import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.contrib.hsgp.approximation import (
  hsgp_squared_exponential,
  hsgp_matern
)

from ._inference import run_inference_svi, posterior_predictive_svi
from ._distributions import multinomial
from ._utils import create_interval_index_array

class DisjointAggPP:
  """
  Disjoint Aggregated Poisson Process model.
  """

  def __init__(
    self,
    df_data: pd.DataFrame,
    df_true: pd.DataFrame,
    L: float = 10.0,
    M: int = 30
  ):
      self.df_data = df_data
      self.df_true = df_true
      self.L = L
      self.M = M
      self.load_data()
      
  def load_data(self):
    """
    Load data from the DataFrame.
    """
    # Grid points (x)
    self.x = self.df_true['x'].values # Grid points
    self.xstd = (self.x - np.mean(self.x)) / np.std(self.x)  # Standardize x

    # Log exposure (log_P)
    self.log_P = self.df_true['log_exposure'].values # Log exposure

    # Total counts for each range (T)
    df_obs = self.df_data.groupby('range', observed=True).agg(
      y_agg=('y', 'sum'),
      N=('y', 'count')
    ).reset_index()
    self.T = df_obs['y_agg'].values # Total counts for each range

    # Number of observations for each range (N)
    range_lengths = {iv: (iv.right - iv.left) for iv in self.df_data['range'].cat.categories}
    self.df_data['R'] = self.df_data['range'].map(range_lengths)
    self.df_N = (
      self.df_data[['x', 'N', 'R']]
      .drop_duplicates()
      .sort_values(by='x')
      .reset_index(drop=True)
    )
    self.N = self.df_N['N'].values # Number of observations for each range
    self.R = self.df_N['R'].values # Range lengths for each range

    # Mapping indices to intervals
    # Create a mapping from range to indices
    int_ind_arr = create_interval_index_array(
      self.df_data['range'],
      grid_start=self.x[0],
      grid_end=self.x[-1],
      grid_step=1
    )
    ind_arr = np.arange(len(int_ind_arr))

    # Maps the interval codes to their corresponding indices
    int_codes = self.df_data['range'].cat.codes.sort_values().unique()
    self.int_map = {int(code): ind_arr[int_ind_arr == code]
                    for code in int_codes}

  def model(self):
    # --- Priors ---
    beta = numpyro.sample('baseline', dist.Normal(0, 1))
    sigma = numpyro.sample('sigma', dist.LogNormal(0, 1))
    lenscale = numpyro.sample('lenscale', dist.LogNormal(0, 1))

    # --- Parameterization ---
    f = hsgp_matern(
      x=self.xstd,
      nu=5/2,
      alpha=sigma,
      length=lenscale,
      ell=self.L,
      m=self.M,
      non_centered=False
    )

    # --- Likelihood ---
    log_rate = self.log_P + (beta + f)
    rate = numpyro.deterministic('rate', jnp.exp(log_rate))
    
    # --- Data augmentation ---
    y_aug = jnp.zeros(len(self.xstd))
    for i, ind in self.int_map.items():
      probs = rate[ind] / jnp.sum(rate[ind])
      y_aug = y_aug.at[ind].set(multinomial(key=jax.random.PRNGKey(0), n=self.T[i], p=probs))

    with numpyro.plate('data', len(self.xstd)):
      numpyro.sample('y', dist.Poisson(rate * self.N / self.R), obs=y_aug)
      
  def run_inference_svi(
    self,
    prng_key: jax.random.PRNGKey,
    guide: callable,
    num_steps: int = 5_000,
    peak_lr: float = 0.01,
    **model_kwargs,
  ):
    """Run stochastic variational inference.

    Parameters
    ----------
    prng_key:
      Random number generator key.
    guide: callable
      The guide function.
    num_steps: int, default=5_000
      Number of steps to run.
    peak_lr: float, default=0.01
      Peak learning rate.
    **model_kwargs
      Additional keyword arguments to pass to the SVI
    """
    self.guide = guide
    self.svi = run_inference_svi(
      prng_key=prng_key,
      model=self.model,
      guide=guide,
      num_steps=num_steps,
      peak_lr=peak_lr,
      **model_kwargs
    )
    
  def posterior_predictive_svi(
    self,
    prng_key,
    guide: callable,
    num_samples: int = 5_000,
    **model_kwargs,
  ) -> dict[str, jax.Array]:
    """Generate posterior predictive samples using SVI.

    Parameters
    ----------
    prng_key:
      Random number generator key.
    guide: callable
      The guide function.
    num_samples: int, default=2000
      Number of samples to draw.
    **model_kwargs
      Additional keyword arguments to pass to the Predictive
    """
    if hasattr(self, 'svi') is False:
      raise AttributeError('run_inferece_svi must be run first.')

    return posterior_predictive_svi(
      prng_key,
      self.model,
      guide,
      self.svi.params,
      num_samples=num_samples,
      **model_kwargs
    )