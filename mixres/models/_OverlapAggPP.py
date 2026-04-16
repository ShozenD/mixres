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
from ._utils import (
  create_interval_grid_index_map,
  create_interval_dict,
  find_overlap_intervals,
  create_overlap_weights
)

class OverlapAggPP:
  """
  Overlapping Aggregated Poisson Process model.
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
    ).reset_index().sort_values(by='range')
    self.T = df_obs['y_agg'].values # Total counts for each range
    
    # Sampling effort is assumed to be uniform for simplicity
    self.sampling_effort = np.ones(len(self.T))

    # Get interval information
    interval_dict = create_interval_dict(self.df_data['range'])
    overlap_dict = find_overlap_intervals(interval_dict)
    
    # Create sampling effort dict using the actual interval codes
    sampling_effort_dict = {}
    for code in interval_dict.keys():
      # Find the index of this code in the sorted T array
      interval_idx = list(interval_dict.keys()).index(code)
      sampling_effort_dict[code] = self.sampling_effort[interval_idx]
      
    self.overlap_weights = create_overlap_weights(
      interval_dict, overlap_dict, sampling_effort_dict
    )

    # Mapping indices to intervals
    self.int_grid_idx_map = create_interval_grid_index_map(
      self.df_data['range'],
      grid_start=self.x[0],
      grid_end=self.x[-1],
      grid_step=1
    )

    # Calculate interval lengths and total length
    self.interval_lengths = np.array([interval.right - interval.left for interval in interval_dict.values()])
    self.cum_interval_length = np.sum(self.interval_lengths)

    # Interval sample counts
    self.interval_N = np.array(
      [
        df_obs.loc[df_obs['range'] == interval, 'N'].values[0]
        for interval in interval_dict.values()
      ]
    )

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
    y_aug = jnp.zeros(self.cum_interval_length)
    rate_aug = jnp.zeros(self.cum_interval_length)
    curr_head = 0
    for i, ind in self.int_grid_idx_map.items():
      weighted_rate = rate[ind] * self.overlap_weights[i]
      probs = weighted_rate / jnp.sum(weighted_rate)
      
      y_aug = y_aug.at[curr_head:curr_head + self.interval_lengths[i]].add(
        multinomial(key=jax.random.PRNGKey(0), n=self.T[i], p=probs)
      )
      rate_aug = rate_aug.at[curr_head:curr_head + self.interval_lengths[i]].set(
        rate[ind] * self.interval_N[i] / self.interval_lengths[i]  # Normalize by interval length
      )
      
      curr_head += self.interval_lengths[i]

    # For overlapping intervals, likelihood should account for all intervals and grid points
    with numpyro.plate('data', self.cum_interval_length):
      numpyro.sample('y', dist.Poisson(rate=rate_aug), obs=y_aug)

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