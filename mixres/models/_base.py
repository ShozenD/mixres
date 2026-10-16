"""
Abstract base class for Bayesian contact models in mixres.

This is a local copy of the ``cntmosaic.models.ContactModel`` interface
(same method names and signatures). When the models move to cntmosaic, this
module is deleted and the models inherit from cntmosaic's class instead.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Literal, Optional

from jax.random import PRNGKey
from numpyro.infer.autoguide import AutoNormal

from ._inference import (
    posterior_predictive_mcmc,
    posterior_predictive_svi,
    run_inference_mcmc,
    run_inference_svi,
)


class ContactModel(ABC):
    """Abstract base class for Bayesian social contact models."""

    def __init__(self) -> None:
        self._mcmc_result: Optional[Any] = None
        self._svi_result: Optional[Any] = None
        self._guide: Optional[Callable] = None
        self.y: Optional[Any] = None  # set by concrete subclasses

    @property
    def inference_method(self) -> Optional[Literal["mcmc", "svi"]]:
        """Return ``"mcmc"``, ``"svi"``, or ``None`` depending on which inference was run."""
        if self._mcmc_result is not None:
            return "mcmc"
        if self._svi_result is not None:
            return "svi"
        return None

    @abstractmethod
    def model(self, y: Optional[Any] = None) -> None:
        """NumPyro model. ``y=None`` samples observations (prior/posterior predictive)."""

    def _check_data(self) -> None:
        if self.y is None:
            raise AttributeError("Observation data (self.y) has not been set.")

    def run_inference_mcmc(
        self,
        prng_key: PRNGKey,
        num_samples: int = 500,
        num_warmup: int = 500,
        num_chains: int = 2,
        target_accept_prob: float = 0.8,
        **kwargs: Any,
    ) -> None:
        """Run NUTS. Extra keyword arguments are forwarded to the model."""
        self._check_data()
        kwargs.setdefault("y", self.y)
        self._mcmc_result = run_inference_mcmc(
            prng_key,
            self.model,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            target_accept_prob=target_accept_prob,
            **kwargs,
        )

    def run_inference_svi(
        self,
        prng_key: PRNGKey,
        guide: Optional[Callable] = None,
        num_steps: int = 5_000,
        peak_lr: float = 0.01,
        **kwargs: Any,
    ) -> None:
        """Run SVI. ``guide`` defaults to ``AutoNormal(self.model)``."""
        self._check_data()
        kwargs.setdefault("y", self.y)
        self._guide = AutoNormal(self.model) if guide is None else guide
        self._svi_result = run_inference_svi(
            prng_key,
            self.model,
            self._guide,
            num_steps=num_steps,
            peak_lr=peak_lr,
            **kwargs,
        )

    def posterior_predictive_mcmc(
        self, prng_key: PRNGKey, **kwargs: Any
    ) -> Dict[str, Any]:
        """Posterior predictive samples from the MCMC draws."""
        if self._mcmc_result is None:
            raise AttributeError("Call run_inference_mcmc() first.")
        return posterior_predictive_mcmc(prng_key, self.model, self._mcmc_result, **kwargs)

    def posterior_predictive_svi(
        self,
        prng_key: PRNGKey,
        guide: Optional[Callable] = None,
        num_samples: int = 1_000,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Posterior predictive samples from the fitted guide (defaults to the stored one)."""
        if self._svi_result is None:
            raise AttributeError("Call run_inference_svi() first.")
        return posterior_predictive_svi(
            prng_key,
            self.model,
            self._guide if guide is None else guide,
            self._svi_result.params,
            num_samples=num_samples,
            **kwargs,
        )
