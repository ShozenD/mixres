"""Posterior sample extraction and summarisation for mixres models."""

from __future__ import annotations

from typing import Any, Literal, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._summary import ContactSummary


Quantity = Literal["log_rate", "rate", "cint", "mcint", "column_mcint"]
Measure = Literal["mean", "median"]


class ModelSummariser:
    """Standardise posterior quantities from a fitted mixres model.

    For :class:`mixres.models.AgeMixAgg`, the quantities follow the convention
    already used by the development notebooks:

    ``rate = exp(log_gamma)``

    ``cint = rate * P[None, None, :]``

    where ``P`` is the fine-age population vector on the contact-age axis.
    Marginal contact intensity is then the sum of ``cint`` over contact ages.

    Parameters
    ----------
    model
        Fitted mixres model. An SVI-fitted ``AgeMixAgg`` must expose
        ``posterior_predictive_svi``, ``guide``, ``svi``, and ``P``.
    posterior_samples
        Optional precomputed posterior dictionary. Supplying this avoids
        drawing samples from ``model`` and is useful for saved results/tests.
        It must contain one of ``log_gamma``, ``rate``, or ``cint``.
    population
        Optional fine-age population vector. If omitted, ``model.P`` is used.
        It is required when contact intensity must be constructed from rate.
    num_samples
        Number of posterior draws requested from an SVI model.
    rng_key
        JAX random key for drawing SVI posterior samples. Defaults lazily to
        ``jax.random.PRNGKey(0)``.
    model_kwargs
        Optional keyword arguments passed to posterior sampling.
    """

    def __init__(
        self,
        model: Any | None = None,
        *,
        posterior_samples: Mapping[str, ArrayLike] | None = None,
        population: ArrayLike | None = None,
        num_samples: int = 3_000,
        rng_key: Any | None = None,
        model_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if model is None and posterior_samples is None:
            raise ValueError("Provide either a fitted model or posterior_samples")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")

        self.model = model
        self.num_samples = int(num_samples)
        self.rng_key = rng_key
        self.model_kwargs = dict(model_kwargs or {})

        if population is None and model is not None and hasattr(model, "P"):
            population = model.P
        self.population = (
            None if population is None else self._validate_population(population)
        )

        self.inference_method = (
            "provided"
            if posterior_samples is not None
            else self._detect_inference_method(model)
        )
        self._raw_samples = (
            None
            if posterior_samples is None
            else self._normalise_sample_mapping(posterior_samples)
        )
        self._quantity_cache: dict[str, NDArray[np.float64]] = {}
        self._summary_cache: dict[tuple[str, float, str], ContactSummary] = {}

    @staticmethod
    def _validate_population(population: ArrayLike) -> NDArray[np.float64]:
        population_array = np.asarray(population, dtype=float)
        if population_array.ndim != 1:
            raise ValueError("population must be a one-dimensional array")
        if population_array.size == 0:
            raise ValueError("population cannot be empty")
        if not np.all(np.isfinite(population_array)):
            raise ValueError("population contains NaN or infinite values")
        if np.any(population_array < 0):
            raise ValueError("population must be non-negative")
        return population_array

    @staticmethod
    def _normalise_sample_mapping(
        posterior_samples: Mapping[str, ArrayLike],
    ) -> dict[str, NDArray[np.float64]]:
        if not isinstance(posterior_samples, Mapping):
            raise TypeError("posterior_samples must be a mapping")
        if not posterior_samples:
            raise ValueError("posterior_samples cannot be empty")

        samples = {
            name: np.asarray(values, dtype=float)
            for name, values in posterior_samples.items()
        }
        for name, values in samples.items():
            if values.ndim == 0:
                raise ValueError(f"posterior sample {name!r} must include a draw axis")
            if values.shape[0] == 0:
                raise ValueError(f"posterior sample {name!r} has no draws")
            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"posterior sample {name!r} contains NaN or infinite values"
                )
        return samples

    @staticmethod
    def _detect_inference_method(model: Any) -> Literal["svi", "custom"]:
        if model is None:
            raise ValueError("A model is required when posterior_samples is omitted")
        if hasattr(model, "svi") and hasattr(model, "posterior_predictive_svi"):
            return "svi"
        if hasattr(model, "draw_posterior_samples"):
            return "custom"
        raise ValueError(
            "The model has no recognised completed inference. Run SVI first, "
            "provide posterior_samples, or implement draw_posterior_samples()."
        )

    def _draw_samples(self) -> dict[str, NDArray[np.float64]]:
        if self.inference_method == "svi":
            if not hasattr(self.model, "guide"):
                raise ValueError("The SVI-fitted model does not expose its guide")

            rng_key = self.rng_key
            if rng_key is None:
                from jax.random import PRNGKey

                rng_key = PRNGKey(0)

            samples = self.model.posterior_predictive_svi(
                rng_key,
                self.model.guide,
                num_samples=self.num_samples,
                **self.model_kwargs,
            )
        elif self.inference_method == "custom":
            rng_key = self.rng_key
            if rng_key is None:
                from jax.random import PRNGKey

                rng_key = PRNGKey(0)
            samples = self.model.draw_posterior_samples(
                rng_key,
                num_samples=self.num_samples,
                **self.model_kwargs,
            )
        else:
            raise RuntimeError(f"Unsupported inference method {self.inference_method!r}")

        return self._normalise_sample_mapping(samples)

    def _ensure_loaded(self) -> None:
        if self._raw_samples is None:
            self._raw_samples = self._draw_samples()

    def _get_rate_samples(self) -> NDArray[np.float64]:
        self._ensure_loaded()
        assert self._raw_samples is not None

        if "rate" in self._raw_samples:
            rate = self._raw_samples["rate"]
        elif "gamma" in self._raw_samples:
            rate = self._raw_samples["gamma"]
        elif "log_gamma" in self._raw_samples:
            rate = np.exp(self._raw_samples["log_gamma"])
        else:
            raise ValueError(
                "Posterior samples do not contain 'log_gamma', 'gamma', or 'rate'"
            )

        self._validate_matrix_draws(rate, "rate")
        return rate

    @staticmethod
    def _validate_matrix_draws(
        samples: NDArray[np.float64],
        name: str,
    ) -> None:
        if samples.ndim != 3:
            raise ValueError(
                f"{name} samples must have shape (draw, age, contact_age); "
                f"got {samples.shape}"
            )

    def _get_cint_samples(self) -> NDArray[np.float64]:
        self._ensure_loaded()
        assert self._raw_samples is not None

        if "cint" in self._raw_samples:
            cint = self._raw_samples["cint"]
            self._validate_matrix_draws(cint, "cint")
            return cint

        rate = self._get_rate_samples()
        if self.population is None:
            raise ValueError(
                "population is required to construct contact intensity from rate"
            )
        if rate.shape[-1] != self.population.size:
            raise ValueError(
                "population length must match the contact-age dimension; "
                f"got {self.population.size} and {rate.shape[-1]}"
            )
        return rate * self.population[None, None, :]

    def get_posterior_samples(self, quantity: Quantity = "cint") -> NDArray[np.float64]:
        """Return posterior draws for a standardised model quantity."""
        if quantity in self._quantity_cache:
            return self._quantity_cache[quantity]

        if quantity == "log_rate":
            rate = self._get_rate_samples()
            if np.any(rate <= 0):
                raise ValueError("rate must be positive to calculate log_rate")
            result = np.log(rate)
        elif quantity == "rate":
            result = self._get_rate_samples()
        elif quantity == "cint":
            result = self._get_cint_samples()
        elif quantity == "mcint":
            result = self._get_cint_samples().sum(axis=-1)
        elif quantity == "column_mcint":
            result = self._get_cint_samples().sum(axis=-2)
        else:
            raise ValueError(
                "quantity must be 'log_rate', 'rate', 'cint', 'mcint', "
                "or 'column_mcint'"
            )

        result = np.asarray(result, dtype=float)
        self._quantity_cache[quantity] = result
        return result

    def summarise(
        self,
        quantity: Quantity = "cint",
        *,
        credible_level: float = 0.95,
        measure: Measure = "median",
    ) -> ContactSummary:
        """Summarise a posterior quantity with an equal-tailed interval."""
        if not 0 < credible_level < 1:
            raise ValueError("credible_level must be between 0 and 1")
        if measure not in ("mean", "median"):
            raise ValueError("measure must be either 'mean' or 'median'")

        cache_key = (quantity, float(credible_level), measure)
        if cache_key in self._summary_cache:
            return self._summary_cache[cache_key]

        samples = self.get_posterior_samples(quantity)
        alpha = 1 - credible_level
        central = (
            np.mean(samples, axis=0)
            if measure == "mean"
            else np.median(samples, axis=0)
        )
        summary = ContactSummary(
            lower=np.quantile(samples, alpha / 2, axis=0),
            central=central,
            upper=np.quantile(samples, 1 - alpha / 2, axis=0),
            measure=measure,
            credible_level=float(credible_level),
        )
        self._summary_cache[cache_key] = summary
        return summary

    def summarise_rate(
        self,
        *,
        credible_level: float = 0.95,
        measure: Measure = "median",
    ) -> ContactSummary:
        """Summarise the fine-age contact-rate matrix."""
        return self.summarise(
            "rate",
            credible_level=credible_level,
            measure=measure,
        )

    def summarise_cint(
        self,
        *,
        credible_level: float = 0.95,
        measure: Measure = "median",
    ) -> ContactSummary:
        """Summarise the fine-age contact-intensity matrix."""
        return self.summarise(
            "cint",
            credible_level=credible_level,
            measure=measure,
        )

    def summarise_mcint(
        self,
        *,
        credible_level: float = 0.95,
        measure: Measure = "median",
    ) -> ContactSummary:
        """Summarise marginal contact intensity over contact ages."""
        return self.summarise(
            "mcint",
            credible_level=credible_level,
            measure=measure,
        )

    def clear_cache(self, *, keep_raw_samples: bool = True) -> None:
        """Clear derived quantities and summaries, optionally including raw draws."""
        self._quantity_cache.clear()
        self._summary_cache.clear()
        if not keep_raw_samples and self.inference_method != "provided":
            self._raw_samples = None
