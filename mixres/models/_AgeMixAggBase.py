"""Shared data handling and NumPyro building blocks for the AgeMixAgg family."""

from abc import ABC
from typing import Any, Dict, Literal, Optional, Tuple

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import Array
from jax.typing import ArrayLike
from numpyro.handlers import scope

from ._aggregation import (
    uniform_allocation_weights,
    validate_allocation_weights,
)
from ._base import ContactModel


def _as_float(x: Optional[ArrayLike]) -> Optional[Array]:
    return None if x is None else jnp.asarray(x, dtype=jnp.float32)


class AgeMixAggBase(ContactModel, ABC):
    """
    Fine-age contact model observed through age-group aggregation.

    Every model in the family shares the latent fine-age log contact rate and
    the way it is aggregated to the observed groups::

        log gamma[a, b] = baseline + f(a, b),          f ~ priors["rate"]
        m[a, d]         = sum_b gamma[a, b] P[b] Q[b, d]
        EZ[c, d]        = sum_{a in c} P[a] m[a, d]

    where ``EZ`` is the expected contact volume between respondent group
    ``c`` and recorded contact group ``d``. Subclasses differ in the
    allocation weights ``Q`` and in the observation model built on ``EZ``.

    Parameters
    ----------
    y : array-like, shape (C, D)
        Contact counts by respondent group and recorded contact group.
    N_c : array-like, shape (C,)
        Number of respondents in each respondent group.
    P : array-like, shape (A,)
        Positive population age distribution on the fine grid.
    comp_map_cnt : array-like, shape (A, D), optional
        Binary composition matrix mapping fine contact ages onto the recorded
        contact groups (see ``make_composition_mapping``). ``None`` keeps fine
        contact ages (``D = A``).
    comp_map_resp : array-like, shape (A, C), optional
        Binary composition matrix mapping fine respondent ages onto the
        respondent groups. ``None`` keeps fine respondent ages (``C = A``).
        Pass ``comp_map_cnt`` here too to group both axes identically.
    age_values : array-like, shape (A,), optional
        Fine ages; defaults to ``0, ..., A - 1``.
    priors : dict
        Must contain ``"rate"``, a cntmosaic ``Prior2D`` (Spline2D, HSGP2D, IGMRF2D, ...).
    allocation_weights : array-like, shape (A, D), optional
        Probability that a contact of true age ``b`` is recorded in group
        ``d``. Rows must sum to one and respect the support of
        ``comp_map_cnt``. When omitted, ``Q`` is built from ``comp_map_cnt``
        by the ``allocation`` rule.
    allocation : {"binary", "uniform"}, default="binary"
        Rule for building ``Q`` from ``comp_map_cnt`` when
        ``allocation_weights`` is omitted. ``"binary"`` uses ``comp_map_cnt``
        itself, so an age in ``k`` overlapping groups is counted ``k`` times.
        ``"uniform"`` sets ``Q[b, d] = comp_map_cnt[b, d] / sum_j
        comp_map_cnt[b, j]``, so the age contributes ``1/k`` to each group.
    likelihood : str, optional
        One of ``ALLOWED_LIKELIHOODS``; defaults to its first entry.
    inv_disp_prior_rate : float, default=1.0
        Rate of the exponential prior on the negative-binomial inverse
        concentration ``inv_disp``.
    """

    ALLOWED_LIKELIHOODS: Tuple[str, ...] = ("poisson",)
    #: Whether the latent rate surface is symmetrised (f = f^T).
    reciprocal: bool = False
    #: Keys of a ``prepare_age_mix_agg_inputs`` dict forwarded by ``from_prepared_inputs``.
    PREPARED_INPUT_KEYS: Tuple[str, ...] = (
        "y",
        "N_c",
        "P",
        "comp_map_cnt",
        "comp_map_resp",
        "allocation_weights",
        "allocation",
        "age_values",
    )

    def __init__(
        self,
        y: ArrayLike,
        N_c: Optional[ArrayLike],
        P: ArrayLike,
        comp_map_cnt: Optional[ArrayLike] = None,
        comp_map_resp: Optional[ArrayLike] = None,
        age_values: Optional[ArrayLike] = None,
        priors: Optional[Dict[str, Any]] = None,
        allocation_weights: Optional[ArrayLike] = None,
        allocation: Literal["binary", "uniform"] = "binary",
        likelihood: Optional[str] = None,
        inv_disp_prior_rate: float = 1.0,
    ) -> None:
        super().__init__()
        if priors is None or "rate" not in priors:
            raise ValueError(
                f"{type(self).__name__} requires priors with a 'rate' Prior2D instance."
            )
        self.priors = dict(priors)
        self.likelihood = (
            self.ALLOWED_LIKELIHOODS[0] if likelihood is None else likelihood
        )
        if self.likelihood not in self.ALLOWED_LIKELIHOODS:
            raise ValueError(f"likelihood must be one of {self.ALLOWED_LIKELIHOODS}.")
        if allocation not in ("binary", "uniform"):
            raise ValueError(
                f"allocation must be 'binary' or 'uniform', got {allocation!r}."
            )
        self.allocation = allocation
        self.inv_disp_prior_rate = float(inv_disp_prior_rate)
        if not jnp.isfinite(self.inv_disp_prior_rate) or self.inv_disp_prior_rate <= 0:
            raise ValueError("inv_disp_prior_rate must be finite and positive.")

        self.y = _as_float(y)
        self.P = _as_float(P)
        self.comp_map_cnt = _as_float(comp_map_cnt)
        self.comp_map_resp = _as_float(comp_map_resp)
        self.age_values = (
            jnp.arange(self.P.shape[0], dtype=jnp.float32)
            if age_values is None
            else _as_float(age_values)
        )
        self._validate_grid()

        A = self.P.shape[0]
        # From here on comp_map_resp is a dense bool (A, C) matrix, the
        # identity for fine respondent ages.
        self.comp_map_resp = (
            jnp.eye(A, dtype=bool)
            if self.comp_map_resp is None
            else self.comp_map_resp.astype(bool)
        )
        self.allocation_weights = self._resolve_allocation_weights(allocation_weights)
        self.N_c = self._resolve_respondent_counts(N_c)

        self.P_respondent = self.comp_map_resp.T.astype(jnp.float32) @ self.P
        self.log_P = jnp.log(self.P)
        self.log_P_respondent = jnp.log(self.P_respondent)
        self.log_N_c = jnp.log(jnp.clip(self.N_c, min=1e-12))

        self._validate_observations()
        self._configure_rate_prior()

    @classmethod
    def from_prepared_inputs(
        cls, inputs: Dict[str, Any], **kwargs: Any
    ) -> "AgeMixAggBase":
        """Construct the model from a ``prepare_age_mix_agg_inputs`` dictionary."""
        prepared = {
            key: inputs[key] for key in cls.PREPARED_INPUT_KEYS if key in inputs
        }
        return cls(**prepared, **kwargs)

    # ------------------------------------------------------------------
    # Input handling (override the hooks to extend)
    # ------------------------------------------------------------------

    @property
    def n_respondent_groups(self) -> int:
        return self.comp_map_resp.shape[1]

    @property
    def n_contact_groups(self) -> int:
        return self.allocation_weights.shape[1]

    def _validate_grid(self) -> None:
        if self.P.ndim != 1 or not bool(jnp.all(jnp.isfinite(self.P))):
            raise ValueError("P must be a finite one-dimensional array.")
        if bool(jnp.any(self.P <= 0)):
            raise ValueError("P must contain positive values.")
        if self.age_values.shape != self.P.shape:
            raise ValueError("age_values and P must have the same length.")

        if self.comp_map_cnt is not None:
            self._check_comp_map(self.comp_map_cnt, "comp_map_cnt")
        if self.comp_map_resp is not None:
            self._check_comp_map(self.comp_map_resp, "comp_map_resp")

    def _check_comp_map(self, comp_map: Array, name: str) -> None:
        """Check that ``comp_map`` is a binary (A, n_groups) composition matrix.

        Parameters
        ----------
        comp_map : Array
            Candidate composition matrix.
        name : str
            Argument name used in error messages.

        Raises
        ------
        ValueError
            If ``comp_map`` is not two-dimensional with A rows, has entries
            other than 0 and 1, or has a group that contains no fine age.
        """
        A = self.P.shape[0]
        if comp_map.ndim != 2 or comp_map.shape[0] != A:
            raise ValueError(f"{name} must have shape ({A}, n_groups).")
        if not bool(jnp.all((comp_map == 0) | (comp_map == 1))):
            raise ValueError(f"{name} must be a binary composition matrix.")
        if bool(jnp.any(comp_map.sum(axis=0) == 0)):
            raise ValueError(f"Every group in {name} must contain at least one age.")

    def _resolve_allocation_weights(
        self, allocation_weights: Optional[ArrayLike]
    ) -> Array:
        A = self.P.shape[0]
        if allocation_weights is not None:
            return validate_allocation_weights(allocation_weights, self.comp_map_cnt, A)
        if self.comp_map_cnt is None:
            return jnp.eye(A, dtype=jnp.float32)
        if self.allocation == "binary":
            return self.comp_map_cnt
        return uniform_allocation_weights(self.comp_map_cnt)

    def _resolve_respondent_counts(self, N_c: Optional[ArrayLike]) -> Array:
        N_c = _as_float(N_c)
        if N_c is None or N_c.shape != (self.n_respondent_groups,):
            raise ValueError(f"N_c must have shape ({self.n_respondent_groups},).")
        return N_c

    def _validate_observations(self) -> None:
        expected = (self.n_respondent_groups, self.n_contact_groups)
        if self.y.shape != expected:
            raise ValueError(f"y must have shape {expected}.")

    def _validate_integer_counts(self) -> None:
        if not bool(jnp.all(jnp.isfinite(self.y))) or bool(jnp.any(self.y < 0)):
            raise ValueError("y must contain finite, non-negative counts.")
        if not bool(jnp.allclose(self.y, jnp.round(self.y))):
            raise ValueError("y must contain integer-valued counts.")

    def _configure_rate_prior(self) -> None:
        rate_prior = self.priors["rate"]
        if hasattr(rate_prior, "set_age_bounds"):
            rate_prior.set_age_bounds(
                int(jnp.min(self.age_values)), int(jnp.max(self.age_values))
            )
        if getattr(rate_prior, "prior_type", None) == "global":
            return
        if (
            hasattr(rate_prior, "set_event_dim")
            and getattr(rate_prior, "event_dim", None) is None
        ):
            rate_prior.set_event_dim(1)
        if hasattr(rate_prior, "set_loc") and getattr(rate_prior, "loc", None) is None:
            rate_prior.set_loc(0.0)

    # ------------------------------------------------------------------
    # NumPyro building blocks
    # ------------------------------------------------------------------

    def _sample_log_rate(self) -> Array:
        """Sample ``baseline`` and the ``rate`` surface; record ``log_gamma``."""
        baseline = numpyro.sample("baseline", dist.Normal(-jnp.mean(self.log_P), 2.5))

        with scope(prefix="rate"):
            f = self.priors["rate"].sample()

        if f.ndim == 3 and f.shape[0] == 1:
            f = f[0]

        A = self.P.shape[0]
        if f.shape != (A, A):
            raise ValueError(
                f"priors['rate'].sample() must return shape {(A, A)} or (1, {A}, {A}), got {f.shape}."
            )

        if self.reciprocal:
            f = jnp.tril(f) + jnp.tril(f, -1).T

        return numpyro.deterministic("log_gamma", baseline + f)

    def _sample_obs(
        self,
        mu: ArrayLike,
        y: Optional[ArrayLike] = None,
        active: Optional[ArrayLike] = None,
    ) -> None:
        """Sample the cell-level count likelihood ``obs``.

        Cells are flattened into a single ``data`` plate. With
        ``likelihood="negbin"`` this also samples ``inv_disp`` and records
        ``concentration = 1 / inv_disp``.

        Parameters
        ----------
        mu : array-like
            Expected count of every cell; same shape as ``y``.
        y : array-like, optional
            Observed counts; ``None`` samples from the predictive distribution.
        active : array-like of bool, optional
            Broadcastable to ``mu``; cells where it is ``False`` are excluded
            from the likelihood. Defaults to all cells.
        """
        mu = jnp.asarray(mu)
        active = (
            jnp.ones(mu.shape, dtype=bool)
            if active is None
            else jnp.broadcast_to(active, mu.shape)
        )
        # NegativeBinomial2 needs a strictly positive mean; masked cells have a
        # structural mean of zero, so give them a placeholder.
        mean = jnp.where(active, mu, 1e-12).ravel()

        if self.likelihood == "negbin":
            inv_disp = numpyro.sample(
                "inv_disp", dist.Exponential(self.inv_disp_prior_rate)
            )
            concentration = numpyro.deterministic("concentration", 1.0 / inv_disp)
            obs_dist = dist.NegativeBinomial2(mean=mean, concentration=concentration)
        else:
            obs_dist = dist.Poisson(rate=mean)

        with numpyro.plate("data", mean.size):
            numpyro.sample(
                "obs",
                obs_dist.mask(active.ravel()),
                obs=None if y is None else jnp.ravel(y),
            )
