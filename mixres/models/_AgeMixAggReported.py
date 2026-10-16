"""Reporting-adjusted reciprocal age-mixing model with cell-level count likelihoods."""

from typing import Any, Dict, Literal, Optional

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import Array
from jax.scipy.special import logsumexp
from jax.typing import ArrayLike

from ._AgeMixAggBase import AgeMixAggBase, _as_float
from ._aggregation import safe_log


class AgeMixAggReported(AgeMixAggBase):
    """
    Fine-age reciprocal contact model for interval-reported contact counts.

    Mathematical model
    ------------------
    ::

        log gamma[a, b] = baseline + f_sym(a, b),     f_sym = tril(f) + tril(f, -1)^T
        mu[c, d]        = N[c] / P[c] * EZ[c, d]
        y[c, d]         ~ NegativeBinomial2(mu, 1 / inv_disp)   or   Poisson(mu)

    where ``EZ`` uses allocation weights ``Q`` (default
    ``allocation="uniform"``, so overlapping groups share each age). Cells of respondent
    strata with no respondents have a structural mean of zero and are masked.

    Repeat-stratified observations
    ------------------------------
    ``y`` may have shape ``(C, D, R)``, indexed by the number of previous
    survey waves ``repeat_levels[r]``. Reporting completeness then decays as

        log completeness[r] = -kappa * s_r / (1 + s_r),   s_r = exp(zeta) * repeat_levels[r]^eta

    and exposure is either grouped ``N_c`` with shape ``(C, R)`` or exact-age
    ``N_age_repeat`` with shape ``(A, R)``. The exact-age form needs the
    respondent groups to partition the fine ages.

    Parameters
    ----------
    likelihood : {"negbin", "poisson"}, default="negbin"
    repeat_levels : array-like, shape (R,), optional
        Unique non-negative integer repeat levels; required for 3-D ``y``.
    N_age_repeat : array-like, shape (A, R), optional
        Exact-age respondent counts per repeat level; replaces ``N_c``.

    See :class:`AgeMixAggBase` for the remaining parameters.
    """

    ALLOWED_LIKELIHOODS = ("negbin", "poisson")
    reciprocal = True
    PREPARED_INPUT_KEYS = AgeMixAggBase.PREPARED_INPUT_KEYS + (
        "repeat_levels",
        "N_age_repeat",
    )

    def __init__(
        self,
        y: ArrayLike,
        N_c: Optional[ArrayLike],
        P: ArrayLike,
        comp_map_cnt: Optional[ArrayLike] = None,
        allocation_weights: Optional[ArrayLike] = None,
        allocation: Literal["binary", "uniform"] = "uniform",
        comp_map_resp: Optional[ArrayLike] = None,
        age_values: Optional[ArrayLike] = None,
        priors: Optional[Dict[str, Any]] = None,
        likelihood: str = "negbin",
        inv_disp_prior_rate: float = 1.0,
        repeat_levels: Optional[ArrayLike] = None,
        N_age_repeat: Optional[ArrayLike] = None,
    ) -> None:
        # Needed by the input hooks that run inside the base constructor.
        self.repeat_levels = _as_float(repeat_levels)
        self.N_age_repeat = _as_float(N_age_repeat)
        super().__init__(
            y=y,
            N_c=N_c,
            P=P,
            comp_map_cnt=comp_map_cnt,
            comp_map_resp=comp_map_resp,
            age_values=age_values,
            priors=priors,
            allocation_weights=allocation_weights,
            allocation=allocation,
            likelihood=likelihood,
            inv_disp_prior_rate=inv_disp_prior_rate,
        )

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    def _resolve_respondent_counts(self, N_c: Optional[ArrayLike]) -> Array:
        N_c = _as_float(N_c)
        A, C = self.comp_map_resp.shape

        if self.y.ndim == 2:
            if self.N_age_repeat is not None or self.repeat_levels is not None:
                raise ValueError("Fatigue inputs require three-dimensional y.")
            if N_c is None or N_c.shape != (C,):
                raise ValueError(f"N_c must have shape ({C},).")
            return N_c

        if self.y.ndim != 3:
            raise ValueError("y must have shape (C, D) or (C, D, R).")
        R = self.y.shape[2]
        if self.N_age_repeat is None:
            if N_c is None or N_c.shape != (C, R):
                raise ValueError(
                    f"Three-dimensional y requires N_c {(C, R)} or N_age_repeat {(A, R)}."
                )
            return N_c

        if N_c is not None:
            raise ValueError("Supply either N_c or N_age_repeat, not both.")
        if self.N_age_repeat.shape != (A, R):
            raise ValueError(f"N_age_repeat must have shape {(A, R)}.")
        if not bool(jnp.all(self.comp_map_resp.sum(axis=1) == 1)):
            raise ValueError(
                "N_age_repeat requires respondent groups to partition ages."
            )
        return self.comp_map_resp.T.astype(jnp.float32) @ self.N_age_repeat

    def _validate_observations(self) -> None:
        self._validate_integer_counts()
        if not bool(jnp.all(jnp.isfinite(self.N_c))) or bool(jnp.any(self.N_c < 0)):
            raise ValueError("Respondent counts must be finite and non-negative.")

        expected = (self.n_respondent_groups, self.n_contact_groups)
        if self.y.shape[:2] != expected:
            raise ValueError(f"The first two dimensions of y must be {expected}.")
        if self.y.ndim == 3:
            self._validate_repeat_levels()
        if bool(jnp.any((self.N_c == 0) & (self.y.sum(axis=1) > 0))):
            raise ValueError("Strata with no respondents cannot have observations.")

    def _validate_repeat_levels(self) -> None:
        R = self.y.shape[2]
        if self.repeat_levels is None or self.repeat_levels.shape != (R,):
            raise ValueError(f"repeat_levels must have shape ({R},).")
        levels = self.repeat_levels
        if (
            bool(jnp.any(levels < 0))
            or not bool(jnp.allclose(levels, jnp.round(levels)))
            or jnp.unique(levels).size != R
        ):
            raise ValueError("repeat_levels must be unique non-negative integers.")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def _sample_log_completeness(self) -> Array:
        kappa = numpyro.sample("fatigue_kappa", dist.HalfNormal(1.0))
        zeta = numpyro.sample("fatigue_zeta", dist.Normal(0.0, 1.0))
        eta = numpyro.sample("fatigue_eta", dist.Exponential(1.0))
        scaled_repeat = jnp.exp(zeta) * self.repeat_levels**eta
        log_completeness = numpyro.deterministic(
            "log_completeness", -kappa * scaled_repeat / (1.0 + scaled_repeat)
        )
        numpyro.deterministic("completeness", jnp.exp(log_completeness))
        return log_completeness

    def _log_mean(self, log_m: Array, log_EZ: Array) -> Array:
        """Log expected counts; ``-inf`` in strata without respondents.

        ``log_m`` is the fine-age ``log m[a, d]`` (A, D), needed when exposure
        is given per exact age; ``log_EZ`` is ``log EZ[c, d]`` (C, D).
        """
        active = jnp.expand_dims(self.N_c > 0, 1)

        if self.y.ndim == 2:
            log_mean = log_EZ + (safe_log(self.N_c) - self.log_P_respondent)[:, None]
            return jnp.where(active, log_mean, -jnp.inf)

        log_completeness = self._sample_log_completeness()
        if self.N_age_repeat is None:
            log_exposure = safe_log(self.N_c) - self.log_P_respondent[:, None]
            log_mean = log_EZ[:, :, None] + log_exposure[:, None, :]
        else:
            # Sum over the exact respondent ages within each group, so that the
            # observed age composition replaces the population composition.
            exact_active = self.N_age_repeat > 0
            age_repeat_terms = (
                log_m[:, :, None] + safe_log(self.N_age_repeat)[:, None, :]
            )
            grouped_terms = jnp.where(
                self.comp_map_resp.T[:, :, None, None] & exact_active[None, :, None, :],
                age_repeat_terms[None, :, :, :],
                # Keep the inactive branch finite so logsumexp has a well-defined
                # derivative for respondent/repeat groups with no exposure.
                jnp.asarray(-1e30, dtype=age_repeat_terms.dtype),
            )
            log_mean = logsumexp(grouped_terms, axis=1)
        return jnp.where(active, log_mean + log_completeness, -jnp.inf)

    def model(self, y: Optional[ArrayLike] = None) -> None:

        log_gamma = self._sample_log_rate()

        # log m[a, d] = log sum_b gamma[a, b] P[b] Q[b, d]
        log_m = logsumexp(
            log_gamma[:, :, None]
            + self.log_P[None, :, None]
            + safe_log(self.allocation_weights)[None, :, :],
            axis=1,
        )

        # log EZ[c, d] = log sum_{a in c} P[a] m[a, d]
        log_comp_map_resp = safe_log(self.comp_map_resp)
        log_EZ = numpyro.deterministic(
            "log_EZ",
            logsumexp(
                (log_m + self.log_P[:, None])[None, :, :]
                + log_comp_map_resp.T[:, :, None],
                axis=1,
            ),
        )

        log_mu = numpyro.deterministic("log_mu", self._log_mean(log_m, log_EZ))

        mu = numpyro.deterministic("mu", jnp.exp(log_mu))

        self._sample_obs(mu, y, active=jnp.expand_dims(self.N_c > 0, 1))
