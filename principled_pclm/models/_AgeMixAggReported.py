"""Reporting-adjusted age-mixing model with cell-level count likelihoods."""

from typing import Literal

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax.scipy.special import logsumexp
from numpyro.handlers import mask, scope

from ._inference import posterior_predictive_svi, run_inference_svi


def _respondent_membership(A, W, W_respondent, aggregation):
    """Return an age-by-respondent-group hard-membership matrix."""
    if W_respondent is not None:
        return W_respondent.astype(bool)
    if aggregation == "both":
        return W.astype(bool)
    return jnp.eye(A, dtype=bool)


def _reporting_matrix(A, W, Q, aggregation):
    """Validate or construct Pr(reported group | true contact age)."""
    if aggregation == "none":
        matrix = jnp.eye(A) if Q is None else jnp.asarray(Q, dtype=jnp.float32)
        if matrix.shape != (A, A):
            raise ValueError("Q must have shape (A, A) when aggregation='none'.")
        support = None
    else:
        support = W.astype(bool)
        coverage = support.sum(axis=1, keepdims=True)
        if bool(jnp.any(coverage == 0)):
            raise ValueError("Every fine contact age must be covered by W.")
        matrix = (
            support.astype(jnp.float32) / coverage
            if Q is None
            else jnp.asarray(Q, dtype=jnp.float32)
        )
        if matrix.shape != support.shape:
            raise ValueError("Q must have the same shape as W.")

    if not bool(jnp.all(jnp.isfinite(matrix))) or bool(jnp.any(matrix < 0)):
        raise ValueError("Q must contain finite, non-negative probabilities.")
    if support is not None and bool(jnp.any((~support) & (matrix != 0))):
        raise ValueError("Q can assign mass only to groups supported by W.")
    if not bool(jnp.allclose(matrix.sum(axis=1), 1.0)):
        raise ValueError("Every row of Q must sum to one.")
    return matrix


def _log_weights(values):
    values = jnp.asarray(values)
    # Avoid evaluating log(0) on the inactive branch: although ``where``
    # returns -inf there, the unused log(0) derivative can still poison SVI
    # initialization when respondent/repeat exposure contains structural zeros.
    safe_values = jnp.where(values > 0, values, 1)
    return jnp.where(values > 0, jnp.log(safe_values), -jnp.inf)


class AgeMixAggReported:
    """Fine-age reciprocal model for interval-reported contact counts.

    ``Q[b, d]`` is the probability that a contact of true age ``b`` is
    recorded in group ``d``. If omitted, it is uniform over the groups in
    ``W`` that contain each age. The default likelihood is cell-level NB2;
    ``likelihood="poisson"`` uses the same expected counts.

    Repeat-stratified observations have shape ``(C, D, R)`` and require
    ``repeat_levels`` plus either grouped ``N_c`` with shape ``(C, R)`` or
    exact-age ``N_age_repeat`` with shape ``(A, R)``. The exact-age form is
    available when respondent groups form a partition of the fine ages.
    """

    def __init__(
        self,
        y,
        N_c,
        P,
        W=None,
        Q=None,
        W_respondent=None,
        aggregation="none",
        age_values=None,
        priors=None,
        likelihood: Literal["negbin", "poisson"] = "negbin",
        inv_disp_prior_rate: float = 1.0,
        repeat_levels=None,
        N_age_repeat=None,
    ):
        if priors is None or "rate" not in priors:
            raise ValueError(
                "AgeMixAggReported requires priors with a 'rate' Prior2D instance."
            )

        self.y = jnp.asarray(y, dtype=jnp.float32)
        self.P = jnp.asarray(P, dtype=jnp.float32)
        self.W = None if W is None else jnp.asarray(W, dtype=jnp.float32)
        self.W_respondent = (
            None
            if W_respondent is None
            else jnp.asarray(W_respondent, dtype=jnp.float32)
        )
        self.aggregation = aggregation
        self.age_values = (
            jnp.arange(len(self.P), dtype=jnp.float32)
            if age_values is None
            else jnp.asarray(age_values, dtype=jnp.float32)
        )
        self.priors = dict(priors)
        self.likelihood = likelihood
        self.inv_disp_prior_rate = float(inv_disp_prior_rate)
        self.repeat_levels = (
            None
            if repeat_levels is None
            else jnp.asarray(repeat_levels, dtype=jnp.float32)
        )
        self.N_age_repeat = (
            None
            if N_age_repeat is None
            else jnp.asarray(N_age_repeat, dtype=jnp.float32)
        )

        self._validate_grid()
        A = len(self.P)
        self.respondent_membership = _respondent_membership(
            A, self.W, self.W_respondent, self.aggregation
        )
        self.Q = _reporting_matrix(A, self.W, Q, self.aggregation)
        self._set_respondent_counts(N_c)
        self._load_data()
        self._validate_observations()
        self._configure_rate_prior()

    @classmethod
    def from_prepared_inputs(cls, inputs, **kwargs):
        """Construct the model from a prepared input dictionary."""
        return cls(
            y=inputs["y"],
            N_c=inputs.get("N_c"),
            P=inputs["P"],
            W=inputs.get("W"),
            Q=inputs.get("Q"),
            W_respondent=inputs.get("W_respondent"),
            aggregation=inputs.get("aggregation", "none"),
            age_values=inputs.get("age_values"),
            repeat_levels=inputs.get("repeat_levels"),
            N_age_repeat=inputs.get("N_age_repeat"),
            **kwargs,
        )

    def _validate_grid(self):
        if self.aggregation not in {"none", "contact", "both"}:
            raise ValueError("aggregation must be 'none', 'contact', or 'both'.")
        if self.P.ndim != 1 or not bool(jnp.all(jnp.isfinite(self.P))):
            raise ValueError("P must be a finite one-dimensional array.")
        if bool(jnp.any(self.P <= 0)):
            raise ValueError("P must contain positive values.")
        if self.age_values.shape != self.P.shape:
            raise ValueError("age_values and P must have the same length.")

        A = len(self.P)
        if self.aggregation == "none":
            if self.W is not None:
                raise ValueError("W must be None when aggregation='none'.")
        elif self.W is None or self.W.ndim != 2 or self.W.shape[0] != A:
            raise ValueError("W must have shape (A, D) when aggregation is enabled.")
        else:
            self._validate_membership(self.W, "W")

        if self.W_respondent is not None:
            if self.W_respondent.ndim != 2 or self.W_respondent.shape[0] != A:
                raise ValueError("W_respondent must have shape (A, C).")
            self._validate_membership(self.W_respondent, "W_respondent")

    @staticmethod
    def _validate_membership(matrix, name):
        if not bool(jnp.all(jnp.isfinite(matrix))) or bool(jnp.any(matrix < 0)):
            raise ValueError(f"{name} must contain finite, non-negative values.")
        if not bool(jnp.all((matrix == 0) | (matrix == 1))):
            raise ValueError(f"{name} must be a binary membership matrix.")
        if bool(jnp.any(matrix.sum(axis=0) == 0)):
            raise ValueError(f"Every group in {name} must contain at least one age.")

    def _set_respondent_counts(self, N_c):
        reported_N = None if N_c is None else jnp.asarray(N_c, dtype=jnp.float32)
        C = self.respondent_membership.shape[1]

        if self.y.ndim == 2:
            if self.N_age_repeat is not None or self.repeat_levels is not None:
                raise ValueError("Fatigue inputs require three-dimensional y.")
            if reported_N is None or reported_N.shape != (C,):
                raise ValueError(f"N_c must have shape ({C},).")
        elif self.y.ndim == 3:
            R = self.y.shape[2]
            if self.N_age_repeat is not None:
                if reported_N is not None:
                    raise ValueError("Supply either N_c or N_age_repeat, not both.")
                if self.N_age_repeat.shape != (len(self.P), R):
                    raise ValueError(
                        f"N_age_repeat must have shape {(len(self.P), R)}."
                    )
                if not bool(jnp.all(self.respondent_membership.sum(axis=1) == 1)):
                    raise ValueError(
                        "N_age_repeat requires respondent groups to partition ages."
                    )
                reported_N = (
                    self.respondent_membership.T.astype(self.N_age_repeat.dtype)
                    @ self.N_age_repeat
                )
            elif reported_N is None or reported_N.shape != (C, R):
                raise ValueError(
                    f"Three-dimensional y requires N_c {(C, R)} or "
                    f"N_age_repeat {(len(self.P), R)}."
                )
        else:
            raise ValueError("y must have shape (C, D) or (C, D, R).")

        self.N_c = reported_N

    def _load_data(self):
        self.P_respondent = self.respondent_membership.T @ self.P
        self.log_P = jnp.log(self.P)
        self.log_P_respondent = jnp.log(self.P_respondent)

    def _validate_observations(self):
        if self.likelihood not in {"negbin", "poisson"}:
            raise ValueError("likelihood must be 'negbin' or 'poisson'.")
        if (
            not bool(jnp.isfinite(self.inv_disp_prior_rate))
            or self.inv_disp_prior_rate <= 0
        ):
            raise ValueError("inv_disp_prior_rate must be finite and positive.")
        if not bool(jnp.all(jnp.isfinite(self.y))) or bool(jnp.any(self.y < 0)):
            raise ValueError("y must contain finite, non-negative counts.")
        if not bool(jnp.allclose(self.y, jnp.round(self.y))):
            raise ValueError("y must contain integer-valued counts.")
        if not bool(jnp.all(jnp.isfinite(self.N_c))) or bool(jnp.any(self.N_c < 0)):
            raise ValueError("Respondent counts must be finite and non-negative.")

        expected_shape = (
            self.respondent_membership.shape[1],
            self.Q.shape[1],
        )
        if self.y.shape[:2] != expected_shape:
            raise ValueError(
                f"The first two dimensions of y must be {expected_shape}."
            )

        if self.y.ndim == 2:
            empty_with_data = (self.N_c == 0) & (self.y.sum(axis=1) > 0)
        else:
            self._validate_repeat_levels()
            empty_with_data = (self.N_c == 0) & (self.y.sum(axis=1) > 0)
        if bool(jnp.any(empty_with_data)):
            raise ValueError("Strata with no respondents cannot have observations.")

    def _validate_repeat_levels(self):
        R = self.y.shape[2]
        if self.repeat_levels is None or self.repeat_levels.shape != (R,):
            raise ValueError(f"repeat_levels must have shape ({R},).")
        if (
            bool(jnp.any(self.repeat_levels < 0))
            or not bool(jnp.allclose(self.repeat_levels, jnp.round(self.repeat_levels)))
            or jnp.unique(self.repeat_levels).size != R
        ):
            raise ValueError("repeat_levels must be unique non-negative integers.")

    def _configure_rate_prior(self):
        rate_prior = self.priors["rate"]
        if hasattr(rate_prior, "set_age_bounds"):
            rate_prior.set_age_bounds(
                int(jnp.min(self.age_values)), int(jnp.max(self.age_values))
            )
        if getattr(rate_prior, "prior_type", None) != "global":
            if hasattr(rate_prior, "set_event_dim") and getattr(
                rate_prior, "event_dim", None
            ) is None:
                rate_prior.set_event_dim(1)
            if (
                hasattr(rate_prior, "set_loc")
                and getattr(rate_prior, "loc", None) is None
            ):
                rate_prior.set_loc(0.0)

    def _sample_rate_prior(self):
        with scope(prefix="rate"):
            surface = self.priors["rate"].sample()
        if surface.ndim == 3 and surface.shape[0] == 1:
            surface = surface[0]
        expected = (len(self.P), len(self.P))
        if surface.shape != expected:
            raise ValueError(
                f"priors['rate'].sample() must return {expected}, got {surface.shape}."
            )
        return surface

    def _log_contact_rate(self, log_gamma):
        """Integrate true contact age through the reporting operator."""
        terms = (
            log_gamma[:, :, None]
            + self.log_P[None, :, None]
            + _log_weights(self.Q)[None, :, :]
        )
        return logsumexp(terms, axis=1)

    def _log_reported_volume(self, log_contact_rate):
        """Aggregate population contact volume over respondent groups."""
        fine_volume = log_contact_rate + self.log_P[:, None]
        terms = jnp.where(
            self.respondent_membership.T[:, :, None],
            fine_volume[None, :, :],
            -jnp.inf,
        )
        return logsumexp(terms, axis=1)

    def _sample_log_completeness(self):
        kappa = numpyro.sample("fatigue_kappa", dist.HalfNormal(1.0))
        zeta = numpyro.sample("fatigue_zeta", dist.Normal(0.0, 1.0))
        eta = numpyro.sample("fatigue_eta", dist.Exponential(1.0))
        scaled_repeat = jnp.exp(zeta) * self.repeat_levels**eta
        log_completeness = numpyro.deterministic(
            "log_completeness",
            -kappa * scaled_repeat / (1.0 + scaled_repeat),
        )
        numpyro.deterministic("completeness", jnp.exp(log_completeness))
        return log_completeness

    def _log_mean(self, log_contact_rate, reported_log_volume):
        if self.y.ndim == 2:
            active = self.N_c > 0
            candidate = (
                reported_log_volume
                + jnp.log(jnp.where(active, self.N_c, 1))[:, None]
                - self.log_P_respondent[:, None]
            )
            return jnp.where(active[:, None], candidate, -jnp.inf)

        log_completeness = self._sample_log_completeness()
        if self.N_age_repeat is None:
            active = self.N_c > 0
            candidate = (
                reported_log_volume[:, :, None]
                + jnp.log(jnp.where(active, self.N_c, 1))[:, None, :]
                - self.log_P_respondent[:, None, None]
                + log_completeness[None, None, :]
            )
            return jnp.where(active[:, None, :], candidate, -jnp.inf)

        exact_active = self.N_age_repeat > 0
        age_repeat_terms = (
            log_contact_rate[:, :, None]
            + jnp.log(jnp.where(exact_active, self.N_age_repeat, 1))[:, None, :]
        )
        grouped_terms = jnp.where(
            self.respondent_membership.T[:, :, None, None]
            & exact_active[None, :, None, :],
            age_repeat_terms[None, :, :, :],
            # Keep the inactive branch finite so logsumexp has a well-defined
            # derivative for respondent/repeat groups with no exposure.
            jnp.asarray(-1e30, dtype=age_repeat_terms.dtype),
        )
        candidate = (
            logsumexp(grouped_terms, axis=1) + log_completeness[None, None, :]
        )
        return jnp.where((self.N_c > 0)[:, None, :], candidate, -jnp.inf)

    def model(self, y=None):
        baseline = numpyro.sample("baseline", dist.Normal(-jnp.mean(self.log_P), 2.5))
        sampled_surface = self._sample_rate_prior()
        reciprocal_surface = jnp.tril(sampled_surface) + jnp.tril(
            sampled_surface, -1
        ).T
        log_gamma = numpyro.deterministic(
            "log_gamma", baseline + reciprocal_surface
        )

        log_contact_rate = self._log_contact_rate(log_gamma)
        aggregated_log_intensity = numpyro.deterministic(
            "aggregated_log_intensity",
            self._log_reported_volume(log_contact_rate),
        )
        log_mu = numpyro.deterministic(
            "log_mu", self._log_mean(log_contact_rate, aggregated_log_intensity)
        )
        mu = numpyro.deterministic("mu", jnp.exp(log_mu))

        observed = None if y is None else jnp.asarray(y).reshape(-1)
        if self.y.ndim == 2:
            active = jnp.broadcast_to((self.N_c > 0)[:, None], mu.shape)
        else:
            active = jnp.broadcast_to((self.N_c > 0)[:, None, :], mu.shape)
        # NumPyro's NegativeBinomial2 requires a strictly positive mean. Empty
        # respondent/repeat strata have a structural mean of zero, so give the
        # base distribution a harmless positive placeholder and mask those cells
        # out of the likelihood. Callers should also zero masked predictive cells.
        distribution_mean = jnp.where(active, mu, jnp.asarray(1e-12, mu.dtype))
        if self.likelihood == "negbin":
            inv_disp = numpyro.sample(
                "inv_disp", dist.Exponential(self.inv_disp_prior_rate)
            )
            concentration = numpyro.deterministic("concentration", 1.0 / inv_disp)
            observation_dist = dist.NegativeBinomial2(
                mean=distribution_mean.reshape(-1), concentration=concentration
            )
        else:
            observation_dist = dist.Poisson(rate=distribution_mean.reshape(-1))

        with numpyro.plate("data", mu.size):
            with mask(mask=active.reshape(-1)):
                numpyro.sample("obs", observation_dist, obs=observed)

    def run_inference_svi(
        self,
        prng_key: jax.random.PRNGKey,
        guide: callable,
        num_steps: int = 5_000,
        peak_lr: float = 0.01,
        **model_kwargs,
    ):
        """Fit the model with stochastic variational inference."""
        self.guide = guide
        model_kwargs.setdefault("y", self.y)
        self.svi = run_inference_svi(
            prng_key=prng_key,
            model=self.model,
            guide=guide,
            num_steps=num_steps,
            peak_lr=peak_lr,
            **model_kwargs,
        )

    def posterior_predictive_svi(
        self,
        prng_key,
        guide: callable,
        num_samples: int = 5_000,
        **model_kwargs,
    ) -> dict[str, jax.Array]:
        """Draw posterior predictive samples after SVI."""
        if not hasattr(self, "svi"):
            raise AttributeError("run_inference_svi must be run first.")
        return posterior_predictive_svi(
            prng_key,
            self.model,
            guide,
            self.svi.params,
            num_samples=num_samples,
            **model_kwargs,
        )


__all__ = ["AgeMixAggReported"]
