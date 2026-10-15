"""Age-aggregated contact model with an overdispersed total count.

For respondent group ``c``, this module implements the factorisation

    alpha ~ Exponential(lambda)
    Y_cA ~ NegativeBinomial2(mean=N_c m_cA, concentration=1 / alpha)
    Y_c. | Y_cA ~ Multinomial(Y_cA, p_c.),

where ``Y_cA = sum_d Y_cd`` and ``m_cA`` integrates the latent contact
intensity over the complete contact-age domain.  ``m_cd`` is calculated with
a row-normalised reporting matrix, so the interval masses sum to ``m_cA``.
The inverse concentration ``alpha`` gives
``Var(Y_cA) = N_c m_cA + alpha (N_c m_cA)^2``.  This gives a coherent
negative-binomial-total plus multinomial-label model and prevents overlapping
ages from contributing repeatedly to the interval likelihood.
"""

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

import numpyro
import numpyro.distributions as dist
from numpyro.handlers import scope

from ._AgeMixAgg import AgeMixAgg, prepare_age_mix_agg_inputs
from ._aggregation import aggregate_log_intensity
from ._inference import posterior_predictive_svi, run_inference_svi


def _aggregate_weighted_log_intensity(
    log_gamma,
    log_P,
    contact_weights,
    respondent_W=None,
):
    """Aggregate contact ages with non-negative reporting probabilities."""
    fine_log_intensity = (
        jnp.asarray(log_gamma)
        + jnp.asarray(log_P)[:, None]
        + jnp.asarray(log_P)[None, :]
    )
    contact_weights = jnp.asarray(contact_weights)
    neg_inf = jnp.asarray(-jnp.inf, dtype=log_gamma.dtype)
    log_weights = jnp.where(
        contact_weights > 0,
        jnp.log(contact_weights),
        neg_inf,
    )
    contact_aggregated = logsumexp(
        fine_log_intensity[:, :, None] + log_weights[None, :, :],
        axis=1,
    )

    if respondent_W is None:
        return contact_aggregated

    respondent_terms = jnp.where(
        jnp.asarray(respondent_W).T[:, :, None].astype(bool),
        contact_aggregated[None, :, :],
        neg_inf,
    )
    return logsumexp(respondent_terms, axis=1)


def prepare_age_mix_agg_total_inputs(*args, **kwargs):
    """Prepare ``AgeMixAggTotal`` inputs and expose observed group totals.

    Parameters are identical to :func:`prepare_age_mix_agg_inputs`.  The
    returned dictionary additionally contains ``y_total`` with one value per
    respondent group.
    """
    inputs = prepare_age_mix_agg_inputs(*args, **kwargs)
    inputs["y_total"] = jnp.asarray(inputs["y"]).sum(axis=1)
    return inputs


class AgeMixAggTotal(AgeMixAgg):
    """Fine-age contact model using the total-intensity overlap likelihood.

    The constructor extends :class:`AgeMixAgg` with ``W_reporting`` and an
    exponential prior rate for the inverse negative-binomial concentration.
    ``W`` is the binary fine-contact-age by recorded-interval membership matrix
    and is retained for raw overlap diagnostics. ``W_reporting`` contains the
    probability that a contact at each fine age receives each recorded label.
    If omitted, it defaults to row-normalised ``W``.

    Deterministic sites
    -------------------
    ``aggregated_log_intensity``
        Log population contact volume assigned to each unique reported label.
    ``aggregated_raw_interval_log_intensity``
        Log population contact volume integrated with binary ``W``.
    ``aggregated_total_log_intensity``
        Log population contact volume integrated over the entire contact-age
        domain, with shape ``(n_respondent_groups,)``.
    ``log_m_interval`` / ``m_interval``
        Per-respondent unique reported-label intensities ``m_cd``.
    ``log_m_interval_raw`` / ``m_interval_raw``
        Raw overlapping interval intensities calculated with binary ``W``.
    ``log_m_total`` / ``m_total``
        Per-respondent total intensities ``m_cA``.
    ``log_mu_interval`` / ``mu_interval``
        Expected unique reported-label counts ``N_c m_cd``.
    ``log_mu_interval_raw`` / ``mu_interval_raw``
        Raw overlapping ``N_c m_cd`` diagnostics calculated with binary ``W``.
    ``log_mu_total`` / ``mu_total``
        Total negative-binomial means ``N_c m_cA``.

    Notes
    -----
    The observation sites are ``total_obs`` for the negative-binomial total and
    ``obs`` for the conditional multinomial labels. ``inv_disp`` is the global
    inverse concentration, so the total-count variance is
    ``mu_total + inv_disp * mu_total**2``.
    """

    def __init__(
        self,
        y,
        N_c,
        P,
        W=None,
        W_reporting=None,
        W_respondent=None,
        aggregation="none",
        age_values=None,
        priors=None,
        total_count_max=None,
        inv_disp_prior_rate=1.0,
    ):
        self.inv_disp_prior_rate = float(inv_disp_prior_rate)
        if not jnp.isfinite(self.inv_disp_prior_rate) or self.inv_disp_prior_rate <= 0:
            raise ValueError("inv_disp_prior_rate must be finite and positive.")
        self.W_reporting = (
            None
            if W_reporting is None
            else jnp.asarray(W_reporting, dtype=jnp.float32)
        )
        super().__init__(
            y=y,
            N_c=N_c,
            P=P,
            W=W,
            W_respondent=W_respondent,
            aggregation=aggregation,
            age_values=age_values,
            priors=priors,
        )
        if total_count_max is None:
            largest_observed_total = float(jnp.max(self.y_total))
            total_count_max = int(
                jnp.ceil(
                    largest_observed_total
                    + 10.0 * jnp.sqrt(largest_observed_total + 1.0)
                    + 10.0
                )
            )
        self.total_count_max = int(total_count_max)
        if self.total_count_max < int(jnp.max(self.y_total)):
            raise ValueError(
                "total_count_max must be at least the largest observed y total."
            )

    def validate_inputs(self):
        super().validate_inputs()

        if bool(jnp.any(self.y < 0)):
            raise ValueError("y must contain non-negative counts.")
        if not bool(jnp.allclose(self.y, jnp.round(self.y))):
            raise ValueError("y must contain integer-valued counts.")

        if self.W is not None:
            contact_coverage = jnp.sum(self.W != 0, axis=1)
            if bool(jnp.any(contact_coverage == 0)):
                raise ValueError(
                    "Every fine contact age must be covered by at least one "
                    "recorded interval."
                )

            if self.W_reporting is None:
                self.W_reporting = self.W / contact_coverage[:, None]
            if self.W_reporting.shape != self.W.shape:
                raise ValueError("W_reporting must have the same shape as W.")
            if bool(jnp.any(self.W_reporting < 0)):
                raise ValueError("W_reporting must contain non-negative values.")
            if bool(jnp.any((self.W == 0) & (self.W_reporting != 0))):
                raise ValueError(
                    "W_reporting can assign mass only to intervals present in W."
                )
            if not bool(jnp.allclose(self.W_reporting.sum(axis=1), 1.0)):
                raise ValueError("Every row of W_reporting must sum to one.")
        elif self.W_reporting is not None:
            raise ValueError("W_reporting must be None when aggregation='none'.")

        self.y_total = self.y.sum(axis=1)

    @classmethod
    def from_prepared_inputs(cls, inputs, **kwargs):
        """Construct the model from a prepared input dictionary."""
        model = cls(
            y=inputs["y"],
            N_c=inputs["N_c"],
            P=inputs["P"],
            W=inputs["W"],
            W_reporting=inputs.get("W_reporting"),
            W_respondent=inputs.get("W_respondent"),
            aggregation=inputs["aggregation"],
            age_values=inputs.get("age_values"),
            **kwargs,
        )

        if "y_total" in inputs and not bool(
            jnp.allclose(jnp.asarray(inputs["y_total"]), model.y_total)
        ):
            raise ValueError("inputs['y_total'] must equal inputs['y'].sum(axis=1).")
        return model

    def _sample_rate_prior(self):
        """Sample and validate the fine-age log-rate prior."""
        with scope(prefix="rate"):
            f = self.priors["rate"].sample()

        if f.ndim == 3 and f.shape[0] == 1:
            f = f[0]

        expected_shape = (self.P.shape[0], self.P.shape[0])
        if f.shape != expected_shape:
            raise ValueError(
                f"priors['rate'].sample() must return shape {expected_shape} "
                f"or (1, {expected_shape[0]}, {expected_shape[1]}), got {f.shape}."
            )
        return f

    def _aggregate_total_log_intensity(self, log_gamma):
        """Integrate over all contact ages and the configured respondent bins."""
        all_contact_ages = jnp.ones(
            (self.P.shape[0], 1), dtype=log_gamma.dtype
        )

        respondent_W = self.W_respondent
        if self.aggregation == "both" and respondent_W is None:
            respondent_W = self.W

        total_log_intensity = aggregate_log_intensity(
            log_gamma,
            self.log_P,
            all_contact_ages,
            aggregation="contact",
            W_respondent=respondent_W,
        )
        return total_log_intensity[:, 0]

    def _aggregate_reporting_log_intensity(self, log_gamma):
        """Aggregate with the unique-label reporting probabilities."""
        if self.aggregation == "none":
            return aggregate_log_intensity(log_gamma, self.log_P)

        respondent_W = self.W_respondent
        if self.aggregation == "both" and respondent_W is None:
            respondent_W = self.W
        return _aggregate_weighted_log_intensity(
            log_gamma,
            self.log_P,
            self.W_reporting,
            respondent_W=respondent_W,
        )

    def model(self, y=None):
        """Define the negative-binomial-total plus interval likelihood."""
        baseline = numpyro.sample(
            "baseline", dist.Normal(-jnp.mean(self.log_P), 2.5)
        )
        f = self._sample_rate_prior()

        inv_disp = numpyro.sample(
            "inv_disp",
            dist.Exponential(rate=self.inv_disp_prior_rate),
        )
        concentration = numpyro.deterministic(
            "concentration",
            1.0 / inv_disp,
        )

        log_gamma = numpyro.deterministic("log_gamma", baseline + f)

        aggregated_raw_interval_log_intensity = numpyro.deterministic(
            "aggregated_raw_interval_log_intensity",
            aggregate_log_intensity(
                log_gamma,
                self.log_P,
                self.W,
                aggregation=self.aggregation,
                W_respondent=self.W_respondent,
            ),
        )
        aggregated_log_intensity = numpyro.deterministic(
            "aggregated_log_intensity",
            self._aggregate_reporting_log_intensity(log_gamma),
        )
        aggregated_total_log_intensity = numpyro.deterministic(
            "aggregated_total_log_intensity",
            self._aggregate_total_log_intensity(log_gamma),
        )

        log_m_interval = numpyro.deterministic(
            "log_m_interval",
            aggregated_log_intensity - self.log_P_respondent[:, None],
        )
        m_interval = numpyro.deterministic(
            "m_interval", jnp.exp(log_m_interval)
        )
        log_m_interval_raw = numpyro.deterministic(
            "log_m_interval_raw",
            aggregated_raw_interval_log_intensity
            - self.log_P_respondent[:, None],
        )
        numpyro.deterministic(
            "m_interval_raw", jnp.exp(log_m_interval_raw)
        )
        log_m_total = numpyro.deterministic(
            "log_m_total",
            aggregated_total_log_intensity - self.log_P_respondent,
        )
        m_total = numpyro.deterministic("m_total", jnp.exp(log_m_total))

        log_mu_interval = numpyro.deterministic(
            "log_mu_interval", log_m_interval + self.log_N_c[:, None]
        )
        mu_interval = numpyro.deterministic(
            "mu_interval", jnp.exp(log_mu_interval)
        )
        log_mu_interval_raw = numpyro.deterministic(
            "log_mu_interval_raw",
            log_m_interval_raw + self.log_N_c[:, None],
        )
        numpyro.deterministic(
            "mu_interval_raw", jnp.exp(log_mu_interval_raw)
        )
        log_mu_total = numpyro.deterministic(
            "log_mu_total", log_m_total + self.log_N_c
        )
        mu_total = numpyro.deterministic("mu_total", jnp.exp(log_mu_total))

        # Backward-compatible aliases now refer to unique reported labels.
        numpyro.deterministic("log_mu", log_mu_interval)
        numpyro.deterministic("mu", mu_interval)

        interval_probabilities = numpyro.deterministic(
            "interval_probabilities",
            jax.nn.softmax(log_m_interval, axis=1),
        )

        observed_y = None if y is None else jnp.asarray(y, dtype=jnp.float32)
        observed_total = (
            None if observed_y is None else observed_y.sum(axis=1)
        )

        with numpyro.plate("respondent_group", self.y.shape[0]):
            sampled_total = numpyro.sample(
                "total_obs",
                dist.NegativeBinomial2(
                    mean=mu_total,
                    concentration=concentration,
                ),
                obs=observed_total,
            )
            numpyro.sample(
                "obs",
                dist.Multinomial(
                    total_count=sampled_total,
                    probs=interval_probabilities,
                    total_count_max=self.total_count_max,
                ),
                obs=observed_y,
            )

        if observed_y is not None:
            numpyro.deterministic(
                "interval_log_likelihood",
                dist.Multinomial(
                    total_count=observed_total,
                    probs=interval_probabilities,
                    total_count_max=self.total_count_max,
                ).log_prob(observed_y).sum(),
            )

    def run_inference_svi(
        self,
        prng_key: jax.random.PRNGKey,
        guide: callable,
        num_steps: int = 5_000,
        peak_lr: float = 0.01,
        **model_kwargs,
    ):
        """Run SVI while conditioning on this model's observed interval counts."""
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
        """Generate total counts and unique interval-label counts."""
        if hasattr(self, "svi") is False:
            raise AttributeError("run_inference_svi must be run first.")

        return posterior_predictive_svi(
            prng_key,
            self.model,
            guide,
            self.svi.params,
            num_samples=num_samples,
            **model_kwargs,
        )


__all__ = ["AgeMixAggTotal", "prepare_age_mix_agg_total_inputs"]
