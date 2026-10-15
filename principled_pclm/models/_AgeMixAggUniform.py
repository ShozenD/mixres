"""AgeMixAgg with uniform contact-bin assignment under overlap.

This module leaves the :class:`AgeMixAgg` likelihood unchanged and replaces
only the binary contact-side membership indicator with

    W_uniform[b, d] = W[b, d] / sum_j W[b, j].

Consequently, a fine contact age covered by k recorded bins contributes 1/k
of its intensity to each compatible bin.  Respondent-age aggregation remains
binary.
"""

import jax.numpy as jnp
from jax.scipy.special import logsumexp

import numpyro
import numpyro.distributions as dist

from ._AgeMixAgg import AgeMixAgg, prepare_age_mix_agg_inputs


def make_uniform_W(W, dtype=jnp.float32):
    """Row-normalise binary ``W`` into equal contact-bin probabilities."""
    W = jnp.asarray(W)
    membership = W != 0
    coverage = membership.sum(axis=1, keepdims=True)
    if bool(jnp.any(coverage == 0)):
        raise ValueError("Every fine contact age must belong to at least one bin.")
    return membership.astype(dtype) / coverage.astype(dtype)


def aggregate_uniform_log_intensity(
    log_gamma,
    log_P,
    W_uniform=None,
    aggregation="none",
    W_respondent=None,
):
    """Aggregate with uniform contact weights and binary respondent groups.

    ``W_uniform`` must already contain the row-normalised contact assignment
    probabilities returned by :func:`make_uniform_W`.
    """
    if aggregation not in {"none", "contact", "both"}:
        raise ValueError("aggregation must be one of 'none', 'contact', or 'both'.")

    log_gamma = jnp.asarray(log_gamma)
    log_P = jnp.asarray(log_P)
    fine_log_intensity = log_gamma + log_P[:, None] + log_P[None, :]

    if aggregation == "none":
        return fine_log_intensity
    if W_uniform is None:
        raise ValueError(
            "W_uniform is required when aggregation is 'contact' or 'both'."
        )

    W_uniform = jnp.asarray(W_uniform, dtype=log_gamma.dtype)
    neg_inf = jnp.asarray(-jnp.inf, dtype=log_gamma.dtype)
    log_W_uniform = jnp.where(
        W_uniform > 0,
        jnp.log(W_uniform),
        neg_inf,
    )
    contact_aggregated = logsumexp(
        fine_log_intensity[:, :, None] + log_W_uniform[None, :, :],
        axis=1,
    )

    if aggregation == "contact" and W_respondent is None:
        return contact_aggregated

    # Only the contact side is probabilistically assigned.  The respondent
    # side continues to use hard membership, including when the same bin
    # definitions are shared by both axes.
    respondent_W = (
        W_uniform if W_respondent is None else jnp.asarray(W_respondent)
    )
    respondent_terms = jnp.where(
        respondent_W.T[:, :, None].astype(bool),
        contact_aggregated[None, :, :],
        neg_inf,
    )
    return logsumexp(respondent_terms, axis=1)


class AgeMixAggUniform(AgeMixAgg):
    """``AgeMixAgg`` with equal assignment across compatible contact bins.

    The constructor and prepared-input format are identical to
    :class:`AgeMixAgg`.  Input ``W`` remains the binary membership matrix made
    by :func:`prepare_age_mix_agg_inputs`; ``W_uniform`` is derived from it and
    is the contact-side matrix used in the expected counts.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.W_uniform = None if self.W is None else make_uniform_W(self.W)

    def model(self, y=None):
        """Use the original independent-Poisson model with uniform weights."""
        len_y = self.y.size if y is None else jnp.asarray(y).size

        baseline = numpyro.sample("baseline", dist.Normal(-jnp.mean(self.log_P), 2.5))
        f = self._sample_rate_prior()

        log_gamma = numpyro.deterministic("log_gamma", baseline + f)
        aggregated_log_intensity = numpyro.deterministic(
            "aggregated_log_intensity",
            aggregate_uniform_log_intensity(
                log_gamma,
                self.log_P,
                self.W_uniform,
                aggregation=self.aggregation,
                W_respondent=self.W_respondent,
            ),
        )
        log_mu = numpyro.deterministic(
            "log_mu",
            aggregated_log_intensity
            + self.log_N_c[:, None]
            - self.log_P_respondent[:, None],
        )
        mu = numpyro.deterministic("mu", jnp.exp(log_mu))

        with numpyro.plate("data", len_y):
            numpyro.sample(
                "obs",
                dist.Poisson(rate=mu.reshape(-1)),
                obs=None if y is None else jnp.asarray(y).reshape(-1),
            )


__all__ = [
    "AgeMixAggUniform",
    "aggregate_uniform_log_intensity",
    "make_uniform_W",
    "prepare_age_mix_agg_inputs",
]
