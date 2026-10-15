import jax.numpy as jnp
from jax.scipy.special import logsumexp


def age_membership(age_values, left_limits, right_limits, dtype=jnp.float32):
    """Return a group-by-age membership matrix for half-open intervals."""
    age_values = jnp.asarray(age_values)
    left_limits = jnp.asarray(left_limits)
    right_limits = jnp.asarray(right_limits)

    in_group = (age_values[None, :] >= left_limits[:, None]) & (
        age_values[None, :] < right_limits[:, None]
    )
    return in_group.astype(dtype)


def make_W(age_values, contact_left, contact_right, dtype=jnp.float32):
    """Build an age aggregation matrix with shape A x C."""
    return age_membership(age_values, contact_left, contact_right, dtype=dtype).T


def aggregate_Z(Z, W=None, aggregation="none", W_respondent=None):
    """Aggregate a fine matrix, optionally with distinct respondent groups."""
    if aggregation not in {"none", "contact", "both"}:
        raise ValueError("aggregation must be one of 'none', 'contact', or 'both'.")
    if aggregation == "none":
        return Z
    if W is None:
        raise ValueError("W is required when aggregation is 'contact' or 'both'.")
    if aggregation == "contact":
        contact_aggregated = Z @ W
        return (
            contact_aggregated
            if W_respondent is None
            else W_respondent.T @ contact_aggregated
        )
    if aggregation == "both":
        respondent_W = W if W_respondent is None else W_respondent
        return respondent_W.T @ Z @ W


def aggregate_log_intensity(
    log_gamma, log_P, W=None, aggregation="none", W_respondent=None
):
    """
    Aggregate fine log contact intensity over binary age groups.

    Returns shape A x A, A x C, or C x C for aggregation modes ``none``,
    ``contact``, or ``both``, respectively.
    """
    log_gamma = jnp.asarray(log_gamma)
    log_P = jnp.asarray(log_P)
    fine_log_intensity = log_gamma + log_P[:, None] + log_P[None, :]

    if aggregation == "none":
        return fine_log_intensity
    if aggregation not in {"contact", "both"}:
        raise ValueError("aggregation must be one of 'none', 'contact', or 'both'.")
    if W is None:
        raise ValueError("W is required when aggregation is 'contact' or 'both'.")

    W = jnp.asarray(W).astype(bool)
    neg_inf = jnp.asarray(-jnp.inf, dtype=log_gamma.dtype)
    contact_terms = jnp.where(
        W[None, :, :],
        fine_log_intensity[:, :, None],
        neg_inf,
    )
    contact_aggregated = logsumexp(contact_terms, axis=1)

    if aggregation == "contact" and W_respondent is None:
        return contact_aggregated

    respondent_W = W if W_respondent is None else jnp.asarray(W_respondent)
    respondent_terms = jnp.where(
        respondent_W.T[:, :, None].astype(bool),
        contact_aggregated[None, :, :],
        neg_inf,
    )
    return logsumexp(respondent_terms, axis=1)
