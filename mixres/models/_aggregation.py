"""Composition matrices, allocation weights and log-scale age aggregation.

Notation used throughout the ``AgeMixAgg`` model family:

* ``A`` fine ages, ``C`` respondent groups, ``D`` recorded contact groups.
* ``comp_map_cnt`` (A x D) is the binary composition matrix mapping fine
  contact ages onto the recorded contact groups.
* ``comp_map_resp`` (A x C) is the binary composition matrix mapping fine
  respondent ages onto the respondent groups.
* Allocation weights ``Q`` (A x D) give the probability that a contact of
  true age ``b`` is recorded in group ``d``.
* An axis is aggregated only when its composition matrix is given; ``None``
  keeps it at fine ages.

Aggregation pipeline
--------------------
The models observe the expected contact volume ``EZ[c, d]`` between
respondent group ``c`` and contact group ``d``, built from the fine-age
contact rates ``gamma`` in two steps::

    m[a, d]  = sum_b gamma[a, b] P[b] Q[b, d]
    EZ[c, d] = sum_{a in c} P[a] m[a, d]

``m[a, d]`` is the contact intensity from fine respondent age ``a`` into
contact group ``d``. :func:`log_expected_volume` computes ``log EZ``.
"""

from typing import Optional

import jax.numpy as jnp
from jax import Array
from jax.scipy.special import logsumexp
from jax.typing import ArrayLike


def make_composition_mapping(
    coords: ArrayLike,
    left_limits: ArrayLike,
    right_limits: ArrayLike,
) -> Array:
    """Binary matrix assigning fine grid points to coarse groups.

    Entry ``[a, c]`` is ``True`` when ``coords[a]`` lies in the half-open
    interval ``[left_limits[c], right_limits[c])``. Groups may overlap, so a
    point can belong to several groups. For a length-A vector ``x``,
    ``x @ comp_map`` returns the C group totals.

    Parameters
    ----------
    coords : array-like, shape (A,)
        Fine-resolution grid points, e.g. single-year ages.
    left_limits : array-like, shape (C,)
        Inclusive lower bound of each group.
    right_limits : array-like, shape (C,)
        Exclusive upper bound of each group.

    Returns
    -------
    Array of bool, shape (A, C)
        Group membership of each grid point.

    Examples
    --------
    >>> make_composition_mapping(jnp.arange(4), jnp.array([0, 2]), jnp.array([2, 4])).astype(int)
    Array([[1, 0],
           [1, 0],
           [0, 1],
           [0, 1]], dtype=int32)
    """
    left_limits, right_limits = jnp.asarray(left_limits), jnp.asarray(right_limits)
    if left_limits.shape != right_limits.shape:
        raise ValueError("left_limits and right_limits must have the same shape.")
    coords = jnp.asarray(coords)[:, None]
    return (coords >= left_limits) & (coords < right_limits)


def uniform_allocation_weights(comp_map_cnt: ArrayLike, dtype=jnp.float32) -> Array:
    """Spread each fine contact age equally over the contact groups containing it.

    ``Q[b, d] = comp_map_cnt[b, d] / sum_j comp_map_cnt[b, j]``, so an age
    covered by ``k`` overlapping groups contributes ``1/k`` of its contacts to
    each of them.

    Parameters
    ----------
    comp_map_cnt : array-like, shape (A, D)
        Composition matrix of the contact age dimension.
    dtype : optional
        Floating dtype of the result (default ``jnp.float32``).

    Returns
    -------
    Array, shape (A, D)
        Row-stochastic allocation weights ``Q``.

    Raises
    ------
    ValueError
        If some fine contact age belongs to no group.
    """
    membership = jnp.asarray(comp_map_cnt) != 0
    coverage = membership.sum(axis=1, keepdims=True)
    if bool(jnp.any(coverage == 0)):
        raise ValueError(
            "Every fine contact age must belong to at least one group in comp_map_cnt."
        )
    return membership.astype(dtype) / coverage.astype(dtype)


def validate_allocation_weights(
    allocation_weights: ArrayLike, comp_map_cnt: Optional[Array], A: int
) -> Array:
    """Check user-supplied allocation weights against ``comp_map_cnt``.

    The weights must be finite, non-negative and row-stochastic, and may only
    put mass on groups that contain the age according to ``comp_map_cnt``.

    Parameters
    ----------
    allocation_weights : array-like, shape (A, D)
        Candidate allocation weights ``Q``. Shape (A, A) when
        ``comp_map_cnt`` is ``None``.
    comp_map_cnt : Array, shape (A, D), optional
        Composition matrix of the contact age dimension.
    A : int
        Number of fine ages.

    Returns
    -------
    Array, shape (A, D)
        The validated weights as ``float32``.

    Raises
    ------
    ValueError
        If any of the conditions above is violated.
    """
    Q = jnp.asarray(allocation_weights, dtype=jnp.float32)
    expected_shape = (A, A) if comp_map_cnt is None else comp_map_cnt.shape
    if Q.shape != expected_shape:
        raise ValueError(f"allocation_weights must have shape {expected_shape}.")
    if not bool(jnp.all(jnp.isfinite(Q))) or bool(jnp.any(Q < 0)):
        raise ValueError(
            "allocation_weights must contain finite, non-negative probabilities."
        )
    if comp_map_cnt is not None and bool(jnp.any((comp_map_cnt == 0) & (Q != 0))):
        raise ValueError(
            "allocation_weights can assign mass only to groups present in comp_map_cnt."
        )
    if not bool(jnp.allclose(Q.sum(axis=1), 1.0)):
        raise ValueError("Every row of allocation_weights must sum to one.")
    return Q


def safe_log(values: ArrayLike) -> Array:
    """Elementwise log that maps zeros to ``-inf`` without evaluating ``log(0)``.

    Although ``where`` returns ``-inf`` on the inactive branch, the derivative
    of an evaluated ``log(0)`` can still poison SVI gradients.

    Parameters
    ----------
    values : array-like
        Non-negative values.

    Returns
    -------
    Array
        ``log(values)`` with ``-inf`` where ``values`` is zero.
    """
    values = jnp.asarray(values)
    positive = values > 0
    return jnp.where(positive, jnp.log(jnp.where(positive, values, 1)), -jnp.inf)


def log_expected_volume(
    log_gamma: ArrayLike,
    log_P: ArrayLike,
    allocation_weights: Optional[ArrayLike] = None,
    comp_map_resp: Optional[ArrayLike] = None,
) -> Array:
    """Log expected contact volume between respondent and contact groups.

    Computes ``log m[a, d] = log sum_b gamma[a, b] P[b] Q[b, d]`` and then
    ``log EZ[c, d] = log sum_{a in c} P[a] m[a, d]``. Pass ``comp_map_cnt``
    as ``allocation_weights`` for binary contact aggregation.
    Passing ``None`` for either matrix keeps that axis at fine ages.

    Parameters
    ----------
    log_gamma : array-like, shape (A, A)
        Fine-age log contact rates.
    log_P : array-like, shape (A,)
        Log population age distribution.
    allocation_weights : array-like, shape (A, D), optional
        Allocation weights ``Q``; ``None`` keeps fine contact ages (``D = A``).
    comp_map_resp : array-like, shape (A, C), optional
        Respondent composition matrix; ``None`` keeps fine respondent ages
        (``C = A``).

    Returns
    -------
    Array, shape (C, D)
        ``log EZ[c, d]``.
    """
    log_gamma, log_P = jnp.asarray(log_gamma), jnp.asarray(log_P)
    if allocation_weights is None:
        log_m = log_gamma + log_P[None, :]
    else:
        log_m = logsumexp(
            log_gamma[:, :, None]
            + log_P[None, :, None]
            + safe_log(allocation_weights)[None, :, :],
            axis=1,
        )
    log_EZ = log_m + log_P[:, None]
    if comp_map_resp is None:
        return log_EZ
    log_comp_map_resp = safe_log(jnp.asarray(comp_map_resp, dtype=jnp.float32))
    return logsumexp(log_EZ[None, :, :] + log_comp_map_resp.T[:, :, None], axis=1)


def sum_EZ(
    EZ: ArrayLike,
    comp_map_cnt: Optional[ArrayLike] = None,
    comp_map_resp: Optional[ArrayLike] = None,
) -> Array:
    """Sum a fine-age expected contact matrix over age groups.

    Computes ``sum_{a in c} sum_{b in d} EZ[a, b]``, the linear-scale
    counterpart of :func:`log_expected_volume`. An axis is aggregated only
    when its composition matrix is given; otherwise it stays at fine ages.

    Parameters
    ----------
    EZ : array-like, shape (A, A)
        Fine-age expected contacts; rows index respondent ages and columns
        contact ages.
    comp_map_cnt : array-like, shape (A, D), optional
        Contact composition matrix; ``None`` keeps fine contact ages
        (``D = A``).
    comp_map_resp : array-like, shape (A, C), optional
        Respondent composition matrix; ``None`` keeps fine respondent ages
        (``C = A``). Pass ``comp_map_cnt`` here too to group both axes
        identically.

    Returns
    -------
    Array, shape (C, D)
        Grouped expected contacts.
    """
    EZ = jnp.asarray(EZ)
    if comp_map_cnt is not None:
        EZ = EZ @ jnp.asarray(comp_map_cnt)
    if comp_map_resp is not None:
        EZ = jnp.asarray(comp_map_resp).T @ EZ
    return EZ
