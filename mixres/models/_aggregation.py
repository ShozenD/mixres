"""Composition matrices, allocation weights and log-scale age aggregation.

Notation used throughout :class:`AgeMixAgg`:

* ``A`` fine ages, ``C`` respondent groups, ``D`` recorded contact groups.
* ``comp_map_cnt`` (A x D) is the binary composition matrix mapping fine
  contact ages onto the recorded contact groups.
* ``comp_map_resp`` (A x C) is the binary composition matrix mapping fine
  respondent ages onto the respondent groups.
* Allocation weights ``Q`` (A x D) give the probability that a contact of
  true age ``b`` is recorded in group ``d``; they are built in
  ``_allocation.py``.
* An axis is aggregated only when its composition matrix is given; ``None``
  keeps it at fine ages.

Aggregation pipeline
--------------------
The model observes the expected contact volume ``EZ[c, d]`` between
respondent group ``c`` and contact group ``d``, built from the fine-age
contact rates ``gamma`` in two steps::

    m[a, d]  = sum_b gamma[a, b] P[b] Q[b, d]
    EZ[c, d] = sum_{a in c} P[a] m[a, d]

``m[a, d]`` is the contact intensity from fine respondent age ``a`` into
contact group ``d``. Both steps are computed on the log scale in
:meth:`AgeMixAgg.model`; :func:`sum_EZ` aggregates a fine ``EZ`` matrix.
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


def sum_EZ(
    EZ: ArrayLike,
    comp_map_cnt: Optional[ArrayLike] = None,
    comp_map_resp: Optional[ArrayLike] = None,
) -> Array:
    """Sum a fine-age expected contact matrix over age groups.

    Computes ``sum_{a in c} sum_{b in d} EZ[a, b]``. An axis is aggregated
    only when its composition matrix is given; otherwise it stays at fine
    ages.

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
