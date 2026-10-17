"""Allocation weights ``Q``: how a contact of true age ``b`` is recorded.

``Q[b, d]`` is the probability that a contact of fine age ``b`` is recorded in
contact age group ``d``. Apart from ``"binary"``, every mode is supported on
the composition matrix ``comp_map`` and row-stochastic. An age covered by a
single group is recorded there with probability one. For an age covered by
``k > 1`` groups, ``Q[b, d]`` is proportional to ``exp(score[d])`` over its
compatible groups, with the score set by the mode. ``n`` are the
``age_group_counts`` and ``L[a, b]`` is ``log_contact_intensity``, the log
intensity of contacts from respondent age ``a`` to contact age ``b``:

* ``"binary"``: ``Q = comp_map`` itself, so the age is counted ``k`` times.
  Not row-stochastic; handled by :class:`AgeMixAgg`, not by this module.
* ``"uniform"``: ``score[d] = 0``, so each group gets ``1/k``.
* ``"empirical"``: ``score[d] = log n[d]``.
* ``"marginal_intensity"``:
  ``score[d] = log n[d] - log sum_{b' in d} sum_a exp(L[a, b'])``.
* ``"conditional_intensity"``: for each respondent age ``a``,
  ``score[a, d] = log n[a, d] - log sum_{b' in d} exp(L[a, b'])``, so ``Q``
  has shape (A, A, D). A row-wise constant in ``L`` (such as respondent
  exposure) cancels. :class:`Allocation` passes counts smoothed by one
  pooled pseudo-contact per age, ``n[a, d] + n[d] / sum_d n[d]``.

Why the conditional counts are smoothed: single-year respondent ages have few
contacts. With raw ``n[a, d]`` a respondent age without contacts, or without
contacts in any group compatible with an ambiguous contact age, would get an
undefined (0/0) row of ``Q``, and a group unseen at age ``a`` would get
probability exactly zero there. The pseudo-contact is the posterior mean under
a Dirichlet prior centred on the pooled group frequencies with unit weight: it
keeps every row defined, makes ages without contacts use the pooled shares,
and barely moves ages with many contacts.

The two intensity modes depend on the latent contact surface and are
recomputed inside ``model()``; they are outcome-derived specifications, not
learned softmaxes. :class:`Allocation` holds the configuration and is built
once, at model construction (running :func:`validate_allocation_inputs`);
:func:`compute_allocation_weights` is traceable and differentiable in
``log_contact_intensity``.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional, get_args

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.special import logsumexp
from jax.typing import ArrayLike

from ._aggregation import safe_log

if TYPE_CHECKING:
    from ._data import AgeMixAggData

AllocationMode = Literal[
    "binary", "uniform", "empirical", "marginal_intensity", "conditional_intensity"
]
#: Row-stochastic modes computed by :func:`compute_allocation_weights`.
WEIGHT_MODES = ("uniform", "empirical", "marginal_intensity", "conditional_intensity")
#: Modes whose ``Q`` depends on the latent intensity and is recomputed in ``model()``.
DYNAMIC_ALLOCATIONS = ("marginal_intensity", "conditional_intensity")


def validate_allocation_inputs(
    comp_map: ArrayLike, mode: str, age_group_counts: Optional[ArrayLike] = None
) -> None:
    """Check the fixed inputs of :func:`compute_allocation_weights` once.

    ``comp_map`` is assumed binary (the models check that); this checks what
    :func:`compute_allocation_weights` additionally relies on.

    Parameters
    ----------
    comp_map : array-like, shape (A, D)
        Binary composition matrix of fine contact ages onto contact groups.
    mode : {"uniform", "empirical", "marginal_intensity", "conditional_intensity"}
        Allocation mode; see the module docstring.
    age_group_counts : array-like, optional
        Raw (unsmoothed) contacts per contact group: shape (D,), or (A, D) by
        respondent age for ``"conditional_intensity"``. Required unless
        ``mode="uniform"``.

    Raises
    ------
    ValueError
        If a fine age or contact group has no support, the counts are
        malformed, or an ambiguous age has no compatible group with a
        positive count (which would make its row of ``Q`` undefined).
    """
    if mode not in WEIGHT_MODES:
        raise ValueError(f"allocation must be one of {WEIGHT_MODES}, got {mode!r}.")
    comp_map = np.asarray(comp_map)
    A, D = comp_map.shape
    coverage = comp_map.sum(axis=1)
    if np.any(coverage == 0) or np.any(comp_map.sum(axis=0) == 0):
        raise ValueError("Every fine age and contact group must have support.")
    if mode == "uniform":
        return

    if age_group_counts is None:
        raise ValueError(f"allocation={mode!r} requires age_group_counts.")
    shape = (A, D) if mode == "conditional_intensity" else (D,)
    counts = np.asarray(age_group_counts, dtype=float)
    if counts.shape != shape or not np.isfinite(counts).all() or np.any(counts < 0):
        raise ValueError(
            f"age_group_counts must have shape {shape} with finite non-negative counts."
        )
    pooled = counts.sum(axis=0) if counts.ndim == 2 else counts
    if mode == "conditional_intensity" and pooled.sum() == 0:
        raise ValueError(
            "conditional_intensity requires positive total age_group_counts."
        )
    if np.any((coverage > 1) & (comp_map @ pooled == 0)):
        raise ValueError("An ambiguous age has zero total compatible group counts.")


def compute_allocation_probs(
    comp_map: ArrayLike,
    mode: str,
    age_group_counts: Optional[ArrayLike] = None,
    log_contact_intensity: Optional[ArrayLike] = None,
) -> Array:
    """Allocation weights ``Q`` for validated inputs; traceable.

    Parameters
    ----------
    comp_map : array-like, shape (A, D)
        Binary composition matrix of fine contact ages onto contact groups.
    mode : {"uniform", "empirical", "marginal_intensity", "conditional_intensity"}
        Allocation mode; see the module docstring.
    age_group_counts : array-like, optional
        Contacts per contact group, shape (D,); for ``"conditional_intensity"``
        the (smoothed) counts by respondent age, shape (A, D). Unused by
        ``"uniform"``.
    log_contact_intensity : array-like, shape (A, A), optional
        ``L[a, b]``, log intensity of contacts from respondent age ``a`` to
        contact age ``b``; required by the intensity modes.

    Returns
    -------
    Array
        ``Q`` with shape (A, D), or (A, A, D) indexed by respondent age first
        for ``"conditional_intensity"``.
    """
    support = jnp.asarray(comp_map, dtype=bool)
    scores = (
        jnp.zeros(support.shape[1])
        if mode == "uniform"
        else safe_log(age_group_counts)  # (D,) or (A, D)
    )
    if mode in DYNAMIC_ALLOCATIONS:
        log_intensity = jnp.asarray(log_contact_intensity)  # (A, A)
        if mode == "marginal_intensity":
            log_intensity = logsumexp(log_intensity, axis=0)  # (A,)
        # Subtract log sum_{b in d} exp(log_intensity[..., b]) for each group d.
        scores = scores - logsumexp(
            jnp.where(support, log_intensity[..., :, None], -jnp.inf), axis=-2
        )
    # Single-group ages are pinned to logit 0 so that an unobserved group
    # (score -inf) cannot turn their one-candidate row into NaN.
    unique = support.sum(axis=1, keepdims=True) == 1
    logits = jnp.where(support, jnp.where(unique, 0.0, scores[..., None, :]), -jnp.inf)
    return jnp.exp(logits - logsumexp(logits, axis=-1, keepdims=True))


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


@dataclass(frozen=True)
class Allocation:
    """Allocation configuration of :class:`AgeMixAgg`, fixed at construction.

    Attributes
    ----------
    mode : str
        One of :data:`AllocationMode`, or ``"custom"`` for user weights.
    support : Array, shape (A, D)
        Binary support of ``Q``: ``comp_map_cnt``, or the identity for fine
        contact ages.
    weights : Array, shape (A, D), or None
        Static ``Q``; ``None`` for the dynamic modes, whose ``Q`` is computed
        in ``model()``.
    age_group_counts : Array or None
        Counts passed to :func:`compute_allocation_weights`: shape (D,), or
        the smoothed (A, D) counts for ``"conditional_intensity"``. ``None``
        for ``"binary"`` and ``"custom"``.
    group_active : Array of bool, shape (D,)
        Contact groups that can receive allocation mass.
    """

    mode: str
    support: Array
    weights: Optional[Array]
    age_group_counts: Optional[Array]
    group_active: Array

    @classmethod
    def build(
        cls,
        data: "AgeMixAggData",
        mode: Optional[str] = None,
        allocation_weights: Optional[ArrayLike] = None,
        age_group_counts: Optional[ArrayLike] = None,
    ) -> "Allocation":
        """Resolve and validate the allocation for ``data``.

        Parameters
        ----------
        data : AgeMixAggData
            Validated model inputs.
        mode : AllocationMode, optional
            Allocation rule; defaults to ``"uniform"``. Cannot be combined
            with ``allocation_weights``.
        allocation_weights : array-like, shape (A, D), optional
            Custom row-stochastic ``Q``; sets ``mode="custom"``.
        age_group_counts : array-like, optional
            Raw contacts per contact group for the count-based modes: shape
            (D,), or (A, D) by respondent age for ``"conditional_intensity"``.
            Defaults to ``y`` summed over respondents (and repeats), or over
            repeats for ``"conditional_intensity"``.

        Raises
        ------
        ValueError
            If the mode or the counts are invalid, or a contact group with
            positive observations would receive no allocation mass.
        """
        A = data.n_ages
        if allocation_weights is not None:
            if mode is not None:
                raise ValueError(
                    "Supply either allocation_weights or allocation, not both."
                )
            mode = "custom"
        elif mode is None:
            mode = "uniform"
        elif mode not in get_args(AllocationMode):
            raise ValueError(
                f"allocation must be one of {get_args(AllocationMode)}, got {mode!r}."
            )

        # Dense counts; cells without a row (unexposed strata) are zero.
        y = data.to_grid(data.y)
        support = jnp.eye(A) if data.comp_map_cnt is None else data.comp_map_cnt
        observed = y.sum(axis=tuple(i for i in range(y.ndim) if i != 1))
        counts = (
            None
            if age_group_counts is None
            else jnp.asarray(age_group_counts, dtype=jnp.float32)
        )

        if mode in ("custom", "binary"):
            weights = (
                validate_allocation_weights(allocation_weights, data.comp_map_cnt, A)
                if mode == "custom"
                else support
            )
            counts, support_Q = None, weights
        else:
            conditional = mode == "conditional_intensity"
            if conditional:
                _check_single_year_respondents(data)
            if mode == "marginal_intensity" and not bool(jnp.any(data.N_c > 0)):
                raise ValueError(
                    "marginal_intensity requires positive respondent exposure."
                )
            if counts is None:
                by_age = y.sum(axis=2) if y.ndim == 3 else y
                counts = by_age if conditional else observed
            validate_allocation_inputs(support, mode, counts)
            pooled = counts.sum(axis=0) if conditional else counts
            if conditional:
                # One pooled pseudo-contact per respondent age (see module docstring).
                counts = counts + pooled / pooled.sum()
            weights = (
                None
                if mode in DYNAMIC_ALLOCATIONS
                else compute_allocation_probs(support, mode, counts)
            )
            # Dynamic Q keeps the positive support of the empirical Q.
            support_Q = (
                weights
                if weights is not None
                else compute_allocation_probs(support, "empirical", pooled)
            )

        group_active = jnp.any(support_Q > 0, axis=0)
        if bool(jnp.any(~group_active & (observed > 0))):
            raise ValueError(
                "Positive observations cannot have zero allocation probability."
            )
        return cls(mode, support, weights, counts, group_active)


def _check_single_year_respondents(data: "AgeMixAggData") -> None:
    A = data.n_ages
    ages = data.age_values
    if (
        data.comp_map_resp.shape != (A, A)
        or not bool(jnp.all(data.comp_map_resp == jnp.eye(A, dtype=bool)))
        or not bool(jnp.all(ages == jnp.round(ages)))
        or not bool(jnp.all(jnp.diff(ages) == 1))
    ):
        raise ValueError(
            "conditional_intensity requires single-year respondents aligned to "
            "the latent age grid; grouped respondents are unsupported."
        )
