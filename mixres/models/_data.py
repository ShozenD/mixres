"""Validated model inputs for :class:`AgeMixAgg`.

:class:`AgeMixAggDataLoader` checks the raw arrays once and ``load()`` returns
an immutable :class:`AgeMixAggData` container, mirroring cntmosaic's
``ContactSurveyLoader`` / ``ModelData`` split. As in cntmosaic, observations
are long-format rows: one per (respondent group, contact group[, repeat level])
cell with respondents, so unexposed cells (structural zeros) never reach the
model. The checks are the private functions below; the loader only
orchestrates them.
"""

from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Tuple

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from ._aggregation import safe_log


def _as_float(x: Optional[ArrayLike]) -> Optional[Array]:
    return None if x is None else jnp.asarray(x, dtype=jnp.float32)


@dataclass(frozen=True)
class AgeMixAggData:
    """Flat container of validated :class:`AgeMixAgg` inputs.

    ``A`` fine ages, ``C`` respondent groups, ``D`` recorded contact groups,
    ``R`` repeat levels and ``n_obs`` observation rows. Row ``i`` is a cell of
    the (C, D[, R]) grid: respondent group ``cid[i]``, contact group
    ``did[i]`` and repeat level ``rid[i]``.

    Attributes
    ----------
    y : Array, shape (n_obs,)
        Integer contact counts per row.
    cid, did : Array of int32, shape (n_obs,)
        Respondent and contact group of each row.
    rid : Array of int32, shape (n_obs,), or None
        Repeat level (number of previous survey waves) of each row, as in
        cntmosaic; indexes the Hill effect directly. ``None`` for
        two-dimensional input.
    log_row_exposure : Array, shape (n_obs, A)
        ``log w[i, a]``, the respondent exposure of fine age ``a`` in row
        ``i``'s stratum, so that ``mu[i] = sum_a w[i, a] m[a, did[i]]``:
        ``P[a] N_c[c(, r)] / P_resp[c]`` for ``a`` in group ``c``, or
        ``N_ar[a, r]`` when exact-age exposure is given; ``-inf`` for ages
        outside the group.
    age_exposure : Array, shape (A,) or (A, L)
        Respondent exposure by fine age and, for 3-D input, by repeat level
        ``0, ..., L - 1 = max(rid)`` (zero for unobserved levels): ``N_ar``,
        or ``N_c`` spread over each group's ages in proportion to ``P``.
    grid_shape : tuple of int
        ``(C, D)`` or ``(C, D, R)``, the shape of the dense input ``y``.
    P : Array, shape (A,)
        Positive population age distribution on the fine grid.
    age_values : Array, shape (A,)
        Fine ages.
    comp_map_cnt : Array, shape (A, D), or None
        Contact composition matrix; ``None`` for fine contact ages.
    comp_map_resp : Array of bool, shape (A, C)
        Respondent composition matrix; the identity for fine respondent ages.
    N_c : Array, shape (C,) or (C, R)
        Respondents per group (and repeat level); derived from ``N_ar`` when
        that is given.
    P_resp : Array, shape (C,)
        Population share of each respondent group.
    log_P : Array, shape (A,)
        ``log P``.
    """

    y: Array
    cid: Array
    did: Array
    rid: Optional[Array]
    log_row_exposure: Array
    age_exposure: Array
    grid_shape: Tuple[int, ...]
    P: Array
    age_values: Array
    comp_map_cnt: Optional[Array]
    comp_map_resp: Array
    N_c: Array
    P_resp: Array
    log_P: Array
    #: Position of each row on the R axis of the grid; used by :meth:`to_grid`.
    _rix: Optional[Array] = None

    @property
    def n_ages(self) -> int:
        return self.P.shape[0]

    @property
    def n_respondent_groups(self) -> int:
        return self.comp_map_resp.shape[1]

    @property
    def n_contact_groups(self) -> int:
        return self.n_ages if self.comp_map_cnt is None else self.comp_map_cnt.shape[1]

    def to_grid(self, values: ArrayLike) -> Array:
        """Scatter per-row ``values`` onto the (C, D[, R]) grid.

        Leading (e.g. sample) dimensions are kept; cells without a row, the
        structural zeros, are 0.

        Parameters
        ----------
        values : array-like, shape (..., n_obs)

        Returns
        -------
        Array, shape (..., *grid_shape)
        """
        values = jnp.asarray(values)
        index = (self.cid, self.did) if self._rix is None else (self.cid, self.did, self._rix)
        grid = jnp.zeros(values.shape[:-1] + self.grid_shape, dtype=values.dtype)
        return grid.at[(..., *index)].set(values)

    def select(self, keep: ArrayLike) -> "AgeMixAggData":
        """Return a copy with only the rows where ``keep`` is ``True``."""
        keep = jnp.asarray(keep, dtype=bool)
        return replace(
            self,
            y=self.y[keep],
            cid=self.cid[keep],
            did=self.did[keep],
            rid=None if self.rid is None else self.rid[keep],
            _rix=None if self._rix is None else self._rix[keep],
            log_row_exposure=self.log_row_exposure[keep],
        )


class AgeMixAggDataLoader:
    """Validate :class:`AgeMixAgg` inputs and load them into :class:`AgeMixAggData`.

    Parameters
    ----------
    y : array-like, shape (C, D) or (C, D, R)
        Integer contact counts by respondent group, recorded contact group
        and (optionally) repeat level.
    N_c : array-like, shape (C,) or (C, R)
        Number of respondents in each respondent group (and repeat level).
        ``None`` when ``N_ar`` is given.
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
    rid : array-like, shape (R,), optional
        Unique non-negative integer repeat levels (previous survey waves) of
        the R axis of ``y``; required for 3-D ``y``.
    N_ar : array-like, shape (A, R), optional
        Exact-age respondent counts per repeat level; replaces ``N_c``. Needs
        respondent groups that partition the fine ages.

    Raises
    ------
    ValueError
        If any input is malformed or inconsistent with the others.

    Examples
    --------
    >>> loader = AgeMixAggDataLoader(y, N_c, P, comp_map_cnt=comp_map_cnt)
    >>> model = AgeMixAgg(loader, priors={"rate": prior})
    """

    #: Keys of a ``prepare_age_mix_agg_inputs`` dict used by :meth:`from_prepared_inputs`.
    PREPARED_INPUT_KEYS = (
        "y",
        "N_c",
        "P",
        "comp_map_cnt",
        "comp_map_resp",
        "age_values",
        "rid",
        "N_ar",
    )

    def __init__(
        self,
        y: ArrayLike,
        N_c: Optional[ArrayLike],
        P: ArrayLike,
        comp_map_cnt: Optional[ArrayLike] = None,
        comp_map_resp: Optional[ArrayLike] = None,
        age_values: Optional[ArrayLike] = None,
        rid: Optional[ArrayLike] = None,
        N_ar: Optional[ArrayLike] = None,
    ) -> None:
        self.y = _as_float(y)
        self.P = _as_float(P)
        self.age_values = (
            jnp.arange(self.P.shape[0], dtype=jnp.float32)
            if age_values is None
            else _as_float(age_values)
        )
        self.comp_map_cnt = _as_float(comp_map_cnt)
        self.rid = _as_float(rid)
        self.N_ar = _as_float(N_ar)

        _check_grid(self.P, self.age_values)
        A = self.P.shape[0]
        if self.comp_map_cnt is not None:
            _check_comp_map(self.comp_map_cnt, "comp_map_cnt", A)
        if comp_map_resp is None:
            self.comp_map_resp = jnp.eye(A, dtype=bool)
        else:
            comp_map_resp = _as_float(comp_map_resp)
            _check_comp_map(comp_map_resp, "comp_map_resp", A)
            self.comp_map_resp = comp_map_resp.astype(bool)

        self.N_c = _resolve_respondent_counts(
            self.y,
            _as_float(N_c),
            self.N_ar,
            self.rid,
            self.comp_map_resp,
        )
        D = A if self.comp_map_cnt is None else self.comp_map_cnt.shape[1]
        _check_counts_and_strata(self.y, self.N_c, (self.comp_map_resp.shape[1], D))
        if self.y.ndim == 3:
            _check_rid(self.rid, self.y.shape[2])
            # Integer levels index the Hill effect evaluated at 0, ..., max.
            self.rid = self.rid.astype(jnp.int32)

    @classmethod
    def from_prepared_inputs(cls, inputs: Dict[str, Any]) -> "AgeMixAggDataLoader":
        """Build a loader from a ``prepare_age_mix_agg_inputs`` dictionary.

        ``N_c`` may be absent when exact-age exposure ``N_ar`` is given.
        """
        return cls(
            **{key: inputs[key] for key in cls.PREPARED_INPUT_KEYS if key in inputs},
            **({} if "N_c" in inputs else {"N_c": None}),
        )

    def load(self) -> AgeMixAggData:
        """Return the observation rows and the grid-level inputs.

        Rows are the cells whose respondent stratum has respondents; cells of
        unexposed strata are structural zeros and are dropped.
        """
        P_resp = self.comp_map_resp.T.astype(jnp.float32) @ self.P
        log_P = jnp.log(self.P)

        exposed = jnp.broadcast_to(jnp.expand_dims(self.N_c > 0, 1), self.y.shape)
        index = tuple(ix.astype(jnp.int32) for ix in jnp.nonzero(exposed))
        cid, did = index[:2]
        rix = index[2] if self.y.ndim == 3 else None  # position on the R axis

        # log w[i, a]: 0 / -inf membership of age a in group cid[i], plus exposure.
        log_member = safe_log(self.comp_map_resp).T[cid]
        if self.N_ar is None:
            N_row = self.N_c[cid] if rix is None else self.N_c[cid, rix]
            log_share = jnp.log(N_row) - jnp.log(P_resp[cid])
            log_row_exposure = log_member + log_P[None, :] + log_share[:, None]
        else:
            log_row_exposure = log_member + safe_log(self.N_ar)[:, rix].T

        # Exposure per fine age (and R-axis position): exact counts, or N_c
        # spread over each group's ages in proportion to P.
        exposure = (
            self.N_ar
            if self.N_ar is not None
            else (self.comp_map_resp * self.P[:, None] / P_resp[None, :]) @ self.N_c
        )
        if rix is None:
            rid, age_exposure = None, exposure
        else:
            # Re-index the R axis by repeat level, so that it pairs with the
            # Hill effect evaluated at levels 0, ..., max(rid).
            rid = self.rid[rix]
            age_exposure = (
                jnp.zeros((self.P.shape[0], int(jnp.max(self.rid)) + 1))
                .at[:, self.rid]
                .set(exposure)
            )

        return AgeMixAggData(
            y=self.y[index],
            cid=cid,
            did=did,
            rid=rid,
            log_row_exposure=log_row_exposure,
            age_exposure=age_exposure,
            grid_shape=tuple(self.y.shape),
            P=self.P,
            age_values=self.age_values,
            comp_map_cnt=self.comp_map_cnt,
            comp_map_resp=self.comp_map_resp,
            N_c=self.N_c,
            P_resp=P_resp,
            log_P=log_P,
            _rix=rix,
        )


# ----------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------


def _check_grid(P: Array, age_values: Array) -> None:
    if P.ndim != 1 or not bool(jnp.all(jnp.isfinite(P))):
        raise ValueError("P must be a finite one-dimensional array.")
    if bool(jnp.any(P <= 0)):
        raise ValueError("P must contain positive values.")
    if age_values.shape != P.shape:
        raise ValueError("age_values and P must have the same length.")


def _check_comp_map(comp_map: Array, name: str, A: int) -> None:
    """Check that ``comp_map`` is a binary (A, n_groups) matrix without empty groups."""
    if comp_map.ndim != 2 or comp_map.shape[0] != A:
        raise ValueError(f"{name} must have shape ({A}, n_groups).")
    if not bool(jnp.all((comp_map == 0) | (comp_map == 1))):
        raise ValueError(f"{name} must be a binary composition matrix.")
    if bool(jnp.any(comp_map.sum(axis=0) == 0)):
        raise ValueError(f"Every group in {name} must contain at least one age.")


def _resolve_respondent_counts(
    y: Array,
    N_c: Optional[Array],
    N_ar: Optional[Array],
    rid: Optional[Array],
    comp_map_resp: Array,
) -> Array:
    """Check the exposure inputs and return ``N_c`` with shape (C,) or (C, R)."""
    A, C = comp_map_resp.shape

    if y.ndim == 2:
        if N_ar is not None or rid is not None:
            raise ValueError("Fatigue inputs require three-dimensional y.")
        if N_c is None or N_c.shape != (C,):
            raise ValueError(f"N_c must have shape ({C},).")
        return N_c

    if y.ndim != 3:
        raise ValueError("y must have shape (C, D) or (C, D, R).")
    R = y.shape[2]
    if N_ar is None:
        if N_c is None or N_c.shape != (C, R):
            raise ValueError(
                f"Three-dimensional y requires N_c {(C, R)} or N_ar {(A, R)}."
            )
        return N_c

    if N_c is not None:
        raise ValueError("Supply either N_c or N_ar, not both.")
    if N_ar.shape != (A, R):
        raise ValueError(f"N_ar must have shape {(A, R)}.")
    if not bool(jnp.all(comp_map_resp.sum(axis=1) == 1)):
        raise ValueError("N_ar requires respondent groups to partition ages.")
    return comp_map_resp.T.astype(jnp.float32) @ N_ar


def _check_counts_and_strata(y: Array, N_c: Array, expected: tuple) -> None:
    if not bool(jnp.all(jnp.isfinite(y))) or bool(jnp.any(y < 0)):
        raise ValueError("y must contain finite, non-negative counts.")
    if not bool(jnp.allclose(y, jnp.round(y))):
        raise ValueError("y must contain integer-valued counts.")
    if not bool(jnp.all(jnp.isfinite(N_c))) or bool(jnp.any(N_c < 0)):
        raise ValueError("Respondent counts must be finite and non-negative.")
    if y.shape[:2] != expected:
        raise ValueError(f"The first two dimensions of y must be {expected}.")
    if bool(jnp.any((N_c == 0) & (y.sum(axis=1) > 0))):
        raise ValueError("Strata with no respondents cannot have observations.")


def _check_rid(rid: Optional[Array], R: int) -> None:
    if rid is None or rid.shape != (R,):
        raise ValueError(f"rid must have shape ({R},).")
    if (
        bool(jnp.any(rid < 0))
        or not bool(jnp.allclose(rid, jnp.round(rid)))
        or jnp.unique(rid).size != R
    ):
        raise ValueError("rid must be unique non-negative integers.")
