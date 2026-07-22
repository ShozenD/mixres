"""1-D B-spline and natural cubic spline basis utilities."""

from __future__ import annotations

from typing import Literal

import numpy as np
import jax.numpy as jnp
from jax.typing import ArrayLike
from scipy.interpolate import BSpline
from scipy.linalg import block_diag
from scipy.special import factorial


def _diff_matrix_truncate(num_nodes: int, order: int) -> ArrayLike:
    D = jnp.zeros((num_nodes - order, num_nodes))
    i_vals = jnp.arange(order + 1)
    coeff = (factorial(order) / (factorial(i_vals) * factorial(order - i_vals))) * (
        -1
    ) ** (order - i_vals)
    for i in range(num_nodes - order):
        D = D.at[i, i : i + order + 1].set(coeff)
    return D


def diff_matrix(
    num_nodes: int,
    order: int,
    boundary: Literal["reflect", "truncate"] = "reflect",
) -> ArrayLike:
    """Construct a finite difference matrix of given order.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the grid.
    order : int
        Order of the finite difference.
    boundary : {"reflect", "truncate"}, optional
        "truncate" (the original behaviour) keeps only differences fully
        inside the node range, shape (num_nodes - order, num_nodes). Boundary
        nodes then participate in fewer difference terms than interior nodes
        (e.g. for order=2 the diagonal of D.T @ D is 1 at the edges vs. 6
        in the interior), so a roughness penalty built from this D
        under-regularizes the boundary -- a major contributor to sharp
        boundary artifacts in penalized-spline fits.
        "reflect" (default) mirror-pads the node index range by `order`
        nodes on each side (numpy 'reflect' convention: mirrors excluding
        the edge value, i.e. whole-point/odd reflection -- the discrete
        analogue of a zero-curvature/Neumann boundary condition) before
        differencing, giving boundary nodes the same number of difference
        terms as interior nodes. Shape becomes (num_nodes + order, num_nodes).

    Returns
    -------
    ArrayLike
        Finite difference matrix. Shape (num_nodes - order, num_nodes) for
        "truncate", (num_nodes + order, num_nodes) for "reflect".
    """
    if boundary == "truncate":
        return _diff_matrix_truncate(num_nodes, order)
    if boundary != "reflect":
        raise ValueError(f"boundary must be 'reflect' or 'truncate', got {boundary!r}")

    pad = order
    idx = np.pad(np.arange(num_nodes), pad, mode="reflect")
    R = jnp.asarray(np.eye(num_nodes)[idx])
    D_ext = _diff_matrix_truncate(num_nodes + 2 * pad, order)
    return D_ext @ R


def _clamped_knot_vector(
    x: np.ndarray,
    n_basis: int,
    knots: np.ndarray | None,
    boundary_ext: float,
    degree: int = 3,
) -> tuple[np.ndarray, int]:
    """Build a clamped B-spline knot vector for input grid x.

    Shared by `make_bspline_basis` and `natural_boundary_elimination_matrix`
    so both work from the same knot vector.

    Returns
    -------
    t : np.ndarray
        Clamped knot vector (boundary knots repeated degree+1 times).
    n_basis : int
        Number of basis functions implied by `t` (== len(t) - degree - 1).
    """
    x = np.asarray(x, dtype=float)
    x_min, x_max = x.min(), x.max()

    if knots is None:
        n_interior = n_basis - degree - 1
        if n_interior < 0:
            raise ValueError(
                f"n_basis={n_basis} is too small for degree={degree}; "
                f"need n_basis >= {degree + 1}."
            )
        # Interior knots always span the true data domain, so every basis
        # function retains support on the evaluation grid regardless of
        # boundary_ext (see below).
        interior_knots = np.linspace(x_min, x_max, n_interior + 2)[1:-1]
    else:
        interior_knots = np.asarray(knots, dtype=float)
        n_basis = len(interior_knots) + degree + 1

    # Only the repeated (clamped) boundary knots move with boundary_ext, scaled
    # to the interior knot spacing rather than the domain width. This keeps
    # the "push the clamp point away from x_min/x_max" effect that reduces
    # boundary pinning, without diluting knot density inside the domain and
    # pushing basis functions entirely off the evaluation grid.
    knot_spacing = (
        interior_knots[1] - interior_knots[0]
        if len(interior_knots) > 1
        else (x_max - x_min)
    )
    x_min_knot = x_min - boundary_ext * knot_spacing
    x_max_knot = x_max + boundary_ext * knot_spacing

    # Clamped knot vector: boundary knots repeated (degree + 1) times
    t = np.concatenate(
        [
            np.repeat(x_min_knot, degree + 1),
            interior_knots,
            np.repeat(x_max_knot, degree + 1),
        ]
    )
    return t, n_basis


def natural_boundary_elimination_matrix(
    t: np.ndarray, degree: int, n_basis: int
) -> np.ndarray:
    """Elimination matrix imposing natural (f''=0) boundary conditions.

    Returns E, shape (n_basis, n_basis - 2), such that for any reduced
    coefficient vector v (length n_basis - 2), w = E @ v satisfies
    f''(x_min) = f''(x_max) = 0 exactly, where f = sum_k w[k] * B_k(x) is
    the spline built from clamped knot vector `t`.

    Because cubic B-splines have compact support, the second derivative at
    the boundary knot depends on only the first (resp. last) 3
    coefficients, so both constraints are solved locally: w[0] and w[-1]
    are eliminated as linear combinations of their immediate neighbours,
    v = w[1:-1] retains the original spatial ordering of the interior
    coefficients unchanged. This keeps the local/banded structure that
    `diff_matrix`-based roughness penalties rely on -- unlike a generic
    null-space (SVD) reduction of the constraint, which would mix all
    coefficients together.

    Parameters
    ----------
    t : np.ndarray
        Clamped knot vector, as returned by `_clamped_knot_vector`.
    degree : int
        Spline degree (3 for cubic).
    n_basis : int
        Number of basis functions (== len(t) - degree - 1).

    Returns
    -------
    np.ndarray, shape (n_basis, n_basis - 2)
    """
    if n_basis < degree + 3:
        raise ValueError(
            f"n_basis={n_basis} is too small for natural boundary elimination; "
            f"need n_basis >= {degree + 3} (degree + 3) so the reduced basis "
            "retains at least one free coefficient."
        )
    d2 = BSpline(t, np.eye(n_basis), degree).derivative(2)
    x_min_knot, x_max_knot = t[degree], t[-(degree + 1)]
    c0, c1, c2 = d2(x_min_knot)[:3]
    cm3, cm2, cm1 = d2(x_max_knot)[-3:]

    E = np.zeros((n_basis, n_basis - 2))
    E[1:-1, :] = np.eye(n_basis - 2)
    E[0, 0], E[0, 1] = -c1 / c0, -c2 / c0
    E[-1, -2], E[-1, -1] = -cm3 / cm1, -cm2 / cm1
    return E


def make_bspline_basis(
    x: np.ndarray,
    n_basis: int = 30,
    knots: np.ndarray | None = None,
    boundary_ext: float = 0.0,
    natural_boundary: bool = False,
    return_elimination_matrix: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Construct a cubic B-spline basis matrix for input grid x.

    Parameters
    ----------
    x : np.ndarray, shape (N,)
        Input grid values (1-D).
    n_basis : int, optional
        Number of basis functions (default 30). Ignored when `knots` is given.
    knots : np.ndarray or None, optional
        Interior knot locations. If None, knots are placed uniformly in
        (x.min(), x.max()) such that the basis has `n_basis` functions.
        When supplied, `n_basis` is inferred as len(knots) + degree + 1.
    boundary_ext : float, optional
        Pushes the repeated (clamped) boundary knots out beyond the data
        domain, in units of the interior knot spacing, to reduce boundary
        pinning artifacts. The clamped boundary knots are placed at
        x_min - boundary_ext * knot_spacing and x_max + boundary_ext *
        knot_spacing. Interior knots always span [x.min(), x.max()]
        regardless of boundary_ext, so every basis function retains support
        on the evaluation grid. Default 0.0 (no extension).
    natural_boundary : bool, optional
        If True, reparametrize the basis so that f''(x_min) = f''(x_max) = 0
        for every coefficient vector (see `natural_boundary_elimination_matrix`),
        removing the 2 boundary degrees of freedom responsible for the
        concentrated/asymmetric leverage of the first and last clamped
        B-spline basis functions. Reduces the basis to n_basis - 2 columns.
        Default False (identical to prior behaviour).
    return_elimination_matrix : bool, optional
        If True (requires natural_boundary=True), also return the
        elimination matrix E used to build the reduced basis, so that
        penalty matrices (e.g. from `diff_matrix`) can be composed with it.

    Returns
    -------
    np.ndarray, shape (N, M)
        Dense basis matrix. M = n_basis (or len(knots) + degree + 1) when
        natural_boundary=False, M - 2 when natural_boundary=True.
    np.ndarray, shape (M, M - 2), optional
        Elimination matrix E, only when return_elimination_matrix=True.
    """
    degree = 3
    x = np.asarray(x, dtype=float)
    t, n_basis = _clamped_knot_vector(x, n_basis, knots, boundary_ext, degree)

    B = BSpline.design_matrix(x, t, degree).toarray()

    E = None
    if natural_boundary:
        E = natural_boundary_elimination_matrix(t, degree, n_basis)
        B = B @ E

    if return_elimination_matrix:
        if E is None:
            raise ValueError(
                "return_elimination_matrix=True requires natural_boundary=True."
            )
        return B, E
    return B


def make_piecewise_bspline_basis(
    x: np.ndarray,
    cut_points: list[int],
    basis_density: float = 0.5,
    boundary_ext: float = 0.0,
    natural_boundary: bool = False,
    return_basis_counts: bool = False,
) -> (
    np.ndarray
    | tuple[np.ndarray, list[int]]
    | tuple[np.ndarray, list[int], list[np.ndarray]]
):
    """Construct a piecewise cubic B-spline basis over cutpoint-defined segments.

    For integer support x = [x_min, ..., x_max] and cut points [c1, c2, ...],
    this creates segments [x_min, c1-1], [c1, c2-1], ..., [ck, x_max].
    The number of basis functions in each segment is
    round(basis_density * segment_length), with a minimum of 4 for cubic splines.

    Parameters
    ----------
    x : np.ndarray, shape (N,)
        Full integer support grid.
    cut_points : list[int]
        Breakpoints that define the segment boundaries. Each value cp must
        satisfy x_min < cp <= x_max.
    basis_density : float, optional
        Fraction of segment length used as the number of basis functions
        (default 0.5). The value is rounded and clipped to a minimum of 4.
    boundary_ext : float, optional
        Boundary extension factor passed to `make_bspline_basis` for each
        segment. Extends the knot vector beyond each segment's domain to
        reduce boundary artifacts. Default 0.0 (no extension).
    natural_boundary : bool, optional
        If True, apply the natural (f''=0) boundary elimination from
        `make_bspline_basis` independently within each segment. Since
        segments are already independent clamped B-splines with no
        continuity constraint across cut points, this removes the
        concentrated-leverage boundary artifact at every internal cut
        point as well as at the two outer domain edges. Reduces each
        segment's basis by 2 columns. Default False.
    return_basis_counts : bool, optional
        If True, also return a list with the number of basis functions per
        segment (post-elimination, i.e. matching B's actual column layout,
        when natural_boundary=True).

    Returns
    -------
    B : np.ndarray, shape (N, M)
        Block-diagonal basis matrix spanning all segments.
    segment_basis_counts : list[int]
        Only returned when return_basis_counts=True.
    elimination_blocks : list[np.ndarray]
        Only returned when natural_boundary=True and return_basis_counts=True.
        Per-segment elimination matrices E (see
        `natural_boundary_elimination_matrix`), needed to compose a
        difference-penalty matrix with the reduced per-segment coefficients
        (see `make_block_difference_matrix`'s `elimination_matrices` param).
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be a 1-D array.")
    if len(x) == 0:
        raise ValueError("x must contain at least one point.")

    x_min = int(np.min(x))
    x_max = int(np.max(x))

    sorted_cut_points = sorted(int(cp) for cp in cut_points)
    for cp in sorted_cut_points:
        if cp <= x_min or cp > x_max:
            raise ValueError(
                f"Cut point {cp} is outside valid range ({x_min}, {x_max}]."
            )

    boundaries = [x_min] + sorted_cut_points + [x_max + 1]
    basis_blocks: list[np.ndarray] = []
    segment_basis_counts: list[int] = []
    elimination_blocks: list[np.ndarray] = []

    for left, right_exclusive in zip(boundaries[:-1], boundaries[1:]):
        mask = (x >= left) & (x < right_exclusive)
        x_segment = x[mask]

        if x_segment.size == 0:
            continue

        segment_length = right_exclusive - left
        n_basis_segment = max(4, int(round(basis_density * segment_length)))
        if natural_boundary:
            B_segment, E_segment = make_bspline_basis(
                x_segment,
                n_basis=n_basis_segment,
                boundary_ext=boundary_ext,
                natural_boundary=True,
                return_elimination_matrix=True,
            )
            elimination_blocks.append(E_segment)
        else:
            B_segment = make_bspline_basis(
                x_segment, n_basis=n_basis_segment, boundary_ext=boundary_ext
            )

        block = np.zeros((x.shape[0], B_segment.shape[1]), dtype=float)
        block[mask, :] = B_segment
        basis_blocks.append(block)
        segment_basis_counts.append(B_segment.shape[1])

    B = np.concatenate(basis_blocks, axis=1)
    if natural_boundary and return_basis_counts:
        return B, segment_basis_counts, elimination_blocks
    if return_basis_counts:
        return B, segment_basis_counts
    return B


def make_natural_cubic_spline_basis(
    x: np.ndarray,
    n_basis: int = 10,
    knots: np.ndarray | None = None,
) -> np.ndarray:
    """Construct a natural cubic spline (NCS) basis matrix for input grid x.

    Uses the truncated power basis parameterisation from Hastie, Tibshirani &
    Friedman (ESL, Chapter 5). With K knots ξ_1 < ... < ξ_K the K basis
    functions are::

        N_1(x) = 1
        N_2(x) = x
        N_{k+2}(x) = d_k(x) - d_{K-1}(x),   k = 1, …, K-2

    where d_k(x) = ((x − ξ_k)_+³ − (x − ξ_K)_+³) / (ξ_K − ξ_k).

    Because all higher-order terms cancel outside [ξ_1, ξ_K] the spline is
    linear (natural) in the tails.

    Parameters
    ----------
    x : np.ndarray, shape (N,)
        Input grid values (1-D).
    n_basis : int, optional
        Number of basis functions, equal to the number of knots (default 10).
        Ignored when `knots` is given.
    knots : np.ndarray or None, optional
        All knot locations (including boundary knots). If None, `n_basis`
        knots are placed uniformly from x.min() to x.max(). When supplied,
        n_basis is inferred as len(knots).

    Returns
    -------
    np.ndarray, shape (N, K)
        Dense basis matrix, where K = n_basis (or len(knots)).
    """
    x = np.asarray(x, dtype=float)
    x_min, x_max = x.min(), x.max()

    if knots is None:
        if n_basis < 2:
            raise ValueError(
                f"n_basis={n_basis} must be >= 2 for natural cubic splines."
            )
        xi = np.linspace(x_min, x_max, n_basis)
    else:
        xi = np.asarray(knots, dtype=float)
        n_basis = len(xi)
        if n_basis < 2:
            raise ValueError("Need at least 2 knots for natural cubic splines.")

    K = n_basis

    def _d(xv: np.ndarray, xi_k: float, xi_K: float) -> np.ndarray:
        return (np.maximum(xv - xi_k, 0.0) ** 3 - np.maximum(xv - xi_K, 0.0) ** 3) / (
            xi_K - xi_k
        )

    N_mat = np.zeros((len(x), K), dtype=float)
    N_mat[:, 0] = 1.0
    if K >= 2:
        N_mat[:, 1] = x
    for k in range(K - 2):
        N_mat[:, k + 2] = _d(x, xi[k], xi[K - 1]) - _d(x, xi[K - 2], xi[K - 1])

    return N_mat


def make_piecewise_natural_cubic_spline_basis(
    x: np.ndarray,
    cut_points: list[int],
    basis_density: float = 0.5,
    return_basis_counts: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[int]]:
    """Construct a piecewise natural cubic spline basis over cutpoint-defined segments.

    For integer support x = [x_min, ..., x_max] and cut points [c1, c2, ...],
    this creates segments [x_min, c1-1], [c1, c2-1], ..., [ck, x_max].
    The number of basis functions in each segment is
    round(basis_density * segment_length), with a minimum of 4.

    Parameters
    ----------
    x : np.ndarray, shape (N,)
        Full integer support grid.
    cut_points : list[int]
        Breakpoints that define the segment boundaries. Each value cp must
        satisfy x_min < cp <= x_max.
    basis_density : float, optional
        Fraction of segment length used as the number of basis functions
        (default 0.5). Rounded and clipped to a minimum of 4.
    return_basis_counts : bool, optional
        If True, also return a list with the number of basis functions per
        segment.

    Returns
    -------
    B : np.ndarray, shape (N, M)
        Block-diagonal basis matrix spanning all segments.
    segment_basis_counts : list[int]
        Only returned when return_basis_counts=True.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be a 1-D array.")
    if len(x) == 0:
        raise ValueError("x must contain at least one point.")

    x_min = int(np.min(x))
    x_max = int(np.max(x))

    sorted_cut_points = sorted(int(cp) for cp in cut_points)
    for cp in sorted_cut_points:
        if cp <= x_min or cp > x_max:
            raise ValueError(
                f"Cut point {cp} is outside valid range ({x_min}, {x_max}]."
            )

    boundaries = [x_min] + sorted_cut_points + [x_max + 1]
    basis_blocks: list[np.ndarray] = []
    segment_basis_counts: list[int] = []

    for left, right_exclusive in zip(boundaries[:-1], boundaries[1:]):
        mask = (x >= left) & (x < right_exclusive)
        x_segment = x[mask]

        if x_segment.size == 0:
            continue

        segment_length = right_exclusive - left
        n_basis_segment = max(4, int(round(basis_density * segment_length)))
        B_segment = make_natural_cubic_spline_basis(x_segment, n_basis=n_basis_segment)

        block = np.zeros((x.shape[0], B_segment.shape[1]), dtype=float)
        block[mask, :] = B_segment
        basis_blocks.append(block)
        segment_basis_counts.append(B_segment.shape[1])

    B = np.concatenate(basis_blocks, axis=1)
    if return_basis_counts:
        return B, segment_basis_counts
    return B


def make_block_difference_matrix(
    segment_basis_counts: list[int],
    order: int = 2,
    boundary: Literal["reflect", "truncate"] = "reflect",
    elimination_matrices: list[np.ndarray] | None = None,
    dtype: type = np.float32,
) -> np.ndarray:
    """Build block-diagonal finite-difference matrix, one block per segment.

    Parameters
    ----------
    segment_basis_counts : list[int]
        Number of basis functions per segment. When `elimination_matrices`
        is given, these are the POST-elimination counts (i.e. matching the
        basis's actual column layout, each 2 less than the corresponding
        pre-elimination segment size).
    order : int, optional
        Order of the finite difference (default 2).
    boundary : {"reflect", "truncate"}, optional
        Passed through to `diff_matrix` for each segment (default "reflect").
        See `diff_matrix` for why "reflect" is recommended: it equalizes the
        roughness-penalty strength between boundary and interior
        coefficients, which "truncate" does not.
    elimination_matrices : list[np.ndarray] or None, optional
        Per-segment natural-boundary elimination matrices E (see
        `natural_boundary_elimination_matrix`, `make_piecewise_bspline_basis`
        with `natural_boundary=True`). When given, each block is built as
        `diff_matrix(n_basis + 2, order, boundary) @ E_segment` -- i.e. the
        difference penalty is computed on the pre-elimination coefficients
        and then folded through the same elimination that reduced the basis,
        so the penalty stays consistent with the natural-boundary basis.
    dtype : type, optional
        NumPy dtype for the output matrix (default np.float32).

    Returns
    -------
    np.ndarray
        Block-diagonal difference matrix.
    """
    if not segment_basis_counts:
        raise ValueError("segment_basis_counts must contain at least one segment.")
    if elimination_matrices is not None and len(elimination_matrices) != len(
        segment_basis_counts
    ):
        raise ValueError(
            "elimination_matrices must have one entry per segment when given."
        )

    blocks = []
    for i, n_basis in enumerate(segment_basis_counts):
        if elimination_matrices is not None:
            E_segment = elimination_matrices[i]
            n_basis_full = n_basis + 2
            D_full = np.asarray(diff_matrix(n_basis_full, order, boundary=boundary))
            block = np.asarray(D_full @ E_segment, dtype=dtype)
        else:
            if n_basis <= order:
                raise ValueError(
                    f"Each segment needs n_basis > order. Got n_basis={n_basis}, order={order}."
                )
            block = np.asarray(
                diff_matrix(int(n_basis), order, boundary=boundary), dtype=dtype
            )
        blocks.append(block)

    return block_diag(*blocks).astype(dtype)


def get_segment_diff_sizes(
    segment_basis_counts: list[int],
    order: int = 2,
    boundary: Literal["reflect", "truncate"] = "reflect",
) -> list[int]:
    """Return number of finite-difference terms per segment.

    Parameters
    ----------
    segment_basis_counts : list[int]
        Number of basis functions per segment.
    order : int, optional
        Finite difference order (default 2).
    boundary : {"reflect", "truncate"}, optional
        Must match the `boundary` argument used to build the corresponding
        difference matrix (`diff_matrix`/`make_block_difference_matrix`),
        since "reflect" produces `n_basis + order` rows per segment instead
        of "truncate"'s `n_basis - order`. Default "reflect".

    Returns
    -------
    list[int]
        Number of difference terms per segment.
    """
    if boundary == "reflect":
        return [int(n_basis + order) for n_basis in segment_basis_counts]
    return [int(n_basis - order) for n_basis in segment_basis_counts]


def make_gmrf_precision(
    n_basis: int,
    order: int = 1,
    tau: float = 1.0,
    jitter: float = 1e-6,
    boundary: Literal["reflect", "truncate"] = "reflect",
) -> np.ndarray:
    """Precision matrix of an order-`order` GMRF smoothness prior.

    Q = tau * D.T @ D + jitter * I, where D is the order-th finite-difference
    matrix over `n_basis` coefficients (see `diff_matrix`). D.T @ D alone is
    only positive semi-definite (rank n_basis - order for "truncate", full
    rank n_basis for "reflect"; its null space for "truncate" is polynomials
    of degree < order for RW1/RW2), so `jitter` is added to the diagonal to
    make Q positive definite and hence invertible.

    Parameters
    ----------
    n_basis : int
        Number of coefficients the GMRF prior is defined over.
    order : int, optional
        GMRF order: 1 for a first-order random walk (penalizes adjacent
        differences), 2 for second-order (penalizes curvature). Default 1.
    tau : float, optional
        Overall precision scale (default 1.0).
    jitter : float, optional
        Diagonal noise added to guarantee positive definiteness (default 1e-6).
    boundary : {"reflect", "truncate"}, optional
        Passed through to `diff_matrix` (default "reflect"). "reflect"
        equalizes the prior precision of boundary and interior coefficients
        (mirror-padding the difference operator at the edges); "truncate"
        is the original behaviour, under which boundary coefficients have
        substantially higher prior variance than interior ones (e.g. ~6x
        for order=2), a major contributor to sharp boundary artifacts.

    Returns
    -------
    np.ndarray, shape (n_basis, n_basis)
        Positive definite GMRF precision matrix.
    """
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, got {order}")
    D = np.asarray(diff_matrix(n_basis, order, boundary=boundary))
    return tau * (D.T @ D) + jitter * np.eye(n_basis)


def gmrf_covariance_cholesky(Q: np.ndarray) -> np.ndarray:
    """Cholesky factor K such that Sigma = inv(Q) = K @ K.T.

    Parameters
    ----------
    Q : np.ndarray, shape (M, M)
        Positive definite precision matrix (e.g. from `make_gmrf_precision`).

    Returns
    -------
    np.ndarray, shape (M, M)
        Lower-triangular Cholesky factor K of the covariance Sigma = inv(Q).
    """
    Sigma = np.linalg.inv(Q)
    return np.linalg.cholesky(Sigma)
