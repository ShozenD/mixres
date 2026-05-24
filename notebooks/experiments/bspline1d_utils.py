"""1-D B-spline and natural cubic spline basis utilities."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from jax.typing import ArrayLike
from scipy.interpolate import BSpline
from scipy.linalg import block_diag
from scipy.special import factorial


def diff_matrix(num_nodes: int, order: int) -> ArrayLike:
    """Construct a finite difference matrix of given order.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the grid.
    order : int
        Order of the finite difference.

    Returns
    -------
    ArrayLike
        Finite difference matrix of shape (num_nodes - order, num_nodes).
    """
    D = jnp.zeros((num_nodes - order, num_nodes))
    i_vals = jnp.arange(order + 1)
    coeff = (factorial(order) / (factorial(i_vals) * factorial(order - i_vals))) * (
        -1
    ) ** (order - i_vals)
    for i in range(num_nodes - order):
        D = D.at[i, i : i + order + 1].set(coeff)
    return D


def make_bspline_basis(
    x: np.ndarray,
    n_basis: int = 30,
    knots: np.ndarray | None = None,
    boundary_ext: float = 0.0,
) -> np.ndarray:
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
        Extends the knot vector beyond the data domain to reduce boundary
        artifacts. The clamped boundary knots are placed at
        x_min - boundary_ext * half_length and x_max + boundary_ext *
        half_length, where half_length = (x_max - x_min) / 2. Default 0.0
        (no extension).

    Returns
    -------
    np.ndarray, shape (N, M)
        Dense basis matrix, where M = n_basis (or len(knots) + degree + 1).
    """
    degree = 3
    x = np.asarray(x, dtype=float)
    x_min, x_max = x.min(), x.max()

    half_length = (x_max - x_min) / 2
    x_min_knot = x_min - boundary_ext * half_length
    x_max_knot = x_max + boundary_ext * half_length

    if knots is None:
        n_interior = n_basis - degree - 1
        if n_interior < 0:
            raise ValueError(
                f"n_basis={n_basis} is too small for degree={degree}; "
                f"need n_basis >= {degree + 1}."
            )
        interior_knots = np.linspace(x_min_knot, x_max_knot, n_interior + 2)[1:-1]
    else:
        interior_knots = np.asarray(knots, dtype=float)
        n_basis = len(interior_knots) + degree + 1

    # Clamped knot vector: boundary knots repeated (degree + 1) times
    t = np.concatenate(
        [
            np.repeat(x_min_knot, degree + 1),
            interior_knots,
            np.repeat(x_max_knot, degree + 1),
        ]
    )

    return BSpline.design_matrix(x, t, degree).toarray()


def make_piecewise_bspline_basis(
    x: np.ndarray,
    cut_points: list[int],
    basis_density: float = 0.5,
    boundary_ext: float = 0.0,
    return_basis_counts: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[int]]:
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
        B_segment = make_bspline_basis(x_segment, n_basis=n_basis_segment, boundary_ext=boundary_ext)

        block = np.zeros((x.shape[0], B_segment.shape[1]), dtype=float)
        block[mask, :] = B_segment
        basis_blocks.append(block)
        segment_basis_counts.append(B_segment.shape[1])

    B = np.concatenate(basis_blocks, axis=1)
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
    dtype: type = np.float32,
) -> np.ndarray:
    """Build block-diagonal finite-difference matrix, one block per segment.

    Parameters
    ----------
    segment_basis_counts : list[int]
        Number of basis functions per segment.
    order : int, optional
        Order of the finite difference (default 2).
    dtype : type, optional
        NumPy dtype for the output matrix (default np.float32).

    Returns
    -------
    np.ndarray
        Block-diagonal difference matrix.
    """
    if not segment_basis_counts:
        raise ValueError("segment_basis_counts must contain at least one segment.")

    blocks = []
    for n_basis in segment_basis_counts:
        if n_basis <= order:
            raise ValueError(
                f"Each segment needs n_basis > order. Got n_basis={n_basis}, order={order}."
            )
        block = np.asarray(diff_matrix(int(n_basis), order), dtype=dtype)
        blocks.append(block)

    return block_diag(*blocks).astype(dtype)


def get_segment_diff_sizes(
    segment_basis_counts: list[int],
    order: int = 2,
) -> list[int]:
    """Return number of finite-difference terms per segment.

    Parameters
    ----------
    segment_basis_counts : list[int]
        Number of basis functions per segment.
    order : int, optional
        Finite difference order (default 2).

    Returns
    -------
    list[int]
        Number of difference terms per segment.
    """
    return [int(n_basis - order) for n_basis in segment_basis_counts]
