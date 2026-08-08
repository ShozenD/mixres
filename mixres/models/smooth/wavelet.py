"""1-D Haar wavelet basis utilities."""

from __future__ import annotations

import numpy as np


def _validate_haar_params(n_levels: int, include_scaling: bool) -> None:
    """Shared validation for `make_haar_basis` and `get_basis_levels`.

    Keeping this in one place guarantees a basis matrix and its level
    metadata can never silently diverge in column count.
    """
    if not isinstance(n_levels, (int, np.integer)) or n_levels < 0:
        raise ValueError(f"n_levels must be a non-negative integer, got {n_levels!r}")
    if n_levels == 0 and not include_scaling:
        raise ValueError(
            "n_levels=0 with include_scaling=False produces an empty basis "
            "(0 columns). Set include_scaling=True or n_levels >= 1."
        )


def make_haar_basis(
    x: np.ndarray,
    n_levels: int = 5,
    domain: tuple[float, float] | None = None,
    include_scaling: bool = True,
) -> np.ndarray:
    """Construct a continuous multiresolution Haar wavelet basis matrix for input grid x.

    Unlike the classical discrete Haar wavelet transform (which requires the
    number of grid points to be a power of 2), this evaluates the standard
    L2([0,1])-orthonormal Haar multiresolution family as piecewise-constant
    functions of x, so it can be sampled at any 1-D grid -- arbitrary
    spacing, arbitrary length -- exactly like `make_bspline_basis` evaluates
    a continuous spline basis at an arbitrary x.

    Parameters
    ----------
    x : np.ndarray, shape (N,)
        Input grid values (1-D).
    n_levels : int, optional
        Number of wavelet resolution levels (default 5). Because a complete
        dyadic multiresolution basis only exists at basis sizes that are
        powers of 2, this parameter is a levels count rather than an
        arbitrary basis size like `make_bspline_basis`'s `n_basis`: level
        `j` contributes `2**j` wavelet functions, so the total basis size is
        `2**n_levels` (including the scaling function) or `2**n_levels - 1`
        (without it).
    domain : tuple[float, float] or None, optional
        (a, b) domain the basis is defined over. If None, defaults to
        (x.min(), x.max()), matching `make_bspline_basis`'s fallback. Points
        of x outside [a, b) contribute 0 to every wavelet column (compactly
        supported, as usual for wavelets) rather than raising -- only the
        scaling column (see below) is unaffected, since it has global, not
        domain-local, support.
    include_scaling : bool, optional
        Whether to include the constant scaling (father wavelet) column,
        phi(x) = 1 for all x. Default True.

        Cubic B-splines already have an exact analogue of this redundancy
        with a model's intercept term: they form a partition of unity
        (`Phi @ ones(M) == 1` on the knot span), and the existing pipeline
        tolerates that via independent weak Normal(0, 1) priors on the
        intercept and the basis coefficients. Defaulting to a complete,
        general-purpose basis here is consistent with that precedent.
        However, the redundancy is stricter for Haar: `phi(x) = 1` is an
        *exact* constant column, not an approximate partition-of-unity sum,
        so set `include_scaling=False` when composing this basis with a
        model that already has an explicit intercept term, to avoid an
        exact linear dependency.

    Returns
    -------
    np.ndarray, shape (N, M)
        Dense basis matrix. M = 2**n_levels if include_scaling, else
        2**n_levels - 1. Columns are ordered coarse-to-fine: the scaling
        column (if present) first, then levels j = 0, ..., n_levels - 1 in
        order, and within each level, positions k = 0, ..., 2**j - 1.

    Notes
    -----
    Each level-j, position-k wavelet has half-open support
    [k/2**j, (k+1)/2**j) in domain-normalized coordinates u = (x-a)/(b-a),
    with value +2**(j/2) on the first half of that support and -2**(j/2) on
    the second half. Because supports are half-open, x == b maps to u == 1.0,
    which falls outside every level's supports under naive floor logic; this
    is corrected by nudging u == 1.0 down to the largest float below 1
    (`np.nextafter`), which lands in the last cell at every level
    simultaneously.
    """
    _validate_haar_params(n_levels, include_scaling)
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be a 1-D array.")

    a, b = domain if domain is not None else (x.min(), x.max())
    if a >= b:
        raise ValueError(
            f"domain must satisfy a < b (got a={a}, b={b}); x appears "
            "constant or domain is degenerate/reversed."
        )

    u = (x - a) / (b - a)
    u = np.where(u >= 1.0, np.nextafter(1.0, 0.0), u)

    N = x.shape[0]
    blocks: list[np.ndarray] = []
    if include_scaling:
        blocks.append(np.ones((N, 1)))

    for j in range(n_levels):
        n_cells = 1 << j
        scaled = u * n_cells
        cell_idx = np.floor(scaled).astype(int)
        frac = scaled - cell_idx
        valid = (cell_idx >= 0) & (cell_idx < n_cells)
        sign = np.where(frac < 0.5, 1.0, -1.0)

        level_block = np.zeros((N, n_cells))
        rows = np.nonzero(valid)[0]
        level_block[rows, cell_idx[rows]] = sign[rows] * (2.0 ** (j / 2))
        blocks.append(level_block)

    return np.hstack(blocks)


def get_basis_levels(n_levels: int, include_scaling: bool = True) -> np.ndarray:
    """Return the resolution level of each column of `make_haar_basis`'s output.

    Parameters
    ----------
    n_levels : int
        Number of wavelet resolution levels, matching the value passed to
        `make_haar_basis`.
    include_scaling : bool, optional
        Whether the scaling column is included, matching the value passed
        to `make_haar_basis`. Default True.

    Returns
    -------
    np.ndarray, shape (M,)
        Level of each column, in the same coarse-to-fine order as
        `make_haar_basis`'s columns: -1 for the scaling column (if
        present), and j for each of the 2**j wavelet columns at level
        j = 0, ..., n_levels - 1. M matches `make_haar_basis`'s column
        count for the same (n_levels, include_scaling). Used to build a
        scale-dependent prior via `make_wavelet_precision`.
    """
    _validate_haar_params(n_levels, include_scaling)
    levels: list[int] = []
    if include_scaling:
        levels.append(-1)
    for j in range(n_levels):
        levels.extend([j] * (1 << j))
    return np.asarray(levels, dtype=int)


def make_wavelet_precision(
    levels: np.ndarray,
    alpha: float = 1.0,
    tau: float = 1.0,
    scaling_precision: float = 1e-6,
    jitter: float = 1e-6,
) -> np.ndarray:
    """Diagonal precision matrix of a Besov-type scale-dependent smoothness prior.

    `make_gmrf_precision` (the B-spline sibling) penalizes adjacent
    coefficients in coefficient space via `D.T @ D`, because spline
    coefficient roughness is a local/neighbour relationship. Haar wavelet
    coefficients are already an orthogonal decomposition by physical scale,
    so there is no neighbour structure to penalize: each coefficient's
    contribution to curve roughness is fully determined by which resolution
    level it belongs to, independent of its siblings at the same level.
    Hence the precision matrix here is exactly diagonal, with the smoothness
    penalty expressed entirely through level-dependent scaling rather than
    through adjacency differencing. `alpha` plays the role `order` plays for
    the GMRF -- larger `alpha` shrinks finer scales harder, giving smoother
    prior draws -- but continuously rather than via a discrete difference
    order.

    Parameters
    ----------
    levels : np.ndarray
        Per-column resolution levels, as returned by `get_basis_levels`
        (-1 for the scaling column, 0, ..., n_levels - 1 for wavelet
        columns).
    alpha : float, optional
        Smoothness/shrinkage exponent: precision at level j scales as
        `4**(alpha * j)` (default 1.0). Larger values shrink finer scales
        more strongly. Negative values invert the intended shrinkage
        direction and are not a typical use case, but are not forbidden.
    tau : float, optional
        Overall precision scale for wavelet columns (default 1.0).
    scaling_precision : float, optional
        Precision assigned to the scaling column, if present (default
        1e-6). Plays the same practical role as `jitter` in
        `make_gmrf_precision`: a small default keeps the scaling
        coefficient's prior effectively flat/uninformative.
    jitter : float, optional
        Diagonal noise added to every entry to guarantee positive
        definiteness (default 1e-6).

    Returns
    -------
    np.ndarray, shape (M, M)
        Diagonal positive definite precision matrix, M = len(levels).
    """
    levels = np.asarray(levels, dtype=int).ravel()
    if levels.size == 0:
        raise ValueError("levels must contain at least one entry.")

    diag = np.empty(levels.shape[0], dtype=float)
    is_wavelet = levels >= 0
    diag[is_wavelet] = tau * (4.0 ** (alpha * levels[is_wavelet])) + jitter
    diag[~is_wavelet] = scaling_precision + jitter
    return np.diag(diag)


def wavelet_covariance_cholesky(Q: np.ndarray) -> np.ndarray:
    """Cholesky factor K such that Sigma = inv(Q) = K @ K.T.

    Duplicated from the identical `gmrf_covariance_cholesky` in
    `bspline.py` rather than imported, to preserve this module's
    self-containment (no local-module imports), matching that module's own
    convention.

    Parameters
    ----------
    Q : np.ndarray, shape (M, M)
        Positive definite precision matrix (e.g. from
        `make_wavelet_precision`).

    Returns
    -------
    np.ndarray, shape (M, M)
        Lower-triangular Cholesky factor K of the covariance Sigma = inv(Q).
    """
    Sigma = np.linalg.inv(Q)
    return np.linalg.cholesky(Sigma)
