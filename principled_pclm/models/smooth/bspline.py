"""1-D B-spline and natural cubic spline basis utilities."""

from __future__ import annotations

from typing import Literal

import numpy as np
import jax.numpy as jnp
from jax.typing import ArrayLike
from scipy.interpolate import BSpline
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
        interior_knots = np.linspace(x_min, x_max, n_interior + 2)[1:-1]
    else:
        interior_knots = np.asarray(knots, dtype=float)
        n_basis = len(interior_knots) + degree + 1

    # Clamped knot vector: boundary knots repeated (degree + 1) times
    t = np.concatenate(
        [
            np.repeat(x_min, degree + 1),
            interior_knots,
            np.repeat(x_max, degree + 1),
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
    t, n_basis = _clamped_knot_vector(x, n_basis, knots, degree)

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
        difference matrix (`diff_matrix`), since "reflect" produces
        `n_basis + order` rows per segment instead of "truncate"'s
        `n_basis - order`. Default "reflect".

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
