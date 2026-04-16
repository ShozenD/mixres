from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _make_grid(domain: tuple[int, int] = (0, 100)) -> np.ndarray:
    """Return integer grid points over *domain* inclusive."""
    return np.arange(domain[0], domain[1] + 1, dtype=float)


def _make_intervals(
    domain: tuple[int, int] = (0, 100),
    interval_width: int = 5,
) -> list[tuple[int, int]]:
    """Return list of (start, end) tuples that partition *domain*.

    Each interval is half-open [start, end) except the last which is closed
    [start, end].  *interval_width* must evenly divide the domain length.
    """
    length = domain[1] - domain[0]
    if length % interval_width != 0:
        raise ValueError(
            f"interval_width={interval_width} does not evenly divide "
            f"domain length {length}."
        )
    starts = range(domain[0], domain[1], interval_width)
    return [(s, s + interval_width) for s in starts]


def _fourier_basis(grid: np.ndarray, n_components: int, L: float) -> np.ndarray:
    """Build Fourier design matrix of shape (len(grid), 2*n_components).

    Columns are ordered [cos_1, sin_1, cos_2, sin_2, ...].
    No constant term so that E[f(x)] = 0.
    """
    cols = []
    for k in range(1, n_components + 1):
        freq = 2.0 * np.pi * k / L
        cols.append(np.cos(freq * grid))
        cols.append(np.sin(freq * grid))
    return np.column_stack(cols)  # shape (N, 2K)


def _sample_coefs(
    rng: np.random.Generator,
    n_components: int,
    amplitude_decay: float,
) -> np.ndarray:
    """Sample Fourier coefficients (a_k, b_k) ~ N(0, k^{-alpha}).

    Returns array of shape (2*n_components,) laid out as
    [a_1, b_1, a_2, b_2, ...].
    """
    coefs = []
    for k in range(1, n_components + 1):
        std = k ** (-amplitude_decay)
        coefs.append(rng.normal(0.0, std, size=2))
    return np.concatenate(coefs)  # shape (2K,)


def _eval_latent(basis: np.ndarray, coefs: np.ndarray) -> np.ndarray:
    """Evaluate latent function: f = basis @ coefs, shape (N,)."""
    return basis @ coefs


def _discrete_interval_means(
    grid: np.ndarray,
    latent: np.ndarray,
    intervals: list[tuple[int, int]],
) -> np.ndarray:
    """Compute the discrete mean of *latent* over each interval.

    Returns array of shape (M,) where M = len(intervals).
    Grid points are assigned to an interval if start <= x < end (last
    interval is inclusive on both sides).
    """
    means = np.empty(len(intervals))
    for j, (start, end) in enumerate(intervals):
        if j < len(intervals) - 1:
            mask = (grid >= start) & (grid < end)
        else:
            mask = (grid >= start) & (grid <= end)
        means[j] = latent[mask].mean()
    return means


def _interval_assignments(
    grid: np.ndarray,
    intervals: list[tuple[int, int]],
) -> np.ndarray:
    """Return integer array of shape (N,) giving the interval index for each grid point."""
    assignment = np.empty(len(grid), dtype=int)
    for j, (start, end) in enumerate(intervals):
        if j < len(intervals) - 1:
            mask = (grid >= start) & (grid < end)
        else:
            mask = (grid >= start) & (grid <= end)
        assignment[mask] = j
    return assignment


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class GaussianDisjoint1D:
    """Data generating process for 1-D Gaussian observations over disjoint intervals.

    The latent function is a random Fourier series (zero-mean by construction).
    Observations are drawn i.i.d. from N(mu_j, sigma2) at every grid point,
    where mu_j is the discrete mean of the latent function over interval I_j.

    Parameters
    ----------
    n_components:
        Number of Fourier basis pairs K.  Higher values give more complex
        latent functions.
    interval_width:
        Width w of each disjoint interval.  Must evenly divide 100.
    sigma2:
        Observation noise variance.
    amplitude_decay:
        Exponent alpha controlling coefficient scale: std(a_k) = k^{-alpha}.
        Larger values give smoother functions.
    seed:
        Seed for the internal RNG.  Fixes both the latent function and the
        noise across calls to generate().
    """

    _DOMAIN: tuple[int, int] = (0, 100)
    _L: float = 100.0

    def __init__(
        self,
        n_components: int = 5,
        interval_width: int = 5,
        sigma2: float = 1.0,
        amplitude_decay: float = 1.0,
        seed: int | None = None,
    ) -> None:
        if sigma2 <= 0:
            raise ValueError("sigma2 must be positive.")

        self.n_components = n_components
        self.interval_width = interval_width
        self.sigma2 = sigma2
        self.amplitude_decay = amplitude_decay
        self.seed = seed

        self._rng = np.random.default_rng(seed)

        # Build fixed structural quantities
        self.grid = _make_grid(self._DOMAIN)
        self.intervals = _make_intervals(self._DOMAIN, interval_width)
        self._assignments = _interval_assignments(self.grid, self.intervals)

        # Sample latent function (fixed for the lifetime of this instance)
        basis = _fourier_basis(self.grid, n_components, self._L)
        coefs = _sample_coefs(self._rng, n_components, amplitude_decay)
        self.latent = _eval_latent(basis, coefs)
        self.interval_means = _discrete_interval_means(
            self.grid, self.latent, self.intervals
        )

    def generate(self) -> pd.DataFrame:
        """Draw one observation per grid point and return a tidy DataFrame.

        Returns
        -------
        pd.DataFrame with columns:
            x   : grid point (int)
            y   : observed value ~ N(mu_j, sigma2)
            mu  : interval mean mu_j for the interval containing x
            f   : latent function value f(x)
        """
        mu_expanded = self.interval_means[self._assignments]
        y = self._rng.normal(mu_expanded, np.sqrt(self.sigma2))

        return pd.DataFrame(
            {
                "x": self.grid.astype(int),
                "y": y,
                "mu": mu_expanded,
                "f": self.latent,
            }
        )


# ---------------------------------------------------------------------------
# Helper for interval label formatting
# ---------------------------------------------------------------------------


def _format_interval_label(start: int, end: int, is_last: bool) -> str:
    """Return a human-readable interval label.

    All intervals are half-open "[start,end)" except the last, which is
    closed "[start,end]".
    """
    if is_last:
        return f"[{start},{end}]"
    return f"[{start},{end})"


# ---------------------------------------------------------------------------
# PoissonDisjoint1D
# ---------------------------------------------------------------------------


class PoissonDisjoint1D:
    """Data generating process for 1-D Poisson counts over disjoint intervals.

    The latent function is a random Fourier series (zero-mean by construction).
    The Poisson rate for interval I_j is lambda_j = exp(mu_j), where mu_j is
    the discrete mean of the latent function over that interval.  For each
    interval, *n_observations* i.i.d. counts are drawn from Poisson(lambda_j).

    Parameters
    ----------
    n_components:
        Number of Fourier basis pairs K.
    interval_width:
        Width w of each disjoint interval.  Must evenly divide 100.
    n_observations:
        Number of i.i.d. draws per interval n.
    amplitude_decay:
        Exponent alpha controlling coefficient scale: std(a_k) = k^{-alpha}.
        Larger values give smoother latent functions.
    seed:
        Seed for the internal RNG.  Fixes both the latent function and the
        noise across calls to generate().
    """

    _DOMAIN: tuple[int, int] = (0, 100)
    _L: float = 100.0

    def __init__(
        self,
        n_components: int = 5,
        interval_width: int = 5,
        n_observations: int = 10,
        amplitude_decay: float = 1.0,
        seed: int | None = None,
    ) -> None:
        if n_observations < 1:
            raise ValueError("n_observations must be at least 1.")

        self.n_components = n_components
        self.interval_width = interval_width
        self.n_observations = n_observations
        self.amplitude_decay = amplitude_decay
        self.seed = seed

        self._rng = np.random.default_rng(seed)

        # Build fixed structural quantities
        self.grid = _make_grid(self._DOMAIN)
        self.intervals = _make_intervals(self._DOMAIN, interval_width)

        # Sample latent function (fixed for the lifetime of this instance)
        basis = _fourier_basis(self.grid, n_components, self._L)
        coefs = _sample_coefs(self._rng, n_components, amplitude_decay)
        self.latent = _eval_latent(basis, coefs)
        self.interval_means = _discrete_interval_means(
            self.grid, self.latent, self.intervals
        )
        self.rates = np.exp(self.interval_means)

        # Pre-compute interval labels (fixed)
        n_intervals = len(self.intervals)
        self._labels = [
            _format_interval_label(s, e, j == n_intervals - 1)
            for j, (s, e) in enumerate(self.intervals)
        ]

    def generate(self) -> pd.DataFrame:
        """Draw n_observations counts per interval and return a tidy DataFrame.

        Returns
        -------
        pd.DataFrame with columns:
            interval_id : integer ID of the interval (0 = leftmost, ascending)
            interval    : string label for the interval, e.g. "[0,5)", "[95,100]"
            y           : Poisson count ~ Poisson(lambda_j)
            lambda      : true Poisson rate lambda_j for that interval
        """
        interval_id_col: list[int] = []
        interval_col: list[str] = []
        y_col: list[int] = []
        lambda_col: list[float] = []

        for j, (label, rate) in enumerate(zip(self._labels, self.rates)):
            counts = self._rng.poisson(rate, size=self.n_observations)
            interval_id_col.extend([j] * self.n_observations)
            interval_col.extend([label] * self.n_observations)
            y_col.extend(counts.tolist())
            lambda_col.extend([rate] * self.n_observations)

        return pd.DataFrame(
            {
                "interval_id": interval_id_col,
                "interval": interval_col,
                "y": y_col,
                "lambda": lambda_col,
            }
        )
