from __future__ import annotations

import re
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


def _make_intervals_from_cuts(
    domain: tuple[int, int] = (0, 100),
    cuts: list[int] | tuple[int, ...] | np.ndarray | None = None,
) -> list[tuple[int, int]]:
    """Return interval partition from explicit interior cut locations.

    For domain (a, b) and cuts [c1, c2, ...], returns
    [(a, c1), (c1, c2), ..., (ck, b)].
    """
    if cuts is None:
        raise ValueError("cuts must be provided.")

    cuts_raw = np.asarray(cuts)
    if cuts_raw.ndim != 1:
        raise ValueError("cuts must be a one-dimensional sequence of integers.")

    if cuts_raw.size > 0 and not np.issubdtype(cuts_raw.dtype, np.integer):
        if not np.issubdtype(cuts_raw.dtype, np.floating) or not np.all(
            cuts_raw == np.floor(cuts_raw)
        ):
            raise ValueError("cuts must contain only integer values.")

    cuts_arr = cuts_raw.astype(int)

    # Ensure strictly interior, unique, ascending cut locations.
    if np.any(cuts_arr <= domain[0]) or np.any(cuts_arr >= domain[1]):
        raise ValueError(
            f"All cuts must lie strictly inside domain {domain}, got {cuts_arr.tolist()}."
        )
    if np.any(np.diff(cuts_arr) <= 0):
        raise ValueError("cuts must be strictly increasing with no duplicates.")

    boundaries = np.concatenate(([domain[0]], cuts_arr, [domain[1]])).tolist()
    return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]


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
# Helper for parsing interval strings
# ---------------------------------------------------------------------------

_INTERVAL_RE = re.compile(r"^\[(-?\d+),(-?\d+)([)\]])$")


def _parse_interval_string(s: str) -> tuple[int, int, bool]:
    """Parse an interval string of the form "[a,b)" or "[a,b]".

    Returns
    -------
    (left, right, right_closed) where *right_closed* is True for "[a,b]".

    Raises
    ------
    ValueError
        If the string does not match the expected format, if left >= right,
        or if the values are not integers.
    """
    m = _INTERVAL_RE.match(s.strip())
    if m is None:
        raise ValueError(
            f"Invalid interval string {s!r}. "
            "Expected format '[a,b)' or '[a,b]' with integer endpoints."
        )
    left = int(m.group(1))
    right = int(m.group(2))
    right_closed = m.group(3) == "]"
    if left >= right:
        raise ValueError(
            f"Interval {s!r} is empty or reversed: left ({left}) must be < right ({right})."
        )
    return left, right, right_closed


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
        Width w of each disjoint interval.  Used when *cut_points* is not
        provided, and must evenly divide 100.
    cut_points:
        Optional explicit interior cut locations for interval boundaries.
        For example, ``cut_points=[18, 65]`` yields intervals [0,18),
        [18,65), and [65,100].  Values must be strictly increasing and lie
        strictly inside the domain (0, 100).
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
        cut_points: list[int] | tuple[int, ...] | np.ndarray | None = None,
        n_observations: int = 10,
        amplitude_decay: float = 1.0,
        seed: int | None = None,
    ) -> None:
        if n_observations < 1:
            raise ValueError("n_observations must be at least 1.")

        self.n_components = n_components
        self.interval_width = interval_width
        self.cut_points = cut_points
        self.n_observations = n_observations
        self.amplitude_decay = amplitude_decay
        self.seed = seed

        self._rng = np.random.default_rng(seed)

        # Build fixed structural quantities
        self.grid = _make_grid(self._DOMAIN)
        if cut_points is None:
            self.intervals = _make_intervals(self._DOMAIN, interval_width)
        else:
            self.intervals = _make_intervals_from_cuts(self._DOMAIN, cut_points)

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


# ---------------------------------------------------------------------------
# PoissonEnveloped1D
# ---------------------------------------------------------------------------


class PoissonEnveloped1D:
    """Data generating process for 1-D Poisson counts with overlapping/enveloping
    observed intervals.

    The true underlying process is identical to PoissonDisjoint1D: a random
    Fourier-series latent function is piecewise-constant over disjoint true
    intervals, with Poisson rate lambda_j = exp(mu_j) in each true interval.

    Observed data is aggregated into a separate set of (possibly overlapping
    or enveloped) intervals.  For each observed interval, *n_observations*
    x-positions are drawn i.i.d. uniformly from the integer grid points that
    fall inside that interval.  A Poisson count y ~ Poisson(lambda_j) is then
    drawn at each x, where lambda_j is the true rate for the true interval
    containing x.

    Parameters
    ----------
    obs_intervals:
        List of observed interval strings in "[a,b)" or "[a,b]" format.
        Intervals may overlap or be enveloped by one another.  Values must
        be integers and must lie within the domain [0, 100].
    n_components:
        Number of Fourier basis pairs K for the latent function.
    interval_width:
        Width w of each true disjoint interval.  Used when *cut_points* is
        not provided, and must evenly divide 100.
    cut_points:
        Optional explicit interior cut locations for the true interval
        boundaries.  For example, ``cut_points=[18, 65]`` yields true
        intervals [0,18), [18,65), and [65,100].
    n_observations:
        Number of i.i.d. observations per observed interval.
    amplitude_decay:
        Exponent alpha controlling Fourier coefficient scale:
        std(a_k) = k^{-alpha}.  Larger values give smoother functions.
    seed:
        Seed for the internal RNG.  Fixes both the latent function and the
        noise across calls to generate().

    Attributes
    ----------
    grid : np.ndarray
        Integer grid points over the domain.
    intervals : list[tuple[int, int]]
        True disjoint interval boundaries.
    latent : np.ndarray
        Latent function values f(x) at each grid point.
    interval_means : np.ndarray
        Discrete mean of the latent function over each true interval.
    rates : np.ndarray
        True Poisson rates lambda_j = exp(interval_means[j]) per true interval.
    """

    _DOMAIN: tuple[int, int] = (0, 100)
    _L: float = 100.0

    def __init__(
        self,
        obs_intervals: list[str],
        n_components: int = 5,
        interval_width: int = 5,
        cut_points: list[int] | tuple[int, ...] | np.ndarray | None = None,
        n_observations: int = 10,
        amplitude_decay: float = 1.0,
        seed: int | None = None,
    ) -> None:
        if not obs_intervals:
            raise ValueError("obs_intervals must be a non-empty list of interval strings.")
        if n_observations < 1:
            raise ValueError("n_observations must be at least 1.")

        self.n_components = n_components
        self.interval_width = interval_width
        self.cut_points = cut_points
        self.n_observations = n_observations
        self.amplitude_decay = amplitude_decay
        self.seed = seed

        self._rng = np.random.default_rng(seed)

        # Build true piecewise structure (identical to PoissonDisjoint1D)
        self.grid = _make_grid(self._DOMAIN)
        if cut_points is None:
            self.intervals = _make_intervals(self._DOMAIN, interval_width)
        else:
            self.intervals = _make_intervals_from_cuts(self._DOMAIN, cut_points)

        basis = _fourier_basis(self.grid, n_components, self._L)
        coefs = _sample_coefs(self._rng, n_components, amplitude_decay)
        self.latent = _eval_latent(basis, coefs)
        self.interval_means = _discrete_interval_means(
            self.grid, self.latent, self.intervals
        )
        self.rates = np.exp(self.interval_means)

        # Map every grid position to its true interval index
        self._true_assignments = _interval_assignments(self.grid, self.intervals)

        # Parse and validate observed interval strings
        self._obs_parsed: list[tuple[int, int, bool]] = []
        for s in obs_intervals:
            left, right, right_closed = _parse_interval_string(s)
            domain_start, domain_end = self._DOMAIN
            if left < domain_start or right > domain_end:
                raise ValueError(
                    f"Observed interval {s!r} lies outside the domain {self._DOMAIN}."
                )
            self._obs_parsed.append((left, right, right_closed))

        self._obs_labels: list[str] = list(obs_intervals)

        # Pre-compute the grid positions that belong to each observed interval
        self._obs_grid_points: list[np.ndarray] = []
        for left, right, right_closed in self._obs_parsed:
            if right_closed:
                mask = (self.grid >= left) & (self.grid <= right)
            else:
                mask = (self.grid >= left) & (self.grid < right)
            pts = self.grid[mask].astype(int)
            if len(pts) == 0:
                closing = "]" if right_closed else ")"
                raise ValueError(
                    f"Observed interval '[{left},{right}{closing}' "
                    "contains no integer grid points."
                )
            self._obs_grid_points.append(pts)

        # Pre-compute the mean Poisson rate for each observed interval:
        # lambda_{I_j} = (1 / N_{I_j}) * sum_{a in I_j} lambda_a
        self._obs_interval_rates: np.ndarray = np.array([
            self.rates[self._true_assignments[pts - self._DOMAIN[0]]].mean()
            for pts in self._obs_grid_points
        ])

    def generate(
        self,
        return_true_rates: bool = False,
    ) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
        """Draw n_observations counts per observed interval and return a DataFrame.

        For each observed interval, *n_observations* x-positions are sampled
        uniformly with replacement from the integer grid points inside that
        interval.  A Poisson count y ~ Poisson(lambda_j) is drawn at each x,
        where lambda_j is the true rate for the true disjoint interval that
        contains x.

        Parameters
        ----------
        return_true_rates:
            If True, also return a second DataFrame containing the true
            Poisson rate at every grid point (useful for plotting).

        Returns
        -------
        obs_df : pd.DataFrame with columns:
            obs_interval_id : integer ID of the observed interval (0 = first in list)
            obs_interval    : the original interval string, e.g. "[0,12)"
            y               : Poisson count
            lambda          : interval mean Poisson rate lambda_{I_j}
        rates_df : pd.DataFrame (only returned when *return_true_rates* is True)
            with columns:
            x      : integer grid point
            lambda : true Poisson rate lambda_a at that grid point
        """
        obs_interval_id_col: list[int] = []
        obs_interval_col: list[str] = []
        y_col: list[int] = []
        lambda_col: list[float] = []

        for i, (label, rate) in enumerate(
            zip(self._obs_labels, self._obs_interval_rates)
        ):
            counts = self._rng.poisson(rate, size=self.n_observations)

            obs_interval_id_col.extend([i] * self.n_observations)
            obs_interval_col.extend([label] * self.n_observations)
            y_col.extend(counts.tolist())
            lambda_col.extend([rate] * self.n_observations)

        obs_df = pd.DataFrame(
            {
                "obs_interval_id": obs_interval_id_col,
                "obs_interval": obs_interval_col,
                "y": y_col,
                "lambda": lambda_col,
            }
        )

        if return_true_rates:
            rates_df = pd.DataFrame(
                {
                    "x": self.grid.astype(int),
                    "lambda": self.rates[self._true_assignments],
                }
            )
            return obs_df, rates_df

        return obs_df
