from __future__ import annotations

import re
from typing import Literal

import numpy as np
import pandas as pd

from .gp_utils import make_kernel, sample_gp

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


def _sample_bandlimited_latent(
    rng: np.random.Generator,
    N: int,
    omega_max: float,
    process_std: float,
    cutoff_ratio: float,
    damping_power: float,
) -> np.ndarray:
    """Sample a band-limited stochastic process g(x) for x = 0, ..., N-1."""
    x = np.arange(N, dtype=float)

    if not 2 * np.pi / N <= omega_max < np.pi:
        raise ValueError("omega_max must be between 2*pi/N and pi.")

    M = int(np.floor(N * omega_max / (2 * np.pi)))
    M = min(M, (N - 1) // 2)
    if M < 1:
        raise ValueError(
            "omega_max is too close to 2*pi/N; no frequency components available."
        )

    omega = 2 * np.pi * np.arange(1, M + 1) / N
    actual_omega_max = omega[-1]

    weights = cutoff_ratio ** ((omega / actual_omega_max) ** damping_power)
    tau = process_std / np.sqrt(np.sum(weights**2))

    a = rng.normal(size=M) * tau * weights
    b = rng.normal(size=M) * tau * weights

    return np.sin(np.outer(x, omega)) @ a + np.cos(np.outer(x, omega)) @ b


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


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class SimGaussian1D:
    """Data generating process for 1-D Gaussian observations over (possibly
    overlapping/enveloping) observed intervals.

    The latent function g(x) is a band-limited stochastic process built from
    a random sin/cos frequency expansion up to omega_max, with per-frequency
    coefficient std decaying by (cutoff_ratio, damping_power). For each
    observed interval, the mean of g over that interval's grid points is the
    Gaussian mean mu_j; n_obs i.i.d. draws y ~ N(mu_j, noise_std^2) are taken
    per observed interval.

    Parameters
    ----------
    obs_intervals:
        List of observed interval strings in "[a,b)" or "[a,b]" format.
        Intervals may overlap or be enveloped by one another. Values must be
        integers and must lie within the domain [0, N-1].
    n_obs:
        Number of i.i.d. observations per observed interval.
    N:
        Number of grid points. The grid is 0, 1, ..., N-1.
    omega_max:
        Maximum angular frequency of the latent process. Must satisfy
        2*pi/N <= omega_max < pi.
    process_std:
        Marginal standard deviation of the latent stochastic process.
    cutoff_ratio:
        Ratio controlling the coefficient standard deviation at the highest
        frequency relative to the lowest.
    damping_power:
        Exponent controlling how strongly the coefficient standard deviation
        decays with frequency.
    noise_std:
        Observation noise standard deviation.
    seed:
        Seed for the internal RNG. Fixes both the latent process and the
        noise across calls to generate().
    """

    def __init__(
        self,
        obs_intervals: list[str],
        n_obs: int = 10,
        N: int = 100,
        omega_max: float = np.pi / 2,
        process_std: float = 1.0,
        cutoff_ratio: float = 0.05,
        damping_power: float = 2.0,
        noise_std: float = 0.1,
        seed: int | None = None,
    ) -> None:
        if not obs_intervals:
            raise ValueError(
                "obs_intervals must be a non-empty list of interval strings."
            )
        if n_obs < 1:
            raise ValueError("n_obs must be at least 1.")
        if noise_std <= 0:
            raise ValueError("noise_std must be positive.")

        self.n_obs = n_obs
        self.N = N
        self.omega_max = omega_max
        self.process_std = process_std
        self.cutoff_ratio = cutoff_ratio
        self.damping_power = damping_power
        self.noise_std = noise_std
        self.seed = seed

        self._rng = np.random.default_rng(seed)

        self.domain: tuple[int, int] = (0, N - 1)
        self.grid = np.arange(N, dtype=float)

        self.latent = _sample_bandlimited_latent(
            self._rng, N, omega_max, process_std, cutoff_ratio, damping_power
        )

        # Parse and validate observed interval strings
        self._obs_parsed: list[tuple[int, int, bool]] = []
        for s in obs_intervals:
            left, right, right_closed = _parse_interval_string(s)
            domain_start, domain_end = self.domain
            if left < domain_start or right > domain_end:
                raise ValueError(
                    f"Observed interval {s!r} lies outside the domain {self.domain}."
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

        # Pre-compute the mean latent value for each observed interval
        self.obs_interval_means: np.ndarray = np.array(
            [self.latent[pts].mean() for pts in self._obs_grid_points]
        )

    def generate(self) -> pd.DataFrame:
        """Draw n_obs observations per observed interval and return a DataFrame.

        Returns
        -------
        pd.DataFrame with columns:
            idx : integer index of the observed interval (0 = first in obs_intervals)
            y   : observed value ~ N(obs_interval_means[idx], noise_std^2)
        """
        idx_col: list[int] = []
        y_col: list[float] = []

        for j, mu in enumerate(self.obs_interval_means):
            y = self._rng.normal(mu, self.noise_std, size=self.n_obs)
            idx_col.extend([j] * self.n_obs)
            y_col.extend(y.tolist())

        return pd.DataFrame({"idx": idx_col, "y": y_col})

    def get_reference_df(self) -> pd.DataFrame:
        """Return the true latent function and interval means at each grid point.

        For each observed interval, every grid point contained in it appears
        as one row. Points covered by multiple (overlapping) observed
        intervals appear once per interval.

        Returns
        -------
        pd.DataFrame with columns:
            idx           : integer index of the observed interval (matches generate()'s idx)
            obs_interval  : the original interval string, e.g. "[0,12)"
            interval_min  : left endpoint of the observed interval
            interval_max  : right endpoint of the observed interval
            x             : integer grid point
            f             : latent function value g(x)
            mu            : mean of g over the observed interval (obs_interval_means[idx])
        """
        idx_col: list[int] = []
        obs_interval_col: list[str] = []
        interval_min_col: list[int] = []
        interval_max_col: list[int] = []
        x_col: list[int] = []
        f_col: list[float] = []
        mu_col: list[float] = []

        for j, (label, (left, right, _), pts, mu) in enumerate(
            zip(
                self._obs_labels,
                self._obs_parsed,
                self._obs_grid_points,
                self.obs_interval_means,
            )
        ):
            idx_col.extend([j] * len(pts))
            obs_interval_col.extend([label] * len(pts))
            interval_min_col.extend([left] * len(pts))
            interval_max_col.extend([right] * len(pts))
            x_col.extend(pts.tolist())
            f_col.extend(self.latent[pts].tolist())
            mu_col.extend([mu] * len(pts))

        return pd.DataFrame(
            {
                "idx": idx_col,
                "obs_interval": obs_interval_col,
                "interval_min": interval_min_col,
                "interval_max": interval_max_col,
                "x": x_col,
                "f": f_col,
                "mu": mu_col,
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
# SimPoisson1D
# ---------------------------------------------------------------------------


class SimPoisson1D:
    """Data generating process for 1-D Poisson counts over (possibly
    overlapping/enveloping) observed intervals.

    The latent log-rate function is beta0 + g(x), where g is a zero-mean
    Gaussian process with the chosen covariance kernel, lengthscale, and
    marginal variance. The Poisson rate at grid point x is
    lambda(x) = exp(beta0 + g(x)). Counts are generated at the fine grid
    resolution first, c(x) ~ Poisson(lambda(x)), and each observed interval's
    count is the sum of the fine-grid counts falling inside it. Since a sum
    of independent Poissons is Poisson with the summed rate, this is
    equivalent to y_j ~ Poisson(lambda_j) with lambda_j = sum of lambda(x)
    over the interval's grid points; n_observations i.i.d. replicates of the
    fine-grid draw are taken per instance.

    Parameters
    ----------
    obs_intervals:
        List of observed interval strings in "[a,b)" or "[a,b]" format.
        Intervals may overlap or be enveloped by one another. Values must be
        integers and must lie within the domain [0, N-1].
    n_observations:
        Number of i.i.d. observations per observed interval.
    N:
        Number of grid points. The grid is 0, 1, ..., N-1.
    kernel:
        Covariance kernel of the GP: "squared_exponential", "matern52", or
        "matern32".
    lenscale:
        Lengthscale of the GP covariance kernel.
    marg_var:
        Marginal variance of the GP.
    beta0:
        Intercept term: the baseline Poisson log rate.
    seed:
        Seed for the internal RNG. Fixes the GP draw and the observation
        noise across calls to generate().

    Attributes
    ----------
    grid : np.ndarray
        Integer grid points over the domain.
    latent : np.ndarray
        Latent log-rate values beta0 + g(x) at each grid point.
    rates : np.ndarray
        True Poisson rates lambda(x) = exp(latent(x)) at each grid point.
    """

    def __init__(
        self,
        obs_intervals: list[str],
        n_observations: int = 10,
        N: int = 100,
        kernel: Literal[
            "squared_exponential", "matern52", "matern32"
        ] = "squared_exponential",
        lenscale: float = 10.0,
        marg_var: float = 1.0,
        beta0: float = -1.0,
        seed: int | None = None,
    ) -> None:
        if not obs_intervals:
            raise ValueError(
                "obs_intervals must be a non-empty list of interval strings."
            )
        if n_observations < 1:
            raise ValueError("n_observations must be at least 1.")
        if lenscale <= 0:
            raise ValueError("lenscale must be positive.")
        if marg_var <= 0:
            raise ValueError("marg_var must be positive.")

        self.n_observations = n_observations
        self.N = N
        self.kernel_name = kernel
        self.lenscale = lenscale
        self.marg_var = marg_var
        self.beta0 = beta0
        self.seed = seed

        self._rng = np.random.default_rng(seed)

        self.domain: tuple[int, int] = (0, N - 1)
        self.grid = np.arange(N, dtype=float)

        gp_kernel = make_kernel(kernel, lenscale, marg_var)
        g = sample_gp(gp_kernel, self.grid, self._rng)
        self.latent = self.beta0 + g
        self.rates = np.exp(self.latent)

        # Parse and validate observed interval strings
        self._obs_parsed: list[tuple[int, int, bool]] = []
        for s in obs_intervals:
            left, right, right_closed = _parse_interval_string(s)
            domain_start, domain_end = self.domain
            if left < domain_start or right > domain_end:
                raise ValueError(
                    f"Observed interval {s!r} lies outside the domain {self.domain}."
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

        # Pre-compute the total Poisson rate for each observed interval — the
        # parameter of the aggregated count Poisson(sum_{a in I_j} lambda_a),
        # since a sum of independent Poissons is Poisson with the summed rate.
        self._obs_interval_rates: np.ndarray = np.array(
            [self.rates[pts].sum() for pts in self._obs_grid_points]
        )

    def generate(
        self,
        return_true_rates: bool = False,
    ) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
        """Draw n_observations replicate counts per observed interval.

        For each replicate, a Poisson count is drawn independently at every
        grid point from Poisson(lambda(x)), where lambda(x) = exp(latent(x))
        is the true fine-resolution rate. An observed interval's count for
        that replicate is the sum of the fine-grid counts falling inside it.
        Because all observed intervals are summed from the same per-replicate
        draw, overlapping/enveloping intervals are correlated within a
        replicate, matching how aggregated count data actually arises.

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
            y               : Poisson count, sum of fine-grid counts in the interval
            lambda          : true total rate lambda_{I_j} = sum_{a in I_j} lambda_a
        rates_df : pd.DataFrame (only returned when *return_true_rates* is True)
            with columns:
            x      : integer grid point
            lambda : continuous true Poisson rate exp(f(x)) at that grid point
        """
        fine_counts = self._rng.poisson(
            self.rates, size=(self.n_observations, len(self.grid))
        )

        obs_interval_id_col: list[int] = []
        obs_interval_col: list[str] = []
        y_col: list[int] = []
        lambda_col: list[float] = []

        for i, (label, pts, rate) in enumerate(
            zip(self._obs_labels, self._obs_grid_points, self._obs_interval_rates)
        ):
            y = fine_counts[:, pts].sum(axis=1)

            obs_interval_id_col.extend([i] * self.n_observations)
            obs_interval_col.extend([label] * self.n_observations)
            y_col.extend(y.tolist())
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
                    "lambda": np.exp(self.latent),
                }
            )
            return obs_df, rates_df

        return obs_df
