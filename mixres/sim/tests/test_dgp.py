import numpy as np
import pandas as pd
import pytest

from mixres.sim import GaussianDisjoint1D


# ---------------------------------------------------------------------------
# Constructor / structural tests
# ---------------------------------------------------------------------------

def test_grid_length():
    dgp = GaussianDisjoint1D()
    assert len(dgp.grid) == 101


def test_grid_values():
    dgp = GaussianDisjoint1D()
    np.testing.assert_array_equal(dgp.grid, np.arange(0, 101, dtype=float))


def test_intervals_tile_domain():
    dgp = GaussianDisjoint1D(interval_width=5)
    intervals = dgp.intervals
    # Correct number of intervals
    assert len(intervals) == 20
    # No gaps or overlaps: each interval starts where the previous ended
    for j in range(1, len(intervals)):
        assert intervals[j][0] == intervals[j - 1][1]
    # Covers full domain
    assert intervals[0][0] == 0
    assert intervals[-1][1] == 100


def test_interval_width_must_divide_100():
    with pytest.raises(ValueError):
        GaussianDisjoint1D(interval_width=7)


def test_sigma2_must_be_positive():
    with pytest.raises(ValueError):
        GaussianDisjoint1D(sigma2=0.0)
    with pytest.raises(ValueError):
        GaussianDisjoint1D(sigma2=-1.0)


def test_interval_means_shape():
    dgp = GaussianDisjoint1D(interval_width=5)
    assert dgp.interval_means.shape == (20,)


def test_latent_shape():
    dgp = GaussianDisjoint1D()
    assert dgp.latent.shape == (101,)


# ---------------------------------------------------------------------------
# generate() output tests
# ---------------------------------------------------------------------------

def test_generate_returns_dataframe():
    dgp = GaussianDisjoint1D(seed=0)
    result = dgp.generate()
    assert isinstance(result, pd.DataFrame)


def test_generate_shape_and_columns():
    dgp = GaussianDisjoint1D(seed=0)
    result = dgp.generate()
    assert result.shape == (101, 4)
    assert list(result.columns) == ["x", "y", "mu", "f"]


def test_generate_x_values():
    dgp = GaussianDisjoint1D(seed=0)
    result = dgp.generate()
    np.testing.assert_array_equal(result["x"].values, np.arange(0, 101))


def test_mu_constant_within_interval():
    """All grid points in the same interval must share the same mu."""
    dgp = GaussianDisjoint1D(interval_width=5, seed=0)
    result = dgp.generate()
    intervals = dgp.intervals
    for j, (start, end) in enumerate(intervals):
        # Last interval is closed on right; all others are half-open [start, end)
        if j < len(intervals) - 1:
            mask = (result["x"] >= start) & (result["x"] < end)
        else:
            mask = (result["x"] >= start) & (result["x"] <= end)
        mus_in_interval = result.loc[mask, "mu"].values
        assert np.all(mus_in_interval == mus_in_interval[0]), (
            f"mu not constant in interval [{start}, {end}]"
        )


def test_f_matches_latent():
    dgp = GaussianDisjoint1D(seed=42)
    result = dgp.generate()
    np.testing.assert_array_almost_equal(result["f"].values, dgp.latent)


def test_seeded_runs_identical():
    dgp1 = GaussianDisjoint1D(seed=7)
    dgp2 = GaussianDisjoint1D(seed=7)
    r1 = dgp1.generate()
    r2 = dgp2.generate()
    pd.testing.assert_frame_equal(r1, r2)


def test_consecutive_generate_differ():
    """Two generate() calls on the same instance should produce different y."""
    dgp = GaussianDisjoint1D(seed=1)
    r1 = dgp.generate()
    r2 = dgp.generate()
    assert not np.allclose(r1["y"].values, r2["y"].values)


def test_latent_fixed_across_generate():
    """Calling generate() twice must yield the same f and mu."""
    dgp = GaussianDisjoint1D(seed=3)
    r1 = dgp.generate()
    r2 = dgp.generate()
    np.testing.assert_array_equal(r1["f"].values, r2["f"].values)
    np.testing.assert_array_equal(r1["mu"].values, r2["mu"].values)


# ---------------------------------------------------------------------------
# Configurable interval width
# ---------------------------------------------------------------------------

def test_interval_width_10():
    dgp = GaussianDisjoint1D(interval_width=10, seed=0)
    assert len(dgp.intervals) == 10
    assert dgp.interval_means.shape == (10,)
    result = dgp.generate()
    assert result.shape == (101, 4)
