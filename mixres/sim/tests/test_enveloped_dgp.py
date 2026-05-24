import numpy as np
import pandas as pd
import pytest

from mixres.sim import PoissonEnveloped1D

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_OBS = ["[0,5)", "[0,12)", "[5,15]", "[10,25)", "[20,50)", "[50,100]"]


# ---------------------------------------------------------------------------
# Constructor / structural tests
# ---------------------------------------------------------------------------


def test_rates_shape_default():
    dgp = PoissonEnveloped1D(_OBS)
    # True intervals: 100 / 5 = 20 disjoint intervals
    assert dgp.rates.shape == (20,)


def test_rates_all_positive():
    dgp = PoissonEnveloped1D(_OBS, seed=0)
    assert np.all(dgp.rates > 0)


def test_rates_equal_exp_interval_means():
    dgp = PoissonEnveloped1D(_OBS, seed=0)
    np.testing.assert_array_almost_equal(dgp.rates, np.exp(dgp.interval_means))


def test_cut_points_define_true_intervals():
    dgp = PoissonEnveloped1D(_OBS, cut_points=[18, 65], seed=0)
    assert dgp.intervals == [(0, 18), (18, 65), (65, 100)]
    assert dgp.rates.shape == (3,)


def test_obs_intervals_required():
    with pytest.raises(TypeError):
        PoissonEnveloped1D()  # type: ignore[call-arg]


def test_empty_obs_intervals_raises():
    with pytest.raises(ValueError):
        PoissonEnveloped1D([])


def test_n_observations_must_be_positive():
    with pytest.raises(ValueError):
        PoissonEnveloped1D(_OBS, n_observations=0)


def test_invalid_interval_string_raises():
    with pytest.raises(ValueError):
        PoissonEnveloped1D(["(0,10)"])  # open left not supported

    with pytest.raises(ValueError):
        PoissonEnveloped1D(["[10,5)"])  # reversed

    with pytest.raises(ValueError):
        PoissonEnveloped1D(["[0,10"])  # missing closing bracket

    with pytest.raises(ValueError):
        PoissonEnveloped1D(["[5,5)"])  # degenerate: left == right


def test_obs_interval_outside_domain_raises():
    with pytest.raises(ValueError):
        PoissonEnveloped1D(["[0,110)"])  # right exceeds domain

    with pytest.raises(ValueError):
        PoissonEnveloped1D(["[-5,10)"])  # left below domain


def test_obs_grid_points_precomputed():
    dgp = PoissonEnveloped1D(["[0,5)", "[5,10]"], seed=0)
    # [0,5) → {0,1,2,3,4}
    np.testing.assert_array_equal(dgp._obs_grid_points[0], np.arange(0, 5))
    # [5,10] → {5,6,7,8,9,10}
    np.testing.assert_array_equal(dgp._obs_grid_points[1], np.arange(5, 11))


def test_overlapping_intervals_accepted():
    # These intervals overlap and envelope each other — should not raise
    dgp = PoissonEnveloped1D(["[0,20)", "[0,10)", "[5,20)"], seed=0)
    assert len(dgp._obs_labels) == 3


# ---------------------------------------------------------------------------
# generate() output tests
# ---------------------------------------------------------------------------


def test_generate_returns_dataframe():
    dgp = PoissonEnveloped1D(_OBS, seed=0)
    result = dgp.generate()
    assert isinstance(result, pd.DataFrame)


def test_generate_columns():
    dgp = PoissonEnveloped1D(_OBS, seed=0)
    result = dgp.generate()
    assert list(result.columns) == ["obs_interval_id", "obs_interval", "y", "lambda"]


def test_generate_row_count():
    n = 7
    dgp = PoissonEnveloped1D(_OBS, n_observations=n, seed=0)
    result = dgp.generate()
    assert len(result) == len(_OBS) * n


def test_y_non_negative_integers():
    dgp = PoissonEnveloped1D(_OBS, seed=1)
    result = dgp.generate()
    assert np.all(result["y"].values >= 0)
    assert pd.api.types.is_integer_dtype(result["y"])


def test_each_obs_interval_appears_n_times():
    n = 8
    dgp = PoissonEnveloped1D(_OBS, n_observations=n, seed=0)
    result = dgp.generate()
    counts = result["obs_interval"].value_counts()
    assert (counts == n).all()


def test_obs_interval_id_range():
    dgp = PoissonEnveloped1D(_OBS, n_observations=3, seed=0)
    result = dgp.generate()
    assert result["obs_interval_id"].min() == 0
    assert result["obs_interval_id"].max() == len(_OBS) - 1


def test_obs_interval_id_maps_to_unique_label():
    dgp = PoissonEnveloped1D(_OBS, n_observations=5, seed=0)
    result = dgp.generate()
    mapping = result.groupby("obs_interval_id")["obs_interval"].nunique()
    assert (mapping == 1).all()


def test_lambda_is_interval_mean_rate():
    """lambda column must equal the mean true rate over the observed interval."""
    dgp = PoissonEnveloped1D(["[0,100]"], n_observations=50, seed=0)
    result = dgp.generate()
    expected_rate = dgp._obs_interval_rates[0]
    assert np.all(np.isclose(result["lambda"], expected_rate))


def test_lambda_matches_interval_mean_rate():
    """lambda column must equal the pre-computed mean rate for each observed interval."""
    dgp = PoissonEnveloped1D(
        ["[0,100]", "[10,50)"],
        n_observations=20,
        seed=0,
    )
    result = dgp.generate()
    for i, label in enumerate(dgp._obs_labels):
        rows = result[result["obs_interval"] == label]
        expected = dgp._obs_interval_rates[i]
        assert np.all(np.isclose(rows["lambda"], expected))


def test_seed_reproducibility():
    dgp1 = PoissonEnveloped1D(_OBS, n_observations=5, seed=42)
    dgp2 = PoissonEnveloped1D(_OBS, n_observations=5, seed=42)
    r1 = dgp1.generate()
    r2 = dgp2.generate()
    pd.testing.assert_frame_equal(r1, r2)


def test_consecutive_calls_differ():
    dgp = PoissonEnveloped1D(_OBS, n_observations=20, seed=0)
    r1 = dgp.generate()
    r2 = dgp.generate()
    assert not r1["y"].equals(r2["y"])


def test_rates_fixed_across_generate_calls():
    dgp = PoissonEnveloped1D(_OBS, seed=0)
    rates_before = dgp.rates.copy()
    dgp.generate()
    dgp.generate()
    np.testing.assert_array_equal(dgp.rates, rates_before)


def test_right_closed_interval_includes_endpoint():
    dgp = PoissonEnveloped1D(["[95,100]"], n_observations=100, seed=0)
    # grid point 100 must be reachable
    pts = dgp._obs_grid_points[0]
    assert 100 in pts


def test_right_open_interval_excludes_endpoint():
    dgp = PoissonEnveloped1D(["[0,10)"], n_observations=100, seed=0)
    pts = dgp._obs_grid_points[0]
    assert 10 not in pts
    assert 0 in pts


def test_cut_points_must_be_interior_and_sorted():
    with pytest.raises(ValueError):
        PoissonEnveloped1D(_OBS, cut_points=[65, 18])
    with pytest.raises(ValueError):
        PoissonEnveloped1D(_OBS, cut_points=[0, 18])
    with pytest.raises(ValueError):
        PoissonEnveloped1D(_OBS, cut_points=[18, 100])
