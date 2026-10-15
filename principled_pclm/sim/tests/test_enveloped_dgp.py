import numpy as np
import pandas as pd
import pytest

from principled_pclm.sim import SimPoisson1D

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_OBS = ["[0,5)", "[0,12)", "[5,15]", "[10,25)", "[20,50)", "[50,99]"]


# ---------------------------------------------------------------------------
# Constructor / structural tests
# ---------------------------------------------------------------------------


def test_rates_shape_default():
    dgp = SimPoisson1D(_OBS, seed=0)
    assert dgp.rates.shape == (100,)


def test_rates_all_positive():
    dgp = SimPoisson1D(_OBS, seed=0)
    assert np.all(dgp.rates > 0)


def test_rates_equal_exp_latent():
    dgp = SimPoisson1D(_OBS, seed=0)
    np.testing.assert_array_almost_equal(dgp.rates, np.exp(dgp.latent))


def test_default_beta0_is_minus_one():
    dgp = SimPoisson1D(_OBS, seed=0)
    assert dgp.beta0 == -1.0


def test_obs_intervals_required():
    with pytest.raises(TypeError):
        SimPoisson1D()  # type: ignore[call-arg]


def test_empty_obs_intervals_raises():
    with pytest.raises(ValueError):
        SimPoisson1D([])


def test_n_observations_must_be_positive():
    with pytest.raises(ValueError):
        SimPoisson1D(_OBS, n_observations=0)


def test_lenscale_must_be_positive():
    with pytest.raises(ValueError):
        SimPoisson1D(_OBS, lenscale=0.0)
    with pytest.raises(ValueError):
        SimPoisson1D(_OBS, lenscale=-1.0)


def test_marg_var_must_be_positive():
    with pytest.raises(ValueError):
        SimPoisson1D(_OBS, marg_var=0.0)
    with pytest.raises(ValueError):
        SimPoisson1D(_OBS, marg_var=-1.0)


def test_invalid_interval_string_raises():
    with pytest.raises(ValueError):
        SimPoisson1D(["(0,10)"])  # open left not supported

    with pytest.raises(ValueError):
        SimPoisson1D(["[10,5)"])  # reversed

    with pytest.raises(ValueError):
        SimPoisson1D(["[0,10"])  # missing closing bracket

    with pytest.raises(ValueError):
        SimPoisson1D(["[5,5)"])  # degenerate: left == right


def test_obs_interval_outside_domain_raises():
    with pytest.raises(ValueError):
        SimPoisson1D(["[0,110)"])  # right exceeds domain (0, N-1) for default N=100

    with pytest.raises(ValueError):
        SimPoisson1D(["[-5,10)"])  # left below domain


def test_obs_grid_points_precomputed():
    dgp = SimPoisson1D(["[0,5)", "[5,10]"], seed=0)
    # [0,5) → {0,1,2,3,4}
    np.testing.assert_array_equal(dgp._obs_grid_points[0], np.arange(0, 5))
    # [5,10] → {5,6,7,8,9,10}
    np.testing.assert_array_equal(dgp._obs_grid_points[1], np.arange(5, 11))


def test_overlapping_intervals_accepted():
    # These intervals overlap and envelope each other — should not raise
    dgp = SimPoisson1D(["[0,20)", "[0,10)", "[5,20)"], seed=0)
    assert len(dgp._obs_labels) == 3


# ---------------------------------------------------------------------------
# GP kernel choice
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kernel", ["squared_exponential", "matern52", "matern32"])
def test_kernel_choice_accepted(kernel):
    dgp = SimPoisson1D(_OBS, kernel=kernel, seed=0)
    assert dgp.latent.shape == (100,)
    assert np.all(np.isfinite(dgp.latent))


def test_unknown_kernel_raises():
    with pytest.raises(ValueError):
        SimPoisson1D(_OBS, kernel="unknown", seed=0)  # type: ignore[arg-type]


def test_small_marg_var_clusters_rates_near_exp_beta0():
    dgp = SimPoisson1D(_OBS, marg_var=1e-8, beta0=-1.0, seed=0)
    np.testing.assert_allclose(dgp.rates, np.exp(-1.0), rtol=1e-2)


def test_latent_reproducible_given_seed():
    dgp1 = SimPoisson1D(_OBS, seed=7)
    dgp2 = SimPoisson1D(_OBS, seed=7)
    np.testing.assert_array_equal(dgp1.latent, dgp2.latent)


def test_latent_differs_across_seeds():
    dgp1 = SimPoisson1D(_OBS, seed=1)
    dgp2 = SimPoisson1D(_OBS, seed=2)
    assert not np.allclose(dgp1.latent, dgp2.latent)


# ---------------------------------------------------------------------------
# generate() output tests
# ---------------------------------------------------------------------------


def test_generate_returns_dataframe():
    dgp = SimPoisson1D(_OBS, seed=0)
    result = dgp.generate()
    assert isinstance(result, pd.DataFrame)


def test_generate_columns():
    dgp = SimPoisson1D(_OBS, seed=0)
    result = dgp.generate()
    assert list(result.columns) == ["obs_interval_id", "obs_interval", "y", "lambda"]


def test_generate_row_count():
    n = 7
    dgp = SimPoisson1D(_OBS, n_observations=n, seed=0)
    result = dgp.generate()
    assert len(result) == len(_OBS) * n


def test_y_non_negative_integers():
    dgp = SimPoisson1D(_OBS, seed=1)
    result = dgp.generate()
    assert np.all(result["y"].values >= 0)
    assert pd.api.types.is_integer_dtype(result["y"])


def test_each_obs_interval_appears_n_times():
    n = 8
    dgp = SimPoisson1D(_OBS, n_observations=n, seed=0)
    result = dgp.generate()
    counts = result["obs_interval"].value_counts()
    assert (counts == n).all()


def test_obs_interval_id_range():
    dgp = SimPoisson1D(_OBS, n_observations=3, seed=0)
    result = dgp.generate()
    assert result["obs_interval_id"].min() == 0
    assert result["obs_interval_id"].max() == len(_OBS) - 1


def test_obs_interval_id_maps_to_unique_label():
    dgp = SimPoisson1D(_OBS, n_observations=5, seed=0)
    result = dgp.generate()
    mapping = result.groupby("obs_interval_id")["obs_interval"].nunique()
    assert (mapping == 1).all()


def test_lambda_is_interval_sum_rate():
    """lambda column must equal the total (summed) true rate over the observed interval."""
    dgp = SimPoisson1D(["[0,99]"], n_observations=50, seed=0)
    result = dgp.generate()
    expected_rate = dgp._obs_interval_rates[0]
    assert np.all(np.isclose(result["lambda"], expected_rate))


def test_lambda_matches_interval_sum_rate():
    """lambda column must equal the pre-computed total rate for each observed interval."""
    dgp = SimPoisson1D(
        ["[0,99]", "[10,50)"],
        n_observations=20,
        seed=0,
    )
    result = dgp.generate()
    for i, label in enumerate(dgp._obs_labels):
        rows = result[result["obs_interval"] == label]
        expected = dgp._obs_interval_rates[i]
        assert np.all(np.isclose(rows["lambda"], expected))


def test_lambda_equals_sum_of_grid_point_rates():
    dgp = SimPoisson1D(_OBS, seed=0)
    for i, pts in enumerate(dgp._obs_grid_points):
        np.testing.assert_allclose(dgp._obs_interval_rates[i], dgp.rates[pts].sum())


def test_y_sample_mean_matches_true_sum_rate():
    dgp = SimPoisson1D(_OBS, n_observations=5000, seed=0)
    result = dgp.generate()
    for i, label in enumerate(dgp._obs_labels):
        y = result.loc[result["obs_interval"] == label, "y"].values
        rate = dgp._obs_interval_rates[i]
        tol = 5 * np.sqrt(rate / dgp.n_observations)
        assert abs(y.mean() - rate) < max(tol, 0.1)


def test_overlapping_intervals_correlated_within_replicate():
    """A contained interval's per-replicate count can never exceed the
    containing interval's count, since both are summed from the same
    shared fine-grid draw."""
    dgp = SimPoisson1D(["[0,50)", "[10,30)"], n_observations=200, seed=0)
    result = dgp.generate()
    outer = result.loc[result["obs_interval"] == "[0,50)", "y"].values
    inner = result.loc[result["obs_interval"] == "[10,30)", "y"].values
    assert np.all(outer >= inner)


def test_return_true_rates_shape_and_columns():
    dgp = SimPoisson1D(_OBS, seed=0)
    obs_df, rates_df = dgp.generate(return_true_rates=True)
    assert list(rates_df.columns) == ["x", "lambda"]
    assert len(rates_df) == 100
    np.testing.assert_array_almost_equal(rates_df["lambda"].values, dgp.rates)


def test_seed_reproducibility():
    dgp1 = SimPoisson1D(_OBS, n_observations=5, seed=42)
    dgp2 = SimPoisson1D(_OBS, n_observations=5, seed=42)
    r1 = dgp1.generate()
    r2 = dgp2.generate()
    pd.testing.assert_frame_equal(r1, r2)


def test_consecutive_calls_differ():
    dgp = SimPoisson1D(_OBS, n_observations=20, seed=0)
    r1 = dgp.generate()
    r2 = dgp.generate()
    assert not r1["y"].equals(r2["y"])


def test_rates_fixed_across_generate_calls():
    dgp = SimPoisson1D(_OBS, seed=0)
    rates_before = dgp.rates.copy()
    dgp.generate()
    dgp.generate()
    np.testing.assert_array_equal(dgp.rates, rates_before)


def test_right_closed_interval_includes_endpoint():
    dgp = SimPoisson1D(["[90,99]"], n_observations=100, seed=0)
    pts = dgp._obs_grid_points[0]
    assert 99 in pts


def test_right_open_interval_excludes_endpoint():
    dgp = SimPoisson1D(["[0,10)"], n_observations=100, seed=0)
    pts = dgp._obs_grid_points[0]
    assert 10 not in pts
    assert 0 in pts
