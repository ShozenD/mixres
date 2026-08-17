import numpy as np
import pandas as pd
import pytest

from principled_pclm.sim import SimGaussian1D

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_OBS = ["[0,10)", "[10,25]", "[20,60)", "[60,99]"]


# ---------------------------------------------------------------------------
# Constructor / validation tests
# ---------------------------------------------------------------------------


def test_obs_intervals_required():
    with pytest.raises(TypeError):
        SimGaussian1D()  # type: ignore[call-arg]


def test_empty_obs_intervals_raises():
    with pytest.raises(ValueError):
        SimGaussian1D([])


def test_n_obs_must_be_positive():
    with pytest.raises(ValueError):
        SimGaussian1D(_OBS, n_obs=0)


def test_noise_std_must_be_positive():
    with pytest.raises(ValueError):
        SimGaussian1D(_OBS, noise_std=0.0)
    with pytest.raises(ValueError):
        SimGaussian1D(_OBS, noise_std=-1.0)


def test_invalid_interval_string_raises():
    with pytest.raises(ValueError):
        SimGaussian1D(["(0,10)"])  # open left not supported

    with pytest.raises(ValueError):
        SimGaussian1D(["[10,5)"])  # reversed

    with pytest.raises(ValueError):
        SimGaussian1D(["[0,10"])  # missing closing bracket

    with pytest.raises(ValueError):
        SimGaussian1D(["[5,5)"])  # degenerate: left == right


def test_obs_interval_outside_domain_raises():
    with pytest.raises(ValueError):
        SimGaussian1D(["[0,100)"])  # right exceeds domain (0, N-1) for default N=100

    with pytest.raises(ValueError):
        SimGaussian1D(["[-5,10)"])  # left below domain


def test_overlapping_intervals_accepted():
    dgp = SimGaussian1D(["[0,20)", "[0,10)", "[5,20)"], seed=0)
    assert len(dgp._obs_labels) == 3


def test_obs_grid_points_precomputed():
    dgp = SimGaussian1D(["[0,5)", "[5,10]"], seed=0)
    np.testing.assert_array_equal(dgp._obs_grid_points[0], np.arange(0, 5))
    np.testing.assert_array_equal(dgp._obs_grid_points[1], np.arange(5, 11))


def test_right_closed_interval_includes_endpoint():
    dgp = SimGaussian1D(["[90,99]"], seed=0)
    pts = dgp._obs_grid_points[0]
    assert 99 in pts


def test_right_open_interval_excludes_endpoint():
    dgp = SimGaussian1D(["[0,10)"], seed=0)
    pts = dgp._obs_grid_points[0]
    assert 10 not in pts
    assert 0 in pts


# ---------------------------------------------------------------------------
# Domain / grid (N) tests
# ---------------------------------------------------------------------------


def test_grid_length_and_values_default_N():
    dgp = SimGaussian1D(_OBS, seed=0)
    assert len(dgp.grid) == 100
    np.testing.assert_array_equal(dgp.grid, np.arange(100, dtype=float))


def test_domain_is_0_to_N_minus_1():
    dgp = SimGaussian1D(_OBS, seed=0)
    assert dgp.domain == (0, 99)


def test_custom_N():
    dgp = SimGaussian1D(["[0,10)", "[10,49]"], N=50, seed=0)
    assert len(dgp.grid) == 50
    assert dgp.domain == (0, 49)

    # Accepted: right-closed at the new domain end
    SimGaussian1D(["[0,49]"], N=50, seed=0)

    # Rejected: right exceeds the new domain end
    with pytest.raises(ValueError):
        SimGaussian1D(["[0,50)"], N=50, seed=0)


# ---------------------------------------------------------------------------
# omega_max bounds
# ---------------------------------------------------------------------------


def test_omega_max_too_small_raises():
    with pytest.raises(ValueError):
        SimGaussian1D(_OBS, N=100, omega_max=(2 * np.pi / 100) / 2, seed=0)


def test_omega_max_too_large_raises():
    with pytest.raises(ValueError):
        SimGaussian1D(_OBS, N=100, omega_max=np.pi, seed=0)


def test_omega_max_at_lower_bound_accepted():
    dgp = SimGaussian1D(_OBS, N=100, omega_max=2 * np.pi / 100, seed=0)
    assert dgp.latent.shape == (100,)


def test_small_N_with_default_omega_max_raises():
    with pytest.raises(ValueError):
        SimGaussian1D(["[0,1]"], N=2, seed=0)


# ---------------------------------------------------------------------------
# Latent process
# ---------------------------------------------------------------------------


def test_latent_shape():
    dgp = SimGaussian1D(_OBS, seed=0)
    assert dgp.latent.shape == (100,)


def test_latent_reproducible_given_seed():
    dgp1 = SimGaussian1D(_OBS, seed=7)
    dgp2 = SimGaussian1D(_OBS, seed=7)
    np.testing.assert_array_equal(dgp1.latent, dgp2.latent)


def test_latent_differs_across_seeds():
    dgp1 = SimGaussian1D(_OBS, seed=1)
    dgp2 = SimGaussian1D(_OBS, seed=2)
    assert not np.allclose(dgp1.latent, dgp2.latent)


# ---------------------------------------------------------------------------
# generate() output tests
# ---------------------------------------------------------------------------


def test_generate_returns_dataframe():
    dgp = SimGaussian1D(_OBS, seed=0)
    result = dgp.generate()
    assert isinstance(result, pd.DataFrame)


def test_generate_columns():
    dgp = SimGaussian1D(_OBS, seed=0)
    result = dgp.generate()
    assert list(result.columns) == ["idx", "y"]
    assert len(result.columns) == 2


def test_generate_row_count():
    n = 7
    dgp = SimGaussian1D(_OBS, n_obs=n, seed=0)
    result = dgp.generate()
    assert len(result) == len(_OBS) * n


def test_each_idx_appears_n_obs_times():
    n = 8
    dgp = SimGaussian1D(_OBS, n_obs=n, seed=0)
    result = dgp.generate()
    counts = result["idx"].value_counts()
    assert (counts == n).all()


def test_idx_range():
    dgp = SimGaussian1D(_OBS, n_obs=3, seed=0)
    result = dgp.generate()
    assert result["idx"].min() == 0
    assert result["idx"].max() == len(_OBS) - 1


def test_y_sample_mean_close_to_obs_interval_mean():
    dgp = SimGaussian1D(_OBS, n_obs=5000, noise_std=0.1, seed=0)
    result = dgp.generate()
    for j, mu in enumerate(dgp.obs_interval_means):
        sample = result.loc[result["idx"] == j, "y"].values
        tol = 4 * dgp.noise_std / np.sqrt(dgp.n_obs)
        assert abs(sample.mean() - mu) < tol


def test_y_sample_std_close_to_noise_std():
    dgp = SimGaussian1D(_OBS, n_obs=5000, noise_std=0.1, seed=0)
    result = dgp.generate()
    for j in range(len(_OBS)):
        sample = result.loc[result["idx"] == j, "y"].values
        assert abs(sample.std() - dgp.noise_std) < 0.05 * dgp.noise_std * 4


def test_seed_reproducibility():
    dgp1 = SimGaussian1D(_OBS, n_obs=5, seed=42)
    dgp2 = SimGaussian1D(_OBS, n_obs=5, seed=42)
    r1 = dgp1.generate()
    r2 = dgp2.generate()
    pd.testing.assert_frame_equal(r1, r2)


def test_consecutive_generate_calls_differ():
    dgp = SimGaussian1D(_OBS, n_obs=20, seed=0)
    r1 = dgp.generate()
    r2 = dgp.generate()
    assert not r1["y"].equals(r2["y"])


def test_obs_interval_means_fixed_across_generate_calls():
    dgp = SimGaussian1D(_OBS, seed=0)
    means_before = dgp.obs_interval_means.copy()
    dgp.generate()
    dgp.generate()
    np.testing.assert_array_equal(dgp.obs_interval_means, means_before)


# ---------------------------------------------------------------------------
# get_reference_df() tests
# ---------------------------------------------------------------------------


def test_get_reference_df_returns_dataframe():
    dgp = SimGaussian1D(_OBS, seed=0)
    result = dgp.get_reference_df()
    assert isinstance(result, pd.DataFrame)


def test_get_reference_df_columns():
    dgp = SimGaussian1D(_OBS, seed=0)
    result = dgp.get_reference_df()
    assert list(result.columns) == [
        "idx",
        "obs_interval",
        "interval_min",
        "interval_max",
        "x",
        "f",
        "mu",
    ]


def test_get_reference_df_interval_min_max_match_parsed_bounds():
    dgp = SimGaussian1D(_OBS, seed=0)
    result = dgp.get_reference_df()
    for j, (left, right, _) in enumerate(dgp._obs_parsed):
        rows = result.loc[result["idx"] == j]
        assert (rows["interval_min"] == left).all()
        assert (rows["interval_max"] == right).all()


def test_get_reference_df_row_count():
    dgp = SimGaussian1D(_OBS, seed=0)
    result = dgp.get_reference_df()
    expected = sum(len(pts) for pts in dgp._obs_grid_points)
    assert len(result) == expected


def test_get_reference_df_f_matches_latent():
    dgp = SimGaussian1D(_OBS, seed=0)
    result = dgp.get_reference_df()
    np.testing.assert_array_almost_equal(
        result["f"].values, dgp.latent[result["x"].values]
    )


def test_get_reference_df_mu_constant_within_idx():
    dgp = SimGaussian1D(_OBS, seed=0)
    result = dgp.get_reference_df()
    for j, mu in enumerate(dgp.obs_interval_means):
        rows = result.loc[result["idx"] == j, "mu"]
        assert np.all(np.isclose(rows.values, mu))


def test_get_reference_df_obs_interval_matches_label():
    dgp = SimGaussian1D(_OBS, seed=0)
    result = dgp.get_reference_df()
    for j, label in enumerate(dgp._obs_labels):
        rows = result.loc[result["idx"] == j, "obs_interval"]
        assert (rows == label).all()


def test_get_reference_df_overlapping_point_appears_once_per_interval():
    dgp = SimGaussian1D(["[0,20)", "[0,10)", "[5,20)"], seed=0)
    result = dgp.get_reference_df()
    # x=7 is covered by all three intervals
    assert (result.loc[result["x"] == 7, "idx"].sort_values().tolist()) == [0, 1, 2]


def test_get_reference_df_deterministic_and_does_not_consume_rng():
    dgp = SimGaussian1D(_OBS, seed=0)
    r1 = dgp.get_reference_df()
    r2 = dgp.get_reference_df()
    pd.testing.assert_frame_equal(r1, r2)

    dgp1 = SimGaussian1D(_OBS, seed=0)
    dgp2 = SimGaussian1D(_OBS, seed=0)
    dgp1.get_reference_df()
    g1 = dgp1.generate()
    g2 = dgp2.generate()
    pd.testing.assert_frame_equal(g1, g2)
