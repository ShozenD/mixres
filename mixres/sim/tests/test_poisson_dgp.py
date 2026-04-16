import numpy as np
import pandas as pd
import pytest

from mixres.sim import PoissonDisjoint1D


# ---------------------------------------------------------------------------
# Constructor / structural tests
# ---------------------------------------------------------------------------

def test_rates_shape_default():
    dgp = PoissonDisjoint1D()
    assert dgp.rates.shape == (20,)  # 100 / 5 = 20 intervals


def test_rates_all_positive():
    dgp = PoissonDisjoint1D(seed=0)
    assert np.all(dgp.rates > 0)


def test_rates_equal_exp_interval_means():
    dgp = PoissonDisjoint1D(seed=0)
    np.testing.assert_array_almost_equal(dgp.rates, np.exp(dgp.interval_means))


def test_n_observations_must_be_positive():
    with pytest.raises(ValueError):
        PoissonDisjoint1D(n_observations=0)


def test_interval_width_must_divide_100():
    with pytest.raises(ValueError):
        PoissonDisjoint1D(interval_width=7)


# ---------------------------------------------------------------------------
# generate() output tests
# ---------------------------------------------------------------------------

def test_generate_returns_dataframe():
    dgp = PoissonDisjoint1D(seed=0)
    result = dgp.generate()
    assert isinstance(result, pd.DataFrame)


def test_generate_columns():
    dgp = PoissonDisjoint1D(seed=0)
    result = dgp.generate()
    assert list(result.columns) == ["interval_id", "interval", "y", "lambda"]


def test_generate_row_count():
    dgp = PoissonDisjoint1D(interval_width=5, n_observations=10, seed=0)
    result = dgp.generate()
    assert len(result) == 20 * 10


def test_generate_row_count_custom():
    dgp = PoissonDisjoint1D(interval_width=10, n_observations=7, seed=0)
    result = dgp.generate()
    assert len(result) == 10 * 7


def test_y_non_negative_integers():
    dgp = PoissonDisjoint1D(seed=1)
    result = dgp.generate()
    assert np.all(result["y"].values >= 0)
    assert result["y"].dtype in (np.int64, int, object) or pd.api.types.is_integer_dtype(result["y"])


def test_each_interval_appears_n_times():
    n = 8
    dgp = PoissonDisjoint1D(n_observations=n, seed=0)
    result = dgp.generate()
    counts = result["interval"].value_counts()
    assert (counts == n).all()


def test_interval_id_range_and_order():
    dgp = PoissonDisjoint1D(interval_width=5, n_observations=3, seed=0)
    result = dgp.generate()
    # IDs run from 0 to M-1
    assert result["interval_id"].min() == 0
    assert result["interval_id"].max() == len(dgp.intervals) - 1
    # Each ID maps to exactly one unique interval label
    mapping = result.groupby("interval_id")["interval"].nunique()
    assert (mapping == 1).all()
    # IDs are ordered by interval start value
    label_per_id = result.groupby("interval_id")["interval"].first()
    starts = [dgp.intervals[i][0] for i in label_per_id.index]
    assert starts == sorted(starts)


def test_lambda_constant_within_interval():
    dgp = PoissonDisjoint1D(seed=0)
    result = dgp.generate()
    for label in result["interval"].unique():
        lambdas = result.loc[result["interval"] == label, "lambda"].values
        assert np.all(lambdas == lambdas[0]), f"lambda not constant in interval {label}"


def test_lambda_equals_exp_interval_mean():
    dgp = PoissonDisjoint1D(seed=0)
    result = dgp.generate()
    for j, label in enumerate(dgp._labels):
        expected = np.exp(dgp.interval_means[j])
        actual = result.loc[result["interval"] == label, "lambda"].iloc[0]
        assert abs(actual - expected) < 1e-10


def test_interval_labels_format():
    dgp = PoissonDisjoint1D(interval_width=5, seed=0)
    labels = dgp._labels
    # All but last are half-open
    for label in labels[:-1]:
        assert label.endswith(")")
    # Last is closed
    assert labels[-1].endswith("]")
    # First starts at 0
    assert labels[0].startswith("[0,")
    # Last ends at 100
    assert labels[-1].endswith(",100]")


def test_seeded_runs_identical():
    dgp1 = PoissonDisjoint1D(seed=42)
    dgp2 = PoissonDisjoint1D(seed=42)
    r1 = dgp1.generate()
    r2 = dgp2.generate()
    pd.testing.assert_frame_equal(r1, r2)


def test_consecutive_generate_differ():
    dgp = PoissonDisjoint1D(seed=5)
    r1 = dgp.generate()
    r2 = dgp.generate()
    assert not np.array_equal(r1["y"].values, r2["y"].values)


def test_rates_fixed_across_generate():
    dgp = PoissonDisjoint1D(seed=3)
    r1 = dgp.generate()
    r2 = dgp.generate()
    np.testing.assert_array_equal(r1["lambda"].values, r2["lambda"].values)
