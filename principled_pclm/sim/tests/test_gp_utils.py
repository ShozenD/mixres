import numpy as np
import pytest

from principled_pclm.sim.gp_utils import covariance_matrix, make_kernel, sample_gp

_X = np.arange(20, dtype=float)


# ---------------------------------------------------------------------------
# make_kernel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kernel_name", ["squared_exponential", "matern52", "matern32"])
def test_make_kernel_accepts_known_names(kernel_name):
    kernel = make_kernel(kernel_name, lenscale=5.0, marg_var=2.0)
    assert callable(kernel)


def test_make_kernel_rejects_unknown_name():
    with pytest.raises(ValueError):
        make_kernel("unknown", lenscale=5.0, marg_var=2.0)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# covariance_matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kernel_name", ["squared_exponential", "matern52", "matern32"])
def test_covariance_matrix_shape_and_symmetry(kernel_name):
    kernel = make_kernel(kernel_name, lenscale=5.0, marg_var=2.0)
    K = covariance_matrix(kernel, _X)
    assert K.shape == (len(_X), len(_X))
    np.testing.assert_array_almost_equal(K, K.T)


@pytest.mark.parametrize("kernel_name", ["squared_exponential", "matern52", "matern32"])
def test_covariance_matrix_diagonal_equals_marg_var(kernel_name):
    kernel = make_kernel(kernel_name, lenscale=5.0, marg_var=3.0)
    K = covariance_matrix(kernel, _X)
    np.testing.assert_array_almost_equal(np.diagonal(K), np.full(len(_X), 3.0))


# ---------------------------------------------------------------------------
# sample_gp
# ---------------------------------------------------------------------------


def test_sample_gp_shape():
    kernel = make_kernel("squared_exponential", lenscale=5.0, marg_var=1.0)
    rng = np.random.default_rng(0)
    g = sample_gp(kernel, _X, rng)
    assert g.shape == (len(_X),)


def test_sample_gp_reproducible_given_same_rng_seed():
    kernel = make_kernel("squared_exponential", lenscale=5.0, marg_var=1.0)
    g1 = sample_gp(kernel, _X, np.random.default_rng(42))
    g2 = sample_gp(kernel, _X, np.random.default_rng(42))
    np.testing.assert_array_equal(g1, g2)


def test_sample_gp_differs_across_seeds():
    kernel = make_kernel("squared_exponential", lenscale=5.0, marg_var=1.0)
    g1 = sample_gp(kernel, _X, np.random.default_rng(1))
    g2 = sample_gp(kernel, _X, np.random.default_rng(2))
    assert not np.allclose(g1, g2)


def test_sample_gp_differs_across_kernels():
    rng_state = 0
    g_se = sample_gp(
        make_kernel("squared_exponential", lenscale=5.0, marg_var=1.0),
        _X,
        np.random.default_rng(rng_state),
    )
    g_matern32 = sample_gp(
        make_kernel("matern32", lenscale=5.0, marg_var=1.0),
        _X,
        np.random.default_rng(rng_state),
    )
    assert not np.allclose(g_se, g_matern32)
