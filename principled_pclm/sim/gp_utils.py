from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Kernel, Matern

KernelName = Literal["squared_exponential", "matern52", "matern32"]


def make_kernel(kernel: KernelName, lenscale: float, marg_var: float) -> Kernel:
    """Build a scikit-learn covariance kernel with the given lengthscale and
    marginal variance: Cov(g(x), g(x)) = marg_var.
    """
    if kernel == "squared_exponential":
        base = RBF(length_scale=lenscale)
    elif kernel == "matern52":
        base = Matern(length_scale=lenscale, nu=2.5)
    elif kernel == "matern32":
        base = Matern(length_scale=lenscale, nu=1.5)
    else:
        raise ValueError(
            f"Unknown kernel {kernel!r}. Expected one of "
            "'squared_exponential', 'matern52', 'matern32'."
        )
    return ConstantKernel(constant_value=marg_var) * base


def covariance_matrix(kernel: Kernel, x: np.ndarray) -> np.ndarray:
    """Evaluate the covariance (Gram) matrix K(x, x) for 1-D grid points x."""
    return kernel(x.reshape(-1, 1))


def sample_gp(kernel: Kernel, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw one sample path from a zero-mean GP(0, kernel) at points x."""
    gpr = GaussianProcessRegressor(kernel=kernel)
    seed = int(rng.integers(0, 2**32 - 1))
    return gpr.sample_y(x.reshape(-1, 1), n_samples=1, random_state=seed).ravel()
