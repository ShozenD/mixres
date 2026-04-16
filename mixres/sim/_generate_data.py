import numpy as np
import pandas as pd


def generate_data(N, x_min=0, x_max=99, exp_decay=80, M=2, seed=None):
    """
    Generate synthetic data for testing.

    Parameters
    ----------
    N : int
            Number of samples to generate.
    x_min : int, optional
            Minimum value of x, by default 0.
    x_max : int, optional
            Maximum value of x, by default 99.
    exp_decay : float, optional
            Exponential decay factor for log exposure, by default 10.
    M : int, optional
            Number of fourier basis function sets, by default 2.
    seed : int, optional
            Random seed for reproducibility, by default None.
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)

    # Define grid
    x = np.arange(x_min, x_max + 1)

    # Calculate log exposure and rate
    log_exposure = -x / exp_decay

    X_sin = np.zeros((len(x), M))
    X_cos = np.zeros((len(x), M))

    for m in range(M):
        X_sin[:, m] = np.sin((m + 1) * x / 10)
        X_cos[:, m] = np.cos((m + 1) * x / 10)

    beta_sin = np.random.normal(size=(M,))
    beta_cos = np.random.normal(size=(M,))

    # Generate latent function using Fourier basis
    f = X_sin @ beta_sin + X_cos @ beta_cos

    rate = np.exp(log_exposure + f)

    # Sample indices with replacement
    idx_list = np.arange(x_max + 1)
    idx = np.random.choice(idx_list, size=N, replace=True)

    # Sample data points and corresponding rates
    s = x[idx]
    y = np.random.poisson(rate[idx])

    # Create DataFrames
    df_true = pd.DataFrame({"x": x, "log_exposure": log_exposure, "f": f, "rate": rate})

    df_data = pd.DataFrame(
        {
            "x": s,
            "log_exposure": log_exposure[idx],
            "f": f[idx],
            "rate": rate[idx],
            "y": y,
        }
    )

    return df_true, df_data
