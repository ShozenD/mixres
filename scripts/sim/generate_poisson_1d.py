"""Generate simulated 1-D Poisson count data and save it for the R pclm experiment."""

import argparse
from pathlib import Path

import yaml

from principled_pclm.sim import SimPoisson1D

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "sim"
DEFAULT_CONFIG = REPO_ROOT / "config" / "poisson_1d.yaml"


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def generate(
    seed: int,
    N: int,
    bin_width: int,
    lenscale: float,
    marg_var: float,
    beta0: float,
) -> Path:
    left_limits = list(range(0, N, bin_width))
    right_limits = [left + bin_width - 1 for left in left_limits]
    intervals = [f"[{left},{right}]" for left, right in zip(left_limits, right_limits)]

    # SimPoisson1D.generate() draws Poisson counts at fine grid-point
    # resolution and sums them per observed interval, so each row it returns
    # is already a proper aggregate bin total: Poisson(sum of the bin's
    # grid-point rates). ungroup::pclm() expects exactly that, so a single
    # replicate (n_observations=1) gives one total count per bin directly.
    dgp = SimPoisson1D(
        obs_intervals=intervals,
        n_observations=1,
        N=N,
        lenscale=lenscale,
        marg_var=marg_var,
        beta0=beta0,
        seed=seed,
    )
    df_obs, df_rates = dgp.generate(return_true_rates=True)

    df_obs = df_obs.rename(columns={"lambda": "bin_lambda"})
    df_obs["bin_left"] = left_limits
    df_obs["bin_right"] = right_limits

    df_rates = df_rates.rename(columns={"lambda": "true_lambda"})
    df_rates["bin_id"] = df_rates["x"] // bin_width

    out = df_rates.merge(
        df_obs[["obs_interval_id", "obs_interval", "y", "bin_lambda", "bin_left", "bin_right"]],
        left_on="bin_id",
        right_on="obs_interval_id",
        how="left",
    ).drop(columns="obs_interval_id")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / f"sim_poisson_1d_{seed}_bw{bin_width}.csv"
    out.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--bin-width", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    seed = args.seed if args.seed is not None else config["seed"]
    bin_width = args.bin_width if args.bin_width is not None else config["bin_width"]

    path = generate(
        seed=seed,
        N=config["N"],
        bin_width=bin_width,
        lenscale=config["lenscale"],
        marg_var=config["marg_var"],
        beta0=config["beta0"],
    )
    print(f"Saved simulated data to {path}")
