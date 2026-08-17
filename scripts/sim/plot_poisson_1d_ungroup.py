"""Plot the true rate function against the ungroup::pclm() estimate."""

import argparse
from pathlib import Path

import altair as alt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "sim"
FIGURES_DIR = REPO_ROOT / "figures"

# "True rate" matches the color used for the same series in
# notebooks/experiments/poisson_1d_pclm.ipynb; "PCLM estimate" uses a new
# color not already claimed by that notebook's other series; "95% CI" is a
# lighter shade of the estimate color for the confidence band.
COLOR_SCALE = alt.Scale(
    domain=["True rate", "PCLM estimate", "95% CI"],
    range=["#de425b", "#2ca25f", "#a1d99b"],
)


def plot(seed: int, bin_width: int) -> Path:
    truth = pd.read_csv(DATA_DIR / f"sim_poisson_1d_{seed}_bw{bin_width}.csv")[["x", "true_lambda"]]
    fit = pd.read_csv(DATA_DIR / f"sim_poisson_1d_{seed}_bw{bin_width}_ungroup_fit.csv")

    chart_ci = (
        alt.Chart(fit.assign(series="95% CI"))
        .mark_area(opacity=0.3)
        .encode(
            x=alt.X("x:Q", title="x"),
            y=alt.Y("lower:Q", title="λ"),
            y2="upper:Q",
            color=alt.Color("series:N", scale=COLOR_SCALE, title=None),
        )
    )
    chart_true = (
        alt.Chart(truth.assign(series="True rate"))
        .mark_line()
        .encode(
            x=alt.X("x:Q", title="x"),
            y=alt.Y("true_lambda:Q", title="λ"),
            color=alt.Color("series:N", scale=COLOR_SCALE, title=None),
        )
    )
    chart_fit = (
        alt.Chart(fit.assign(series="PCLM estimate"))
        .mark_line(strokeDash=[4, 3])
        .encode(
            x="x:Q",
            y=alt.Y("fitted:Q", title="λ"),
            color=alt.Color("series:N", scale=COLOR_SCALE, title=None),
        )
    )
    chart = (chart_ci + chart_true + chart_fit).properties(
        title=f"ungroup::pclm() estimate vs. true rate (seed={seed}, bin width={bin_width})"
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / f"poisson_1d_ungroup_seed{seed}_bw{bin_width}.svg"
    chart.save(out_path)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--bin-width", type=int, default=5)
    args = parser.parse_args()

    path = plot(args.seed, args.bin_width)
    print(f"Saved plot to {path}")
