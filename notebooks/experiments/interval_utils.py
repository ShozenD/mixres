"""Utilities for interval-indexed data."""

from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd
import jax.numpy as jnp


_INTERVAL_PATTERN = re.compile(
    r"^\s*([\[(])\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*([\])])\s*$"
)


def expand_interval_dataframe(
    df: pd.DataFrame,
    interval_col: str | None = None,
    x_col: str = "x",
) -> pd.DataFrame:
    """Expand interval-indexed rows into one row per integer point."""
    if df.empty:
        out = df.copy()
        out[x_col] = pd.Series(dtype="int64")
        return out

    working_df = df.copy()

    if interval_col is None:
        candidate_cols = []
        for col in working_df.columns:
            series = working_df[col].dropna().astype(str)
            if not series.empty and series.str.match(_INTERVAL_PATTERN).all():
                candidate_cols.append(col)

        if not candidate_cols:
            raise ValueError(
                "Could not infer the interval column. Pass interval_col explicitly."
            )
        interval_col = candidate_cols[0]

    expanded_rows = []

    for _, row in working_df.iterrows():
        interval_value = row[interval_col]

        if isinstance(interval_value, pd.Interval):
            left = float(interval_value.left)
            right = float(interval_value.right)
            left_closed = interval_value.closed in {"left", "both"}
            right_closed = interval_value.closed in {"right", "both"}
        else:
            match = _INTERVAL_PATTERN.match(str(interval_value))
            if match is None:
                raise ValueError(
                    f"Could not parse interval {interval_value!r} in column {interval_col!r}."
                )

            left_bracket, left_str, right_str, right_bracket = match.groups()
            left = float(left_str)
            right = float(right_str)
            left_closed = left_bracket == "["
            right_closed = right_bracket == "]"

        start = math.ceil(left) if left_closed else math.floor(left) + 1
        end = math.floor(right) if right_closed else math.ceil(right) - 1

        for x in range(start, end + 1):
            new_row = row.to_dict()
            new_row[x_col] = x
            expanded_rows.append(new_row)

    return pd.DataFrame(expanded_rows).reset_index(drop=True)


def compute_interval_widths(intervals: list[str], dtype=np.float32) -> np.ndarray:
    """Compute integer widths for interval strings like "[0,5)" or "[95,100]"."""
    if not intervals:
        raise ValueError("intervals must contain at least one interval string.")

    widths = []
    for interval in intervals:
        match = _INTERVAL_PATTERN.match(str(interval))
        if match is None:
            raise ValueError(f"Could not parse interval: {interval!r}")

        left_bracket, left_str, right_str, right_bracket = match.groups()
        left = float(left_str)
        right = float(right_str)

        start = math.ceil(left) if left_bracket == "[" else math.floor(left) + 1
        end = math.floor(right) if right_bracket == "]" else math.ceil(right) - 1

        if end < start:
            raise ValueError(f"Interval contains no integer points: {interval!r}")

        widths.append(end - start + 1)

    return np.asarray(widths, dtype=dtype)


def build_interval_sum_matrix(
    intervals: list[str],
    dtype=jnp.float32,
    return_support: bool = False,
) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
    """Build a JAX matrix that sums a latent vector over integer intervals.

    Each row corresponds to one interval; columns span the integer support.
    Entries are 1 inside the interval and 0 elsewhere.
    """
    if not intervals:
        raise ValueError("intervals must contain at least one interval string.")

    parsed_intervals = []
    min_x = np.inf
    max_x = -np.inf

    for interval in intervals:
        match = _INTERVAL_PATTERN.match(str(interval))
        if match is None:
            raise ValueError(f"Could not parse interval: {interval!r}")

        left_bracket, left_str, right_str, right_bracket = match.groups()
        left = float(left_str)
        right = float(right_str)

        start = math.ceil(left) if left_bracket == "[" else math.floor(left) + 1
        end = math.floor(right) if right_bracket == "]" else math.ceil(right) - 1

        if end < start:
            raise ValueError(f"Interval contains no integer points: {interval!r}")

        parsed_intervals.append((start, end))
        min_x = min(min_x, start)
        max_x = max(max_x, end)

    width = int(max_x - min_x + 1)
    matrix = np.zeros((len(parsed_intervals), width), dtype=np.float32)

    for row_idx, (start, end) in enumerate(parsed_intervals):
        matrix[row_idx, int(start - min_x) : int(end - min_x + 1)] = 1.0

    matrix = jnp.asarray(matrix, dtype=dtype)

    if return_support:
        support = jnp.arange(int(min_x), int(max_x) + 1)
        return matrix, support

    return matrix
