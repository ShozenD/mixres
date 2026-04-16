import os
import sys

import numpy as np
import pandas as pd
import pytest

# Add the project root to the path to allow direct imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

# Import directly from the _utils module to avoid dependency issues
from mixres.models._utils import create_interval_dict, find_overlap_intervals


def test_find_overlap_intervals_no_overlap():
    """Test intervals with no overlaps"""
    interval_series = pd.Series(
        [
            pd.Interval(1, 2, closed="left"),
            pd.Interval(3, 4, closed="left"),
            pd.Interval(5, 6, closed="left"),
            pd.Interval(7, 8, closed="left"),
        ],
        dtype="category",
    )
    intervals = create_interval_dict(interval_series)
    result = find_overlap_intervals(intervals)
    # Should have entry for each interval with empty lists
    assert len(result) == 4
    for code in result:
        assert result[code] == []


def test_find_overlap_intervals_single_overlap():
    """Test intervals with single overlap"""
    interval_series = pd.Series(
        [pd.Interval(1, 4, closed="left"), pd.Interval(3, 5, closed="left")],
        dtype="category",
    )
    intervals = create_interval_dict(interval_series)
    result = find_overlap_intervals(intervals)
    # Should have 2 intervals, each referencing the other
    assert len(result) == 2
    assert 1 in result[0]  # Interval 0 overlaps with interval 1
    assert 0 in result[1]  # Interval 1 overlaps with interval 0


def test_find_overlap_intervals_multiple_overlaps():
    """Test intervals with multiple overlaps"""
    interval_series = pd.Series(
        [
            pd.Interval(1, 3, closed="left"),
            pd.Interval(2, 4, closed="left"),
            pd.Interval(3, 5, closed="left"),
        ],
        dtype="category",
    )
    intervals = create_interval_dict(interval_series)
    result = find_overlap_intervals(intervals)
    assert len(result) == 3  # Should have 3 intervals
    # Interval 0 [1,3) overlaps with interval 1 [2,4)
    assert 1 in result[0]
    assert 0 in result[1]
    # Interval 1 [2,4) overlaps with interval 2 [3,5) at [3,4)
    assert 2 in result[1]
    assert 1 in result[2]
    # Interval 0 [1,3) and interval 2 [3,5) touch at boundary but don't overlap
    assert 2 not in result[0]
    assert 0 not in result[2]


def test_find_overlap_intervals_nested():
    """Test nested intervals"""
    interval_series = pd.Series(
        [pd.Interval(1, 5, closed="left"), pd.Interval(2, 3, closed="left")],
        dtype="category",
    )
    intervals = create_interval_dict(interval_series)
    result = find_overlap_intervals(intervals)
    assert len(result) == 2  # Should have 2 intervals
    assert 1 in result[0]  # Larger interval contains smaller
    assert 0 in result[1]  # Smaller interval is contained in larger


def test_find_overlap_intervals_identical():
    """Test identical intervals"""
    interval_series = pd.Series(
        [pd.Interval(1, 3, closed="left"), pd.Interval(1, 3, closed="left")],
        dtype="category",
    )
    intervals = create_interval_dict(interval_series)
    result = find_overlap_intervals(intervals)
    # Note: identical intervals in pandas categorical will be deduplicated, so only one interval
    assert len(result) == 1
    assert result[0] == []  # Single interval has no overlaps with itself


def test_find_overlap_intervals_empty():
    """Test empty intervals"""
    interval_series = pd.Series([], dtype="category")
    intervals = create_interval_dict(interval_series)
    result = find_overlap_intervals(intervals)
    assert len(result) == 0


def test_find_overlap_intervals_single_interval():
    """Test single interval"""
    interval_series = pd.Series([pd.Interval(1, 3, closed="left")], dtype="category")
    intervals = create_interval_dict(interval_series)
    result = find_overlap_intervals(intervals)
    assert len(result) == 1
    assert result[0] == []  # Single interval has no overlaps


def test_find_overlap_intervals_touching_boundaries():
    """Test intervals that touch at boundaries"""
    interval_series = pd.Series(
        [pd.Interval(1, 3, closed="left"), pd.Interval(3, 5, closed="left")],
        dtype="category",
    )
    intervals = create_interval_dict(interval_series)
    result = find_overlap_intervals(intervals)
    # Intervals [1,3) and [3,5) don't overlap since intervals are left-closed
    assert len(result) == 2
    assert result[0] == []  # No overlaps
    assert result[1] == []  # No overlaps
