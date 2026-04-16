import os
import sys

import numpy as np
import pandas as pd
import pytest

# Add the project root to the path to allow direct imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

# Import directly from the _utils module to avoid dependency issues
from mixres.models._utils import (
    create_interval_dict,
    create_overlap_weights,
    find_overlap_intervals,
)


def test_create_overlap_weights_no_overlaps():
    """Test overlap weights with no overlapping intervals"""
    # Create non-overlapping intervals
    intervals = [
        pd.Interval(0, 3, closed="left"),
        pd.Interval(5, 8, closed="left"),
        pd.Interval(10, 12, closed="left"),
    ]

    interval_series = pd.Series(intervals, dtype="category")
    interval_dict = create_interval_dict(interval_series)
    overlap_dict = find_overlap_intervals(interval_dict)
    sampling_effort_dict = {0: 1.0, 1: 2.0, 2: 1.5}

    weights = create_overlap_weights(interval_dict, overlap_dict, sampling_effort_dict)

    # With no overlaps, weights should equal 1.0 everywhere (sampling_effort / sampling_effort)
    assert len(weights) == 3
    assert np.allclose(weights[0], np.ones(3))  # Interval [0,3) has length 3
    assert np.allclose(weights[1], np.ones(3))  # Interval [5,8) has length 3
    assert np.allclose(weights[2], np.ones(2))  # Interval [10,12) has length 2


def test_create_overlap_weights_single_overlap():
    """Test overlap weights with two overlapping intervals"""
    # Create overlapping intervals [0,5) and [3,8)
    intervals = [pd.Interval(0, 5, closed="left"), pd.Interval(3, 8, closed="left")]

    interval_series = pd.Series(intervals, dtype="category")
    interval_dict = create_interval_dict(interval_series)
    overlap_dict = find_overlap_intervals(interval_dict)
    sampling_effort_dict = {0: 1.0, 1: 1.0}

    weights = create_overlap_weights(interval_dict, overlap_dict, sampling_effort_dict)

    # Interval 0 [0,5): positions 0,1,2 no overlap (weight=1.0), positions 3,4 overlap (weight=0.5)
    expected_weights_0 = np.array([1.0, 1.0, 1.0, 0.5, 0.5])
    # Interval 1 [3,8): positions 0,1 overlap (weight=0.5), positions 2,3,4 no overlap (weight=1.0)
    expected_weights_1 = np.array([0.5, 0.5, 1.0, 1.0, 1.0])

    assert len(weights) == 2
    assert np.allclose(weights[0], expected_weights_0)
    assert np.allclose(weights[1], expected_weights_1)


def test_create_overlap_weights_different_sampling_efforts():
    """Test overlap weights with different sampling efforts"""
    # Create overlapping intervals [0,4) and [2,6)
    intervals = [pd.Interval(0, 4, closed="left"), pd.Interval(2, 6, closed="left")]

    interval_series = pd.Series(intervals, dtype="category")
    interval_dict = create_interval_dict(interval_series)
    overlap_dict = find_overlap_intervals(interval_dict)
    sampling_effort_dict = {0: 1.0, 1: 2.0}  # Different sampling efforts

    weights = create_overlap_weights(interval_dict, overlap_dict, sampling_effort_dict)

    # Interval 0 [0,4):
    # - positions 0,1: only interval 0 (total_effort=1.0, weight=1.0/1.0=1.0)
    # - positions 2,3: both intervals (total_effort=1.0+2.0=3.0, weight=1.0/3.0≈0.333)
    expected_weights_0 = np.array([1.0, 1.0, 1.0 / 3.0, 1.0 / 3.0])

    # Interval 1 [2,6):
    # - positions 0,1: both intervals (total_effort=2.0+1.0=3.0, weight=2.0/3.0≈0.667)
    # - positions 2,3: only interval 1 (total_effort=2.0, weight=2.0/2.0=1.0)
    expected_weights_1 = np.array([2.0 / 3.0, 2.0 / 3.0, 1.0, 1.0])

    assert len(weights) == 2
    assert np.allclose(weights[0], expected_weights_0)
    assert np.allclose(weights[1], expected_weights_1)


def test_create_overlap_weights_nested_intervals():
    """Test overlap weights with nested intervals"""
    # Create nested intervals [0,6) containing [2,4)
    intervals = [
        pd.Interval(0, 6, closed="left"),  # Outer interval
        pd.Interval(2, 4, closed="left"),  # Inner interval
    ]

    interval_series = pd.Series(intervals, dtype="category")
    interval_dict = create_interval_dict(interval_series)
    overlap_dict = find_overlap_intervals(interval_dict)
    sampling_effort_dict = {0: 1.0, 1: 1.0}

    weights = create_overlap_weights(interval_dict, overlap_dict, sampling_effort_dict)

    # Interval 0 [0,6):
    # - positions 0,1: only interval 0 (weight=1.0)
    # - positions 2,3: both intervals (weight=0.5)
    # - positions 4,5: only interval 0 (weight=1.0)
    expected_weights_0 = np.array([1.0, 1.0, 0.5, 0.5, 1.0, 1.0])

    # Interval 1 [2,4):
    # - positions 0,1: both intervals (weight=0.5)
    expected_weights_1 = np.array([0.5, 0.5])

    assert len(weights) == 2
    assert np.allclose(weights[0], expected_weights_0)
    assert np.allclose(weights[1], expected_weights_1)


def test_create_overlap_weights_multiple_overlaps():
    """Test overlap weights with multiple overlapping intervals"""
    # Create three overlapping intervals
    intervals = [
        pd.Interval(0, 4, closed="left"),  # Overlaps with 1, 2
        pd.Interval(2, 6, closed="left"),  # Overlaps with 0, 2
        pd.Interval(3, 5, closed="left"),  # Overlaps with 0, 1
    ]

    interval_series = pd.Series(intervals, dtype="category")
    interval_dict = create_interval_dict(interval_series)
    overlap_dict = find_overlap_intervals(interval_dict)
    sampling_effort_dict = {0: 1.0, 1: 1.0, 2: 1.0}

    weights = create_overlap_weights(interval_dict, overlap_dict, sampling_effort_dict)

    # Interval 0 [0,4):
    # - positions 0,1: only interval 0 (weight=1.0)
    # - position 2: intervals 0,1 (weight=0.5)
    # - position 3: intervals 0,1,2 (weight=1/3)
    expected_weights_0 = np.array([1.0, 1.0, 0.5, 1.0 / 3.0])

    # Interval 1 [2,6):
    # - position 0: intervals 0,1 (weight=0.5)
    # - position 1: intervals 0,1,2 (weight=1/3)
    # - position 2: intervals 1,2 (weight=0.5)
    # - position 3: only interval 1 (weight=1.0)
    expected_weights_1 = np.array([0.5, 1.0 / 3.0, 0.5, 1.0])

    # Interval 2 [3,5):
    # - position 0: intervals 0,1,2 (weight=1/3)
    # - position 1: intervals 1,2 (weight=0.5)
    expected_weights_2 = np.array([1.0 / 3.0, 0.5])

    assert len(weights) == 3
    assert np.allclose(weights[0], expected_weights_0)
    assert np.allclose(weights[1], expected_weights_1)
    assert np.allclose(weights[2], expected_weights_2)


def test_create_overlap_weights_empty_intervals():
    """Test overlap weights with empty interval dictionary"""
    interval_dict = {}
    overlap_dict = {}
    sampling_effort_dict = {}

    weights = create_overlap_weights(interval_dict, overlap_dict, sampling_effort_dict)

    assert len(weights) == 0
    assert weights == {}


def test_create_overlap_weights_single_interval():
    """Test overlap weights with single interval"""
    intervals = [pd.Interval(0, 3, closed="left")]

    interval_series = pd.Series(intervals, dtype="category")
    interval_dict = create_interval_dict(interval_series)
    overlap_dict = find_overlap_intervals(interval_dict)
    sampling_effort_dict = {0: 2.0}

    weights = create_overlap_weights(interval_dict, overlap_dict, sampling_effort_dict)

    # Single interval with no overlaps should have weight 1.0 everywhere
    assert len(weights) == 1
    assert np.allclose(weights[0], np.ones(3))


def test_create_overlap_weights_touching_boundaries():
    """Test overlap weights with intervals that touch at boundaries"""
    # Create touching intervals [0,3) and [3,6) - these don't overlap
    intervals = [pd.Interval(0, 3, closed="left"), pd.Interval(3, 6, closed="left")]

    interval_series = pd.Series(intervals, dtype="category")
    interval_dict = create_interval_dict(interval_series)
    overlap_dict = find_overlap_intervals(interval_dict)
    sampling_effort_dict = {0: 1.0, 1: 1.0}

    weights = create_overlap_weights(interval_dict, overlap_dict, sampling_effort_dict)

    # No overlaps, so weights should be 1.0 everywhere
    assert len(weights) == 2
    assert np.allclose(weights[0], np.ones(3))
    assert np.allclose(weights[1], np.ones(3))


def test_create_overlap_weights_zero_sampling_effort():
    """Test overlap weights with zero sampling effort"""
    intervals = [pd.Interval(0, 3, closed="left"), pd.Interval(1, 4, closed="left")]

    interval_series = pd.Series(intervals, dtype="category")
    interval_dict = create_interval_dict(interval_series)
    overlap_dict = find_overlap_intervals(interval_dict)
    sampling_effort_dict = {0: 0.0, 1: 1.0}  # Zero effort for interval 0

    weights = create_overlap_weights(interval_dict, overlap_dict, sampling_effort_dict)

    # Interval 0 has zero sampling effort
    # - position 0: only interval 0 exists, so total_effort=0.0, weight=0.0/0.0=NaN
    # - positions 1,2: both intervals exist, total_effort=0.0+1.0=1.0, weight=0.0/1.0=0.0
    assert len(weights) == 2
    assert np.isnan(weights[0][0])  # First position gives NaN due to 0/0
    assert np.allclose(weights[0][1:], np.zeros(2))  # Remaining positions are 0.0

    # Interval 1: all positions should have weight 1.0 since interval 0 contributes nothing
    assert np.allclose(weights[1], np.ones(3))
