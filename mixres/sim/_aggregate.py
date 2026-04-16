import numpy as np
import pandas as pd

def assign_disjoint_bins(
	x : np.ndarray,
	bin_size : int = 5
) -> pd.Series:
	"""
	Assign disjoint bins to the data based on the 'x' values.
	This function creates bins of size 5 from 0 to 100.
	"""
	x_min = x.min()
	x_max = x.max()

	return pd.cut(
		x,
    bins=np.arange(x_min, x_max + 2, bin_size),
    right=False
  )
 
def assign_overlapping_bins(
	x : np.ndarray,
	left_limits : list,
	right_limits : list,
	sampling_effort : list
) -> pd.Series:
	"""
	Assign overlapping bins to the data based on the 'x' values.
	This function creates bins of size 5 with an overlap of 2.
	"""

	assert len(left_limits) == len(right_limits), "Left and right limits must have the same length."

	intervals = []
	for left, right in zip(left_limits, right_limits):
		intervals.append(pd.Interval(left, right, closed='left'))
	
	# For each x value, find which interval it belongs to
	assigned_intervals = []

	for x_val in x:
		# Find all intervals that contain this x value
		containing_intervals = []
		containing_efforts = []
		
		for i, interval in enumerate(intervals):
			if x_val >= interval.left and x_val < interval.right:
				containing_intervals.append(interval)
				containing_efforts.append(sampling_effort[i])
		
		if containing_intervals:
			if len(containing_intervals) == 1:
				# Only one interval contains this point
				assigned_intervals.append(containing_intervals[0])
			else:
				# Multiple intervals contain this point, choose based on sampling effort
				total_effort = sum(containing_efforts)
				probabilities = [effort / total_effort for effort in containing_efforts]
				
				# Sample based on probabilities
				chosen_idx = np.random.choice(len(containing_intervals), p=probabilities)
				assigned_intervals.append(containing_intervals[chosen_idx])
		else:
			# No interval contains this point
			assigned_intervals.append(None)
 
	# Convert to categorical Series
	return pd.Series(assigned_intervals, dtype='category')