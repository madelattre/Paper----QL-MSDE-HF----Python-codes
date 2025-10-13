"""
Data extraction and visualization utilities for Mixed SDE simulations.

This module provides functions to extract simulated data over a specified time interval
with a new time step and to plot the extracted data using Matplotlib.

Main Functions:
---------------
- extract_data: Extract and resample simulated paths over a user-defined time interval.

Dependencies:
-------------
- JAX (https://github.com/google/jax)
"""

import jax


def extract_data(y, time_mat, start_time, end_time, new_time_step):
    """
    Extract and resample simulated paths over a specified time interval and time step.

    Parameters
    ----------
    y : jax.numpy.ndarray
        Simulated paths, each row representing a trajectory.
    time_mat : jax.numpy.ndarray
        Matrix of time points, repeated for each trajectory.
    start_time : float
        Start time of the extraction interval.
    end_time : float
        End time of the extraction interval.
    new_time_step : float
        Time step for resampling within the interval.

    Returns
    -------
    new_y : jax.numpy.ndarray
        Resampled simulated paths, containing values at the new time points.
    new_time_mat : jax.numpy.ndarray
        Matrix of new time points, repeated for each trajectory.
    """
    new_time_points = jax.numpy.arange(
        start_time, end_time + new_time_step, new_time_step
    )

    original_time_points = time_mat[0]
    indices = jax.numpy.searchsorted(original_time_points, new_time_points)

    new_y = y[:, indices]

    new_time_mat = jax.numpy.tile(new_time_points, (y.shape[0], 1))

    return new_y, new_time_mat
