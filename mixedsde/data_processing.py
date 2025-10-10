"""
Module for data extraction and visualization.

This module provides functions to extract simulated data over a specified time interval
with a new time step and to plot the extracted data using Matplotlib.

Functions:
----------
- extract_data(y, time_mat, start_time, end_time, new_time_step)
"""

import jax


def extract_data(y, time_mat, start_time, end_time, new_time_step):
    """
    Extract data from the simulated paths on a smaller time interval with a larger time step.

    Args:
    ----
    y : jax.numpy.ndarray
        Simulated paths, where each row represents a different trajectory.
    time_mat : jax.numpy.ndarray
        Matrix of time points, repeated for each trajectory.
    start_time : float
        Start time of the new interval.
    end_time : float
        End time of the new interval.
    new_time_step : float
        New time step for resampling.

    Returns:
    -------
    tuple
        new_y : jax.numpy.ndarray
            Resampled simulated paths, containing values corresponding to the new time points.
        new_time_mat : jax.numpy.ndarray
            Matrix of new time points, repeated for each trajectory.

    Example:
    --------
    import jax.numpy as jnp

    # Example simulated paths and time matrix
    y = jnp.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    time_mat = jnp.array([[0, 1, 2, 3]])

    new_y, new_time_mat = extract_data(y, time_mat, start_time=1, end_time=3, new_time_step=1)
    """
    # Generate new time points for the smaller interval with the larger time step
    new_time_points = jax.numpy.arange(
        start_time, end_time + new_time_step, new_time_step
    )

    # Find the indices of the new time points in the original time points
    original_time_points = time_mat[0]
    indices = jax.numpy.searchsorted(original_time_points, new_time_points)

    # Extract the corresponding values from the simulated paths
    new_y = y[:, indices]

    # Create a new time matrix with the new time points
    new_time_mat = jax.numpy.tile(new_time_points, (y.shape[0], 1))

    return new_y, new_time_mat
