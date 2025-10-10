"""
Module for Mixed Stochastic Differential Equations (SDEs).

This module defines the `Mixedsde` class, which represents a mixed stochastic differential equation model. The class allows users to specify drift and diffusion functions, along with their parameters, and provides methods to compute the drift and diffusion components, as well as to generate simulated
trajectories of the mixed SDE.

Classes:
--------
- Mixedsde: A class for defining and working with mixed stochastic differential equations.
"""

import inspect
from pprint import pprint
from .simulation import _drift, _diffusion, _generate_mixed_sde
from .fit import (
    _estim_tau_vectorized,
    _estim_diffusion_param,
    _estim_drift_param,
    precompute_indices,
)


class Mixedsde:
    """
    A class to represent a mixed stochastic differential equation (SDE).

    This class allows the user to define a mixed SDE by specifying the drift and diffusion functions,
    their parameters, and the distributions of random effects. It provides methods to compute the drift
    and diffusion components and to generate simulated trajectories.

    Attributes:
    -----------
    drift_func : callable
        The function representing the drift component of the SDE.
    drift_re_param : dict
        A dictionary containing parameters for the random effect in the drift.
    diffusion_func : callable
        The function representing the diffusion component of the SDE.
    diffusion_fixed_effect : float
        The fixed effect parameter in the diffusion coefficient.
    diffusion_re_dist : str
        The distribution type for the random effect in the diffusion coefficient.
    diffusion_re_param : dict
        A dictionary containing parameters for the random effect in the diffusion coefficient.

    Parameters:
    -----------
    drift_func : callable
        The drift function to be used in the SDE.
    drift_re_param : dict
        Parameters for the random effect in the drift (e.g., mean and covariance).
    diffusion_func : callable
        The diffusion function to be used in the SDE.
    diffusion_fixed_effect : float
        The fixed effect in the diffusion coefficient.
    diffusion_re_dist : str
        The distribution of the random effect in the diffusion coefficient.
    diffusion_re_param : dict
        Parameters for the random effect in the diffusion coefficient.
    """

    def __init__(
        self,
        drift_func,
        drift_re_param,
        diffusion_func,
        diffusion_fixed_effect,
        diffusion_re_dist,
        diffusion_re_param,
    ):
        self.drift_func = drift_func
        self.drift_re_param = drift_re_param
        self.diffusion_func = diffusion_func
        self.diffusion_fixed_effect = diffusion_fixed_effect
        self.diffusion_re_dist = diffusion_re_dist
        self.diffusion_re_param = diffusion_re_param
        self.nb_re_drift = len(drift_re_param['mu'])

    def __str__(self):
        """
        Prints the expressions of the drift and diffusion functions, along with model parameters.

        This method outputs the source code of the drift and diffusion functions, as well as
        details about the random effects and fixed effects used in the model.
        """
        print("---Expression of the drift function:")
        print("------------------------------------")
        print(inspect.getsource(self.drift_func))
        print("---Expression of the diffusion coefficient:")
        print("-------------------------------------------")
        print(inspect.getsource(self.diffusion_func))
        print("---Model Parameters:")
        print("-------------------- ")
        print(
            "1. The random effect in the drift (phi) follows a Gaussian distribution with mean "
        )
        print(self.drift_re_param["mu"])
        print("and covariance matrix ")
        print(self.drift_re_param["omega2"])
        print(
            "2. The random effect in the diffusion coefficient (tau) follows a "
            + self.diffusion_re_dist
            + " distribution with parameters "
        )
        pprint(self.diffusion_re_param)
        print("3. The fixed effect in the diffusion coefficient (eta) is equal to ")
        print(self.diffusion_fixed_effect)

    def drift(self, y, phi, tau, time):
        """
        Computes the drift component of the SDE.

        Args:
        ----
        y : float
            The current value of the process.
        phi : float
            The random effect in the drift.
        tau : float
            The random effect in the diffusion coefficient.
        time : float
            The current time point.

        Returns:
        -------
        float
            The computed drift value at the given time.
        """
        return _drift(self.drift_func, y, phi, tau, time)

    def diffusion(self, eta, y, tau, time):
        """
        Computes the diffusion component of the SDE.

        Args:
        ----
        eta : float
            The fixed effect in the diffusion coefficient.
        y : float
            The current value of the process.
        tau : float
            The random effect in the diffusion coefficient.
        time : float
            The current time point.

        Returns:
        -------
        float
            The computed diffusion value at the given time.
        """
        return _diffusion(self.diffusion_func, eta, y, tau, time)

    def generate_mixed_sde(self, nb_trajectories, y0, t0, t_max, h_euler, key):
        """
        Generates simulated trajectories of the mixed stochastic differential equation (SDE).

        This method uses the specified drift and diffusion functions, along with their parameters, to generate a specified number of trajectories of the mixed SDE using the Euler-Maruyama method.

        Args:
        ----
        nb_trajectories : int
            The number of trajectories to simulate.
        y0 : float
            The initial value of the process at time t0.
        t0 : float
            The starting time for the simulation.
        t_max : float
            The ending time for the simulation.
        h_euler : float
            The time step size for the Euler-Maruyama method.
        key : jax.random.PRNGKey
            A random key for generating random numbers, used for stochastic components.

        Returns:
        -------
        jax.numpy.ndarray
            A 2D array containing the simulated trajectories, where each row corresponds to a trajectory and each column corresponds to a time point.

        Example:
        --------
        import jax.numpy as jnp
        from jax import random

        # Define drift and diffusion functions
        def drift_func(y, phi, tau, time):
            return y + phi  # Example drift function

        def diffusion_func(eta, y, tau, time):
            return eta * y + tau  # Example diffusion function

        # Initialize parameters
        drift_re_param = {'mu': 0, 'omega2': 1}
        diffusion_re_param = {'param1': 0.5}  # Example parameters
        diffusion_fixed_effect = 1.0
        diffusion_re_dist = 'normal'

        # Create an instance of Mixedsde
        mixed_sde = Mixedsde(drift_func, drift_re_param, diffusion_func, diffusion_fixed_effect, diffusion_re_dist, diffusion_re_param)

        # Generate trajectories
        key = random.PRNGKey(0)
        trajectories = mixed_sde.generate_mixed_sde(nb_trajectories=10, y0=0.0, t0=0.0, t_max=1.0, h_euler=0.01, key=key)
        """
        return _generate_mixed_sde(
            self.drift_func,
            self.diffusion_func,
            self.diffusion_fixed_effect,
            self.diffusion_re_dist,
            self.diffusion_re_param,
            self.drift_re_param,
            nb_trajectories,
            y0,
            t0,
            t_max,
            h_euler,
            key,
        )

    def estim_tau(self, y, eta, time):
        """
        This method estimates the random effects in the diffusion coefficient for the model using a vectorized approach. It calls the helper function _estim_tau_vectorized to perform the estimation using the provided data and model parameters.

        Arguments:
        ----------
            y (array-like): The observed data points.
            eta (array-like): Model parameters related to the stochastic process.
            time (array-like): Time steps or time matrix associated with the data.

        Returns:
        --------
            The estimated tau parameter using the vectorized estimation method.
        """
        return _estim_tau_vectorized(self.diffusion_func, y, eta, time)

    def estim_diffusion_param(self, y, time, init):
        """
        This method estimates the diffusion parameters for the model. It calls the helper function _estim_diffusion_param to perform the estimation based on the provided data, initial conditions, and the diffusion-related functions.

        Arguments:
        ----------
            y (array-like): The observed data points.

            time (array-like): Time matrix or time steps corresponding to the data.

            init (dict): A dictionary containing the initial guesses for the parameters, including 'diffusion', which is the initial guess for the diffusion parameters.

        Returns:
        --------

        The estimated diffusion parameters (eta_hat, theta_tau_hat, tau_hat) based on the given data.
        """
        return _estim_diffusion_param(
            y, time, self.diffusion_re_dist, init["diffusion"], self.diffusion_func
        )

    def estim_drift_param(self, y, time_mat, eta, tau, init, covariance_to_estimate, method):
        """
        This method estimates the drift parameters for the model. It first computes the indices of the elements to estimate in the covariance matrix using the precompute_indices function. Then, it calls the helper function _estim_drift_param to estimate the drift parameters based on the provided data, model functions, initial conditions, and the pre-computed covariance matrix indices.

        Arguments:
        ----------

        y (array-like): The observed data points.

        time_mat (array-like): Time matrix corresponding to the data.

        eta (array-like): Parameters related to the model's stochastic process.

        tau (array-like): The estimated tau parameter.

        init (dict): A dictionary containing the initial guesses for the parameters, including 'drift', which is the initial guess for the drift parameters.

        covariance_to_estimate (array-like): The covariance matrix that contains the elements to estimate.

        Returns:
        --------
        The estimated drift parameters (mu_hat, omega2_hat) based on the given data and model.
        """
        assert covariance_to_estimate.shape[0] == covariance_to_estimate.shape[1], "The covariance matrix must be square."
        assert covariance_to_estimate.shape[0] == self.nb_re_drift, "The covariance matrix must be of size nb_re_drift x nb_re_drift."

        covariance_to_estimate_indices = precompute_indices(
            covariance_to_estimate)
        return _estim_drift_param(
            self.diffusion_func,
            self.drift_func,
            self.nb_re_drift,
            y,
            time_mat,
            eta,
            tau,
            init["drift"],
            covariance_to_estimate_indices,
            method
        )

    def fit_mixed_sde(self, y, time_mat, init, covariance_to_estimate, method):
        """
        This method fits a mixed Stochastic Differential Equation (SDE) model to the provided data. It sequentially estimates the diffusion and drift parameters by calling the estim_diffusion_param and estim_drift_param methods. Finally, it returns the estimated parameters.

        Arguments:
        ----------
        y (array-like): The observed data points.

        time_mat (array-like): The time matrix or time steps corresponding to the data.

        init (dict): A dictionary containing the initial guesses for the parameters, including initial guesses for both the diffusion and drift parameters.

        covariance_to_estimate (array-like): The covariance matrix for which elements need to be estimated.

        Returns:
        --------

        A tuple containing the estimated parameters: (eta_hat, theta_tau_hat, tau_hat, mu_hat, omega2_hat).
        """
        eta_hat, theta_tau_hat, tau_hat = self.estim_diffusion_param(
            y, time_mat[:, 1:], init
        )
        mu_hat, omega2_hat = self.estim_drift_param(
            y, time_mat[:, 1:], eta_hat, tau_hat, init, covariance_to_estimate,
            method
        )
        return (eta_hat, theta_tau_hat, tau_hat, mu_hat, omega2_hat)
