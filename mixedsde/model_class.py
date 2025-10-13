"""
Mixed Stochastic Differential Equations (SDEs) Framework.

This module provides the `Mixedsde` class for simulating and estimating stochastic differential equations with both fixed and random effects. It leverages JAX for efficient computation, vectorization, and just-in-time (JIT) compilation.

Features:
---------
- Simulation of mixed SDE trajectories using Euler-Maruyama.
- Estimation of drift and diffusion parameters with random effects.
- Flexible specification of drift and diffusion functions.
- Parameter management for both fixed and random effects.

Dependencies:
-------------
- JAX (https://github.com/google/jax)

Classes:
--------
Mixedsde
    Main class for defining, simulating, and fitting mixed SDE models.
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
    Mixed Stochastic Differential Equation (SDE) Model.

    The `Mixedsde` class represents SDEs with both fixed and random effects in drift and diffusion terms. It provides methods for simulation, parameter estimation, and model fitting.

    Effects:
    --------
    - Fixed effect: Parameter constant across all trajectories.
    - Random effect: Parameter varying across trajectories, modeled by a distribution.

    Attributes:
    -----------
    drift_func : callable
        Function for the drift term: drift(y, phi, tau, time).
    drift_re_param : dict
        Parameters for the random effect in drift (e.g., mean 'mu', covariance 'omega2').
    diffusion_func : callable
        Function for the diffusion term: diffusion(eta, y, tau, time).
    diffusion_fixed_effect : float
        Fixed effect parameter in the diffusion coefficient.
    diffusion_re_dist : str
        Distribution type for the random effect in diffusion (e.g., 'normal').
    diffusion_re_param : dict
        Parameters for the random effect in diffusion.

    Methods:
    --------
    - drift(y, phi, tau, time): Compute drift value.
    - diffusion(eta, y, tau, time): Compute diffusion value.
    - generate_mixed_sde(...): Simulate trajectories.
    - estim_tau(...): Estimate random effects in diffusion.
    - estim_diffusion_param(...): Estimate diffusion parameters.
    - estim_drift_param(...): Estimate drift parameters.
    - fit_mixed_sde(...): Fit the mixed SDE model to data.

    Parameters:
    -----------
    drift_func : callable
        Drift function.
    drift_re_param : dict
        Random effect parameters for drift.
    diffusion_func : callable
        Diffusion function.
    diffusion_fixed_effect : float
        Fixed effect in diffusion.
    diffusion_re_dist : str
        Distribution for random effect in diffusion.
    diffusion_re_param : dict
        Random effect parameters for diffusion.
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
        """
        Initialize a Mixedsde model instance.

        Parameters
        ----------
        drift_func : callable
            Function specifying the drift term of the SDE: drift(y, phi, tau, time).
        drift_re_param : dict
            Dictionary of parameters for the random effect in the drift (e.g., mean 'mu', covariance 'omega2').
        diffusion_func : callable
            Function specifying the diffusion term of the SDE: diffusion(eta, y, tau, time).
        diffusion_fixed_effect : float
            Fixed effect parameter for the diffusion coefficient.
        diffusion_re_dist : str
            Name of the distribution for the random effect in diffusion (e.g., 'normal').
        diffusion_re_param : dict
            Dictionary of parameters for the random effect in diffusion.
        """
        self.drift_func = drift_func
        self.drift_re_param = drift_re_param
        self.diffusion_func = diffusion_func
        self.diffusion_fixed_effect = diffusion_fixed_effect
        self.diffusion_re_dist = diffusion_re_dist
        self.diffusion_re_param = diffusion_re_param
        self.nb_re_drift = len(drift_re_param['mu'])

    def __str__(self):
        """
        Return a detailed string representation of the Mixedsde model.

        This includes:
        - The source code of the drift and diffusion functions.
        - The parameters for random effects in drift (mean and covariance).
        - The distribution and parameters for random effects in diffusion.
        - The value of the fixed effect in the diffusion coefficient.

        Returns
        -------
        str
            A multi-line string describing the model's functions and parameters.
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
        Evaluate the drift component of the SDE at a given state and time.

        Parameters
        ----------
        y : float or array-like
            Current value(s) of the process.
        phi : float or array-like
            Random effect(s) in the drift term.
        tau : float or array-like
            Random effect(s) in the diffusion coefficient.
        time : float or array-like
            Current time point(s).

        Returns
        -------
        float or array-like
            The computed drift value(s) for the given input(s).
        """
        return _drift(self.drift_func, y, phi, tau, time)

    def diffusion(self, eta, y, tau, time):
        """
        Evaluate the diffusion component of the SDE at a given state and time.

        Parameters
        ----------
        eta : float
            Fixed effect parameter in the diffusion coefficient.
        y : float or array-like
            Current value(s) of the process.
        tau : float or array-like
            Random effect(s) in the diffusion coefficient.
        time : float or array-like
            Current time point(s).

        Returns
        -------
        float or array-like
            The computed diffusion value(s) for the given input(s).
        """
        return _diffusion(self.diffusion_func, eta, y, tau, time)

    def generate_mixed_sde(self, nb_trajectories, y0, t0, t_max, h_euler, key):
        """
        Simulate multiple trajectories of the mixed SDE using the Euler-Maruyama method.

        Parameters
        ----------
        nb_trajectories : int
            Number of independent trajectories to simulate.
        y0 : float
            Initial value of the process at time t0.
        t0 : float
            Start time for simulation.
        t_max : float
            End time for simulation.
        h_euler : float
            Time step size for the Euler-Maruyama discretization.
        key : jax.random.PRNGKey
            JAX random key for stochastic sampling.

        Returns
        -------
        jax.numpy.ndarray
            Array of shape (nb_trajectories, num_time_points) containing simulated trajectories.
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
        Estimate the random effects (tau) in the diffusion coefficient using a vectorized approach.

        Parameters
        ----------
        y : array-like
            Observed data points for each trajectory.
        eta : array-like
            Fixed effect parameter(s) for the diffusion coefficient.
        time : array-like
            Time points or time matrix associated with the data.

        Returns
        -------
        array-like
            Estimated values of the random effect tau for each trajectory.
        """
        return _estim_tau_vectorized(self.diffusion_func, y, eta, time)

    def estim_diffusion_param(self, y, time, init):
        """
        Estimate the diffusion parameters of the mixed SDE model.

        Parameters
        ----------
        y : array-like
            Observed data points for each trajectory.
        time : array-like
            Time matrix or time steps corresponding to the data.
        init : dict
            Dictionary containing initial guesses for the diffusion parameters (key: 'diffusion').

        Returns
        -------
        tuple
            Estimated diffusion parameters:
            - eta_hat: Estimated fixed effect in diffusion.
            - theta_tau_hat: Estimated parameters for the random effect distribution in diffusion.
            - tau_hat: Estimated random effects in diffusion for each trajectory.
        """
        return _estim_diffusion_param(
            y, time, self.diffusion_re_dist, init["diffusion"], self.diffusion_func
        )

    def estim_drift_param(self, y, time_mat, eta, tau, init, covariance_to_estimate, method):
        """
        Estimate the drift parameters of the mixed SDE model.

        Parameters
        ----------
        y : array-like
            Observed data points for each trajectory.
        time_mat : array-like
            Time matrix or time steps corresponding to the data.
        eta : array-like
            Estimated fixed effect(s) in the diffusion coefficient.
        tau : array-like
            Estimated random effects in the diffusion coefficient for each trajectory.
        init : dict
            Dictionary containing initial guesses for the drift parameters (key: 'drift').
        covariance_to_estimate : array-like
            Covariance matrix specifying which elements to estimate for the drift random effects.
        method : str
            Estimation method to use (e.g., optimization algorithm name).

        Returns
        -------
        tuple
            Estimated drift parameters:
            - mu_hat: Estimated mean of the random effect in drift.
            - omega2_hat: Estimated covariance matrix of the random effect in drift.
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
        Fit the mixed SDE model to observed data by sequentially estimating diffusion and drift parameters.

        Parameters
        ----------
        y : array-like
            Observed data points for each trajectory.
        time_mat : array-like
            Time matrix or time steps corresponding to the data.
        init : dict
            Dictionary containing initial guesses for both diffusion and drift parameters.
        covariance_to_estimate : array-like
            Covariance matrix specifying which elements to estimate for the drift random effects.
        method : str
            Estimation method to use for drift parameters (e.g., optimization algorithm name).

        Returns
        -------
        tuple
            Estimated parameters:
            - eta_hat: Estimated fixed effect in diffusion.
            - theta_tau_hat: Estimated parameters for the random effect distribution in diffusion.
            - tau_hat: Estimated random effects in diffusion for each trajectory.
            - mu_hat: Estimated mean of the random effect in drift.
            - omega2_hat: Estimated covariance matrix of the random effect in drift.

        Notes
        -----
        This method first estimates the diffusion parameters, then uses those results to estimate the drift parameters.
        """
        eta_hat, theta_tau_hat, tau_hat = self.estim_diffusion_param(
            y, time_mat[:, 1:], init
        )
        mu_hat, omega2_hat = self.estim_drift_param(
            y, time_mat[:, 1:], eta_hat, tau_hat, init, covariance_to_estimate,
            method
        )
        return (eta_hat, theta_tau_hat, tau_hat, mu_hat, omega2_hat)
