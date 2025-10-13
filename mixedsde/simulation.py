"""
Simulation utilities for Mixed Stochastic Differential Equations (SDEs).

This module provides functions to define and simulate mixed SDEs with both fixed and random effects.
It includes routines for computing drift and diffusion components, simulating multiple trajectories
using the Euler-Maruyama method, and generating random effects from various distributions.
All computations leverage JAX for efficient vectorization and automatic differentiation.

Main Functions:
---------------
- _drift: Compute the drift component of the SDE.
- _diffusion: Compute the diffusion component of the SDE.
- simulate_several_paths: Simulate multiple SDE trajectories.
- _generate_mixed_sde: Generate mixed SDE trajectories with random effects.
- case_exponential, case_weibull, case_gamma, case_lognormal: Generate random samples from specified distributions.
- generate_random_effects: Generate random effects for drift and diffusion terms.

Dependencies:
-------------
- JAX (https://github.com/google/jax)
"""

import jax
from functools import partial
import jax.scipy.stats


@partial(jax.jit, static_argnums=0)
def _drift(drift_func, y, phi, tau, time):
    """
    Compute the drift component of the SDE at a given state and time.

    Parameters
    ----------
    drift_func : callable
        Function representing the drift term, typically drift(y, time).
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

    Notes
    -----
    The drift is computed as: tau * phi @ drift_func(y, time).
    This function is JIT-compiled for efficiency.
    """

    return tau * phi @ drift_func(y, time)


@partial(jax.jit, static_argnums=0)
def _diffusion(diffusion_func, diffusion_fixed_effect, y, tau, time):
    """
    Compute the diffusion component of the SDE at a given state and time.

    Parameters
    ----------
    diffusion_func : callable
        Function representing the diffusion term.
    diffusion_fixed_effect : float
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

    Notes
    -----
    The diffusion is computed as: sqrt(tau) * diffusion_func(y, diffusion_fixed_effect, time).
    This function is JIT-compiled for efficiency.
    """
    return jax.numpy.sqrt(tau) * diffusion_func(y, diffusion_fixed_effect, time)


@partial(jax.jit, static_argnums=(0, 1))
def simulate_several_paths(
    drift_func, diffusion_func, diffusion_fixed_effect, phis, taus, keys, y0, tim_points
):
    """
    Simulate multiple trajectories of the mixed SDE using the Euler-Maruyama method.

    Parameters
    ----------
    drift_func : callable
        Function representing the drift term.
    diffusion_func : callable
        Function representing the diffusion term.
    diffusion_fixed_effect : float
        Fixed effect parameter in the diffusion coefficient.
    phis : jax.numpy.ndarray
        Array of random effects for the drift.
    taus : jax.numpy.ndarray
        Array of random effects for the diffusion.
    keys : jax.random.PRNGKey
        Array of random keys for generating noise.
    y0 : float
        Initial value of the process.
    tim_points : jax.numpy.ndarray
        Array of time points for the simulation.

    Returns
    -------
    jax.numpy.ndarray
        Simulated trajectories (shape: [nb_trajectories, num_time_points]).
    jax.numpy.ndarray
        Time matrix repeated for each trajectory.

    Notes
    -----
    Each trajectory is simulated independently with its own random effects and noise.
    """

    nb_trajectories = phis.shape[0]
    time_mat = jax.numpy.tile(tim_points, (nb_trajectories, 1))
    deltas = jax.numpy.diff(tim_points)

    def simulate_one_path(
        drift_func,
        diffusion_func,
        diffusion_fixed_effect,
        phi,
        tau,
        y0,
        t0,
        deltas,
        key,
    ):
        state_init = (y0, t0, key)

        def euler_iter(
            drift_func, diffusion_func, diffusion_fixed_effect, state, delta, phi, tau
        ):
            y, t, key = state
            key, subkey = jax.random.split(key)
            dw = jax.random.normal(subkey)
            dy = _drift(drift_func, y, phi, tau, t) * delta + _diffusion(
                diffusion_func, diffusion_fixed_effect, y, tau, t
            ) * dw * jax.numpy.sqrt(delta)
            new_y = y + dy
            new_state = (new_y, t + delta, subkey)
            return new_state, y

        def euler_step(state, delta):
            return euler_iter(
                drift_func,
                diffusion_func,
                diffusion_fixed_effect,
                state,
                delta,
                phi,
                tau,
            )

        _, y = jax.lax.scan(euler_step, state_init, deltas)

        return y

    def partial_simulate_one_path(phi, tau, key):
        return simulate_one_path(
            drift_func=drift_func,
            diffusion_func=diffusion_func,
            diffusion_fixed_effect=diffusion_fixed_effect,
            phi=phi,
            tau=tau,
            y0=y0,
            t0=tim_points[0],
            deltas=deltas,
            key=key,
        )

    y = jax.vmap(partial_simulate_one_path)(phis, taus, keys)
    return y, time_mat


def _generate_mixed_sde(
    drift_func,
    diffusion_func,
    diffusion_fixed_effect,
    diffusion_re_dist,
    diffusion_re_param,
    drift_re_param,
    nb_trajectories,
    y0,
    t0,
    t_max,
    h_euler,
    key,
):
    """
    Generate simulated trajectories of the mixed SDE model.

    Parameters
    ----------
    drift_func : callable
        Function representing the drift term.
    diffusion_func : callable
        Function representing the diffusion term.
    diffusion_fixed_effect : float
        Fixed effect parameter in the diffusion coefficient.
    diffusion_re_dist : str
        Distribution type for random effects in diffusion ('exponential', 'weibull', 'gamma', 'lognormal').
    diffusion_re_param : dict
        Parameters for the random effect distribution in diffusion.
    drift_re_param : dict
        Parameters for the random effect distribution in drift.
    nb_trajectories : int
        Number of trajectories to simulate.
    y0 : float
        Initial value of the process.
    t0 : float
        Start time for simulation.
    t_max : float
        End time for simulation.
    h_euler : float
        Time step size for Euler-Maruyama discretization.
    key : jax.random.PRNGKey
        Random key for stochastic sampling.

    Returns
    -------
    jax.numpy.ndarray
        Simulated trajectories (shape: [nb_trajectories, num_time_points]).
    jax.numpy.ndarray
        Random effects tau for diffusion.
    jax.numpy.ndarray
        Random effects phi for drift.
    jax.numpy.ndarray
        Time matrix repeated for each trajectory.

    Notes
    -----
    Random effects are generated for each trajectory and used in simulation.
    """

    subkeys = jax.random.split(key, nb_trajectories)
    tim_points = jax.numpy.arange(t0, t_max, h_euler)
    tau, phi = generate_random_effects(
        n=nb_trajectories,
        distribution=diffusion_re_dist,
        key=key,
        theta_tau=diffusion_re_param,
        theta_phi=drift_re_param,
    )

    y, time_mat = simulate_several_paths(
        drift_func,
        diffusion_func,
        diffusion_fixed_effect,
        phi,
        tau,
        subkeys,
        y0,
        tim_points,
    )

    return y.squeeze(), tau, phi, time_mat


def case_exponential(key, n, param):
    """
    Generate samples from an exponential distribution.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for generating samples.
    n : int
        Number of samples to generate.
    param : dict
        Dictionary with key 'lambda' (rate parameter, positive float).

    Returns
    -------
    jax.numpy.DeviceArray
        Array of shape (n,) with exponential samples.

    Raises
    ------
    ValueError
        If n is not positive or param is missing 'lambda'.
    """

    if not isinstance(n, int) or n <= 0:
        raise ValueError("n should be a positive integer value")

    if not isinstance(param, dict) or not "lambda" in param:
        raise ValueError("param should be a dictionary with key lambda.")
    else:
        scale = 1.0 / param["lambda"]
    return jax.random.exponential(shape=(n,), key=key) * scale


def case_weibull(key, n, param):
    """
    Generate samples from a Weibull distribution.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for generating samples.
    n : int
        Number of samples to generate.
    param : dict
        Dictionary with keys 'c' (concentration) and 'scale' (scale parameter).

    Returns
    -------
    jax.numpy.ndarray
        Array of shape (n,) with Weibull samples.

    Raises
    ------
    ValueError
        If n is not positive or param is missing required keys.
    """

    if not isinstance(n, int) or n <= 0:
        raise ValueError("n should be a positive integer value")

    if not isinstance(param, dict) or not set(["c", "scale"]).issubset(param.keys()):
        raise ValueError("param should be a dictionary with keys c and scale.")
    else:
        c = param["c"]
        scale = param["scale"]

    return jax.random.weibull_min(key=key, shape=(n,), concentration=c, scale=scale)


def case_gamma(key, n, param):
    """
    Generate samples from a gamma distribution.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for generating samples.
    n : int
        Number of samples to generate.
    param : dict
        Dictionary with keys 'shape' and 'scale'.

    Returns
    -------
    jax.numpy.DeviceArray
        Array of shape (n,) with gamma samples.

    Raises
    ------
    ValueError
        If n is not positive or param is missing required keys.
    """

    if not isinstance(n, int) or n <= 0:
        raise ValueError("n should be a positive integer value")

    if not isinstance(param, dict) or not set(["shape", "scale"]).issubset(
        param.keys()
    ):
        raise ValueError(
            "param should be a dictionary with keys shape and scale.")
    else:
        shape = param["shape"]
        scale = param["scale"]

    return jax.random.gamma(key=key, shape=(n,), a=shape) * scale


def case_lognormal(key, n, param):
    """
    Generate samples from a log-normal distribution.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for generating samples.
    n : int
        Number of samples to generate.
    param : dict
        Dictionary with keys 'mean' and 'sigma' for the underlying normal distribution.

    Returns
    -------
    jax.numpy.ndarray
        Array of shape (n,) with log-normal samples.

    Raises
    ------
    ValueError
        If n is not positive or param is missing required keys.
    """

    if not isinstance(n, int) or n <= 0:
        raise ValueError("n should be a positive integer value")

    if not isinstance(param, dict) or not set(["mean", "sigma"]).issubset(param.keys()):
        raise ValueError("param should be a dictionary with keys mu and sigma.")
    else:
        mean = param["mean"]
        sigma = param["sigma"]
        normal_samples = jax.random.normal(key, shape=(n,))

    return jax.numpy.exp(mean + sigma * normal_samples)


def generate_random_effects(n, distribution, key, theta_tau, theta_phi):
    """
    Generate random effects for the mixed SDE model.

    Parameters
    ----------
    n : int
        Number of random effects to generate (number of trajectories).
    distribution : str
        Distribution for random effects in diffusion ('exponential', 'weibull', 'gamma', 'lognormal').
    key : jax.random.PRNGKey
        Random key for generating samples.
    theta_tau : dict
        Parameters for the diffusion random effect distribution.
    theta_phi : dict
        Parameters for the drift random effect distribution (keys: 'mu', 'omega2').

    Returns
    -------
    jax.numpy.ndarray
        Array of random effects tau for diffusion.
    jax.numpy.ndarray
        Array of random effects phi for drift.

    Raises
    ------
    ValueError
        If input parameters are invalid or required keys are missing.

    Notes
    -----
    The function supports several distributions for tau and generates phi as Gaussian or multivariate normal.
    """

    if not isinstance(theta_tau, dict):
        raise ValueError("Dictionary is expected for argument theta_tau.")

    if not isinstance(theta_phi, dict):
        raise ValueError("Dictionary is expected for argument theta_phi.")

    if distribution == "exponential":
        tau = case_exponential(key=key, n=n, param=theta_tau)
    elif distribution == "weibull":
        tau = case_weibull(key=key, n=n, param=theta_tau)
    elif distribution == "gamma":
        tau = case_gamma(key=key, n=n, param=theta_tau)
    elif distribution == "lognormal":
        tau = case_lognormal(key=key, n=n, param=theta_tau)
    else:
        raise ValueError(
            "Unknown distribution. Supported distributions are exponential, \
            weibull, gamma and lognormal."
        )

    if set(["mu", "omega2"]).issubset(theta_phi.keys()):
        if theta_phi['omega2'].shape[0] == 1:
            phi = theta_phi['mu'] + jax.numpy.sqrt(theta_phi['omega2']) * jax.random.normal(
                key=key,
                shape=(n,)
            ).reshape(-1, 1)
        elif (
            jax.numpy.linalg.matrix_rank(theta_phi["omega2"])
            < theta_phi["omega2"].shape[0]
        ):
            z = jax.random.normal(key, shape=(n, theta_phi["mu"].shape[0]))
            L = jax.numpy.linalg.cholesky(
                theta_phi["omega2"]
                + 1e-30 * jax.numpy.eye(theta_phi["omega2"].shape[0])
            )
            phi = theta_phi["mu"] + jax.numpy.dot(z, L.T)
        else:
            phi = jax.random.multivariate_normal(
                key=key,
                mean=theta_phi["mu"],
                cov=theta_phi["omega2"],
                shape=(n,),
            )
    else:
        raise ValueError(
            "theta_phi should be structured into normal, mu and omega2 keys."
        )

    return tau, phi
