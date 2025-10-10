"""
Module for simulating mixed stochastic differential equations (SDEs) with random effects.

This module provides functions to define and simulate mixed SDEs, including the computation of
drift and diffusion components, as well as generating random effects from various distributions.
The functions leverage JAX for efficient computation and automatic differentiation.

Functions:
----------
- _drift(drift_func, y, phi, tau, time)
- _diffusion(diffusion_func, diffusion_fixed_effect, y, tau, time)
- simulate_several_paths(drift_func, diffusion_func, diffusion_fixed_effect, phis, taus, keys, y0, tim_points)
- _generate_mixed_sde(drift_func, diffusion_func, diffusion_fixed_effect, diffusion_re_dist, diffusion_re_param, drift_re_param, nb_trajectories, y0, t0, t_max, h_euler, key)
- case_exponential(key, n, param)
- case_weibull(key, n, param)
- case_gamma(key, n, param)
- case_lognormal(key, n, param)
- generate_random_effects(n, distribution, key, theta_tau, theta_phi)
"""

import jax
from functools import partial
import jax.scipy.stats


@partial(jax.jit, static_argnums=0)
def _drift(drift_func, y, phi, tau, time):
    """
    Computes the drift component of the SDE.

    Args:
    ----
    drift_func : callable
        The function representing the drift component.
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

    return tau * phi @ drift_func(y, time)


@partial(jax.jit, static_argnums=0)
def _diffusion(diffusion_func, diffusion_fixed_effect, y, tau, time):
    """
    Computes the diffusion component of the SDE.

    Args:
    ----
    diffusion_func : callable
        The function representing the diffusion component.
    diffusion_fixed_effect : float
        The fixed effect parameter in the diffusion coefficient.
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
    return jax.numpy.sqrt(tau) * diffusion_func(y, diffusion_fixed_effect, time)


@partial(jax.jit, static_argnums=(0, 1))
def simulate_several_paths(
    drift_func, diffusion_func, diffusion_fixed_effect, phis, taus, keys, y0, tim_points
):
    """
    Simulates several paths of the mixed SDE using the Euler-Maruyama method.

    Args:
    ----
    drift_func : callable
        The function representing the drift component.
    diffusion_func : callable
        The function representing the diffusion component.
    diffusion_fixed_effect : float
        The fixed effect parameter in the diffusion coefficient.
    phis : jax.numpy.ndarray
        Array of random effects for the drift.
    taus : jax.numpy.ndarray
        Array of random effects for the diffusion.
    keys : jax.random.PRNGKey
        Random keys for generating random numbers.
    y0 : float
        The initial value of the process.
    tim_points : jax.numpy.ndarray
        Array of time points for the simulation.

    Returns:
    -------
    jax.numpy.ndarray
        A 2D array containing the simulated trajectories, where each row corresponds to a trajectory
        and each column corresponds to a time point.
    jax.numpy.ndarray
        A 2D array of time points repeated for each trajectory.
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

    Parameters:
    key (jax.random.PRNGKey): The random key used for generating random numbers.
    n (int): The number of samples to generate. Must be a positive integer.
    param (dict): A dictionary containing the parameter "lambda" for the exponential distribution.
                  The dictionary must have the key "lambda" with a positive value.

    Returns:
    jax.numpy.DeviceArray: An array of shape (n,) containing samples from the exponential distribution.

    Raises:
    ValueError: If `n` is not a positive integer.
    ValueError: If `param` is not a dictionary or does not contain the key "lambda".
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
    Generates random samples from a Weibull distribution using JAX.

    Parameters:
    key (jax.random.PRNGKey): The random key used for generating random numbers.
    n (int): The number of samples to generate. Must be a positive integer.
    param (dict): A dictionary containing the parameters of the Weibull distribution:
        - c (float): The concentration parameter of the Weibull distribution.
        - scale (float): The scale parameter of the Weibull distribution.

    Returns:
    jax.numpy.ndarray: An array of shape (n,) containing the generated samples.

    Raises:
    ValueError: If `n` is not a positive integer.
    ValueError: If `param` is not a dictionary with keys 'c' and 'scale'.
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
    Generates samples from a gamma distribution using the given parameters.
    Parameters:
    key (jax.random.PRNGKey): The random key used for generating random numbers.
    n (int): The number of samples to generate. Must be a positive integer.
    param (dict): A dictionary containing the parameters for the gamma distribution.
                  Must include 'shape' and 'scale' keys.
    Returns:
    jax.numpy.DeviceArray: An array of samples from the gamma distribution.
    Raises:
    ValueError: If 'n' is not a positive integer.
    ValueError: If 'param' is not a dictionary with 'shape' and 'scale' keys.
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
    Generate log-normal distributed samples.

    Parameters:
    key (jax.random.PRNGKey): The random key used for generating random numbers.
    n (int): The number of samples to generate. Must be a positive integer.
    param (dict): A dictionary containing the parameters 'mean' and 'sigma' for the log-normal distribution.
                  - 'mean' (float): The mean of the underlying normal distribution.
                  - 'sigma' (float): The standard deviation of the underlying normal distribution.

    Returns:
    jax.numpy.ndarray: An array of log-normal distributed samples.

    Raises:
    ValueError: If 'n' is not a positive integer.
    ValueError: If 'param' is not a dictionary with keys 'mean' and 'sigma'.
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
    Generate the model's random effects, with distribution specified in "distribution"
    for the random effects in the diffusion coefficient and conditionally gaussian
    random effects in the drift

    Args:
    n            : number of random effects to be generated (int)
                   corresponds to the number of trajectories that will form the sample
    distribution : distribution of the random effects in the diffusion coefficient,
                   'exponential', 'weibull', 'gamma' or 'lognormal'
    key          : random generation key
    theta_tau    : parameters of the distribution of the random effects in the
                   diffusion coefficient (dictionary)
    theta_phi    : parameters of the distribution of the random effects in the drift
                   (dictionary)

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
