"""
Module for estimating parameters in mixed stochastic differential equations (SDEs) with random effects.`
This module provides optimized JAX implementations of various mathematical functions, including matrix transformations, diffusion coefficient estimations, and parameter estimation using maximum likelihood methods. The functions leverage JAX's jit compilation and vectorization (vmap) for efficient computation.
"""

import jax
from functools import partial
from jax.scipy.optimize import minimize
from jax.scipy.stats import norm, expon, gamma


# Fonctions utilitaires
################################################################################


def increments(y):
    # For one trajectory
    """
    Calculate discrete differences along an array.

    Args:
    -----
    - y (array-like): Input values.

    Returns:
    --------
    ndarray: Discrete differences of the array.
    """

    return jax.numpy.diff(y)


def precompute_indices(covariance_to_estimate):
    """
    This function computes the indices of the elements to be estimated in a covariance matrix. It identifies the upper triangular part of the matrix and returns the indices of the non-zero elements.

    Arguments:
    ----------
    covariance_to_estimate (array-like): A square matrix for which the indices of the upper triangular elements will be computed.

    Returns:
    --------
    array-like: Indices of the elements in the upper triangular part of the matrix that are to be estimated.
    """
    return jax.numpy.where(jax.numpy.triu(covariance_to_estimate).ravel())[0]


def inverse_svd(matrix):
    u, s, vh = jax.numpy.linalg.svd(matrix)
    inv_matrix = jax.numpy.dot(vh.T, jax.numpy.dot(jax.numpy.diag(1 / s), u.T))
    return inv_matrix


@partial(jax.jit, static_argnums=0)
def c_vectorized(diffusion_func, diffusion_fixed_effect, y, time):
    """
    Compute the vectorized diffusion function.

    Args:
    -----
    - diffusion_func (callable): Diffusion function.
    - diffusion_fixed_effect (float or array-like): Fixed effect parameter.
    - y (array-like): Input values.
    -  time (array-like): Corresponding time points.

    Returns:
    --------
    ndarray: Vectorized diffusion function output.
    """

    n = len(y)
    cy = jax.vmap(
        lambda x: diffusion_func(x[0], diffusion_fixed_effect, x[1]), in_axes=0
    )(jax.numpy.stack((y[: n - 1], time[: n - 1]), axis=-1))
    return cy


@partial(jax.jit, static_argnums=0)
def c_matrix(diffusion_func, diffusion_fixed_effect, y, time_mat):
    """
    Compute diffusion function values for multiple trajectories.

    Args:
    -----
    - diffusion_func (callable): Diffusion function.
    - diffusion_fixed_effect (float or array-like): Fixed effect parameter.
    - y (array-like): Observed data matrix.
    - time (array-like): Time points.

    Returns:
    --------
    ndarray: Diffusion function matrix.
    """
    nb_obs_per_trajectory = y.shape[1]
    cy = jax.vmap(diffusion_func, in_axes=(0, None, 0))(
        y[:, 0: nb_obs_per_trajectory - 1],
        diffusion_fixed_effect,
        time_mat[:, 0: nb_obs_per_trajectory - 1],
    )
    return cy


def estim_diffusion_coef(diffusion_func, y, eta, time):
    """
    Estimate the diffusion coefficient for a given path.
    This function computes an estimate of the diffusion coefficient based on the provided path, a fixed parameter `eta`, and the time points.

    Args:
    ----
        diffusion_func (callable): Diffusion function.
        y (array-like): The observed time series data.
        eta (float or array-like): A parameter used in the computation of the diffusion coefficient.
        time (array-like): The time points corresponding to the observations in `y`.
    Returns:
    --------
        float: The estimated diffusion coefficient.
    Notes:
    ------
        - The function uses JAX for numerical computations, which allows for efficient differentiation and parallelization.
        - The computation involves calculating increments of the time series, squaring them, and normalizing by a function of `eta` and `time`.
    """

    h = jax.numpy.diff(time)[0,]

    value = (
        jax.numpy.sum(
            jax.lax.mul(
                jax.numpy.squeeze(jax.lax.square(increments(y=y))),
                jax.numpy.squeeze(
                    jax.lax.reciprocal(
                        jax.lax.square(
                            c_vectorized(
                                y=y,
                                diffusion_fixed_effect=eta,
                                time=time,
                                diffusion_func=diffusion_func,
                            )
                        )
                    )
                ),
            )
        )
        * 1
        / h
    )

    return value


@partial(jax.jit, static_argnums=0)
def estim_tau(diffusion_func, y, eta, time):
    """
    Compute an estimation of the random effect in the diffusion coefficient when fixed-effects values are known. This function calculates the random effects in the diffusion coefficient
    based on the observations of a single path, a fixed-effect value, and the corresponding time points.

    Args:
    -----
        diffusion_func (callable): Diffusion function.
        y (array-like): A 1D array containing the observations of one path.
        eta (float): The fixed-effect value in the diffusion coefficient.
        time (array-like): A vector of time points corresponding to the observations.

    Returns:
    --------
        float: The estimated value of the random effect in the diffusion coefficient.
    """

    nb_obs_per_trajectory = len(y) - 1
    value = (
        1
        / nb_obs_per_trajectory
        * estim_diffusion_coef(y=y, eta=eta, time=time, diffusion_func=diffusion_func)
    )

    return value


@partial(jax.jit, static_argnums=0)
def _estim_tau_vectorized(diffusion_func, y, eta, time_mat):
    """
    Vectorized estimation of tau values for a given dataset.
    This function applies the `estim_tau` function to each path of the input array `y` in a vectorized manner using JAX's `vmap`. The results are then reshaped into a column vector.

    Args:
    -----
    diffusion_func (callable): Diffusion function.
    y : array-like
        Input data array where each row corresponds to an individual data point.
    eta : float or array-like
        Parameter(s) required by the `estim_tau` function.
    time : array-like
        Time points corresponding to the data in `y`.
    Returns:
    --------
    numpy.ndarray
        A column vector containing the estimated tau values for each row in `y`.
    """
    value = jax.vmap(estim_tau, in_axes=(None, 0, None, 0))(
        diffusion_func, y, eta, time_mat)
    return value.reshape(-1, 1)


@partial(jax.jit, static_argnums=0)
def eta_estim_function(diffusion_func, eta, y, time_mat):
    """
    Computes the estimation function of eta based on the provided parameters.
    This function uses the mean of the logarithm
    of two components: the result of `estim_tau_vectorized` and the square of
    the result of `c_matrix`. Both components are evaluated with the given inputs `y`, `eta`, and `time`.
    Args:
    -----
        diffusion_func (callable): Diffusion function.
        eta (float or array-like): The current fixed-effect parameter eta.
        y (array-like): Observed data.
        time (array-like): Time points corresponding to the observed data.
    Returns:
    --------
        float: The computed estimation function value for the given value of eta.
    """

    value = 1 / 2 * jax.numpy.mean(
        jax.lax.log(
            _estim_tau_vectorized(
                y=y, eta=eta, time_mat=time_mat, diffusion_func=diffusion_func
            )
        )
    ) + 1 / 2 * jax.numpy.mean(
        jax.lax.log(
            jax.lax.square(
                c_matrix(
                    y=y,
                    diffusion_fixed_effect=eta,
                    time_mat=time_mat,
                    diffusion_func=diffusion_func,
                )
            )
        )
    )
    return value


@partial(jax.jit, static_argnums=0)
def estim_eta(diffusion_func, y, time_mat, init):
    """
    Estimate the parameter eta using optimization.

    This function estimates the value of eta by minimizing a given objective
    function `eta_estim_function` using the BFGS optimization method.

    Args:
        diffusion_func (callable): Diffusion function.
        y (array-like): Observed data.
        time (array-like): Time points corresponding to the observed data.
        init (float or array-like): Initial guess for the eta parameter.

    Returns:
        float or array-like: The estimated value of eta that minimizes the objective function.
    """

    assert init.shape[0] == 1, "The initial value of eta must be a scalar."

    def partial_eta_est(eta):
        return eta_estim_function(diffusion_func, eta, y, time_mat)

    res = minimize(partial_eta_est, init, method="BFGS")

    return res.x


def estim_theta_tau(distribution, tau, init):
    """
    Estimate the parameters of a given distribution using maximum likelihood estimation.

    Parameters:
    -----------
    distribution : str
        The name of the distribution to estimate. Supported values are:
        - "exponential"
        - "gamma"
        - "weibull"
        - "lognormal"
    tau : array-like
        The data samples used for parameter estimation.
    init : list or array-like
        Initial guesses for the parameters of the distribution. The number of
        initial values must match the number of parameters for the specified distribution:
        - "exponential": 1 parameter (rate)
        - "gamma": 2 parameters (shape, scale)
        - "weibull": 2 parameters (shape, scale)
        - "lognormal": 2 parameters (mean, standard deviation)

    Returns:
    --------
    numpy.ndarray
        The estimated parameters of the specified distribution.

    Raises:
    -------
    ValueError
        If the `distribution` is not supported or if the number of initial values
        in `init` does not match the expected number of parameters for the specified distribution.

    Notes:
    ------
    - The function uses the `jax` library for vectorized computations and the
      `scipy.optimize.minimize` function with the "BFGS" method for optimization.
    - The Weibull distribution is implemented manually as it is not available in
      `jax.scipy.stats`.
    """

    if distribution == "exponential":
        if not len(init) == 1:
            raise ValueError(
                "init does contain the correct number of initial values.")
        else:

            def logpdf_exponential(lambda_):
                value = -jax.numpy.mean(
                    jax.vmap(lambda x: expon.logpdf(x, scale=1 / lambda_))(tau)
                )
                return value

            res = minimize(logpdf_exponential, init, method="BFGS")

    elif distribution == "gamma":
        if not len(init) == 2:
            raise ValueError(
                "init does contain the correct number of initial values.")
        else:

            def logpdf_gamma(param):
                shape, scale = param
                value = -jax.numpy.mean(
                    jax.vmap(lambda x: gamma.logpdf(
                        x, a=shape, scale=scale))(tau)
                )
                return value

            res = minimize(logpdf_gamma, init, method="BFGS")

    elif distribution == "weibull":
        # The Weibull distribution is not implemented in jax.scipy.stats
        if not len(init) == 2:
            raise ValueError(
                "init does contain the correct number of initial values.")
        else:

            def logpdf_weibull_ind(x, k, lam):
                return (
                    (k - 1) * jax.numpy.log(x / lam)
                    - (x / lam) ** k
                    + jax.numpy.log(k / lam)
                )

            def logpdf_weibull(param):
                shape, scale = param
                value = -jax.numpy.mean(
                    jax.vmap(lambda x: logpdf_weibull_ind(
                        x, lam=scale, k=shape))(tau)
                )
                return value

            res = minimize(logpdf_weibull, init, method="BFGS")

    elif distribution == "weibull_loc":
        if not len(init) == 3:
            raise ValueError(
                "init does contain the correct number of initial values.")
        else:

            def logpdf_weibull_loc_ind(x, k, lam, loc):
                return (
                    (k - 1) * jax.numpy.log((x - loc) / lam)
                    - ((x - loc) / lam) ** k
                    + jax.numpy.log(k / lam)
                )

            def logpdf_weibull_loc(param):
                shape, scale, loc = param
                value = -jax.numpy.mean(
                    jax.vmap(lambda x: logpdf_weibull_loc_ind(
                        x, lam=scale, k=shape, loc=loc))(tau)
                )
                return value

            res = minimize(logpdf_weibull_loc, init, method="BFGS")

    elif distribution == "lognormal":
        if not len(init) == 2:
            raise ValueError(
                "init does contain the correct number of initial values.")
        else:

            def logpdf_lognormal(param):
                mu, sigma = param
                value = -jax.numpy.mean(
                    jax.vmap(lambda x: norm.logpdf(x, loc=mu, scale=sigma))(
                        jax.numpy.log(tau)
                    )
                )
                return value

        res = minimize(logpdf_lognormal, init, method="BFGS")
    else:
        raise ValueError("Unknown distribution.")

    return res.x


@partial(jax.jit, static_argnums=(2, 4))
def _estim_diffusion_param(y, time_mat, tau_distribution, init, diffusion_func):
    """
    Estimate the diffusion parameters of the mixed sde model.
    This function estimates the parameters of a diffusion process by sequentially
    estimating the eta parameter, the tau vector, and the theta_tau parameter.
    Args:
    -----
        y (array-like): Observed data points.
        time (array-like): Time points corresponding to the observed data.
        tau_distribution (str): The distribution type for the tau parameter.
        init (dict): A dictionary containing initial values for the parameters:
            - 'eta': Initial value for the eta parameter.
            - 'theta_tau': Initial value for the theta_tau parameter.
        diffusion_func (callable): Diffusion function.
    Returns:
    --------
        tuple: A tuple containing:
            - eta_hat (float): Estimated value of the eta parameter.
            - theta_tau_hat (any): Estimated value of the theta_tau parameter.
            - tau_hat (array-like): Estimated tau vector.
    """
    init_eta = init["eta"]
    init_theta_tau = jax.numpy.array(list(init["theta_tau"].values()))
    eta_hat = estim_eta(y=y, time_mat=time_mat, init=init_eta,
                        diffusion_func=diffusion_func)
    tau_hat = _estim_tau_vectorized(
        y=y, eta=eta_hat, time_mat=time_mat, diffusion_func=diffusion_func
    )
    theta_tau_hat = estim_theta_tau(
        distribution=tau_distribution, tau=tau_hat, init=init_theta_tau
    )
    return (eta_hat, theta_tau_hat, tau_hat)


###############################################################################
# Partie drift
###############################################################################

@partial(jax.jit, static_argnums=0)
def _a_matrix(drift_func, y, time):
    """
    Compute the drift function output reshaped as a column matrix.

    Args:
    -----
    - drift_func (callable): Function representing the drift.
    - y (array-like): Input values.
    - time (array-like): Corresponding time points.

    Returns:
    --------
    ndarray: Reshaped output matrix.
    """
    return drift_func(y, time).reshape(-1, 1)


@partial(jax.jit, static_argnums=0)
def _squared_a_matrix(drift_func, y, time):
    """
    Compute the squared transformation of the drift function matrix.

    Args:
    -----
    - drift_func (callable): Function representing the drift.
    - y (array-like): Input values.
    - time (array-like): Corresponding time points.

    Returns:
    --------
    ndarray: Squared matrix transformation.
    """

    nb_obs_per_trajectory = len(y)

    a2 = jax.numpy.squeeze(
        jax.vmap(
            lambda x: _a_matrix(drift_func, x[0], x[1])
            @ _a_matrix(drift_func, x[0], x[1]).T
        )(jax.numpy.stack((y[: nb_obs_per_trajectory - 1], time[: nb_obs_per_trajectory - 1]), axis=-1))
    )
    return a2


###########
# Useful statistics
###########

@partial(jax.jit, static_argnums=(0, 1, 2))
def mi_inv(diffusion_func, drift_func, nb_re_drift, y, time, eta, tau):

    h = jax.numpy.diff(time)[0,]

    nb_obs_per_trajectory = len(y)

    invcy2 = jax.lax.reciprocal(
        jax.lax.square(
            c_vectorized(
                y=y,
                diffusion_fixed_effect=eta,
                time=time,
                diffusion_func=diffusion_func,
            )
        )
    ).reshape(-1, 1)

    value = jax.numpy.linalg.inv(
        h
        * tau
        * jax.numpy.sum(
            _squared_a_matrix(drift_func=drift_func, y=y, time=time).reshape(
                nb_obs_per_trajectory-1, nb_re_drift, nb_re_drift)
            * jax.numpy.broadcast_to(invcy2[:, None],
                                     (invcy2.shape[0], nb_re_drift, nb_re_drift)),
            axis=0,
        )
    )

    return value


@partial(jax.jit, static_argnums=(0, 1, 2))
def vi(diffusion_func, drift_func, nb_re_drift, y, time, eta):

    nb_obs_per_trajectory = len(y)

    invcy2 = jax.lax.reciprocal(
        jax.lax.square(
            c_vectorized(
                y=y,
                diffusion_fixed_effect=eta,
                time=time,
                diffusion_func=diffusion_func,
            )
        )
    ).reshape(-1, 1)

    ay = jax.numpy.squeeze(
        jax.vmap(lambda x: _a_matrix(drift_func, x[0], x[1]))(
            jax.numpy.stack(
                (y[: nb_obs_per_trajectory - 1],
                 time[: nb_obs_per_trajectory - 1]),
                axis=-1,
            )
        )
    ).reshape(nb_obs_per_trajectory-1, nb_re_drift)

    value = jax.numpy.sum(
        jax.lax.mul(ay * invcy2, increments(y=y).reshape(-1, 1)),
        axis=0
    ).reshape(nb_re_drift, 1)

    return value


@partial(jax.jit, static_argnums=(0, 1, 2))
def theta_phi_est_function(diffusion_func, drift_func, nb_re_drift, y, time, eta, tau, mu, omega2):
    """
    Computes the log-likelihood value for a given set of parameters in the context of mixed stochastic differential equations (SDEs).

    Parameters:
    -----------
    diffusion_func (callable): Diffusion function.
    y : array-like
        Observed data points for the trajectories.
    time : array-like
        Time points corresponding to the observations in `y`.
    eta : array-like
        Parameter vector used in the computation of the `c_vectorized` function.
    tau : float
        Scaling factor for the covariance matrix.
    mu : array-like
        Mean vector for the prior distribution.
    omega2 : array-like
        Covariance matrix for the prior distribution.

    Returns:
    --------
    value : float
        The computed log-likelihood value based on the input parameters.
    """

    mi_inv_ind = mi_inv(diffusion_func, drift_func,
                        nb_re_drift, y, time, eta, tau)
    vi_ind = vi(diffusion_func, drift_func, nb_re_drift, y, time, eta)

    value = -1 / 2 * (mu.reshape(nb_re_drift, 1) - mi_inv_ind @ vi_ind).T @ jax.numpy.linalg.inv(omega2 + mi_inv_ind) @ (
        mu.reshape(nb_re_drift, 1) - mi_inv_ind @ vi_ind
    ) - 1 / 2 * jax.numpy.log(jax.numpy.linalg.det(omega2 + mi_inv_ind))

    return value


@partial(jax.jit, static_argnums=(0, 1, 2))
def mu_init_stepwise(diffusion_func, drift_func, nb_re_drift, y, time_mat, eta, tau):

    identity_matrix = jax.numpy.identity(nb_re_drift)

    mi_inv_vect = jax.vmap(mi_inv, in_axes=(None, None, None, 0, 0, None, 0)
                           )(diffusion_func, drift_func, nb_re_drift, y, time_mat, eta, tau)

    vi_vect = jax.vmap(vi, in_axes=(None, None, None, 0, 0, None)
                       )(diffusion_func, drift_func, nb_re_drift, y, time_mat, eta)

    mat = jax.vmap(inverse_svd)(mi_inv_vect + identity_matrix)

    mat2 = jax.vmap(jax.numpy.matmul, in_axes=(0, 0))(mat, mi_inv_vect)

    sum_mat = jax.numpy.sum(mat, axis=0)
    sum_mat2 = jax.numpy.sum(
        jax.vmap(jax.numpy.matmul, in_axes=(0, 0))(mat2, vi_vect), axis=0)

    value = inverse_svd(sum_mat) @ sum_mat2

    return value


@partial(jax.jit, static_argnums=(0, 1, 2, 9))
def _estim_drift_param(
    diffusion_func,
    drift_func,
    nb_re_drift,
    y,
    time_mat,
    eta,
    tau,
    init,
    covariance_to_estimate_indices,
    method

):
    """
    This function estimates the drift and diffusion parameters (theta_phi) of a model based on a given dataset. It uses a minimization approach to optimize the parameters by minimizing the negative log-likelihood.

    Arguments:
    ----------
    diffusion_func (callable): The function representing the diffusion process in the model.

    drift_func (callable): The function representing the drift process in the model.

    y (array-like): The observed data points.

    time_mat (array-like): The matrix of time steps associated with the data.

    eta (array-like): Parameters related to the model's stochastic process.

    tau (array-like): Time discretization or other time-related parameters.

    init (dict): A dictionary containing the initial guesses for parameters, such as "mu" (mean) and "omega2" (covariance).

    covariance_to_estimate_indices (array-like): Indices of the elements in the covariance matrix to be estimated.

    method ...

    Returns:
    --------

    A tuple (mu_hat, omega2_hat) representing the estimated mean vector (mu_hat) and covariance matrix (omega2_hat).
    """
    assert init["mu"].shape[0] == nb_re_drift, "The initial mean vector must have the same length as nb_re_drift."
    assert init['mu'].shape[0] == drift_func(
        0, 0).shape[0], "The initial mean vector must have the same length as the drift function output."
    assert init["omega2"].shape[0] == init["omega2"].shape[1], "The initial covariance matrix must be square."
    assert init["omega2"].shape[1] == nb_re_drift, "The initial covariance matrix must be nb_re_drift x nb_re_drift."

    omega2_vect = from_covariance_to_vector(
        init["omega2"], covariance_to_estimate_indices
    )

    def estimating_theta_phi(theta):
        mu = theta[:nb_re_drift]
        omega2 = from_vector_to_covariance(
            theta[nb_re_drift:], nb_re_drift, covariance_to_estimate_indices
        )

        terms = jax.vmap(
            theta_phi_est_function, in_axes=(
                None, None, None, 0, 0, None, 0, None, None)
        )(diffusion_func, drift_func, nb_re_drift, y, time_mat, eta, tau, mu, omega2)
        return -jax.numpy.mean(terms)

    if method == 'joint':  # Joint estimation of mu and omega2

        vectorized_init = jax.numpy.concatenate([init["mu"], omega2_vect])

        res = minimize(estimating_theta_phi, vectorized_init, method="BFGS")

        mu_hat = res.x[0:nb_re_drift]
        omega2_hat = from_vector_to_covariance(jax.numpy.squeeze(
            res.x[nb_re_drift:]), nb_re_drift, covariance_to_estimate_indices)

    elif method == 'stepwise':  # Stepwise estimation of mu and omega2
        mu0 = mu_init_stepwise(diffusion_func, drift_func,
                               nb_re_drift, y, time_mat, eta, tau)

        def estimating_omega2(theta):
            omega2 = from_vector_to_covariance(
                theta, nb_re_drift, covariance_to_estimate_indices
            )
            terms = jax.vmap(
                theta_phi_est_function, in_axes=(
                    None, None, None, 0, 0, None, 0, None, None)
            )(diffusion_func, drift_func, nb_re_drift, y, time_mat, eta, tau, mu0, omega2)
            return -jax.numpy.mean(terms)

        res = minimize(estimating_omega2, omega2_vect, method="BFGS").x

        mu_hat = mu0
        if nb_re_drift == 1:
            mu_hat = mu0

            vectorized_theta0 = jax.numpy.concatenate([mu0, res.reshape(1, 1)])

            grad_estim_theta_phi = jax.grad(estimating_theta_phi)(
                jax.numpy.squeeze(vectorized_theta0)).reshape(len(vectorized_theta0), 1)
            hessian_estim_theta_phi = hessian(estimating_theta_phi)(
                jax.numpy.squeeze(vectorized_theta0))

            corrected_theta = vectorized_theta0 - \
                inverse_svd(hessian_estim_theta_phi) @ grad_estim_theta_phi

            mu_hat = corrected_theta[:nb_re_drift]

            omega2_hat = from_vector_to_covariance(jax.numpy.squeeze(
                corrected_theta[nb_re_drift:]), nb_re_drift, covariance_to_estimate_indices)

        else:
            vectorized_theta0 = jax.numpy.concatenate(
                [jax.numpy.squeeze(mu0), res])

            grad_estim_theta_phi = jax.grad(estimating_theta_phi)(
                vectorized_theta0).reshape(len(vectorized_theta0), 1)
            hessian_estim_theta_phi = hessian(
                estimating_theta_phi)(vectorized_theta0)

            corrected_theta = vectorized_theta0.reshape(len(
                vectorized_theta0), 1) - inverse_svd(hessian_estim_theta_phi) @ grad_estim_theta_phi

            mu_hat = corrected_theta[:nb_re_drift]

            omega2_hat = from_vector_to_covariance(jax.numpy.squeeze(
                corrected_theta[nb_re_drift:]), nb_re_drift, covariance_to_estimate_indices)

    return (mu_hat, omega2_hat)


# Automatic differentiation to compute gradient and hessian
def hessian(f):
    return jax.jacfwd(jax.grad(f))


def from_vector_to_covariance(theta, mu_size, covariance_to_estimate_indices):
    """
    Reconstruct a symmetric covariance matrix from a parameter vector.

    Args:
    -----
        theta (array-like): A 1D vector containing the covariance parameters.
        mu_size (int): The size of the covariance matrix (mu_size, mu_size).
        covariance_to_estimate_indices (array-like): Indices of the elements to be updated.

    Returns:
    --------
        array-like: A symmetric covariance matrix of size (mu_size, mu_size).
    """
    omega2_p = jax.numpy.zeros((mu_size * mu_size,))
    omega2_p = omega2_p.at[covariance_to_estimate_indices].set(theta)
    omega2_p = omega2_p.reshape(mu_size, mu_size)
    omega2 = omega2_p + omega2_p.T - jax.numpy.diag(jax.numpy.diag(omega2_p))
    return omega2


@jax.jit
def from_covariance_to_vector(omega2, selected_indices):
    """
    This function extracts a vector of elements from the upper triangular part of a covariance matrix, based on pre-calculated indices.

    Arguments:
    ----------
    omega2 (ndarray): The covariance matrix from which elements will be extracted.

    selected_indices (ndarray): Indices of the elements to extract from the upper triangular part of the matrix.

    Returns:
    --------
    ndarray: A 1D vector of the selected elements from the covariance matrix.
    """
    upper_triangle_indices = jax.numpy.triu_indices_from(omega2)
    upper_triangle_elements = omega2[upper_triangle_indices]
    return upper_triangle_elements[selected_indices]
