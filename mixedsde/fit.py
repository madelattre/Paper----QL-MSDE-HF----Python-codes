"""
Parameter estimation utilities for Mixed Stochastic Differential Equations (SDEs) with random effects.

This module provides optimized JAX implementations for estimating drift and diffusion parameters in mixed SDE models.
It includes functions for matrix transformations, diffusion coefficient estimation, random effect estimation, and maximum likelihood parameter estimation.
All computations leverage JAX's just-in-time (jit) compilation and vectorization (vmap) for efficient and scalable computation.

Main Functions:
---------------
- increments: Compute discrete differences along an array.
- precompute_indices: Identify indices of covariance matrix elements to estimate.
- inverse_svd: Compute the inverse of a matrix using singular value decomposition.
- c_vectorized, c_matrix: Compute vectorized diffusion function outputs.
- estim_diffusion_coef: Estimate the diffusion coefficient for a trajectory.
- estim_tau, _estim_tau_vectorized: Estimate random effects in diffusion.
- eta_estim_function, estim_eta: Estimate fixed effect parameter eta.
- estim_theta_tau: Estimate parameters of the random effect distribution in diffusion.
- _estim_diffusion_param: Sequentially estimate diffusion parameters.
- _a_matrix, _squared_a_matrix: Compute drift function outputs and their squared transformations.
- mi_inv, vi: Compute statistics for drift parameter estimation.
- theta_phi_est_function, mu_init_stepwise, _estim_drift_param: Estimate drift parameters using likelihood-based methods.
- hessian: Compute the Hessian matrix via automatic differentiation.
- from_vector_to_covariance, from_covariance_to_vector: Convert between covariance matrices and parameter vectors.

Dependencies:
-------------
- JAX (https://github.com/google/jax)
"""

import jax
from functools import partial
from jax.scipy.optimize import minimize
from jax.scipy.stats import norm, expon, gamma


def increments(y):
    """
    Compute discrete differences along an array.

    Parameters
    ----------
    y : array-like
        Input values.

    Returns
    -------
    ndarray
        Discrete differences of the array (y[1:] - y[:-1]).
    """

    return jax.numpy.diff(y)


def precompute_indices(covariance_to_estimate):
    """
    Identify indices of non-zero elements in the upper triangular part of a covariance matrix.

    Parameters
    ----------
    covariance_to_estimate : array-like
        Square matrix indicating which elements to estimate.

    Returns
    -------
    array-like
        Indices of elements in the upper triangular part to be estimated.
    """
    return jax.numpy.where(jax.numpy.triu(covariance_to_estimate).ravel())[0]


def inverse_svd(matrix):
    """
    Compute the inverse of a matrix using singular value decomposition (SVD).

    Parameters
    ----------
    matrix : array-like
        Input matrix to invert.

    Returns
    -------
    ndarray
        Inverse of the input matrix.
    """
    u, s, vh = jax.numpy.linalg.svd(matrix)
    inv_matrix = jax.numpy.dot(vh.T, jax.numpy.dot(jax.numpy.diag(1 / s), u.T))
    return inv_matrix


@partial(jax.jit, static_argnums=0)
def c_vectorized(diffusion_func, diffusion_fixed_effect, y, time):
    """
    Compute the vectorized output of the diffusion function for a single trajectory.

    Parameters
    ----------
    diffusion_func : callable
        Diffusion function.
    diffusion_fixed_effect : float or array-like
        Fixed effect parameter.
    y : array-like
        Observed values.
    time : array-like
        Time points.

    Returns
    -------
    ndarray
        Vectorized diffusion function output for each time step.
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

    Parameters
    ----------
    diffusion_func : callable
        Diffusion function.
    diffusion_fixed_effect : float or array-like
        Fixed effect parameter.
    y : array-like
        Observed data matrix (trajectories).
    time_mat : array-like
        Time points matrix.

    Returns
    -------
    ndarray
        Diffusion function matrix for all trajectories and time steps.
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
    Estimate the diffusion coefficient for a single trajectory.

    Parameters
    ----------
    diffusion_func : callable
        Diffusion function.
    y : array-like
        Observed time series data.
    eta : float or array-like
        Fixed effect parameter for diffusion.
    time : array-like
        Time points.

    Returns
    -------
    float
        Estimated diffusion coefficient for the trajectory.

    Notes
    -----
    Uses increments and normalizes by the squared diffusion function.
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
    Estimate the random effect (tau) in the diffusion coefficient for a single trajectory.

    Parameters
    ----------
    diffusion_func : callable
        Diffusion function.
    y : array-like
        Observed data for one trajectory.
    eta : float
        Fixed effect parameter for diffusion.
    time : array-like
        Time points.

    Returns
    -------
    float
        Estimated value of tau for the trajectory.
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
    Vectorized estimation of tau values for multiple trajectories.

    Parameters
    ----------
    diffusion_func : callable
        Diffusion function.
    y : array-like
        Data matrix (trajectories).
    eta : float or array-like
        Fixed effect parameter for diffusion.
    time_mat : array-like
        Time points matrix.

    Returns
    -------
    ndarray
        Column vector of estimated tau values for each trajectory.
    """
    value = jax.vmap(estim_tau, in_axes=(None, 0, None, 0))(
        diffusion_func, y, eta, time_mat)
    return value.reshape(-1, 1)


@partial(jax.jit, static_argnums=0)
def eta_estim_function(diffusion_func, eta, y, time_mat):
    """
    Compute the objective function for eta estimation.

    Parameters
    ----------
    diffusion_func : callable
        Diffusion function.
    eta : float or array-like
        Current value of eta.
    y : array-like
        Observed data.
    time_mat : array-like
        Time points.

    Returns
    -------
    float
        Value of the estimation function for eta.
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
    Estimate the fixed effect parameter eta using optimization.

    Parameters
    ----------
    diffusion_func : callable
        Diffusion function.
    y : array-like
        Observed data.
    time_mat : array-like
        Time points.
    init : float or array-like
        Initial guess for eta.

    Returns
    -------
    float or array-like
        Estimated value of eta.
    """

    assert init.shape[0] == 1, "The initial value of eta must be a scalar."

    def partial_eta_est(eta):
        return eta_estim_function(diffusion_func, eta, y, time_mat)

    res = minimize(partial_eta_est, init, method="BFGS")

    return res.x


def estim_theta_tau(distribution, tau, init):
    """
        Estimate parameters of the specified distribution for tau using maximum likelihood.

    Parameters
    ----------
    distribution : str
        Distribution name ('exponential', 'gamma', 'weibull', 'lognormal', 'weibull_loc').
    tau : array-like
        Data samples for estimation.
    init : list or array-like
        Initial guesses for distribution parameters.

    Returns
    -------
    ndarray
        Estimated parameters of the distribution.

    Raises
    ------
    ValueError
        If distribution is unsupported or initial values are invalid.
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
    Sequentially estimate the diffusion parameters of the mixed SDE model.

    Parameters
    ----------
    y : array-like
        Observed data points.
    time_mat : array-like
        Time points.
    tau_distribution : str
        Distribution type for tau.
    init : dict
        Initial values for eta and theta_tau.
    diffusion_func : callable
        Diffusion function.

    Returns
    -------
    tuple
        (eta_hat, theta_tau_hat, tau_hat): Estimated eta, distribution parameters, and tau vector.
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
# Drift part
###############################################################################

@partial(jax.jit, static_argnums=0)
def _a_matrix(drift_func, y, time):
    """
    Compute the drift function output reshaped as a column matrix.

    Parameters
    ----------
    drift_func : callable
        Drift function.
    y : array-like
        Input values.
    time : array-like
        Time points.

    Returns
    -------
    ndarray
        Column matrix of drift function outputs.
    """
    return drift_func(y, time).reshape(-1, 1)


@partial(jax.jit, static_argnums=0)
def _squared_a_matrix(drift_func, y, time):
    """
    Compute the squared transformation of the drift function matrix.

    Parameters
    ----------
    drift_func : callable
        Drift function.
    y : array-like
        Input values.
    time : array-like
        Time points.

    Returns
    -------
    ndarray
        Squared matrix transformation for each time step.
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
    """
    Compute the inverse of mi matrix for drift parameter estimation.

    Parameters
    ----------
    diffusion_func : callable
        Diffusion function.
    drift_func : callable
        Drift function.
    nb_re_drift : int
        Number of random effects in drift.
    y : array-like
        Observed data for one trajectory.
    time : array-like
        Time points.
    eta : array-like
        Fixed effect parameter for diffusion.
    tau : array-like
        Random effect parameter for diffusion.

    Returns
    -------
    ndarray
        Inverse mi matrix for drift estimation.
    """
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
    """
    Compute the vi statistics for drift parameter estimation.

    Parameters
    ----------
    diffusion_func : callable
        Diffusion function.
    drift_func : callable
        Drift function.
    nb_re_drift : int
        Number of random effects in drift.
    y : array-like
        Observed data for one trajectory.
    time : array-like
        Time points.
    eta : array-like
        Fixed effect parameter for diffusion.

    Returns
    -------
    ndarray
        vi statistics vector for drift estimation.
    """
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
    Compute the log-likelihood value for drift parameter estimation in mixed SDEs.

    Parameters
    ----------
    diffusion_func : callable
        Diffusion function.
    drift_func : callable
        Drift function.
    nb_re_drift : int
        Number of random effects in drift.
    y : array-like
        Observed data for one trajectory.
    time : array-like
        Time points.
    eta : array-like
        Fixed effect parameter for diffusion.
    tau : float
        Random effect parameter for diffusion.
    mu : array-like
        Mean vector for drift random effects.
    omega2 : array-like
        Covariance matrix for drift random effects.

    Returns
    -------
    float
        Log-likelihood value for the given parameters.
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
    """
    Compute the initial estimate of the mean vector (mu) for drift random effects using a stepwise approach.

    Parameters
    ----------
    diffusion_func : callable
        Diffusion function.
    drift_func : callable
        Drift function.
    nb_re_drift : int
        Number of random effects in drift.
    y : array-like
        Data matrix (trajectories).
    time_mat : array-like
        Time points matrix.
    eta : array-like
        Fixed effect parameter for diffusion.
    tau : array-like
        Random effect parameter for diffusion.

    Returns
    -------
    ndarray
        Initial estimate of mu for drift random effects.
    """
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
    Estimate drift parameters (mean and covariance) in mixed SDEs using likelihood-based optimization.

    Parameters
    ----------
    diffusion_func : callable
        Diffusion function.
    drift_func : callable
        Drift function.
    nb_re_drift : int
        Number of random effects in drift.
    y : array-like
        Data matrix (trajectories).
    time_mat : array-like
        Time points matrix.
    eta : array-like
        Fixed effect parameter for diffusion.
    tau : array-like
        Random effect parameter for diffusion.
    init : dict
        Initial guesses for 'mu' and 'omega2'.
    covariance_to_estimate_indices : array-like
        Indices of covariance matrix elements to estimate.
    method : str
        Estimation method ('joint' or 'stepwise').

    Returns
    -------
    tuple
        (mu_hat, omega2_hat): Estimated mean vector and covariance matrix for drift random effects.
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


def hessian(f):
    """
    Compute the Hessian matrix of a scalar-valued function using automatic differentiation.

    Parameters
    ----------
    f : callable
        Function for which to compute the Hessian.

    Returns
    -------
    callable
        Function that computes the Hessian matrix at a given point.
    """
    return jax.jacfwd(jax.grad(f))


def from_vector_to_covariance(theta, mu_size, covariance_to_estimate_indices):
    """
    Reconstruct a symmetric covariance matrix from a parameter vector.

    Parameters
    ----------
    theta : array-like
        1D vector containing covariance parameters.
    mu_size : int
        Size of the covariance matrix (mu_size x mu_size).
    covariance_to_estimate_indices : array-like
        Indices of elements to update in the covariance matrix.

    Returns
    -------
    ndarray
        Symmetric covariance matrix of size (mu_size, mu_size).
    """
    omega2_p = jax.numpy.zeros((mu_size * mu_size,))
    omega2_p = omega2_p.at[covariance_to_estimate_indices].set(theta)
    omega2_p = omega2_p.reshape(mu_size, mu_size)
    omega2 = omega2_p + omega2_p.T - jax.numpy.diag(jax.numpy.diag(omega2_p))
    return omega2


@jax.jit
def from_covariance_to_vector(omega2, selected_indices):
    """
    Extract a vector of elements from the upper triangular part of a covariance matrix.

    Parameters
    ----------
    omega2 : ndarray
        Covariance matrix.
    selected_indices : ndarray
        Indices of elements to extract from the upper triangular part.

    Returns
    -------
    ndarray
        1D vector of selected elements from the covariance matrix.
    """
    upper_triangle_indices = jax.numpy.triu_indices_from(omega2)
    upper_triangle_elements = omega2[upper_triangle_indices]
    return upper_triangle_elements[selected_indices]
