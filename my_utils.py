import jax
from mixedsde.fit import from_covariance_to_vector, from_vector_to_covariance, precompute_indices, estim_theta_tau
from mixedsde.data_processing import extract_data
from mixedsde.model_class import *


def oneRun(seed,
           nb_trajectories,
           model,
           h_seq,
           t_max_seq,
           h_euler,
           y0,
           t0,
           covariance_to_estimate):

    std_factor = 0.25
    nb_theta_phi = len(
        model.drift_re_param['mu']) + len(precompute_indices(covariance_to_estimate))

    eta_est = jax.numpy.zeros((1, len(h_seq), len(t_max_seq)))
    theta_tau_est = jax.numpy.zeros(
        (len(model.diffusion_re_param), len(h_seq), len(t_max_seq)))
    theta_true_tau_est = jax.numpy.zeros(
        (len(model.diffusion_re_param), len(h_seq), len(t_max_seq)))
    theta_phi_est_joint = jax.numpy.zeros(
        (nb_theta_phi, len(h_seq), len(t_max_seq)))
    theta_phi_est_stepwise = jax.numpy.zeros(
        (nb_theta_phi, len(h_seq), len(t_max_seq)))
    tau_est = jax.numpy.zeros((nb_trajectories, len(h_seq), len(t_max_seq)))

    # Define random generation keys for the entire simulated example
    key = jax.random.PRNGKey(seed)
    subkey = jax.random.split(key, 2)

    # Simulate the data

    y_init, tau, phi, time_mat_init = model.generate_mixed_sde(
        nb_trajectories, jax.numpy.array([y0]), t0, max(t_max_seq), h_euler, key)

    # Generate intial values for parameter estimation
    epsilon = jax.random.normal(subkey[0,], shape=(
        1+len(model.diffusion_re_param) + nb_theta_phi,))

    if model.diffusion_re_dist == 'exponential':
        init = {
            'diffusion': {
                'eta': model.diffusion_fixed_effect + std_factor * jax.lax.mul(jax.numpy.abs(model.diffusion_fixed_effect), epsilon[0]),
                'theta_tau': {'lambda': model.diffusion_re_param['lambda'] + std_factor * jax.lax.mul(jax.numpy.abs(model.diffusion_re_param['lambda']), epsilon[1])}
            },
            'drift': {
                'mu': model.drift_re_param['mu'] + std_factor * jax.lax.mul(jax.numpy.abs(model.drift_re_param['mu']), epsilon[2:2+len(model.drift_re_param['mu'])]),
                'omega2': model.drift_re_param['omega2'] + std_factor * jax.lax.mul(jax.numpy.abs(model.drift_re_param['omega2']), from_vector_to_covariance(epsilon[2+len(model.drift_re_param['mu']):], len(model.drift_re_param['mu']), precompute_indices(covariance_to_estimate)))},
        }
    elif model.diffusion_re_dist == 'weibull':
        init = {
            'diffusion': {
                'eta': model.diffusion_fixed_effect + std_factor * jax.lax.mul(jax.numpy.abs(model.diffusion_fixed_effect), epsilon[0]),
                'theta_tau': {'c': model.diffusion_re_param['c'] + std_factor * jax.lax.mul(jax.numpy.abs(model.diffusion_re_param['c']), epsilon[1]), 'scale': model.diffusion_re_param['scale'] + std_factor * jax.lax.mul(jax.numpy.abs(model.diffusion_re_param['scale']), epsilon[2])}
            },
            'drift': {
                'mu': model.drift_re_param['mu'] + std_factor * jax.lax.mul(jax.numpy.abs(model.drift_re_param['mu']), epsilon[3:3+len(model.drift_re_param['mu'])]),
                'omega2': model.drift_re_param['omega2'] + std_factor * jax.lax.mul(jax.numpy.abs(model.drift_re_param['omega2']), from_vector_to_covariance(epsilon[3+len(model.drift_re_param['mu']):], len(model.drift_re_param['mu']), precompute_indices(covariance_to_estimate)))},
        }
    elif model.diffusion_re_dist == 'gamma':
        init = {
            'diffusion': {
                'eta': model.diffusion_fixed_effect + std_factor * jax.lax.mul(jax.numpy.abs(model.diffusion_fixed_effect), epsilon[0]),
                'theta_tau': {'shape': model.diffusion_re_param['shape'] + std_factor * jax.lax.mul(jax.numpy.abs(model.diffusion_re_param['shape']), epsilon[1]), 'scale': model.diffusion_re_param['scale'] + std_factor * jax.lax.mul(jax.numpy.abs(model.diffusion_re_param['scale']), epsilon[2])}
            },
            'drift': {
                'mu': model.drift_re_param['mu'] + std_factor * jax.lax.mul(jax.numpy.abs(model.drift_re_param['mu']), epsilon[3:3+len(model.drift_re_param['mu'])]),
                'omega2': model.drift_re_param['omega2'] + std_factor * jax.lax.mul(jax.numpy.abs(model.drift_re_param['omega2']), from_vector_to_covariance(epsilon[3+len(model.drift_re_param['mu']):], len(model.drift_re_param['mu']), precompute_indices(covariance_to_estimate)))},
        }
    elif model.diffusion_re_dist == 'lognormal':
        init = {
            'diffusion': {
                'eta': model.diffusion_fixed_effect + std_factor * jax.lax.mul(jax.numpy.abs(model.diffusion_fixed_effect), epsilon[0]),
                'theta_tau': {'mean': model.diffusion_re_param['mean'] + std_factor * jax.lax.mul(jax.numpy.abs(model.diffusion_re_param['mean']), epsilon[1]), 'sigma': model.diffusion_re_param['sigma'] + std_factor * jax.lax.mul(jax.numpy.abs(model.diffusion_re_param['sigma']), epsilon[2])}
            },
            'drift': {
                'mu': model.drift_re_param['mu'] + std_factor * jax.lax.mul(jax.numpy.abs(model.drift_re_param['mu']), epsilon[3:3+len(model.drift_re_param['mu'])]),
                'omega2': model.drift_re_param['omega2'] + std_factor * jax.lax.mul(jax.numpy.abs(model.drift_re_param['omega2']), from_vector_to_covariance(epsilon[3+len(model.drift_re_param['mu']):], len(model.drift_re_param['mu']), precompute_indices(covariance_to_estimate)))},
        }

    for t_max in t_max_seq:
        t_index = [i for i, valeur in enumerate(t_max_seq) if valeur == t_max]
        for h in h_seq:
            h_index = [i for i, valeur in enumerate(h_seq) if valeur == h]

            y, time_mat = extract_data(y_init, time_mat_init, 0, t_max, h)

            eta_hat, theta_tau_hat, tau_hat, mu_hat, omega2_hat = model.fit_mixed_sde(
                y, time_mat, init, covariance_to_estimate, method='joint')
            omega2_hat_vect = from_covariance_to_vector(
                omega2_hat, precompute_indices(covariance_to_estimate))

            _, _, _, mu_hat_2, omega2_hat_2 = model.fit_mixed_sde(
                y, time_mat, init, covariance_to_estimate, method='stepwise')
            omega2_hat_2_vect = from_covariance_to_vector(
                omega2_hat_2, precompute_indices(covariance_to_estimate))

            eta_est = eta_est.at[0, h_index, t_index].set(eta_hat)

            theta_tau_est = theta_tau_est.at[:, h_index, t_index].set(
                theta_tau_hat.reshape(theta_tau_hat.shape[0], 1))

            temp = estim_theta_tau(model.diffusion_re_dist, tau[:, None], jax.numpy.array(
                list(init['diffusion']['theta_tau'].values())))
            theta_true_tau_est = theta_true_tau_est.at[:, h_index, t_index].set(
                temp.reshape(temp.shape[0], 1))

            theta_phi_est_joint = theta_phi_est_joint.at[0:model.drift_re_param['mu'].shape[0], h_index, t_index].set(
                mu_hat.reshape(mu_hat.shape[0], 1))

            theta_phi_est_joint = theta_phi_est_joint.at[model.drift_re_param['mu'].shape[0]:, h_index, t_index].set(
                omega2_hat_vect.reshape(omega2_hat_vect.shape[0], 1))

            theta_phi_est_stepwise = theta_phi_est_stepwise.at[0:model.drift_re_param['mu'].shape[0], h_index, t_index].set(
                mu_hat_2.reshape(mu_hat_2.shape[0], 1))

            theta_phi_est_stepwise = theta_phi_est_stepwise.at[model.drift_re_param['mu'].shape[0]:, h_index, t_index].set(
                omega2_hat_2_vect.reshape(omega2_hat_2_vect.shape[0], 1))

            tau_est = tau_est.at[:, h_index, t_index].set(tau_hat)

    res = {'eta_est': eta_est,
           'theta_phi_est_joint': theta_phi_est_joint,
           'theta_phi_est_stepwise': theta_phi_est_stepwise,
           'theta_tau_est': theta_tau_est,
           'theta_true_tau_est': theta_true_tau_est,
           'tau_est': tau_est,
           'phi': phi,
           'tau': tau,
           'seed': key}

    return res
