from itertools import product
import os
import pickle
import argparse
import jax
import pandas as pd
from scipy.special import digamma, polygamma


from pathlib import Path
DIRECTORY_PATH = Path.cwd() / 'res'

# Define parser
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Script compiling simulation results.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--model", help="Simulated model", type=str, default='m1')

args = parser.parse_args()

# General settings
# ------------------------------------------------------------------------------

h_seq = [0.005, 0.001]
H_EULER = 0.0001
t_max_seq = [5, 10]

transformed_h = [h for h in h_seq for _ in range(len(t_max_seq))]
transformed_T = t_max_seq*len(h_seq)
h_T = list(product(h_seq, t_max_seq))


NB_REP = 500
rep_h_T = h_T * NB_REP
rep_h = transformed_h * NB_REP
rep_T = transformed_T * NB_REP

rep_N = [200, 500] * NB_REP
model = args.model

for nb_trajectories in [200, 500]:
    DF_NAME = f'res{nb_trajectories}_trajectories_model_{model}'
    files = []
    for filename in os.listdir(DIRECTORY_PATH):
        if (model in filename) and (f'N{nb_trajectories}' in filename):
            files.append(filename)
    eta_ = jax.numpy.zeros((NB_REP, len(h_seq), len(t_max_seq)))

    match model:
        case 'm1':
            mu_j = jax.numpy.zeros((NB_REP, len(h_seq), len(t_max_seq)))
            mu_s = jax.numpy.zeros((NB_REP, len(h_seq), len(t_max_seq)))
            omega2_j = jax.numpy.zeros((NB_REP, len(h_seq), len(t_max_seq)))
            omega2_s = jax.numpy.zeros((NB_REP, len(h_seq), len(t_max_seq)))
            m_ = jax.numpy.zeros((NB_REP, len(h_seq), len(t_max_seq)))
            sigma_ = jax.numpy.zeros((NB_REP, len(h_seq), len(t_max_seq)))
            m_t = jax.numpy.zeros((NB_REP, len(h_seq), len(t_max_seq)))
            sigma_t = jax.numpy.zeros((NB_REP, len(h_seq), len(t_max_seq)))
            k = 0
            for file in sorted(files):
                with open(os.path.join(DIRECTORY_PATH, file), "rb") as fichier:
                    results = pickle.load(fichier)
                    eta_ = eta_.at[k, :, :].set(results['eta_est'][0, :, :])
                    m_ = m_.at[k, :, :].set(
                        results['theta_tau_est'][0, :, :])
                    sigma_ = sigma_.at[k, :, :].set(
                        results['theta_tau_est'][1, :, :])
                    m_t = m_t.at[k, :, :].set(
                        results['theta_true_tau_est'][0, :, :])
                    sigma_t = sigma_t.at[k, :, :].set(
                        results['theta_true_tau_est'][1, :, :])
                    mu_j = mu_j.at[k, :, :].set(
                        results['theta_phi_est_joint'][0, :, :])
                    omega2_j = omega2_j.at[k, :, :].set(
                        results['theta_phi_est_joint'][1, :, :])
                    mu_s = mu_s.at[k, :, :].set(
                        results['theta_phi_est_stepwise'][0, :, :])
                    omega2_s = omega2_s.at[k, :, :].set(
                        results['theta_phi_est_stepwise'][1, :, :])
                    k += 1
            locals()[DF_NAME] = pd.DataFrame({
                'eta': eta_.reshape(-1),
                'm': m_.reshape(-1),
                'sigma': sigma_.reshape(-1),
                'm_t': m_t.reshape(-1),
                'sigma_t': sigma_t.reshape(-1),
                'mu_j': mu_j.reshape(-1),
                'mu_s': mu_s.reshape(-1),
                'omega2_j': omega2_j.reshape(-1),
                'omega2_s': omega2_s.reshape(-1),
                '(h,T)': rep_h_T,
                'h': rep_h,
                'T': rep_T,
            })
        case 'm2':
            mu_j = jax.numpy.zeros((NB_REP, 2, len(h_seq), len(t_max_seq)))
            mu_s = jax.numpy.zeros((NB_REP, 2, len(h_seq), len(t_max_seq)))
            omega2_j = jax.numpy.zeros((NB_REP, 1, len(h_seq), len(t_max_seq)))
            omega2_s = jax.numpy.zeros((NB_REP, 1, len(h_seq), len(t_max_seq)))
            alpha = jax.numpy.zeros((NB_REP, len(h_seq), len(t_max_seq)))
            lambda_ = jax.numpy.zeros((NB_REP, len(h_seq), len(t_max_seq)))
            alpha_t = jax.numpy.zeros((NB_REP, len(h_seq), len(t_max_seq)))
            lambda_t = jax.numpy.zeros((NB_REP, len(h_seq), len(t_max_seq)))
            k = 0
            for file in sorted(files):
                with open(os.path.join(DIRECTORY_PATH, file), "rb") as fichier:
                    results = pickle.load(fichier)
                    eta_ = eta_.at[k, :, :].set(results['eta_est'][0, :, :])
                    alpha = alpha.at[k, :, :].set(
                        results['theta_tau_est'][0, :, :])
                    lambda_ = lambda_.at[k, :, :].set(
                        results['theta_tau_est'][1, :, :])
                    alpha_t = alpha_t.at[k, :, :].set(
                        results['theta_true_tau_est'][0, :, :])
                    lambda_t = lambda_t.at[k, :, :].set(
                        results['theta_true_tau_est'][1, :, :])
                    mu_j = mu_j.at[k, :, :].set(
                        results['theta_phi_est_joint'][:2, :, :])
                    omega2_j = omega2_j.at[k, :, :].set(
                        results['theta_phi_est_joint'][2:, :, :])
                    mu_s = mu_s.at[k, :, :].set(
                        results['theta_phi_est_stepwise'][:2, :, :])
                    omega2_s = omega2_s.at[k, :, :].set(
                        results['theta_phi_est_stepwise'][2:, :, :])
                    k += 1
            locals()[DF_NAME] = pd.DataFrame({
                'eta': eta_.reshape(-1),
                'alpha': alpha.reshape(-1),
                'lambda': lambda_.reshape(-1),
                'alpha_t': alpha_t.reshape(-1),
                'lambda_t': lambda_t.reshape(-1),
                'mu1_j': mu_j[:, 0, :, :].reshape(-1),
                'mu1_s': mu_s[:, 0, :, :].reshape(-1),
                'mu2_j': mu_j[:, 1, :, :].reshape(-1),
                'mu2_s': mu_s[:, 1, :, :].reshape(-1),
                'omega21_j': omega2_j[:, 0, :, :].reshape(-1),
                'omega21_s': omega2_s[:, 0, :, :].reshape(-1),
                '(h,T)': rep_h_T,
                'h': rep_h,
                'T': rep_T})
        case 'm3':
            mu_j = jax.numpy.zeros((NB_REP, 2, len(h_seq), len(t_max_seq)))
            mu_s = jax.numpy.zeros((NB_REP, 2, len(h_seq), len(t_max_seq)))
            omega2_j = jax.numpy.zeros((NB_REP, 3, len(h_seq), len(t_max_seq)))
            omega2_s = jax.numpy.zeros((NB_REP, 3, len(h_seq), len(t_max_seq)))
            alpha = jax.numpy.zeros((NB_REP, len(h_seq), len(t_max_seq)))
            alpha_t = jax.numpy.zeros((NB_REP, len(h_seq), len(t_max_seq)))
            k = 0
            for file in sorted(files):
                with open(os.path.join(DIRECTORY_PATH, file), "rb") as fichier:
                    results = pickle.load(fichier)
                    eta_ = eta_.at[k, :, :].set(results['eta_est'][0, :, :])
                    alpha = alpha.at[k, :, :].set(
                        results['theta_tau_est'][0, :, :])
                    alpha_t = alpha_t.at[k, :, :].set(
                        results['theta_true_tau_est'][0, :, :])
                    mu_j = mu_j.at[k, :, :].set(
                        results['theta_phi_est_joint'][:2, :, :])
                    omega2_j = omega2_j.at[k, :, :].set(
                        results['theta_phi_est_joint'][2:, :, :])
                    mu_s = mu_s.at[k, :, :].set(
                        results['theta_phi_est_stepwise'][:2, :, :])
                    omega2_s = omega2_s.at[k, :, :].set(
                        results['theta_phi_est_stepwise'][2:, :, :])
                    k += 1
            locals()[DF_NAME] = pd.DataFrame({
                'eta': eta_.reshape(-1),
                'alpha': alpha.reshape(-1),
                'alpha_t': alpha_t.reshape(-1),
                'mu1_j': mu_j[:, 0, :, :].reshape(-1),
                'mu1_s': mu_s[:, 0, :, :].reshape(-1),
                'mu2_j': mu_j[:, 1, :, :].reshape(-1),
                'mu2_s': mu_s[:, 1, :, :].reshape(-1),
                'omega21_j': omega2_j[:, 0, :, :].reshape(-1),
                'omega21_s': omega2_s[:, 0, :, :].reshape(-1),
                'omega22_j': omega2_j[:, 1, :, :].reshape(-1),
                'omega22_s': omega2_s[:, 1, :, :].reshape(-1),
                'omega23_j': omega2_j[:, 2, :, :].reshape(-1),
                'omega23_s': omega2_s[:, 2, :, :].reshape(-1),
                '(h,T)': rep_h_T,
                'h': rep_h,
                'T': rep_T})

df1 = locals()[f'res{200}_trajectories_model_{model}']
df1['N'] = 200
df2 = locals()[f'res{500}_trajectories_model_{model}']
df2['N'] = 500
dat = pd.concat([df1, df2], ignore_index=True)


file_name_dat = f'res_{model}.pkl'


match model:
    case 'm1':
        groupedDiff = dat.groupby(['N', 'T', 'h']).agg({
            'eta': ['mean', 'std'],
            'm': ['mean', 'std'],
            'sigma': ['mean', 'std'],
            'm_t': ['mean', 'std'],
            'sigma_t': ['mean', 'std'],
        }).reset_index()
        groupedDiff.columns = ['_'.join(col).strip()
                               for col in groupedDiff.columns.values]
        groupedDiff.rename(columns={
            'N_': 'N', 'T_': 'T', 'h_': 'h',
            'eta_mean': r'$m_{\eta}$', 'eta_std': r'$s_{\eta}$',
            'm_mean': r'$m_{m}$', 'm_std': r'$s_{m}$',
            'sigma_mean': r'$m_{\sigma}$', 'sigma_std': r'$s_{\sigma}$',
            'm_t_mean': r'$m_{m_t}$', 'm_t_std': r'$s_{m_t}$',
            'sigma_t_mean': r'$m_{\sigma_t}$', 'sigma_t_std': r'$s_{\sigma_t}$',
        }, inplace=True)
        groupedDiff = groupedDiff.round(2)
        groupedDrift_joint = dat.groupby(['N', 'T', 'h']).agg({
            'mu_j': ['mean', 'std'],
            'omega2_j': ['mean', 'std']
        }).reset_index()
        groupedDrift_stepwise = dat.groupby(['N', 'T', 'h']).agg({
            'mu_s': ['mean', 'std'],
            'omega2_s': ['mean', 'std']
        }).reset_index()
        groupedDrift_joint.columns = [
            '_'.join(col).strip() for col in groupedDrift_joint.columns.values]
        groupedDrift_joint.rename(columns={
            'N_': 'N', 'T_': 'T', 'h_': 'h',
            'mu_j_mean': r'$m_{\mu_j}$', 'mu_j_std': r'$s_{\mu_j}$',
            'omega2_j_mean': r'$m_{\omega^2_j}$', 'omega2_j_std': r'$s_{\omega^2_j}$'
        }, inplace=True)
        groupedDrift_stepwise.columns = [
            '_'.join(col).strip() for col in groupedDrift_stepwise.columns.values]
        groupedDrift_stepwise.rename(columns={
            'N_': 'N', 'T_': 'T', 'h_': 'h',
            'mu_s_mean': r'$m_{\mu_s}$', 'mu_s_std': r'$s_{\mu_s}$',
            'omega2_s_mean': r'$m_{\omega^2_s}$', 'omega2_s_std': r'$s_{\omega^2_s}$'
        }, inplace=True)
        groupedDrift_joint = groupedDrift_joint.round(2)
        groupedDrift_stepwise = groupedDrift_stepwise.round(2)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.expand_frame_repr', False)
        os.chdir(DIRECTORY_PATH)
        dat.to_pickle(file_name_dat)
        groupedDrift_joint.to_pickle('ResDrift_joint_m1.pkl')
        groupedDrift_stepwise.to_pickle('ResDrift_stepwise_m1.pkl')
        groupedDiff.to_pickle('ResDiffusion_m1.pkl')
    case 'm2':
        groupedDiff = dat.groupby(['N', 'T', 'h']).agg({
            'eta': ['mean', 'std'],
            'alpha': ['mean', 'std'],
            'lambda': ['mean', 'std'],
            'alpha_t': ['mean', 'std'],
            'lambda_t': ['mean', 'std']
        }).reset_index()
        groupedDiff.columns = ['_'.join(col).strip()
                               for col in groupedDiff.columns.values]
        groupedDiff.rename(columns={
            'N_': 'N', 'T_': 'T', 'h_': 'h',
            'eta_mean': r'$m_{\eta}$', 'eta_std': r'$s_{\eta}$',
            'alpha_mean': r'$m_{\alpha}$', 'alpha_std': r'$s_{\alpha}$',
            'lambda_mean': r'$m_{\lambda}$', 'lambda_std': r'$s_{\lambda}$',
            'alpha_t_mean': r'$m_{\alpha_t}$', 'alpha_t_std': r'$s_{\alpha_t}$',
            'lambda_t_mean': r'$m_{\lambda_t}$', 'lambda_t_std': r'$s_{\lambda_t}$'
        }, inplace=True)
        groupedDiff = groupedDiff.round(2)
        groupedDrift_joint = dat.groupby(['N', 'T', 'h']).agg({
            'mu1_j': ['mean', 'std'],
            'mu2_j': ['mean', 'std'],
            'omega21_j': ['mean', 'std']
        }).reset_index()
        groupedDrift_stepwise = dat.groupby(['N', 'T', 'h']).agg({
            'mu1_s': ['mean', 'std'],
            'mu2_s': ['mean', 'std'],
            'omega21_s': ['mean', 'std']
        }).reset_index()
        groupedDrift_joint.columns = [
            '_'.join(col).strip() for col in groupedDrift_joint.columns.values]
        groupedDrift_joint.rename(columns={
            'N_': 'N', 'T_': 'T', 'h_': 'h',
            'mu1_j_mean': r'$m_{\mu1_j}$', 'mu1_j_std': r'$s_{\mu1_j}$',
            'mu2_j_mean': r'$m_{\mu2_j}$', 'mu2_j_std': r'$s_{\mu2_j}$',
            'omega21_j_mean': r'$m_{\omega1^2_j}$', 'omega21_j_std': r'$s_{\omega1^2_j}$'
        }, inplace=True)
        groupedDrift_stepwise.columns = [
            '_'.join(col).strip() for col in groupedDrift_stepwise.columns.values]
        groupedDrift_stepwise.rename(columns={
            'N_': 'N', 'T_': 'T', 'h_': 'h',
            'mu1_s_mean': r'$m_{\mu1_s}$', 'mu1_s_std': r'$s_{\mu1_s}$',
            'mu2_s_mean': r'$m_{\mu2_s}$', 'mu2_s_std': r'$s_{\mu2_s}$',
            'omega21_s_mean': r'$m_{\omega1^2_s}$', 'omega21_s_std': r'$s_{\omega1^2_s}$'
        }, inplace=True)
        groupedDrift_joint = groupedDrift_joint.round(2)
        groupedDrift_stepwise = groupedDrift_stepwise.round(2)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.expand_frame_repr', False)
        os.chdir(DIRECTORY_PATH)
        dat.to_pickle(file_name_dat)
        groupedDrift_joint.to_pickle('ResDrift_joint_m2.pkl')
        groupedDrift_stepwise.to_pickle('ResDrift_stepwise_m2.pkl')
        groupedDiff.to_pickle('ResDiffusion_m2.pkl')
    case 'm3':
        groupedDiff = dat.groupby(['N', 'T', 'h']).agg({
            'eta': ['mean', 'std'],
            'alpha': ['mean', 'std'],
            'alpha_t': ['mean', 'std']
        }).reset_index()
        groupedDiff.columns = ['_'.join(col).strip()
                               for col in groupedDiff.columns.values]
        groupedDiff.rename(columns={
            'N_': 'N', 'T_': 'T', 'h_': 'h',
            'eta_mean': r'$m_{\eta}$', 'eta_std': r'$s_{\eta}$',
            'alpha_mean': r'$m_{\alpha}$', 'alpha_std': r'$s_{\alpha}$',
            'alpha_t_mean': r'$m_{\alpha_t}$', 'alpha_t_std': r'$s_{\alpha_t}$'
        }, inplace=True)
        groupedDiff = groupedDiff.round(2)
        groupedDrift_joint = dat.groupby(['N', 'T', 'h']).agg({
            'mu1_j': ['mean', 'std'],
            'mu2_j': ['mean', 'std'],
            'omega21_j': ['mean', 'std'],
            'omega22_j': ['mean', 'std'],
            'omega23_j': ['mean', 'std'],
        }).reset_index()
        groupedDrift_stepwise = dat.groupby(['N', 'T', 'h']).agg({
            'mu1_s': ['mean', 'std'],
            'mu2_s': ['mean', 'std'],
            'omega21_s': ['mean', 'std'],
            'omega22_s': ['mean', 'std'],
            'omega23_s': ['mean', 'std'],
        }).reset_index()
        groupedDrift_joint.columns = [
            '_'.join(col).strip() for col in groupedDrift_joint.columns.values]
        groupedDrift_joint.rename(columns={
            'N_': 'N', 'T_': 'T', 'h_': 'h',
            'mu1_j_mean': r'$m_{\mu1_j}$', 'mu1_j_std': r'$s_{\mu1_j}$',
            'mu2_j_mean': r'$m_{\mu2_j}$', 'mu2_j_std': r'$s_{\mu2_j}$',
            'omega21_j_mean': r'$m_{\omega1^2_j}$', 'omega21_j_std': r'$s_{\omega1^2_j}$',
            'omega22_j_mean': r'$m_{\omega2^2_j}$', 'omega22_j_std': r'$s_{\omega2^2_j}$',
            'omega23_j_mean': r'$m_{\omega3^2_j}$', 'omega23_j_std': r'$s_{\omega3^2_j}$'
        }, inplace=True)
        groupedDrift_stepwise.columns = [
            '_'.join(col).strip() for col in groupedDrift_stepwise.columns.values]
        groupedDrift_stepwise.rename(columns={
            'N_': 'N', 'T_': 'T', 'h_': 'h',
            'mu1_s_mean': r'$m_{\mu1_s}$', 'mu1_s_std': r'$s_{\mu1_s}$',
            'mu2_s_mean': r'$m_{\mu2_s}$', 'mu2_s_std': r'$s_{\mu2_s}$',
            'omega21_s_mean': r'$m_{\omega1^2_s}$', 'omega21_s_std': r'$s_{\omega1^2_s}$',
            'omega22_s_mean': r'$m_{\omega2^2_s}$', 'omega22_s_std': r'$s_{\omega2^2_s}$',
            'omega23_s_mean': r'$m_{\omega3^2_s}$', 'omega23_s_std': r'$s_{\omega3^2_s}$'
        }, inplace=True)
        groupedDrift_joint = groupedDrift_joint.round(2)
        groupedDrift_stepwise = groupedDrift_stepwise.round(2)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.expand_frame_repr', False)
        os.chdir(DIRECTORY_PATH)
        dat.to_pickle(file_name_dat)
        groupedDrift_joint.to_pickle('ResDrift_joint_m3.pkl')
        groupedDrift_stepwise.to_pickle('ResDrift_stepwise_m3.pkl')
        groupedDiff.to_pickle('ResDiffusion_m3.pkl')
