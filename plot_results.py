import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import pandas as pd
import os
import numpy as np

from pathlib import Path
DIRECTORY_PATH = Path.cwd() / 'res'


## ---------------------------------- ##
## -- Plotting results for model 1 -- ##
## ---------------------------------- ##


true_values = {
    r'$\eta$': 0.5,
    r'$\alpha$': -0.7,
    r'$\sigma$': 0.7,
    r'$\mu$': 2.0,
    r'$\omega^2$': 1.0,
}

file = os.path.join(DIRECTORY_PATH, 'res_m1.pkl')

df = pd.read_pickle(file)
df['n'] = df['T'] / df['h']
df['n'] = df['n'].astype(int)
df = df[df['h'] != 0.01]
df_clean = df.drop(columns=['(h,T)', 'h', 'm_t', 'sigma_t', 'mu_s', 'omega2_s'])
df_clean = df_clean.rename(columns={
    'eta': r'$\eta$',
    'm': r'$\alpha$',
    'sigma': r'$\sigma$',
    'mu_j': r'$\mu$',
    'omega2_j': r'$\omega^2$',
})


id_cols = ['n', 'T', 'N']
df_long = df_clean.melt(
    id_vars=id_cols, var_name='parameter', value_name='value')
df_long['scenario'] = 'T_' + df_long['T'].astype(
    str) + '_n_' + df_long['n'].astype(str) + '_N_' + df_long['N'].astype(str)


unique_scenarios = df_long['scenario'].unique()
n_values = df_long[['scenario', 'N']].drop_duplicates().set_index('scenario')

palette = {}
for N_value in sorted(df_long['N'].unique()):
    scenarios_N = n_values[n_values['N'] == N_value].index.tolist()
    nb_colors = len(scenarios_N)

    cmap_name = 'Blues' if N_value == 100 else 'Greens' if N_value == 200 else 'Oranges'
    gradient = sns.color_palette(cmap_name, nb_colors)

    for scenario, color in zip(scenarios_N, gradient):
        palette[scenario] = to_hex(color)

parameters = df_long['parameter'].unique()
n_params = len(parameters)
n_cols = 2
n_rows = int(np.ceil(n_params / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(
    6 * n_cols, 5 * n_rows), constrained_layout=True)
axes = axes.flatten()

for i, param in enumerate(parameters):
    ax = axes[i]
    df_param = df_long[df_long['parameter'] == param]

    sns.boxplot(data=df_param, x='scenario', y='value', ax=ax,
                palette=palette)

    if param in true_values:
        ax.axhline(y=true_values[param], linestyle='--',
                   color='black', label='True value')

    ax.set_title(f"Parameter: {param}", fontsize=14)
    ax.tick_params(axis='x', rotation=45)
    ax.set_xlabel('')
    ax.set_ylabel('Estimated value')


for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.savefig(DIRECTORY_PATH / 'boxplots-model1.png', dpi=300)


## ---------------------------------- ##
## -- Plotting results for model 2 -- ##
## ---------------------------------- ##


true_values = {
    r'$\eta$': 0.5,
    r'$\alpha$': 1.0,
    r'$\lambda$': 0.6,
    r'$\mu_1$': 2.0,
    r'$\mu_2$': 1.0,
    r'$\omega_1^2$': 1.0,
}

file = os.path.join(DIRECTORY_PATH, 'res_m2.pkl')

df = pd.read_pickle(file)
df['n'] = df['T'] / df['h']
df['n'] = df['n'].astype(int)
df = df[df['h'] != 0.01]
df_clean = df.drop(columns=['(h,T)', 'h', 'alpha_t',
                   'lambda_t', 'mu1_s', 'mu2_s', 'omega21_s'])
df_clean = df_clean.rename(columns={
    'eta': r'$\eta$',
    'alpha': r'$\alpha$',
    'lambda': r'$\lambda$',
    'mu1_j': r'$\mu_1$',
    'mu2_j': r'$\mu_2$',
    'omega21_j': r'$\omega_1^2$',
})


id_cols = ['n', 'T', 'N']
df_long = df_clean.melt(
    id_vars=id_cols, var_name='parameter', value_name='value')
df_long['scenario'] = 'T_' + df_long['T'].astype(
    str) + '_n_' + df_long['n'].astype(str) + '_N_' + df_long['N'].astype(str)


unique_scenarios = df_long['scenario'].unique()
n_values = df_long[['scenario', 'N']].drop_duplicates().set_index('scenario')

palette = {}
for N_value in sorted(df_long['N'].unique()):
    scenarios_N = n_values[n_values['N'] == N_value].index.tolist()
    nb_colors = len(scenarios_N)

    cmap_name = 'Blues' if N_value == 100 else 'Greens' if N_value == 200 else 'Oranges'
    gradient = sns.color_palette(cmap_name, nb_colors)

    for scenario, color in zip(scenarios_N, gradient):
        palette[scenario] = to_hex(color)


parameters = df_long['parameter'].unique()
n_params = len(parameters)
n_cols = 2
n_rows = int(np.ceil(n_params / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(
    6 * n_cols, 5 * n_rows), constrained_layout=True)
axes = axes.flatten()

for i, param in enumerate(parameters):
    ax = axes[i]
    df_param = df_long[df_long['parameter'] == param]

    sns.boxplot(data=df_param, x='scenario', y='value', ax=ax,
                palette=palette)

    if param in true_values:
        ax.axhline(y=true_values[param], linestyle='--',
                   color='black', label='True value')

    ax.set_title(f"Parameter: {param}", fontsize=14)
    ax.tick_params(axis='x', rotation=45)
    ax.set_xlabel('')
    ax.set_ylabel('Estimated value')

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.savefig(DIRECTORY_PATH / 'boxplots-model2.png', dpi=300)


## ---------------------------------- ##
## -- Plotting results for model 3 -- ##
## ---------------------------------- ##


true_values = {
    r'$\eta$': 0.5,
    r'$\lambda$': 1.0,
    r'$\mu_1$': 2.0,
    r'$\mu_2$': 1.0,
    r'$\omega_1^2$': 1.0,
    r'$\omega_2^2$': 0.5,
    r'$\omega_3$': -0.2,
}

file = os.path.join(DIRECTORY_PATH, 'res_m3.pkl')
df = pd.read_pickle(file)
df['n'] = df['T'] / df['h']
df['n'] = df['n'].astype(int)
df = df[df['h'] != 0.01]
df_clean = df.drop(columns=['(h,T)', 'h', 'alpha_t', 'mu1_s',
                   'mu2_s', 'omega21_s', 'omega22_s', 'omega23_s'])
df_clean = df_clean.rename(columns={
    'eta': r'$\eta$',
    'alpha': r'$\lambda$',
    'mu1_j': r'$\mu_1$',
    'mu2_j': r'$\mu_2$',
    'omega21_j': r'$\omega_1^2$',
    'omega22_j': r'$\omega_3$',
    'omega23_j': r'$\omega_2^2$',
})

id_cols = ['n', 'T', 'N']
df_long = df_clean.melt(
    id_vars=id_cols, var_name='parameter', value_name='value')
df_long['scenario'] = 'T_' + df_long['T'].astype(
    str) + '_n_' + df_long['n'].astype(str) + '_N_' + df_long['N'].astype(str)

unique_scenarios = df_long['scenario'].unique()
n_values = df_long[['scenario', 'N']].drop_duplicates().set_index('scenario')

palette = {}
for N_value in sorted(df_long['N'].unique()):
    scenarios_N = n_values[n_values['N'] == N_value].index.tolist()
    nb_colors = len(scenarios_N)

    cmap_name = 'Blues' if N_value == 100 else 'Greens' if N_value == 200 else 'Oranges'
    gradient = sns.color_palette(cmap_name, nb_colors)

    for scenario, color in zip(scenarios_N, gradient):
        palette[scenario] = to_hex(color)

parameters = df_long['parameter'].unique()
n_params = len(parameters)
n_cols = 2
n_rows = int(np.ceil(n_params / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(
    6 * n_cols, 5 * n_rows), constrained_layout=True)
axes = axes.flatten()

for i, param in enumerate(parameters):
    ax = axes[i]
    df_param = df_long[df_long['parameter'] == param]

    sns.boxplot(data=df_param, x='scenario', y='value', ax=ax,
                palette=palette)

    if param in true_values:
        ax.axhline(y=true_values[param], linestyle='--',
                   color='black', label='True value')

    ax.set_title(f"Parameter: {param}", fontsize=14)
    ax.tick_params(axis='x', rotation=45)
    ax.set_xlabel('')
    ax.set_ylabel('Estimated value')

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.savefig(DIRECTORY_PATH / 'boxplots-model3.png', dpi=300)
