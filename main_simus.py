

import jax
import pickle
import argparse
import os
from my_utils import *


from mixedsde.fit import *
from mixedsde.model_class import *
from mixedsde.data_processing import *

# -- Define parser
parser = argparse.ArgumentParser(
    description="Script simulating N trajectories.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument("--seed", help="Seed to be used", type=int, default=1)
parser.add_argument("--N", help="Number of trajectories",
                    type=int, default=100)
parser.add_argument("--model", help="Simulated model", type=str, default='m1')

args = parser.parse_args()


# -- General settings
##

output_dir = "res"

# -- True parameter values

eta = jax.numpy.array([0.5])

theta_tau = {
    'exponential': {'lambda': 1.0},
    'weibull': {'c': 1.0, 'scale': 0.6},
    'gamma': {'shape': 1.0, 'scale': 0.5},
    'lognormal': {'mean': -0.7, 'sigma': 0.7}
}


# --  Model settings
# -- Drift and diffusion functions

def a2(y, time):
    a_value = jax.numpy.array(
        [-y / jax.numpy.sqrt(1 + y * y), -1.0 /
         jax.numpy.sqrt(1 + y * y) + 0.0 * time]
    )
    return a_value


def a3(y, time):
    a_value = jax.numpy.array(
        [-1 / jax.numpy.sqrt(1 + y * y) + 0.0 * time]
    )
    return a_value


def c(y, eta, time):
    c_value = jax.numpy.exp(jax.numpy.arctan(y) * eta) + 0.0 * time
    return c_value


def c2(y, eta, time):
    c_value = jax.numpy.exp(time * eta/2) + 0.0 * y
    return c_value


# Model 1
# -- 1 random effect and no fixed effect in the drift, log-normal distribution for the random effect in the diffusion coefficient

theta_phi_1 = {
    'normal': {'mu': jax.numpy.array([2.0]), 'omega2': jax.numpy.array([1.0]).reshape(1, 1)}
}

m1 = Mixedsde(drift_func=a3, diffusion_func=c2,
              drift_re_param=theta_phi_1['normal'], diffusion_fixed_effect=eta, diffusion_re_dist='lognormal', diffusion_re_param=theta_tau['lognormal'])


# Model 2
# -- 1 random effect and 1 fixed effect in the drift, weibull distribution for the random effect in the diffusion coefficient

theta_phi_2 = {
    'normal': {'mu': jax.numpy.array([2.0, 1.0]), 'omega2': jax.numpy.array([[1.0, 0.0], [0.0, 0.0]])}
}

m2 = Mixedsde(drift_func=a2, diffusion_func=c,
              drift_re_param=theta_phi_2['normal'], diffusion_fixed_effect=eta, diffusion_re_dist='weibull', diffusion_re_param=theta_tau['weibull'])

# Model 3
# -- 2 correlated random effects in the drift, exponential distribution for the random effect in the diffusion coefficient

theta_phi_3 = {
    'normal': {'mu': jax.numpy.array([2.0, 1.0]), 'omega2': jax.numpy.array([[1.0, -0.2], [-0.2, 0.5]])}
}

m3 = Mixedsde(drift_func=a2, diffusion_func=c,
              drift_re_param=theta_phi_3['normal'], diffusion_fixed_effect=eta, diffusion_re_dist='exponential', diffusion_re_param=theta_tau['exponential'])


# -- Design for the simulations

h_seq = [0.005, 0.001]
h_euler = 0.0001
t_max_seq = [5, 10]

y0 = 0.0
t0 = 0.0


# -- User settings

seed = args.seed

nb_trajectories = args.N

if args.model == 'm1':
    model = m1
    covariance_to_estimate = jax.numpy.array([[1.0]], dtype=int).reshape(1, 1)
elif args.model == 'm2':
    model = m2
    covariance_to_estimate = jax.numpy.array(
        [[1.0, 0.0], [0.0, 0.0]], dtype=int)
elif args.model == 'm3':
    model = m3
    covariance_to_estimate = jax.numpy.array(
        [[1.0, 1.0], [1.0, 1.0]], dtype=int)
else:
    raise ValueError("Unknown model")

filename = os.path.join(
    output_dir, f"res_N{nb_trajectories}_{seed}_{args.model}.pickle")

# Main loop

res = oneRun(seed,
             nb_trajectories,
             model,
             h_seq,
             t_max_seq,
             h_euler,
             y0,
             t0,
             covariance_to_estimate)

with open(filename, "wb") as fichier:
    pickle.dump(res, fichier)
