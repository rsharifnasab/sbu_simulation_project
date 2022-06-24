import numpy as np
from utils import calculate_survival_time, simulate_survival_by_search, simulate_isolation_by_search, simulate_isolation_by_distribution, simulate_isolation_survival_by_pareto_shape
import matplotlib.pyplot as plt

EPOCHS = 1000
N_UP_LIMIT = 200
# 'Static' or 'Dynamic'
MODE = 'Static'
# 'Exponential' or 'Pareto'
L_DISTRIBUTION = 'Exponential'
EXPO_LAMBDA = 2
PARETO_SCALE = 1
PARETO_SHAPE = 3


if L_DISTRIBUTION == 'Exponential':
    l = np.random.exponential(scale=1/EXPO_LAMBDA, size=N_UP_LIMIT)
elif L_DISTRIBUTION == 'Pareto':
    l = np.random.pareto(PARETO_SCALE, size=N_UP_LIMIT) * PARETO_SHAPE

# Part a
calculate_survival_time(N_UP_LIMIT, l, mode=MODE)

# Part b
simulate_survival_by_search(N_UP_LIMIT, number_of_simulations=EPOCHS//10, pareto_scale=PARETO_SCALE, pareto_shape = PARETO_SHAPE, expo_lambda = EXPO_LAMBDA, number_of_plotting_points=100)

# Part c
simulate_isolation_by_search(N_UP_LIMIT, number_of_simulations=EPOCHS//10, pareto_scale=PARETO_SCALE, pareto_shape = PARETO_SHAPE, expo_lambda = EXPO_LAMBDA)

# Part d
simulate_isolation_by_distribution(N_UP_LIMIT//10, number_of_simulations=EPOCHS, pareto_scale=PARETO_SCALE)

# Part e
simulate_isolation_survival_by_pareto_shape(N_UP_LIMIT//10, number_of_simulations=EPOCHS, scale=PARETO_SCALE, number_of_plotting_points=1000)
