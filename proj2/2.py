import numpy as np
from utils import calculate_survival_time
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
    l = np.random.pareto(PARETO_SHAPE, size=N_UP_LIMIT) * PARETO_SCALE

calculate_survival_time(N_UP_LIMIT, l, mode=MODE)