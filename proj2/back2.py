import numpy as np
from shutil import copyfile as cp
from utils import calculate_survival_time, \
        simulate_isolation_by_distribution, \
        simulate_isolation_survival_by_pareto_shape

EPOCHS = 1000
N_UP_LIMIT = 200
# 'Static' or 'Dynamic'
MODE = 'Static'
EXPO_LAMBDA = 2
PARETO_SCALE = 1
PARETO_SHAPE = 3


def result(src):
    return cp(src, "templates/result.html")

def back2(l_distro, mode):
    l = {
        "expo": lambda: np.random.exponential(scale=1/EXPO_LAMBDA, size=N_UP_LIMIT),
        "pareto": lambda:  np.random.pareto(PARETO_SCALE, size=N_UP_LIMIT) * PARETO_SHAPE,
    }[l_distro]()

    
    if mode == "a":
        calculate_survival_time(N_UP_LIMIT, l, mode=MODE)
        result("simulation_survival_time.html")
    elif mode == "d":
        simulate_isolation_by_distribution(
            N_UP_LIMIT//10, number_of_simulations=EPOCHS, pareto_scale=PARETO_SCALE)
        result("simulate_isolation_by_distribution.html")
    elif mode == "e":
        simulate_isolation_survival_by_pareto_shape(
            N_UP_LIMIT//10, number_of_simulations=EPOCHS, scale=PARETO_SCALE, number_of_plotting_points=1000)
        result("simulate_isolation_survival_by_pareto_shape.html")

    else:
        raise NotImplementedError(f"{mode} not implemented yet")
