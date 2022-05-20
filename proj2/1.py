#!/usr/bin/env python3
import numpy as np
from utils import calculate_irreducablity_probability_mean
import matplotlib.pyplot as plt

EPOCHS = 1000
DRAW_POINTS = 30

GRAPH_TYPES = ["ER", "WS", "BA"]


def calculate_and_save_fig(graph_type):
    p_list = np.linspace(0, 1, 10, dtype=float)

    figure, ax = plt.subplots(nrows=2, ncols=5,
                              figsize=(20, 8), sharex=True, sharey=True)
    for i, n in zip(range(10), np.geomspace(2, DRAW_POINTS, 10, dtype=int), strict=True):
        probability_mean_list, equation_3s = calculate_irreducablity_probability_mean(
            n, p_list, EPOCHS, graph_type)
        ax[i//5, i % 5].plot(p_list, probability_mean_list,
                             'b-', label='Simulation')
        ax[i//5, i % 5].plot(p_list, equation_3s, 'r--', label='Equation 3')
        ax[i//5, i % 5].set_title(f'n = {n}')
        ax[i//5, i % 5].legend()

    figure.suptitle(f'Graph type: {graph_type}', fontweight="bold")
    figure.supxlabel('p')
    figure.supylabel('irreducabliry-probability')

    plt.savefig(f"{graph_type}.png")


def main_interactive():
    graph_type = input(f"Enter graph type {'-'.join(GRAPH_TYPES)} ")
    assert graph_type in GRAPH_TYPES
    calculate_and_save_fig(graph_type)


def main_all():
    for graph_type in GRAPH_TYPES:
        calculate_and_save_fig(graph_type)
        print(f"-->{graph_type} saved")


if __name__ == "__main__":
    main_all()
