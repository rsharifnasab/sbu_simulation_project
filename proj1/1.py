#!/usr/bin/env python
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def draw_graph(G):
    options = {
        "font_size": 10,
        "node_size": 300,
        "edgecolors": "black",
        "linewidths": 2,
        "width": 1,
        "with_labels": False,
    }
    nx.draw(G, **options)
    plt.show()


def calculate_expected_edges(n, p):
    """ Calculate the expected number of edges """
    return n * (n - 1) / 2 * p

def calculate_actual_edges(n, p):
    """ Do an experiment and return the number of edges """
    G = nx.erdos_renyi_graph(n, p)
    return nx.number_of_edges(G)

def simulation_iter(n, p):
    actual_edges = calculate_actual_edges(n, p)
    expected_edges = calculate_expected_edges(n, p)
    diff = abs(actual_edges - expected_edges) / expected_edges

    print(f"Act: {actual_edges}, exp: {int(expected_edges)}, diff: {diff:.3f}")

    assert diff < 0.05

def main():
    starting_n, ending_n = 2000,3000

    for n in range(starting_n, ending_n, 200):
        for p in np.linspace(0.01, 0.99, 10):
            simulation_iter(n, p)

    # draw_graph(G)


if __name__ == "__main__":
    main()
