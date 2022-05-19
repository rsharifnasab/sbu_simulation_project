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
        "with_labels": True,
    }
    nx.draw(G, **options)
    plt.show()

def create_initial_matrix(n, p):
    """
    Create a 'A' matrix with a shape of n*n. All of the elements except diagonals are either 1 or 0.

    Args:
        n (int): shape of matrix
        p (float): probability threshold for being 1 or 0

    Returns:
        np.array: the 'A' matrix
    """
    mat = np.random.rand(n, n)
    np.fill_diagonal(mat, 0)
    return mat >= p

def is_irreducable(G):
    """
    Checks if given 'G' matrix is reducable.

    Args:
        G (networkx.Graph)
    """
    return nx.algorithms.all_pairs_node_connectivity(G)

def calculate_irreducablity_probability_mean(n, p_list, epochs, graph_type):
    mean_list = []
    for p in p_list:
        mean = 0
        for i in range(epochs):
            if graph_type == 'ER':
                graph = nx.erdos_renyi_graph(n=n, p=p)
            elif graph_type == 'WS':
                graph = nx.newman_watts_strogatz_graph(n=n, k=n//10, p=p)
            elif graph_type == 'BA':
                graph = nx.barabasi_albert_graph(n=n+1, m=int(n**p))
            else:
                raise RuntimeError
            # draw_graph(graph)
            if nx.is_connected(graph):
                mean += 1
        mean /= epochs
        # print(mean)
        mean_list.append(mean)
    return mean_list