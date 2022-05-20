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


graph_providers = {
    'ER': lambda n, p: nx.erdos_renyi_graph(n=n, p=p), # ok
    'WS': lambda n, p: nx.newman_watts_strogatz_graph(n=n, k=int(0.4 * n), p=p),
    'BA': lambda n, p: nx.barabasi_albert_graph(n=n+1, m=int(n**p))
}
GRAPH_TYPES = graph_providers.keys()


def calculate_irreducablity_probability_mean(n, p_list, epochs, graph_provider):
    mean_list = []
    equation_3s = []
    for p in p_list:
        no_connected = 0
        equation_tmp = []
        for _ in range(epochs):
            graph = graph_provider(n, p)
            # draw_graph(graph)
            if nx.is_connected(graph):
                no_connected += 1
            equation_tmp.append(calculate_equation_3(graph, p))

        # print(no_connected / epochs)
        equation_3s.append(sum(equation_tmp) / len(equation_tmp))
        mean_list.append(no_connected / epochs)
    return mean_list, equation_3s


def calculate_equation_3(G, p):
    summ = np.sum([p**k_i[1] for k_i in G.degree])
    return np.e**(-(1-p)*summ)
