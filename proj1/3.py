import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random

def add_new_edge(graph):
    non_edges = list(nx.non_edges(graph))
    a, b = random.choice(non_edges)
    graph.add_edge(a, b)
    return

def new_simulation(dist, all_edges):
    G = nx.empty_graph(n)
    for i in range(all_edges):
        add_new_edge(G)
        if nx.is_connected(G):
            dist[i:] += 1
            break

def simulation(number_of_iterations, n):
    all_edges = n*(n-1)//2
    dist_count = np.zeros(all_edges, dtype=int)
    for _ in range(number_of_iterations):
        new_simulation(dist_count, all_edges)
    dist = dist_count / number_of_iterations
    return dist

n = 50
distribution = simulation(100, n)
plt.plot(distribution)
one = np.log(n)/n
plt.axvline(x = one, color = 'r')
plt.show()
