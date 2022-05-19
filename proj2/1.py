import numpy as np
import networkx as nx
from utils import is_irreducable, draw_graph, calculate_irreducablity_probability_mean
import matplotlib.pyplot as plt

EPOCHS = 1000

graph_type = input("Please input graph type(ER\WS\BA): ")

p_list = np.linspace(0, 1, 10, dtype=float)

figure, ax = plt.subplots(nrows=2, ncols=5, figsize=(20, 8), sharex=True, sharey=True)
for i, n in zip(range(10), np.geomspace(2, 100, 10, dtype=int)):
    probability_mean_list = calculate_irreducablity_probability_mean(n, p_list, EPOCHS, graph_type)
    ax[i//5, i%5].plot(p_list, probability_mean_list)
    ax[i//5, i%5].set_title(f'n = {n}')

figure.suptitle(f'Graph type: {graph_type}', fontweight ="bold") 
figure.supxlabel('p')
figure.supylabel('irreducabliry-probability')

plt.savefig('{graph_type}.png')