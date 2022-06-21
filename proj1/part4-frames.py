#!/usr/bin/env python
# coding: utf-8

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

def draw_graph(G):
    options = {
        "font_size": 10,
        "node_size": 300,
        "edgecolors": "black",
        "linewidths": 2,
        "width": 1,
        "with_labels" : False,
    }
    return nx.draw(G, **options)


def gen_graph(N, M):
    G = nx.gnm_random_graph(n=N, m=M)
    fitness = np.random.standard_normal(N)
    fitness += abs(fitness.min())
    return G, fitness

def uniform_random_edge(G):
    all_edges = list(G.edges)
    return random.choice(all_edges)

def second(arr):
    return np.array(list(map(lambda a: a[1],(arr))))

def fitnessed_node(G, fitness):
    degrees = second(G.degree)
    final_fitness = fitness * degrees #* np.arange(0,N) why?
    final_fitness = final_fitness / np.sum(final_fitness)
    nodes = list(G.nodes)
    return np.random.choice(a=nodes, p=final_fitness, size=degrees.shape[0])

def swap_edges(G, fitness):
    i,j = uniform_random_edge(G)
    m = None
    for node in fitnessed_node(G, fitness):
        if not G.has_edge(i, node) and i != node:
            m = node
            break
    if m is None:
        return
    G.remove_edge(i,j)
    G.add_edge(i,m)

def calculate_degrees_mean(graph):
    return graph.number_of_edges() * 2.0 / graph.number_of_nodes()

def calculate_degrees_freq(graph):
    l = graph.number_of_nodes()
    degrees_freq = np.zeros((l, ), dtype=int)
    for node in graph.nodes:
        node_degree = graph.degree[node]
        degrees_freq[node_degree] += 1
    return degrees_freq



def init_ax(subname, x_low=-1, x_high=+1, y_low=-1, y_high=+1):
    fig_dim = 10
    # size of figure
    x_step = (x_high - x_low) / 10
    y_step = (y_high - y_low) / 10
    # distance between ticks in figure

    _, ax = plt.subplots(figsize=(fig_dim, fig_dim))
    # create a figure

    ax.set_xlabel("k")
    ax.set_ylabel("P(k)")
    ax.set_title(f"Scatter {subname}")
    # set figure titles and labels

    ax.set_xlim(x_low, x_high)
    ax.set_ylim(y_low, y_high)
    ax.set_xticks(np.arange(x_low, x_high, x_step))
    ax.set_yticks(np.arange(y_low, y_high, y_step))
    # set figure thicks

    return ax

def scatter(ax, matrix):
    x = matrix[:, 0]
    y_org = matrix[:, 1]
    y = y_org[:]

    ax.scatter(x, y, color="blue", s=.8)



def frame(G, N, i):
    degrees_freq = calculate_degrees_freq(G)
    ax = init_ax('subproblem 2',
                    x_low=0,
                    x_high=N,
                    y_low=0,
                    y_high=1
                )

    ax.plot(degrees_freq/N, color='red', label ='Expected')
    ax.legend(loc ="upper right")

    print(i)
    plt.savefig(f"./frames/frame-{i}.png")
    #plt.show()


def simulation(N, M, swap_count=None):
    if swap_count is None:
        swap_count = N*N
    G, fitness = gen_graph(N, M)
 
    FRAME_COUNT = 400
    for i in tqdm(range(FRAME_COUNT)):
        for _ in range(swap_count//FRAME_COUNT):
            swap_edges(G, fitness)
        
        frame(G, N, i)

    return G



if __name__ == "__main__":
    N = 90
    M = 2000
    G = simulation(N, M)


    degrees_mean = calculate_degrees_mean(G)
    after_degrees_freq = calculate_degrees_freq(G)
    print(degrees_mean)





