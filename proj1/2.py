#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# In[8]:


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


# In[58]:


G = nx.gnm_random_graph(n=4, m=2)
draw_graph(G)


# In[4]:


l = len(G.nodes)
degrees_freq = np.zeros((l, ), dtype=int)
for node in G.nodes:
    node_degree = G.degree[node]
    degrees_freq[node_degree] += 1


# In[5]:


degrees_freq


# In[6]:


plt.plot(range(degrees_freq.shape[0]), degrees_freq/l, '-')
plt.show()


# In[7]:


def init_ax(subname, x_low=-1, x_high=+1, y_low=-1, y_high=+1):
    fig_dim = 10
    # size of figure
    x_step = (x_high - x_low) / 10
    y_step = (y_high - y_low) / 10
    # distance between ticks in figure

    _, ax = plt.subplots(figsize=(fig_dim, fig_dim))
    # create a figure

    ax.set_xlabel("x")
    ax.set_ylabel("y")
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


# In[8]:


# G.degree


# In[9]:


ax = init_ax('subproblem 2',
                x_low=0,
                x_high=l,
                y_low=0,
                y_high=1
            )
ax.plot(degrees_freq/l, color='black')

