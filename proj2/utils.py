import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.stats import pareto, expon
from tqdm import tqdm


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
    Checks if given 'G' matrix is irreducable.

    Args:
        G (networkx.Graph)
    """
    return nx.algorithms.all_pairs_node_connectivity(G)

def is_irreducable_2(G):
    n = G.number_of_nodes
    mat = np.zeros((n,n))
    for i, node1 in enumerate(G.nodes):
        for j, node2 in enumerate(G.nodes):
            mat[i,j] =  nx.node_connectivity(node1, node2)
    
    expected = np.ones((n,n)) * (-2) + 1
    return expected == mat


graph_providers = {
    'ER': lambda n, p: nx.erdos_renyi_graph(n=n, p=p),  # ok
    'WS': lambda n, p: nx.newman_watts_strogatz_graph(n=n, k=int(0.1 * n), p=p),
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


def calculate_survival_time(n, l, s=None,
                            p_list=np.linspace(0.1, 1, 10, dtype=float),
                            # p_list=[0.5],
                            mode='Static',
                            number_of_simulations=100):
    print("A) calculating survival time")
    df = None
    for p in tqdm(p_list):
        k_Et_dict = {}
        for _ in range(number_of_simulations):
            k_list = []
            graph = graph_providers["ER"](n, p)
            remove_nodes_priority = [(x, y)
                                     for x, y in sorted(zip(l, graph.nodes))]
            for node in remove_nodes_priority:
                graph.remove_node(node[1])
                if (graph.number_of_nodes() == 0 or not nx.is_connected(graph)):
                    break
                k = (graph.number_of_edges() * 2) // graph.number_of_nodes()
                k_list.append(k)
            # calculate how long it takes from each <k> for a graph to become disconnected
            k_prev = None
            k_prev_counter = 0
            k_prev_sum = 0
            for i in range(len(k_list)):
                k = k_list[i]
                curr_time = remove_nodes_priority[-i][0]
                # print(k, curr_time)
                # print(f'k: {k},\tk_prev: {k_prev},\tk_prev_counter: {k_prev_counter},\tk_prev_sum: {k_prev_sum},\tk_list_size: {k_list_size}')
                if (k != k_prev):
                    if k_prev is None:
                        k_prev = k
                        k_prev_counter = 1
                        k_prev_sum = curr_time
                    else:
                        if k_prev in k_Et_dict.keys():
                            k_Et_dict[k_prev] += k_prev_sum / k_prev_counter
                        else:
                            k_Et_dict[k_prev] = k_prev_sum / k_prev_counter
                        # print(f'k_prev: {k_prev},\tk_Et_dict[k_prev]: {k_Et_dict[k_prev]}')
                        k_prev = k
                        k_prev_counter = 1
                        k_prev_sum = curr_time
                else:
                    k_prev_counter += 1
                    k_prev_sum += curr_time
        if len(k_Et_dict) == 0:
            continue
        lists = sorted(k_Et_dict.items())
        x, y = zip(*lists)
        if df is None:
            df = pd.DataFrame({'k': x, 'E_t': np.array(
                y)/number_of_simulations, 'p': [p for _ in range(len(x))]})
        else:
            _ = pd.DataFrame({'k': x, 'E_t': np.array(
                y)/number_of_simulations, 'p': [p for _ in range(len(x))]})
            df = pd.concat([df, _], ignore_index=True)
    
    print("done")
    df.to_csv('simulation_output_survival_time.csv', index=False)
    fig = px.line(df, x='k', y='E_t', color='p')
    # fig.show()
    fig.write_html("simulation_survival_time.html")

def simulate_survival_by_search(n, number_of_simulations=100, pareto_scale=1, pareto_shape = 3, expo_lambda = 2, number_of_plotting_points=100):
    _expo = None
    _pareto = None
    s_mean_list = [[], []]
    t_mean_list = [[], []]
    for distro in ['expon', 'pareto']:
        for multiply in tqdm(np.linspace(0.1, 100, 10), position=1):
            s_mean = []
            t_mean = []
            for _ in tqdm(range(number_of_simulations), position=0, ascii=True):
                s_list = []
                t_list = []
                graph = nx.random_regular_graph(d=10, n=n)
                l = None
                events = None
                if distro == 'expon':
                    l = [(x, y, z) for x, y, z in sorted(zip(np.random.exponential(scale=(1/expo_lambda)*multiply, size=n), np.arange(n), ['l' for _ in np.arange(n)]))]
                else:
                    l = [(x, y, z) for x, y, z in sorted(zip(np.random.pareto(pareto_scale, size=n) * pareto_shape*multiply/100, np.arange(n), ['l' for _ in np.arange(n)]))]
                events = l

                for event in events:
                    if event[2] == 'l':
                        neighbors = graph.neighbors(event[1])
                        graph.remove_node(event[1])
                        for neighbor in neighbors:
                            if len(list(graph.neighbors(neighbor))) == 0:
                                t_list.append(event[0])
                                # if np.random.random() > 0.5:
                                s_ = None
                                if distro == 'expon':
                                    s_ = np.random.exponential(scale=(1/expo_lambda) * multiply/10)
                                else:
                                    s_ = np.random.pareto(pareto_scale) * pareto_shape * multiply/100/10
                                events.append((s_, neighbor, 's'))
                                s_list.append(s_)
                    elif event[2] == 's':
                        if not graph.has_node(event[1]):
                            continue
                        non_neighbors = list(nx.non_neighbors(graph, event[1]))
                        if len(non_neighbors) == 0:
                            continue
                        to_add_node = np.random.choice(non_neighbors)
                        graph.add_edge(event[1], to_add_node)
                    events = sorted(events)

                s_mean.append(np.mean(s_list))
                t_mean.append(np.mean(t_list))
            if distro == 'expon':
                s_mean_list[0].append(np.mean(s_mean))
                t_mean_list[0].append(-np.mean(t_mean))
            else:
                s_mean_list[1].append(np.mean(s_mean))
                t_mean_list[1].append(-np.mean(t_mean))

    _expo = pd.DataFrame({'E[s]': s_mean_list[0], 'E[t]': t_mean_list[0]-2*np.min(t_mean_list[0]), 'distribution': 'pareto'})
    _pareto = pd.DataFrame({'E[s]': s_mean_list[1], 'E[t]': t_mean_list[1]-2*np.min(t_mean_list[1]), 'distribution': 'expon'})

    df = pd.concat([_expo, _pareto], ignore_index=True)
    df.to_csv('simulation_output_survival_by_search.csv', index=False)
    fig = px.scatter(df, x='E[s]', y='E[t]', color='distribution')
    # fig.show()
    fig.write_html("simulate_survival_by_search.html")

def simulate_isolation_by_search(n, number_of_simulations=100, pareto_scale=1, pareto_shape = 3, expo_lambda = 2, number_of_plotting_points=100):
    s_mean_list = [[], []]
    phi_mean_list = [[], []]
    for multiply in np.linspace(0.1, 100, number_of_plotting_points):
        s_mean = []
        phi_mean = []
        for _ in range(number_of_simulations):
            s_list = []
            phi_list = []
            graph = nx.random_regular_graph(d=10, n=n)
            l = None
            events = None
            l = [(x, y, z) for x, y, z in sorted(zip(np.random.exponential(scale=(1/expo_lambda)*multiply, size=n), np.arange(n), ['l' for _ in np.arange(n)]))]
            events = l

            for event in events:
                if event[2] == 'l':
                    neighbors = graph.neighbors(event[1])
                    graph.remove_node(event[1])
                    for neighbor in neighbors:
                        if len(list(graph.neighbors(neighbor))) == 0:
                            phi_list.append(event[0])
                            # if np.random.random() > 0.5:
                            s_ = None
                            s_ = np.random.exponential(scale=(1/expo_lambda) * multiply/10)
                            events.append((s_, neighbor, 's'))
                            s_list.append(s_)
                elif event[2] == 's':
                    if not graph.has_node(event[1]):
                        continue
                    non_neighbors = list(nx.non_neighbors(graph, event[1]))
                    if len(non_neighbors) == 0:
                        continue
                    to_add_node = np.random.choice(non_neighbors)
                    graph.add_edge(event[1], to_add_node)
                events = sorted(events)
                phi_list.append([d for _, d in graph.degree].count(0) / n)

            s_mean.append(np.mean(s_list))
            phi_mean.append(np.mean(phi_list))
        s_mean_list[0].append(np.mean(s_mean))
        phi_mean_list[0].append(np.mean(phi_mean))

    df = pd.DataFrame({'E[s]': s_mean_list[0], 'Phi': phi_mean_list[0], 'distribution': 'pareto'})

    df.to_csv('simulation_output_isolation_by_search.csv', index=False)
    fig = px.scatter(df, x='E[s]', y='Phi', color='distribution')
    # fig.show()
    fig.write_html("simulate_isolation_by_search.html")

def simulate_isolation_by_distribution(n, number_of_simulations=100, pareto_scale=2.62, number_of_plotting_points=100):
    print("D) calculating isolation by distro")
    expo_list = []
    expo_isolation_list = []
    pareto_list = []
    pareto_isolation_list = []
    for distro in [expon, pareto]:
        if distro.name == 'expon':
            p_list = [distro.pdf(x) for x in np.linspace(distro.ppf(
                0.01), distro.ppf(0.99), number_of_plotting_points)]
        else:
            p_list = [distro.pdf(x, pareto_scale) for x in np.linspace(distro.ppf(
                0.01, pareto_scale), distro.ppf(0.99, pareto_scale), number_of_plotting_points)]

        for p in tqdm(p_list):
            isolation_rate = 0
            for _ in range(number_of_simulations):
                graph = graph_providers["ER"](n, p)
                if nx.is_connected(graph):
                    isolation_rate += 1
            isolation_rate /= number_of_simulations
            if distro.name == 'expon':
                expo_list.append(p)
                expo_isolation_list.append(isolation_rate)
            else:
                pareto_list.append(p)
                pareto_isolation_list.append(isolation_rate)
    _expo = pd.DataFrame({'p': expo_list, 'isolation_rate': expo_isolation_list, 'distribution': [
                         'expo' for _ in range(number_of_plotting_points)]})
    _pareto = pd.DataFrame({'p': pareto_list, 'isolation_rate': pareto_isolation_list, 'distribution': [
                           'pareto' for _ in range(number_of_plotting_points)]})

    df = pd.concat([_expo, _pareto], ignore_index=True)
    
    print("done")
    df.to_csv('simulation_output_isolation_by_distribution.csv', index=False)
    fig = px.line(df, x='p', y='isolation_rate', color='distribution')
    #fig.show()
    fig.write_html("simulate_isolation_by_distribution.html")


def simulate_isolation_survival_by_pareto_shape(n, number_of_simulations=100, scale=1, number_of_plotting_points=100):
    print("E) calculating isolation survival  by pareto shape")
    shape_list = []
    p_list = []
    isolation_list = []
    survival_list = []

    for shape in tqdm(np.geomspace(0.01, 10, 100)):
        pareto_list = [(shape*pareto.pdf(x, scale)) for x in np.linspace(
            pareto.ppf(0.01, scale), pareto.ppf(0.99, scale), number_of_plotting_points)]
        survival_time = np.max(pareto_list)
        for p in pareto_list:
            isolation_rate = 0
            for _ in range(number_of_simulations):
                graph = graph_providers["ER"](n, p)
                if nx.is_connected(graph):
                    isolation_rate += 1
            isolation_rate /= number_of_simulations
            p_list.append(p)
            isolation_list.append(isolation_rate)
            shape_list.append(shape)
            survival_list.append(survival_time)

    df = pd.DataFrame({'p': p_list, 'isolation_rate': isolation_list,
                      'shape': shape_list, 'survival_time': survival_list})

    print("done")

    df.to_csv(
        'simulation_output_isolation_survival_by_pareto_shape.csv', index=False)
    fig = px.scatter_3d(df, x='p', y='isolation_rate',
                        z='survival_time', color='shape')
    #fig.show()
    fig.write_html("simulate_isolation_survival_by_pareto_shape.html")


    fig = px.line(df, x='p', y='isolation_rate', color='shape')
    fig.write_html("simulate_isolation_by_pareto_shape.html")


    fig = px.line(df, x='p', y='survival_time', color='shape')
    fig.write_html("simulate_survival_by_pareto_shape.html")
