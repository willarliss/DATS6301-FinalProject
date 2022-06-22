import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

SEED = 0


def plot_graph(G, return_fig=False, figsize=(10,10), title=None, **kwargs):
    """Plot 2D representation of graph using Kamada-Kawai algorithm.
    """

    pos = nx.kamada_kawai_layout(G, **kwargs)
    edges_0 = [e[:-1] for e in G.edges(data='label') if e[-1]=='0']
    edges_1 = [e[:-1] for e in G.edges(data='label') if e[-1]=='1']

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    ax.set_title('Malicious (red) and Benign (blue) Network Flows' if title is None else str(title))

    nx.draw_networkx(G=G, pos=pos, ax=ax, edgelist=edges_1,
                     width=0.2, alpha=0.5, with_labels=False,
                     arrows=False, nodelist=[], edge_color='r')
    nx.draw_networkx(G=G, pos=pos, ax=ax, edgelist=edges_0,
                     width=0.2, alpha=0.5, with_labels=False,
                     arrows=False, nodelist=[], edge_color='b')
    nx.draw_networkx(G=G, pos=pos, ax=ax,
                     node_size=50, alpha=0.55, with_labels=False,
                     edgelist=[], linewidths=0., node_color='k')

    if return_fig:
        return fig
    return None


def plot_centrality(G, figsize=(14,4), bins=15, density=False, log=False, return_fig=False):
    """Plot histograms of (in/out/total) node centrality.
    """

    eps = 1e-8
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=figsize)

    centrality = np.array(list(nx.degree_centrality(G).values()))
    mean = centrality.mean().round(5)
    if log:
        ln_centrality = np.log(centrality+eps)
        lmean = ln_centrality.mean().round(5)
        ax1.hist(ln_centrality, bins=bins, density=density, label=f'mean={mean} \nln(mean)={lmean}')
        ax1.set_title('Logged Degree Centrality')
    else:
        ax1.hist(centrality, bins=bins, density=density, label=f'mean={mean}')
        ax1.set_title('Degree Centrality')
    ax1.legend()

    centrality = np.array(list(nx.in_degree_centrality(G).values()))
    mean = centrality.mean().round(5)
    if log:
        ln_centrality = np.log(centrality+eps)
        lmean = ln_centrality.mean().round(5)
        ax2.hist(ln_centrality, bins=bins, density=density, label=f'mean={mean} \nln(mean)={lmean}')
        ax2.set_title('Logged In-Degree Centrality')
    else:
        ax2.hist(centrality, bins=bins, density=density, label=f'mean={mean}')
        ax2.set_title('In-Degree Centrality')
    ax2.legend()

    centrality = np.array(list(nx.out_degree_centrality(G).values()))
    mean = centrality.mean().round(5)
    if log:
        ln_centrality = np.log(centrality+eps)
        lmean = ln_centrality.mean().round(5)
        ax3.hist(ln_centrality, bins=bins, density=density, label=f'mean={mean} \nln(mean)={lmean}')
        ax3.set_title('Logged In-Degree Centrality')
    else:
        ax3.hist(centrality, bins=bins, density=density, label=f'mean={mean}')
        ax3.set_title('Out-Degree Centrality')
    ax3.legend()

    if return_fig:
        return fig
    return None


def plot_closeness(G, figsize=(6,4), log=False, bins=15, density=False, return_fig=False):
    """Plot histogram of closeness centrality.
    """

    eps = 1e-8
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    closeness = np.array(list(dict(nx.closeness_centrality(G)).values()))
    mean = closeness.mean().round(5)

    if log:
        ln_closeness = np.log(closeness+eps)
        lmean = ln_closeness.mean().round(5)
        ax.hist(ln_closeness, bins=bins, density=density, label=f'mean={mean} \nln(mean)={lmean}')
        ax.set_title('Logged Closeness Centrality')
    else:
        ax.hist(closeness, bins=bins, density=density, label=f'mean={mean}')
        ax.set_title('Closeness Centrality')
    ax.legend()

    if return_fig:
        return fig
    return None


def plot_betweenness(G, figsize=(11,4), bins=15, density=False, log=False, return_fig=False):
    """Plot histograms of (node/edge) betweenness centrality.
    """

    eps = 1e-8
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=figsize)

    betweenness = np.array(list(nx.betweenness_centrality(G).values()))
    mean = betweenness.mean().round(3)
    if log:
        ln_betweenness = np.log(betweenness+eps)
        lmean = ln_betweenness.mean().round(3)
        ax1.hist(ln_betweenness, bins=bins, density=density, label=f'mean={mean} \nln(mean)={lmean}')
        ax1.set_title('Logged Node-Betweenness Centrality')
    else:
        ax1.hist(betweenness, bins=bins, density=density, label=f'mean={mean}')
        ax1.set_title('Node-Betweenness Centrality')
    ax1.legend()

    betweenness = np.array(list(nx.in_degree_centrality(G).values()))
    mean = betweenness.mean().round(3)
    if log:
        ln_betweenness = np.log(betweenness+eps)
        lmean = ln_betweenness.mean().round(3)
        ax2.hist(ln_betweenness, bins=bins, density=density, label=f'mean={mean} \nln(mean)={lmean}')
        ax2.set_title('Logged Edge-Betweenness Centrality')
    else:
        ax2.hist(betweenness, bins=bins, density=density, label=f'mean={mean}')
        ax1.set_title('Edge-Betweenness Centrality')
    ax2.legend()

    if return_fig:
        return fig
    return None


if __name__ == '__main__':
    """Exploratory data analysis on graph data.
    """

    graph = nx.read_gml('../data/train_graph.gml.gz')

    print('nodes:', nx.number_of_nodes(graph))
    print('edges:', nx.number_of_edges(graph))
    print('connections:', nx.number_of_edges(nx.Graph(graph)))
    print('density:', nx.density(graph))

    plot_centrality(graph, log=True, bins=20, return_fig=True)
    plt.savefig('./artifacts/eda/graph_centrality.png')

    subsample_size = 5000
    edges = list(graph.edges)
    idx = np.random.default_rng(SEED).choice(len(edges), size=subsample_size, replace=False)
    nodes = np.array([v[:-1] for i,v in enumerate(edges) if i in idx]).flatten()
    graph_sub = graph.subgraph(set(nodes))

    plot_closeness(graph_sub, log=True, bins=20, return_fig=True)
    plt.savefig('./artifacts/eda/graph_closeness.png')

    plot_betweenness(graph_sub, log=True, bins=20, return_fig=True)
    plt.savefig('./artifacts/eda/graph_betweenness.png')

    gplot = plot_graph(graph_sub, return_fig=True, figsize=(11,9), scale=2.)
    plt.savefig('./artifacts/eda/graph_layout.png')
    # gplot = plot_graph(graph_sub, return_fig=True, figsize=(11,9), scale=1.5)
    # plt.savefig('./artifacts/eda/graph_layout_large.png')
