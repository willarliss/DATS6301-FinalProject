import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt

SEED = 0


def feature_histogram(data, target, title=None, log=False, figsize=(15,5), return_fig=False):

    if log:
        data -= data.min()
        data = np.log(data+1)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=figsize)

    ax1.hist(data, bins=20, alpha=0.9, density=False, label=col)
    ax1.legend()
    ax1.set_ylabel('Count')
    ax1.set_xlabel('Domain')

    ax2.hist(data[target==1], color='r', alpha=0.5, bins=25, density=True, label='malicious')
    ax2.hist(data[target==0], color='b', alpha=0.5, bins=25, density=True, label='benign')
    ax2.legend()
    ax2.set_ylabel('Density')
    ax2.set_xlabel('Domain')

    fig.suptitle('Histograms' if title is None else str(title))

    if return_fig:
        return fig
    return None


def plot_graph(G, return_fig=False, figsize=(10,10), title=None):

    pos = nx.kamada_kawai_layout(G)
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


if __name__ == '__main__':

    dtype_map = {'dest_ip': str, 'dest_port': str, 'scr_ip': str, 'src_port': str}
    df = pd.read_csv('../data/train.csv.gz', dtype=dtype_map)

    df['time_end_dt'] = pd.to_datetime(df['time_end']/1e6, unit='s')
    df['time_start_dt'] = pd.to_datetime(df['time_start']/1e6, unit='s')

    df['dayofweek'] = df['time_start_dt'].dt.dayofweek.astype(str)
    df['timeofday'] = df['time_start_dt'].dt.time.apply(lambda x: (x.hour*3600 + x.minute*60 + x.second)/86400)

    df['duration_computed'] = (df['time_end']-df['time_start'])/1e6

    print(df.shape, df.columns.tolist())
    print(df.describe().T)
    df.info()
    print()

    for col in df.columns:
        print(df[col].nunique())
        print(df[col].value_counts().reset_index()[:10])
        print()

    features = ['avg_ipt', 'bytes_in', 'bytes_out', 'entropy', 'num_pkts_out', 'num_pkts_in', 'proto',
                'total_entropy', 'duration', 'duration_computed', 'dayofweek', 'timeofday']

    X_train = pd.get_dummies(df[features], columns=['dayofweek', 'proto'])
    y_train = df['label']
    print(X_train.shape)
    print()

    for col in X_train.columns:
        if col.startswith('proto_'):
            continue
        if col.startswith('dayofweek_'):
            continue
        hist = feature_histogram(X_train[col].values, y_train, title=f'Histogram of logged {col}', log=True, return_fig=True)
        plt.savefig(f'./artifacts/eda/{col}_hist.png')
        plt.close()

    freq = df.groupby('label')['dayofweek'].value_counts().rename('Count').to_frame().reset_index()
    plt.figure(figsize=(7,5))
    sns.catplot(data=freq, x='dayofweek', y='Count', hue='label', kind='bar')
    plt.savefig('./artifacts/eda/dayofweek_bar.png')
    plt.close()

    freq = df.groupby('label')['proto'].value_counts().rename('Count').to_frame().reset_index()
    plt.figure(figsize=(7,5))
    sns.catplot(data=freq, x='proto', y='Count', hue='label', kind='bar')
    plt.savefig('./artifacts/eda/proto_bar.png')
    plt.close()

    graph = nx.read_gml('../data/train_graph.gml.gz')
    print(len(graph.nodes), 'nodes', len(graph.edges), 'edges'))

    subsample_size = 1000
    edges = list(graph.edges)
    idx = np.random.default_rng(SEED).choice(len(edges), size=subsample_size, replace=False)
    nodes = np.array([v[:-1] for i,v in enumerate(edges) if i in idx]).flatten()
    graph_sub = graph.subgraph(set(nodes))

    gplot = plot_graph(graph_sub, return_fig=True)
    plt.savefig('./artifacts/eda/graph_layout.png')
