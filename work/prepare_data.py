import os

import pandas as pd
import networkx as nx

SEED = 0


def extract(dir_path, sample=0.1, seed=None):
    """Walk the directory, extract csv files, store a sample, apply some basic preprocessing.
    """

    dtype_map = {'dest_ip': str, 'dest_port': str, 'scr_ip': str, 'src_port': str}
    raw_data = []

    for path, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.csv'):

                df = pd.read_csv(os.path.join(path, file), dtype=dtype_map)
                df = df[df['label']!='outlier'].dropna(inplace=False)
                df = df.sample(frac=sample, replace=False, random_state=seed)
                df['src'] = df['src_ip'].astype(str) + ':' + df['src_port'].astype(str)
                df['dst'] = df['dest_ip'].astype(str) + ':' + df['dest_port'].astype(str)
                df['label'] = df['label'].map({'benign': 0, 'malicious': 1}).astype(int)

                raw_data.append(df)

    return pd.concat(raw_data, ignore_index=True, axis=0)


def build_graph(df):
    """Build directed graph from src and dst columns with other features as attributes.
    """

    G = nx.MultiDiGraph()
    features = ['label', 'avg_ipt', 'bytes_in', 'bytes_out', 'entropy', 'num_pkts_out',
                'num_pkts_in', 'proto', 'time_end', 'time_start', 'total_entropy', 'duration']

    for _, row in df.iterrows():

        u, v = row['src'], row['dst']

        attrs = dict(zip(['ip', 'port'], u.split(':')))
        G.add_node(u, **attrs)

        attrs = dict(zip( ['ip', 'port'], v.split(':') ))
        G.add_node(v, **attrs)

        attrs = row[features].to_dict()
        G.add_edge(u, v, **attrs)

    return G


if __name__ == '__main__':
    """Download and unzip data from LUFlow Network Intrusion Detection Data Set on Kaggle:
    https://www.kaggle.com/datasets/mryanm/luflow-network-intrusion-detection-data-set?select=2020
    """

    df_train = extract('../data/archive/2020', sample=0.01, seed=SEED)
    G_train = build_graph(df_train)
    df_train.to_csv('../data/train.csv.gz', header=True, index=False, compression='gzip')
    nx.write_gml(G_train, '../data/train_graph.gml.gz')
    del df_train, G_train

    df_test = extract('../data/archive/2021', sample=0.01, seed=SEED)
    G_test = build_graph(df_test)
    df_test.to_csv('../data/test.csv.gz', header=True, index=False, compression='gzip')
    nx.write_gml(G_test, '../data/test_graph.gml.gz')
    del df_test, G_test
