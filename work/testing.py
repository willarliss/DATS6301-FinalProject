import json

import joblib
import pandas as pd

from utils import plot_cm, plot_roc, plot_hist


def process_inputs(df):

    features = ['avg_ipt', 'bytes_in', 'bytes_out', 'entropy', 'num_pkts_out', 'num_pkts_in', 'proto',
                'total_entropy', 'duration', 'duration_computed', 'dayofweek', 'timeofday']
    required_columns = ['avg_ipt', 'bytes_in', 'bytes_out', 'entropy', 'num_pkts_out',
                        'num_pkts_in', 'total_entropy', 'duration', 'duration_computed',
                        'timeofday', 'dayofweek_0', 'dayofweek_1', 'dayofweek_2', 'dayofweek_3',
                        'dayofweek_4', 'dayofweek_5', 'dayofweek_6', 'proto_1', 'proto_6',
                        'proto_17', 'proto_58']

    df['time_end_dt'] = pd.to_datetime(df['time_end']/1e6, unit='s')
    df['time_start_dt'] = pd.to_datetime(df['time_start']/1e6, unit='s')
    df['dayofweek'] = df['time_start_dt'].dt.dayofweek.astype(str)
    df['timeofday'] = df['time_start_dt'].dt.time.apply(lambda x: (x.hour*3600 + x.minute*60 + x.second)/86400)
    df['duration_computed'] = (df['time_end']-df['time_start'])/1e6

    X, y = (
        pd.get_dummies(df[features], columns=['dayofweek', 'proto']),
        df['label'].values,
    )

    for col in set(required_columns)-set(X.columns):
        X[col] = 0

    return X[required_columns], y


if __name__ == '__main__':

    data = pd.read_csv('../data/test.csv.gz')
    X_test, y_test = process_inputs(data)
    del data

    clf = joblib.load('./artifacts/model.joblib')

