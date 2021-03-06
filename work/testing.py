import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, classification_report, recall_score

from utils.plot import plot_cm, plot_roc, plot_hist

RUN = 3


def process_inputs(df):
    """Process raw data for input to pipeline. Copied from `training.py` with one adjustment to
    validate that the proper columns are present.
    """

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
    """Test a model on the holdout set. Plot some metrics.
    """

    data = pd.read_csv('../data/test.csv.gz')
    X_test, y_test = process_inputs(data)
    del data

    clf = joblib.load(f'./artifacts/model_{RUN}.joblib')

    y_proba = clf.predict_proba(X_test)[:,1]
    try:
        y_score = clf.decision_function(X_test)
    except AttributeError:
        y_score = np.log(y_proba/(1-y_proba))
    y_pred = clf.predict(X_test)

    fig = plot_hist(y_test, y_score, title=f'Scoring Histogram {RUN}', return_fig=True)
    plt.savefig(f'./artifacts/eval/scoring_hist_{RUN}.png')

    fig, thresh = plot_roc(y_test, y_proba, title=f'ROC Curve {RUN}', return_fig=True,  return_t=True)
    plt.savefig(f'./artifacts/eval/roc_curve_{RUN}.png')

    fig = plot_cm(y_test, y_pred, title=f'Confusion Matrix {RUN}', return_fig=True)
    plt.savefig(f'./artifacts/eval/confusion_matrix_{RUN}.png')

    auc = roc_auc_score(y_true=y_test, y_score=y_proba)
    acc = (y_test==y_pred).mean()
    f1 = f1_score(y_true=y_test, y_pred=y_pred)
    rec = recall_score(y_true=y_test, y_pred=y_pred)
    print('Accuracy:', acc)
    print('ROC-AUC:', auc)
    print('Recall:', rec)
    print('F1:', f1)

    report = classification_report(
        y_true=y_test,
        y_pred=y_pred,
        target_names=('benign_class', 'malicious_class'),
        output_dict=True,
    )
    report['auc'] = auc
    report['recall'] = rec

    with open(f'./artifacts/eval/report_{RUN}.json', 'w') as outfile:
        outfile.write(json.dumps(report, indent=2))
