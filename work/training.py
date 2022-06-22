import json

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier

from utils.objects import NamedStandardScaler

SEED = 0
RUN = 2


def run_gridsearch(X, y, estimators, param_grids):
    """Run grid-search over multiple estimators.
    """

    est_names = list(estimators.keys())
    assert set(est_names) == set(param_grids.keys())

    results = []

    for name in est_names:
        print('Gridsearching:', name)

        search = GridSearchCV(
            estimator=estimators[name],
            param_grid=param_grids[name],
            cv=3,
            refit=False,
            return_train_score=False,
            #scoring='roc_auc',
            scoring='recall',
            n_jobs=7,
            verbose=2,
        )

        search.fit(X, y)

        res = {
            'algorithm': name,
            'best_params': search.best_params_,
            'best_score': float(search.best_score_),
        }

        results.append(res)
        with open(f'./artifacts/gridsearch_{RUN}.json', 'a') as outfile:
            outfile.write(json.dumps(res)+'\n')

    return results


def process_inputs(df):
    """Process raw data for input to pipeline.
    """

    features = ['avg_ipt', 'bytes_in', 'bytes_out', 'entropy', 'num_pkts_out', 'num_pkts_in', 'proto',
                'total_entropy', 'duration', 'duration_computed', 'dayofweek', 'timeofday']

    df['time_end_dt'] = pd.to_datetime(df['time_end']/1e6, unit='s')
    df['time_start_dt'] = pd.to_datetime(df['time_start']/1e6, unit='s')
    df['dayofweek'] = df['time_start_dt'].dt.dayofweek.astype(str)
    df['timeofday'] = df['time_start_dt'].dt.time.apply(lambda x: (x.hour*3600 + x.minute*60 + x.second)/86400)
    df['duration_computed'] = (df['time_end']-df['time_start'])/1e6

    X, y = (
        pd.get_dummies(df[features], columns=['dayofweek', 'proto']),
        df['label'].values,
    )

    return X, y


if __name__ == '__main__':
    """Train/tune models. Store best model and grid-search results.
    """

    data = pd.read_csv('../data/train.csv.gz')
    X_train, y_train = process_inputs(data)
    del data

    models = {
        'bagged': RandomForestClassifier(
            random_state=SEED,
            n_estimators=100,
        ),
        'adaboost': AdaBoostClassifier(
            base_estimator=DecisionTreeClassifier(
                random_state=SEED,
            ),
            n_estimators=100,
            algorithm='SAMME.R'
        ),
        'gradboost': Pipeline([
            ('sc', NamedStandardScaler(
                names=[c for c in X_train.columns if not (c.startswith('proto_') or c.startswith('dayofweek_'))],
            )),
            ('clf', GradientBoostingClassifier(
                random_state=SEED,
                n_estimators=100,
                loss='log_loss',
            )),
        ]),
    }

    # parameters = {
    #     'bagged': {
    #         'max_depth': [None, 8, 6, 4, 2],
    #         'criterion': ['gini', 'entropy'],
    #         'max_features': [1.0, 0.83, 0.67, 0.5],
    #         'max_samples': [1.0, 0.75, 0.5, 0.25],
    #         'class_weight': ['balanced', None, 'balanced_subsample'],
    #     },
    #     'adaboost': {
    #         'learning_rate': [1.0, 0.1, 0.01, 0.001],
    #         'base_estimator__criterion': ['gini', 'entropy'],
    #         'base_estimator__class_weight': ['balanced', None],
    #         'base_estimator__max_features': [1.0, 0.83, 0.67, 0.5],
    #         'base_estimator__max_depth': [None, 8, 6, 4, 2],
    #     },
    #     'gradboost': {
    #         'clf__learning_rate': [1.0, 0.1, 0.01, 0.001],
    #         'clf__subsample': [1., 0.75, 0.5, 0.25],
    #         'clf__max_depth': [None, 8, 6, 4, 2],
    #         'clf__max_features': [1.0, 0.83, 0.67, 0.5],
    #         'clf__criterion': ['friedman_mse', 'squared_error'],
    #     },
    # }
    parameters = {
        'bagged': {
            'max_depth': [8, 7, 6, 5, 4],
            'class_weight': ['balanced', None],
        },
        'adaboost': {
            'learning_rate': [1.0, 0.1, 0.01, 0.001],
            'base_estimator__class_weight': ['balanced', None],
            'base_estimator__max_depth': [8, 7, 6, 5, 4],
        },
        'gradboost': {
            'clf__learning_rate': [1.0, 0.1, 0.01, 0.001],
            'clf__max_depth': [8, 7, 6, 5, 4],
        },
    }

    gs_results = run_gridsearch(
        X=X_train,
        y=y_train,
        estimators=models,
        param_grids=parameters,
    )

    best_result = max(gs_results, key=lambda x: x['best_score'])

    print('Training:', best_result['algorithm'])
    clf = models[best_result['algorithm']]
    clf.set_params(**best_result['best_params'])
    clf.fit(X_train, y_train)

    joblib.dump(clf, f'./artifacts/model_{RUN}.joblib')
