import joblib

from utils.objects import VotingClassifier2

RUN = 3

if __name__ == '__main__':
    """Create soft voting ensemble from pre-trained models.
    """

    models = [
        ('mod_1_roc', joblib.load(filename='./artifacts/model_1.joblib')),
        ('mod_2_recall', joblib.load(filename='./artifacts/model_2.joblib')),
    ]

    clf = VotingClassifier2(
        estimators=models,
        voting='soft',
        weights=(1., 2.),
    ).fit()

    joblib.dump(clf, f'./artifacts/model_{RUN}.joblib')
