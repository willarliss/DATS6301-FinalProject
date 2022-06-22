from typing import List, Any

from numpy import ndarray
from pandas import DataFrame
from sklearn.utils import Bunch
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.base import TransformerMixin, BaseEstimator


class VotingClassifier2(VotingClassifier):
    """Additional functionality for VotingClassifier to allow using pre-fit models.
    """

    def __init__(self, estimators: list, *,
                 voting: str = 'soft',
                 weights: ndarray = None,
                 n_jobs: int = None,
                 flatten_transform: bool = True,
                 verbose: int = False):

        super().__init__(
            estimators=estimators,
            voting=voting,
            weights=weights,
            n_jobs=n_jobs,
            flatten_transform=flatten_transform,
            verbose=verbose,
        )

    def fit(self, X: Any = None, y: any = None, sample_weight: Any = None):

        self.named_estimators_ = Bunch(**dict(self.estimators))
        self.estimators_ = [e for n,e in self.estimators]
        self.le_ = LabelEncoder()
        self.le_.classes_ = self.estimators_[0].classes_

        return self


class NamedStandardScaler(TransformerMixin, BaseEstimator):
    """Additional functionality for StandardScaler to limit transformation to subset of columns.
    """

    def __init__(self,
                 copy: bool = True,
                 with_mean: bool = True,
                 with_std: bool = True,
                 names: List[str] = None):

        self.copy = copy
        self.with_mean = with_mean
        self.with_std = with_std
        self.names = names

        self.scaler = StandardScaler(
            copy=True,
            with_mean=self.with_mean,
            with_std=self.with_std,
        )

    def fit(self, X: DataFrame, y: Any = None):

        if not isinstance(X, DataFrame):
            raise ValueError

        if self.names is None:
            self.names_ = X.columns.tolist()
        else:
            self.names_ = list(self.names)

        self.scaler.fit(X[self.names_])

        return self

    def partial_fit(self, X: DataFrame, y: Any = None):

        if not isinstance(X, DataFrame):
            raise ValueError

        if not hasattr(self, 'names_'):
            if self.names is None:
                self.names_ = X.columns.tolist()
            else:
                self.names_ = list(self.names)

        self.scaler.partial_fit(X[self.names_])

        return self

    def transform(self, X: DataFrame) -> DataFrame:

        if not isinstance(X, DataFrame):
            raise ValueError

        Xt = X.copy() if self.copy else X
        Xt[self.names_] = self.scaler.transform(Xt[self.names_])

        return Xt
