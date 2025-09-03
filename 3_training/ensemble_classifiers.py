import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin, clone


class StackingClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, estimators):
        self.estimators = estimators
        self.meta_model = LogisticRegression(
            class_weight={0: 1, 1: 5},
            penalty=None
        )

    def fit(self, X, y):
        self.estimators_ = [clone(est) for est in self.estimators]
        for estimator in self.estimators_:
            estimator.fit(X, y)
        meta_features = np.column_stack([estimator.predict_proba(X)[:, 1] for estimator in self.estimators_])
        self.meta_model_ = clone(self.meta_model)
        self.meta_model_.fit(meta_features, y)
        return self

    def predict(self, X):
        base_predictions = np.column_stack([estimator.predict_proba(X)[:, 1] for estimator in self.estimators_])
        return self.meta_model_.predict(base_predictions)

    def get_params(self, deep=True):
        params = {
            "estimators": self.estimators,
            "meta_model": self.meta_model,
        }
        return params

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


class VotingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimators):
        self.estimators = estimators

    def fit(self, X, y):
        self.estimators_ = [clone(est) for est in self.estimators]
        for estimator in self.estimators_:
            estimator.fit(X, y)
        return self

    def predict(self, X):
        probabilities = np.column_stack([model.predict_proba(X)[:, 1] for model in self.estimators_])
        avg_probabilities = np.mean(probabilities, axis=1)
        return (avg_probabilities >= 0.5).astype(int)

    def get_params(self, deep=True):
        params = {"estimators": self.estimators}
        return params

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
