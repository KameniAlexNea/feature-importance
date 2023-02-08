import collections

import numpy as np


class Voting:
    def __init__(self, models) -> None:
        self.models = models
    
    def predict(self, X):
        return np.array([
            model.predict(X) for model in self.models
        ]).T
    
    def predict_proba(self, X):
        return np.array([
            model.predict_proba(X)[:, -1] for model in self.models
        ]).T


class VotingClassifier(Voting):
    def __init__(self, models, weights: np.ndarray) -> None:
        super().__init__(models)
        self.weights = weights
    
    def predict(self, X):
        def _apply_row(row):
            data = collections.defaultdict(float)
            for clazz, weight in zip(row, self.weights):
                data[clazz] += weight
            return max(data.keys(), key=lambda x: data[x])
        preds = super().predict(X)
        return np.apply_along_axis(_apply_row, 1, preds)
    
    def predict_proba(self, X):
        preds = super().predict_proba(X)
        preds = preds * self.weights.reshape((1, -1))
        return preds.mean(axis=1)


class VotingRegressor(Voting):
    def __init__(self, models, weights: np.ndarray) -> None:
        super().__init__(models)
        self.weights = weights
    
    def predict(self, X):
        preds = super().predict(X)
        preds = preds * self.weights.reshape((1, -1))
        return preds.mean(axis=1)
    
    def predict_proba(self, X):
        raise NotImplementedError("cannot predict proba for regression")