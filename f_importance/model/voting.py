import collections

import numpy as np


class Voting:
    def __init__(self, models) -> None:
        self.models = models

    def predict(self, X):
        res = [model.predict(X).reshape(len(X), -1) for model in self.models]
        return res


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
        res = [
            np.apply_along_axis(
                _apply_row, 1, np.array([pred[:, i] for pred in preds]).T
            )
            for i in range(preds[0].shape[1])
        ]
        res = np.array(res).T
        return res


class VotingRegressor(Voting):
    def __init__(self, models, weights: np.ndarray) -> None:
        super().__init__(models)
        self.weights = weights

    def predict(self, X):
        preds = super().predict(X)
        preds = [
            (
                np.array([pred[:, i] for pred in preds]).T
                * self.weights.reshape((1, -1))
            ).sum(axis=1)
            for i in range(preds[0].shape[1])
        ]
        return np.array(preds).T
