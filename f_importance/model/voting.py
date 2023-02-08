import collections

import numpy as np
import pandas as pd


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
            pd.DataFrame(np.array([pred[:, i] for pred in preds]))
            .apply(_apply_row, axis=0)
            .values
            for i in range(preds[0].shape[1])
        ]
        return np.array(res)


class VotingRegressor(Voting):
    def __init__(self, models, weights: np.ndarray) -> None:
        super().__init__(models)
        self.weights = weights

    def predict(self, X):
        preds = super().predict(X)
        preds = preds * self.weights.reshape((1, -1))
        return preds.mean(axis=1)
