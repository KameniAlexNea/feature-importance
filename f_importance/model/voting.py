import collections

import numpy as np


class Voting:
    """
    The base class for creating a voting ensemble model. It has a `predict` method that aggregates the predictions
    of multiple models.
    """

    def __init__(self, models) -> None:
        self.models = models

    def predict(self, X):
        res = [model.predict(X).reshape(len(X), -1) for model in self.models]
        return res


class VotingClassifier(Voting):
    """
    A class for creating a voting ensemble classifier. It aggregates the predictions of multiple classifiers by
    taking a weighted average of their predictions.
    """

    def __init__(self, models, weights: np.ndarray) -> None:
        """
        Initializes the VotingClassifier class.

        :param models: a list of scikit-learn like models to be used for voting.
        :type models: List[Model]
        :param weights: an array of weights for each model. The shape of the array should be (n_models,).
        :type weights: np.ndarray
        """
        super().__init__(models)
        self.weights = weights

    def predict(self, X):
        """
        Makes predictions for the input `X` using the list of models passed during initialization.

        :param X: input features
        :type X: an array of predictions of shape (n_samples, n_classes)
        """

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
    """
    A class for creating a voting ensemble regressor. It aggregates the predictions of multiple regressors by
    taking a weighted average of their predictions.
    """

    def __init__(self, models, weights: np.ndarray) -> None:
        """
        Initializes the VotingRegressor class

        :param models: a list of scikit-learn like models to be used for voting.
        :type models: List[Model]
        :param weights: an array of weights for each model. The shape of models
        :type weights: np.ndarray
        """
        super().__init__(models)
        self.weights = weights

    def predict(self, X):
        """
        Makes predictions for the input `X` using the list of models passed during initialization.

        :param X: input features
        :type X: an array of predictions of shape (n_samples, n_classes)
        """
        preds = super().predict(X)
        preds = [
            (
                np.array([pred[:, i] for pred in preds]).T
                * self.weights.reshape((1, -1))
            ).sum(axis=1)
            for i in range(preds[0].shape[1])
        ]
        return np.array(preds).T
