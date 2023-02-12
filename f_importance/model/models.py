import collections
import os
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from concurrent.futures import wait
from typing import Union

import numpy as np
import pandas as pd
from sklearn.multioutput import ClassifierChain
from sklearn.multioutput import RegressorChain

from f_importance.dataset import data
from f_importance.metrics import METRICS
from f_importance.model import CLASSIFIERS
from f_importance.model import REGRESSORS
from f_importance.model.voting import VotingClassifier
from f_importance.model.voting import VotingRegressor


def _train_pred_evaluate(col: str, splits: list, model, metric, refit=True):
    """
    is a private function used to evaluate the performance of a model on a dataset.
    It trains the model and computes the prediction for each sample in the dataset.
    The function takes the following arguments:

    :param col: string, name of the column/feature in the dataset.
    :type col: str
    :param splits: list of tuples, each tuple consists of two arrays (X_train, y_train), (X_test, y_test)
    representing the train and test data for a specific fold in the cross-validation
    :type splits: list
    :param model: machine learning model.
    :type model: XGBoost
    :param metric: scoring function used to evaluate the performance of the model.
    :type metric: function
    :param refit: boolean, whether to refit the model for each fold, defaults to True
    :type refit: bool, optional
    :return: The function returns the name of the column/feature and a list of scores,
    each score is the evaluation result of the model on a fold in the cross-validation.
    :rtype: str, list[float]
    """
    scores = []
    for (X_train, y_train), (X_test, y_test) in splits:
        if refit:
            model.fit(X_train, y_train)
        preds = model.predict(X_test)
        score = metric(preds, y_test)
        scores.append(score)
    return col, scores


def _fit_voting_classifier(splits: list, model_name: str, is_m_output: bool, scorer):
    """
     is a private function used to fit a voting classifier/regressor on a dataset. It takes the following arguments:

    :param splits: list of tuples, each tuple consists of two arrays (X_train, y_train), (X_test, y_test)
    representing the train and test data for a specific fold in the cross-validation.
    :type splits: list
    :param model_name: string, name of the base model.
    :type model_name: str
    :param is_m_output: boolean, whether the model is a multi-output model.
    :type is_m_output: bool
    :param scorer: scoring function used to evaluate the performance of the model.
    :type scorer: function
    :return: The function returns the fitted voting classifier/regressor.
    :rtype: Voting Model
    """

    def _softmax(x, coef):
        x = np.array(x) * coef
        return np.exp(x) / (1 + np.exp(x))

    models = []
    scores = []
    is_classif = 1 if model_name in CLASSIFIERS else -1
    for (X_train, y_train), (X_test, y_test) in splits:
        model = get_model(model_name, is_m_output)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = scorer(y_pred, y_test)
        models.append(model)
        scores.append(score)
    clazz = VotingClassifier if is_classif == 1 else VotingRegressor
    model = clazz(models, weights=_softmax(scores, is_classif))
    return model


def get_model(model_name: str, is_m_output: bool):
    """
    is a function used to get the machine learning model. It takes the following arguments:

    :param model_name: string, name of the model.
    :type model_name: str
    :param is_m_output: boolean, whether the model is a multi-output model.
    :type is_m_output: bool
    :return: The function returns the machine learning model, which can be a single model or a chain of models.
    :rtype: XGBoost or Voting
    """
    base_model = (
        CLASSIFIERS[model_name]()
        if model_name in CLASSIFIERS
        else REGRESSORS[model_name]()
    )
    if is_m_output:
        base_model = (
            ClassifierChain(base_model)
            if model_name in CLASSIFIERS
            else RegressorChain(base_model)
        )
    return base_model


class Model:
    """
    is a class used to perform feature importance computation. It has the following attributes:

    _model: string, name of the model.

    _is_m_output: boolean, whether the model is a multi-output model.

    _dataset: Data object, either Data, DataFold, or DataSample, which stores the preprocessed dataset and the splits for cross-validation.

    _method: string, name of the method used for cross-validation, either "DataFold" or "DataSample".

    _metric: scoring function used to evaluate the performance of the
    """

    def __init__(
        self,
        model_name: str,
        method: str,
        metric: str,
        dataset: Union[str, pd.DataFrame],
        targets: Union[str, list],
        n_gram=(1, 1),
        val_rate=0.15,
        shuffle=True,
        n=5,
        is_regression=False,
        n_jobs=os.cpu_count(),
        refit=None,
    ) -> None:
        self._model = model_name
        self._is_m_output = (not isinstance(targets, str)) and (len(targets) > 1)
        if isinstance(dataset, str):
            dataset = pd.read_csv(dataset, low_memory=False)
        self._dataset: Union[data.Data, data.DataFold, data.DataSample] = data.__dict__[
            method
        ](dataset, targets, n_gram, val_rate, shuffle, n)
        self._method = method
        self._metric = METRICS[metric]
        self._n_split = n
        self._is_regression = -1 if is_regression else 1
        self._n_jobs = n_jobs
        if refit is None:
            refit = method != "DataSample"
        self._refit = refit

    def compute_contrib(self):
        """
        Parallele computation of feature importance

        :return: dataframe with features contribution
        :rtype: pd.DataFrame
        """
        scores = collections.defaultdict(float)
        cross_scores = collections.defaultdict(list)
        futures = []
        model = None
        if not self._refit:
            assert (
                self._method == "DataSample"
            ), "Misconfiguration, refit is False only if you're using Sample Strategy"
            # Train model here
            model = _fit_voting_classifier(
                self._dataset[0][1], self._model, self._is_m_output, self._metric
            )
        with ThreadPoolExecutor(self._n_jobs) as executor:
            for col, splits in self._dataset:
                futures.append(
                    executor.submit(
                        _train_pred_evaluate,
                        col=col,
                        splits=splits,
                        model=get_model(self._model, self._is_m_output)
                        if self._refit
                        else model,
                        metric=self._metric,
                        refit=self._refit,
                    )
                )
            for future in as_completed(futures):
                col, perfs = future.result()
                scores[str(col)] = sum(perfs) / len(perfs)
                cross_scores[str(col)] = perfs

        wait(futures)  # wait while all thread finish
        for col in scores:
            if col != "[]":
                scores[col] = self._is_regression * (scores[col] - scores["[]"])
        contrib_perfs = pd.DataFrame(
            scores.values(), columns=["Contribution"], index=scores.keys()
        )
        cross_perfs = pd.DataFrame(
            cross_scores.values(),
            columns=["Split" + str(i) for i in range(self._n_split)],
            index=cross_scores.keys(),
        )
        perfs = pd.concat((contrib_perfs, cross_perfs), axis=1)
        return perfs.sort_values(by="Contribution", ascending=False)
