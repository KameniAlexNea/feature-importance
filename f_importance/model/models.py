import collections
import os
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from concurrent.futures import wait
from typing import Union

import pandas as pd
import numpy as np
from sklearn.multioutput import ClassifierChain
from sklearn.multioutput import RegressorChain
from sklearn.ensemble import VotingClassifier, VotingRegressor

from f_importance.dataset import data
from f_importance.metrics import METRICS
from f_importance.model import CLASSIFIERS
from f_importance.model import REGRESSORS


def _train_pred_evaluate(col: str, splits: list, model, metric, refit=True):
    scores = []
    for (X_train, y_train), (X_test, y_test) in splits:
        if refit:
            model.fit(X_train, y_train)
        preds = model.predict(X_test)
        score = metric(preds, y_test)
        scores.append(score)
    return col, scores

def _fit_voting_classifier(splits: list, model_name: str, is_m_output: bool, scorer):
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
    model.__sklearn_is_fitted__ = lambda: True
    model.estimators_ = models
    return model


def get_model(model_name, is_m_output):
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
        refit=None
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
        scores = collections.defaultdict(float)
        cross_scores = collections.defaultdict(list)
        futures = []
        model = None
        if not self._refit:
            assert self._method == "DataSample", "Misconfiguration, refit is False only if you're using Sample Strategy"
            # Train model here
            model = _fit_voting_classifier(self._dataset[0][0], self._model, self._is_m_output, self._metric)
        with ThreadPoolExecutor(self._n_jobs) as executor:
            for col, splits in self._dataset:
                futures.append(
                    executor.submit(
                        _train_pred_evaluate,
                        col=col,
                        splits=splits,
                        model=get_model(self._model, self._is_m_output) if self._refit else model,
                        metric=self._metric,
                        refit=self._refit
                    )
                )
            for future in as_completed(futures):
                col, perfs = future.result()
                scores[str(col)] = sum(perfs) / len(perfs)
                cross_scores[str(col)] = perfs

        wait(futures)
        for col in scores:
            if col != "[]":
                scores[col] = self._is_regression * (scores["[]"] - scores[col])
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
