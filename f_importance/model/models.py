import collections
import os
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from concurrent.futures import wait
from typing import Union

import pandas as pd
from sklearn.multioutput import ClassifierChain
from sklearn.multioutput import RegressorChain

from f_importance.dataset import data
from f_importance.metrics import METRICS
from f_importance.model import CLASSIFIERS
from f_importance.model import REGRESSORS


def _train_pred_evaluate(col: str, splits: list, model, metric):
    scores = []
    for (X_train, y_train), (X_test, y_test) in splits:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        score = metric(preds, y_test)
        scores.append(score)
    return col, scores


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

    def compute_contrib(self):
        scores = collections.defaultdict(float)
        cross_scores = collections.defaultdict(list)
        futures = []
        with ThreadPoolExecutor(self._n_jobs) as executor:
            for col, splits in self._dataset:
                futures.append(
                    executor.submit(
                        _train_pred_evaluate,
                        col=col,
                        splits=splits,
                        model=get_model(self._model, self._is_m_output),
                        metric=self._metric,
                    )
                )
            for future in as_completed(futures):
                col, perfs = future.result()
                scores[str(col)] = sum(perfs) / len(perfs)
                cross_scores[str(col)] = perfs

        wait(futures)
        for col in scores:
            if col != "['']":
                scores[col] = self._is_regression * (scores["['']"] - scores[col])
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
