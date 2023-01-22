import collections
from typing import Union

import pandas as pd

from f_importance.dataset import data
from f_importance.metrics import METRICS
from f_importance.model import CLASSIFIERS
from f_importance.model import REGRESSORS


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
    ) -> None:
        self._model = (
            CLASSIFIERS[model_name]()
            if model_name in CLASSIFIERS
            else REGRESSORS[model_name]()
        )
        if isinstance(dataset, str):
            dataset = pd.read_csv(dataset, low_memory=False)
        self._dataset: Union[data.Data, data.DataFold, data.DataSample] = data.__dict__[
            method
        ](dataset, targets, n_gram, val_rate, shuffle, n)
        self._method = method
        self._metric = METRICS[metric]
        self._n_split = n

    def compute_contrib(self):
        scores = collections.defaultdict(int)
        cross_scores = collections.defaultdict(list)
        for col, splits in self._dataset:
            for (X_train, y_train), (X_test, y_test) in splits:
                self._model.fit(X_train, y_train)
                preds = self._model.predict(X_test)
                score = self._metric(preds, y_test)
                scores[str(col)] += score
                cross_scores[str(col)].append(score)
            scores[str(col)] /= len(splits)
            if col != [""]:
                scores[str(col)] = scores["['']"] - scores[str(col)]
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
