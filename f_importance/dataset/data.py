import pandas as pd
from typing import Union, List
import itertools as it


class Data:
    def __init__(self, dataset: pd.DataFrame, targets: Union[str, list], n_gram=(1, 1)) -> None:
        if isinstance(targets, str):
            targets = [targets]
        self._targets = targets
        self._columns = [col for col in dataset.columns if col not in self._targets]
        self._dataset = dataset
        self._n_gram = n_gram
        self._build_n_gram()

    def _build_n_gram(self):
        grams = []
        for i in range(self._n_gram[0], min(len(self._columns), self._n_gram[1]+1)):
            grams.extend(it.combinations(self._columns), i)
        self._grams = grams

    def __iter__(self):
        self._pos = 0
        return self

    def __next__(self):
        if self._pos >= len(self._grams):
            raise StopIteration
        val = self[self._pos]
        self._pos += 1
        return val

    def __item__(self, pos: int):
        col: Union[List[str], str] = self._n_gram[pos]
        if isinstance(col, str):
            col = [col]
        return col, self._dataset[[i for i in self._columns if i not in col]].copy(), self._dataset[self._targets].copy()
