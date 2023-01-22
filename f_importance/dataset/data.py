"""
This code defines three classes: Data, DataFold, and DataSample.
"""

import itertools as it
from typing import List
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


class Data:
    """
    Data is a class for handling a dataset and its targets, it's used as a parent class for other two classes.
    Data class takes in dataset, targets, n_gram, val_rate, and shuffle.

        dataset is a DataFrame that needs to be handled
        targets is a string or list of strings representing the target column(s) of the dataset
        n_gram is a tuple representing the minimum and maximum number of columns to be used in a feature set
        val_rate is a float representing the percentage of data to be used for validation
        shuffle is a boolean representing whether the dataset should be shuffled or not
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        targets: Union[str, list],
        n_gram=(1, 1),
        val_rate=0.15,
        shuffle=True,
    ) -> None:
        if isinstance(targets, str):
            targets = [targets]
        self._targets = targets
        self._columns = [col for col in dataset.columns if col not in self._targets]
        self._dataset = dataset
        self._split = True
        if shuffle:
            self._dataset = self._dataset.sample(frac=1.0)
        self._shuffle = shuffle
        self._n_gram = n_gram
        self._val_rate = int(val_rate * len(self._dataset))
        self._build_n_gram()

    def _build_n_gram(self):
        grams = [""]  # first index for base accuracy
        for i in range(self._n_gram[0], min(len(self._columns), self._n_gram[1] + 1)):
            grams.extend(it.combinations(self._columns, i))
        self._grams = grams

    def __len__(self):
        return len(self._dataset)

    def __iter__(self):
        self._pos = 0
        return self

    def __next__(self):
        if self._pos >= len(self._grams):
            raise StopIteration
        val = self[self._pos]
        self._pos += 1
        return val

    def __getitem__(self, pos: int):
        col: Union[List[str], str] = self._grams[pos]
        if isinstance(col, str):
            col = [col]
        X = self._dataset[[i for i in self._columns if i not in col]].copy()
        y = self._dataset[self._targets].copy()
        if not self._split:
            return col, X, y
        return (
            col,
            (X.head(len(X) - self._val_rate), y.head(len(X) - self._val_rate)),
            (X.tail(self._val_rate), y.tail(self._val_rate)),
        )


class DataFold(Data):
    """
    DataFold is a class that inherits from the Data class, it's used for handling cross-validation.
    DataFold class takes in dataset, targets, n_gram, val_rate, shuffle, and n_fold.

    n is an integer representing the number of folds for cross-validation

    DataFold class has one method:
        __getitem__ which returns the dataset and target column(s) based on the position of the iterator and
    splits the dataset into folds according to the n_fold value.
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        targets: Union[str, list],
        n_gram=(1, 1),
        val_rate=0.15,
        shuffle=True,
        n=5,
    ) -> None:
        super().__init__(dataset, targets, n_gram, val_rate, shuffle)
        self._n_fold = n
        self._split = False

    def __getitem__(self, pos: int):
        col, X, y = super().__getitem__(pos)
        splits = [
            (
                (X.loc[train_index], y.loc[train_index]),
                (X.loc[test_index], y.loc[test_index]),
            )
            for train_index, test_index in KFold(
                self._n_fold, shuffle=self._shuffle
            ).split(X, y)
        ]
        return col, splits


class DataSample(Data):
    """

    DataSample is a class that inherits from the Data class, it's used for handling random sampling.
    DataSample class takes in dataset, targets, n_gram, val_rate, shuffle, and n_sample.

    n is an integer representing the number of samples for random sampling

    DataSample class has several methods:
        __getitem__ which returns the dataset and target column(s) based on the position of the iterator and
    splits the dataset into samples according to the n_sample value.
        _get_split_X_y which splits the dataset into training and validation set randomly
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        targets: Union[str, list],
        n_gram=(1, 1),
        val_rate=0.15,
        shuffle=True,
        n=5,
    ) -> None:
        super().__init__(dataset, targets, n_gram, val_rate, shuffle)
        self._n_sample = n
        self._split = False

    def _get_split_X_y(self):
        perm = np.random.permutation(len(self._dataset))
        return perm[: -self._val_rate], perm[-self._val_rate :]

    def __getitem__(self, pos: int):
        col, X, y = super().__getitem__(pos)
        splits = []
        for _ in range(self._n_sample):
            train_index, test_index = self._get_split_X_y()
            splits.append(
                (
                    (X.loc[train_index], y.loc[train_index]),
                    (X.loc[test_index], y.loc[test_index]),
                )
            )
        return col, splits
