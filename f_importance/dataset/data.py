"""
This code defines three classes: Data, DataFold, and DataSample.
"""

import itertools as it
from typing import List
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold


class Data:

    def __init__(
        self,
        dataset: pd.DataFrame,
        targets: Union[str, list],
        n_gram=(1, 1),
        val_rate=0.15,
        shuffle=False,
        data_sample=False,
        seed=41
    ) -> None:
        """
        Data is a class for handling a dataset and its targets, it's used as a parent class for other two classes.
        Data class takes in dataset, targets, n_gram, val_rate, and shuffle.

        :param dataset: is a DataFrame that needs to be handled
        :type dataset: pd.DataFrame
        :param targets: is a string or list of strings representing the target column(s) of the dataset
        :type targets: Union[str, list]
        :param n_gram: is a tuple representing the minimum and maximum number of columns to be used in a feature set, defaults to (1, 1)
        :type n_gram: tuple, optional
        :param val_rate: is a float representing the percentage of data to be used for validation, defaults to 0.15
        :type val_rate: float, optional
        :param shuffle: is a boolean representing whether the dataset should be shuffled or not, defaults to False
        :type shuffle: bool, optional
        :param data_sample: is DataSample strategy class, defaults to False
        :type data_sample: bool, optional
        :param seed: current seed of training process, defaults to 41
        :type seed: int, optional
        """
        if isinstance(targets, str):
            targets = [targets]
        self._targets = targets
        self._columns = list(
            filter(lambda col: col not in self._targets, dataset.columns)
        )
        self._dataset = dataset
        self._split = True
        if shuffle:
            self._dataset[self._dataset.columns] = self._dataset.sample(frac=1.0).values
        self._shuffle = shuffle
        self._n_gram = n_gram
        self._val_rate = int(val_rate * len(self._dataset))
        self.data_sample = data_sample
        self._seed = seed
        self._build_n_gram()

    def _build_n_gram(self):
        """
        Build n_gram combination of column features
        """
        grams = [[]]  # first index for base accuracy
        for i in range(self._n_gram[0], min(len(self._columns), self._n_gram[1] + 1)):
            grams.extend([list(k) for k in it.combinations(self._columns, i)])
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
        """
        Get n_gram item at pos

        :param pos: position of item
        :type pos: int
        :return: item
        :rtype: Union(
            tuple(
                str, (np.ndarray, np.ndarray)
            ),
            tuple(
                str, tuple(np.ndarray, np.ndarray), tuple(np.ndarray, np.ndarray)
            )
        )
        """
        col: List[str] = self._grams[pos]
        X = self._dataset[
            [i for i in self._columns if (i not in col) or self.data_sample]
        ].copy()
        y = self._dataset[self._targets].copy()
        if not self._split:
            return col, X, y
        return (
            col,
            (X.head(len(X) - self._val_rate), y.head(len(X) - self._val_rate)),
            (X.tail(self._val_rate), y.tail(self._val_rate)),
        )


class DataFold(Data):

    def __init__(
        self,
        dataset: pd.DataFrame,
        targets: Union[str, list],
        n_gram=(1, 1),
        val_rate=0.15,
        shuffle=True,
        n=5,
        is_regression=False,
        seed = 41,
    ) -> None:
        super().__init__(dataset, targets, n_gram, val_rate, shuffle, seed=seed)
        self._n_fold = n
        self._split = False
        self._is_regression = is_regression

    def __getitem__(self, pos: int):
        """
        Get n_gram item at pos

        :param pos: position of item
        :type pos: int
        :return: item
        :rtype: tuple(
            str, list[tuple(tuple(np.ndarray, np.ndarray), tuple(np.ndarray, np.ndarray))]
        )
        """
        col, X, y = super().__getitem__(pos)
        clazz = KFold if self._is_regression else StratifiedKFold
        splits = [
            (
                (X.loc[train_index], y.loc[train_index]),
                (X.loc[test_index], y.loc[test_index]),
            )
            for train_index, test_index in clazz(
                self._n_fold, shuffle=self._shuffle, random_state=self._seed
            ).split(X, y)
        ]
        return col, splits


class DataSample(Data):

    def __init__(
        self,
        dataset: pd.DataFrame,
        targets: Union[str, list],
        n_gram=(1, 1),
        val_rate=0.15,
        shuffle=True,
        n=5,
        is_regression=False,
        seed = 41,
    ) -> None:
        super().__init__(dataset, targets, n_gram, val_rate, shuffle, data_sample=True, seed=seed)
        self._n_sample = n
        self._split = False

    def _get_split_X_y(self):
        perm = np.random.permutation(len(self._dataset))
        return perm[: -self._val_rate], perm[-self._val_rate :]

    def _permute(self, X: pd.DataFrame, pos: int, col: List[str]):
        X = X.copy()
        if pos != 0:
            X[col] = X[col].sample(frac=1.0).values
        return X

    def __getitem__(self, pos: int):
        col, X, y = super().__getitem__(pos)
        splits = []
        train_index, test_index = self._get_split_X_y()
        for _ in range(self._n_sample):
            splits.append(
                (
                    (X.loc[train_index], y.loc[train_index]),
                    (self._permute(X.loc[test_index], pos, col), y.loc[test_index]),
                )
            )
        return col, splits
