from typing import Union

import numpy as np
import pandas as pd
from sklearn import tree


class GINI:
    def __init__(
        self, clazz: Union[tree.DecisionTreeClassifier, tree.DecisionTreeRegressor]
    ) -> None:
        self.clazz = clazz

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "GINI":
        self.model: Union[
            tree.DecisionTreeClassifier, tree.DecisionTreeRegressor
        ] = self.clazz(max_depth=len(X.columns))
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def score(self, X: pd.DataFrame, y: np.ndarray) -> float:
        return self.model.score(X, y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict_log_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_log_proba(X)


class GINIClassifier(GINI):
    def __init__(self):
        super().__init__(tree.DecisionTreeClassifier)


class GINIRegressor(GINI):
    def __init__(self):
        super().__init__(tree.DecisionTreeRegressor)
