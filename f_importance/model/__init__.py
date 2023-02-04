"""
    In this package, we manipulate everything
    about models
"""

from lightgbm import LGBMClassifier
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBClassifier
from xgboost import XGBRegressor

CLASSIFIERS = dict(
    XGBClassifier=XGBClassifier,
    LGBMClassifier=LGBMClassifier,
    RandomForestClassifier=RandomForestClassifier,
    GradientBoostingClassifier=GradientBoostingClassifier,
    DecisionTreeClassifier=DecisionTreeClassifier,
)

REGRESSORS = dict(
    XGBRegressor=XGBRegressor,
    LGBMRegressor=LGBMRegressor,
    RandomForestRegressor=RandomForestRegressor,
    GradientBoostingRegressor=GradientBoostingRegressor,
    DecisionTreeRegressor=DecisionTreeRegressor,
)
