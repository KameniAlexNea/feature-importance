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
from xgboost import XGBClassifier
from xgboost import XGBRegressor

from f_importance.model.gini import GINIClassifier
from f_importance.model.gini import GINIRegressor

CLASSIFIERS = dict(
    XGBClassifier=XGBClassifier,
    LGBMClassifier=LGBMClassifier,
    RandomForestClassifier=RandomForestClassifier,
    GradientBoostingClassifier=GradientBoostingClassifier,
    GINIClassifier=GINIClassifier,
)

REGRESSORS = dict(
    XGBRegressor=XGBRegressor,
    LGBMRegressor=LGBMRegressor,
    RandomForestRegressor=RandomForestRegressor,
    GradientBoostingRegressor=GradientBoostingRegressor,
    GINIRegressor=GINIRegressor,
)
