"""
    In this package, we manipulate everything
    about models
"""

from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

CLASSIFIERS = dict(
    XGBClassifier=XGBClassifier, LGBMClassifier=LGBMClassifier, RandomForestClassifier=RandomForestClassifier, GradientBoostingClassifier=GradientBoostingClassifier
)

REGRESSORS = dict(
    XGBRegressor=XGBRegressor, LGBMRegressor=LGBMRegressor, RandomForestRegressor=RandomForestRegressor, GradientBoostingRegressor=GradientBoostingRegressor
)
