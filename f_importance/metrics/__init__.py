"""
    In this package, we manipulate everything
    about metrics
"""
from sklearn import metrics

METRICS = {name: metrics.__dict__[name] for name in metrics.__all__}
