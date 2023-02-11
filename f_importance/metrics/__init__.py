"""
    In this package, we manipulate everything about metrics

    It supports all sklearn metrics implementations
"""
from sklearn import metrics

METRICS = {name: metrics.__dict__[name] for name in metrics.__all__}
