"""
    In this package, we manipulate everything about metrics

    It supports all sklearn metrics implementations
"""
from sklearn import metrics

"""
List of proposed metrics same as sklearn metrics
"""
METRICS = {name: metrics.__dict__[name] for name in metrics.__all__}
