import numpy as np
from sklearn import metrics

from f_importance.metrics import METRICS


def test_accuracy():
    preds = np.random.randint(0, 5, 100)
    target = np.random.randint(0, 5, 100)
    assert METRICS["accuracy_score"](preds, target) == metrics.accuracy_score(
        preds, target
    )

    assert METRICS["f1_score"](preds, target, average="macro") == metrics.f1_score(
        preds, target, average="macro"
    )

    assert METRICS["precision_score"](
        preds, target, average="macro"
    ) == metrics.precision_score(preds, target, average="macro")


def test_mse():
    preds = np.random.rand(100)
    target = np.random.rand(100)
    assert METRICS["mean_absolute_error"](preds, target) == metrics.mean_absolute_error(
        preds, target
    )

    assert METRICS["mean_squared_error"](preds, target) == metrics.mean_squared_error(
        preds, target
    )
