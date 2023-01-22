import numpy as np
import pandas as pd
from pytest import fixture

from f_importance.dataset import data as dataset
from f_importance.model import CLASSIFIERS
from f_importance.model.models import Model


@fixture
def data():
    data = np.random.rand(100, 3)
    data = pd.DataFrame(data, columns=["A1", "A2", "A3"])
    data["y"] = np.random.randint(0, 2, len(data))
    return data


def test_init(data):
    model = Model(
        "XGBClassifier", "DataFold", "accuracy_score", data, "y", (1, 1), 0.15, True, 5
    )
    assert isinstance(model, Model)
    assert isinstance(model._model, CLASSIFIERS["XGBClassifier"])
    assert isinstance(model._dataset, dataset.__dict__["DataFold"])


def test_compute(data):
    model = Model(
        "XGBClassifier", "DataFold", "accuracy_score", data, "y", (1, 1), 0.15, True, 2
    )
    contrib = model.compute_contrib()
    assert len(contrib) == 4


def test_compute2(data):
    model = Model(
        "XGBClassifier", "DataFold", "accuracy_score", data, "y", (1, 2), 0.15, True, 2
    )
    contrib = model.compute_contrib()
    assert len(contrib) == 7
