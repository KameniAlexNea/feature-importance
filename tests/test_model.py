import numpy as np
import pandas as pd
from pytest import fixture

from f_importance.dataset import data as dataset
from f_importance.model import CLASSIFIERS
from f_importance.model.models import Model
from f_importance.model.models import get_model


@fixture
def data():
    dat = np.random.rand(100, 3)
    dat = pd.DataFrame(dat, columns=["A1", "A2", "A3"])
    dat["y"] = np.random.randint(0, 2, len(dat))
    return dat


@fixture
def data_reg():
    dat = np.random.rand(100, 3)
    dat = pd.DataFrame(dat, columns=["A1", "A2", "A3"])
    dat["y"] = dat["A1"] ** 2 - dat["A2"] * 2 + dat["A3"] * dat["A2"]
    return dat


def test_init(data):
    model = Model(
        "XGBClassifier", "DataFold", "accuracy_score", data, "y", (1, 1), 0.15, True, 5
    )
    assert isinstance(model, Model)
    assert isinstance(get_model(model._model), CLASSIFIERS["XGBClassifier"])
    assert isinstance(model._dataset, dataset.__dict__["DataFold"])


def test_compute(data):
    model = Model(
        "XGBClassifier", "DataFold", "accuracy_score", data, "y", (1, 1), 0.15, True, 2
    )
    contrib = model.compute_contrib()
    assert len(contrib) == 4


def test_compute2(data):
    model = Model(
        "XGBClassifier",
        "DataSample",
        "accuracy_score",
        data,
        "y",
        (1, 2),
        0.15,
        True,
        2,
    )
    contrib = model.compute_contrib()
    assert len(contrib) == 7


def test_compute3(data_reg):
    model = Model(
        "XGBRegressor",
        "DataSample",
        "mean_absolute_error",
        data_reg,
        "y",
        (1, 2),
        0.15,
        True,
        2,
        True,
    )
    contrib = model.compute_contrib()
    assert len(contrib) == 7
