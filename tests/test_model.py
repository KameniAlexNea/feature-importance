import numpy as np
import pandas as pd
from pytest import fixture
from sklearn.multioutput import ClassifierChain

from f_importance.dataset import data as dataset
from f_importance.model import CLASSIFIERS
from f_importance.model.models import Model
from f_importance.model.models import get_model


@fixture
def data():
    dat = np.random.rand(150, 5)
    dat = pd.DataFrame(dat, columns=["A1", "A2", "A3", "A4", "A5"])
    dat["y"] = np.random.randint(0, 2, len(dat))
    return dat


@fixture
def data_reg():
    dat = np.random.rand(150, 4)
    dat = pd.DataFrame(dat, columns=["A1", "A2", "A3", "A4"])
    dat["y"] = dat["A1"] ** 2 - dat["A2"] * 2 + dat["A3"] * dat["A2"]
    return dat


def test_init(data):
    model = Model(
        "XGBClassifier", "DataFold", "accuracy_score", data, "y", (1, 1), 0.15, True, 5
    )
    assert isinstance(model, Model)
    assert isinstance(
        get_model(model._model, model._is_m_output), CLASSIFIERS["XGBClassifier"]
    )
    assert isinstance(model._dataset, dataset.DataFold)


def test_init_multi_output(data):
    model = Model(
        "XGBClassifier",
        "DataFold",
        "accuracy_score",
        data,
        ["y"],
        (1, 1),
        0.15,
        True,
        5,
    )
    assert isinstance(model, Model)
    assert not model._is_m_output

    model = Model(
        "XGBClassifier",
        "DataFold",
        "accuracy_score",
        data,
        ["y", "A1"],
        (1, 1),
        0.15,
        True,
        5,
    )
    assert isinstance(model, Model)
    assert model._is_m_output

    chain = get_model(model._model, model._is_m_output)
    assert isinstance(chain, ClassifierChain)
    assert isinstance(chain.base_estimator, CLASSIFIERS["XGBClassifier"])


def test_compute(data):
    model = Model(
        "XGBClassifier", "DataFold", "accuracy_score", data, "y", (1, 1), 0.15, True, 2
    )
    contrib = model.compute_contrib()
    assert len(contrib) == len(data.columns)


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
    n = len(data.columns) - 1
    assert len(contrib) == len(data.columns) + n * (n - 1) // 2


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
    n = len(data_reg.columns) - 1
    assert len(contrib) == 1 + n + n * (n - 1) // 2


def test_compute4(data_reg):
    model = Model(
        "GradientBoostingRegressor",
        "DataSample",
        "mean_absolute_error",
        data_reg,
        ["y", "A1"],
        (1, 2),
        0.15,
        True,
        2,
        True,
    )
    contrib = model.compute_contrib()
    n = len(data_reg.columns) - 2
    assert len(contrib) == 1 + n + n * (n - 1) // 2


def test_compute5(data):
    data["A1"] = data["A1"].round()
    model = Model(
        "LGBMClassifier",
        "DataSample",
        "accuracy_score",
        data,
        ["y", "A1"],
        (1, 1),
        0.15,
        True,
        2,
    )
    contrib = model.compute_contrib()
    assert len(contrib) == len(data.columns) - 1


def test_compute6(data):
    data["A1"] = data["A1"].round()
    model = Model(
        "DecisionTreeClassifier",
        "DataSample",
        "accuracy_score",
        data,
        ["y", "A1"],
        (1, 1),
        0.15,
        True,
        2,
    )
    contrib = model.compute_contrib()
    assert len(contrib) == len(data.columns) - 1
