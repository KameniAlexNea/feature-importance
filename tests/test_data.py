from f_importance.dataset.data import Data, DataFold, DataSample
import numpy as np
import pandas as pd
from pytest import fixture

@fixture
def data():
    data = np.random.rand(100, 4)
    data = pd.DataFrame(data, columns=["A1", "A2", "A3", "y"])
    return data

def test_check_n_grams(data):
    data_cls = Data(data, 'y', (1, 1), 0.15, True)
    assert len(data_cls._grams) == 4

    data_cls = Data(data, 'y', (1, 2), 0.15, True)
    assert len(data_cls._grams) == 7

    assert all(
        i in data_cls._grams for i in [
            ("A1", "A2"), ("A1", "A3"), ("A2", "A3")
        ]
    )


def test_iterator(data):
    data_cls = Data(data, 'y', (1, 1), 0.15, True)
    n = 0
    for col, (x_train, y_train), (x_test, y_test) in data_cls:
        n += 1
        assert len(x_train) + len(x_test) == len(data_cls)
        assert len(x_train) == len(y_train)
        assert len(x_test) == len(y_test)
    assert n == 4

def test_iterator2(data):
    data_cls = Data(data, 'y', (1, 2), 0.15, True)
    n = 0
    for col, (x_train, y_train), (x_test, y_test) in data_cls:
        n += 1
        assert len(x_train) + len(x_test) == len(data_cls)
        assert len(x_train) == len(y_train)
        assert len(x_test) == len(y_test)
    assert n == 7

def test_get_item(data):
    data_cls = Data(data, 'y', (1, 1), 0.15, True)
    col, (x_train, y_train), (x_test, y_test) = data_cls[0]
    assert col == ['']
    assert all(i in x_train.columns for i in ["A1", "A2", "A3"])
    assert len(y_train.columns) == 1
    assert "y" in y_test.columns
    assert len(x_test.columns) == 3

def test_data_folder(data):
    data_cls = DataFold(data, 'y', (1, 1), 0.15, True, n_fold=5)

    n = 0
    for col, splits in data_cls:
        n += 1
        assert len(splits) == 5
    assert n == 4

def test_data_sample(data):
    data_cls = DataSample(data, 'y', (1, 1), 0.15, True, n_sample=5)

    n = 0
    for col, splits in data_cls:
        n += 1
        assert len(splits) == 5
    assert n == 4