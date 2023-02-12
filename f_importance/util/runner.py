import os
from argparse import ArgumentParser
from argparse import Namespace
from typing import Union

import pandas as pd
from sklearn import datasets

from f_importance import dataset as ds
from f_importance import metrics
from f_importance import model
from f_importance.model.models import Model


def get_sample_dataframe():
    dtset = datasets.load_wine(as_frame=True)
    data = dtset["data"]
    data["target"] = dtset["target"]
    return data


def get_arguments():
    parser = ArgumentParser()

    parser.add_argument(
        "--model", default="XGBClassifier", choices=model.__all__, dest="model_name"
    )
    parser.add_argument("--method", default="DataFold", choices=ds.__all__)
    parser.add_argument("--metric", default="accuracy_score", choices=metrics.METRICS)

    parser.add_argument("--val_rate", default=0.15, type=float)
    parser.add_argument("--n_jobs", default=os.cpu_count(), type=int)

    parser.add_argument(
        "--n_gram",
        default=(1, 1),
        nargs="+",
        type=int,
        help="range of feature groups to compute importance",
    )
    parser.add_argument("--no_shuffle", action="store_false", dest="shuffle")
    parser.add_argument("--regression", action="store_true", dest="is_regression")
    parser.add_argument(
        "--n_try",
        default=5,
        type=int,
        dest="n",
        help="Number of training models/splits",
    )

    return parser.parse_args()


def compute_importance_from_args(args: Namespace):
    return compute_importance(**args.__dict__)


def compute_importance(
    model_name: str = "XGBoostClassifier",
    method: str = "DataSample",
    metric: str = "accuracy_score",
    dataset: Union[str, pd.DataFrame] = get_sample_dataframe(),
    targets: Union[str, list] = "target",
    n_gram: tuple[int, int] = (1, 1),
    val_rate: float = 0.15,
    shuffle: bool = True,
    n: int = 5,
    is_regression: bool = False,
    n_jobs: int = os.cpu_count(),
    refit: bool = None,
):
    model = Model(
        model_name,
        method,
        metric,
        dataset,
        targets,
        n_gram,
        val_rate,
        shuffle,
        n,
        is_regression,
        n_jobs,
        refit=refit,
    )
    contrib = model.compute_contrib()
    return contrib


if __name__ == "__main__":
    args = get_arguments()
    contrib = compute_importance_from_args(args)
    print(contrib)
