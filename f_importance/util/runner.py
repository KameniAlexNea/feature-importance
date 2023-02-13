"""
The runner script allow you to run a feature importance estimation on you own data

:return: _description_
:rtype: _type_
"""
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


def get_sample_dataframe() -> pd.DataFrame:
    """
    loads the wine dataset from scikit-learn and adds the target column to the dataframe.

    :return: Wine dataset loaded from sklearn
    :rtype: pd.DataFrame
    """
    dtset = datasets.load_wine(as_frame=True)
    data = dtset["data"]
    data["target"] = dtset["target"]
    return data


def get_arguments() -> Namespace:
    """
    parses the command line arguments using ArgumentParser. 
    The parsed arguments include the machine learning model, 
    the dataset creation method, the evaluation metric, the validation rate, 
    the number of CPU cores to use, the range of feature groups to compute importance, 
    whether to shuffle the data, whether it's a regression problem, and the number of models/splits to train.

    :return: argument parsed from cmd
    :rtype: Namespace
    """
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


def compute_importance_from_args(args: Namespace) -> pd.DataFrame:
    """
    calls compute_importance function with the parsed arguments.

    :param args: arguments parsed from cmd
    :type args: Namespace
    :return: feature importance estimated
    :rtype: pd.DataFrame
    """
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
) -> pd.DataFrame:
    """
    the main function that computes the feature importance based on the input arguments. 
    It first creates an instance of the Model class with the given arguments, 
    then computes the contribution of each feature by calling compute_contrib method of the instance.

    :param model_name: The name of the model to use for feature importance estimation.
    Choices include: "XGBoostClassifier", "RandomForestClassifier", etc., defaults to "XGBoostClassifier"
    :type model_name: str, optional
    :param method: The method to use for dataset splitting.
    Choices include: "DataSample", "DataFold", etc, defaults to "DataSample"
    :type method: str, optional
    :param metric: The metric to use for model evaluation.
    Choices include: "accuracy_score", "mean_squared_error", etc., defaults to "accuracy_score"
    :type metric: str, optional
    :param dataset: The dataset to use for feature importance estimation.
    Can be either a file path to a .csv file or a Pandas DataFrame., defaults to get_sample_dataframe()
    :type dataset: Union[str, pd.DataFrame], optional
    :param targets: The target(s) in the dataset to predict.
    Can be either a single column name as a string or a list of column names., defaults to "target"
    :type targets: Union[str, list], optional
    :param n_gram: The range of feature groups to compute importance for.
    For example, setting n_gram=(1, 2) would compute feature importances for both
    single features and combinations of 2 features., defaults to (1, 1)
    :type n_gram: tuple[int, int], optional
    :param val_rate: The validation rate to use for dataset splitting., defaults to 0.15
    :type val_rate: float, optional
    :param shuffle: Whether to shuffle the data before splitting it., defaults to True
    :type shuffle: bool, optional
    :param n: The number of training models/splits to use for feature importance estimation., defaults to 5
    :type n: int, optional
    :param is_regression: Whether the problem being solved is a regression problem or a classification problem., defaults to False
    :type is_regression: bool, optional
    :param n_jobs: The number of CPU cores to use for parallel processing., defaults to os.cpu_count()
    :type n_jobs: int, optional
    :param refit: Whether to refit the model on the entire dataset after feature selection.
    None (default) will use the value set in the Model class, which is False by default., defaults to None
    :type refit: bool, optional
    :return: A DataFrame containing the computed feature importances.
    :rtype: pd.DataFrame
    """
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
