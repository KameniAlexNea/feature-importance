# F-Importance

**Feature Importance** is a crucial technique in the field of machine learning that helps in assigning scores to each feature of a model based on the extent of its influence on the model's predictions. For instance, when you are buying a house, location is a major factor that determines your decision-making process. Similarly, Feature Importance ranks the features based on their impact on the model's predictions.

This method is extremely useful in understanding data, improving model performance, and interpreting the results of a model. It enables us to comprehend the relationship between features and the target variable, reduce the complexity of a model, and determine which features contribute the most to the model's predictive power.

## Removal importance

Another method for determining feature importance is to use each feature as an individual input to a model and evaluate its impact on the model's predictions. This can be done by considering the feature's individual prediction power as a measure of its importance. Depending on the specific machine learning problem, it may be possible to define multiple measurements, such as different error and accuracy-based metrics, and combine them. However, it's important to note that an individual feature's prediction power may not always be indicative of its overall importance in the model. In some cases, a feature may have a low individual prediction power, but when combined with other features, it may still contribute to significant performance improvements in the model.

## Permutation Feature Importance

Permutation Feature Importance involves evaluating the impact of a feature on the model's performance by shuffling its values and comparing the resulting change in error. This method is model-agnostic, meaning it can be applied to any machine learning model, and is simple to implement as it does not require complex mathematical formulas. The process of permutation feature importance involves calculating the mean squared error with the original feature values, shuffling the values, making predictions, calculating the mean squared error with the shuffled values, comparing the differences, and finally sorting the differences in descending order to determine the most to least important features.

## Gini Importance

Gini Importance is a method used to calculate the impurity of a node in a decision tree, which further helps in determining the importance of each feature. The feature importance is determined by the reduction in impurity of a node, which is then weighted by the proportion of samples that reach that node out of the total number of samples. This weighting, known as node probability, ensures that features that have a larger impact on more samples are considered more important.

It is important to note that Gini Importance is a measure of how much a feature can reduce the impurity of a node. If a feature can split the data into more pure subsets (lower gini impurity), it will be considered more important. It reflects how much a feature helps in separating the samples into different classes, which is an indicator of its relevance in predicting the target variable.


## Model importance such RandomForest

Model-specific feature importance is a technique to assess the impact of each input feature on the prediction outcome of a particular machine learning model. This method can either be extracted directly from the model or used as a standalone method.

For instance, linear regression feature importance involves fitting a linear regression model and determining the importance of each input feature by calculating the coefficients. It's important to keep in mind that this method assumes that the input features have the same scale or were normalized prior to fitting the model.

Another model-specific feature importance method is Decision Tree feature importance. This method assesses the importance of features by measuring the reduction in the criterion used to select split points. The scores are usually calculated using Gini or entropy impurity, and can be applied to other algorithms based on decision trees such as Random Forests and Gradient Boosting.

# Package Installation and Usage
The following code is a script to compute feature importance using the `f_importance` package. This script uses the [scikit-learn wine](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html#sklearn.datasets.load_wine) dataset as an example, but you can use your own dataset as well. The script provides a number of options for choosing the machine learning model, the method for computing feature importance, and the evaluation metric to use.

Before you can use the script, you need to install the `f_importance` package. This can be done by running the following command in your terminal:

```bash
pip install f_importance
```

The code uses the following main components:

* The Model class from the f_importance.model.models module, which provides the interface for computing feature importance using different machine learning models.
* The compute_contrib method of the Model class, which computes the contribution of each feature to the prediction performance.

The script starts by importing the required modules and defining the necessary functions. The get_arguments function uses the argparse module to define the command-line arguments that can be passed to the script. The `compute_importance` function takes these arguments and uses them to create an instance of the Model class and compute the feature importance using the `compute_contrib` method.

Finally, the script runs the `compute_importance` function and prints the results. You can run the script by executing the following command in your terminal:

```bash
python f_importance/util/runner.py
```

You can also run the script with different options by passing the appropriate command-line arguments. For example, you can change the machine learning model used to compute feature importance by using the `--model` option. You can also change the evaluation metric by using the `--metric` option. The full list of options can be found in the `get_arguments` function.

## Expected output :

|                                | **Importance** | **Split0** | **Split1** | **Split2** | **Split3** | **Split4** |
|--------------------------------|------------------|------------|------------|------------|------------|------------|
|                                | _0.954762_       | 1.000000   | 1.000000   | 0.916667   | 0.942857   | 0.914286   |
| 'proline'                      | **0.011429**     | 0.916667   | 1.000000   | 1.000000   | 0.857143   | 0.942857   |
| 'od280/od315_of_diluted_wines' | **0.000000**     | 0.944444   | 1.000000   | 0.972222   | 0.885714   | 0.971429   |
| 'malic_acid'                   | **-0.000476**    | 0.944444   | 0.944444   | 0.944444   | 0.971429   | 0.971429   |
| 'nonflavanoid_phenols'         | **-0.005714**    | 0.944444   | 1.000000   | 0.972222   | 0.914286   | 0.971429   |
| 'alcohol'                      | -0.006032        | 0.944444   | 0.944444   | 0.972222   | 0.971429   | 0.971429   |
| 'ash'                          | -0.006349        | 0.944444   | 0.944444   | 0.916667   | 1.000000   | 1.000000   |
| 'color_intensity'              | -0.011429        | 0.944444   | 1.000000   | 0.972222   | 0.971429   | 0.942857   |
| 'alcalinity_of_ash'            | -0.017143        | 1.000000   | 0.944444   | 0.972222   | 1.000000   | 0.942857   |
| 'hue'                          | -0.017143        | 0.972222   | 0.944444   | 1.000000   | 0.971429   | 0.971429   |
|                    'magnesium' | -0.017143        | 1.000000   | 0.972222   | 0.944444   | 0.942857   | 1.000000   |
| 'total_phenols'                | -0.017302        | 0.944444   | 1.000000   | 0.944444   | 0.971429   | 1.000000   |
| 'flavanoids'                   | _-0.022540_      | 1.000000   | 0.972222   | 1.000000   | 0.942857   | 0.971429   |
| 'proanthocyanins'              | _-0.022857_      | 0.944444   | 1.000000   | 0.972222   | 1.000000   | 0.971429   |


## Help
```
usage: runner.py [-h]
                 [--model {XGBClassifier,LGBMClassifier,RandomForestClassifier,GradientBoostingClassifier,DecisionTreeClassifier,XGBRegressor,LGBMRegressor,RandomForestRegressor,GradientBoostingRegressor,DecisionTreeRegressor}]
                 [--method {DataFold,DataSample}]
                 [--metric {accuracy_score,adjusted_mutual_info_score,...SCORERS,get_scorer_names,silhouette_samples,silhouette_score,top_k_accuracy_score,v_measure_score,zero_one_loss,brier_score_loss}]
                 [--val_rate VAL_RATE] [--n_jobs N_JOBS] [--n_gram N_GRAM [N_GRAM ...]] [--no_shuffle] [--regression] [--n_try N]

options:
  -h, --help            show this help message and exit
  --model {XGBClassifier,LGBMClassifier,RandomForestClassifier,GradientBoostingClassifier,DecisionTreeClassifier,XGBRegressor,LGBMRegressor,RandomForestRegressor,GradientBoostingRegressor,DecisionTreeRegressor}
  --method {DataFold,DataSample}
  --metric {accuracy_score,adjusted_mutual_info_score,adjusted_rand_score,auc,average_precision_score,...}
  --val_rate VAL_RATE
  --n_jobs N_JOBS
  --n_gram N_GRAM [N_GRAM ...]
                        range of feature groups to compute importance
  --no_shuffle
  --regression
  --n_try N             Number of training models/splits
```

# Coverage

---------- coverage: platform linux, python 3.11.1-final-0 -----------

| **Name**                           | **Stmts** | **Miss** | **Cover** |
|------------------------------------|-----------|----------|-----------|
| f_importance/__init__.py           | 3         | 0        | 100%      |
| f_importance/dataset/__init__.py   | 1         | 0        | 100%      |
| f_importance/dataset/data.py       | 73        | 0        | 100%      |
| f_importance/metrics/__init__.py   | 3         | 0        | 100%      |
| f_importance/model/__init__.py     | 16        | 0        | 100%      |
| f_importance/model/models.py       | 86        | 1        | 99%       |
| f_importance/model/voting.py       | 30        | 0        | 100%      |
| f_importance/util/__init__.py      | 0         | 0        | 100%      |
| f_importance/util/runner.py        | 37        | 37       | 0%        |
| f_importance/visualize/__init__.py | 0         | 0        | 100%      |
|                          **TOTAL** | 249       | 38       | **85%**   |