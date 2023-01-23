# F-Importance - Python Apps for feature importance

ML Features Importance Implementation

## Dev

Utilisation de pyenv : https://github.com/pyenv/pyenv

### Version python

3.7, 3.8, 3.9, 3.10

### Needed

    pyenv install 3.6.15 3.7.15 3.8.15 3.9.15 3.10.8 3.11.0
    python3 -m venv .venv
    . .venv/bin/activate
    pip install -r requirements

### CI

    pyenv local 3.9 3.7 3.8 3.10 3.11
    tox

### Build

    python setup.py bdist_wheel

### Install

    python -m pip install -e .

## Package utilities

Feature Importance is a technique that assigns a score to each input feature of a given model, indicating the degree of influence the feature has on the model's prediction. For example, when buying a new house near your workplace, location is likely the most important factor in your decision-making process. Similarly, Feature Importance ranks features based on their impact on the model's prediction.

Feature Importance is a valuable tool for understanding data, improving model performance, and interpreting model results. It allows for understanding the relationship between features and target variable, reducing the dimensionality of a model, and determining which features attribute the most to the predictive power of a model.

### F-Imp Using Gini

Gini importance is a method used to calculate the impurity of a node in a decision tree, which can then be used to determine the importance of each feature. The feature importance is determined by the reduction in impurity of a node, which is weighted by the proportion of samples that reach that node out of the total number of samples. This weighting, known as node probability, is used to ensure that features that have a larger impact on more samples are considered more important.

It's important to note that Gini importance is a measure of how much a feature is able to reduce the impurity of a node. When a feature is able to split the data into more pure subsets(lower gini impurity) it will be considered more important. It is a measure of how much a feature helps to separate the samples in different classes. This is a good indicator of how relevant a feature is to predict the target variable.

Another point to consider is that Gini importance is only applicable for classification problems, it's not suitable for regression problems.

### F-Imp Using Data Permutation

The permutation feature importance method involves evaluating the impact of a feature on the model's performance by shuffling its values and comparing the resulting change in error. This method is model-agnostic, meaning it can be applied to any machine learning model, and it is simple to implement as it does not require complex mathematical formulas. The process of permutation feature importance includes calculating the mean squared error with the original feature values, shuffling the values, making predictions, calculating the mean squared error with the shuffled values, comparing the difference, and finally sorting the differences in descending order to determine the most to least important features.

### Removal importance

An alternative method for determining feature importance is to use each feature as an individual input to a model, and then evaluate its impact on the model's predictions. This can be done by considering the feature's individual prediction power as a measure of importance. Depending on the specific machine learning problem, it may be possible to define multiple measurements, such as different error and accuracy-based metrics, and combine them. However, it's important to note that an individual feature's prediction power may not always be indicative of its overall importance in the model. In some cases, a feature may have a low individual prediction power but when combined with other features, it may still contribute to significant performance improvements in the model.

### Model importance such RandomForest

Model-dependent feature importance is a method of determining feature importance that is specific to a particular machine learning model. These methods can often be directly extracted from the model, but they can also be used as separate methods for determining feature importance without necessarily using the model for making predictions.

One example of a model-dependent feature importance method is linear regression feature importance. This method involves fitting a linear regression model and then extracting the coefficients, which can be used to show the importance of each input variable. However, it is important to note that this method assumes that the input features have the same scale or were scaled before the model was fitted.

Another example is decision tree feature importance. Decision tree algorithms provide feature importance scores based on reducing the criterion used to select split points. These scores are typically based on Gini or entropy impurity measurements and can also be used in algorithms based on decision trees such as random forests and gradient boosting.