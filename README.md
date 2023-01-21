# F-Importance - Python Apps for feature importance

ML Features Importance Implementation

## Dev

Utilisation de pyenv : https://github.com/pyenv/pyenv

## Version python

3.6, 3.7, 3.8, 3.9, 3.10

## Needed

    pyenv install 3.6.15 3.7.15 3.8.15 3.9.15 3.10.8 3.11.0
    python3 -m venv .venv
    . .venv/bin/activate
    pip install -r requirements

## CI

    pyenv local 3.9 3.6 3.7 3.8 3.10 3.11
    tox

## Build

    python setup.py bdist_wheel

## Install

    python -m pip install -e .

## Publish

    twine upload -r innova dist/*

## Package utilities

Feature Importance is a technique that assigns a score to each input feature of a given model, indicating the degree of influence the feature has on the model's prediction. For example, when buying a new house near your workplace, location is likely the most important factor in your decision-making process. Similarly, Feature Importance ranks features based on their impact on the model's prediction.

Feature Importance is a valuable tool for understanding data, improving model performance, and interpreting model results. It allows for understanding the relationship between features and target variable, reducing the dimensionality of a model, and determining which features attribute the most to the predictive power of a model.

### F-Imp Using Gini

### F-Imp Using Data Permutation

### Removal importance

### Model importance such RandomForest
