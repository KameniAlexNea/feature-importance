# F-Importance - Python Apps for feature importance

ML Features Importance Implementation

## Dev

Utilisation de pyenv : https://github.com/pyenv/pyenv

## Version python

3.6, 3.7, 3.8, 3.9, 3.10

## Needed

> pyenv install 3.6.15 3.7.15 3.8.15 3.9.15 3.10.8 3.11.0
> python3 -m venv .venv
> . .venv/bin/activate
> pip install -r requirements

## CI

> pyenv local 3.9 3.6 3.7 3.8 3.10 3.11
> tox

## Build

> python setup.py bdist_wheel

## Install

> python -m pip install -e .

## Publish

> twine upload -r innova dist/*
