"""
pytest conftest fixture to read data and model
"""
import joblib
import pytest
import pandas as pd


@pytest.fixture(scope='session')
def data():
    return pd.read_csv("./data/census_cleaned.csv")


@pytest.fixture(scope='session')
def model():
    return joblib.load("./models/rfc_model.pkl")

