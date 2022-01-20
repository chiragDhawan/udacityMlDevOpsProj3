"""
pytest conftest fixture to read data
"""
import joblib
import pytest
import pandas as pd


@pytest.fixture(scope='session')
def data():
    return pd.read_csv("../data/census_cleaned.csv")



