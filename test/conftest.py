"""
pytest conftest fixture to read data
"""
import joblib
import pytest
import pandas as pd
import os
from pathlib import Path

CWD = os.path.dirname(os.path.realpath(__file__))
path = Path(CWD)
DATA_PATH = os.path.join(path.parent.absolute(), "data/census_cleaned.csv")

@pytest.fixture(scope='session')
def data():
    return pd.read_csv(DATA_PATH)



