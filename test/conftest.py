"""
pytest conftest fixture to read data
"""
import joblib
import pytest
import pandas as pd
import os
from pathlib import Path
import data
CWD = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(CWD, "data/census_cleaned.csv")
print(DATA_PATH)
@pytest.fixture(scope='session')
def data():
    return pd.read_csv(CWD)



