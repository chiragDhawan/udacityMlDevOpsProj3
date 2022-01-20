"""
Main Script which calls the training
"""
import pandas as pd

from test import slice_check
from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import logging
from ml import model_func, preprocess_data, train_model
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

CWD = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(CWD, "data/census_cleaned.csv")
MODEL_PATH = os.path.join(CWD, "models/rfc_model.pkl")
ENCODER_PATH = os.path.join(CWD, "data/encoder.pkl")
LB_PATH = os.path.join(CWD, "data/lb.pkl")

app = FastAPI()


@app.get("/")
async def say_hello():
    return {"welcome": "Welcome to the model app!! Happy inferencing"}


class ApiInfer(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias='education-num')
    marital_status: str = Field(alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias='capital-gain')
    capital_loss: int = Field(alias='capital-gain')
    hours_per_week: float = Field(alias='hours-per-week')
    native_country: str = Field(alias='native-country')


def create_data_payload(ApiInfer: ApiInfer):
    columns = ['age',
               'workclass',
               'fnlgt',
               'education',
               'education-num',
               'marital-status',
               'occupation',
               'relationship',
               'race',
               'sex',
               'capital-gain',
               'capital-loss',
               'hours-per-week',
               'native-country'
               ]

    raw_data = pd.DataFrame([[ApiInfer.age, ApiInfer.workclass, ApiInfer.fnlgt, ApiInfer.education,
                              ApiInfer.education_num, ApiInfer.marital_status, ApiInfer.occupation,
                              ApiInfer.relationship, ApiInfer.race, ApiInfer.sex, ApiInfer.capital_gain,
                              ApiInfer.capital_loss, ApiInfer.hours_per_week, ApiInfer.native_country]]
                            , columns=columns)
    '''
    raw_data = pd.DataFrame([[self.age, self.workclass, self.fnlgt, self.education,
                              self.education_num, self.marital_status, self.occupation,
                              self.relationship, self.race, self.sex, self.capital_gain,
                              self.capital_loss, self.hours_per_week, self.native_country]])
    '''
    return raw_data



@app.post("/infer/")
async def infer(apiInfer: ApiInfer):
    # load the model
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    lb = joblib.load(LB_PATH)
    # call the inference function

    raw_data = create_data_payload(apiInfer)

    logger.info("raw data {}".format(raw_data))
    logger.info("raw columns {}".format(raw_data.columns))
    logger.info("raw shape {}".format(raw_data.shape))
    X, y, _, _ = preprocess_data.process_data(raw_data, train_model.get_cat_features(), None,
                                              training=False, encoder=encoder, lb=lb)
    logger.info("X shape {}".format(X.shape))

    result = model_func.inference(model, X)

    logger.info("result shape {}".format(result.shape))
    logger.info("result value {}".format(result))
    return {"inference": "{}".format(result)}


if __name__ == "__main__":
    slice_check.slice()


