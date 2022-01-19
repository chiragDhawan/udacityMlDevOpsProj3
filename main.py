"""
Main Script which calls the training
"""
import pandas as pd

from test import slice_check
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Union
from ml import model_func, preprocess_data, train_model
import joblib



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

    def createDataPayload(self):
        '''
        raw_data = pd.DataFrame([[ApiInfer.age, ApiInfer.workclass, ApiInfer.fnlgt, ApiInfer.education,
                                  ApiInfer.education_num, ApiInfer.marital_status, ApiInfer.occupation,
                                  ApiInfer.relationship, ApiInfer.race, ApiInfer.sex, ApiInfer.capital_gain,
                                  ApiInfer.capital_loss, ApiInfer.hours_per_week, ApiInfer.native_country]])
        '''
        raw_data = pd.DataFrame([[self.age, self.workclass, self.fnlgt, self.education,
                                  self.education_num, self.marital_status, self.occupation,
                                  self.relationship, self.race, self.sex, self.capital_gain,
                                  self.capital_loss, self.hours_per_week, self.native_country]])


        return raw_data

@app.get("/infer/", response_model=ApiInfer)
async def infer():
    #load the model
    model = joblib.load("./models/rfc_model.pkl")
    encoder = joblib.load("./models/encoder.pkl")
    lb = joblib.load("./models/lb.pkl")
    # call the inference function

    raw_data = ApiInfer.createDataPayload()

    X, y, _, _ = preprocess_data.process_data(raw_data, train_model.get_cat_features(), None,
                                            training=False, encoder=encoder, lb=lb)

    result = model_func.inference(model, X)

    return {"inference": result}

if __name__ == "__main__":
    slice_check.slice()
