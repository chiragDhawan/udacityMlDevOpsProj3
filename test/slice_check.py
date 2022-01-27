import pandas as pd
import joblib
import logging
import numpy as np
import os
from pathlib import Path
import sys

CWD = os.path.dirname(os.path.realpath(__file__))
curr_file_path = Path(CWD)
sys.path.append(str(curr_file_path.parent))

from ml import model_func, train_model, preprocess_data

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

CWD = os.path.dirname(os.path.realpath(__file__))
path = Path(CWD)
DATA_PATH = os.path.join(path.parent.absolute(), "data/census_cleaned.csv")
MODEL_PATH = os.path.join(path.parent.absolute(), "models/rfc_model.pkl")
ENCODER_PATH = os.path.join(path.parent.absolute(), "models/encoder.pkl")
LB_PATH = os.path.join(path.parent.absolute(), "models/lb.pkl")

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

def slice_data(data):
    """
    This function slices on the salary and calculates statistics of education-num
    :param data:
    :param feature:
    :return:
    """

    for cls in data["salary"].unique():
        df_temp = data[data["salary"] == cls]
        mean = df_temp["education-num"].mean()
        stddev = df_temp["education-num"].std()
        logger.info(f"Class: {cls}")
        logger.info(f"education-num mean: {mean:.4f}")
        logger.info(f"education-num stddev: {stddev:.4f}")


def slice_model_salary(data, model, encoder, lb, cat_feature):
    """
    This function slices the salary and check the model performance on each class of it
    :param data:
    :param cat_feature: the feature on which to perform the slice
    :param model:
    :return:
    """
    slice_output_file = open("slice_output.txt", 'w')
    for cls in data[cat_feature].unique():
        df_temp = data[data[cat_feature] == cls]
        logger.info("df temp shape\n {}".format(df_temp.shape))
        X, y, _, _ = preprocess_data.process_data(df_temp, train_model.get_cat_features(),
                                                    'salary', False, encoder, lb)
        logger.info("X\n {}".format(X.shape))
        predictions = model_func.inference(model, X)
        precision, recall, fbeta = model_func.compute_model_metrics(y, predictions)
        slice_output_file.write(f"Class: {cls} \n")
        slice_output_file.write(f"precision {precision} \n")
        slice_output_file.write(f"recall {recall} \n ")
        slice_output_file.write(f"fbeta {fbeta} \n")



if __name__ == "__main__":
    data = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    lb = joblib.load(LB_PATH)
    slice_data(data)
    slice_model_salary(data, model, encoder, lb, "race")
