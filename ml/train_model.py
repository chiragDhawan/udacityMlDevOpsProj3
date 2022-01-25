# Script to train machine learning model.
import pandas as pd
import joblib
import logging
from sklearn.model_selection import train_test_split
import os
from pathlib import Path
import sys
CWD = os.path.dirname(os.path.realpath(__file__))
curr_file_path = Path(CWD)
sys.path.append(str(curr_file_path.parent))
from ml.preprocess_data import process_data
from ml.model_func import training_model, compute_model_metrics, inference

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def load_data(path):
    # Add code to load in the data.
    data = pd.read_csv(path)
    logger.info("data sample")
    logger.info(data.head())
    return data


def split_data(data):
    train, test = train_test_split(data, test_size=0.20)
    return train, test


def get_cat_features():
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
    return cat_features




if __name__=="__main__":
    data = load_data("data/census_cleaned.csv")
    cat_features = get_cat_features()
    train, test = split_data(data)
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True)

    # Proces the test data with the process_data function.
    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )
    # Train
    model = training_model(X_train, y_train)

    # Model metrics train
    predictions_train = inference(model, X_train)
    precision_train, recall_train, fbeta_train = compute_model_metrics(y_train, predictions_train)
    logger.info("precision_train {}\n recall_train {}\n fbeta_train {}".format(precision_train, recall_train, fbeta_train))

    # Model metrics test
    predictions_test = inference(model, X_test)
    precision_test, recall_test, fbeta_test = compute_model_metrics(y_test, predictions_test)
    logger.info("precision_test {}\n recall_test {}\n fbeta_test {}".format(precision_test, recall_test, fbeta_test))

    # Save model, encoder, lb
    joblib.dump(model, 'models/rfc_model.pkl')
    joblib.dump(encoder, 'models/encoder.pkl')
    joblib.dump(lb, 'models/lb.pkl')
