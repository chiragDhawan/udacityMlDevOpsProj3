import pandas as pd
import joblib
import logging
from ml import model_func, train_model, preprocess_data

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


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


def slice_model_salary(data, model, encoder, lb):
    """
    This function slices the salary and check the model performance on each class of it
    :param data:
    :param model:
    :return:
    """
    for cls in data["salary"].unique():
        df_temp = data[data["salary"] == cls]
        y = df_temp.pop('salary')
        logger.info("df temp shape\n {}".format(df_temp.shape))
        logger.info("y\n {}".format(y))
        X, y, _, _ = preprocess_data.process_data(df_temp, train_model.get_cat_features(),
                                                    None, False, encoder, lb)
        predictions = model_func.inference(model, X)
        logger.info("predictions {}".format(predictions))
        logger.info("predictions shape {}".format(predictions.shape))
        precision, recall, fbeta = model_func.compute_model_metrics(y, predictions)
        logger.info(f"Class: {cls}")
        logger.info(f"precision {precision}")
        logger.info(f"recall {recall}")
        logger.info(f"fbeta {fbeta}")


#if __name__ == "__main__":
def slice():
    data = pd.read_csv("./data/census_cleaned.csv")
    model = joblib.load("./models/rfc_model.pkl")
    encoder = joblib.load("./models/encoder.pkl")
    lb = joblib.load("./models/lb.pkl")
    slice_data(data)
    slice_model_salary(data, model, encoder, lb)
