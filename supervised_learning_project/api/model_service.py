import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
import numpy as np
import utils
import pandas as pd


def clean_customer_df(customer_df: pd.DataFrame) -> pd.DataFrame:
    df = customer_df.copy()
    utils.validate_not_empty(df)
    missing = [col for col in utils.TOTAL_FEATURES if col not in df.columns]
    if missing:
        raise ValueError(f"Input file is missing required columns: {missing}")

    df = df.replace(r"^\s*$", np.nan, regex=True)
    utils.validate_unique_ids(df)
    df = utils.clean_numeric_columns(df)
    df = utils.clean_total_charges(df)
    return df


def load_model(model_path: Path) -> Pipeline:
    utils.require_file(model_path, "Train and save the model first.")
    model = joblib.load(model_path)
    return model

def predict_churn(model: Pipeline, customer_df: pd.DataFrame) -> float:
    if len(customer_df) != 1:
        raise ValueError("predict_churn expects exactly one customer")

    customer_df = clean_customer_df(customer_df)
    class_1_index = np.where(model.classes_ == 1)[0][0]
    prob = model.predict_proba(customer_df)[:, class_1_index]
    return float(prob[0])


def predict_churn_batch(model: Pipeline, customers_df: pd.DataFrame) -> pd.DataFrame:
    customers_df = clean_customer_df(customers_df)
    class_1_index = np.where(model.classes_ == 1)[0][0]
    probabilities = model.predict_proba(customers_df)[:, class_1_index]
    return pd.DataFrame({utils.ID_COLUMN: customers_df[utils.ID_COLUMN], "churn_probability": probabilities})