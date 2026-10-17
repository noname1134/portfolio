import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
import numpy as np
import utils
import pandas as pd


def clean_single_customer_df(customer_df: pd.DataFrame) -> pd.DataFrame:
    df = customer_df.copy()
    utils.validate_not_empty(df)
    if len(df) != 1:
        raise ValueError("predict_churn expects exactly one customer")
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
    customer_df = clean_single_customer_df(customer_df)
    prob = model.predict_proba(customer_df)
    class_1_index = np.where(model.classes_ == 1)[0][0]
    prob_class_1 = prob[:, class_1_index]
    return float(prob_class_1[0])