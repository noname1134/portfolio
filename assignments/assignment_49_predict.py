"""Load a saved churn model and write churn probabilities for a CSV of customers.

Usage:
    python predict.py                                   # score the default test file
    python predict.py --input new_customers.csv --output predictions.csv
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import utils as utils

MODEL_PATH = Path("models/churn_model.joblib")
FEATURE_NAMES_PATH = Path("models/churn_model_customer_features_names.csv")
PREDICTIONS_PATH = Path("result/churn_model_predictions.csv")
TEST_DATA_PATH = Path("test/test_customers.csv")

CONTACT_CAPACITY = 125
ID_COLUMN = "customerID"
NA_TOKENS = ["", "NA", "N/A", "n/a", "-", "missing", "null", "None"]
PROBABILITY_DECIMALS = 4
VALID_SENIOR_CITIZEN_VALUES = (0, 1)
WHOLE_NUMBER_FEATURES = ("tenure", "SeniorCitizen")
MAX_ROWS_SHOWN = 5
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]


def load_feature_names(path: Path) -> list[str]:
    """Feature names the model was trained on, saved at training time."""
    return pd.read_csv(path)["feature"].tolist()


def verify_campaign(results: pd.DataFrame, expected_customers: int, capacity: int) -> None:
    """Raise ValueError listing every rule the campaign table breaks."""
    problems = []

    if len(results) != expected_customers:
        problems.append(f"expected {expected_customers} customers, got {len(results)}")

    if results[ID_COLUMN].duplicated().any():
        problems.append(f"duplicate {ID_COLUMN} values in the results")

    probabilities = results["churn_probability"]
    if not probabilities.between(0, 1).all():  # also false for NaN
        problems.append("churn_probability must be between 0 and 1 with no missing values")
    if not probabilities.is_monotonic_decreasing:
        problems.append("rows are not sorted by churn_probability, highest first")

    if results["customer_rank"].tolist() != list(range(1, len(results) + 1)):
        problems.append("customer_rank must run 1..n with no gaps or repeats")

    should_select = results["customer_rank"] <= capacity
    if (results["selected_for_contact"] != should_select).any():
        problems.append("selected_for_contact does not match customer_rank <= capacity")
    if results["selected_for_contact"].sum() != min(capacity, expected_customers):
        problems.append(f"expected {min(capacity, expected_customers)} customers selected for contact")

    if problems:
        raise ValueError("Campaign verification failed:\n- " + "\n- ".join(problems))


def build_campaign(model: Pipeline, df: pd.DataFrame, features: list[str], capacity: int) -> pd.DataFrame:
    """Rank customers by churn risk (1 = highest) and flag exactly `capacity` of them for contact."""
    campaign = df[[ID_COLUMN]].copy()
    campaign["churn_probability"] = model.predict_proba(df[features])[:, 1]

    campaign = campaign.sort_values(["churn_probability", ID_COLUMN], ascending=[False, True]).reset_index(drop=True)
    campaign["customer_rank"] = np.arange(1, len(campaign) + 1)
    campaign["selected_for_contact"] = campaign["customer_rank"] <= capacity
    return campaign


def predict(model_path: Path, input_path: Path, output_path: Path, capacity: int) -> pd.DataFrame:
    """Score every customer in `input_path` and save the probabilities to `output_path`."""
    utils.require_file(model_path, "Train and save the model first.")
    utils.require_file(FEATURE_NAMES_PATH, "It is written by the training script.")
    utils.require_file(input_path, "Pass a customer CSV with --input.")

    model = joblib.load(model_path)
    features = load_feature_names(FEATURE_NAMES_PATH)
    df = utils.load_features(input_path, features)
    df = utils.clean_total_charges(df)
    result = build_campaign(model, df, features, capacity)
    verify_campaign(result, expected_customers=len(df), capacity=capacity)
    result["churn_probability"] = result["churn_probability"].round(PROBABILITY_DECIMALS)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=TEST_DATA_PATH, help=f"CSV with customer features (default: {TEST_DATA_PATH})",)
    parser.add_argument("--output", type=Path, default=PREDICTIONS_PATH, help=f"where to write the results (default: {PREDICTIONS_PATH})",)
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help=f"path to the saved model (default: {MODEL_PATH})",)
    parser.add_argument("--capacity", type=int, default=CONTACT_CAPACITY, help=f"number of customers to select for contact (default: {CONTACT_CAPACITY})")
    args = parser.parse_args()
    if args.capacity < 1:
        parser.error("--capacity must be at least 1")

    result = predict(args.model, args.input, args.output, args.capacity)
    print(f"Scored {len(result)} customers -> {args.output}")


if __name__ == "__main__":
    main()