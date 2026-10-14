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


def require_file(path: Path, hint: str) -> None:
    """Fail early with a clear message if a required file is missing."""
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. {hint}")


def validate_not_empty(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("Input CSV contains no customers")


def load_feature_names(path: Path) -> list[str]:
    """Feature names the model was trained on, saved at training time."""
    return pd.read_csv(path)["feature"].tolist()


def _rows(mask: pd.Series) -> list:
    """Row labels where `mask` is true, capped so error messages stay short."""
    return mask[mask].index.tolist()[:MAX_ROWS_SHOWN]


def find_numeric_problems(df: pd.DataFrame) -> list[str]:
    """Describe every non-numeric, negative or out-of-range value in the numeric features."""
    problems = []
    for column in NUMERIC_FEATURES:
        converted = pd.to_numeric(df[column], errors="coerce")
        not_numeric = df[column].notna() & converted.isna()
        if not_numeric.any():
            problems.append(f"'{column}' has non-numeric values at rows {_rows(not_numeric)}")
        negative = converted < 0
        if negative.any():
            problems.append(f"'{column}' has negative values at rows {_rows(negative)}")
        finite = converted.notna() & ~np.isfinite(converted)
        if finite.any():
            problems.append(f"'{column}' has infinity values at rows {_rows(finite)}")
        if column in WHOLE_NUMBER_FEATURES:
            fractional = finite & (converted % 1 != 0)
            if fractional.any():
                problems.append(f"'{column}' must be a whole number, found fractions at rows {_rows(fractional)}")

    senior = pd.to_numeric(df["SeniorCitizen"], errors="coerce")
    not_binary = senior.notna() & ~senior.isin(VALID_SENIOR_CITIZEN_VALUES)
    if not_binary.any():
        problems.append(f"'SeniorCitizen' must be 0 or 1, found other values at rows {_rows(not_binary)}")
    return problems



def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the numeric features, convert them to numbers, and zero-fill new customers' TotalCharges."""
    problems = find_numeric_problems(df)
    if problems:
        raise ValueError("Invalid input data:\n- " + "\n- ".join(problems))

    df = df.copy()
    for column in NUMERIC_FEATURES:
        df[column] = pd.to_numeric(df[column])

    # Blank TotalCharges belong to new customers (tenure == 0) who haven't been billed yet.
    new_customers = df["TotalCharges"].isna() & (df["tenure"] == 0)
    df.loc[new_customers, "TotalCharges"] = 0
    return df


def load_features(path: Path, features: list[str]) -> pd.DataFrame:
    """Read the customer CSV, check it has every required column, and clean it."""
    df = pd.read_csv(path, na_values=NA_TOKENS)
    validate_not_empty(df)
    required_columns = [ID_COLUMN] + features
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Input file is missing required columns: {missing}")

    df = df.replace(r"^\s*$", np.nan, regex=True)
    validate_unique_ids(df)
    return clean_numeric_columns(df)


def validate_unique_ids(df: pd.DataFrame) -> None:
    if df[ID_COLUMN].isna().any():
        raise ValueError(f"Input file has customers with a missing {ID_COLUMN}")
    duplicated = df.loc[df[ID_COLUMN].duplicated(), ID_COLUMN].unique()
    if len(duplicated):
        raise ValueError(f"Input file has duplicate {ID_COLUMN} values, e.g. {list(duplicated[:5])}")



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
    require_file(model_path, "Train and save the model first.")
    require_file(FEATURE_NAMES_PATH, "It is written by the training script.")
    require_file(input_path, "Pass a customer CSV with --input.")

    model = joblib.load(model_path)
    features = load_feature_names(FEATURE_NAMES_PATH)
    df = load_features(input_path, features)
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