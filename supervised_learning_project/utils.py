from pathlib import Path

import numpy as np
import pandas as pd

ID_COLUMN = "customerID"
CHURN_COLUMN = "Churn"
PROBABILITY_COLUMN = "churn_probability"

NA_TOKENS = ["", "NA", "N/A", "n/a", "-", "missing", "null", "None"]
VALID_SENIOR_CITIZEN_VALUES = (0, 1)
WHOLE_NUMBER_FEATURES = ("tenure", "SeniorCitizen")
MAX_ROWS_SHOWN = 5

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling",
    "MultipleLines", "InternetService", "PaymentMethod", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract",
]
TOTAL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [ID_COLUMN]

ORIGINAL_DATA_PATH = Path("data/Telco-Customer-Churn.csv")
MODEL_PATH = Path("models/churn_model.joblib")

TEST_DATA_PATH = Path("test/test_customers.csv")
DRIFTED_TEST_DATA_PATH = Path("test/drifted_test_customers.csv")
TEST_CHURN_DATA_PATH = Path("data/actual_churn.csv")

PREDICTIONS_PATH = Path("result/churn_model_predictions.csv")
DRIFTED_PREDICTION_PATH = Path("result/drifted_churn_model_predictions.csv")

FEATURE_NAMES_PATH = Path("models/churn_model_customer_features_names.csv")
REFERENCE_DATA_PATH = Path("data/reference_data.csv")


def require_file(path: Path, hint: str) -> None:
    """Fail early with a clear message if a required file is missing."""
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. {hint}")


def save_csv(output_path: Path, data: pd.DataFrame, index: bool = False) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=index)


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
        non_finite = converted.notna() & ~np.isfinite(converted)
        if non_finite.any():
            problems.append(f"'{column}' has infinity values at rows {_rows(non_finite)}")
        if column in WHOLE_NUMBER_FEATURES:
            is_finite = converted.notna() & np.isfinite(converted)
            fractional = is_finite & (converted % 1 != 0)
            if fractional.any():
                problems.append(f"'{column}' must be a whole number, found fractions at rows {_rows(fractional)}")

    senior = pd.to_numeric(df["SeniorCitizen"], errors="coerce")
    not_binary = senior.notna() & ~senior.isin(VALID_SENIOR_CITIZEN_VALUES)
    if not_binary.any():
        problems.append(f"'SeniorCitizen' must be 0 or 1, found other values at rows {_rows(not_binary)}")
    return problems


def clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """zero-fill new customers' TotalCharges."""
    clean_df = df.copy()
    new_customers = clean_df["TotalCharges"].isna() & (clean_df["tenure"] == 0)
    clean_df.loc[new_customers, "TotalCharges"] = 0
    return clean_df


def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and convert numeric features while preserving missing values."""
    problems = find_numeric_problems(df)
    if problems:
        raise ValueError("Invalid input data:\n- " + "\n- ".join(problems))

    df = df.copy()
    for column in NUMERIC_FEATURES:
        df[column] = pd.to_numeric(df[column])

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


def validate_same_customers(actual_ids: pd.Series, predicted_ids: pd.Series, label: str) -> None:
    """The predictions must cover exactly the customers that have an actual churn outcome."""
    without_predictions = set(actual_ids) - set(predicted_ids)
    without_outcomes = set(predicted_ids) - set(actual_ids)
    if without_predictions or without_outcomes:
        raise ValueError(
            f"{label}: predictions and actual churn cover different customers "
            f"({len(without_predictions)} without predictions, {len(without_outcomes)} without actual churn)"
        )


def load_csv_by_columns(path: Path, column_names: list[str]) -> pd.DataFrame:
    available = pd.read_csv(path, nrows=0).columns
    missing = [column for column in column_names if column not in available]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    df = pd.read_csv(path, usecols=column_names)
    validate_not_empty(df)
    validate_unique_ids(df)
    return df


def validate_scores(y_true: np.ndarray, probabilities: np.ndarray) -> None:
    """Raise one ValueError listing every problem with the labels and probabilities."""
    problems = []
    if len(y_true) != len(probabilities):
        problems.append("labels and probabilities must have the same length")
    if not np.isin(y_true, [0, 1]).all():
        problems.append("actual churn labels must contain only 0 and 1")
    if len(np.unique(y_true)) < 2:
        problems.append("both churn classes are required to compute ROC-AUC")
    if not np.isfinite(probabilities).all():
        problems.append("probabilities must be finite")
    elif ((probabilities < 0) | (probabilities > 1)).any():
        problems.append("probabilities must be between 0 and 1")
    if problems:
        raise ValueError("Invalid input data:\n- " + "\n- ".join(problems))


def get_pob_and_churn_from_csv(real_churn_path: Path, predictions_path: Path, data_type: str = "original") -> tuple[np.ndarray, np.ndarray]:
    require_file(real_churn_path, "Pass the actual churn CSV with --real_churn.")
    require_file(predictions_path, "Pass the model predictions CSV with --input or --drifted_input.")

    real_churn_df = load_csv_by_columns(real_churn_path, [ID_COLUMN, "Churn"])
    predicted_df = load_csv_by_columns(predictions_path, [ID_COLUMN, "churn_probability"])

    validate_same_customers(real_churn_df[ID_COLUMN], predicted_df[ID_COLUMN], data_type)

    merged_df = predicted_df.merge(real_churn_df, on=ID_COLUMN, how="inner", validate="one_to_one")
    y_true = merged_df[CHURN_COLUMN].to_numpy()
    probabilities = merged_df[PROBABILITY_COLUMN].to_numpy()
    validate_scores(y_true, probabilities)

    return probabilities, y_true
