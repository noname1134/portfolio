"""Evaluate a churn model on the held-out test set under a contact budget."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path("data/Telco-Customer-Churn.csv")
ID_COLUMN = "customerID"
TARGET = "Churn"
NA_TOKENS = ["", "NA", "N/A", "n/a", "-", "missing", "null", "None"]

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Campaign economics (the contact budget is defined relative to the training set size)
VALUE_PER_CHURNER_CAUGHT = 500
COST_PER_CONTACT = 50
MAX_CONTACTS = 500
THRESHOLDS = [0.5, 0.7]

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling",
    "MultipleLines", "InternetService", "PaymentMethod", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract",
]


def load_data(path: Path) -> pd.DataFrame:
    """Load the CSV, fix blank TotalCharges, and encode the target as 0/1."""
    df = pd.read_csv(path, na_values=NA_TOKENS).drop(columns=ID_COLUMN)
    df = df.replace(r"^\s*$", np.nan, regex=True)

    # Blank TotalCharges belong to new customers (tenure == 0) who haven't been billed yet.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    new_customers = df["TotalCharges"].isna() & (df["tenure"] == 0)
    df.loc[new_customers, "TotalCharges"] = 0

    df[TARGET] = df[TARGET].map({"No": 0, "Yes": 1})
    if df[TARGET].isna().any():
        raise ValueError(f"Unexpected values in target column '{TARGET}'")
    return df


def build_model() -> Pipeline:
    """Preprocessing + logistic regression in a single pipeline."""
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", drop="if_binary")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=5000)),
    ])


def campaign_profit(churners_caught: int, contacted: int) -> int:
    """Value of the churners caught minus the cost of contacting everyone selected."""
    return VALUE_PER_CHURNER_CAUGHT * churners_caught - COST_PER_CONTACT * contacted


def scaled_capacity(n_samples: int, n_reference: int) -> int:
    """Scale the contact budget from the reference set (train) to a set of n_samples."""
    if n_samples <= 0 or n_reference <= 0:
        raise ValueError("Sample counts must be positive")
    capacity = round(MAX_CONTACTS * n_samples / n_reference)
    if capacity < 1:
        raise ValueError(f"Scaled capacity rounds to {capacity}; it must be at least 1")
    return capacity


def evaluate_selection(y_true: np.ndarray, selected: np.ndarray) -> dict[str, float]:
    """Confusion counts, metrics and profit for contacting the customers flagged in `selected`."""
    tp = int(y_true[selected].sum())
    fp = int(selected.sum()) - tp
    fn = int(y_true.sum()) - tp
    tn = len(y_true) - tp - fp - fn

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "contacted": tp + fp,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / len(y_true),
        "net_profit": campaign_profit(tp, tp + fp),
    }


def top_n_mask(probabilities: np.ndarray, n: int) -> np.ndarray:
    """Boolean mask selecting the n customers with the highest churn probability."""
    mask = np.zeros(len(probabilities), dtype=bool)
    mask[np.argsort(probabilities)[::-1][:n]] = True
    return mask


def compare_strategies(y_true: np.ndarray, probabilities: np.ndarray, capacity: int) -> pd.DataFrame:
    """One row per strategy: top-n by probability, then each probability threshold."""
    selections = {f"top {capacity}": top_n_mask(probabilities, capacity)}
    for threshold in THRESHOLDS:
        selections[f"threshold {threshold}"] = probabilities >= threshold

    rows = {name: evaluate_selection(y_true, mask) for name, mask in selections.items()}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("strategy")


def main() -> None:
    df = load_data(DATA_PATH)
    X, y = df.drop(columns=TARGET), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    model = build_model()
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]

    capacity = scaled_capacity(len(y_test), len(y_train))
    results = compare_strategies(y_test.to_numpy(), probabilities, capacity)

    print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.3f}\n")
    print(results.round(3).to_string())


if __name__ == "__main__":
    main()