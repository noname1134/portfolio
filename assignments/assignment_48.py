"""Compare raw and calibrated logistic regression probabilities on the Telco churn dataset."""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path("data/Telco-Customer-Churn.csv")
ID_COLUMN = "customerID"
TARGET = "Churn"
NA_TOKENS = ["", "NA", "N/A", "n/a", "-", "missing", "null", "None"]

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
CALIBRATION_FOLDS = 3  # inner folds used to fit each calibrator

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


def calibrated(method: str) -> CalibratedClassifierCV:
    """A fresh logistic-regression pipeline wrapped in a probability calibrator."""
    return CalibratedClassifierCV(estimator=build_model(), method=method, cv=CALIBRATION_FOLDS)


def get_models() -> dict[str, BaseEstimator]:
    """Models to compare: a constant baseline, the raw model, and two calibrated versions."""
    return {
        "baseline": DummyClassifier(strategy="prior"),
        "logistic_regression": build_model(),
        "calibrated_sigmoid": calibrated("sigmoid"),
        "calibrated_isotonic": calibrated("isotonic"),
    }


def evaluate(models: dict[str, BaseEstimator], X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Out-of-fold ROC-AUC, Brier score and log loss per model, best Brier score first."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows = {}
    for name, model in models.items():
        scores = cross_validate(model, X, y, cv=cv, scoring=["roc_auc", "neg_brier_score", "neg_log_loss"])
        rows[name] = {
            "roc_auc": scores["test_roc_auc"].mean(),
            "log_loss": -scores["test_neg_log_loss"].mean(),
            "brier_score": -scores["test_neg_brier_score"].mean(),
        }
    eval_df = pd.DataFrame.from_dict(rows, orient="index").rename_axis("model").sort_values("brier_score")
    return eval_df


def main() -> None:
    df = load_data(DATA_PATH)
    X, y = df.drop(columns=TARGET), df[TARGET]
    X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    results = evaluate(get_models(), X_train, y_train)
    print(results.round(4).to_string())


if __name__ == "__main__":
    main()