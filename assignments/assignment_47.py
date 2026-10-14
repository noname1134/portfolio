from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss)
from sklearn.dummy import DummyClassifier

DATA_PATH = Path("data/Telco-Customer-Churn.csv")
ID_COLUMN = "customerID"
TARGET = "Churn"
NA_TOKENS = ["", "NA", "N/A", "n/a", "-", "missing", "null", "None"]
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

VALUE_PER_CHURNER_CAUGHT = 500
COST_PER_CONTACT = 50
MAX_CONTACTS = 500

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService",
    "PaperlessBilling", "MultipleLines", "InternetService", "PaymentMethod",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract",
]


def campaign_profit(churners_caught: int, contacted: int) -> int:
    return VALUE_PER_CHURNER_CAUGHT * churners_caught - COST_PER_CONTACT * contacted


def baseline_probabilities(y_train: pd.Series, n: int) -> np.ndarray:
    """Predict the training churn rate for every customer."""
    return np.full(n, y_train.mean())


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, na_values=NA_TOKENS).drop(columns=ID_COLUMN)
    df = df.replace(r"^\s*$", np.nan, regex=True)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    mask = (df["TotalCharges"].isna() & (df["tenure"] == 0))
    df.loc[mask, "TotalCharges"] = 0

    df[TARGET] = df[TARGET].map({"No": 0, "Yes": 1})
    if df[TARGET].isna().any():
        raise ValueError(f"Unexpected values in target column '{TARGET}'")
    return df

def build_model() -> Pipeline:
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

def get_calibration_table(y_true: pd.Series, probabilities: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Customers, mean predicted probability and actual churn rate per equal-width probability bin."""
    edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.digitize(probabilities, edges[1:-1])

    df = pd.DataFrame({"bin": bin_idx, "probability": probabilities, "churn": y_true.to_numpy()})
    table = (
        df.groupby("bin")
        .agg(
            customers=("churn", "size"),
            mean_predicted_probability=("probability", "mean"),
            actual_churn_rate=("churn", "mean"),
        )
        .reindex(range(n_bins))  # keep empty bins visible
    )
    table["customers"] = table["customers"].fillna(0).astype(int)
    table.index = [f"{edges[i]:.1f}–{edges[i + 1]:.1f}" for i in range(n_bins)]
    table.index.name = "probability_bin"
    return table


def evaluate(model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    probabilities = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    return get_calibration_table(y_train, probabilities)


def final_test(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, threshold: float) -> dict[str, float]:
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }


def compare_brier(model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """Out-of-fold Brier score of the model vs. a constant-prevalence baseline."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    baseline = DummyClassifier(strategy="prior")

    scores = {}
    for name, estimator in {"baseline": baseline, "logistic_regression": model}.items():
        probs = cross_val_predict(estimator, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        scores[name] = brier_score_loss(y_train, probs)
    table = pd.DataFrame({"brier_score": scores})
    # Brier skill score: 0 = no better than baseline, 1 = perfect, negative = worse.
    table["skill_vs_baseline"] = 1 - table["brier_score"] / scores["baseline"]
    return table


def main() -> None:
    df = load_data(DATA_PATH)
    X, y = df.drop(columns=TARGET), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    model = build_model()
    calibration_table = evaluate(model, X_train, y_train)
    dummy_table = compare_brier(build_model(), X_train, y_train)
    print(f"calibration_table:\n{calibration_table.round(3).to_string()}")
    print(f"dummy_table:\n{dummy_table.round(3).to_string()}")
    """
    model.fit(X_train, y_train)
    for name, value in final_test(model, X_test, y_test, threshold).items():
        print(f"{name:<10}{value:.3f}")
    """

if __name__ == "__main__":
    main()