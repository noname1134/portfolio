from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix)

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
THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService",
    "PaperlessBilling", "MultipleLines", "InternetService", "PaymentMethod",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract",
]

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

def campaign_profit(churners_caught: int, contacted: int) -> int:
    return VALUE_PER_CHURNER_CAUGHT * churners_caught - COST_PER_CONTACT * contacted


def cross_validated_probabilities(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]


def threshold_table(y: pd.Series, probabilities: np.ndarray) -> pd.DataFrame:
    y_true = y.to_numpy()
    rows = []
    for threshold in THRESHOLDS:
        predicted = (probabilities >= threshold).astype(int)
        contacted = int(predicted.sum())
        caught = int((y_true & predicted).sum())
        rows.append({
            "threshold": threshold,
            "precision": precision_score(y_true, predicted, zero_division=0),
            "recall": recall_score(y_true, predicted),
            "f1": f1_score(y_true, predicted, zero_division=0),
            "accuracy": accuracy_score(y_true, predicted),
            "contacted": contacted,
            "churners_caught": caught,
            "within_capacity": contacted <= MAX_CONTACTS,
            "net_profit": campaign_profit(caught, contacted),
        })
    return pd.DataFrame(rows).set_index("threshold")


def top_n_strategy(y: pd.Series, probabilities: np.ndarray, n: int = MAX_CONTACTS) -> dict:
    top_idx = np.argsort(probabilities)[::-1][:n]
    caught = int(y.to_numpy()[top_idx].sum())
    return {
        "strategy": f"top {n}",
        "contacted": len(top_idx),
        "churners_caught": caught,
        "net_profit": campaign_profit(caught, len(top_idx)),
    }


def best_threshold(table: pd.DataFrame) -> Optional[float]:
    feasible = table[table["within_capacity"]]
    return None if feasible.empty else feasible["net_profit"].idxmax()


def threshold_strategy(table: pd.DataFrame, threshold: float) -> dict:
    row = table.loc[threshold]
    return {
        "strategy": f"threshold {threshold}",
        "contacted": int(row["contacted"]),
        "churners_caught": int(row["churners_caught"]),
        "net_profit": int(row["net_profit"]),
    }


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



def main() -> None:
    df = load_data(DATA_PATH)
    X, y = df.drop(columns=TARGET), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    model = build_model()
    probabilities = cross_validated_probabilities(model, X_train, y_train)
    table = threshold_table(y_train, probabilities)
    print(table.round(3).to_string())
    threshold = best_threshold(table)
    if threshold is None:
        print(f"\nNo threshold stays within the {MAX_CONTACTS}-contact capacity.")
        return

    chosen = threshold_strategy(table, threshold)
    print(f"\nBest threshold: {threshold}")
    print(f"Expected net profit: {chosen['net_profit']}")

    comparison = pd.DataFrame([top_n_strategy(y_train, probabilities), chosen]).set_index("strategy")
    print(f"\n{comparison.to_string()}")
    """
    model.fit(X_train, y_train)
    print()
    for name, value in final_test(model, X_test, y_test, threshold).items():
        print(f"{name:<10}{value:.3f}")
    """

if __name__ == "__main__":
    main()