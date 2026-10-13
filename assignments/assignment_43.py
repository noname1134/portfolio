from pathlib import Path
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


def evaluate(model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    probabilities = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba"
    )[:, 1]
    rows = []
    for threshold in THRESHOLDS:
        predictions = (probabilities >= threshold).astype(int)
        cm = confusion_matrix(y_train, predictions)
        tn, fp, fn, tp = cm.ravel()
        net_profit = 500 * tp - 50 * (tp+fp)

        row = {"threshold" : threshold,
        "precision": precision_score(y_train, predictions),
        "recall": recall_score(y_train, predictions),
        "f1": f1_score(y_train, predictions),
        "accuracy": accuracy_score(y_train, predictions),
        "contacted_customers": predictions.sum(),
        "is_capacity": (predictions.sum() <= MAX_CONTACTS),
        "net_profit": net_profit}
        rows.append(row)
    result_df = pd.DataFrame(rows).set_index("threshold").sort_values("threshold", ascending=True)
    return result_df


def finale_test(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }

def get_best_threshold(feasible: pd.DataFrame) -> tuple[float, float]:
    best_threshold = None
    best_profit = None
    if feasible.empty:
        print("No feasible threshold found.")
    else:
        best_threshold = feasible["net_profit"].idxmax()
        best_profit = feasible.loc[best_threshold, "net_profit"]
    return best_threshold, best_profit



def main() -> None:
    df = load_data(DATA_PATH)
    X, y = df.drop(columns=TARGET), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    model = build_model()
    result_df = evaluate(model, X_train, y_train)
    feasible_df = result_df[result_df["is_capacity"] == True]
    best_threshold, best_profit = get_best_threshold(feasible_df)
    print("Best threshold:", best_threshold)
    print("Expected net profit:", best_profit)
    """
    model.fit(X_train, y_train)
    for name, value in finale_test(model, X_test, y_test).items():
        print(f"{name:<10}{value:.3f}")
    """

if __name__ == "__main__":
    main()