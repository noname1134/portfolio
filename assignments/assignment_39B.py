from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = Path("data/Telco-Customer-Churn.csv")
ID_COLUMN = "customerID"
TARGET = "Churn"
NA_TOKENS = ["", "NA", "N/A", "n/a", "-", "missing", "null", "None"]
RANDOM_STATE = 42
TEST_SIZE = 0.2

CV_FOLDS = 5
SCORING = ["roc_auc", "recall", "precision"]

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

def build_preprocessor() -> ColumnTransformer:
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
    return preprocessor


def build_model(classifier: BaseEstimator) -> Pipeline:
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", classifier),
    ])

def get_models() -> dict[str, Pipeline]:
    return {
        "logistic_regression": build_model(LogisticRegression(max_iter=5000)),
        "decision_tree": build_model(DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)),
        "random_forest": build_model(RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
        "knn": build_model(KNeighborsClassifier(n_neighbors=5)),
    }


def evaluate(models: dict[str, Pipeline], X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, model in models.items():
        scores = cross_validate(model, X, y, cv=cv, scoring=SCORING, n_jobs=-1)
        row = {"model": name}
        for metric in SCORING:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = values.mean()
            if metric == "roc_auc":
                row["roc_auc_std"] = values.std()
        rows.append(row)

    results_df = pd.DataFrame(rows).set_index("model").sort_values("roc_auc_mean", ascending=False).round(3)
    return results_df


def main() -> None:
    df = load_data(DATA_PATH)
    X, y = df.drop(columns=TARGET), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    models = get_models()
    results_df = evaluate(models, X_train, y_train)
    print(results_df.to_string())

if __name__ == "__main__":
    main()