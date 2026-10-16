"""Train the churn model and export an unlabeled test batch."""

from pathlib import Path
import utils

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data(path: Path) -> pd.DataFrame:
    """Load the CSV, fix blank TotalCharges, and encode the target as 0/1."""
    df = pd.read_csv(path, na_values=utils.NA_TOKENS)
    df = df.replace(r"^\s*$", np.nan, regex=True)

    # Blank TotalCharges belong to new customers (tenure == 0) who haven't been billed yet.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    new_customers = df["TotalCharges"].isna() & (df["tenure"] == 0)
    df.loc[new_customers, "TotalCharges"] = 0

    df[utils.CHURN_COLUMN] = df[utils.CHURN_COLUMN].map({"No": 0, "Yes": 1})
    if df[utils.CHURN_COLUMN].isna().any():
        raise ValueError(f"Unexpected values in target column '{utils.CHURN_COLUMN}'")
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
        ("num", numeric, utils.NUMERIC_FEATURES),
        ("cat", categorical, utils.CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=5000)),
    ])


def save_model(model: Pipeline, features: list[str]) -> None:
    """Persist the fitted pipeline and the feature names it expects."""
    utils.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    utils.FEATURE_NAMES_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, utils.MODEL_PATH)
    pd.Series(features, name="feature").to_csv(utils.FEATURE_NAMES_PATH, index=False)


def main() -> None:
    df = load_data(utils.ORIGINAL_DATA_PATH)
    X, y = df[utils.TOTAL_FEATURES], df[utils.CHURN_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    y_test_with_id = pd.DataFrame(y_test, columns=[utils.CHURN_COLUMN])
    y_test_with_id[utils.ID_COLUMN] = X_test[utils.ID_COLUMN].values
    utils.save_csv(utils.REFERENCE_DATA_PATH, X_train)
    utils.save_csv(utils.TEST_CHURN_DATA_PATH, y_test_with_id)

    X_train = X_train.drop(columns=utils.ID_COLUMN)
    model = build_model().fit(X_train, y_train)
    save_model(model, list(X_train.columns))

    utils.save_csv(utils.TEST_DATA_PATH, X_test)
    print(f"Saved model -> {utils.MODEL_PATH}")
    print(f"Saved reference data -> {utils.REFERENCE_DATA_PATH}")
    print(f"Saved test data -> {utils.TEST_DATA_PATH}")


if __name__ == "__main__":
    main()