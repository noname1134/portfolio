import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer


if __name__ == "__main__":
    df = pd.DataFrame({
        "tenure": [12, 24, np.nan, 48, 6, 36],
        "MonthlyCharges": [50, 70, 60, np.nan, 40, 90],
        "Contract": [
            "Month-to-month",
            "One year",
            "Month-to-month",
            "Two year",
            np.nan,
            "One year"
        ],
        "Churn": [1, 0, 1, 0, 1, 0]
    })

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("num", numerical_pipeline, ["tenure", "MonthlyCharges"]),
        ("cat", categorical_pipeline, ["Contract"])
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression())
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"predictions: {predictions}, y_test: {y_test.tolist()}")
    accuracy = accuracy_score(y_test, predictions)
    print(f"accuracy: {accuracy}")
    cm = confusion_matrix(y_test, predictions)
    print(f"confusion matrix:\n{cm}")