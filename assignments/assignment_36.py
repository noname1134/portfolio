import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


def add_features(X):
    X = X.copy()
    X["EstimatedSpending"] = X["tenure"] * X["MonthlyCharges"]
    X["IsNewCustomer"] = (X["tenure"] < 6).astype(int)
    X["HighMonthlyCharges"] = (X["MonthlyCharges"] > 80).astype(int)
    return X


if __name__ == "__main__":
    df = pd.DataFrame({
        "tenure": [2, 48, 6, 36, 12, 24, 3, 60],
        "MonthlyCharges": [90, 40, 85, 50, 70, 60, 95, 35],
        "Contract": [
            "Month-to-month",
            "Two year",
            "Month-to-month",
            "One year",
            "Month-to-month",
            "One year",
            "Month-to-month",
            "Two year"
        ],
        "Churn": [1, 0, 1, 0, 1, 0, 1, 0]
    })
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    feature_engineer = FunctionTransformer(add_features)
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), ["tenure", "MonthlyCharges", "EstimatedSpending", "IsNewCustomer", "HighMonthlyCharges"]),
        ("cat", OneHotEncoder(), ["Contract"])
    ])

    model = Pipeline([
        ("feature_engineering", feature_engineer),
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
