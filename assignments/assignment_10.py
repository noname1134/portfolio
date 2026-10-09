import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

telco = pd.DataFrame({
    "age": [25, 42, 31, 55, 28, 47],
    "monthly_charges": [40.5, 90.2, 65.3, 110.0, 55.0, 85.0],
    "contract": [
        "Month-to-month",
        "One year",
        "Two year",
        "Month-to-month",
        "One year",
        "Two year"
    ],
    "internet_service": [
        "DSL",
        "Fiber optic",
        "DSL",
        "No",
        "Fiber optic",
        "DSL"
    ],
    "churn": [0, 1, 0, 1, 0, 1]
})


if __name__ == "__main__":
    df = pd.DataFrame(telco)
    X = df.drop("churn", axis=1)
    y = df["churn"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ["age", "monthly_charges"]),
            ("cat", OneHotEncoder(), ["contract", "internet_service"])
        ]
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression())
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"predictions: {predictions}, y_test: {y_test.tolist()}")
    accuracy = accuracy_score(y_test, predictions)
    print(f"accuracy: {accuracy}")
    cm = confusion_matrix(y_test, predictions)
    print(f"confusion matrix: {cm}")