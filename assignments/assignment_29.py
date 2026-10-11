from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report



# Load the dataset
data = load_breast_cancer()
X = data.data
y = data.target


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        ("preprocessor", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=5000))
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, predictions)
    print("Test accuracy:", accuracy)
    cm = confusion_matrix(y_test, predictions)
    print(cm)
    results = pd.DataFrame({"actual_class": y_test, "predicted_class": predictions, "malignant_probability": probabilities[:, 0]})
    errors = results[results["actual_class"] != results["predicted_class"]]
    print("Number of errors:", len(errors))
    print("errors:\n", errors)


