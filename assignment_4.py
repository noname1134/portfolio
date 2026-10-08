from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

students = {
    "hours_studied": [2, 5, 3, 8, 7, 1, 6, 4, 9, 2],
    "attendance": [60, 85, 70, 95, 90, 50, 88, 75, 98, 65],
    "previous_score": [55, 72, 64, 91, 85, 45, 78, 69, 94, 58],
    "passed": [0, 1, 0, 1, 1, 0, 1, 0, 1, 0]
}


if __name__ == "__main__":
    df = pd.DataFrame(students)
    X = df.drop("passed", axis=1)
    y = df["passed"]
    depth = 5
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    for depth in [1, 2, 3, 5]:
        model = DecisionTreeClassifier(
            max_depth=depth,
            random_state=42
        )

        model.fit(X_train, y_train)

        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)

        train_accuracy = accuracy_score(y_train, train_predictions)
        test_accuracy = accuracy_score(y_test, test_predictions)

        print(
            f"depth={depth}, "
            f"train={train_accuracy:.2f}, "
            f"test={test_accuracy:.2f}"
        )