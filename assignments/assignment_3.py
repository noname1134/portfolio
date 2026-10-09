import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
    model = DecisionTreeClassifier(random_state=42, max_depth=depth)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    print(f"X_train.shape: {X_train.shape}")
    print(f"X_test.shape: {X_test.shape}")
    print(f"y_train.shape: {y_train.shape}")
    print(f"y_test.shape: {y_test.shape}")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"predictions: {predictions}")
    print(f"y_test: {y_test}")
    accuracy = accuracy_score(y_test, predictions)
    print(f"max_depth = {depth}\naccuracy = {accuracy}")