from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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

    # Make predictions
    predictions = model.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, predictions)
    print("Test accuracy:", accuracy)
    probabilities = model.predict_proba(X_test)

    print("First 5 predictions:", predictions[:5])
    print("First 5 probabilities:\n", probabilities[:5])
    print("Class order:", model.classes_)