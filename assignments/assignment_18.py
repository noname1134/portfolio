from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Load a built-in classification dataset
data = load_breast_cancer()

X = data.data
y = data.target

print("X shape:", X.shape)
print("y shape:", y.shape)

model = DecisionTreeClassifier(random_state=42, max_depth=3)

scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print("Fold accuracies:", scores)
print("Mean accuracy:", np.mean(scores))
print("Standard deviation:", np.std(scores))