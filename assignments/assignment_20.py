from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = load_breast_cancer()

X = data.data
y = data.target

model = DecisionTreeClassifier(random_state=42)

param_grid = {
    "max_depth": [2, 3, 5, 10, None],
    "min_samples_leaf": [1, 2, 5]
}


X_train, X_test, y_train, y_test = train_test_split(X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
predictions = grid_search.best_estimator_.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation accuracy:", grid_search.best_score_)
print("Test accuracy:", accuracy)