from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score


data = load_breast_cancer()

X = data.data
y = data.target

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}


if __name__ == "__main__":
    random_forest_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scores_without_grid_search = cross_val_score(
        random_forest_model,
        X_train,
        y_train,
        cv=5,
        scoring="roc_auc"
    )

    print(f"Original Random Forest mean CV ROC-AUC: {scores_without_grid_search.mean()}")

    grid_search = GridSearchCV(
        estimator=random_forest_model,
        param_grid=param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    probabilities = grid_search.best_estimator_.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, probabilities)
    print("Tuned Random Forest mean CV ROC-AUC:", grid_search.best_score_)
    print("Best parameters:", grid_search.best_params_)
    print("Test AUC:", test_auc)