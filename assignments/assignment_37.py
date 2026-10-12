from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

data = load_breast_cancer()

X = data.data
y = data.target


if __name__ == "__main__":
    logistic_regression_model = Pipeline([
        ("preprocessor", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=5000))
    ])
    decision_tree_model = DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    )
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
    results = []
    models = {"logistic_regression_model": logistic_regression_model, "decision_tree_model": decision_tree_model, "random_forest_model": random_forest_model}
    for name, model in models.items():
        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=5,
            scoring="roc_auc"
        )
        results.append({
            "model": name,
            "mean_ROC-AUC": scores.mean(),
            "standard_deviation": scores.std()
        })
    scores_df = pd.DataFrame(results)
    print(scores_df)