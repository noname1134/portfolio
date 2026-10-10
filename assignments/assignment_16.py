from sklearn.metrics import roc_auc_score

y_test = [0, 0, 1, 1, 0, 1]
probabilities = [0.1, 0.4, 0.8, 0.7, 0.3, 0.9]
auc = roc_auc_score(y_test, probabilities)

print(f"ROC-AUC: {auc:.3f}")