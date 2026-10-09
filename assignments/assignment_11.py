from sklearn.metrics import f1_score

y_test = [0, 0, 1, 1, 0, 1, 0, 1]
predictions = [0, 1, 1, 1, 0, 0, 0, 1]

print("F1:", f1_score(y_test, predictions))