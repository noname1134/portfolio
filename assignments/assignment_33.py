import numpy as np


def find_best_threshold(y_true, probabilities):
    best_threshold = None
    best_net_profit = float("-inf")
    thresholds = np.round(np.arange(0.10, 0.91, 0.05), 2)
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        tp = ((predictions == 1) & (y_true == 1)).sum()
        fp = ((predictions == 1) & (y_true == 0)).sum()
        cost = (tp + fp) * 50
        profit = tp * 500
        net_profit = profit - cost
        if net_profit > best_net_profit:
            best_threshold = threshold
            best_net_profit = net_profit
    return best_threshold, best_net_profit

if __name__ == "__main__":
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0])

    probabilities = np.array([
        0.90, 0.80, 0.70, 0.40, 0.20,
        0.10, 0.65, 0.35, 0.55, 0.45
    ])
    best_threshold, best_net_profit = find_best_threshold(y_true, probabilities)
    print("Best threshold:", best_threshold)
    print("Best profit:", best_net_profit)