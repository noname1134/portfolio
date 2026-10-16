"""Compare the churn model's predictions on production data with the actual churn outcomes."""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score, log_loss, brier_score_loss, roc_auc_score
import utils

REPORT_OUTPUT_PATH = Path("result/performance_evaluation.csv")

VALUE_PER_CHURNER_CAUGHT = 500
COST_PER_CONTACT = 50

DEFAULT_CAPACITY = 125
DEFAULT_THRESHOLD = 0.5


def campaign_profit(churners_caught: int, contacted: int) -> int:
    return VALUE_PER_CHURNER_CAUGHT * churners_caught - COST_PER_CONTACT * contacted


def top_n_mask(probabilities: np.ndarray, n: int) -> np.ndarray:
    """Boolean mask of the n highest probabilities; ties keep their original row order."""
    if n < 1:
        raise ValueError("n must be at least 1")
    mask = np.zeros(len(probabilities), dtype=bool)
    mask[np.argsort(-probabilities, kind="stable")[:n]] = True
    return mask


def probability_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    """Metrics that score the probabilities themselves, independent of who gets contacted."""
    return {
        "roc_auc": roc_auc_score(y_true, probabilities),
        "log_loss": log_loss(y_true, probabilities),
        "brier_score": brier_score_loss(y_true, probabilities),
    }


def selection_metrics(y_true: np.ndarray, selected: np.ndarray) -> dict[str, float]:
    """Classification and profit metrics for contacting the customers flagged in `selected`."""
    predictions = selected.astype(int)
    contacted = int(selected.sum())
    caught = int(y_true[selected].sum())
    return {
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "contacted_customers": contacted,
        "churners_caught": caught,
        "net_profit": campaign_profit(caught, contacted),
    }


def build_performance_report(real_churn_path: Path, predictions_path: Path, data_type: str, capacity: int = DEFAULT_CAPACITY, threshold: float = DEFAULT_THRESHOLD,) -> pd.DataFrame:
    """One row per contact strategy (threshold and top-n), scored against the actual churn."""
    probabilities, y_true = utils.get_pob_and_churn_from_csv(real_churn_path, predictions_path, data_type)
    top_n = min(capacity, len(y_true))
    selections = {
        f"threshold >= {threshold}": probabilities >= threshold,
        f"top {top_n}": top_n_mask(probabilities, top_n),
    }

    scores = probability_metrics(y_true, probabilities)
    rows = [{"data_type": data_type, "method": method, **scores, **selection_metrics(y_true, selected)} for method, selected in selections.items()]
    result_df = pd.DataFrame(rows).set_index(["data_type", "method"])
    return result_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=utils.PREDICTIONS_PATH, help=f"CSV with model predictions (default: {utils.PREDICTIONS_PATH})")
    parser.add_argument("--drifted_input", type=Path, default=utils.DRIFTED_PREDICTION_PATH, help=f"CSV with predictions on drifted data (default: {utils.DRIFTED_PREDICTION_PATH})")
    parser.add_argument("--real_churn", type=Path, default=utils.TEST_CHURN_DATA_PATH, help=f"CSV with the actual churn outcomes (default: {utils.TEST_CHURN_DATA_PATH})")
    parser.add_argument("--output", type=Path, default=REPORT_OUTPUT_PATH, help=f"where to write the evaluation report (default: {REPORT_OUTPUT_PATH})")
    parser.add_argument("--capacity", type=int, default=DEFAULT_CAPACITY, help=f"customers that can be contacted (default: {DEFAULT_CAPACITY})")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help=f"churn probability at or above which a customer is contacted (default: {DEFAULT_THRESHOLD})")

    args = parser.parse_args()
    if args.capacity < 1:
        parser.error("--capacity must be at least 1")
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")

    prediction_files = {"original": args.input, "drifted": args.drifted_input}
    reports = [build_performance_report(args.real_churn, path, data_type, args.capacity, args.threshold) for data_type, path in prediction_files.items()]
    report_df = pd.concat(reports, ignore_index=False).sort_values("net_profit", ascending=False)
    utils.save_csv(args.output, report_df, True)
    print(f"report -> {args.output}")

if __name__ == "__main__":
    main()