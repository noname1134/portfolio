"""Compare the churn model's predictions on production data with the actual churn outcomes."""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score, log_loss, brier_score_loss, roc_auc_score
import utils

MODEL_PREDICTION_PATH = Path("result/churn_model_predictions.csv")
MODEL_DRIFTED_PREDICTION_PATH = Path("result/drifted_churn_model_predictions.csv")
ACTUAL_CHURN_PATH = Path("data/actual_churn.csv")
EVAL_PATH = Path("result/performance_evaluation.csv")

VALUE_PER_CHURNER_CAUGHT = 500
COST_PER_CONTACT = 50

DEFAULT_CAPACITY = 125
DEFAULT_THRESHOLD = 0.5

CHURN_COLUMN = "Churn"
PROBABILITY_COLUMN = "churn_probability"


def campaign_profit(churners_caught: int, contacted: int) -> int:
    return VALUE_PER_CHURNER_CAUGHT * churners_caught - COST_PER_CONTACT * contacted


def validate_same_customers(actual_ids: pd.Series, predicted_ids: pd.Series, label: str) -> None:
    """The predictions must cover exactly the customers that have an actual churn outcome."""
    without_predictions = set(actual_ids) - set(predicted_ids)
    without_outcomes = set(predicted_ids) - set(actual_ids)
    if without_predictions or without_outcomes:
        raise ValueError(
            f"{label}: predictions and actual churn cover different customers "
            f"({len(without_predictions)} without predictions, {len(without_outcomes)} without actual churn)"
        )


def validate_scores(y_true: np.ndarray, probabilities: np.ndarray) -> None:
    """Raise one ValueError listing every problem with the labels and probabilities."""
    problems = []
    if len(y_true) != len(probabilities):
        problems.append("labels and probabilities must have the same length")
    if not np.isin(y_true, [0, 1]).all():
        problems.append("actual churn labels must contain only 0 and 1")
    if len(np.unique(y_true)) < 2:
        problems.append("both churn classes are required to compute ROC-AUC")
    if not np.isfinite(probabilities).all():
        problems.append("probabilities must be finite")
    elif ((probabilities < 0) | (probabilities > 1)).any():
        problems.append("probabilities must be between 0 and 1")
    if problems:
        raise ValueError("Invalid input data:\n- " + "\n- ".join(problems))


def load_csv(path: Path, column_names: list[str]) -> pd.DataFrame:
    available = pd.read_csv(path, nrows=0).columns
    missing = [column for column in column_names if column not in available]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    df = pd.read_csv(path, usecols=column_names)
    utils.validate_not_empty(df)
    utils.validate_unique_ids(df)
    return df


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
    utils.require_file(real_churn_path, "Pass the actual churn CSV with --real_churn.")
    utils.require_file(predictions_path, "Pass the model predictions CSV with --input or --drifted_input.")

    real_churn_df = load_csv(real_churn_path, [utils.ID_COLUMN, "Churn"])
    predicted_df = load_csv(predictions_path, [utils.ID_COLUMN, "churn_probability"])

    validate_same_customers(real_churn_df[utils.ID_COLUMN], predicted_df[utils.ID_COLUMN], data_type)

    merged_df = predicted_df.merge(real_churn_df, on=utils.ID_COLUMN, how="inner", validate="one_to_one")
    y_true = merged_df[CHURN_COLUMN].to_numpy()
    probabilities = merged_df[PROBABILITY_COLUMN].to_numpy()
    validate_scores(y_true, probabilities)

    top_n = min(capacity, len(y_true))
    selections = {
        f"threshold >= {threshold}": probabilities >= threshold,
        f"top {top_n}": top_n_mask(probabilities, top_n),
    }

    scores = probability_metrics(y_true, probabilities)
    rows = [
        {"data_type": data_type, "method": method, **scores, **selection_metrics(y_true, selected)}
        for method, selected in selections.items()
    ]

    result_df = pd.DataFrame(rows).set_index(["data_type", "method"])
    return result_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=MODEL_PREDICTION_PATH, help=f"CSV with model predictions (default: {MODEL_PREDICTION_PATH})")
    parser.add_argument("--drifted_input", type=Path, default=MODEL_DRIFTED_PREDICTION_PATH, help=f"CSV with predictions on drifted data (default: {MODEL_DRIFTED_PREDICTION_PATH})")
    parser.add_argument("--real_churn", type=Path, default=ACTUAL_CHURN_PATH, help=f"CSV with the actual churn outcomes (default: {ACTUAL_CHURN_PATH})")
    parser.add_argument("--output", type=Path, default=EVAL_PATH, help=f"where to write the evaluation report (default: {EVAL_PATH})")
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