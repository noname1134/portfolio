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


def campaign_profit(churners_caught: int, contacted: int) -> int:
    return VALUE_PER_CHURNER_CAUGHT * churners_caught - COST_PER_CONTACT * contacted

def find_df_problems(churn_df: pd.Series, compere_df: pd.Series, compere_name: str) -> None:
    problems = []
    if set(churn_df) != set(compere_df):
        problems.append(f"{compere_name} predictions customers do not match actual churn customers")
    if len(churn_df) != len(compere_df):
        problems.append(f"{compere_name} predictions customers not the same size as actual churn customers")
    if problems:
        raise ValueError("Invalid input data:\n- " + "\n- ".join(problems))


def find_data_problems(y_true: np.ndarray, probs: np.ndarray, capacity: int = DEFAULT_CAPACITY) -> None:
    problems = []
    if len(y_true) != len(probs):
        problems.append("Labels and probabilities must have the same length")
    if capacity <= 0:
        problems.append("Capacity must be positive")
    if not np.isfinite(probs).all():
        problems.append("Probabilities must be finite")
    if not np.isin(y_true, [0, 1]).all():
        problems.append("Actual churn labels must contain only 0 and 1")
    if len(np.unique(y_true)) < 2:
        problems.append("ROC-AUC requires both churn classes")
    if ((probs < 0) | (probs > 1)).any():
        problems.append("Probabilities must be between 0 and 1")
    if problems:
        raise ValueError("Invalid input data:\n- " + "\n- ".join(problems))


def load_csv(path: Path, column_names: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=column_names)
    utils.validate_not_empty(df)
    utils.validate_unique_ids(df)
    return df



def evaluate(y_true: np.ndarray, probs: np.ndarray, data_type: str, predictions: np.ndarray, churners_caught: int, contacted: int, method: str) -> pd.DataFrame:
    row = {
        "data_type": data_type,
        "method": method,
        "roc_auc": roc_auc_score(y_true, probs),
        "log_loss": log_loss(y_true, probs),
        "brier_score": brier_score_loss(y_true, probs),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "contacted_customers": contacted,
        "churners_caught": churners_caught,
        "net_profit": campaign_profit(churners_caught, contacted)
    }
    result_df = pd.DataFrame([row])
    return result_df


def threshold_eval(y_true: np.ndarray, probs: np.ndarray, data_type: str, threshold: float = DEFAULT_THRESHOLD) -> pd.DataFrame:
    threshold_predictions = (probs >= threshold).astype(int)
    threshold_contacted = int(threshold_predictions.sum())
    threshold_churners_caught = int(np.sum(y_true * threshold_predictions))
    threshold_method = f"threshold >= {threshold}"

    result_df = evaluate(y_true, probs, data_type, threshold_predictions, threshold_churners_caught, threshold_contacted,threshold_method)
    return result_df


def top_n_eval(y_true: np.ndarray, probs: np.ndarray, data_type: str, capacity: int = DEFAULT_CAPACITY) -> pd.DataFrame:
    find_data_problems(y_true, probs, capacity)
    top_n_contacted = min(capacity, len(y_true))

    selected_indices = np.argsort(-probs, kind="stable")[:top_n_contacted]
    top_n_predictions = np.zeros(len(y_true), dtype=int)
    top_n_predictions[selected_indices] = 1

    top_n_churners_caught = y_true[selected_indices].sum()
    top_n_method = f"top {top_n_contacted}"

    result_df = evaluate(y_true, probs, data_type, top_n_predictions, top_n_churners_caught, top_n_contacted, top_n_method)
    return result_df

def build_eval_reports(real_churn_path: Path, model_pred_path: Path, output_path: Path, model_drifted_pred_path: Path) -> pd.DataFrame:
    utils.require_file(real_churn_path, "Pass a churn data CSV with --reference_data.")
    utils.require_file(model_pred_path, "Pass a model prediction CSV with --input.")
    utils.require_file(model_drifted_pred_path, "Pass a model drifted prediction CSV with --input.")

    real_churn_df = load_csv(real_churn_path, [utils.ID_COLUMN, "Churn"])
    model_prob_df = load_csv(model_pred_path, [utils.ID_COLUMN, "churn_probability"])
    model_drifted_prob_df = load_csv(model_drifted_pred_path, [utils.ID_COLUMN, "churn_probability"])

    find_df_problems(real_churn_df[utils.ID_COLUMN], model_prob_df[utils.ID_COLUMN], "original")
    find_df_problems(real_churn_df[utils.ID_COLUMN], model_drifted_prob_df[utils.ID_COLUMN], "drifted")

    data_df = pd.merge(model_prob_df, real_churn_df, on=utils.ID_COLUMN, how="inner", validate="one_to_one")
    drifted_data_df = pd.merge(model_drifted_prob_df, real_churn_df, on=utils.ID_COLUMN, how="inner", validate="one_to_one")

    find_data_problems(data_df["Churn"].to_numpy(), data_df["churn_probability"].to_numpy())
    find_data_problems(drifted_data_df["Churn"].to_numpy(), drifted_data_df["churn_probability"].to_numpy())

    threshold_result_df = threshold_eval(data_df["Churn"].to_numpy(), data_df["churn_probability"].to_numpy(), "original")
    top_n_result_df = top_n_eval(data_df["Churn"].to_numpy(), data_df["churn_probability"].to_numpy(), "original")

    threshold_drifted_result_df = threshold_eval(drifted_data_df["Churn"].to_numpy(), drifted_data_df["churn_probability"].to_numpy(), "drifted")
    drifted_top_n_result_df = top_n_eval(drifted_data_df["Churn"].to_numpy(), drifted_data_df["churn_probability"].to_numpy(), "drifted")
    combined_df = pd.concat([threshold_result_df, top_n_result_df, threshold_drifted_result_df, drifted_top_n_result_df],ignore_index=True).set_index("data_type").sort_values("net_profit", ascending=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_path)
    return combined_df

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=MODEL_PREDICTION_PATH, help=f"CSV with model predictions (default: {MODEL_PREDICTION_PATH})")
    parser.add_argument("--drifted_input", type=Path, default=MODEL_DRIFTED_PREDICTION_PATH,help=f"CSV with model drifted predictions (default: {MODEL_DRIFTED_PREDICTION_PATH})")
    parser.add_argument("--output", type=Path, default=EVAL_PATH, help=f"where to write the evaluation report (default: {EVAL_PATH})")
    parser.add_argument("--real_churn", type=Path, default=ACTUAL_CHURN_PATH, help=f"CSV with real churn (default: {ACTUAL_CHURN_PATH})")
    args = parser.parse_args()

    build_eval_reports(args.real_churn, args.input, args.output, args.drifted_input)
    print(f"reports -> {args.output}")

if __name__ == "__main__":
    main()