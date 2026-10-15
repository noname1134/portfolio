"""Detect data drift between the model's reference (training) data and production data."""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, ks_2samp
from statsmodels.stats.multitest import multipletests

import utils

REFERENCE_DATA_PATH = Path("models/reference_data.csv")
TEST_DATA_PATH = Path("test/test_customers.csv")
REPORT_FOLDER_PATH = Path("result")
ORIGINAL_REPORT = "drift_report_original.csv"
SIMULATED_REPORT = "drift_report_simulated.csv"

ALPHA = 0.05
SIMULATED_CHARGE_INCREASE = 30

MISSING_LABEL = "__missing__"
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling",
    "MultipleLines", "InternetService", "PaymentMethod", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# SeniorCitizen is a 0/1 flag, so it is compared by category counts rather than with the KS test.
CONTINUOUS_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
DISCRETE_FEATURES = CATEGORICAL_FEATURES + ["SeniorCitizen"]


def simulate_drift(df: pd.DataFrame) -> pd.DataFrame:
    """A copy of `df` with an obvious shift, to confirm the detector can actually fire."""
    drifted = df.copy()
    drifted["MonthlyCharges"] += SIMULATED_CHARGE_INCREASE
    drifted["tenure"] //= 2  # halve tenure, staying in whole months
    drifted["Contract"] = "Month-to-month"

    rng = np.random.default_rng(42)
    missing_indices = rng.choice(drifted.index,size=int(len(drifted) * 0.30),replace=False)
    drifted.loc[missing_indices, "PaymentMethod"] = np.nan
    return drifted


def numeric_drift(reference: pd.Series, production: pd.Series) -> tuple[float, float, float]:
    """KS test. Returns (statistic, p_value, effect_size); the KS statistic is the effect size."""
    ref_clean = reference.dropna()
    prod_clean = production.dropna()
    if len(ref_clean) == 0 or len(prod_clean) == 0:
        return np.nan, np.nan, np.nan
    result = ks_2samp(reference.dropna(), production.dropna())
    return result.statistic, result.pvalue,  result.statistic


def cramers_v(chi2: float, counts: pd.DataFrame) -> float:
    """Cramér's V (0 = same category mix, 1 = completely different) from a chi-square statistic."""
    n = counts.to_numpy().sum()
    k = min(counts.shape) - 1
    return float(np.sqrt(chi2 / (n * k))) if n and k else 0.0


def category_counts(series: pd.Series) -> pd.Series:
    """Category counts, with missing values counted under their own MISSING_LABEL."""
    counts = series.value_counts(dropna=False)
    counts.index = [MISSING_LABEL if pd.isna(value) else value for value in counts.index]
    return counts


def categorical_drift(reference: pd.Series, production: pd.Series) -> tuple[float, float, float]:
    """Chi-square test on category counts. Returns (statistic, p_value, cramers_v)."""
    reference_counts = category_counts(reference)
    production_counts = category_counts(production)
    categories = reference_counts.index.union(production_counts.index, sort=False)
    if len(categories) == 1:
        return 0.0, 1.0, 0.0
    counts = pd.DataFrame({
        "reference": reference_counts.reindex(categories, fill_value=0),
        "production": production_counts.reindex(categories, fill_value=0),
    })
    statistic, p_value, _, _ = chi2_contingency(counts.to_numpy())
    return statistic, p_value, cramers_v(statistic, counts)


def add_adjusted_drift(report: pd.DataFrame, alpha: float) -> pd.DataFrame:
    valid_mask = report["p_value"].notna()
    drifted = pd.Series(False, index=report.index)
    adjusted = pd.Series(np.nan, index=report.index)

    if valid_mask.sum() > 0:
        valid_drifted, valid_adjusted, _, _ = multipletests(report.loc[valid_mask, "p_value"], alpha=alpha, method="fdr_bh")
        drifted.loc[valid_mask] = valid_drifted
        adjusted.loc[valid_mask] = valid_adjusted

    report["p_value_adjusted"] = adjusted
    report["drift_detected"] = drifted
    return report


def detect_drift(reference: pd.DataFrame, production: pd.DataFrame, alpha: float = ALPHA) -> pd.DataFrame:
    """Test every feature for drift, correcting p-values for testing many features at once."""
    tests = [(col, "Numeric", numeric_drift) for col in CONTINUOUS_FEATURES]
    tests += [(col, "Categorical", categorical_drift) for col in DISCRETE_FEATURES]

    rows = []
    for column, kind, test in tests:
        statistic, p_value, effect_size = test(reference[column], production[column])
        reference_missing = reference[column].isna().mean()
        production_missing = production[column].isna().mean()
        missing_difference = abs(production_missing - reference_missing)
        row = {
            "feature_name": column,
            "type": kind,
            "statistic": statistic,
            "p_value": p_value,
            "effect_size": effect_size,
            "training_missing_rate": reference_missing,
            "production_missing_rate": production_missing,
            "missing_difference": missing_difference
        }
        rows.append(row)
    report = pd.DataFrame(rows).set_index("feature_name")
    report = add_adjusted_drift(report, alpha)
    return report.sort_values("p_value")


def build_drift_reports(reference_path: Path, input_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drift reports for the real production data and for a simulated drifted copy."""
    utils.require_file(reference_path, "Pass a reference data CSV with --reference_data.")
    utils.require_file(input_path, "Pass a customer CSV with --input.")

    reference = utils.load_features(reference_path, FEATURES)
    production = utils.load_features(input_path, FEATURES)
    return detect_drift(reference, production), detect_drift(reference, simulate_drift(production))


def save_reports(reports: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Write each report to `output_dir`, keeping the feature names as the first column."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, report in reports.items():
        report.to_csv(output_dir / filename)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=TEST_DATA_PATH, help=f"CSV with customer features (default: {TEST_DATA_PATH})")
    parser.add_argument("--output", type=Path, default=REPORT_FOLDER_PATH, help=f"where to write the reports (default: {REPORT_FOLDER_PATH})")
    parser.add_argument("--reference_data", type=Path, default=REFERENCE_DATA_PATH, help=f"CSV with reference data (default: {REFERENCE_DATA_PATH})")
    args = parser.parse_args()

    reports = dict(zip([ORIGINAL_REPORT, SIMULATED_REPORT],build_drift_reports(args.reference_data, args.input)))
    save_reports(reports, args.output)

    for name, report in reports.items():
        print(f"{name}: {int(report['drift_detected'].sum())} of {len(report)} features drifted")
    print(f"reports -> {args.output}")


if __name__ == "__main__":
    main()