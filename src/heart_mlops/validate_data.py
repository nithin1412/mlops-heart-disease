"""Validate the acquired dataset and create a cleaned binary-target dataset."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "heart_disease_raw.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
CLEAN_PATH = PROCESSED_DIR / "heart_disease_clean.csv"
RESULTS_DIR = ROOT / "reports" / "results"
REPORT_PATH = RESULTS_DIR / "data_validation.json"

EXPECTED_FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


def validate_and_clean() -> pd.DataFrame:
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {RAW_PATH}. Run `python -m src.heart_mlops.data` first."
        )

    frame = pd.read_csv(RAW_PATH, na_values=["?", " ?", ""])
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    if "num" not in frame.columns:
        raise ValueError(f"Expected original target column 'num'; found {list(frame.columns)}")

    missing_features = [name for name in EXPECTED_FEATURES if name not in frame.columns]
    if missing_features:
        raise ValueError(f"Dataset schema is missing expected features: {missing_features}")

    for column in EXPECTED_FEATURES + ["num"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    invalid_target_count = int(frame["num"].isna().sum())
    valid_targets = set(frame["num"].dropna().unique().tolist())
    if not valid_targets.issubset({0, 1, 2, 3, 4}):
        raise ValueError(f"Unexpected target values in `num`: {sorted(valid_targets)}")
    if not valid_targets:
        raise ValueError("No valid target values were found.")

    before_dedup = len(frame)
    duplicate_rows = int(frame.duplicated().sum())
    frame = frame.drop_duplicates().copy()

    # Convert the original severity target to the assignment's binary target.
    frame = frame.loc[frame["num"].notna()].copy()
    frame["target"] = (frame["num"] > 0).astype("int64")
    clean = frame[EXPECTED_FEATURES + ["target"]].copy()

    if clean.empty:
        raise ValueError("No usable rows remain after target validation.")
    if clean["target"].nunique() != 2:
        raise ValueError("Expected both binary classes (0 and 1) in the dataset.")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    clean.to_csv(CLEAN_PATH, index=False)

    report = {
        "raw_file": str(RAW_PATH.relative_to(ROOT)),
        "clean_file": str(CLEAN_PATH.relative_to(ROOT)),
        "raw_rows": int(before_dedup),
        "duplicate_rows_removed": duplicate_rows,
        "rows_removed_for_invalid_or_missing_target": invalid_target_count,
        "clean_rows": len(clean),
        "feature_count": len(EXPECTED_FEATURES),
        "features": EXPECTED_FEATURES,
        "target": "target",
        "target_mapping": "num == 0 -> 0 (absence); num in {1,2,3,4} -> 1 (presence)",
        "missing_values_per_feature": {k: int(v) for k, v in clean[EXPECTED_FEATURES].isna().sum().items()},
        "target_counts": {str(k): int(v) for k, v in clean["target"].value_counts().sort_index().items()},
        "validation": "PASS",
        "notes": [
            "Feature missing values are retained as missing for training-time imputation.",
            "Imputation/scaling must be fitted only on training folds to avoid data leakage.",
            "Exact duplicate rows are removed before modelling."
        ],
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOGGER.info("Validation passed. Clean dataset: %s", CLEAN_PATH)
    LOGGER.info("Validation report: %s", REPORT_PATH)
    return clean


if __name__ == "__main__":
    validate_and_clean()
