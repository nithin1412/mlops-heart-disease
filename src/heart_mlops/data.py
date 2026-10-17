"""Acquire the official UCI Heart Disease dataset and record provenance."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
MANIFEST_DIR = ROOT / "data" / "manifests"
RAW_PATH = RAW_DIR / "heart_disease_raw.csv"
MANIFEST_PATH = MANIFEST_DIR / "dataset_provenance.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


def acquire_dataset() -> Path:
    """Fetch UCI dataset 45, combine features/target, and save a raw CSV."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    try:
        dataset = fetch_ucirepo(id=45)
        features = dataset.data.features.copy()
        targets = dataset.data.targets.copy()
    except Exception as exc:
        raise RuntimeError(
            "Could not retrieve UCI Heart Disease dataset (ID 45). "
            "Check internet access and the ucimlrepo installation; no fallback "
            "dataset was substituted."
        ) from exc

    if features.empty or targets.empty:
        raise ValueError("UCI returned an empty feature or target table.")
    if len(features) != len(targets):
        raise ValueError("Feature and target row counts do not match.")

    frame = pd.concat([features, targets], axis=1)
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    if "num" not in frame.columns:
        # The UCI wrapper may expose the target under a different label.
        target_columns = [str(column).strip().lower() for column in targets.columns]
        if len(target_columns) == 1:
            frame = frame.rename(columns={target_columns[0]: "num"})
        else:
            raise ValueError(f"Expected one target column; got {target_columns}")

    # Preserve raw values as supplied, but normalize the common missing marker.
    frame = frame.replace(r"^\s*\?\s*$", pd.NA, regex=True)
    frame.to_csv(RAW_PATH, index=False)

    metadata = {
        "dataset_name": "Heart Disease",
        "uci_dataset_id": 45,
        "source_url": "https://archive.ics.uci.edu/dataset/45/heart+disease",
        "doi": "10.24432/C52P4X",
        "license": "CC BY 4.0",
        "retrieval_method": "ucimlrepo.fetch_ucirepo(id=45)",
        "raw_file": str(RAW_PATH.relative_to(ROOT)),
        "rows": int(frame.shape[0]),
        "columns": list(frame.columns),
        "feature_columns": [c for c in frame.columns if c != "num"],
        "original_target": "num",
        "binary_target_mapping": {"0": 0, "1,2,3,4": 1},
        "missing_marker": "? (normalized to blank/NA)",
        "note": "Review target values and missingness during validation before modelling."
    }
    MANIFEST_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    LOGGER.info("Saved raw dataset to %s (%d rows, %d columns)", RAW_PATH, *frame.shape)
    LOGGER.info("Saved provenance manifest to %s", MANIFEST_PATH)
    return RAW_PATH


if __name__ == "__main__":
    acquire_dataset()
