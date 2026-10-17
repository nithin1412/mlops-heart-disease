"""Create EDA visualizations from the cleaned UCI Heart Disease data."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
CLEAN_PATH = ROOT / "data" / "processed" / "heart_disease_clean.csv"
FIGURE_DIR = ROOT / "reports" / "figures"
FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)
sns.set_theme(context="notebook", style="whitegrid")


def run_eda() -> list[Path]:
    if not CLEAN_PATH.exists():
        raise FileNotFoundError(
            f"Clean dataset not found at {CLEAN_PATH}. Run `python -m src.heart_mlops.validate_data` first."
        )
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(CLEAN_PATH)
    outputs: list[Path] = []

    # Histograms for numeric and integer-coded features.
    numeric = frame[FEATURES].apply(pd.to_numeric, errors="coerce")
    numeric.hist(figsize=(14, 10), bins=15, edgecolor="black")
    plt.gcf().suptitle("Heart Disease Dataset: Feature Distributions", y=1.02)
    plt.tight_layout()
    path = FIGURE_DIR / "feature_histograms.png"
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close("all")
    outputs.append(path)

    # Correlation heatmap is descriptive only; correlation does not imply causation.
    corr = frame[FEATURES + ["target"]].corr(numeric_only=True)
    plt.figure(figsize=(12, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
    plt.title("Feature and Target Correlation Heatmap")
    plt.tight_layout()
    path = FIGURE_DIR / "correlation_heatmap.png"
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    outputs.append(path)

    plt.figure(figsize=(7, 5))
    counts = frame["target"].value_counts().reindex([0, 1], fill_value=0)
    sns.barplot(x=counts.index.astype(str), y=counts.values)
    plt.xlabel("Target (0 = absence, 1 = presence)")
    plt.ylabel("Number of records")
    plt.title("Binary Target Class Balance")
    plt.tight_layout()
    path = FIGURE_DIR / "class_balance.png"
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    outputs.append(path)

    LOGGER.info("Generated %d EDA figures in %s", len(outputs), FIGURE_DIR)
    for output in outputs:
        LOGGER.info(" - %s", output)
    return outputs


if __name__ == "__main__":
    run_eda()
