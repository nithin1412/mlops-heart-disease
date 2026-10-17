
"""Load the trained pipeline and predict heart disease status."""

import json
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "artifacts" / "best_model.joblib"
METADATA_PATH = ROOT / "artifacts" / "model_metadata.json"


def load_model():
    """Load the saved model and metadata."""
    model = joblib.load(MODEL_PATH)
    with METADATA_PATH.open(encoding="utf-8") as file:
        metadata = json.load(file)
    return model, metadata


def predict(features: dict) -> dict:
    """Predict the binary target for one patient."""
    model, metadata = load_model()
    feature_names = metadata["feature_names"]

    missing = [name for name in feature_names if name not in features]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    input_data = pd.DataFrame(
        [[features[name] for name in feature_names]],
        columns=feature_names,
    )

    prediction = int(model.predict(input_data)[0])
    probabilities = model.predict_proba(input_data)[0]
    confidence = float(probabilities[prediction])

    return {
        "prediction": prediction,
        "label": (
            "heart disease present"
            if prediction == 1
            else "heart disease absent"
        ),
        "confidence": round(confidence, 4),
    }
