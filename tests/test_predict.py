
from src.heart_mlops.predict import predict

SAMPLE_PATIENT = {
    "age": 55,
    "sex": 1,
    "cp": 2,
    "trestbps": 130,
    "chol": 250,
    "fbs": 0,
    "restecg": 1,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 2,
    "ca": 0,
    "thal": 2,
}


def test_prediction_returns_expected_fields():
    result = predict(SAMPLE_PATIENT)

    assert result["prediction"] in (0, 1)
    assert result["label"] in (
        "heart disease present",
        "heart disease absent",
    )
    assert 0.0 <= result["confidence"] <= 1.0


def test_missing_features_raise_error():
    incomplete_patient = SAMPLE_PATIENT.copy()
    del incomplete_patient["age"]

    try:
        predict(incomplete_patient)
    except ValueError as error:
        assert "Missing required features" in str(error)
    else:
        raise AssertionError("Expected ValueError for missing features")
