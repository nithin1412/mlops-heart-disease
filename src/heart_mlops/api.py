
"""FastAPI application for heart disease predictions."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.heart_mlops.predict import predict

app = FastAPI(
    title="Heart Disease Prediction API",
    description="Prediction API for the MLOps assignment.",
    version="1.0.0",
)


class PatientFeatures(BaseModel):
    age: float = Field(..., ge=0, le=120)
    sex: float
    cp: float
    trestbps: float = Field(..., ge=0)
    chol: float = Field(..., ge=0)
    fbs: float
    restecg: float
    thalach: float = Field(..., ge=0)
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_endpoint(patient: PatientFeatures):
    try:
        return predict(patient.model_dump())
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
