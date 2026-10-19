
"""FastAPI application for heart disease predictions."""

import logging
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

from src.heart_mlops.predict import predict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("heart_mlops.api")

REQUEST_COUNT = Counter(
    "heart_api_requests_total",
    "Total HTTP requests handled by the API",
    ["method", "endpoint", "status"],
)

REQUEST_DURATION = Histogram(
    "heart_api_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
)

PREDICTION_COUNT = Counter(
    "heart_api_predictions_total",
    "Total successful prediction requests",
    ["prediction"],
)

app = FastAPI(
    title="Heart Disease Prediction API",
    description="Prediction API for the MLOps assignment.",
    version="1.0.0",
)


@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start_time = time.perf_counter()
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception:
        logger.exception(
            "Request failed: method=%s path=%s",
            request.method,
            request.url.path,
        )
        raise
    finally:
        endpoint = request.url.path
        duration = time.perf_counter() - start_time

        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=endpoint,
            status=str(status_code),
        ).inc()

        REQUEST_DURATION.labels(
            method=request.method,
            endpoint=endpoint,
        ).observe(duration)

        logger.info(
            "HTTP request method=%s path=%s status=%s duration_seconds=%.4f",
            request.method,
            endpoint,
            status_code,
            duration,
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


@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/predict")
def predict_endpoint(patient: PatientFeatures):
    try:
        result = predict(patient.model_dump())
        PREDICTION_COUNT.labels(
            prediction=str(result["prediction"])
        ).inc()
        return result
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
