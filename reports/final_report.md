# MLOps Assignment 01: Heart Disease Classification Pipeline

## 1. Project Overview

This project implements an end-to-end MLOps pipeline for binary heart disease classification. It covers dataset acquisition, exploratory data analysis, preprocessing, model training, experiment tracking, model packaging, automated testing, API development, containerization, Kubernetes deployment, and application monitoring.

## 2. Dataset Acquisition and Validation

- Dataset: UCI Heart Disease dataset, dataset ID 45.
- Raw dataset dimensions: 303 rows and 14 columns.
- Target column: `num`.
- Target mapping: `num = 0` means absence of heart disease; values 1–4 mean presence of heart disease.
- Model input features: 13.
- Missing values: `ca` has 4 missing values and `thal` has 2.
- Class distribution: 164 absence cases and 139 presence cases.
- No duplicate rows were removed.

The project stores the raw and processed datasets, a provenance manifest, and data-validation results. Imputation and scaling are performed within model pipelines to prevent preprocessing from leaking information from the test set into training.

## 3. Exploratory Data Analysis

EDA visualizations are stored in `reports/figures/`:

- `class_balance.png` — target class distribution.
- `correlation_heatmap.png` — feature correlation heatmap.
- `feature_histograms.png` — feature distributions.

These figures support inspection of class balance, feature relationships, and data distributions.

## 4. Model Development

Two classification pipelines were evaluated:

**Logistic Regression**
- Missing-value imputation and standard scaling.
- Hyperparameter search over regularization strength and class weighting.

**Random Forest**
- Missing-value imputation.
- Hyperparameter search over tree count, maximum depth, and minimum samples required to split a node.

The data was divided using a stratified 80/20 train-test split with random seed 42. Five-fold stratified cross-validation and GridSearchCV were used for hyperparameter selection, with ROC-AUC as the selection metric.

## 5. Experiment Tracking

MLflow was used to track experiments under `heart-disease-classification`. The tracking database is `mlflow.db`. The recorded runs support comparison of model configurations and evaluation results.

## 6. Model Evaluation

Held-out test-set results:

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 0.86885 | 0.90164 |
| Precision | 0.81250 | 0.84375 |
| Recall | 0.92857 | 0.96429 |
| F1-score | 0.86667 | 0.90000 |
| ROC-AUC | 0.95671 | 0.95509 |
| Best cross-validation ROC-AUC | 0.900775 | 0.883743 |

Logistic Regression was selected as the final model because it achieved the higher best cross-validation ROC-AUC: 0.900775 compared with 0.883743 for Random Forest. Random Forest achieved higher accuracy, recall, and F1-score on this particular test split, while Logistic Regression achieved a slightly higher test ROC-AUC.

The selected model is saved at `artifacts/best_model.joblib`, with metadata in `artifacts/model_metadata.json`.

Model comparison results, classification reports, confusion matrices, and ROC curves are stored under `reports/results/` and `reports/figures/`.

## 7. Prediction Module and REST API

The prediction module loads the saved model, validates required input features, and returns a predicted class, human-readable label, and confidence score.

The FastAPI application provides:

- `GET /health` — health check.
- `POST /predict` — prediction endpoint.
- `GET /metrics` — Prometheus-format metrics endpoint.

The API includes request logging, request-duration measurements, request counters, and prediction counters. Local endpoint checks returned valid responses, and the metrics endpoint exposed request and prediction metrics.

**Medical disclaimer:** This is an academic demonstration, not a clinically validated diagnostic system. Its predictions must not be used as medical advice or as a substitute for professional evaluation.

## 8. Testing and CI/CD

Automated tests are located in `tests/test_predict.py`. The test suite completed successfully with two tests passing.

Ruff linting also completed successfully using `ruff check src tests`. The GitHub Actions workflow is defined in `.github/workflows/ci.yml` to automate code-quality checks and tests.

## 9. Containerization

The project includes a Dockerfile based on Python 3.11 slim. It installs dependencies, copies the application source and model artifact, and starts the FastAPI service with Uvicorn on port 8000.

Supporting Docker and API screenshots are stored in `screenshots/`.

## 10. Kubernetes Deployment

Kubernetes manifests are stored in `deployment/kubernetes/`:

- `deployment.yaml` defines the application deployment, image, resource requests and limits, and readiness and liveness probes.
- `service.yaml` exposes the application using a NodePort service, mapping port 30080 to container port 8000.

The local Kubernetes Desktop cluster reached one available replica. The application pod was running and ready with zero restarts at the time of verification. Health and prediction requests were tested through port forwarding.

The deployment uses `imagePullPolicy: Never` because the image was imported into the local Kubernetes environment. A remote cluster would require an accessible image registry and an appropriate image-publishing workflow.

## 11. Monitoring and Logging

The `/metrics` endpoint exposes Prometheus-format request totals, request-duration measurements, and prediction counts by predicted class. Application request logging supports basic operational troubleshooting.

This implementation provides application-level instrumentation. A full production monitoring stack with persistent metric storage, dashboards, and alert rules is not claimed.

## 12. Reproducibility and Project Structure

The repository separates data, source code, model artifacts, tests, reports, screenshots, and deployment manifests.

Important project paths include:

- `data/raw/` — raw dataset.
- `data/processed/` — processed dataset.
- `data/manifests/` — dataset provenance.
- `src/heart_mlops/` — training, prediction, and API modules.
- `artifacts/` — selected model and metadata.
- `tests/` — automated tests.
- `reports/figures/` — EDA and model evaluation figures.
- `reports/results/` — validation and evaluation results.
- `screenshots/` — implementation evidence.
- `deployment/kubernetes/` — Kubernetes manifests.
- `.github/workflows/ci.yml` — CI workflow.

The fixed random seed, saved model artifact, experiment records, dependency definitions, and automated tests support reproducibility. Results may vary if dependencies, code, or execution environments change.

## 13. Limitations and Future Improvements

- The dataset is relatively small and may not represent all patient populations.
- Reported performance comes from a single held-out test split; independent validation is needed.
- The model is not clinically validated.
- Production use would require a complete monitoring backend, dashboards, alerting, and operational procedures.
- A remote deployment would require image-registry configuration and appropriate security, secrets, and environment-specific settings.

## 14. Conclusion

This project demonstrates an end-to-end MLOps workflow for heart disease classification, from data acquisition and validation to model training, experiment tracking, REST API serving, automated testing, Docker containerization, Kubernetes deployment, and application monitoring.

The selected Logistic Regression pipeline is saved and served through FastAPI. Evaluation outputs, tests, deployment manifests, and screenshots provide supporting evidence of the implementation.
