
"""Train and evaluate Logistic Regression and Random Forest models."""

import json
import logging
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "heart_disease_clean.csv"
ARTIFACT_DIR = ROOT / "artifacts"
RESULT_DIR = ROOT / "reports" / "results"
FIGURE_DIR = ROOT / "reports" / "figures"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    for directory in (ARTIFACT_DIR, RESULT_DIR, FIGURE_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Clean dataset not found at {DATA_PATH}. "
            "Run the data acquisition and validation steps first."
        )

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["target"])
    y = df["target"].astype(int)

    # Keep the test set untouched until final evaluation.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    numeric_features = X_train.select_dtypes(include=["number"]).columns.tolist()

    # Imputation and scaling are fitted only on training folds.
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        ],
        remainder="drop",
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    candidates = {
        "logistic_regression": (
            LogisticRegression(max_iter=2000, random_state=42),
            {
                "model__C": [0.1, 1.0, 10.0],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "random_forest": (
            RandomForestClassifier(random_state=42),
            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 5, 10],
                "model__min_samples_split": [2, 5],
            },
        ),
    }

    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("heart-disease-classification")

    results = []
    best_score = -1.0
    best_model = None
    best_name = None

    for model_name, (estimator, param_grid) in candidates.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessing", preprocessing),
                ("model", estimator),
            ]
        )

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
            error_score="raise",
        )

        with mlflow.start_run(run_name=model_name):
            search.fit(X_train, y_train)
            fitted_model = search.best_estimator_
            predictions = fitted_model.predict(X_test)
            probabilities = fitted_model.predict_proba(X_test)[:, 1]

            metrics = {
                "accuracy": float(accuracy_score(y_test, predictions)),
                "precision": float(
                    precision_score(y_test, predictions, zero_division=0)
                ),
                "recall": float(
                    recall_score(y_test, predictions, zero_division=0)
                ),
                "f1": float(f1_score(y_test, predictions, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_test, probabilities)),
                "cv_best_roc_auc": float(search.best_score_),
            }

            mlflow.log_params(search.best_params_)
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("cv_folds", 5)
            mlflow.log_param("random_state", 42)
            mlflow.log_metrics(metrics)

            # Save a confusion matrix for this model.
            cm_path = FIGURE_DIR / f"{model_name}_confusion_matrix.png"
            ConfusionMatrixDisplay.from_predictions(
                y_test,
                predictions,
                display_labels=["Absence (0)", "Presence (1)"],
            )
            plt.title(f"{model_name}: Confusion Matrix")
            plt.tight_layout()
            plt.savefig(cm_path, dpi=150)
            plt.close()

            roc_path = FIGURE_DIR / f"{model_name}_roc_curve.png"
            RocCurveDisplay.from_predictions(y_test, probabilities)
            plt.title(f"{model_name}: ROC Curve")
            plt.tight_layout()
            plt.savefig(roc_path, dpi=150)
            plt.close()

            report_path = RESULT_DIR / f"{model_name}_classification_report.json"
            report_path.write_text(
                json.dumps(
                    classification_report(
                        y_test,
                        predictions,
                        output_dict=True,
                        zero_division=0,
                    ),
                    indent=2,
                ),
                encoding="utf-8",
            )

            mlflow.log_artifact(str(cm_path))
            mlflow.log_artifact(str(roc_path))
            mlflow.log_artifact(str(report_path))

            results.append(
                {
                    "model": model_name,
                    "best_params": search.best_params_,
                    **metrics,
                }
            )

            logger.info("%s metrics: %s", model_name, metrics)
            logger.info("%s best parameters: %s", model_name, search.best_params_)

            if search.best_score_ > best_score:
                best_score = search.best_score_
                best_model = fitted_model
                best_name = model_name

    model_path = ARTIFACT_DIR / "best_model.joblib"
    joblib.dump(best_model, model_path)

    results_path = RESULT_DIR / "model_comparison.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    metadata_path = ARTIFACT_DIR / "model_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "best_model": best_name,
                "selection_metric": "mean cross-validation ROC-AUC",
                "best_cv_roc_auc": best_score,
                "feature_names": X.columns.tolist(),
                "target_mapping": "0=absence, 1=presence",
                "test_size": 0.20,
                "random_state": 42,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info("Best model: %s", best_name)
    logger.info("Saved model: %s", model_path)
    logger.info("Saved comparison: %s", results_path)
    logger.info("Saved metadata: %s", metadata_path)


if __name__ == "__main__":
    main()
