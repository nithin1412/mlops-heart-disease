# MLOps Heart Disease Prediction

An assignment project for AIMLCZG523 MLOps Assignment 01. The project uses the UCI Heart Disease dataset and is built incrementally to keep code, evidence, and the report consistent.

> **Important:** This is an educational ML demonstration, not a clinically validated diagnostic tool.

## Current stage

Stage 1 scaffolding is provided: UCI dataset acquisition, data validation, binary-target construction, and EDA. Model training, MLflow tracking, CI/CD, Docker, Kubernetes, monitoring, and the final report will be added and verified in subsequent stages.

## Dataset source

- UCI Machine Learning Repository: Heart Disease, dataset ID 45
- Official page: https://archive.ics.uci.edu/dataset/45/heart+disease
- Citation: Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). *Heart Disease* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X
- UCI describes 303 Cleveland instances, 13 input features, and a `num` target with values 0–4. This project maps `num == 0` to absence (0) and `num > 0` to presence (1), matching the assignment's binary classification objective.

## Setup

Python 3.11 is recommended for a straightforward local setup.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS/Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Stage 1 commands

Run from the repository root:

```bash
python -m src.heart_mlops.data
python -m src.heart_mlops.validate_data
python -m src.heart_mlops.eda
```

Outputs:
- `data/raw/heart_disease_raw.csv` — raw downloaded dataset
- `data/processed/heart_disease_clean.csv` — cleaned dataset with binary target `target`
- `data/manifests/dataset_provenance.json` — source and target mapping metadata
- `reports/figures/` — EDA plots
- `reports/results/data_validation.json` — machine-readable validation summary

The acquisition step requires internet access. It intentionally fails with a clear error rather than silently substituting a different dataset.

## Target and missing-value handling

The UCI Cleveland data uses `num` as the original target (0–4). The project derives `target = 0` when `num == 0`, otherwise `target = 1`. A literal `?` is treated as missing. The cleaned dataset retains rows and represents missing feature values as blank CSV fields; the modelling stage will fit imputers using training data only.

## Evidence policy

Do not report metrics, screenshots, or deployment status until the corresponding command has actually run. Record environment and command output for reproducibility.
