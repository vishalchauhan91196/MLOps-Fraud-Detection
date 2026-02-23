# mlops-fraud-detection

Enterprise fraud detection ML project with multi-environment configuration (`dev`, `uat`, `prd`), DVC-based reproducible pipeline, MLflow tracking/registry integration, drift detection, champion/challenger promotion flow, and FastAPI inference service.

## Scope
- This repo contains ML code and training/evaluation/promotion logic.
- Kubernetes, Argo CD/Workflows, KServe, Gateway/Istio, and monitoring workloads are maintained in:
  - `MLOps-Fraud-Detection-K8s-Workload`

## High-level architecture
- Data lifecycle:
  - ingestion -> validation -> preprocessing -> feature engineering
- Model lifecycle:
  - model building -> evaluation -> registration -> promotion
- Reliability controls:
  - config lint, schema validation, structured logging, centralized exceptions
- Drift controls:
  - data drift, concept drift, model drift with retrain signal outputs
- Serving:
  - FastAPI app for online prediction from trained artifacts

## Repository layout
- `src/core`: generic/shared runtime (`logger`, `exceptions`, `io`, `settings`, `s3`, `config_lint`, transformers)
- `src/data`: ingestion, validation, preprocessing, transformation
- `src/feature`: feature engineering and feature manifest generation
- `src/model`: building, evaluation, registration, promotion
- `src/drift_detection`: separate drift modules (`data`, `concept`, `model`) + runner
- `configs/environments`: base + env-specific config overlays
- `configs/schemas`: dataset/schema contracts
- `configs/contracts`: runtime required keys/contracts
- `app/app.py`: FastAPI inference API
- `tests`: unit, integration, smoke, performance tests
- `dvc.yaml`: end-to-end reproducible pipeline stages

## Environment model
- Select environment using `APP_ENV`:
  - `dev`
  - `uat`
  - `prd`
- Config resolution is environment-aware via `src/core/settings.py`.
- No hardcoded environment values should be added in code paths.

## Pipeline (DVC)
Defined in `dvc.yaml`:
1. `config_lint`
2. `data_ingestion`
3. `data_validation`
4. `data_preprocessing`
5. `feature_engineering`
6. `model_building`
7. `model_evaluation`
8. `drift_detection`
9. `model_registration`
10. `model_promotion`

## Quickstart
```powershell
cd mlops-fraud-detection
python -m venv .venv
. .venv/Scripts/Activate.ps1
pip install -r requirements.txt
```

Run full pipeline:
```powershell
$env:APP_ENV='dev'
$env:PYTHONPATH='.'
dvc repro
```

Run single stage:
```powershell
$env:APP_ENV='dev'
$env:PYTHONPATH='.'
dvc repro model_building
```

Run tests:
```powershell
$env:PYTHONPATH='.'
pytest -q
```

## FastAPI inference
Run locally:
```powershell
$env:APP_ENV='dev'
$env:PYTHONPATH='.'
uvicorn app.app:app --host 0.0.0.0 --port 8000
```

Endpoints:
- `GET /health`
- `POST /predict`

`/predict` expects `rows` (list of feature dictionaries) and returns predictions + probabilities.

## Data and model expectations
- Source dataset: `notebooks/insuranceFraud.csv`
- DVC-managed data zones:
  - `data/raw`
  - `data/interim`
  - `data/processed`
- Artifacts:
  - `models/model.pkl`
  - `models/feature_manifest.json`
  - evaluation/registration/promotion reports in `reports/`

## Drift and retraining
- Drift output artifacts:
  - `reports/drift_report.json`
  - `reports/retrain_signal.json`
- Retraining decision can be triggered by:
  - data drift
  - concept drift
  - model drift/performance degradation

## Security and operations
- Use OIDC-based auth for cloud access; no static cloud keys in code.
- Keep secrets/config secrets out of git.
- Use MLflow tracking URI per environment and S3-backed model/artifact persistence.

## Related repo
- `MLOps-Fraud-Detection-K8s-Workload` consumes promoted model outputs and handles GitOps-based deployment to KServe.
