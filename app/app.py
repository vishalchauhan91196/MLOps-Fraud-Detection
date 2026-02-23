from contextlib import asynccontextmanager
import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.core.logger import logging


class PredictRequest(BaseModel):
    """Define the `PredictRequest` API schema.

    Provides structured request/response validation for FastAPI endpoints.
    """
    rows: List[Dict[str, Any]] = Field(..., min_length=1, description="Batch rows for prediction")


class PredictResponse(BaseModel):
    """Define the `PredictResponse` API schema.

    Provides structured request/response validation for FastAPI endpoints.
    """
    predictions: List[int]
    prediction_probabilities: List[float]


class AppState:
    """Hold shared `AppState` application state.

    Stores runtime artifacts needed across endpoint calls and lifecycle events.
    """
    model = None
    expected_columns: List[str] = []


def _load_artifacts() -> None:
    """Provide internal support for load artifacts.

    Used by this module to keep the main workflow functions focused and readable.
    """
    models_dir = Path(os.getenv("INFERENCE_MODELS_DIR", "models"))
    model_file = Path(os.getenv("INFERENCE_MODEL_FILE", "model.pkl"))
    manifest_file = Path(os.getenv("INFERENCE_MANIFEST_FILE", "feature_manifest.json"))

    model_path = Path(os.getenv("INFERENCE_MODEL_PATH", str(models_dir / model_file)))
    manifest_path = Path(os.getenv("INFERENCE_MANIFEST_PATH", str(models_dir / manifest_file)))

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Feature manifest not found: {manifest_path}")

    import pickle
    import json

    with open(model_path, "rb") as file:
        AppState.model = pickle.load(file)
    with open(manifest_path, "r", encoding="utf-8") as file:
        manifest = json.load(file)

    numeric_cols = manifest.get("numeric_columns", [])
    categorical_cols = manifest.get("categorical_columns", [])
    text_col = manifest.get("text_feature_column")
    expected = list(numeric_cols) + list(categorical_cols)
    if text_col:
        expected.append(text_col)
    AppState.expected_columns = expected
    logging.info("Inference app initialized with model=%s and %s features", model_path, len(expected))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Execute lifespan as part of the module workflow.

    Encapsulates a focused unit of pipeline logic for reuse and testing.
    """
    if os.getenv("APP_SKIP_MODEL_LOAD", "false").strip().lower() != "true":
        _load_artifacts()
    yield


app = FastAPI(
    title="Fraud Detection Inference API",
    version="1.0.0",
    description="Asynchronous fraud detection prediction service.",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> Dict[str, str]:
    """Execute health as part of the module workflow.

    Encapsulates a focused unit of pipeline logic for reuse and testing.
    Returns `Dict[str, str]`.
    """
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict(payload: PredictRequest) -> PredictResponse:
    """Execute predict as part of the module workflow.

    Encapsulates a focused unit of pipeline logic for reuse and testing.
    Returns `PredictResponse`.
    """
    try:
        if AppState.model is None:
            raise HTTPException(status_code=503, detail="Model is not loaded")

        frame = pd.DataFrame(payload.rows)
        missing = [col for col in AppState.expected_columns if col not in frame.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing required feature columns: {missing}")

        ordered = frame[AppState.expected_columns].copy()
        predictions = AppState.model.predict(ordered).tolist()
        probabilities = AppState.model.predict_proba(ordered)[:, 1].tolist()
        return PredictResponse(
            predictions=[int(item) for item in predictions],
            prediction_probabilities=[float(item) for item in probabilities],
        )
    except HTTPException:
        raise
    except Exception as exc:
        logging.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc

