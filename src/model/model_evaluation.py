import os
import pickle
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.core.exceptions import ModelEvaluationError
from src.core.io import atomic_write_json, safe_join
from src.core.settings import load_settings, require_nested
from src.core.logger import logging


def setup_tracking(mlflow_cfg: dict):
    """Configure tracking for the current runtime environment.

    Initializes client or tracking settings before pipeline operations begin.
    """
    try:
        enabled = bool(require_nested(mlflow_cfg, "enabled"))
        if not enabled:
            return None

        import mlflow
        import mlflow.sklearn

        mlflow.set_tracking_uri(str(require_nested(mlflow_cfg, "tracking_uri")))
        return mlflow
    except Exception as exc:
        raise ModelEvaluationError("Failed to initialize MLflow tracking") from exc


def load_model(file_path: str):
    """Load model from the configured source.

    Returns a parsed in-memory object for downstream processing.
    """
    try:
        with open(file_path, "rb") as file:
            model = pickle.load(file)
        logging.info("Loaded model from %s", file_path)
        return model
    except Exception as exc:
        raise ModelEvaluationError(f"Failed to load model from {file_path}") from exc


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from the configured source.

    Returns a parsed in-memory object for downstream processing.
    """
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            raise ModelEvaluationError(f"Evaluation dataset is empty: {file_path}")
        logging.info("Loaded evaluation data from %s shape=%s", file_path, df.shape)
        return df
    except ModelEvaluationError:
        raise
    except Exception as exc:
        raise ModelEvaluationError(f"Failed to load evaluation data: {file_path}") from exc


def evaluate_model(clf, X_test: pd.DataFrame, y_test: np.ndarray) -> Tuple[Dict[str, float], Dict[str, object]]:
    """Execute evaluate model as part of the module workflow.

    Encapsulates a focused unit of pipeline logic for reuse and testing.
    Returns `Tuple[Dict[str, float], Dict[str, object]]`.
    """
    try:
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
            log_loss,
            precision_score,
            recall_score,
            roc_auc_score,
            average_precision_score,
        )

        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        metrics_dict = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "auc": float(roc_auc_score(y_test, y_pred_proba)),
            "pr_auc": float(average_precision_score(y_test, y_pred_proba)),
            "log_loss": float(log_loss(y_test, y_pred_proba)),
        }

        artifacts = {
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        }
        logging.info("Computed evaluation metrics and diagnostics")
        return metrics_dict, artifacts
    except Exception as exc:
        raise ModelEvaluationError("Model evaluation failed") from exc


def save_json(payload: dict, file_path: str, label: str) -> None:
    """Persist json to the target destination.

    Creates or overwrites the target artifact needed by later pipeline stages.
    """
    try:
        atomic_write_json(file_path, payload)
        logging.info("Saved %s to %s", label, file_path)
    except Exception as exc:
        raise ModelEvaluationError(f"Failed to save {label}: {file_path}") from exc


def save_model_info(payload: dict, file_path: str) -> None:
    """Persist model info to the target destination.

    Creates or overwrites the target artifact needed by later pipeline stages.
    """
    save_json(payload, file_path, "model info")


def _stringify_params(params: dict) -> dict:
    """Provide internal support for stringify params.

    Used by this module to keep the main workflow functions focused and readable.
    """
    safe_params = {}
    for key, value in params.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe_params[key] = value
        else:
            safe_params[key] = str(value)
    return safe_params


def main() -> None:
    """Run this module as a script entrypoint.

    Coordinates the stage workflow, logs failures, and re-raises exceptions to make pipeline execution status explicit.
    """
    try:
        settings = load_settings()
        config = settings.config
        env = settings.env

        eval_cfg = require_nested(config, "model_evaluation")
        mlflow_cfg = require_nested(eval_cfg, "mlflow")
        target_col = str(require_nested(require_nested(config, "pipeline"), "target_column"))

        processed_dir = safe_join(settings.root_dir, str(require_nested(config, "data.processed_dir")))
        models_dir = safe_join(settings.root_dir, str(require_nested(config, "data.models_dir")))
        reports_dir = safe_join(settings.root_dir, str(require_nested(config, "data.reports_dir")))

        test_file = safe_join(processed_dir, str(require_nested(eval_cfg, "test_input_file")))
        model_file = safe_join(models_dir, str(require_nested(require_nested(config, "model_building"), "output_model_file")))
        metrics_file = safe_join(reports_dir, str(require_nested(eval_cfg, "metrics_output_file")))
        model_info_file = safe_join(reports_dir, str(require_nested(eval_cfg, "model_info_output_file")))
        diagnostics_file = safe_join(reports_dir, "evaluation_diagnostics.json")

        mlflow = setup_tracking(mlflow_cfg)
        clf = load_model(model_file)
        test_data = load_data(test_file)

        if target_col not in test_data.columns:
            raise ModelEvaluationError(f"Target column missing in test data: {target_col}")

        X_test = test_data.drop(columns=[target_col])
        y_test = test_data[target_col].values

        metrics, diagnostics = evaluate_model(clf, X_test, y_test)
        save_json(metrics, metrics_file, "metrics")
        save_json(diagnostics, diagnostics_file, "evaluation diagnostics")

        if mlflow is None:
            save_model_info(
                {
                    "model_uri": model_file,
                    "run_id": None,
                    "environment": env,
                    "metrics": metrics,
                },
                model_info_file,
            )
            return

        experiment_name = str(require_nested(mlflow_cfg, "experiment_name"))
        mlflow.set_experiment(experiment_name)
        artifact_path = str(require_nested(mlflow_cfg, "log_model_artifact_path"))

        with mlflow.start_run() as run:
            mlflow.log_params({"environment": env, "target_column": target_col})
            mlflow.log_params(_stringify_params(clf.get_params() if hasattr(clf, "get_params") else {}))
            mlflow.log_metrics(metrics)

            model_info = mlflow.sklearn.log_model(clf, artifact_path)
            mlflow.log_artifact(metrics_file)
            mlflow.log_artifact(diagnostics_file)

            save_model_info(
                {
                    "model_uri": model_info.model_uri,
                    "run_id": run.info.run_id,
                    "environment": env,
                    "metrics": metrics,
                },
                model_info_file,
            )
    except Exception:
        logging.exception("Model evaluation failed")
        raise


if __name__ == "__main__":
    main()

