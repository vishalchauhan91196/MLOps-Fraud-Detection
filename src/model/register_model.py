import json
import warnings
from typing import Optional

from src.core.exceptions import ModelRegistrationError
from src.core.io import atomic_write_json, safe_join
from src.core.settings import get_nested, load_settings, require_nested
from src.core.logger import logging

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")


def setup_tracking(mlflow_cfg: dict):
    """Configure tracking for the current runtime environment.

    Initializes client or tracking settings before pipeline operations begin.
    """
    try:
        enabled = bool(require_nested(mlflow_cfg, "enabled"))
        if not enabled:
            return None
        import mlflow

        mlflow.set_tracking_uri(str(require_nested(mlflow_cfg, "tracking_uri")))
        return mlflow
    except Exception as exc:
        raise ModelRegistrationError("Failed to initialize registration tracking") from exc


def load_model_info(file_path: str) -> dict:
    """Load model info from the configured source.

    Returns a parsed in-memory object for downstream processing.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            payload = json.load(file)
        if "model_uri" not in payload:
            raise ModelRegistrationError(f"model_uri missing in model info file: {file_path}")
        return payload
    except ModelRegistrationError:
        raise
    except Exception as exc:
        raise ModelRegistrationError(f"Failed to load model info file: {file_path}") from exc


def register_candidate(
    mlflow,
    model_name: str,
    model_uri: str,
    stage: str,
    alias: Optional[str] = None,
) -> dict:
    """Execute lifecycle actions for candidate in the model registry flow.

    Updates model versions, aliases, or metadata according to promotion policy.
    """
    try:
        model_version = mlflow.register_model(model_uri, model_name)
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(name=model_name, version=model_version.version, stage=stage)
        if alias:
            client.set_registered_model_alias(model_name, alias, model_version.version)
        return {
            "model_name": model_name,
            "version": str(model_version.version),
            "stage": stage,
            "alias": alias,
            "registered_model_uri": model_uri,
        }
    except Exception as exc:
        raise ModelRegistrationError("Failed to register candidate model") from exc


def main() -> None:
    """Run this module as a script entrypoint.

    Coordinates the stage workflow, logs failures, and re-raises exceptions to make pipeline execution status explicit.
    """
    try:
        settings = load_settings()
        config = settings.config
        env = settings.env

        reg_cfg = require_nested(config, "model_registration")
        mlflow = setup_tracking(require_nested(reg_cfg, "mlflow"))
        if mlflow is None:
            logging.info("Model registration skipped because model_registration.mlflow.enabled is false")
            return

        reports_dir = safe_join(settings.root_dir, str(require_nested(config, "data.reports_dir")))
        model_info_path = safe_join(reports_dir, str(require_nested(reg_cfg, "model_info_input_file")))
        registration_out = safe_join(reports_dir, str(require_nested(reg_cfg, "registration_output_file")))

        model_info = load_model_info(model_info_path)
        alias = get_nested(config, "model_promotion.aliases.challenger", default=None)
        result = register_candidate(
            mlflow=mlflow,
            model_name=str(require_nested(reg_cfg, "model_name")),
            model_uri=str(require_nested(model_info, "model_uri")),
            stage=str(require_nested(reg_cfg, "candidate_stage")),
            alias=str(alias) if alias else None,
        )
        result["environment"] = env
        atomic_write_json(registration_out, result)
        logging.info("Model registration completed version=%s", result["version"])
    except Exception:
        logging.exception("Model registration failed")
        raise


if __name__ == "__main__":
    main()

