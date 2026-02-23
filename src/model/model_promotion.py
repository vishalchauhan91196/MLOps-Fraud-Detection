import json
import warnings
from typing import Optional, Tuple

from src.core.exceptions import ModelRegistrationError
from src.core.io import safe_join
from src.core.s3 import get_json_if_exists, get_s3_client, put_json, upload_file
from src.core.settings import load_settings, require_nested
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
        raise ModelRegistrationError("Failed to initialize promotion tracking") from exc


def load_json(file_path: str, label: str) -> dict:
    """Load json from the configured source.

    Returns a parsed in-memory object for downstream processing.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            payload = json.load(file)
        if not isinstance(payload, dict):
            raise ModelRegistrationError(f"{label} file is not a JSON object: {file_path}")
        return payload
    except ModelRegistrationError:
        raise
    except Exception as exc:
        raise ModelRegistrationError(f"Failed to load {label}: {file_path}") from exc


def _get_registered_version(registration_payload: dict) -> str:
    """Return the registered model version from registration payload.

    Supports both the normalized `version` key and legacy `candidate_version`
    key for backward compatibility.
    """
    version = registration_payload.get("version")
    if version is None:
        version = registration_payload.get("candidate_version")
    if version is None:
        raise ModelRegistrationError("Registration payload missing required key: version")
    return str(version)


def build_s3_keys(s3_cfg: dict, env: str) -> Tuple[str, str]:
    """Construct s3 keys from the provided configuration and inputs.

    Returns an initialized object ready for model training or inference workflows.
    """
    prefix = str(require_nested(s3_cfg, "key_prefix")).strip("/")
    model_file = str(require_nested(s3_cfg, "model_file_name"))
    metrics_file = str(require_nested(s3_cfg, "metrics_file_name"))
    return f"{prefix}/{env}/{model_file}", f"{prefix}/{env}/{metrics_file}"


def read_best_score_from_s3(s3_cfg: dict, env: str, metric_name: str) -> Optional[float]:
    """Execute read best score from s3 as part of the module workflow.

    Encapsulates a focused unit of pipeline logic for reuse and testing.
    Returns `Optional[float]`.
    """
    if not bool(require_nested(s3_cfg, "enabled")):
        return None
    client = get_s3_client(str(require_nested(s3_cfg, "region")))
    bucket = str(require_nested(s3_cfg, "bucket"))
    _, metrics_key = build_s3_keys(s3_cfg, env)
    payload = get_json_if_exists(client, bucket, metrics_key)
    if payload is None:
        return None
    metric_value = payload.get("metrics", {}).get(metric_name)
    if metric_value is None:
        return None
    return float(metric_value)


def write_best_to_s3(s3_cfg: dict, env: str, model_local_path: str, payload: dict) -> None:
    """Persist best to s3 to the target destination.

    Creates or overwrites the target artifact needed by later pipeline stages.
    """
    if not bool(require_nested(s3_cfg, "enabled")):
        return
    client = get_s3_client(str(require_nested(s3_cfg, "region")))
    bucket = str(require_nested(s3_cfg, "bucket"))
    model_key, metrics_key = build_s3_keys(s3_cfg, env)
    upload_file(client, model_local_path, bucket, model_key)
    put_json(client, bucket, metrics_key, payload)


def should_promote(current_score: float, best_score: Optional[float], higher_is_better: bool) -> bool:
    """Execute should promote as part of the module workflow.

    Encapsulates a focused unit of pipeline logic for reuse and testing.
    Returns `bool`.
    """
    if best_score is None:
        return True
    if higher_is_better:
        return current_score > best_score
    return current_score < best_score


def get_alias_version(client, model_name: str, alias_name: str) -> Optional[str]:
    """Return alias version derived from the provided inputs and runtime context.

    Encapsulates lookup and fallback logic in a single reusable call.
    """
    try:
        mv = client.get_model_version_by_alias(model_name, alias_name)
        return str(mv.version)
    except Exception:
        return None


def promote_candidate(
    client,
    model_name: str,
    candidate_version: str,
    champion_version: Optional[str],
    champion_alias: str,
    challenger_alias: str,
    champion_stage: str,
    archived_stage: str,
) -> None:
    """Execute lifecycle actions for candidate in the model registry flow.

    Updates model versions, aliases, or metadata according to promotion policy.
    """
    try:
        if champion_version and champion_version != candidate_version:
            client.transition_model_version_stage(name=model_name, version=champion_version, stage=archived_stage)
            # Keep challenger pointing to the previous champion after promotion.
            client.set_registered_model_alias(model_name, challenger_alias, champion_version)

        client.transition_model_version_stage(name=model_name, version=candidate_version, stage=champion_stage)
        client.set_registered_model_alias(model_name, champion_alias, candidate_version)

        # Candidate moves from challenger to champion. If challenger alias still
        # points to candidate (first run / no previous champion), try to clear it.
        try:
            challenger_version = get_alias_version(client, model_name, challenger_alias)
            if challenger_version == candidate_version:
                client.delete_registered_model_alias(model_name, challenger_alias)
        except Exception:
            # Not all MLflow client versions expose delete alias APIs.
            pass
    except Exception as exc:
        raise ModelRegistrationError("Failed to promote model candidate") from exc


def keep_candidate_as_challenger(client, model_name: str, candidate_version: str, challenger_alias: str) -> None:
    """Execute keep candidate as challenger as part of the module workflow.

    Encapsulates a focused unit of pipeline logic for reuse and testing.
    """
    try:
        client.set_registered_model_alias(model_name, challenger_alias, candidate_version)
    except Exception as exc:
        raise ModelRegistrationError("Failed to set challenger alias") from exc


def main() -> None:
    """Run this module as a script entrypoint.

    Coordinates the stage workflow, logs failures, and re-raises exceptions to make pipeline execution status explicit.
    """
    try:
        settings = load_settings()
        config = settings.config
        env = settings.env

        promotion_cfg = require_nested(config, "model_promotion")
        mlflow = setup_tracking(require_nested(promotion_cfg, "mlflow"))
        if mlflow is None:
            logging.info("Model promotion skipped because model_promotion.mlflow.enabled is false")
            return

        reports_dir = safe_join(settings.root_dir, str(require_nested(config, "data.reports_dir")))
        models_dir = safe_join(settings.root_dir, str(require_nested(config, "data.models_dir")))
        registration_file = safe_join(reports_dir, str(require_nested(promotion_cfg, "registration_input_file")))
        metrics_file = safe_join(reports_dir, str(require_nested(promotion_cfg, "metrics_input_file")))
        retrain_signal_file = safe_join(reports_dir, str(require_nested(promotion_cfg, "retrain_signal_input_file")))
        local_model_file = safe_join(models_dir, str(require_nested(require_nested(config, "model_building"), "output_model_file")))

        registration_payload = load_json(registration_file, "registration payload")
        metrics = load_json(metrics_file, "metrics payload")
        retrain_signal = load_json(retrain_signal_file, "retrain signal payload")

        if bool(retrain_signal.get("should_retrain", False)):
            logging.warning("Promotion skipped for env=%s because retrain_signal.should_retrain is true", env)
            return

        model_name = str(require_nested(promotion_cfg, "model_name"))
        candidate_version = _get_registered_version(registration_payload)
        promotion_metric = str(require_nested(promotion_cfg, "promotion_metric"))
        metric_higher_is_better = bool(require_nested(promotion_cfg, "metric_higher_is_better"))
        champion_stage = str(require_nested(promotion_cfg, "champion_stage"))
        archived_stage = str(require_nested(promotion_cfg, "archived_stage"))
        aliases_cfg = require_nested(promotion_cfg, "aliases")
        champion_alias = str(require_nested(aliases_cfg, "champion"))
        challenger_alias = str(require_nested(aliases_cfg, "challenger"))
        s3_cfg = require_nested(promotion_cfg, "s3")

        if promotion_metric not in metrics:
            raise ModelRegistrationError(f"Promotion metric '{promotion_metric}' missing in metrics payload")
        current_score = float(metrics[promotion_metric])
        best_score = read_best_score_from_s3(s3_cfg=s3_cfg, env=env, metric_name=promotion_metric)

        client = mlflow.tracking.MlflowClient()
        champion_version = get_alias_version(client, model_name, champion_alias)

        if should_promote(current_score, best_score, metric_higher_is_better):
            promote_candidate(
                client=client,
                model_name=model_name,
                candidate_version=candidate_version,
                champion_version=champion_version,
                champion_alias=champion_alias,
                challenger_alias=challenger_alias,
                champion_stage=champion_stage,
                archived_stage=archived_stage,
            )
            write_best_to_s3(
                s3_cfg=s3_cfg,
                env=env,
                model_local_path=local_model_file,
                payload={
                    "environment": env,
                    "model_name": model_name,
                    "model_version": candidate_version,
                    "promotion_metric": promotion_metric,
                    "metrics": metrics,
                },
            )
            logging.info("Promoted candidate version=%s to champion for env=%s", candidate_version, env)
        else:
            keep_candidate_as_challenger(
                client=client,
                model_name=model_name,
                candidate_version=candidate_version,
                challenger_alias=challenger_alias,
            )
            logging.info("Candidate version=%s retained as challenger for env=%s", candidate_version, env)
    except Exception:
        logging.exception("Model promotion failed")
        raise


if __name__ == "__main__":
    main()

