from typing import Optional

from src.core.exceptions import ModelEvaluationError
from src.core.logger import logging
from src.core.s3 import get_json_if_exists, get_s3_client
from src.core.settings import require_nested


def load_baseline_metric_from_s3(baseline_cfg: dict, env: str, metric_name: str) -> Optional[float]:
    """Load baseline metric from s3 from the configured source.

    Returns a parsed in-memory object for downstream processing.
    """
    source = str(require_nested(baseline_cfg, "source")).lower()
    if source != "s3":
        return None
    s3_cfg = require_nested(baseline_cfg, "s3")
    if not bool(require_nested(s3_cfg, "enabled")):
        return None

    try:
        client = get_s3_client(str(require_nested(s3_cfg, "region")))
        bucket = str(require_nested(s3_cfg, "bucket"))
        key_prefix = str(require_nested(s3_cfg, "key_prefix")).strip("/")
        metrics_file = str(require_nested(s3_cfg, "metrics_file_name"))
        key = f"{key_prefix}/{env}/{metrics_file}"

        payload = get_json_if_exists(client, bucket, key)
        if payload is None:
            return None
        metric_value = payload.get("metrics", {}).get(metric_name)
        if metric_value is None:
            return None
        return float(metric_value)
    except Exception as exc:
        logging.warning("Unable to fetch baseline metric from S3; continuing without baseline: %s", exc)
        return None


def detect_model_drift(current_metrics: dict, baseline_metric: Optional[float], cfg: dict) -> dict:
    """Detect model drift using input data and configured thresholds.

    Produces a structured drift or quality assessment for monitoring and reporting.
    """
    metric_name = str(require_nested(cfg, "metric_name"))
    relative_drop_threshold = float(require_nested(cfg, "relative_drop_threshold"))

    if metric_name not in current_metrics:
        raise ModelEvaluationError(f"Model drift metric '{metric_name}' missing from current metrics payload")
    current_metric = float(current_metrics[metric_name])

    if baseline_metric is None:
        return {
            "status": False,
            "message": "No baseline metric available; treating as no model drift for bootstrap run",
            "metric_name": metric_name,
            "current_metric": current_metric,
            "baseline_metric": None,
            "relative_drop": 0.0,
            "relative_drop_threshold": relative_drop_threshold,
        }

    denom = abs(baseline_metric) if abs(baseline_metric) > 1e-6 else 1.0
    relative_drop = max(0.0, (baseline_metric - current_metric) / denom)
    return {
        "status": relative_drop > relative_drop_threshold,
        "metric_name": metric_name,
        "current_metric": current_metric,
        "baseline_metric": baseline_metric,
        "relative_drop": relative_drop,
        "relative_drop_threshold": relative_drop_threshold,
    }
