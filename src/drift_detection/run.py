from src.core.io import atomic_write_json, safe_join
from src.core.settings import load_settings, require_nested
from src.core.logger import logging
from src.drift_detection.common import load_csv, load_json
from src.drift_detection.concept_drift import detect_concept_drift
from src.drift_detection.data_drift import detect_data_drift
from src.drift_detection.model_drift import detect_model_drift, load_baseline_metric_from_s3


def main() -> None:
    """Run this module as a script entrypoint.

    Coordinates the stage workflow, logs failures, and re-raises exceptions to make pipeline execution status explicit.
    """
    try:
        settings = load_settings()
        config = settings.config
        env = settings.env
        drift_cfg = require_nested(config, "drift_detection")
        target_col = str(require_nested(config, "pipeline.target_column"))

        processed_dir = safe_join(settings.root_dir, str(require_nested(config, "data.processed_dir")))
        reports_dir = safe_join(settings.root_dir, str(require_nested(config, "data.reports_dir")))

        reference_path = safe_join(processed_dir, str(require_nested(drift_cfg, "reference_data_file")))
        current_path = safe_join(processed_dir, str(require_nested(drift_cfg, "current_data_file")))
        metrics_path = safe_join(reports_dir, str(require_nested(drift_cfg, "metrics_input_file")))
        report_path = safe_join(reports_dir, str(require_nested(drift_cfg, "output_report_file")))
        retrain_signal_path = safe_join(reports_dir, str(require_nested(drift_cfg, "output_retrain_signal_file")))

        reference_df = load_csv(reference_path, "reference dataset")
        current_df = load_csv(current_path, "current dataset")
        current_metrics = load_json(metrics_path, "current metrics")

        data_drift = detect_data_drift(
            reference_df=reference_df,
            current_df=current_df,
            target_col=target_col,
            cfg=require_nested(drift_cfg, "data_drift"),
        )
        concept_drift = detect_concept_drift(
            reference_df=reference_df,
            current_df=current_df,
            target_col=target_col,
            cfg=require_nested(drift_cfg, "concept_drift"),
        )
        metric_name = str(require_nested(require_nested(drift_cfg, "model_drift"), "metric_name"))
        baseline_metric = load_baseline_metric_from_s3(require_nested(drift_cfg, "baseline"), env=env, metric_name=metric_name)
        model_drift = detect_model_drift(
            current_metrics=current_metrics,
            baseline_metric=baseline_metric,
            cfg=require_nested(drift_cfg, "model_drift"),
        )

        should_retrain = bool(data_drift["status"] or concept_drift["status"] or model_drift["status"])
        report_payload = {
            "environment": env,
            "data_drift": data_drift,
            "concept_drift": concept_drift,
            "model_drift": model_drift,
            "should_retrain": should_retrain,
        }
        retrain_payload = {
            "environment": env,
            "should_retrain": should_retrain,
            "reason": {
                "data_drift": bool(data_drift["status"]),
                "concept_drift": bool(concept_drift["status"]),
                "model_drift": bool(model_drift["status"]),
            },
        }

        atomic_write_json(report_path, report_payload)
        atomic_write_json(retrain_signal_path, retrain_payload)
        logging.info("Drift detection completed should_retrain=%s env=%s", should_retrain, env)
    except Exception:
        logging.exception("Drift detection failed")
        raise


if __name__ == "__main__":
    main()

