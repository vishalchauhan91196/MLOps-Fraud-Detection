import pandas as pd

from src.drift_detection.concept_drift import detect_concept_drift
from src.drift_detection.data_drift import detect_data_drift
from src.drift_detection.model_drift import detect_model_drift


def test_data_drift_detects_numeric_shift():
    """Validate that data drift detects numeric shift behaves as expected.

    This test guards against regressions in the associated workflow or API behavior.
    """
    reference = pd.DataFrame({"age": [30, 31, 29, 30], "fraud_reported": [0, 0, 1, 0]})
    current = pd.DataFrame({"age": [80, 82, 79, 81], "fraud_reported": [0, 1, 1, 1]})
    cfg = {
        "numeric_mean_shift_threshold": 0.1,
        "categorical_psi_threshold": 0.2,
        "drifted_feature_ratio_threshold": 0.1,
    }
    result = detect_data_drift(reference, current, target_col="fraud_reported", cfg=cfg)
    assert result["status"] is True
    assert result["drifted_feature_ratio"] > 0


def test_concept_drift_detects_target_shift():
    """Validate that concept drift detects target shift behaves as expected.

    This test guards against regressions in the associated workflow or API behavior.
    """
    reference = pd.DataFrame({"fraud_reported": [0, 0, 0, 1]})
    current = pd.DataFrame({"fraud_reported": [1, 1, 1, 1]})
    result = detect_concept_drift(reference, current, "fraud_reported", {"target_rate_delta_threshold": 0.2})
    assert result["status"] is True


def test_model_drift_detects_relative_drop():
    """Validate that model drift detects relative drop behaves as expected.

    This test guards against regressions in the associated workflow or API behavior.
    """
    result = detect_model_drift(
        current_metrics={"auc": 0.70},
        baseline_metric=0.85,
        cfg={"metric_name": "auc", "relative_drop_threshold": 0.1},
    )
    assert result["status"] is True

