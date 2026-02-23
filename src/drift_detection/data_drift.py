from typing import Dict, List

import numpy as np
import pandas as pd

from src.core.settings import require_nested


def _distribution(values: pd.Series) -> Dict[str, float]:
    """Provide internal support for distribution.

    Used by this module to keep the main workflow functions focused and readable.
    """
    normalized = values.astype(str).value_counts(normalize=True, dropna=False)
    return {str(k): float(v) for k, v in normalized.items()}


def _compute_psi(reference: Dict[str, float], current: Dict[str, float], epsilon: float = 1e-6) -> float:
    """Provide internal support for compute psi.

    Used by this module to keep the main workflow functions focused and readable.
    """
    bins = sorted(set(reference.keys()) | set(current.keys()))
    ref = np.array([reference.get(item, epsilon) for item in bins], dtype=float)
    cur = np.array([current.get(item, epsilon) for item in bins], dtype=float)
    ref = np.clip(ref, epsilon, None)
    cur = np.clip(cur, epsilon, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def detect_data_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame, target_col: str, cfg: dict) -> dict:
    """Detect data drift using input data and configured thresholds.

    Produces a structured drift or quality assessment for monitoring and reporting.
    """
    numeric_threshold = float(require_nested(cfg, "numeric_mean_shift_threshold"))
    psi_threshold = float(require_nested(cfg, "categorical_psi_threshold"))
    ratio_threshold = float(require_nested(cfg, "drifted_feature_ratio_threshold"))

    feature_cols = [col for col in reference_df.columns if col in current_df.columns and col != target_col]
    drifted: List[dict] = []

    for col in feature_cols:
        ref_series = reference_df[col]
        cur_series = current_df[col]

        if pd.api.types.is_numeric_dtype(ref_series) and pd.api.types.is_numeric_dtype(cur_series):
            ref_mean = float(pd.to_numeric(ref_series, errors="coerce").mean())
            cur_mean = float(pd.to_numeric(cur_series, errors="coerce").mean())
            denom = abs(ref_mean) if abs(ref_mean) > 1e-6 else 1.0
            shift = abs(cur_mean - ref_mean) / denom
            if shift > numeric_threshold:
                drifted.append({"feature": col, "type": "numeric_mean_shift", "score": shift, "threshold": numeric_threshold})
        else:
            psi_score = _compute_psi(_distribution(ref_series), _distribution(cur_series))
            if psi_score > psi_threshold:
                drifted.append({"feature": col, "type": "categorical_psi", "score": psi_score, "threshold": psi_threshold})

    total_features = len(feature_cols)
    ratio = (len(drifted) / total_features) if total_features else 0.0
    return {
        "status": ratio > ratio_threshold,
        "drifted_feature_ratio": ratio,
        "drifted_feature_ratio_threshold": ratio_threshold,
        "drifted_features": drifted,
    }
