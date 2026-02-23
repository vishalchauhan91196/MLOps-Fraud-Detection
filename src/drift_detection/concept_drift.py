import pandas as pd

from src.core.settings import require_nested


def detect_concept_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame, target_col: str, cfg: dict) -> dict:
    """Detect concept drift using input data and configured thresholds.

    Produces a structured drift or quality assessment for monitoring and reporting.
    """
    threshold = float(require_nested(cfg, "target_rate_delta_threshold"))
    if target_col not in reference_df.columns or target_col not in current_df.columns:
        return {"status": False, "message": f"Target column '{target_col}' not present for concept drift check"}

    ref_rate = float(pd.to_numeric(reference_df[target_col], errors="coerce").mean())
    cur_rate = float(pd.to_numeric(current_df[target_col], errors="coerce").mean())
    delta = abs(cur_rate - ref_rate)
    return {
        "status": delta > threshold,
        "reference_target_rate": ref_rate,
        "current_target_rate": cur_rate,
        "delta": delta,
        "threshold": threshold,
    }
