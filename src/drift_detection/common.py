from typing import Dict

import pandas as pd

from src.core.exceptions import ModelEvaluationError


def load_csv(path: str, label: str) -> pd.DataFrame:
    """Load csv from the configured source.

    Returns a parsed in-memory object for downstream processing.
    """
    try:
        df = pd.read_csv(path)
        if df.empty:
            raise ModelEvaluationError(f"{label} is empty: {path}")
        return df
    except ModelEvaluationError:
        raise
    except Exception as exc:
        raise ModelEvaluationError(f"Failed loading {label} from {path}") from exc


def load_json(path: str, label: str) -> Dict:
    """Load json from the configured source.

    Returns a parsed in-memory object for downstream processing.
    """
    try:
        import json

        with open(path, "r", encoding="utf-8") as file:
            payload = json.load(file)
        if not isinstance(payload, dict):
            raise ModelEvaluationError(f"{label} is not a JSON object: {path}")
        return payload
    except ModelEvaluationError:
        raise
    except Exception as exc:
        raise ModelEvaluationError(f"Failed loading {label} from {path}") from exc
