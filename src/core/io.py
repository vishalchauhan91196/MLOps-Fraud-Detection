import json
import os
import tempfile

import pandas as pd
import yaml

from src.core.exceptions import PipelineIOError, SecurityError


def safe_join(root_dir: str, relative_path: str) -> str:
    """Safely resolve a relative path under project root."""
    try:
        root_abs = os.path.abspath(root_dir)
        candidate_abs = os.path.abspath(os.path.join(root_abs, relative_path))
        if os.path.commonpath([root_abs, candidate_abs]) != root_abs:
            raise SecurityError(f"Path escapes project root: {relative_path}")
        return candidate_abs
    except SecurityError:
        raise
    except Exception as exc:
        raise SecurityError(f"Failed to resolve safe path for: {relative_path}") from exc


def atomic_write_yaml(path: str, payload: dict) -> None:
    """Atomically write YAML content to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".yaml", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            yaml.safe_dump(payload, file, sort_keys=False)
        os.replace(tmp_path, path)
    except Exception as exc:
        raise PipelineIOError(f"Failed atomic YAML write to {path}") from exc
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def atomic_write_json(path: str, payload: dict) -> None:
    """Atomically write JSON content to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=4)
        os.replace(tmp_path, path)
    except Exception as exc:
        raise PipelineIOError(f"Failed atomic JSON write to {path}") from exc
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def atomic_write_csv(df: pd.DataFrame, path: str) -> None:
    """Atomically write CSV content to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".csv", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as file:
            df.to_csv(file, index=False)
        os.replace(tmp_path, path)
    except Exception as exc:
        raise PipelineIOError(f"Failed atomic CSV write to {path}") from exc
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

