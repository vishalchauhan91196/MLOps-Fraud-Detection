import os
from typing import List, Tuple

import pandas as pd

from src.core.exceptions import FeatureEngineeringError
from src.core.io import atomic_write_csv, atomic_write_json, safe_join
from src.core.settings import load_settings, require_nested
from src.core.logger import logging


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from the configured source.

    Returns a parsed in-memory object for downstream processing.
    """
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            raise FeatureEngineeringError(f"Input data is empty: {file_path}")
        logging.info("Loaded data from %s with shape=%s", file_path, df.shape)
        return df
    except FeatureEngineeringError:
        raise
    except Exception as exc:
        raise FeatureEngineeringError(f"Failed to load feature input: {file_path}") from exc


def build_feature_frame(
    df: pd.DataFrame,
    target_col: str,
    text_source_columns: List[str],
    combined_text_column: str,
    include_original_text_columns: bool,
) -> pd.DataFrame:
    """Construct feature frame from the provided configuration and inputs.

    Returns an initialized object ready for model training or inference workflows.
    """
    if target_col not in df.columns:
        raise FeatureEngineeringError(f"Target column missing in feature input: {target_col}")

    working = df.copy()
    available_text_cols = [col for col in text_source_columns if col in working.columns]

    if available_text_cols:
        working[combined_text_column] = (
            working[available_text_cols]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
            .str.strip()
        )
    else:
        working[combined_text_column] = ""

    if not include_original_text_columns and available_text_cols:
        working = working.drop(columns=available_text_cols)

    non_target_cols = [col for col in working.columns if col != target_col]
    ordered_cols = non_target_cols + [target_col]
    return working[ordered_cols]


def get_feature_manifest(df: pd.DataFrame, target_col: str, combined_text_column: str) -> dict:
    """Return feature manifest derived from the provided inputs and runtime context.

    Encapsulates lookup and fallback logic in a single reusable call.
    """
    feature_cols = [col for col in df.columns if col != target_col]
    numeric_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(df[col])]
    categorical_cols = [col for col in feature_cols if col not in numeric_cols and col != combined_text_column]

    return {
        "target_column": target_col,
        "text_feature_column": combined_text_column,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
    }


def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Persist data to the target destination.

    Creates or overwrites the target artifact needed by later pipeline stages.
    """
    try:
        atomic_write_csv(df, file_path)
        logging.info("Saved feature data to %s shape=%s", file_path, df.shape)
    except Exception as exc:
        raise FeatureEngineeringError(f"Failed to save feature data: {file_path}") from exc


def save_manifest(manifest: dict, file_path: str) -> None:
    """Persist manifest to the target destination.

    Creates or overwrites the target artifact needed by later pipeline stages.
    """
    try:
        atomic_write_json(file_path, manifest)
        logging.info("Saved feature manifest to %s", file_path)
    except Exception as exc:
        raise FeatureEngineeringError(f"Failed to save feature manifest: {file_path}") from exc


def main() -> None:
    """Run this module as a script entrypoint.

    Coordinates the stage workflow, logs failures, and re-raises exceptions to make pipeline execution status explicit.
    """
    try:
        settings = load_settings()
        config = settings.config

        fe_cfg = require_nested(config, "feature_engineering")
        target_col = str(require_nested(fe_cfg, "target_column"))
        text_source_columns = list(require_nested(fe_cfg, "text_source_columns"))
        combined_text_column = str(require_nested(fe_cfg, "combined_text_column"))
        include_original_text_columns = bool(require_nested(fe_cfg, "include_original_text_columns"))

        interim_dir = safe_join(settings.root_dir, str(require_nested(config, "data.interim_dir")))
        processed_dir = safe_join(settings.root_dir, str(require_nested(config, "data.processed_dir")))
        models_dir = safe_join(settings.root_dir, str(require_nested(config, "data.models_dir")))

        train_in = safe_join(interim_dir, str(require_nested(fe_cfg, "train_input_file")))
        test_in = safe_join(interim_dir, str(require_nested(fe_cfg, "test_input_file")))
        train_out = safe_join(processed_dir, str(require_nested(fe_cfg, "train_output_file")))
        test_out = safe_join(processed_dir, str(require_nested(fe_cfg, "test_output_file")))
        manifest_out = safe_join(models_dir, str(require_nested(fe_cfg, "output_manifest_file")))

        train_df = load_data(train_in)
        test_df = load_data(test_in)

        train_features = build_feature_frame(
            train_df,
            target_col=target_col,
            text_source_columns=text_source_columns,
            combined_text_column=combined_text_column,
            include_original_text_columns=include_original_text_columns,
        )
        test_features = build_feature_frame(
            test_df,
            target_col=target_col,
            text_source_columns=text_source_columns,
            combined_text_column=combined_text_column,
            include_original_text_columns=include_original_text_columns,
        )

        save_data(train_features, train_out)
        save_data(test_features, test_out)

        manifest = get_feature_manifest(train_features, target_col=target_col, combined_text_column=combined_text_column)
        save_manifest(manifest, manifest_out)

    except Exception:
        logging.exception("Feature engineering failed")
        raise


if __name__ == "__main__":
    main()

