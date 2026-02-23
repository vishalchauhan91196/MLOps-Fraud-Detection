import os
import pickle
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import LabelEncoder

from src.core.exceptions import DataPreprocessingError
from src.core.io import atomic_write_csv, safe_join
from src.core.settings import load_settings, require_nested
from src.data.transformation import apply_smoten, transform_dataframe
from src.core.logger import logging


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from the configured source.

    Returns a parsed in-memory object for downstream processing.
    """
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            raise DataPreprocessingError(f"Input file is empty: {file_path}")
        logging.info("Loaded dataframe from %s with shape %s", file_path, df.shape)
        return df
    except DataPreprocessingError:
        raise
    except Exception as exc:
        raise DataPreprocessingError(f"Failed to load dataframe from {file_path}") from exc


def rename_columns(df: pd.DataFrame, rename_map: Dict[str, str]) -> pd.DataFrame:
    """Execute rename columns as part of the module workflow.

    Encapsulates a focused unit of pipeline logic for reuse and testing.
    Returns `pd.DataFrame`.
    """
    renamed_df = df.rename(columns=rename_map).copy()
    renamed_df.columns = [col.strip().lower().replace("-", "_").replace(" ", "_") for col in renamed_df.columns]
    return renamed_df


def normalize_categorical_values(df: pd.DataFrame, value_maps: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    """Transform categorical values into the format required by later steps.

    Applies deterministic preprocessing so training and inference stay consistent.
    """
    normalized_df = df.copy()
    object_cols = normalized_df.select_dtypes(include=["object"]).columns

    if len(object_cols) == 0:
        return normalized_df

    normalized_df[object_cols] = normalized_df[object_cols].apply(lambda s: s.astype(str).str.strip())

    for col, mapper in value_maps.items():
        if col in normalized_df.columns:
            lowered = normalized_df[col].astype(str).str.lower()
            normalized_df[col] = lowered.map(lambda x: mapper.get(x, x.upper()))

    return normalized_df


def replace_placeholder_missing_values(df: pd.DataFrame, null_tokens: list) -> pd.DataFrame:
    """Execute replace placeholder missing values as part of the module workflow.

    Encapsulates a focused unit of pipeline logic for reuse and testing.
    Returns `pd.DataFrame`.
    """
    cleaned_df = df.copy()
    object_cols = cleaned_df.select_dtypes(include=["object"]).columns

    if len(object_cols) > 0:
        null_set = {str(t).strip().lower() for t in null_tokens}
        cleaned_df[object_cols] = cleaned_df[object_cols].apply(
            lambda s: s.mask(s.astype(str).str.strip().str.lower().isin(null_set), np.nan)
        )

    return cleaned_df


def drop_duplicates_and_empty_rows(df: pd.DataFrame, drop_duplicates: bool, drop_all_na_rows: bool) -> pd.DataFrame:
    """Execute drop duplicates and empty rows as part of the module workflow.

    Encapsulates a focused unit of pipeline logic for reuse and testing.
    Returns `pd.DataFrame`.
    """
    clean_df = df
    if drop_duplicates:
        clean_df = clean_df.drop_duplicates()
    if drop_all_na_rows:
        clean_df = clean_df.dropna(how="all")
    return clean_df


def coerce_numeric_columns(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Transform numeric columns into the format required by later steps.

    Applies deterministic preprocessing so training and inference stay consistent.
    """
    coerced_df = df.copy()
    for col in coerced_df.columns:
        if col == target_col:
            continue
        if coerced_df[col].dtype == "object":
            candidate = pd.to_numeric(coerced_df[col], errors="coerce")
            if float(candidate.notna().mean()) >= 0.90:
                coerced_df[col] = candidate
    return coerced_df


def fill_missing_values(df: pd.DataFrame, categorical_fill_value: str) -> pd.DataFrame:
    """Execute fill missing values as part of the module workflow.

    Encapsulates a focused unit of pipeline logic for reuse and testing.
    Returns `pd.DataFrame`.
    """
    filled_df = df.copy()

    numeric_cols = filled_df.select_dtypes(include=[np.number]).columns
    categorical_cols = filled_df.select_dtypes(exclude=[np.number]).columns

    if len(numeric_cols) > 0:
        medians = filled_df[numeric_cols].median()
        filled_df[numeric_cols] = filled_df[numeric_cols].fillna(medians)

    for col in categorical_cols:
        mode_value = filled_df[col].mode(dropna=True)
        replacement = mode_value.iloc[0] if not mode_value.empty else categorical_fill_value
        filled_df[col] = filled_df[col].fillna(replacement)

    return filled_df


def encode_target_column(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, LabelEncoder]:
    """Transform target column into the format required by later steps.

    Applies deterministic preprocessing so training and inference stay consistent.
    """
    label_encoder = LabelEncoder()

    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df[target_col] = label_encoder.fit_transform(train_df[target_col].astype(str))
    mapping: Dict[str, int] = {class_label: int(index) for index, class_label in enumerate(label_encoder.classes_)}
    test_df[target_col] = test_df[target_col].astype(str).map(mapping)

    if test_df[target_col].isna().any():
        unknown_labels = test_df.loc[test_df[target_col].isna(), target_col].astype(str).unique().tolist()
        raise DataPreprocessingError(f"Unknown labels found in test data: {unknown_labels}")

    test_df[target_col] = test_df[target_col].astype(int)
    return train_df, test_df, label_encoder


def save_dataframe(df: pd.DataFrame, output_path: str) -> None:
    """Persist dataframe to the target destination.

    Creates or overwrites the target artifact needed by later pipeline stages.
    """
    atomic_write_csv(df, output_path)
    logging.info("Saved dataframe to %s with shape %s", output_path, df.shape)


def save_label_encoder(label_encoder: LabelEncoder, output_path: str) -> None:
    """Persist label encoder to the target destination.

    Creates or overwrites the target artifact needed by later pipeline stages.
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as file:
            pickle.dump(label_encoder, file)
        logging.info("Saved label encoder to %s", output_path)
    except Exception as exc:
        raise DataPreprocessingError(f"Failed saving label encoder to {output_path}") from exc


def ensure_validation_passed(report_path: str) -> None:
    """Execute ensure validation passed as part of the module workflow.

    Encapsulates a focused unit of pipeline logic for reuse and testing.
    """
    if not os.path.exists(report_path):
        raise DataPreprocessingError(f"Validation report not found at {report_path}. Run validation first.")

    with open(report_path, "r", encoding="utf-8") as file:
        report = yaml.safe_load(file)

    if not isinstance(report, dict):
        raise DataPreprocessingError("Validation report is not a valid YAML object.")
    if "validation_status" not in report:
        raise DataPreprocessingError("Validation report missing required key: validation_status")
    if "message" not in report:
        raise DataPreprocessingError("Validation report missing required key: message")

    status = bool(report["validation_status"])
    message = str(report["message"])
    if not status:
        raise DataPreprocessingError(f"Validation failed. Preprocessing aborted. Details: {message}")


def main() -> None:
    """Run this module as a script entrypoint.

    Coordinates the stage workflow, logs failures, and re-raises exceptions to make pipeline execution status explicit.
    """
    try:
        settings = load_settings()
        config = settings.config
        env = settings.env

        target_col = str(require_nested(config, "pipeline.target_column"))
        raw_dir = safe_join(settings.root_dir, str(require_nested(config, "data.raw_dir")))
        interim_dir = safe_join(settings.root_dir, str(require_nested(config, "data.interim_dir")))
        reports_dir = safe_join(settings.root_dir, str(require_nested(config, "data.reports_dir")))
        models_dir = safe_join(settings.root_dir, str(require_nested(config, "data.models_dir")))

        train_raw_path = os.path.join(raw_dir, "train.csv")
        test_raw_path = os.path.join(raw_dir, "test.csv")
        train_out_path = os.path.join(interim_dir, "train_processed.csv")
        test_out_path = os.path.join(interim_dir, "test_processed.csv")
        report_path = os.path.join(reports_dir, "report.yaml")
        encoder_path = os.path.join(models_dir, "label_encoder.pkl")

        null_tokens = list(require_nested(config, "preprocessing.placeholder_null_tokens"))
        drop_duplicates = bool(require_nested(config, "preprocessing.drop_duplicates"))
        drop_all_na_rows = bool(require_nested(config, "preprocessing.drop_all_na_rows"))
        apply_smote = bool(require_nested(config, "preprocessing.apply_smoten"))
        save_encoder = bool(require_nested(config, "preprocessing.save_label_encoder"))
        rename_map = dict(require_nested(config, "preprocessing.rename_columns_map"))
        value_maps = dict(require_nested(config, "preprocessing.categorical_value_maps"))
        categorical_fill_value = str(require_nested(config, "preprocessing.categorical_fill_value"))
        random_state = int(require_nested(config, "pipeline.random_state"))
        transformation_config = require_nested(config, "transformation")

        logging.info("Starting preprocessing for env=%s", env)
        ensure_validation_passed(report_path)

        train_df = load_data(train_raw_path)
        test_df = load_data(test_raw_path)

        train_df = rename_columns(train_df, rename_map=rename_map)
        test_df = rename_columns(test_df, rename_map=rename_map)

        train_df = normalize_categorical_values(train_df, value_maps=value_maps)
        test_df = normalize_categorical_values(test_df, value_maps=value_maps)

        train_df = replace_placeholder_missing_values(train_df, null_tokens)
        test_df = replace_placeholder_missing_values(test_df, null_tokens)

        train_df = drop_duplicates_and_empty_rows(train_df, drop_duplicates, drop_all_na_rows)
        test_df = drop_duplicates_and_empty_rows(test_df, drop_duplicates, drop_all_na_rows)

        train_df = coerce_numeric_columns(train_df, target_col)
        test_df = coerce_numeric_columns(test_df, target_col)

        train_df = transform_dataframe(train_df, config=transformation_config)
        test_df = transform_dataframe(test_df, config=transformation_config)

        train_df = fill_missing_values(train_df, categorical_fill_value=categorical_fill_value)
        test_df = fill_missing_values(test_df, categorical_fill_value=categorical_fill_value)

        train_df = train_df.dropna(subset=[target_col])
        test_df = test_df.dropna(subset=[target_col])

        train_df, test_df, label_encoder = encode_target_column(train_df, test_df, target_col)

        X_train, y_train = train_df.drop(columns=[target_col]), train_df[target_col]
        X_train, y_train = apply_smoten(X_train, y_train, enabled=apply_smote, random_state=random_state)
        train_final = pd.concat([X_train, y_train], axis=1)

        # Keep raw engineered columns (no one-hot here). Feature engineering/model pipeline handles encoding/scaling/tfidf.
        save_dataframe(train_final, train_out_path)
        save_dataframe(test_df, test_out_path)

        if save_encoder:
            save_label_encoder(label_encoder, encoder_path)

        logging.info("Preprocessing completed for env=%s", env)
    except Exception:
        logging.exception("Preprocessing failed")
        raise


if __name__ == "__main__":
    main()

