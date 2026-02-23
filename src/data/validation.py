import os
from typing import Any, Dict, List, Tuple

import pandas as pd
import yaml

from src.core.exceptions import DataValidationError
from src.core.io import atomic_write_yaml, safe_join
from src.core.settings import load_settings, require_nested
from src.core.logger import logging


def load_schema(schema_path: str) -> dict:
    """Load schema from the configured source.

    Returns a parsed in-memory object for downstream processing.
    """
    try:
        with open(schema_path, "r", encoding="utf-8") as file:
            schema = yaml.safe_load(file)
        if not isinstance(schema, dict):
            raise DataValidationError(f"Schema must be a mapping object: {schema_path}")
        logging.info("Loaded schema from %s", schema_path)
        return schema
    except Exception as exc:
        raise DataValidationError(f"Failed to load schema: {schema_path}") from exc


def require_schema_key(schema: dict, key: str, expected_type: type) -> Any:
    """Validate schema key against expected constraints.

    Returns validation feedback or raises an error when required conditions are not met.
    """
    if key not in schema:
        raise DataValidationError(f"Schema missing required key: {key}")
    value = schema[key]
    if not isinstance(value, expected_type):
        raise DataValidationError(f"Schema key '{key}' has invalid type: expected {expected_type.__name__}")
    return value


def validate_schema_structure(df: pd.DataFrame, schema: dict, file_name: str, strict_column_match: bool, fail_on_extra: bool) -> List[str]:
    """Validate schema structure against expected constraints.

    Returns validation feedback or raises an error when required conditions are not met.
    """
    messages: List[str] = []

    expected_columns = require_schema_key(schema, "columns", list)
    expected_num_columns = require_schema_key(schema, "number_of_columns", int)
    numerical_columns = require_schema_key(schema, "numerical_columns", list)
    categorical_columns = require_schema_key(schema, "categorical_columns", list)

    if expected_num_columns is not None and df.shape[1] != int(expected_num_columns):
        messages.append(f"{file_name}: expected {expected_num_columns} columns, found {df.shape[1]}.")

    missing_expected = sorted(set(expected_columns) - set(df.columns))
    extra_columns = sorted(set(df.columns) - set(expected_columns))

    if strict_column_match and missing_expected:
        messages.append(f"{file_name}: missing expected columns: {missing_expected}.")

    if fail_on_extra and extra_columns:
        messages.append(f"{file_name}: unexpected extra columns found: {extra_columns}.")

    missing_numerical = sorted(set(numerical_columns) - set(df.columns))
    missing_categorical = sorted(set(categorical_columns) - set(df.columns))

    if missing_numerical:
        messages.append(f"{file_name}: missing numerical columns: {missing_numerical}.")
    if missing_categorical:
        messages.append(f"{file_name}: missing categorical columns: {missing_categorical}.")

    return messages


def validate_data_quality(df: pd.DataFrame, file_name: str, config: dict) -> List[str]:
    """Validate data quality against expected constraints.

    Returns validation feedback or raises an error when required conditions are not met.
    """
    messages: List[str] = []
    null_tokens = {str(token).strip().lower() for token in require_nested(config, "validation.null_tokens")}

    duplicate_ratio = float(df.duplicated().mean())
    max_duplicate_ratio = float(require_nested(config, "validation.max_duplicate_ratio"))
    if duplicate_ratio > max_duplicate_ratio:
        messages.append(f"{file_name}: duplicate ratio {duplicate_ratio:.4f} exceeds threshold {max_duplicate_ratio:.4f}.")

    critical_thresholds = require_nested(config, "validation.critical_missing_thresholds")
    for column, threshold in critical_thresholds.items():
        if column not in df.columns:
            messages.append(f"{file_name}: critical column missing for null check: {column}.")
            continue

        series = df[column]
        lowered = series.astype(str).str.strip().str.lower()
        missing_ratio = float((series.isna() | lowered.isin(null_tokens)).mean())
        if missing_ratio > float(threshold):
            messages.append(f"{file_name}: missing ratio in {column} is {missing_ratio:.4f}, threshold {float(threshold):.4f}.")

    target_col = str(require_nested(config, "pipeline.target_column"))
    if bool(require_nested(config, "validation.enforce_target_values")):
        allowed_values = set(str(v).upper() for v in require_nested(config, "validation.target_allowed_values"))
        if target_col in df.columns:
            observed = set(df[target_col].dropna().astype(str).str.upper().unique())
            invalid = sorted(v for v in observed if v not in allowed_values)
            if invalid:
                messages.append(f"{file_name}: invalid target labels found: {invalid}.")
        else:
            messages.append(f"{file_name}: target column missing: {target_col}.")

    numeric_bounds = require_nested(config, "validation.numeric_bounds")
    for column, bounds in numeric_bounds.items():
        if column not in df.columns:
            continue
        numeric_series = pd.to_numeric(df[column], errors="coerce")
        min_value = bounds.get("min")
        max_value = bounds.get("max")
        if min_value is not None:
            low_count = int((numeric_series < float(min_value)).sum())
            if low_count > 0:
                messages.append(f"{file_name}: {column} has {low_count} rows below min {min_value}.")
        if max_value is not None:
            high_count = int((numeric_series > float(max_value)).sum())
            if high_count > 0:
                messages.append(f"{file_name}: {column} has {high_count} rows above max {max_value}.")

    if {"injury_claim", "property_claim", "vehicle_claim", "total_claim_amount"}.issubset(df.columns):
        claim_total = (
            pd.to_numeric(df["injury_claim"], errors="coerce")
            + pd.to_numeric(df["property_claim"], errors="coerce")
            + pd.to_numeric(df["vehicle_claim"], errors="coerce")
        )
        mismatch_ratio = float((claim_total != pd.to_numeric(df["total_claim_amount"], errors="coerce")).mean())
        tolerance = float(require_nested(config, "validation.claim_total_mismatch_ratio"))
        if mismatch_ratio > tolerance:
            messages.append(f"{file_name}: claim total mismatch ratio {mismatch_ratio:.4f} exceeds {tolerance:.4f}.")

    if {"policy_bind_date", "incident_date"}.issubset(df.columns):
        bind_date = pd.to_datetime(df["policy_bind_date"], errors="coerce")
        incident_date = pd.to_datetime(df["incident_date"], errors="coerce")
        invalid_temporal_ratio = float((incident_date < bind_date).fillna(False).mean())
        max_temporal_ratio = float(require_nested(config, "validation.max_temporal_violation_ratio"))
        if invalid_temporal_ratio > max_temporal_ratio:
            messages.append(f"{file_name}: temporal violation ratio {invalid_temporal_ratio:.4f} exceeds {max_temporal_ratio:.4f}.")

    return messages


def validate_dataframe(df: pd.DataFrame, schema: dict, file_name: str, config: dict) -> Tuple[bool, List[str]]:
    """Validate dataframe against expected constraints.

    Returns validation feedback or raises an error when required conditions are not met.
    """
    strict_column_match = bool(require_nested(config, "validation.strict_column_match"))
    fail_on_extra = bool(require_nested(config, "validation.fail_on_extra_columns"))

    messages: List[str] = []
    messages.extend(validate_schema_structure(df, schema, file_name, strict_column_match, fail_on_extra))
    messages.extend(validate_data_quality(df, file_name, config))

    is_valid = len(messages) == 0
    if is_valid:
        messages.append(f"{file_name}: validation passed.")
        logging.info("Validation passed for %s", file_name)
    else:
        logging.warning("Validation failed for %s with %s issues", file_name, len(messages))

    return is_valid, messages


def write_validation_report(report_path: str, status: bool, message: str, details: Dict[str, List[str]], env: str) -> None:
    """Persist validation report to the target destination.

    Creates or overwrites the target artifact needed by later pipeline stages.
    """
    try:
        report_payload = {
            "environment": env,
            "validation_status": bool(status),
            "message": message,
            "details": details,
        }
        atomic_write_yaml(report_path, report_payload)
        logging.info("Validation report written to %s", report_path)
    except Exception as exc:
        raise DataValidationError(f"Failed to write validation report to {report_path}") from exc


def main() -> None:
    """Run this module as a script entrypoint.

    Coordinates the stage workflow, logs failures, and re-raises exceptions to make pipeline execution status explicit.
    """
    try:
        settings = load_settings()
        config = settings.config
        env = settings.env

        raw_dir = safe_join(settings.root_dir, str(require_nested(config, "data.raw_dir")))
        report_rel = os.path.join(str(require_nested(config, "data.reports_dir")), "report.yaml")
        report_path = safe_join(settings.root_dir, report_rel)
        required_files = require_nested(config, "validation.required_files")

        logging.info("Starting validation for env=%s", env)
        schema = load_schema(settings.schema_path)

        overall_status = True
        details: Dict[str, List[str]] = {}

        for file_name in required_files:
            file_path = os.path.join(raw_dir, file_name)
            if not os.path.exists(file_path):
                overall_status = False
                details[file_name] = [f"{file_name}: file not found at {file_path}."]
                logging.error("%s missing at %s", file_name, file_path)
                continue

            df = pd.read_csv(file_path)
            if df.empty:
                overall_status = False
                details[file_name] = [f"{file_name}: file is empty."]
                logging.error("%s is empty", file_name)
                continue

            file_status, file_messages = validate_dataframe(df, schema=schema, file_name=file_name, config=config)
            details[file_name] = file_messages
            overall_status = overall_status and file_status

        message = "Validation succeeded" if overall_status else "Validation failed"
        write_validation_report(report_path, overall_status, message, details, env)
        logging.info("Validation completed with status=%s for env=%s", overall_status, env)

        if not overall_status:
            raise DataValidationError("Validation failed. Check reports/report.yaml for details.")
    except Exception:
        logging.exception("Validation stage failed")
        raise


if __name__ == "__main__":
    main()

