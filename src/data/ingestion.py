import os
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.core.exceptions import ConfigurationError, DataIngestionError
from src.core.io import atomic_write_csv, safe_join
from src.core.settings import load_settings, require_nested
from src.core.logger import logging


def load_data(data_path: str) -> pd.DataFrame:
    """Load data from the configured source.

    Returns a parsed in-memory object for downstream processing.
    """
    try:
        df = pd.read_csv(data_path)
        if df.empty:
            raise DataIngestionError(f"Input dataset is empty: {data_path}")
        logging.info("Data loaded from %s with shape %s", data_path, df.shape)
        return df
    except DataIngestionError:
        raise
    except Exception as exc:
        raise DataIngestionError(f"Failed to load dataset from {data_path}") from exc


def split_data(df: pd.DataFrame, test_size: float, random_state: int, target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into deterministic partitions for downstream stages.

    Used to create consistent training and evaluation datasets.
    """
    try:
        stratify = df[target_col] if target_col in df.columns else None
        if stratify is None:
            logging.warning("Target column %s not found. Using non-stratified split.", target_col)

        train_data, test_data = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
        logging.info("Completed split with test_size=%s random_state=%s", test_size, random_state)
        return train_data, test_data
    except Exception as exc:
        raise DataIngestionError("Failed during train/test split") from exc


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, output_dir: str) -> None:
    """Persist data to the target destination.

    Creates or overwrites the target artifact needed by later pipeline stages.
    """
    try:
        train_path = os.path.join(output_dir, "train.csv")
        test_path = os.path.join(output_dir, "test.csv")

        atomic_write_csv(train_data, train_path)
        atomic_write_csv(test_data, test_path)

        logging.info("Saved train data to %s with shape %s", train_path, train_data.shape)
        logging.info("Saved test data to %s with shape %s", test_path, test_data.shape)
    except Exception as exc:
        raise DataIngestionError("Failed to persist train/test datasets") from exc


def main() -> None:
    """Run this module as a script entrypoint.

    Coordinates the stage workflow, logs failures, and re-raises exceptions to make pipeline execution status explicit.
    """
    try:
        settings = load_settings()
        config = settings.config
        env = settings.env

        source_path = safe_join(settings.root_dir, str(require_nested(config, "data.source_file")))
        output_dir = safe_join(settings.root_dir, str(require_nested(config, "data.raw_dir")))
        test_size = float(require_nested(config, "ingestion.test_size"))
        random_state = int(require_nested(config, "pipeline.random_state"))
        target_col = str(require_nested(config, "pipeline.target_column"))

        if not 0.0 < test_size < 1.0:
            raise ConfigurationError(f"Invalid ingestion.test_size={test_size}. Must be between 0 and 1.")

        logging.info("Starting ingestion for env=%s", env)
        df = load_data(source_path)
        train_data, test_data = split_data(df, test_size=test_size, random_state=random_state, target_col=target_col)
        save_data(train_data, test_data, output_dir)
        logging.info("Ingestion completed for env=%s", env)
    except Exception:
        logging.exception("Ingestion failed")
        raise


if __name__ == "__main__":
    main()

