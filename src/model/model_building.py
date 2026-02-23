import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler

from src.core.exceptions import ModelBuildingError
from src.core.io import atomic_write_json, safe_join
from src.core.settings import load_settings, require_nested
from src.core.transformers import SparseToDenseTransformer
from src.core.logger import logging


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from the configured source.

    Returns a parsed in-memory object for downstream processing.
    """
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            raise ModelBuildingError(f"Training data is empty: {file_path}")
        logging.info("Loaded training data from %s shape=%s", file_path, df.shape)
        return df
    except ModelBuildingError:
        raise
    except Exception as exc:
        raise ModelBuildingError(f"Failed to load training data: {file_path}") from exc


def build_preprocessor(X: pd.DataFrame, pipeline_cfg: dict) -> Tuple[ColumnTransformer, bool]:
    """Construct preprocessor from the provided configuration and inputs.

    Returns an initialized object ready for model training or inference workflows.
    """
    text_col = str(require_nested(pipeline_cfg, "text_feature_column"))
    tfidf_cfg = require_nested(pipeline_cfg, "tfidf")

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols and col != text_col]

    transformers = []
    if numeric_cols:
        transformers.append(("num", "passthrough", numeric_cols))
    if categorical_cols:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown=str(require_nested(pipeline_cfg, "one_hot_handle_unknown"))),
                categorical_cols,
            )
        )
    if text_col in X.columns:
        transformers.append(
            (
                "txt",
                TfidfVectorizer(
                    max_features=int(require_nested(tfidf_cfg, "max_features")),
                    ngram_range=tuple(require_nested(tfidf_cfg, "ngram_range")),
                    min_df=require_nested(tfidf_cfg, "min_df"),
                    max_df=require_nested(tfidf_cfg, "max_df"),
                ),
                text_col,
            )
        )

    if not transformers:
        raise ModelBuildingError("No valid feature columns found for preprocessing pipeline")

    return ColumnTransformer(transformers=transformers, remainder="drop"), bool(numeric_cols)


def build_scaler_grid(pipeline_cfg: dict):
    """Construct scaler grid from the provided configuration and inputs.

    Returns an initialized object ready for model training or inference workflows.
    """
    scaler_options = [str(item).lower() for item in require_nested(pipeline_cfg, "scaler_options")]
    dense_minmax = Pipeline(steps=[("to_dense", SparseToDenseTransformer()), ("scale", MinMaxScaler())])
    dense_robust = Pipeline(steps=[("to_dense", SparseToDenseTransformer()), ("scale", RobustScaler())])
    mapping = {
        "standard": StandardScaler(with_mean=False),
        "minmax": dense_minmax,
        "robust": dense_robust,
        "none": "passthrough",
    }
    scalers = []
    for name in scaler_options:
        if name not in mapping:
            raise ModelBuildingError(f"Unsupported scaler option: {name}")
        scalers.append(mapping[name])
    return scalers


def create_estimator(algo_cfg: dict, random_state: int):
    """Construct estimator from the provided configuration and inputs.

    Returns an initialized object ready for model training or inference workflows.
    """
    name = str(require_nested(algo_cfg, "name"))
    params = dict(require_nested(algo_cfg, "params"))
    if "random_state" not in params:
        params["random_state"] = random_state

    if name == "logistic_regression":
        return LogisticRegression(**params)
    if name == "random_forest":
        return RandomForestClassifier(**params)
    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except Exception as exc:
            raise ModelBuildingError("xgboost is not installed but xgboost algorithm is enabled") from exc
        if "eval_metric" not in params:
            params["eval_metric"] = "logloss"
        return XGBClassifier(**params)
    raise ModelBuildingError(f"Unsupported algorithm name: {name}")


def create_param_grid(algo_cfg: dict, use_scaler: bool, has_numeric: bool, pipeline_cfg: dict) -> Dict[str, List]:
    """Construct param grid from the provided configuration and inputs.

    Returns an initialized object ready for model training or inference workflows.
    """
    grid_cfg = require_nested(algo_cfg, "grid_search")
    param_grid: Dict[str, List] = {}
    for key, value in grid_cfg.items():
        param_grid[f"clf__{key}"] = list(value)

    if use_scaler and has_numeric:
        param_grid["scaler"] = build_scaler_grid(pipeline_cfg)
    else:
        param_grid["scaler"] = ["passthrough"]
    return param_grid


def train_candidate(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    preprocessor: ColumnTransformer,
    has_numeric: bool,
    pipeline_cfg: dict,
    gs_cfg: dict,
    algo_cfg: dict,
    random_state: int,
):
    """Train candidate using the supplied data and model configuration.

    Returns fitted artifacts and metrics used for model selection.
    """
    try:
        algo_name = str(require_nested(algo_cfg, "name"))
        estimator = create_estimator(algo_cfg, random_state=random_state)
        use_scaler = bool(require_nested(algo_cfg, "use_scaler"))

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("scaler", "passthrough"),
                ("clf", estimator),
            ]
        )
        param_grid = create_param_grid(
            algo_cfg=algo_cfg,
            use_scaler=use_scaler,
            has_numeric=has_numeric,
            pipeline_cfg=pipeline_cfg,
        )

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            n_iter=int(require_nested(gs_cfg, "n_iter")),
            cv=int(require_nested(gs_cfg, "cv")),
            scoring=str(require_nested(gs_cfg, "scoring")),
            n_jobs=int(require_nested(gs_cfg, "n_jobs")),
            refit=True,
            random_state=random_state,
        )
        search.fit(X_train, y_train)

        best_params = {}
        for key, value in search.best_params_.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                best_params[key] = value
            else:
                best_params[key] = str(value)

        summary = {
            "algorithm": algo_name,
            "best_cv_score": float(search.best_score_),
            "best_params": best_params,
        }
        logging.info("Candidate completed algo=%s score=%s", algo_name, summary["best_cv_score"])
        return search.best_estimator_, summary
    except ModelBuildingError:
        raise
    except Exception as exc:
        raise ModelBuildingError("Candidate model tuning failed") from exc


def train_and_select_best(X_train: pd.DataFrame, y_train: np.ndarray, model_cfg: dict, random_state: int):
    """Train and select best using the supplied data and model configuration.

    Returns fitted artifacts and metrics used for model selection.
    """
    pipeline_cfg = require_nested(model_cfg, "pipeline")
    gs_cfg = require_nested(model_cfg, "grid_search")
    algorithms_cfg = list(require_nested(model_cfg, "algorithms"))

    enabled_algorithms = [cfg for cfg in algorithms_cfg if bool(require_nested(cfg, "enabled"))]
    if not enabled_algorithms:
        raise ModelBuildingError("No enabled algorithms found under model_building.algorithms")

    preprocessor, has_numeric = build_preprocessor(X_train, pipeline_cfg)
    best_model = None
    best_summary = None
    candidate_summaries = []

    for algo_cfg in enabled_algorithms:
        candidate_model, candidate_summary = train_candidate(
            X_train=X_train,
            y_train=y_train,
            preprocessor=preprocessor,
            has_numeric=has_numeric,
            pipeline_cfg=pipeline_cfg,
            gs_cfg=gs_cfg,
            algo_cfg=algo_cfg,
            random_state=random_state,
        )
        candidate_summaries.append(candidate_summary)

        if best_summary is None or candidate_summary["best_cv_score"] > best_summary["best_cv_score"]:
            best_summary = candidate_summary
            best_model = candidate_model

    if best_model is None or best_summary is None:
        raise ModelBuildingError("Model selection failed to produce a best model")

    selection_summary = {
        "selection_metric": str(require_nested(gs_cfg, "scoring")),
        "selected_algorithm": best_summary["algorithm"],
        "selected_cv_score": best_summary["best_cv_score"],
        "candidates": candidate_summaries,
    }
    logging.info("Selected model algorithm=%s score=%s", best_summary["algorithm"], best_summary["best_cv_score"])
    return best_model, selection_summary


def save_model(model, file_path: str) -> None:
    """Persist model to the target destination.

    Creates or overwrites the target artifact needed by later pipeline stages.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file:
            pickle.dump(model, file)
        logging.info("Pipeline model saved to %s", file_path)
    except Exception as exc:
        raise ModelBuildingError(f"Failed to save model: {file_path}") from exc


def main() -> None:
    """Run this module as a script entrypoint.

    Coordinates the stage workflow, logs failures, and re-raises exceptions to make pipeline execution status explicit.
    """
    try:
        settings = load_settings()
        config = settings.config

        model_cfg = require_nested(config, "model_building")
        pipeline_cfg = require_nested(config, "pipeline")
        target_col = str(require_nested(pipeline_cfg, "target_column"))
        random_state = int(require_nested(pipeline_cfg, "random_state"))

        processed_dir = safe_join(settings.root_dir, str(require_nested(config, "data.processed_dir")))
        models_dir = safe_join(settings.root_dir, str(require_nested(config, "data.models_dir")))

        train_file = safe_join(processed_dir, str(require_nested(model_cfg, "train_input_file")))
        model_file = safe_join(models_dir, str(require_nested(model_cfg, "output_model_file")))
        summary_file = safe_join(models_dir, str(require_nested(model_cfg, "training_summary_output_file")))

        train_data = load_data(train_file)
        if target_col not in train_data.columns:
            raise ModelBuildingError(f"Target column missing in training data: {target_col}")

        X_train = train_data.drop(columns=[target_col])
        y_train = train_data[target_col].values

        model, training_summary = train_and_select_best(
            X_train=X_train,
            y_train=y_train,
            model_cfg=model_cfg,
            random_state=random_state,
        )
        save_model(model, model_file)
        atomic_write_json(summary_file, training_summary)
        logging.info("Model training summary saved to %s", summary_file)
    except Exception:
        logging.exception("Model building failed")
        raise


if __name__ == "__main__":
    main()

