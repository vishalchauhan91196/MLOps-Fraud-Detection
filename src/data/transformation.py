import re
from string import punctuation
from typing import Iterable, List, Optional, Set, Tuple

import pandas as pd

try:
    from imblearn.over_sampling import SMOTEN
except Exception:
    SMOTEN = None

from src.core.exceptions import DataTransformationError
from src.core.logger import logging


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
ALNUM_PATTERN = re.compile(r"[a-z0-9]+")

_NLTK_CACHE: Optional[Tuple[object, Set[str], object]] = None


def _require(config: dict, key: str):
    """Provide internal support for require.

    Used by this module to keep the main workflow functions focused and readable.
    """
    if key not in config:
        raise DataTransformationError(f"Missing transformation config key: {key}")
    return config[key]


def _get_nltk_resources(download_if_missing: bool = True) -> Tuple[Optional[object], Set[str], Optional[object]]:
    """Provide internal support for get nltk resources.

    Used by this module to keep the main workflow functions focused and readable.
    """
    global _NLTK_CACHE
    if _NLTK_CACHE is not None:
        return _NLTK_CACHE

    try:
        import nltk
        from nltk.corpus import stopwords
    except Exception as exc:
        _NLTK_CACHE = (None, set(), None)
        logging.warning("NLTK is unavailable. Stopword removal will be skipped: %s", exc)
        return _NLTK_CACHE

    tokenize = None
    lemmatizer = None

    if download_if_missing:
        for resource, package in [
            ("tokenizers/punkt", "punkt"),
            ("corpora/stopwords", "stopwords"),
            ("corpora/wordnet", "wordnet"),
        ]:
            try:
                nltk.data.find(resource)
            except LookupError:
                nltk.download(package, quiet=True)

    try:
        stop_words = set(stopwords.words("english"))
    except Exception as exc:
        stop_words = set()
        logging.warning("Could not load NLTK stopwords. Stopword removal will be skipped: %s", exc)

    try:
        from nltk.tokenize import word_tokenize

        tokenize = word_tokenize
    except Exception as exc:
        logging.warning("Could not load NLTK tokenizer. Falling back to regex tokenization: %s", exc)

    try:
        from nltk.stem import WordNetLemmatizer

        lemmatizer = WordNetLemmatizer()
    except Exception as exc:
        logging.warning("Could not load NLTK lemmatizer. Lemmatization will be skipped: %s", exc)

    _NLTK_CACHE = (tokenize, stop_words, lemmatizer)
    logging.info("NLTK resources loaded and cached")
    return _NLTK_CACHE


def normalize_text(
    text: str,
    lowercase: bool,
    remove_urls: bool,
    remove_non_alphanumeric: bool,
    remove_stopwords: bool,
    apply_lemmatization: bool,
    nltk_download_if_missing: bool,
) -> str:
    """Transform text into the format required by later steps.

    Applies deterministic preprocessing so training and inference stay consistent.
    """
    try:
        if pd.isna(text):
            return ""

        raw_text = str(text)
        if lowercase:
            raw_text = raw_text.lower()
        if remove_urls:
            raw_text = URL_PATTERN.sub(" ", raw_text)

        tokenize, stop_words, lemmatizer = _get_nltk_resources(download_if_missing=nltk_download_if_missing)
        if tokenize is None:
            tokens = ALNUM_PATTERN.findall(raw_text.lower())
        else:
            try:
                tokens = tokenize(raw_text)
            except Exception:
                tokens = ALNUM_PATTERN.findall(raw_text.lower())

        cleaned_tokens: List[str] = []
        for token in tokens:
            if token in punctuation:
                continue
            if remove_non_alphanumeric and not ALNUM_PATTERN.fullmatch(token):
                continue
            if remove_stopwords and token in stop_words:
                continue
            cleaned_tokens.append(token)

        if apply_lemmatization and lemmatizer is not None:
            try:
                cleaned_tokens = [lemmatizer.lemmatize(token) for token in cleaned_tokens]
            except Exception:
                pass

        return " ".join(cleaned_tokens).strip()
    except Exception as exc:
        raise DataTransformationError("Failed to normalize text value") from exc


def transform_text_columns(df: pd.DataFrame, text_columns: Iterable[str], config: dict) -> pd.DataFrame:
    """Transform text columns into the format required by later steps.

    Applies deterministic preprocessing so training and inference stay consistent.
    """
    try:
        transformed_df = df.copy()
        applied_columns: List[str] = []

        lowercase = bool(_require(config, "lowercase"))
        remove_urls = bool(_require(config, "remove_urls"))
        remove_non_alphanumeric = bool(_require(config, "remove_non_alphanumeric"))
        remove_stopwords = bool(_require(config, "remove_stopwords"))
        apply_lemmatization = bool(_require(config, "apply_lemmatization"))
        nltk_download_if_missing = bool(_require(config, "nltk_download_if_missing"))

        for column in text_columns:
            if column in transformed_df.columns:
                transformed_df[column] = transformed_df[column].astype(str).map(
                    lambda value: normalize_text(
                        value,
                        lowercase=lowercase,
                        remove_urls=remove_urls,
                        remove_non_alphanumeric=remove_non_alphanumeric,
                        remove_stopwords=remove_stopwords,
                        apply_lemmatization=apply_lemmatization,
                        nltk_download_if_missing=nltk_download_if_missing,
                    )
                )
                applied_columns.append(column)

        logging.info("Text transformations applied on columns=%s", applied_columns)
        return transformed_df
    except Exception as exc:
        raise DataTransformationError("Failed while transforming text columns") from exc


def enrich_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transform date features into the format required by later steps.

    Applies deterministic preprocessing so training and inference stay consistent.
    """
    try:
        transformed_df = df.copy()
        generated_features = []

        for date_col in ["policy_bind_date", "incident_date"]:
            if date_col in transformed_df.columns:
                parsed = pd.to_datetime(transformed_df[date_col], errors="coerce")
                year_col = f"{date_col}_year"
                month_col = f"{date_col}_month"
                day_col = f"{date_col}_day"
                transformed_df[year_col] = parsed.dt.year
                transformed_df[month_col] = parsed.dt.month
                transformed_df[day_col] = parsed.dt.day
                generated_features.extend([year_col, month_col, day_col])

        if {"policy_bind_date", "incident_date"}.issubset(transformed_df.columns):
            bind_date = pd.to_datetime(transformed_df["policy_bind_date"], errors="coerce")
            incident_date = pd.to_datetime(transformed_df["incident_date"], errors="coerce")
            transformed_df["policy_to_incident_days"] = (incident_date - bind_date).dt.days
            generated_features.append("policy_to_incident_days")

        date_columns = [col for col in ["policy_bind_date", "incident_date"] if col in transformed_df.columns]
        if date_columns:
            transformed_df = transformed_df.drop(columns=date_columns)

        logging.info("Date feature enrichment complete. Added=%s, dropped=%s", generated_features, date_columns)
        return transformed_df
    except Exception as exc:
        raise DataTransformationError("Failed while enriching date features") from exc


def apply_smoten(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    enabled: bool,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Transform smoten into the format required by later steps.

    Applies deterministic preprocessing so training and inference stay consistent.
    """
    try:
        if not enabled:
            logging.info("Skipping SMOTEN because it is disabled by config")
            return X_train, y_train

        if SMOTEN is None:
            logging.info("Skipping SMOTEN because imblearn is not installed")
            return X_train, y_train

        class_counts = y_train.value_counts()
        if len(class_counts) < 2:
            logging.info("Skipping SMOTEN because target has fewer than 2 classes")
            return X_train, y_train

        X_nominal = X_train.astype(str)
        smoten = SMOTEN(random_state=random_state)
        X_resampled, y_resampled = smoten.fit_resample(X_nominal, y_train)

        X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
        y_resampled = pd.Series(y_resampled, name=y_train.name)

        logging.info("Applied SMOTEN: train shape changed from %s to %s", X_train.shape, X_resampled.shape)
        return X_resampled, y_resampled
    except Exception as exc:
        logging.warning("SMOTEN could not be applied. Continuing without resampling: %s", exc)
        return X_train, y_train


def transform_dataframe(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Transform dataframe into the format required by later steps.

    Applies deterministic preprocessing so training and inference stay consistent.
    """
    try:
        logging.info("Starting dataframe transformations with shape=%s", df.shape)

        text_columns = _require(config, "text_columns")
        enrich_dates = bool(_require(config, "enrich_dates"))

        transformed_df = transform_text_columns(df, text_columns=text_columns, config=config)

        if enrich_dates:
            transformed_df = enrich_date_features(transformed_df)

        logging.info("Completed dataframe transformations with shape=%s", transformed_df.shape)
        return transformed_df
    except DataTransformationError:
        raise
    except Exception as exc:
        raise DataTransformationError("Failed to transform dataframe") from exc

