from typing import Any, Dict, Optional


class FraudDetectionError(Exception):
    """Base exception for the fraud detection platform."""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Provide internal support for init.

        Used by this module to keep the main workflow functions focused and readable.
        """
        super().__init__(message)
        self.context = context or {}


class ConfigurationError(FraudDetectionError):
    """Raised for configuration loading/validation failures."""


class SecurityError(FraudDetectionError):
    """Raised for path traversal or unsafe filesystem access."""


class PipelineIOError(FraudDetectionError):
    """Raised for pipeline read/write I/O failures."""


class DataIngestionError(FraudDetectionError):
    """Raised for ingestion-stage failures."""


class DataValidationError(FraudDetectionError):
    """Raised for validation-stage failures."""


class DataPreprocessingError(FraudDetectionError):
    """Raised for preprocessing-stage failures."""


class DataTransformationError(FraudDetectionError):
    """Raised for transformation-stage failures."""


class FeatureEngineeringError(FraudDetectionError):
    """Raised for feature engineering-stage failures."""


class ModelBuildingError(FraudDetectionError):
    """Raised for model building-stage failures."""


class ModelEvaluationError(FraudDetectionError):
    """Raised for model evaluation-stage failures."""


class ModelRegistrationError(FraudDetectionError):
    """Raised for model registration-stage failures."""
