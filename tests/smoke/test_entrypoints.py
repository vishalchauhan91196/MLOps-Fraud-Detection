from pathlib import Path


def test_critical_entrypoints_exist():
    """Validate that critical entrypoints exist behaves as expected.

    This test guards against regressions in the associated workflow or API behavior.
    """
    required = [
        "src/data/ingestion.py",
        "src/data/validation.py",
        "src/data/preprocessing.py",
        "src/data/transformation.py",
        "src/feature/feature_engineering.py",
        "src/model/model_building.py",
        "src/model/model_evaluation.py",
        "src/drift_detection/run.py",
        "src/model/register_model.py",
        "src/model/model_promotion.py",
        "src/core/config_lint.py",
    ]
    for file_path in required:
        assert Path(file_path).exists(), f"Missing required entrypoint: {file_path}"


