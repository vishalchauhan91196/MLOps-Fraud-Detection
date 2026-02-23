import os

import numpy as np
import pytest
from fastapi.testclient import TestClient


class DummyModel:
    """Encapsulate `DummyModel` behavior used by this module.

    Groups related state and methods into a reusable, testable abstraction.
    """
    def predict(self, frame):
        """Execute predict as part of the module workflow.

        Encapsulates a focused unit of pipeline logic for reuse and testing.
        """
        return np.ones(len(frame), dtype=int)

    def predict_proba(self, frame):
        """Execute predict proba as part of the module workflow.

        Encapsulates a focused unit of pipeline logic for reuse and testing.
        """
        probs = np.full(len(frame), 0.8, dtype=float)
        return np.vstack([1.0 - probs, probs]).T


@pytest.fixture(scope="session", autouse=True)
def _skip_model_load():
    """Provide internal support for skip model load.

    Used by this module to keep the main workflow functions focused and readable.
    """
    os.environ["APP_SKIP_MODEL_LOAD"] = "true"


@pytest.fixture()
def fastapi_client():
    """Execute fastapi client as part of the module workflow.

    Encapsulates a focused unit of pipeline logic for reuse and testing.
    """
    from app.app import AppState, app

    AppState.model = DummyModel()
    AppState.expected_columns = ["age", "combined_text"]
    with TestClient(app) as client:
        yield client
