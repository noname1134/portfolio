"""Shared fixtures: run the API against a mock model instead of the real one."""
from unittest.mock import Mock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api import main

FAKE_CHURN_PROBABILITY = 0.9


@pytest.fixture
def mock_model():
    model = Mock()
    model.classes_ = np.array([0, 1])
    model.predict_proba.side_effect = lambda df: np.tile([1 - FAKE_CHURN_PROBABILITY, FAKE_CHURN_PROBABILITY], (len(df), 1))
    return model


@pytest.fixture
def client(mock_model, monkeypatch):
    """TestClient whose lifespan loads the mock model instead of the joblib file."""
    monkeypatch.setattr(main.ms, "load_model", lambda model_path: mock_model)
    with TestClient(main.app) as test_client:
        yield test_client