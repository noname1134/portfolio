"""Shared fixtures: run the API against a mock model instead of the real one."""
from unittest.mock import Mock
import zlib
import numpy as np
import pytest
from fastapi.testclient import TestClient

import utils
from api import main

FAKE_CHURN_PROBABILITY = 0.9

def probability_for(customer_id: str) -> float:
    """Deterministic fake churn probability in [0.05, 0.95], different per customer ID."""
    bucket = zlib.crc32(customer_id.encode()) % 1000
    return 0.05 + 0.90 * bucket / 999


@pytest.fixture
def expected_probability():
    """The same function the mock uses, so tests can compute what to expect."""
    return probability_for


@pytest.fixture
def mock_model():
    model = Mock()
    model.classes_ = np.array([0, 1])
    def fake_predict_proba(df):
        churn = df[utils.ID_COLUMN].map(probability_for).to_numpy(dtype=float)
        return np.column_stack([1 - churn, churn])

    model.predict_proba.side_effect = fake_predict_proba
    return model


@pytest.fixture
def client(mock_model, monkeypatch):
    """TestClient whose lifespan loads the mock model instead of the joblib file."""
    monkeypatch.setattr(main.ms, "load_model", lambda model_path: mock_model)
    with TestClient(main.app) as test_client:
        yield test_client