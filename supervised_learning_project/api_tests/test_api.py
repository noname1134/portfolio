import pytest
from conftest import FAKE_CHURN_PROBABILITY

@pytest.fixture
def get_valid_full_customer():
    return {
        "customerID": "7590-VHVEG",
        "tenure": 1,
        "MonthlyCharges": 29.85,
        "Contract": "Month-to-month",
        "TotalCharges": 29.85,
        "SeniorCitizen": 0,
        "gender": "Female",
        "Partner": "Yes",
        "Dependents": "No",
        "PhoneService": "No",
        "PaperlessBilling": "Yes",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "PaymentMethod": "Electronic check",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No"
    }


@pytest.fixture
def get_valid_customer():
    return {
        "customer_id": "7590-VHVEG",
        "tenure": 1,
        "monthly_charges": 29.85,
        "contract": "Month-to-month"
    }


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


@pytest.mark.parametrize(
    "payload_overrides, expected_status",
    [
        ({}, 200),
        ({"tenure": -10}, 422),
        ({"contract": "Five years"}, 422),
    ],
    ids=["valid_customer", "negative_tenure", "invalid_contract"]
)
def test_customer_validation(client, get_valid_customer, payload_overrides, expected_status):
    payload = {**get_valid_customer, **payload_overrides}

    response = client.post("/customers/validate", json=payload)
    assert response.status_code == expected_status

    if expected_status == 200:
        assert response.json() == {
            "valid": True,
            "customer_id": payload["customer_id"]
        }
    elif expected_status == 422:
        details = response.json()["detail"]
        assert isinstance(details, list)
        assert len(details) > 0


def test_single_prediction(client, mock_model, get_valid_full_customer):
    response = client.post("/predict", json=get_valid_full_customer)

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["customer_id"] == get_valid_full_customer["customerID"]
    assert data["churn_probability"] == pytest.approx(FAKE_CHURN_PROBABILITY)
    mock_model.predict_proba.assert_called_once()


def test_batch_predictions(client, mock_model, get_valid_full_customer):
    customer_one = get_valid_full_customer.copy()
    customer_two = {**get_valid_full_customer, "customerID": "1234-ABCD"}
    payload = {"customers": [customer_one, customer_two]}

    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "predictions" in data
    assert len(data["predictions"]) == 2

    for pred in data["predictions"]:
        assert "customer_id" in pred
        assert "churn_probability" in pred
        assert pred["churn_probability"] == pytest.approx(FAKE_CHURN_PROBABILITY)

    input_ids = [customer["customerID"] for customer in payload["customers"]]
    output_ids = [prediction["customer_id"] for prediction in data["predictions"]]
    assert output_ids == input_ids
    mock_model.predict_proba.assert_called_once()

def test_empty_batch(client):
    payload = {"customers": []}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 422


def test_duplicate_customer_ids(client, get_valid_full_customer):
    customer_one = get_valid_full_customer.copy()
    customer_two = get_valid_full_customer
    payload = {"customers": [customer_one, customer_two]}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 422


def test_invalid_customer_in_batch(client, get_valid_full_customer):
    customer_one = get_valid_full_customer.copy()
    customer_two = {**get_valid_full_customer, "customerID": f"1234-ABCDE", "tenure": -1}
    payload = {"customers": [customer_one, customer_two]}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 422


def test_batch_size_limit(client, get_valid_full_customer):
    customers = [{**get_valid_full_customer, "customerID": f"{i:04d}-ABCDE"} for i in range(1001)]
    payload = {"customers": customers}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 422