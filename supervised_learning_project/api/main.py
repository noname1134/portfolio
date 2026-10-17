from fastapi import FastAPI
from typing import Literal
from pydantic import BaseModel, Field, field_validator
from contextlib import asynccontextmanager
import pandas as pd
from . import model_service as ms
import utils
from collections import Counter


def _get_customer_id(customer):
    for name in ("customer_id", "customerID"):
        if isinstance(customer, dict):
            if name in customer:
                return customer[name]
        elif hasattr(customer, name):
            return getattr(customer, name)
    raise AttributeError(
        "Customer has neither 'customer_id' nor 'customerID'"
    )


def validate_unique_customer_ids(customers: list) -> list:
    ids = [_get_customer_id(c) for c in customers]
    counts = Counter(ids)
    duplicates = [customer_id for customer_id, count in counts.items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate customer IDs: {duplicates}")
    return customers


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = ms.load_model(utils.MODEL_PATH)
    try:
        yield
    finally:
        app.state.model = None

app = FastAPI(title="Customer Churn Prediction API", version="1.0.0", lifespan=lifespan)

class Customer(BaseModel):
    customer_id: str = Field(..., pattern=r"\S")
    tenure: int = Field(..., ge=0)
    monthly_charges: float = Field(..., ge=0)
    contract: Literal["Month-to-month", "One year", "Two year"]


class ChurnPredictionRequest(BaseModel):
    customerID: str = Field(..., pattern=r"\S")
    tenure: int = Field(..., ge=0)
    MonthlyCharges: float = Field(..., ge=0)
    Contract: Literal["Month-to-month", "One year", "Two year"]
    TotalCharges: float | None = Field(default=None, ge=0)
    SeniorCitizen: Literal[0, 1]
    gender: Literal["Female", "Male"]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    PhoneService: Literal["Yes", "No"]
    PaperlessBilling: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    PaymentMethod: Literal["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]

    @field_validator("TotalCharges", mode="before")
    @classmethod
    def clean_total_charges(cls, value):
        if isinstance(value, str) and not value.strip():
            return None
        return value

class BatchPredictionRequest(BaseModel):
    customers: list[ChurnPredictionRequest] = Field(min_length=1, max_length=1000)

    @field_validator("customers", mode="after")
    @classmethod
    def check_duplicates(cls, customers):
        return validate_unique_customer_ids(customers)

class ChurnPredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float = Field(ge=0, le=1)

class BatchPredictionResponse(BaseModel):
    predictions: list[ChurnPredictionResponse] = Field(min_length=1, max_length=1000)

    @field_validator("predictions", mode="after")
    @classmethod
    def check_duplicates(cls, predictions):
        return validate_unique_customer_ids(predictions)


@app.post("/customers/validate")
def validate_customer(customer: Customer) -> dict:
    return {
        "valid": True,
        "customer_id": customer.customer_id
    }

@app.get("/health")
def health_check() -> dict:
    return {"status": "healthy"}

@app.get("/customers/{customer_id}")
def get_customer(customer_id: str) -> dict:
    return {
        "customer_id": customer_id,
        "message": "Customer information requested"
    }


@app.post("/predict", response_model=ChurnPredictionResponse)
def predict_churn(customer: ChurnPredictionRequest) -> dict:
    customer_df = pd.DataFrame([customer.model_dump()])
    model = app.state.model
    probability = ms.predict_churn(model, customer_df)

    return {
        "customer_id": customer.customerID,
        "churn_probability": probability
    }

@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_churn_batch(customers: BatchPredictionRequest) -> dict:
    customer_df = pd.DataFrame([customer.model_dump() for customer in customers.customers])
    model = app.state.model
    probabilities_df = ms.predict_churn_batch(model, customer_df)
    probabilities_df.rename(columns={"customerID": "customer_id"}, inplace=True)
    predictions = probabilities_df.to_dict(orient="records")

    return {"predictions": predictions}