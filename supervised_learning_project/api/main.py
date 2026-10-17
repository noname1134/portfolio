from fastapi import FastAPI
from typing import Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator
from contextlib import asynccontextmanager
import pandas as pd
from . import model_service as ms
import utils


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


class ChurnPredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float = Field(ge=0, le=1)


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