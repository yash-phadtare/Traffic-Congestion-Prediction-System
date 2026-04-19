from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from traffic_pipeline import (
    get_data_payload,
    get_forecast,
    get_realtime_simulation,
    predict_traffic,
    train_and_evaluate,
)


class PredictionRequest(BaseModel):
    timestamp: datetime
    area_name: str = Field(..., description="Named traffic area")
    temperature_c: float
    humidity_pct: float
    precipitation_mm: float = 0.0
    weather_condition: str = "Clear"
    traffic_volume: int
    avg_speed_kmph: float
    road_incident_count: int = 0
    latitude: float | None = None
    longitude: float | None = None
    route_distance_km: float | None = None


class PredictionResponse(BaseModel):
    predicted_travel_time_minutes: float
    predicted_congestion_level: str
    model_version: str
    input_features: dict[str, Any]


class TrainingResponse(BaseModel):
    status: str
    metrics: dict[str, Any]


class DataResponse(BaseModel):
    rows: int
    columns: list[str]
    summary: dict[str, Any]
    preview: list[dict[str, Any]]


class ForecastResponse(BaseModel):
    forecast: list[dict[str, Any]]


app = FastAPI(
    title="Traffic Congestion Prediction System",
    description="Traffic analytics system with forecasting, dashboarding, and model retraining.",
    version="1.0.0",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    return PredictionResponse(**predict_traffic(payload.model_dump()))


@app.post("/train", response_model=TrainingResponse)
def train() -> TrainingResponse:
    metrics = train_and_evaluate(save_artifacts=True)
    return TrainingResponse(status="training_completed", metrics=metrics)


@app.get("/data", response_model=DataResponse)
def data() -> DataResponse:
    return DataResponse(**get_data_payload())


@app.get("/forecast", response_model=ForecastResponse)
def forecast() -> ForecastResponse:
    forecast_df = get_forecast()
    forecast_df["timestamp"] = forecast_df["timestamp"].astype(str)
    return ForecastResponse(forecast=forecast_df.to_dict(orient="records"))


@app.get("/simulate")
def simulate() -> list[dict[str, Any]]:
    simulation_df = get_realtime_simulation()
    simulation_df["timestamp"] = simulation_df["timestamp"].astype(str)
    return simulation_df.to_dict(orient="records")
