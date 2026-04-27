from __future__ import annotations

from fastapi import FastAPI

from traffic_pipeline import (
    analyze_route,
    get_data_payload,
    get_forecast,
    get_realtime_simulation,
    predict_traffic,
    train_and_evaluate,
)
from traffic_system.schemas import (
    DataResponse,
    ForecastResponse,
    PredictionRequest,
    PredictionResponse,
    RouteAnalysisRequest,
    RouteAnalysisResponse,
    TrainingResponse,
)


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


@app.post("/route-analyze", response_model=RouteAnalysisResponse)
def route_analyze(payload: RouteAnalysisRequest) -> RouteAnalysisResponse:
    return RouteAnalysisResponse(**analyze_route(payload.model_dump()))


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
def simulate() -> list[dict]:
    simulation_df = get_realtime_simulation()
    simulation_df["timestamp"] = simulation_df["timestamp"].astype(str)
    return simulation_df.to_dict(orient="records")
