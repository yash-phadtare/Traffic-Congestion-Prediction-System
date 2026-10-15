from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

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
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(ValueError)
async def value_error_handler(_, exc: ValueError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(FileNotFoundError)
async def missing_artifact_handler(_, exc: FileNotFoundError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    try:
        return PredictionResponse(**predict_traffic(payload.model_dump()))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/route-analyze", response_model=RouteAnalysisResponse)
def route_analyze(payload: RouteAnalysisRequest) -> RouteAnalysisResponse:
    try:
        return RouteAnalysisResponse(**analyze_route(payload.model_dump()))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/train", response_model=TrainingResponse)
def train() -> TrainingResponse:
    try:
        metrics = train_and_evaluate(save_artifacts=True)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Training failed: {exc}") from exc
    return TrainingResponse(status="training_completed", metrics=metrics)


@app.get("/data", response_model=DataResponse)
def data(limit: int = Query(default=20, ge=1, le=200)) -> DataResponse:
    return DataResponse(**get_data_payload(limit=limit))


@app.get("/forecast", response_model=ForecastResponse)
def forecast(force_refresh: bool = Query(default=False)) -> ForecastResponse:
    forecast_df = get_forecast(force_refresh=force_refresh)
    forecast_df["timestamp"] = forecast_df["timestamp"].astype(str)
    return ForecastResponse(forecast=forecast_df.to_dict(orient="records"))


@app.get("/simulate")
def simulate(
    sample_size: int = Query(default=25, ge=1, le=500),
    seed: int = Query(default=42, ge=0),
) -> list[dict]:
    try:
        simulation_df = get_realtime_simulation(sample_size=sample_size, seed=seed)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    simulation_df["timestamp"] = simulation_df["timestamp"].astype(str)
    return simulation_df.to_dict(orient="records")
