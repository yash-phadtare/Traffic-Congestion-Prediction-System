from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


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


class RouteAnalysisRequest(BaseModel):
    timestamp: datetime
    start_area: str
    end_area: str
    temperature_c: float = 28.0
    humidity_pct: float = 65.0
    precipitation_mm: float = 0.0
    weather_condition: str = "Clear"
    road_incident_count: int = 0


class RouteAnalysisResponse(BaseModel):
    route: dict[str, Any]
    prediction: dict[str, Any]
    analysis: dict[str, Any]
    trend: list[dict[str, Any]]


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
