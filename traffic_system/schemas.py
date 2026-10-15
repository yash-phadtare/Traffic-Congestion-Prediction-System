from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from traffic_system.config import AREA_PROFILE_MAP

WeatherCondition = Literal["Clear", "Cloudy", "Rain", "Storm"]


def _validate_area(value: str) -> str:
    if value not in AREA_PROFILE_MAP:
        raise ValueError(f"Unknown area '{value}'. Valid areas: {sorted(AREA_PROFILE_MAP)}")
    return value


class PredictionRequest(BaseModel):
    timestamp: datetime
    area_name: str = Field(..., description="Named traffic area")
    temperature_c: float = Field(..., ge=-10, le=55)
    humidity_pct: float = Field(..., ge=0, le=100)
    precipitation_mm: float = Field(default=0.0, ge=0, le=200)
    weather_condition: WeatherCondition = "Clear"
    traffic_volume: int = Field(..., ge=0, le=10000)
    avg_speed_kmph: float = Field(..., gt=0, le=150)
    road_incident_count: int = Field(default=0, ge=0, le=20)
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    route_distance_km: float | None = Field(default=None, gt=0, le=200)

    @field_validator("area_name")
    @classmethod
    def validate_area(cls, value: str) -> str:
        return _validate_area(value)


class PredictionResponse(BaseModel):
    predicted_travel_time_minutes: float
    predicted_congestion_level: str
    model_version: str
    input_features: dict[str, Any]


class RouteAnalysisRequest(BaseModel):
    timestamp: datetime
    start_area: str
    end_area: str
    temperature_c: float = Field(default=28.0, ge=-10, le=55)
    humidity_pct: float = Field(default=65.0, ge=0, le=100)
    precipitation_mm: float = Field(default=0.0, ge=0, le=200)
    weather_condition: WeatherCondition = "Clear"
    road_incident_count: int = Field(default=0, ge=0, le=20)

    @field_validator("start_area", "end_area")
    @classmethod
    def validate_areas(cls, value: str) -> str:
        return _validate_area(value)


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
