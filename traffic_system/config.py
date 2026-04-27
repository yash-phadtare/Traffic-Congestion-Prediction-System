from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Settings:
    base_dir: Path = Path(__file__).resolve().parents[1]
    raw_data_path: Path = base_dir / "data" / "raw" / "traffic_data.csv"
    processed_data_path: Path = base_dir / "data" / "processed" / "traffic_features.csv"
    model_bundle_path: Path = base_dir / "artifacts" / "models" / "model_bundle.joblib"
    metrics_path: Path = base_dir / "artifacts" / "models" / "training_metrics.json"
    feature_importance_path: Path = base_dir / "artifacts" / "models" / "feature_importance.csv"
    forecast_path: Path = base_dir / "artifacts" / "forecasts" / "future_congestion_forecast.csv"
    plot_dir: Path = base_dir / "artifacts" / "plots"
    spark_output_dir: Path = base_dir / "artifacts" / "spark"
    random_state: int = 42

settings = Settings()

AREA_PROFILES = [
    {"area_name": "Shivaji Nagar", "latitude": 18.5308, "longitude": 73.8475, "route_distance_km": 6.4, "base_traffic_volume": 980, "base_speed_kmph": 34},
    {"area_name": "Hinjewadi", "latitude": 18.5910, "longitude": 73.7389, "route_distance_km": 14.0, "base_traffic_volume": 870, "base_speed_kmph": 38},
    {"area_name": "Kothrud", "latitude": 18.5074, "longitude": 73.8077, "route_distance_km": 8.2, "base_traffic_volume": 760, "base_speed_kmph": 37},
    {"area_name": "Kharadi", "latitude": 18.5513, "longitude": 73.9497, "route_distance_km": 12.1, "base_traffic_volume": 920, "base_speed_kmph": 36},
    {"area_name": "Baner", "latitude": 18.5590, "longitude": 73.7868, "route_distance_km": 10.2, "base_traffic_volume": 730, "base_speed_kmph": 35},
    {"area_name": "Hadapsar", "latitude": 18.5089, "longitude": 73.9259, "route_distance_km": 11.6, "base_traffic_volume": 860, "base_speed_kmph": 33},
    {"area_name": "Wakad", "latitude": 18.5975, "longitude": 73.7637, "route_distance_km": 13.4, "base_traffic_volume": 810, "base_speed_kmph": 34},
    {"area_name": "Viman Nagar", "latitude": 18.5679, "longitude": 73.9143, "route_distance_km": 9.3, "base_traffic_volume": 780, "base_speed_kmph": 36},
]
AREA_PROFILE_MAP = {item["area_name"]: item for item in AREA_PROFILES}

RUSH_HOURS = {7, 8, 9, 10, 17, 18, 19, 20}
HOLIDAY_MONTH_DAY = {(1, 1), (1, 26), (8, 15), (10, 2), (12, 25)}
WEATHER_SPEED_PENALTY = {"Clear": 0.00, "Cloudy": 0.04, "Rain": 0.11, "Storm": 0.18}
WEATHER_VOLUME_BOOST = {"Clear": 0.00, "Cloudy": 0.02, "Rain": 0.07, "Storm": 0.11}

FEATURE_COLUMNS = [
    "area_name", "latitude", "longitude", "temperature_c", "humidity_pct", "precipitation_mm",
    "weather_condition", "traffic_volume", "avg_speed_kmph", "road_incident_count",
    "route_distance_km", "hour", "day_of_week", "month", "is_weekend", "is_rush_hour",
    "is_holiday", "time_band", "congestion_index",
]
NUMERIC_FEATURES = [
    "latitude", "longitude", "temperature_c", "humidity_pct", "precipitation_mm",
    "traffic_volume", "avg_speed_kmph", "road_incident_count", "route_distance_km",
    "hour", "day_of_week", "month", "is_weekend", "is_rush_hour", "is_holiday", "congestion_index",
]
CATEGORICAL_FEATURES = ["area_name", "weather_condition", "time_band"]
NUMERIC_COLUMNS = [
    "latitude", "longitude", "temperature_c", "humidity_pct", "precipitation_mm",
    "traffic_volume", "avg_speed_kmph", "road_incident_count", "route_distance_km",
]

def ensure_directories(paths: list[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
