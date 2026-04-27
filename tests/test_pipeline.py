from pathlib import Path
from uuid import uuid4

from traffic_pipeline import analyze_route, clean_traffic_data, engineer_time_features, generate_synthetic_traffic_data, settings


def _workspace_temp_file(name: str) -> Path:
    temp_dir = settings.base_dir / "artifacts" / "test_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir / f"{name}_{uuid4().hex}.csv"


def test_generated_dataset_contains_required_columns():
    output_path = _workspace_temp_file("traffic")
    df = generate_synthetic_traffic_data(output_path=output_path, num_days=3, seed=7)

    required_columns = {
        "timestamp",
        "area_name",
        "temperature_c",
        "humidity_pct",
        "traffic_volume",
        "avg_speed_kmph",
        "travel_time_minutes",
        "congestion_level",
    }

    assert required_columns.issubset(df.columns)


def test_preprocessing_engineers_operational_features():
    output_path = _workspace_temp_file("traffic")
    raw_df = generate_synthetic_traffic_data(output_path=output_path, num_days=2, seed=11)
    processed_df = engineer_time_features(clean_traffic_data(raw_df))

    for column in ["hour", "day_of_week", "is_weekend", "is_rush_hour", "congestion_index"]:
        assert column in processed_df.columns


def test_route_analysis_returns_start_end_metrics():
    route_result = analyze_route(
        {
            "timestamp": "2024-04-15T08:30:00",
            "start_area": "Baner",
            "end_area": "Hadapsar",
            "temperature_c": 28.0,
            "humidity_pct": 65.0,
            "precipitation_mm": 1.5,
            "weather_condition": "Cloudy",
            "road_incident_count": 1,
        }
    )

    assert route_result["route"]["estimated_distance_km"] > 0
    assert route_result["analysis"]["route_risk_score"] >= 0
    assert route_result["prediction"]["predicted_travel_time_minutes"] > 0
    assert len(route_result["trend"]) == 24
