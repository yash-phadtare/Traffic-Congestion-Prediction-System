from traffic_pipeline import analyze_route, clean_traffic_data, engineer_time_features, generate_synthetic_traffic_data
from traffic_system.config import CONGESTION_RATIO_LOW_MAX, CONGESTION_RATIO_MED_MAX
from traffic_system.inference import _classify_ratio


def test_generated_dataset_contains_required_columns(tmp_path):
    output_path = tmp_path / "traffic.csv"
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


def test_preprocessing_engineers_operational_features(tmp_path):
    output_path = tmp_path / "traffic.csv"
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
    assert 0 <= route_result["analysis"]["route_risk_score"] <= 100
    assert route_result["prediction"]["predicted_travel_time_minutes"] > 0
    assert len(route_result["trend"]) == 24


def test_route_risk_score_discriminates():
    low = analyze_route(
        {
            "timestamp": "2024-04-15T03:00:00",
            "start_area": "Baner",
            "end_area": "Hadapsar",
            "weather_condition": "Clear",
            "road_incident_count": 0,
        }
    )
    high = analyze_route(
        {
            "timestamp": "2024-04-15T08:30:00",
            "start_area": "Hinjewadi",
            "end_area": "Kharadi",
            "weather_condition": "Storm",
            "precipitation_mm": 20.0,
            "road_incident_count": 5,
        }
    )
    assert low["analysis"]["route_risk_score"] < high["analysis"]["route_risk_score"]
    assert high["analysis"]["route_risk_score"] <= 100


def test_unknown_area_raises_helpful_error():
    import pytest

    with pytest.raises(ValueError, match="Unknown area"):
        analyze_route(
            {
                "timestamp": "2024-04-15T08:30:00",
                "start_area": "Nowhere",
                "end_area": "Hadapsar",
            }
        )


def test_congestion_ratio_boundaries_match_training_labels():
    assert _classify_ratio(CONGESTION_RATIO_LOW_MAX) == "Low"
    assert _classify_ratio(CONGESTION_RATIO_LOW_MAX + 0.01) == "Medium"
    assert _classify_ratio(CONGESTION_RATIO_MED_MAX) == "Medium"
    assert _classify_ratio(CONGESTION_RATIO_MED_MAX + 0.01) == "High"
