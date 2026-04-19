from traffic_pipeline import clean_traffic_data, engineer_time_features, generate_synthetic_traffic_data


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
