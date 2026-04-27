from __future__ import annotations

# Re-export configuration and constants
from traffic_system.config import (
    settings,
    AREA_PROFILES,
    AREA_PROFILE_MAP,
    RUSH_HOURS,
    HOLIDAY_MONTH_DAY,
    WEATHER_SPEED_PENALTY,
    WEATHER_VOLUME_BOOST,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_COLUMNS,
)

# Re-export data processing and ingestion pipelines
from traffic_system.data import (
    generate_synthetic_traffic_data,
    load_raw_data,
    clean_traffic_data,
    engineer_time_features,
    preprocess_and_save,
    get_processed_data,
    generate_visualizations,
    get_data_payload,
    get_area_summary,
)

# Re-export ML modeling components
from traffic_system.ml import (
    build_preprocessor,
    train_and_evaluate,
    forecast_congestion_trend,
    get_forecast,
    load_metrics,
)

# Re-export inference and downstream analytics functions
from traffic_system.inference import (
    get_realtime_simulation,
    predict_traffic,
    analyze_route,
)

def bootstrap_project() -> dict[str, str]:
    raw_df = generate_synthetic_traffic_data(output_path=settings.raw_data_path)
    processed_df = preprocess_and_save(settings.raw_data_path, settings.processed_data_path)
    generate_visualizations(processed_df, settings.plot_dir)
    train_and_evaluate(save_artifacts=True)
    forecast_congestion_trend(output_path=settings.forecast_path)

    spark_status = "skipped"
    try:
        from spark.traffic_spark_job import run_spark_pipeline
        run_spark_pipeline(input_path=settings.raw_data_path, output_dir=settings.spark_output_dir)
        spark_status = "completed"
    except Exception as exc:  # pragma: no cover
        spark_status = f"skipped ({exc})"

    return {
        "raw_rows": f"{len(raw_df):,}",
        "processed_rows": f"{len(processed_df):,}",
        "model_bundle": str(settings.model_bundle_path),
        "forecast_file": str(settings.forecast_path),
        "spark_pipeline": spark_status,
    }
