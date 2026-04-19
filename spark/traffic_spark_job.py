from __future__ import annotations

from pathlib import Path

from traffic_pipeline import settings


def run_spark_pipeline(
    input_path: Path | None = None,
    output_dir: Path | None = None,
) -> None:
    """Run scalable traffic aggregations using PySpark."""

    from pyspark.sql import SparkSession
    from pyspark.sql.functions import avg, col, dayofweek, hour, month, to_timestamp

    input_path = input_path or settings.raw_data_path
    output_dir = output_dir or settings.spark_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    spark = (
        SparkSession.builder.appName("TrafficCongestionPredictionSpark")
        .master("local[*]")
        .getOrCreate()
    )

    df = spark.read.csv(str(input_path), header=True, inferSchema=True)
    df = df.withColumn("timestamp", to_timestamp(col("timestamp")))
    df = df.dropDuplicates()
    df = df.na.fill(
        {
            "temperature_c": 28.0,
            "humidity_pct": 65.0,
            "avg_speed_kmph": 30.0,
        }
    )

    enriched_df = (
        df.withColumn("hour", hour("timestamp"))
        .withColumn("day_of_week", dayofweek("timestamp"))
        .withColumn("month", month("timestamp"))
    )

    area_hourly = (
        enriched_df.groupBy("area_name", "hour")
        .agg(
            avg("traffic_volume").alias("avg_traffic_volume"),
            avg("avg_speed_kmph").alias("avg_speed_kmph"),
            avg("travel_time_minutes").alias("avg_travel_time_minutes"),
        )
        .orderBy("area_name", "hour")
    )

    area_hourly.write.mode("overwrite").parquet(str(output_dir / "area_hourly_summary"))
    spark.stop()


if __name__ == "__main__":
    run_spark_pipeline()
    print(f"Spark output written to {settings.spark_output_dir}")
