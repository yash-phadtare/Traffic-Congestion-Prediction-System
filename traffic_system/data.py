import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from traffic_system.config import (
    settings, AREA_PROFILES, RUSH_HOURS, HOLIDAY_MONTH_DAY, 
    AREA_PROFILE_MAP, NUMERIC_COLUMNS, ensure_directories
)

def _assign_weather_condition(precipitation: pd.Series, humidity: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [precipitation >= 16, precipitation >= 4, humidity >= 78],
            ["Storm", "Rain", "Cloudy"],
            default="Clear",
        )
    )

def generate_synthetic_traffic_data(
    output_path: Path | None = None, num_days: int = 120, seed: int = settings.random_state
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    output_path = output_path or settings.raw_data_path
    ensure_directories([output_path.parent])

    timestamps = pd.date_range("2024-01-01", periods=num_days * 24, freq="h")
    time_df = pd.DataFrame({"timestamp": timestamps})
    area_df = pd.DataFrame(AREA_PROFILES)
    df = time_df.assign(_key=1).merge(area_df.assign(_key=1), on="_key").drop(columns="_key")

    hour = df["timestamp"].dt.hour
    day_of_week = df["timestamp"].dt.dayofweek
    month = df["timestamp"].dt.month
    day_of_year = df["timestamp"].dt.dayofyear
    is_weekend = day_of_week >= 5
    is_rush_hour = hour.isin(RUSH_HOURS)
    is_holiday = pd.Series(
        [(month_value, day_value) in HOLIDAY_MONTH_DAY for month_value, day_value in zip(month, df["timestamp"].dt.day)]
    )

    commute_wave = np.sin((hour - 7) / 24 * 2 * np.pi) + np.sin((hour - 17) / 24 * 2 * np.pi)
    seasonal_temp = 27 + 5 * np.sin(day_of_year / 365 * 2 * np.pi)
    weekend_factor = np.where(is_weekend, 0.84, 1.0)
    holiday_factor = np.where(is_holiday, 0.72, 1.0)
    rush_factor = np.where(is_rush_hour, 1.32, 0.90)

    monsoon_prob = np.where(month.isin([6, 7, 8, 9]), 0.44, 0.12)
    precipitation = np.where(rng.random(len(df)) < monsoon_prob, rng.gamma(shape=2.2, scale=3.2, size=len(df)), 0.0)
    temperature_c = seasonal_temp + rng.normal(0, 2.8, len(df))
    humidity_pct = 58 + precipitation * 1.8 + np.where(month.isin([6, 7, 8, 9]), 16, 0) + rng.normal(0, 8, len(df))
    humidity_pct = np.clip(humidity_pct, 30, 99)
    weather_condition = _assign_weather_condition(pd.Series(precipitation), pd.Series(humidity_pct))

    incident_lambda = 0.10 + np.where(is_rush_hour, 0.18, 0.0) + (precipitation > 0) * 0.12
    road_incident_count = rng.poisson(lam=incident_lambda)
    traffic_volume = (
        df["base_traffic_volume"] * rush_factor * weekend_factor * holiday_factor
        + (commute_wave + 1.6) * 80
        + precipitation * 4.5
        + road_incident_count * 55
        + np.where((day_of_week == 4) & (hour >= 18), 110, 0)
        + rng.normal(0, 55, len(df))
    )
    traffic_volume = np.clip(traffic_volume.round(), 180, None)

    speed_drop = 0.018 * traffic_volume + precipitation * 0.6 + road_incident_count * 4.5
    avg_speed_kmph = df["base_speed_kmph"] + rng.normal(0, 3, len(df)) - speed_drop / 10
    avg_speed_kmph = np.clip(avg_speed_kmph, 8, 65)

    signal_delay = 1 + 0.006 * road_incident_count + precipitation / 150
    travel_time_minutes = (df["route_distance_km"] / avg_speed_kmph) * 60 * signal_delay
    travel_time_minutes = travel_time_minutes + np.where(is_rush_hour, 6.0, 1.5) + rng.normal(0, 1.8, len(df))
    travel_time_minutes = np.clip(travel_time_minutes, 8, None)

    generated_df = pd.DataFrame(
        {
            "timestamp": df["timestamp"],
            "area_name": df["area_name"],
            "latitude": df["latitude"],
            "longitude": df["longitude"],
            "temperature_c": temperature_c.round(2),
            "humidity_pct": humidity_pct.round(2),
            "precipitation_mm": np.round(precipitation, 2),
            "weather_condition": weather_condition,
            "traffic_volume": traffic_volume.astype(int),
            "avg_speed_kmph": np.round(avg_speed_kmph, 2),
            "road_incident_count": road_incident_count.astype(int),
            "route_distance_km": df["route_distance_km"],
            "travel_time_minutes": np.round(travel_time_minutes, 2),
            "congestion_level": pd.cut(
                traffic_volume / avg_speed_kmph,
                bins=[-np.inf, 24, 38, np.inf],
                labels=["Low", "Medium", "High"],
            ).astype(str),
        }
    )

    for column in ["temperature_c", "humidity_pct", "avg_speed_kmph"]:
        missing_idx = rng.choice(generated_df.index, size=max(20, len(generated_df) // 250), replace=False)
        generated_df.loc[missing_idx, column] = np.nan

    duplicate_sample = generated_df.sample(n=max(10, len(generated_df) // 400), random_state=seed)
    generated_df = pd.concat([generated_df, duplicate_sample], ignore_index=True)
    generated_df.to_csv(output_path, index=False)
    return generated_df

def load_raw_data(raw_path: Path | None = None) -> pd.DataFrame:
    raw_path = raw_path or settings.raw_data_path
    if not raw_path.exists():
        generate_synthetic_traffic_data(output_path=raw_path)
    return pd.read_csv(raw_path, parse_dates=["timestamp"])

def clean_traffic_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned_df = df.drop_duplicates().sort_values("timestamp").reset_index(drop=True).copy()
    for area_name, profile in AREA_PROFILE_MAP.items():
        area_mask = cleaned_df["area_name"] == area_name
        cleaned_df.loc[area_mask, "latitude"] = cleaned_df.loc[area_mask, "latitude"].fillna(profile["latitude"])
        cleaned_df.loc[area_mask, "longitude"] = cleaned_df.loc[area_mask, "longitude"].fillna(profile["longitude"])

    for column in NUMERIC_COLUMNS:
        cleaned_df[column] = pd.to_numeric(cleaned_df[column], errors="coerce")
        cleaned_df[column] = cleaned_df.groupby("area_name")[column].transform(lambda series: series.fillna(series.median()))
        cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].median())

    for column in ["area_name", "weather_condition", "congestion_level"]:
        cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].mode().iloc[0])

    return cleaned_df

def engineer_time_features(df: pd.DataFrame) -> pd.DataFrame:
    featured_df = df.copy()
    featured_df["hour"] = featured_df["timestamp"].dt.hour
    featured_df["day_of_week"] = featured_df["timestamp"].dt.dayofweek
    featured_df["month"] = featured_df["timestamp"].dt.month
    featured_df["day_name"] = featured_df["timestamp"].dt.day_name()
    featured_df["is_weekend"] = (featured_df["day_of_week"] >= 5).astype(int)
    featured_df["is_rush_hour"] = featured_df["hour"].isin(RUSH_HOURS).astype(int)
    featured_df["is_holiday"] = featured_df["timestamp"].apply(lambda ts: int((ts.month, ts.day) in HOLIDAY_MONTH_DAY))
    featured_df["time_band"] = pd.cut(
        featured_df["hour"],
        bins=[-1, 5, 11, 16, 20, 24],
        labels=["Night", "Morning", "Afternoon", "Evening", "LateNight"],
    ).astype(str)
    featured_df["congestion_index"] = featured_df["traffic_volume"] / featured_df["avg_speed_kmph"].clip(lower=1)
    return featured_df

def preprocess_and_save(raw_path: Path | None = None, processed_path: Path | None = None) -> pd.DataFrame:
    raw_path = raw_path or settings.raw_data_path
    processed_path = processed_path or settings.processed_data_path
    ensure_directories([processed_path.parent])
    processed_df = engineer_time_features(clean_traffic_data(load_raw_data(raw_path)))
    processed_df.to_csv(processed_path, index=False)
    return processed_df

def get_processed_data(force_refresh: bool = False) -> pd.DataFrame:
    if force_refresh or not settings.processed_data_path.exists():
        return preprocess_and_save()
    return pd.read_csv(settings.processed_data_path, parse_dates=["timestamp"])

def generate_visualizations(df: pd.DataFrame, output_dir: Path | None = None) -> None:
    output_dir = output_dir or settings.plot_dir
    ensure_directories([output_dir])
    sns.set_theme(style="whitegrid")

    numeric_columns = [
        "temperature_c", "humidity_pct", "precipitation_mm", "traffic_volume",
        "avg_speed_kmph", "road_incident_count", "travel_time_minutes", "congestion_index",
    ]

    plt.figure(figsize=(10, 6))
    sns.heatmap(df[numeric_columns].corr(), cmap="YlOrRd", annot=True, fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=200)
    plt.close()

    hourly_trend = df.groupby("hour", as_index=False)["travel_time_minutes"].mean()
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=hourly_trend, x="hour", y="travel_time_minutes", marker="o")
    plt.title("Average Travel Time by Hour")
    plt.tight_layout()
    plt.savefig(output_dir / "hourly_travel_trend.png", dpi=200)
    plt.close()

    heatmap_source = df.pivot_table(values="traffic_volume", index="area_name", columns="hour", aggfunc="mean")
    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_source.sort_index(), cmap="rocket_r")
    plt.title("Congestion Heatmap by Area and Hour")
    plt.tight_layout()
    plt.savefig(output_dir / "area_hour_heatmap.png", dpi=200)
    plt.close()

def get_data_payload(limit: int = 20) -> dict[str, Any]:
    df = get_processed_data()
    preview_df = df.head(limit).copy()
    preview_df["timestamp"] = preview_df["timestamp"].astype(str)
    return {
        "rows": int(len(df)),
        "columns": df.columns.tolist(),
        "summary": {
            "average_travel_time_minutes": round(float(df["travel_time_minutes"].mean()), 2),
            "average_traffic_volume": round(float(df["traffic_volume"].mean()), 2),
            "congestion_distribution": df["congestion_level"].value_counts().to_dict(),
            "areas": sorted(df["area_name"].unique().tolist()),
        },
        "preview": preview_df.to_dict(orient="records"),
    }

def get_area_summary() -> pd.DataFrame:
    df = get_processed_data()
    summary_df = df.groupby("area_name", as_index=False).agg(
        latitude=("latitude", "median"),
        longitude=("longitude", "median"),
        average_traffic_volume=("traffic_volume", "mean"),
        average_speed_kmph=("avg_speed_kmph", "mean"),
        average_travel_time_minutes=("travel_time_minutes", "mean"),
        average_route_distance_km=("route_distance_km", "mean"),
        congestion_mode=("congestion_level", lambda series: series.mode().iloc[0]),
    )
    return summary_df
