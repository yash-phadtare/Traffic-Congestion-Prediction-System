import pandas as pd
import numpy as np
from typing import Any

from traffic_system.config import AREA_PROFILE_MAP, RUSH_HOURS, HOLIDAY_MONTH_DAY, WEATHER_VOLUME_BOOST, WEATHER_SPEED_PENALTY
from traffic_system.data import get_processed_data
from traffic_system.ml import ensure_model_bundle

def get_realtime_simulation(sample_size: int = 25, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    latest_df = get_processed_data().sort_values("timestamp").tail(sample_size).copy()
    latest_df["traffic_volume"] = (latest_df["traffic_volume"] * rng.uniform(0.9, 1.1, len(latest_df))).round().astype(int)
    latest_df["avg_speed_kmph"] = (latest_df["avg_speed_kmph"] * rng.uniform(0.88, 1.08, len(latest_df))).round(2)
    latest_df["travel_time_minutes"] = (latest_df["travel_time_minutes"] * rng.uniform(0.92, 1.12, len(latest_df))).round(2)
    latest_df["timestamp"] = pd.Timestamp.utcnow().floor("min")
    return latest_df

def build_prediction_features(payload: dict[str, Any]) -> pd.DataFrame:
    timestamp = pd.Timestamp(payload["timestamp"])
    hour = timestamp.hour
    day_of_week = timestamp.dayofweek
    profile = AREA_PROFILE_MAP.get(payload["area_name"], {})

    latitude = payload.get("latitude") or profile.get("latitude", 18.5204)
    longitude = payload.get("longitude") or profile.get("longitude", 73.8567)
    route_distance_km = payload.get("route_distance_km") or profile.get("route_distance_km", 10.0)

    row = {
        "area_name": payload["area_name"],
        "latitude": latitude,
        "longitude": longitude,
        "temperature_c": payload["temperature_c"],
        "humidity_pct": payload["humidity_pct"],
        "precipitation_mm": payload.get("precipitation_mm", 0.0),
        "weather_condition": payload.get("weather_condition", "Clear"),
        "traffic_volume": payload["traffic_volume"],
        "avg_speed_kmph": payload["avg_speed_kmph"],
        "road_incident_count": payload.get("road_incident_count", 0),
        "route_distance_km": route_distance_km,
        "hour": hour,
        "day_of_week": day_of_week,
        "month": timestamp.month,
        "is_weekend": int(day_of_week >= 5),
        "is_rush_hour": int(hour in RUSH_HOURS),
        "is_holiday": int((timestamp.month, timestamp.day) in HOLIDAY_MONTH_DAY),
        "time_band": "Night" if hour <= 5 else "Morning" if hour <= 11 else "Afternoon" if hour <= 16 else "Evening" if hour <= 20 else "LateNight",
        "congestion_index": payload["traffic_volume"] / max(payload["avg_speed_kmph"], 1),
    }
    return pd.DataFrame([row])

def predict_traffic(payload: dict[str, Any]) -> dict[str, Any]:
    bundle = ensure_model_bundle()
    features = build_prediction_features(payload)
    predicted_time = float(bundle["regressor"].predict(features)[0])
    encoded_level = int(bundle["classifier"].predict(features)[0])
    predicted_level = bundle["label_encoder"].inverse_transform([encoded_level])[0]
    return {
        "predicted_travel_time_minutes": round(predicted_time, 2),
        "predicted_congestion_level": predicted_level,
        "model_version": "v1.0",
        "input_features": features.to_dict(orient="records")[0],
    }

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    lat1_rad, lon1_rad = np.radians([lat1, lon1])
    lat2_rad, lon2_rad = np.radians([lat2, lon2])
    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad
    a_value = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2
    )
    return float(2 * radius_km * np.arcsin(np.sqrt(a_value)))

def _get_area_hour_snapshot(area_name: str, hour: int) -> dict[str, Any]:
    df = get_processed_data()
    area_df = df[df["area_name"] == area_name]
    hour_df = area_df[area_df["hour"] == hour]
    base_df = hour_df if not hour_df.empty else area_df

    return {
        "traffic_volume": float(base_df["traffic_volume"].mean()),
        "avg_speed_kmph": float(base_df["avg_speed_kmph"].mean()),
        "travel_time_minutes": float(base_df["travel_time_minutes"].mean()),
        "latitude": float(area_df["latitude"].median()),
        "longitude": float(area_df["longitude"].median()),
        "route_distance_km": float(area_df["route_distance_km"].mean()),
    }

def _estimate_route_distance_km(start_area: str, end_area: str) -> float:
    start_profile = AREA_PROFILE_MAP[start_area]
    end_profile = AREA_PROFILE_MAP[end_area]
    if start_area == end_area:
        return float(start_profile["route_distance_km"])

    geo_distance = _haversine_km(
        start_profile["latitude"], start_profile["longitude"],
        end_profile["latitude"], end_profile["longitude"],
    )
    return round(max(geo_distance * 1.28, 2.5), 2)

def _estimate_route_metrics(
    start_area: str, end_area: str, timestamp: pd.Timestamp,
    weather_condition: str, precipitation_mm: float, road_incident_count: int,
) -> dict[str, float]:
    start_snapshot = _get_area_hour_snapshot(start_area, timestamp.hour)
    end_snapshot = _get_area_hour_snapshot(end_area, timestamp.hour)
    route_distance_km = _estimate_route_distance_km(start_area, end_area)

    base_volume = (start_snapshot["traffic_volume"] + end_snapshot["traffic_volume"]) / 2
    base_speed = 2 / ((1 / max(start_snapshot["avg_speed_kmph"], 1)) + (1 / max(end_snapshot["avg_speed_kmph"], 1)))

    rush_multiplier = 1.16 if timestamp.hour in RUSH_HOURS else 0.97
    weekend_multiplier = 0.90 if timestamp.dayofweek >= 5 else 1.0
    weather_boost = WEATHER_VOLUME_BOOST.get(weather_condition, 0.0)
    weather_penalty = WEATHER_SPEED_PENALTY.get(weather_condition, 0.0)

    adjusted_volume = base_volume * rush_multiplier * weekend_multiplier * (1 + weather_boost)
    adjusted_volume += route_distance_km * 12 + road_incident_count * 60 + precipitation_mm * 4

    adjusted_speed = base_speed * (1 - weather_penalty)
    adjusted_speed -= road_incident_count * 2.8
    adjusted_speed -= precipitation_mm * 0.08
    adjusted_speed = max(adjusted_speed, 8.0)

    midpoint_latitude = (start_snapshot["latitude"] + end_snapshot["latitude"]) / 2
    midpoint_longitude = (start_snapshot["longitude"] + end_snapshot["longitude"]) / 2

    return {
        "traffic_volume": round(adjusted_volume),
        "avg_speed_kmph": round(adjusted_speed, 2),
        "route_distance_km": route_distance_km,
        "latitude": round(midpoint_latitude, 6),
        "longitude": round(midpoint_longitude, 6),
        "start_volume": round(start_snapshot["traffic_volume"]),
        "end_volume": round(end_snapshot["traffic_volume"]),
        "start_speed": round(start_snapshot["avg_speed_kmph"], 2),
        "end_speed": round(end_snapshot["avg_speed_kmph"], 2),
    }

def _route_trend_for_day(
    start_area: str, end_area: str, route_distance_km: float, weather_condition: str,
) -> list[dict[str, Any]]:
    trend_rows: list[dict[str, Any]] = []
    current_day = pd.Timestamp("2024-04-15")
    for hour in range(24):
        timestamp = current_day + pd.Timedelta(hours=hour)
        estimated = _estimate_route_metrics(
            start_area=start_area, end_area=end_area, timestamp=timestamp,
            weather_condition=weather_condition, precipitation_mm=0.0, road_incident_count=0,
        )
        estimated_time = round((route_distance_km / max(estimated["avg_speed_kmph"], 1)) * 60 + (6 if hour in RUSH_HOURS else 2), 2)
        congestion_level = "Low" if estimated["traffic_volume"] / max(estimated["avg_speed_kmph"], 1) < 24 else "Medium" if estimated["traffic_volume"] / max(estimated["avg_speed_kmph"], 1) < 38 else "High"
        trend_rows.append(
            {
                "hour": hour,
                "estimated_travel_time_minutes": estimated_time,
                "estimated_avg_speed_kmph": estimated["avg_speed_kmph"],
                "estimated_traffic_volume": estimated["traffic_volume"],
                "estimated_congestion_level": congestion_level,
            }
        )
    return trend_rows

def _recommended_departure(start_area: str, end_area: str, timestamp: pd.Timestamp, weather_condition: str) -> dict[str, Any]:
    options: list[dict[str, Any]] = []
    for offset in range(0, 6):
        candidate_time = timestamp + pd.Timedelta(hours=offset)
        estimate = _estimate_route_metrics(
            start_area=start_area, end_area=end_area, timestamp=candidate_time,
            weather_condition=weather_condition, precipitation_mm=0.0, road_incident_count=0,
        )
        proxy_minutes = round((estimate["route_distance_km"] / max(estimate["avg_speed_kmph"], 1)) * 60 + (6 if candidate_time.hour in RUSH_HOURS else 2), 2)
        options.append({"timestamp": candidate_time.isoformat(), "proxy_travel_time_minutes": proxy_minutes})
    return min(options, key=lambda item: item["proxy_travel_time_minutes"])

def analyze_route(payload: dict[str, Any]) -> dict[str, Any]:
    start_area = payload["start_area"]
    end_area = payload["end_area"]
    timestamp = pd.Timestamp(payload["timestamp"])
    weather_condition = payload.get("weather_condition", "Clear")
    precipitation_mm = float(payload.get("precipitation_mm", 0.0))
    road_incident_count = int(payload.get("road_incident_count", 0))
    temperature_c = float(payload.get("temperature_c", 28.0))
    humidity_pct = float(payload.get("humidity_pct", 65.0))

    route_metrics = _estimate_route_metrics(
        start_area=start_area, end_area=end_area, timestamp=timestamp,
        weather_condition=weather_condition, precipitation_mm=precipitation_mm,
        road_incident_count=road_incident_count,
    )

    model_payload = {
        "timestamp": timestamp,
        "area_name": start_area,
        "temperature_c": temperature_c,
        "humidity_pct": humidity_pct,
        "precipitation_mm": precipitation_mm,
        "weather_condition": weather_condition,
        "traffic_volume": int(route_metrics["traffic_volume"]),
        "avg_speed_kmph": float(route_metrics["avg_speed_kmph"]),
        "road_incident_count": road_incident_count,
        "latitude": route_metrics["latitude"],
        "longitude": route_metrics["longitude"],
        "route_distance_km": route_metrics["route_distance_km"],
    }
    prediction = predict_traffic(model_payload)

    free_flow_speed = max(
        (AREA_PROFILE_MAP[start_area]["base_speed_kmph"] + AREA_PROFILE_MAP[end_area]["base_speed_kmph"]) / 2, 18,
    )
    free_flow_time = round((route_metrics["route_distance_km"] / free_flow_speed) * 60, 2)
    
    # FIX: Delay calculations maxed with 0 to make it physically legitimate and accurate
    delay_minutes = max(0.0, round(prediction["predicted_travel_time_minutes"] - free_flow_time, 2))
    delay_percent = round((delay_minutes / max(free_flow_time, 1)) * 100, 2)
    
    congestion_ratio = route_metrics["traffic_volume"] / max(route_metrics["avg_speed_kmph"], 1)
    route_risk_score = min(
        100,
        round(
            congestion_ratio + road_incident_count * 8 + precipitation_mm * 0.8
            + (10 if timestamp.hour in RUSH_HOURS else 0), 2,
        ),
    )
    hotspot_area = start_area if route_metrics["start_volume"] >= route_metrics["end_volume"] else end_area

    return {
        "route": {
            "start_area": start_area,
            "end_area": end_area,
            "midpoint_latitude": route_metrics["latitude"],
            "midpoint_longitude": route_metrics["longitude"],
            "estimated_distance_km": route_metrics["route_distance_km"],
            "hotspot_area": hotspot_area,
        },
        "prediction": prediction,
        "analysis": {
            "free_flow_travel_time_minutes": free_flow_time,
            "expected_delay_minutes": delay_minutes,
            "delay_percent": delay_percent,
            "route_risk_score": route_risk_score,
            "estimated_corridor_speed_kmph": route_metrics["avg_speed_kmph"],
            "estimated_corridor_volume": route_metrics["traffic_volume"],
            "recommended_departure": _recommended_departure(start_area, end_area, timestamp, weather_condition),
        },
        "trend": _route_trend_for_day(start_area, end_area, route_metrics["route_distance_km"], weather_condition),
    }
