from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_endpoint():
    payload = {
        "timestamp": "2024-04-15T08:30:00",
        "area_name": "Hinjewadi",
        "temperature_c": 29.0,
        "humidity_pct": 70.0,
        "precipitation_mm": 3.0,
        "weather_condition": "Rain",
        "traffic_volume": 1120,
        "avg_speed_kmph": 24.0,
        "road_incident_count": 1,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "predicted_travel_time_minutes" in body
    assert body["predicted_congestion_level"] in {"Low", "Medium", "High"}


def test_route_analyze_endpoint():
    payload = {
        "timestamp": "2024-04-15T08:30:00",
        "start_area": "Hinjewadi",
        "end_area": "Kharadi",
        "temperature_c": 29.0,
        "humidity_pct": 70.0,
        "precipitation_mm": 3.0,
        "weather_condition": "Rain",
        "road_incident_count": 1,
    }
    response = client.post("/route-analyze", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["route"]["start_area"] == "Hinjewadi"
    assert body["route"]["end_area"] == "Kharadi"
    assert body["prediction"]["predicted_congestion_level"] in {"Low", "Medium", "High"}
    assert len(body["trend"]) == 24
