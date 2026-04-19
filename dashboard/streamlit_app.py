from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import folium
    from streamlit_folium import st_folium

    FOLIUM_AVAILABLE = True
except ImportError:  # pragma: no cover - UI fallback
    FOLIUM_AVAILABLE = False

from traffic_pipeline import AREA_PROFILES, get_forecast, get_processed_data, predict_traffic


st.set_page_config(page_title="Traffic Congestion Dashboard", layout="wide")
st.title("Traffic Congestion Prediction System")
st.caption("Interactive dashboard for travel-time prediction, congestion analysis, and future trend forecasting.")

df = get_processed_data()
forecast_df = get_forecast()
area_names = [area["area_name"] for area in AREA_PROFILES]

with st.sidebar:
    st.header("Prediction Inputs")
    area_name = st.selectbox("Area", options=area_names)
    timestamp = st.datetime_input("Timestamp", value=pd.Timestamp("2024-04-15 08:30:00"))
    temperature_c = st.slider("Temperature (C)", min_value=15.0, max_value=42.0, value=29.0)
    humidity_pct = st.slider("Humidity (%)", min_value=20.0, max_value=100.0, value=68.0)
    precipitation_mm = st.slider("Rainfall (mm)", min_value=0.0, max_value=50.0, value=2.0)
    weather_condition = st.selectbox("Weather", options=["Clear", "Cloudy", "Rain", "Storm"])
    traffic_volume = st.slider("Traffic Volume", min_value=150, max_value=2200, value=1050)
    avg_speed_kmph = st.slider("Average Speed (kmph)", min_value=8.0, max_value=70.0, value=26.0)
    road_incident_count = st.slider("Road Incidents", min_value=0, max_value=6, value=1)

    if st.button("Predict Congestion", use_container_width=True):
        result = predict_traffic(
            {
                "timestamp": pd.Timestamp(timestamp).to_pydatetime(),
                "area_name": area_name,
                "temperature_c": temperature_c,
                "humidity_pct": humidity_pct,
                "precipitation_mm": precipitation_mm,
                "weather_condition": weather_condition,
                "traffic_volume": traffic_volume,
                "avg_speed_kmph": avg_speed_kmph,
                "road_incident_count": road_incident_count,
            }
        )
        st.success(
            f"Predicted congestion: {result['predicted_congestion_level']} | "
            f"Travel time: {result['predicted_travel_time_minutes']} minutes"
        )

top_left, top_right = st.columns(2)

with top_left:
    st.subheader("Traffic Trend by Hour")
    hourly_df = df.groupby("hour", as_index=False).agg(
        average_travel_time=("travel_time_minutes", "mean"),
        average_traffic_volume=("traffic_volume", "mean"),
    )
    fig_hourly = px.line(
        hourly_df,
        x="hour",
        y="average_travel_time",
        markers=True,
        title="Average Travel Time Through the Day",
    )
    st.plotly_chart(fig_hourly, use_container_width=True)

with top_right:
    st.subheader("Congestion Mix")
    congestion_counts = df["congestion_level"].value_counts().reset_index()
    congestion_counts.columns = ["congestion_level", "count"]
    fig_pie = px.pie(
        congestion_counts,
        names="congestion_level",
        values="count",
        color="congestion_level",
        color_discrete_map={"Low": "#2E8B57", "Medium": "#F4A261", "High": "#D62828"},
    )
    st.plotly_chart(fig_pie, use_container_width=True)

middle_left, middle_right = st.columns(2)

with middle_left:
    st.subheader("Forecasted Future Congestion")
    fig_forecast = px.line(
        forecast_df,
        x="timestamp",
        y="forecast_travel_time_minutes",
        color="forecast_congestion_level",
        title="24-Hour Travel-Time Forecast",
    )
    st.plotly_chart(fig_forecast, use_container_width=True)

with middle_right:
    st.subheader("Area Performance")
    area_summary = df.groupby("area_name", as_index=False).agg(
        average_speed=("avg_speed_kmph", "mean"),
        average_volume=("traffic_volume", "mean"),
        average_travel_time=("travel_time_minutes", "mean"),
    )
    fig_area = px.bar(
        area_summary.sort_values("average_travel_time", ascending=False),
        x="area_name",
        y="average_travel_time",
        color="average_volume",
        title="Average Travel Time by Area",
    )
    st.plotly_chart(fig_area, use_container_width=True)

st.subheader("Map View")
map_df = (
    df.groupby(["area_name", "latitude", "longitude"], as_index=False)["traffic_volume"]
    .mean()
    .rename(columns={"traffic_volume": "average_traffic_volume"})
)

if FOLIUM_AVAILABLE:
    map_center = [map_df["latitude"].mean(), map_df["longitude"].mean()]
    folium_map = folium.Map(location=map_center, zoom_start=11, tiles="CartoDB positron")
    for _, row in map_df.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=max(row["average_traffic_volume"] / 180, 6),
            popup=f"{row['area_name']} - Avg volume: {row['average_traffic_volume']:.0f}",
            color="#D62828",
            fill=True,
            fill_opacity=0.6,
        ).add_to(folium_map)
    st_folium(folium_map, use_container_width=True, height=420)
else:
    st.map(map_df.rename(columns={"latitude": "lat", "longitude": "lon"}))

st.subheader("Processed Data Snapshot")
st.dataframe(df.head(25), use_container_width=True)
