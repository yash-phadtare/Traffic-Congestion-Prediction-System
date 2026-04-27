from __future__ import annotations

import sys
from datetime import datetime
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

from traffic_pipeline import AREA_PROFILES, analyze_route, get_forecast, get_processed_data, predict_traffic


def apply_premium_chart_styling(fig):
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_family="Outfit, sans-serif",
        margin=dict(l=10, r=10, t=50, b=10),
        hovermode="x unified"
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.1)", zeroline=False)
    return fig


def inject_custom_css() -> None:
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
        
        html, body, [class*="css"]  {
            font-family: 'Outfit', sans-serif;
        }
        
        div[data-testid="metric-container"] {
            background-color: var(--secondary-background-color);
            border: 1px solid rgba(128,128,128, 0.15);
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        div[data-testid="metric-container"]:hover {
            transform: translateY(-4px);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
        }
        div[data-testid="stMetricValue"] {
            font-size: 2.2rem !important;
            font-weight: 700 !important;
            color: var(--primary-color) !important;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 1.05rem !important;
            opacity: 0.8;
            font-weight: 500 !important;
        }
        
        .stButton>button {
            border-radius: 10px !important;
            font-weight: 600 !important;
            padding: 0.6rem 2rem !important;
            transition: all 0.2s ease !important;
        }
        .stButton>button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 12px rgba(0,0,0, 0.1) !important;
        }
        
        /* Glassmorphic effect on sidebar */
        [data-testid="stSidebar"] {
            border-right: 1px solid rgba(128,128,128,0.1);
        }
        </style>
    """, unsafe_allow_html=True)


def init_state() -> None:
    st.session_state.setdefault("area_result", None)
    st.session_state.setdefault("area_error", None)
    st.session_state.setdefault("route_result", None)
    st.session_state.setdefault("route_error", None)


def build_common_inputs(area_names: list[str]) -> dict:
    with st.sidebar:
        st.markdown("### 🎛️ Simulation Controls")
        st.markdown("Adjust the parameters to preview realtime AI estimations.")
        st.divider()

        with st.expander("⏱️ Date & Time", expanded=True):
            selected_date = st.date_input("Date", value=pd.Timestamp("2024-04-15").date())
            selected_time = st.time_input("Time", value=datetime.strptime("08:30:00", "%H:%M:%S").time())
            
        with st.expander("☁️ Environment Settings", expanded=True):
            weather_condition = st.selectbox("Weather", options=["Clear", "Cloudy", "Rain", "Storm"])
            temperature_c = st.slider("Temperature (°C)", min_value=15.0, max_value=45.0, value=29.0)
            humidity_pct = st.slider("Humidity (%)", min_value=20.0, max_value=100.0, value=68.0)
            precipitation_mm = st.slider("Rainfall (mm)", min_value=0.0, max_value=50.0, value=2.0)
            
        with st.expander("🚧 Road Incidents", expanded=True):
            road_incident_count = st.slider("Active Incidents", min_value=0, max_value=6, value=1)

    return {
        "timestamp": datetime.combine(selected_date, selected_time),
        "temperature_c": temperature_c,
        "humidity_pct": humidity_pct,
        "precipitation_mm": precipitation_mm,
        "weather_condition": weather_condition,
        "road_incident_count": road_incident_count,
        "area_names": area_names,
    }


def render_area_prediction(common_inputs: dict) -> None:
    st.markdown("### 🏙️ Single Area AI Predictor")
    st.markdown("Select a target area and simulate traffic loads to pinpoint exact congestion categories.")
    form_left, form_right = st.columns(2)

    with form_left:
        area_name = st.selectbox("Select Target Area", options=common_inputs["area_names"])
        traffic_volume = st.slider("Simulated Traffic Volume", min_value=150, max_value=2200, value=1050)
    with form_right:
        avg_speed_kmph = st.slider("Estimated Average Speed (kmph)", min_value=8.0, max_value=70.0, value=26.0)

    st.write("") # Spacer
    if st.button("🚀 Run Area Prediction", use_container_width=True):
        try:
            with st.spinner("Predicting congestion..."):
                st.session_state["area_result"] = predict_traffic(
                    {
                        "timestamp": common_inputs["timestamp"],
                        "area_name": area_name,
                        "temperature_c": common_inputs["temperature_c"],
                        "humidity_pct": common_inputs["humidity_pct"],
                        "precipitation_mm": common_inputs["precipitation_mm"],
                        "weather_condition": common_inputs["weather_condition"],
                        "traffic_volume": traffic_volume,
                        "avg_speed_kmph": avg_speed_kmph,
                        "road_incident_count": common_inputs["road_incident_count"],
                    }
                )
                st.session_state["area_error"] = None
        except Exception as exc:  # pragma: no cover - UI safeguard
            st.session_state["area_result"] = None
            st.session_state["area_error"] = str(exc)

    if st.session_state["area_error"]:
        st.error(f"Area prediction failed: {st.session_state['area_error']}")

    if st.session_state["area_result"]:
        st.success("Analysis Complete!")
        result = st.session_state["area_result"]
        metric_left, metric_mid, metric_right = st.columns(3)
        metric_left.metric("AI Congestion Status", result["predicted_congestion_level"])
        metric_mid.metric("Expected Travel Time", f"{result['predicted_travel_time_minutes']} min")
        metric_right.metric("Monitored Volume", str(result["input_features"]["traffic_volume"]))


def render_route_map(start_area: str, end_area: str) -> None:
    if not FOLIUM_AVAILABLE:
        return

    profile_lookup = {area["area_name"]: area for area in AREA_PROFILES}
    start_profile = profile_lookup[start_area]
    end_profile = profile_lookup[end_area]

    route_map = folium.Map(
        location=[
            (start_profile["latitude"] + end_profile["latitude"]) / 2,
            (start_profile["longitude"] + end_profile["longitude"]) / 2,
        ],
        zoom_start=11,
        tiles="CartoDB positron",
    )
    folium.Marker(
        [start_profile["latitude"], start_profile["longitude"]],
        popup=f"Start: {start_area}",
        icon=folium.Icon(color="green"),
    ).add_to(route_map)
    folium.Marker(
        [end_profile["latitude"], end_profile["longitude"]],
        popup=f"End: {end_area}",
        icon=folium.Icon(color="red"),
    ).add_to(route_map)
    folium.PolyLine(
        [
            [start_profile["latitude"], start_profile["longitude"]],
            [end_profile["latitude"], end_profile["longitude"]],
        ],
        color="#1d3557",
        weight=5,
        opacity=0.8,
    ).add_to(route_map)
    st_folium(route_map, use_container_width=True, height=420)


def render_route_analysis(common_inputs: dict) -> None:
    st.markdown("### 🗺️ Start-to-End Route Intelligence")
    st.markdown("Calculate complex multi-area dynamics, expected ETA delays, and route risk metrics.")
    route_left, route_right = st.columns(2)

    with route_left:
        start_area = st.selectbox("Depature Node", options=common_inputs["area_names"], index=0)
    with route_right:
        end_area = st.selectbox("Arrival Node", options=common_inputs["area_names"], index=min(3, len(common_inputs["area_names"]) - 1))

    st.write("")
    if st.button("📍 Calculate Best Route", use_container_width=True):
        try:
            with st.spinner("Analyzing route..."):
                st.session_state["route_result"] = analyze_route(
                    {
                        "timestamp": common_inputs["timestamp"],
                        "start_area": start_area,
                        "end_area": end_area,
                        "temperature_c": common_inputs["temperature_c"],
                        "humidity_pct": common_inputs["humidity_pct"],
                        "precipitation_mm": common_inputs["precipitation_mm"],
                        "weather_condition": common_inputs["weather_condition"],
                        "road_incident_count": common_inputs["road_incident_count"],
                    }
                )
                st.session_state["route_error"] = None
        except Exception as exc:  # pragma: no cover - UI safeguard
            st.session_state["route_result"] = None
            st.session_state["route_error"] = str(exc)

    if start_area == end_area:
        st.warning("Choose different start and end areas for route analysis.")

    if st.session_state["route_error"]:
        st.error(f"Route analysis failed: {st.session_state['route_error']}")

    if st.session_state["route_result"] and start_area != end_area:
        route_result = st.session_state["route_result"]
        route_metrics = st.columns(4)
        route_metrics[0].metric("Route Distance", f"{route_result['route']['estimated_distance_km']} km")
        route_metrics[1].metric("ETA (Arrival)", f"{route_result['prediction']['predicted_travel_time_minutes']} min")
        route_metrics[2].metric("Inferred Delay", f"{route_result['analysis']['expected_delay_minutes']} min")
        route_metrics[3].metric("Risk Index", f"{route_result['analysis']['route_risk_score']}/100")

        st.info(
            f"**🔥 Hotspot Intersect:** {route_result['route']['hotspot_area']}  |  "
            f"**🕒 Optimal Departure Time:** {route_result['analysis']['recommended_departure']['timestamp'].replace('T', ' ')}"
        )

        route_trend_df = pd.DataFrame(route_result["trend"])
        route_chart = px.line(
            route_trend_df,
            x="hour",
            y="estimated_travel_time_minutes",
            color="estimated_congestion_level",
            markers=True,
            title=f"24-Hour Travel Time Trend: {route_result['route']['start_area']} → {route_result['route']['end_area']}",
        )
        st.plotly_chart(apply_premium_chart_styling(route_chart), use_container_width=True)
        render_route_map(route_result["route"]["start_area"], route_result["route"]["end_area"])


def render_area_map(df: pd.DataFrame) -> None:
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


def render_analytics(df: pd.DataFrame, forecast_df: pd.DataFrame) -> None:
    st.markdown("### 📊 Global System Analytics & Forecasting")
    st.markdown("Macro-level city trends, dataset ingestion stats, and time-series ARIMA prediction curves.")
    st.write("")
    
    top_left, top_right = st.columns(2)

    with top_left:
        hourly_df = df.groupby("hour", as_index=False).agg(
            average_travel_time=("travel_time_minutes", "mean"),
            average_traffic_volume=("traffic_volume", "mean"),
        )
        fig_hourly = px.line(
            hourly_df,
            x="hour",
            y="average_travel_time",
            markers=True,
            title="Hourly Baseline Travel Thresholds",
        )
        fig_hourly.update_traces(line=dict(width=3, color='#3b82f6'), marker=dict(size=8))
        st.plotly_chart(apply_premium_chart_styling(fig_hourly), use_container_width=True)

    with top_right:
        congestion_counts = df["congestion_level"].value_counts().reset_index()
        congestion_counts.columns = ["congestion_level", "count"]
        fig_pie = px.pie(
            congestion_counts,
            names="congestion_level",
            values="count",
            hole=0.4,
            title="Congestion Category Distribution",
            color="congestion_level",
            color_discrete_map={"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"},
        )
        fig_pie.update_layout(font_family="Outfit", margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    middle_left, middle_right = st.columns(2)

    with middle_left:
        fig_forecast = px.area(
            forecast_df,
            x="timestamp",
            y="forecast_travel_time_minutes",
            color="forecast_congestion_level",
            title="ARIMA 24-Hour Congestion Projection",
        )
        fig_forecast.update_traces(opacity=0.6)
        st.plotly_chart(apply_premium_chart_styling(fig_forecast), use_container_width=True)

    with middle_right:
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
            color_continuous_scale="Blues",
            title="Performance Disparity by Region",
        )
        st.plotly_chart(apply_premium_chart_styling(fig_area), use_container_width=True)

    st.markdown("### 🗺️ Geographical Node Graph")
    render_area_map(df)
    st.markdown("### 🗃️ Underling Processed Data Snapshot")
    st.dataframe(df.head(25), use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Traffic Intel AI", layout="wide", page_icon="🚦")
    
    inject_custom_css()
    
    st.title("⚡ Traffic Intelligence & Prediction")
    st.markdown("<p style='font-size: 1.15rem; color: #64748b; margin-top: -10px;'>End-to-End Area Forecasting, Route Analytics, and Realtime Congestion Simulation</p>", unsafe_allow_html=True)
    st.write("")

    init_state()
    df = get_processed_data()
    forecast_df = get_forecast()
    area_names = [area["area_name"] for area in AREA_PROFILES]
    common_inputs = build_common_inputs(area_names)

    prediction_tab, route_tab, analytics_tab = st.tabs(["Area Prediction", "Route Analysis", "Analytics"])

    with prediction_tab:
        render_area_prediction(common_inputs)

    with route_tab:
        render_route_analysis(common_inputs)

    with analytics_tab:
        render_analytics(df, forecast_df)


if __name__ == "__main__":
    main()
