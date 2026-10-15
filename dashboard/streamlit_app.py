from __future__ import annotations

import html
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

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


CONGESTION_COLORS = {
    "Low": "#2dd4bf",
    "Medium": "#fbbf24",
    "High": "#fb7185",
}


@st.cache_data(ttl=300, show_spinner=False)
def load_dashboard_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    return get_processed_data(), get_forecast()


def apply_chart_styling(fig, *, hovermode: str | bool = "x unified"):
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, ui-sans-serif, system-ui, sans-serif", color="#b8c6d9", size=13),
        title=dict(font=dict(color="#f4f8fc", size=17), x=0, xanchor="left"),
        margin=dict(l=8, r=12, t=54, b=12),
        hovermode=hovermode,
        hoverlabel=dict(bgcolor="#111f32", bordercolor="#2b405b", font_color="#f4f8fc"),
        legend=dict(title=None, orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="#263950", tickfont_color="#8fa1b8")
    fig.update_yaxes(gridcolor="rgba(143,161,184,0.13)", zeroline=False, tickfont_color="#8fa1b8")
    return fig


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --canvas: #07111f;
            --surface: #0d1a2b;
            --surface-raised: #122238;
            --border: #263950;
            --text: #f4f8fc;
            --muted: #9bacc1;
            --accent: #22d3ee;
            --success: #2dd4bf;
        }

        html, body, [class*="css"] {
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at 82% -12%, rgba(34, 211, 238, 0.08), transparent 28rem),
                var(--canvas);
            color: var(--text);
        }

        .block-container {
            max-width: 1440px;
            padding: 2rem 2.4rem 4rem;
        }

        [data-testid="stHeader"] {
            background: rgba(7, 17, 31, 0.88);
            border-bottom: 1px solid rgba(38, 57, 80, 0.7);
            backdrop-filter: blur(12px);
        }

        [data-testid="stSidebar"] {
            background: #091526;
            border-right: 1px solid var(--border);
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1.35rem;
        }

        [data-testid="stSidebar"] hr {
            border-color: var(--border);
        }

        .sidebar-brand {
            color: var(--text);
            font-size: 1.05rem;
            font-weight: 750;
            letter-spacing: -0.015em;
            margin-bottom: 0.2rem;
        }

        .sidebar-copy {
            color: var(--muted);
            font-size: 0.82rem;
            line-height: 1.5;
            margin-bottom: 1.5rem;
        }

        .sidebar-section {
            color: #d7e1ed;
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            margin: 1.25rem 0 0.45rem;
            text-transform: uppercase;
        }

        .app-header {
            align-items: flex-start;
            display: flex;
            gap: 2rem;
            justify-content: space-between;
            margin: 0.25rem 0 1.45rem;
        }

        .app-header h1 {
            color: var(--text);
            font-size: 2.15rem;
            font-weight: 760;
            letter-spacing: -0.03em;
            line-height: 1.08;
            margin: 0;
        }

        .app-header p {
            color: var(--muted);
            font-size: 0.98rem;
            line-height: 1.6;
            margin: 0.65rem 0 0;
            max-width: 68ch;
        }

        .system-status {
            align-items: center;
            background: #0c2130;
            border: 1px solid #164052;
            border-radius: 999px;
            color: #b9f6ff;
            display: inline-flex;
            flex: 0 0 auto;
            font-size: 0.78rem;
            font-weight: 700;
            gap: 0.55rem;
            margin-top: 0.15rem;
            padding: 0.48rem 0.75rem;
        }

        .status-dot {
            background: var(--success);
            border-radius: 50%;
            box-shadow: 0 0 0 4px rgba(45, 212, 191, 0.12);
            height: 0.45rem;
            width: 0.45rem;
        }

        .context-strip {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 14px;
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            margin: 0 0 1.7rem;
            overflow: hidden;
        }

        .context-item {
            border-right: 1px solid var(--border);
            min-width: 0;
            padding: 0.85rem 1rem;
        }

        .context-item:last-child { border-right: 0; }

        .context-label {
            color: var(--muted);
            display: block;
            font-size: 0.72rem;
            font-weight: 650;
            letter-spacing: 0.04em;
            margin-bottom: 0.18rem;
            text-transform: uppercase;
        }

        .context-value {
            color: var(--text);
            display: block;
            font-size: 0.92rem;
            font-variant-numeric: tabular-nums;
            font-weight: 650;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .section-heading {
            color: var(--text);
            font-size: 1.32rem;
            font-weight: 720;
            letter-spacing: -0.02em;
            margin: 1.45rem 0 0.25rem;
        }

        .section-copy {
            color: var(--muted);
            font-size: 0.91rem;
            line-height: 1.55;
            margin: 0 0 1.1rem;
            max-width: 72ch;
        }

        div[data-testid="stMetric"] {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 14px;
            min-height: 7rem;
            padding: 1rem 1.05rem;
        }

        div[data-testid="stMetricValue"] {
            color: var(--text);
            font-size: 1.65rem;
            font-variant-numeric: tabular-nums;
            font-weight: 730;
            letter-spacing: -0.025em;
        }

        div[data-testid="stMetricLabel"] {
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 650;
            letter-spacing: 0.025em;
        }

        .result-banner {
            align-items: center;
            background: var(--surface-raised);
            border: 1px solid var(--border);
            border-radius: 14px;
            display: flex;
            justify-content: space-between;
            margin: 1.25rem 0 1rem;
            padding: 1rem 1.15rem;
        }

        .result-banner span {
            color: var(--muted);
            display: block;
            font-size: 0.76rem;
            font-weight: 650;
            letter-spacing: 0.04em;
            margin-bottom: 0.18rem;
            text-transform: uppercase;
        }

        .result-banner strong { color: var(--text); font-size: 1.1rem; }

        .level-chip {
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 750;
            padding: 0.38rem 0.68rem;
        }

        .level-low { background: rgba(45, 212, 191, 0.12); color: #7aead9; }
        .level-medium { background: rgba(251, 191, 36, 0.13); color: #fcd76d; }
        .level-high { background: rgba(251, 113, 133, 0.13); color: #fda4b3; }

        .empty-state {
            border: 1px dashed #304660;
            border-radius: 14px;
            color: var(--muted);
            margin-top: 1.1rem;
            padding: 1.2rem;
            text-align: center;
        }

        .empty-state strong {
            color: #dbe6f2;
            display: block;
            font-size: 0.92rem;
            margin-bottom: 0.25rem;
        }

        .route-insight {
            background: #0c2130;
            border: 1px solid #164052;
            border-radius: 12px;
            color: #c6ddec;
            line-height: 1.55;
            margin: 1rem 0 0.4rem;
            padding: 0.85rem 1rem;
        }

        .route-insight b { color: #eaf7fb; }

        .stButton > button,
        [data-testid="stFormSubmitButton"] > button {
            border-radius: 10px;
            font-weight: 700;
            min-height: 2.75rem;
            transition: background-color 180ms ease, border-color 180ms ease, transform 180ms ease;
        }

        .stButton > button:hover,
        [data-testid="stFormSubmitButton"] > button:hover { transform: translateY(-1px); }

        .stButton > button:focus-visible,
        [data-testid="stFormSubmitButton"] > button:focus-visible,
        input:focus-visible,
        [data-baseweb="select"] > div:focus-within {
            outline: 3px solid rgba(34, 211, 238, 0.32) !important;
            outline-offset: 2px;
        }

        button:disabled {
            cursor: not-allowed !important;
            opacity: 0.48 !important;
            transform: none !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            border-bottom: 1px solid var(--border);
            gap: 1.6rem;
        }

        .stTabs [data-baseweb="tab"] {
            background: transparent;
            border-radius: 0;
            color: var(--muted);
            font-size: 0.88rem;
            font-weight: 650;
            padding-left: 0;
            padding-right: 0;
        }

        .stTabs [aria-selected="true"] { color: var(--text) !important; }

        [data-testid="stDataFrame"],
        [data-testid="stPlotlyChart"],
        iframe[title="streamlit_folium.st_folium"] {
            border-radius: 14px;
            overflow: hidden;
        }

        [data-testid="stAlert"] { border-radius: 12px; }

        ::selection { background: rgba(34, 211, 238, 0.28); color: #ffffff; }

        @media (max-width: 900px) {
            .block-container { padding: 1.5rem 1rem 3rem; }
            .app-header { gap: 1rem; }
            .app-header h1 { font-size: 1.75rem; }
            .context-strip { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .context-item:nth-child(2) { border-right: 0; }
            .context-item:nth-child(-n+2) { border-bottom: 1px solid var(--border); }
        }

        @media (max-width: 560px) {
            .app-header { display: block; }
            .system-status { margin-top: 1rem; }
            .context-strip { grid-template-columns: 1fr; }
            .context-item { border-bottom: 1px solid var(--border); border-right: 0; }
            .context-item:last-child { border-bottom: 0; }
            .result-banner { align-items: flex-start; flex-direction: column; gap: 0.75rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state() -> None:
    st.session_state.setdefault("area_result", None)
    st.session_state.setdefault("area_error", None)
    st.session_state.setdefault("route_result", None)
    st.session_state.setdefault("route_error", None)


def build_common_inputs(area_names: list[str]) -> dict[str, Any]:
    with st.sidebar:
        st.markdown('<div class="sidebar-brand">Traffic Intel</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sidebar-copy">Set the operating conditions used across every analysis.</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-section">Time window</div>', unsafe_allow_html=True)
        selected_date = st.date_input("Date", value=pd.Timestamp("2024-04-15").date())
        selected_time = st.time_input("Time", value=datetime.strptime("08:30:00", "%H:%M:%S").time())

        st.markdown('<div class="sidebar-section">Road conditions</div>', unsafe_allow_html=True)
        weather_condition = st.selectbox("Weather", options=["Clear", "Cloudy", "Rain", "Storm"])
        precipitation_mm = st.slider("Rainfall (mm)", min_value=0.0, max_value=50.0, value=2.0)
        road_incident_count = st.slider("Active incidents", min_value=0, max_value=6, value=1)

        st.markdown('<div class="sidebar-section">Environment</div>', unsafe_allow_html=True)
        temperature_c = st.slider("Temperature (°C)", min_value=15.0, max_value=45.0, value=29.0)
        humidity_pct = st.slider("Humidity (%)", min_value=20.0, max_value=100.0, value=68.0)

        st.divider()
        if st.button("Clear analysis results", width="stretch", type="secondary"):
            st.session_state["area_result"] = None
            st.session_state["area_error"] = None
            st.session_state["route_result"] = None
            st.session_state["route_error"] = None

    return {
        "timestamp": datetime.combine(selected_date, selected_time),
        "temperature_c": temperature_c,
        "humidity_pct": humidity_pct,
        "precipitation_mm": precipitation_mm,
        "weather_condition": weather_condition,
        "road_incident_count": road_incident_count,
        "area_names": area_names,
    }


def render_page_header(df: pd.DataFrame, common_inputs: dict[str, Any]) -> None:
    area_count = int(df["area_name"].nunique())
    timestamp_label = common_inputs["timestamp"].strftime("%d %b %Y · %H:%M")
    weather_label = html.escape(str(common_inputs["weather_condition"]))
    st.markdown(
        f"""
        <div class="app-header">
            <div>
                <h1>Traffic intelligence</h1>
                <p>Model congestion, compare route conditions, and inspect network-wide forecasts for Pune's monitored corridors.</p>
            </div>
            <div class="system-status"><span class="status-dot"></span>{area_count} areas online</div>
        </div>
        <div class="context-strip">
            <div class="context-item"><span class="context-label">Scenario time</span><span class="context-value">{timestamp_label}</span></div>
            <div class="context-item"><span class="context-label">Weather</span><span class="context-value">{weather_label}</span></div>
            <div class="context-item"><span class="context-label">Rainfall</span><span class="context-value">{common_inputs['precipitation_mm']:.1f} mm</span></div>
            <div class="context-item"><span class="context-label">Incidents</span><span class="context-value">{common_inputs['road_incident_count']} active</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_intro(title: str, copy: str) -> None:
    st.markdown(f'<div class="section-heading">{html.escape(title)}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-copy">{html.escape(copy)}</div>', unsafe_allow_html=True)


def render_area_prediction(common_inputs: dict[str, Any]) -> None:
    render_section_intro(
        "Area prediction",
        "Test a traffic scenario for one monitored area using observed speed, volume, weather, and incident conditions.",
    )

    with st.form("area_prediction_form"):
        form_left, form_right = st.columns(2)
        with form_left:
            area_name = st.selectbox("Target area", options=common_inputs["area_names"])
            traffic_volume = st.slider(
                "Traffic volume (vehicles/hour)",
                min_value=150,
                max_value=2200,
                value=1050,
                help="Estimated vehicles passing through the monitored corridor in one hour.",
            )
        with form_right:
            avg_speed_kmph = st.slider(
                "Average speed (km/h)",
                min_value=8.0,
                max_value=70.0,
                value=26.0,
                help="Observed or assumed average corridor speed.",
            )
            st.caption("The model combines these values with the shared scenario controls in the sidebar.")

        submitted = st.form_submit_button("Run area prediction", type="primary", width="stretch")

    if submitted:
        try:
            with st.spinner("Calculating area conditions…"):
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
        st.error(f"Prediction unavailable: {st.session_state['area_error']} Check the inputs and try again.")

    result = st.session_state["area_result"]
    if result:
        level = str(result["predicted_congestion_level"])
        level_class = f"level-{level.lower()}"
        result_area = str(result["input_features"]["area_name"])
        st.markdown(
            f"""
            <div class="result-banner">
                <div><span>Prediction complete</span><strong>{html.escape(result_area)} scenario</strong></div>
                <div class="level-chip {level_class}">{html.escape(level)} congestion</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        metric_left, metric_mid, metric_right = st.columns(3)
        metric_left.metric("Travel time", f"{result['predicted_travel_time_minutes']:.1f} min")
        metric_mid.metric("Traffic volume", f"{result['input_features']['traffic_volume']:,}")
        metric_right.metric("Average speed", f"{result['input_features']['avg_speed_kmph']:.1f} km/h")
        st.caption(f"Prediction generated with model {result['model_version']}.")
    else:
        st.markdown(
            '<div class="empty-state"><strong>No prediction yet</strong>Set the area conditions above, then run the model to view results.</div>',
            unsafe_allow_html=True,
        )


def render_route_map(start_area: str, end_area: str) -> None:
    profile_lookup = {area["area_name"]: area for area in AREA_PROFILES}
    start_profile = profile_lookup[start_area]
    end_profile = profile_lookup[end_area]

    if not FOLIUM_AVAILABLE:
        st.info("Interactive maps are unavailable in this environment. Showing the built-in map instead.")
        st.map(
            pd.DataFrame(
                [
                    {"lat": start_profile["latitude"], "lon": start_profile["longitude"]},
                    {"lat": end_profile["latitude"], "lon": end_profile["longitude"]},
                ]
            )
        )
        return

    route_map = folium.Map(
        location=[
            (start_profile["latitude"] + end_profile["latitude"]) / 2,
            (start_profile["longitude"] + end_profile["longitude"]) / 2,
        ],
        zoom_start=11,
        tiles="OpenStreetMap",
    )
    folium.CircleMarker(
        [start_profile["latitude"], start_profile["longitude"]],
        radius=8,
        tooltip=f"Departure: {start_area}",
        color="#22d3ee",
        fill=True,
        fill_color="#22d3ee",
        fill_opacity=0.95,
    ).add_to(route_map)
    folium.CircleMarker(
        [end_profile["latitude"], end_profile["longitude"]],
        radius=8,
        tooltip=f"Arrival: {end_area}",
        color="#fbbf24",
        fill=True,
        fill_color="#fbbf24",
        fill_opacity=0.95,
    ).add_to(route_map)
    folium.PolyLine(
        [
            [start_profile["latitude"], start_profile["longitude"]],
            [end_profile["latitude"], end_profile["longitude"]],
        ],
        color="#67e8f9",
        weight=5,
        opacity=0.82,
    ).add_to(route_map)
    st_folium(route_map, use_container_width=True, height=420, returned_objects=[])


def render_route_analysis(common_inputs: dict[str, Any]) -> None:
    render_section_intro(
        "Route analysis",
        "Compare two monitored areas to estimate corridor travel time, delay, risk, and the best departure window.",
    )

    with st.form("route_analysis_form"):
        route_left, route_right = st.columns(2)
        with route_left:
            start_area = st.selectbox("Departure area", options=common_inputs["area_names"], index=0)
        with route_right:
            end_area = st.selectbox(
                "Arrival area",
                options=common_inputs["area_names"],
                index=min(3, len(common_inputs["area_names"]) - 1),
            )

        same_area = start_area == end_area
        route_submitted = st.form_submit_button(
            "Analyze route", type="primary", width="stretch", disabled=same_area
        )

    if same_area:
        st.warning("Choose two different areas to analyze a route.")

    if route_submitted:
        try:
            with st.spinner("Evaluating corridor conditions…"):
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

    if st.session_state["route_error"]:
        st.error(f"Route analysis unavailable: {st.session_state['route_error']} Check the inputs and try again.")

    route_result = st.session_state["route_result"]
    if not route_result:
        st.markdown(
            '<div class="empty-state"><strong>No route analyzed</strong>Select a departure and arrival area to compare corridor conditions.</div>',
            unsafe_allow_html=True,
        )
        return

    route = route_result["route"]
    analysis = route_result["analysis"]
    prediction = route_result["prediction"]
    route_metrics = st.columns(4)
    route_metrics[0].metric("Distance", f"{route['estimated_distance_km']:.1f} km")
    route_metrics[1].metric("Estimated time", f"{prediction['predicted_travel_time_minutes']:.1f} min")
    route_metrics[2].metric("Expected delay", f"{analysis['expected_delay_minutes']:.1f} min")
    route_metrics[3].metric("Risk score", f"{analysis['route_risk_score']:.0f} / 100")

    departure = html.escape(str(analysis["recommended_departure"]["timestamp"]).replace("T", " "))
    st.markdown(
        f"""
        <div class="route-insight">
            <b>Priority hotspot:</b> {html.escape(str(route['hotspot_area']))}
            &nbsp;&nbsp;·&nbsp;&nbsp;
            <b>Recommended departure:</b> {departure}
        </div>
        """,
        unsafe_allow_html=True,
    )

    route_trend_df = pd.DataFrame(route_result["trend"])
    route_chart = px.line(
        route_trend_df,
        x="hour",
        y="estimated_travel_time_minutes",
        color="estimated_congestion_level",
        color_discrete_map=CONGESTION_COLORS,
        category_orders={"estimated_congestion_level": ["Low", "Medium", "High"]},
        markers=True,
        labels={
            "hour": "Hour of day",
            "estimated_travel_time_minutes": "Travel time (min)",
            "estimated_congestion_level": "Congestion",
        },
        title=f"24-hour route outlook · {route['start_area']} to {route['end_area']}",
    )
    route_chart.update_traces(line=dict(width=3), marker=dict(size=7))
    st.plotly_chart(apply_chart_styling(route_chart), width="stretch")
    render_route_map(route["start_area"], route["end_area"])


def render_area_map(df: pd.DataFrame) -> None:
    map_df = (
        df.groupby(["area_name", "latitude", "longitude"], as_index=False)["traffic_volume"]
        .mean()
        .rename(columns={"traffic_volume": "average_traffic_volume"})
    )

    if not FOLIUM_AVAILABLE:
        st.info("Interactive maps are unavailable in this environment. Showing the built-in map instead.")
        st.map(map_df.rename(columns={"latitude": "lat", "longitude": "lon"}))
        return

    map_center = [map_df["latitude"].mean(), map_df["longitude"].mean()]
    folium_map = folium.Map(location=map_center, zoom_start=11, tiles="OpenStreetMap")
    for _, row in map_df.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=max(row["average_traffic_volume"] / 180, 6),
            tooltip=f"{row['area_name']} · {row['average_traffic_volume']:.0f} vehicles/hour",
            color="#22d3ee",
            fill=True,
            fill_color="#22d3ee",
            fill_opacity=0.48,
        ).add_to(folium_map)
    st_folium(folium_map, use_container_width=True, height=430, returned_objects=[])


def render_analytics(df: pd.DataFrame, forecast_df: pd.DataFrame) -> None:
    render_section_intro(
        "Network overview",
        "Explore hourly demand, congestion distribution, regional performance, and the next 24-hour forecast.",
    )

    peak_hour = int(df.groupby("hour")["traffic_volume"].mean().idxmax())
    summary_metrics = st.columns(4)
    summary_metrics[0].metric("Average travel time", f"{df['travel_time_minutes'].mean():.1f} min")
    summary_metrics[1].metric("Average speed", f"{df['avg_speed_kmph'].mean():.1f} km/h")
    summary_metrics[2].metric("Peak demand hour", f"{peak_hour:02d}:00")
    summary_metrics[3].metric("Processed records", f"{len(df):,}")

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
            labels={"hour": "Hour of day", "average_travel_time": "Travel time (min)"},
            title="Hourly travel-time baseline",
        )
        fig_hourly.update_traces(line=dict(width=3, color="#22d3ee"), marker=dict(size=7, color="#67e8f9"))
        st.plotly_chart(apply_chart_styling(fig_hourly), width="stretch")

    with top_right:
        congestion_counts = (
            df["congestion_level"].value_counts().rename_axis("congestion_level").reset_index(name="count")
        )
        fig_distribution = px.bar(
            congestion_counts,
            x="count",
            y="congestion_level",
            orientation="h",
            color="congestion_level",
            color_discrete_map=CONGESTION_COLORS,
            category_orders={"congestion_level": ["High", "Medium", "Low"]},
            labels={"count": "Records", "congestion_level": "Congestion"},
            title="Congestion distribution",
        )
        fig_distribution.update_layout(showlegend=False)
        fig_distribution.update_traces(marker_line_width=0, hovertemplate="%{y}: %{x:,} records<extra></extra>")
        st.plotly_chart(apply_chart_styling(fig_distribution, hovermode=False), width="stretch")

    middle_left, middle_right = st.columns(2)
    with middle_left:
        fig_forecast = px.line(
            forecast_df,
            x="timestamp",
            y="forecast_travel_time_minutes",
            color="forecast_congestion_level",
            color_discrete_map=CONGESTION_COLORS,
            category_orders={"forecast_congestion_level": ["Low", "Medium", "High"]},
            labels={
                "timestamp": "Forecast time",
                "forecast_travel_time_minutes": "Travel time (min)",
                "forecast_congestion_level": "Congestion",
            },
            title="24-hour congestion forecast",
        )
        fig_forecast.update_traces(line=dict(width=3))
        st.plotly_chart(apply_chart_styling(fig_forecast), width="stretch")

    with middle_right:
        area_summary = df.groupby("area_name", as_index=False).agg(
            average_speed=("avg_speed_kmph", "mean"),
            average_volume=("traffic_volume", "mean"),
            average_travel_time=("travel_time_minutes", "mean"),
        )
        area_summary = area_summary.sort_values("average_travel_time", ascending=True)
        fig_area = px.bar(
            area_summary,
            x="average_travel_time",
            y="area_name",
            orientation="h",
            color="average_volume",
            color_continuous_scale=["#164e63", "#22d3ee"],
            labels={
                "area_name": "Area",
                "average_travel_time": "Travel time (min)",
                "average_volume": "Traffic volume",
            },
            title="Travel time by area",
        )
        fig_area.update_traces(marker_line_width=0)
        st.plotly_chart(apply_chart_styling(fig_area, hovermode=False), width="stretch")

    render_section_intro("Monitored area map", "Marker size reflects average traffic volume across the dataset.")
    render_area_map(df)

    with st.expander("Inspect processed records"):
        st.caption(f"Showing the first 25 of {len(df):,} processed observations.")
        st.dataframe(df.head(25), width="stretch", hide_index=True)


def main() -> None:
    st.set_page_config(page_title="Traffic Intel", layout="wide", initial_sidebar_state="expanded")
    inject_custom_css()
    init_state()

    try:
        with st.spinner("Loading network snapshot…"):
            df, forecast_df = load_dashboard_data()
    except Exception as exc:  # pragma: no cover - UI safeguard
        st.error(f"Dashboard data could not be loaded: {exc} Run the bootstrap process, then refresh this page.")
        st.stop()

    area_names = [area["area_name"] for area in AREA_PROFILES]
    common_inputs = build_common_inputs(area_names)
    render_page_header(df, common_inputs)

    prediction_tab, route_tab, analytics_tab = st.tabs(
        ["Area prediction", "Route analysis", "Network overview"]
    )

    with prediction_tab:
        render_area_prediction(common_inputs)

    with route_tab:
        render_route_analysis(common_inputs)

    with analytics_tab:
        render_analytics(df, forecast_df)


if __name__ == "__main__":
    main()
