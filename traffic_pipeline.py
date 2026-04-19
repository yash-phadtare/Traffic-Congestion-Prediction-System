from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from statsmodels.tsa.arima.model import ARIMA

try:
    from xgboost import XGBClassifier, XGBRegressor

    XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover
    XGBClassifier = None
    XGBRegressor = None
    XGBOOST_AVAILABLE = False


@dataclass(frozen=True)
class Settings:
    base_dir: Path = Path(__file__).resolve().parent
    raw_data_path: Path = base_dir / "data" / "raw" / "traffic_data.csv"
    processed_data_path: Path = base_dir / "data" / "processed" / "traffic_features.csv"
    model_bundle_path: Path = base_dir / "artifacts" / "models" / "model_bundle.joblib"
    metrics_path: Path = base_dir / "artifacts" / "models" / "training_metrics.json"
    feature_importance_path: Path = base_dir / "artifacts" / "models" / "feature_importance.csv"
    forecast_path: Path = base_dir / "artifacts" / "forecasts" / "future_congestion_forecast.csv"
    plot_dir: Path = base_dir / "artifacts" / "plots"
    spark_output_dir: Path = base_dir / "artifacts" / "spark"
    random_state: int = 42


settings = Settings()

AREA_PROFILES = [
    {"area_name": "Shivaji Nagar", "latitude": 18.5308, "longitude": 73.8475, "route_distance_km": 6.4, "base_traffic_volume": 980, "base_speed_kmph": 34},
    {"area_name": "Hinjewadi", "latitude": 18.5910, "longitude": 73.7389, "route_distance_km": 14.0, "base_traffic_volume": 870, "base_speed_kmph": 38},
    {"area_name": "Kothrud", "latitude": 18.5074, "longitude": 73.8077, "route_distance_km": 8.2, "base_traffic_volume": 760, "base_speed_kmph": 37},
    {"area_name": "Kharadi", "latitude": 18.5513, "longitude": 73.9497, "route_distance_km": 12.1, "base_traffic_volume": 920, "base_speed_kmph": 36},
    {"area_name": "Baner", "latitude": 18.5590, "longitude": 73.7868, "route_distance_km": 10.2, "base_traffic_volume": 730, "base_speed_kmph": 35},
    {"area_name": "Hadapsar", "latitude": 18.5089, "longitude": 73.9259, "route_distance_km": 11.6, "base_traffic_volume": 860, "base_speed_kmph": 33},
    {"area_name": "Wakad", "latitude": 18.5975, "longitude": 73.7637, "route_distance_km": 13.4, "base_traffic_volume": 810, "base_speed_kmph": 34},
    {"area_name": "Viman Nagar", "latitude": 18.5679, "longitude": 73.9143, "route_distance_km": 9.3, "base_traffic_volume": 780, "base_speed_kmph": 36},
]
AREA_PROFILE_MAP = {item["area_name"]: item for item in AREA_PROFILES}
RUSH_HOURS = {7, 8, 9, 10, 17, 18, 19, 20}
HOLIDAY_MONTH_DAY = {(1, 1), (1, 26), (8, 15), (10, 2), (12, 25)}

FEATURE_COLUMNS = [
    "area_name", "latitude", "longitude", "temperature_c", "humidity_pct", "precipitation_mm",
    "weather_condition", "traffic_volume", "avg_speed_kmph", "road_incident_count",
    "route_distance_km", "hour", "day_of_week", "month", "is_weekend", "is_rush_hour",
    "is_holiday", "time_band", "congestion_index",
]
NUMERIC_FEATURES = [
    "latitude", "longitude", "temperature_c", "humidity_pct", "precipitation_mm",
    "traffic_volume", "avg_speed_kmph", "road_incident_count", "route_distance_km",
    "hour", "day_of_week", "month", "is_weekend", "is_rush_hour", "is_holiday", "congestion_index",
]
CATEGORICAL_FEATURES = ["area_name", "weather_condition", "time_band"]
NUMERIC_COLUMNS = [
    "latitude", "longitude", "temperature_c", "humidity_pct", "precipitation_mm",
    "traffic_volume", "avg_speed_kmph", "road_incident_count", "route_distance_km",
]


def ensure_directories(paths: list[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_joblib(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)


def load_joblib(path: Path) -> Any:
    return joblib.load(path)


def _assign_weather_condition(precipitation: pd.Series, humidity: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [precipitation >= 16, precipitation >= 4, humidity >= 78],
            ["Storm", "Rain", "Cloudy"],
            default="Clear",
        )
    )


def generate_synthetic_traffic_data(
    output_path: Path | None = None,
    num_days: int = 120,
    seed: int = settings.random_state,
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


def generate_visualizations(df: pd.DataFrame, output_dir: Path | None = None) -> None:
    output_dir = output_dir or settings.plot_dir
    ensure_directories([output_dir])
    sns.set_theme(style="whitegrid")

    numeric_columns = [
        "temperature_c",
        "humidity_pct",
        "precipitation_mm",
        "traffic_volume",
        "avg_speed_kmph",
        "road_incident_count",
        "travel_time_minutes",
        "congestion_index",
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


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def _build_regression_models() -> dict[str, Pipeline]:
    models: dict[str, Any] = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=180,
            max_depth=16,
            min_samples_split=6,
            random_state=settings.random_state,
            n_jobs=-1,
        ),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoostRegressor"] = XGBRegressor(
            n_estimators=220,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=settings.random_state,
            objective="reg:squarederror",
        )
    else:
        models["GradientBoostingRegressor"] = GradientBoostingRegressor(random_state=settings.random_state)

    return {name: Pipeline([("preprocessor", build_preprocessor()), ("model", model)]) for name, model in models.items()}


def _build_classification_models() -> dict[str, Pipeline]:
    models: dict[str, Any] = {
        "LogisticRegression": LogisticRegression(max_iter=400),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=220,
            max_depth=14,
            min_samples_split=6,
            random_state=settings.random_state,
            n_jobs=-1,
        ),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoostClassifier"] = XGBClassifier(
            n_estimators=220,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=settings.random_state,
            objective="multi:softprob",
            eval_metric="mlogloss",
            num_class=3,
        )
    else:
        models["GradientBoostingClassifier"] = GradientBoostingClassifier(random_state=settings.random_state)

    return {name: Pipeline([("preprocessor", build_preprocessor()), ("model", model)]) for name, model in models.items()}


def _evaluate_regressors(
    models: dict[str, Pipeline],
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[dict[str, dict[str, float]], str]:
    metrics: dict[str, dict[str, float]] = {}
    for name, pipeline in models.items():
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        metrics[name] = {
            "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
            "mae": float(mean_absolute_error(y_test, predictions)),
        }
    return metrics, min(metrics, key=lambda name: metrics[name]["rmse"])


def _evaluate_classifiers(
    models: dict[str, Pipeline],
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[dict[str, dict[str, float]], str]:
    metrics: dict[str, dict[str, float]] = {}
    for name, pipeline in models.items():
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        metrics[name] = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "f1_score": float(f1_score(y_test, predictions, average="weighted")),
        }
    return metrics, max(metrics, key=lambda name: metrics[name]["f1_score"])


def _get_search_space(model_name: str) -> dict[str, list[Any]]:
    if "Forest" in model_name:
        return {
            "model__n_estimators": [120, 180, 260],
            "model__max_depth": [10, 14, 18],
            "model__min_samples_split": [2, 4, 6],
        }
    if "Boost" in model_name:
        return {
            "model__n_estimators": [120, 180, 240],
            "model__learning_rate": [0.03, 0.06, 0.1],
            "model__max_depth": [3, 4, 6],
        }
    return {}


def _tune_pipeline(
    pipeline: Pipeline,
    model_name: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    scoring: str,
) -> Pipeline:
    search_space = _get_search_space(model_name)
    if not search_space:
        pipeline.fit(x_train, y_train)
        return pipeline

    sampled_rows = min(len(x_train), 8000)
    sampled_index = x_train.sample(n=sampled_rows, random_state=settings.random_state).index
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=search_space,
        n_iter=4,
        cv=3,
        scoring=scoring,
        random_state=settings.random_state,
        n_jobs=-1,
    )
    search.fit(x_train.loc[sampled_index], y_train.loc[sampled_index])
    return search.best_estimator_


def _extract_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
    if hasattr(model, "feature_importances_"):
        scores = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        scores = np.abs(coef if coef.ndim == 1 else coef.mean(axis=0))
    else:
        scores = np.zeros(len(feature_names))
    return pd.DataFrame({"feature": feature_names, "importance": scores}).sort_values("importance", ascending=False)


def _save_feature_importance_plot(importance_df: pd.DataFrame) -> None:
    top_features = importance_df.head(12).sort_values("importance")
    plt.figure(figsize=(10, 6))
    plt.barh(top_features["feature"], top_features["importance"], color="#D62828")
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.savefig(settings.plot_dir / "feature_importance.png", dpi=200)
    plt.close()


def train_and_evaluate(save_artifacts: bool = True) -> dict[str, Any]:
    df = preprocess_and_save()
    generate_visualizations(df)

    x_data = df[FEATURE_COLUMNS]
    y_regression = df["travel_time_minutes"]
    y_classification = df["congestion_level"]

    x_train, x_test, y_train_reg, y_test_reg, y_train_cls, y_test_cls = train_test_split(
        x_data,
        y_regression,
        y_classification,
        test_size=0.2,
        random_state=settings.random_state,
        stratify=y_classification,
    )

    label_encoder = LabelEncoder()
    y_train_cls_encoded = pd.Series(label_encoder.fit_transform(y_train_cls), index=y_train_cls.index)
    y_test_cls_encoded = pd.Series(label_encoder.transform(y_test_cls), index=y_test_cls.index)

    regression_models = _build_regression_models()
    regression_metrics, best_regressor_name = _evaluate_regressors(regression_models, x_train, x_test, y_train_reg, y_test_reg)
    regressor = _tune_pipeline(
        regression_models[best_regressor_name],
        best_regressor_name,
        x_train,
        y_train_reg,
        "neg_root_mean_squared_error",
    )
    regressor.fit(x_train, y_train_reg)
    regression_predictions = regressor.predict(x_test)
    regression_metrics[f"{best_regressor_name}_tuned"] = {
        "rmse": float(np.sqrt(mean_squared_error(y_test_reg, regression_predictions))),
        "mae": float(mean_absolute_error(y_test_reg, regression_predictions)),
    }

    classification_models = _build_classification_models()
    classification_metrics, best_classifier_name = _evaluate_classifiers(
        classification_models, x_train, x_test, y_train_cls_encoded, y_test_cls_encoded
    )
    classifier = _tune_pipeline(
        classification_models[best_classifier_name],
        best_classifier_name,
        x_train,
        y_train_cls_encoded,
        "f1_weighted",
    )
    classifier.fit(x_train, y_train_cls_encoded)
    class_predictions = classifier.predict(x_test)
    classification_metrics[f"{best_classifier_name}_tuned"] = {
        "accuracy": float(accuracy_score(y_test_cls_encoded, class_predictions)),
        "f1_score": float(f1_score(y_test_cls_encoded, class_predictions, average="weighted")),
    }

    metrics = {
        "dataset": {"rows": int(len(df)), "features": len(FEATURE_COLUMNS)},
        "regression_metrics": regression_metrics,
        "classification_metrics": classification_metrics,
        "best_regressor": best_regressor_name,
        "best_classifier": best_classifier_name,
    }
    feature_importance_df = _extract_feature_importance(regressor)

    if save_artifacts:
        save_joblib(
            {
                "regressor": regressor,
                "classifier": classifier,
                "label_encoder": label_encoder,
            },
            settings.model_bundle_path,
        )
        save_json(metrics, settings.metrics_path)
        feature_importance_df.to_csv(settings.feature_importance_path, index=False)
        _save_feature_importance_plot(feature_importance_df)

    return metrics


def forecast_congestion_trend(horizon_hours: int = 24, output_path: Path | None = None) -> pd.DataFrame:
    output_path = output_path or settings.forecast_path
    ensure_directories([output_path.parent])

    df = preprocess_and_save()
    hourly_series = df.set_index("timestamp").resample("h")["travel_time_minutes"].mean().ffill()
    fitted = ARIMA(hourly_series, order=(2, 1, 2)).fit()
    forecast_values = fitted.forecast(steps=horizon_hours)

    forecast_df = pd.DataFrame(
        {
            "timestamp": pd.date_range(hourly_series.index.max() + pd.Timedelta(hours=1), periods=horizon_hours, freq="h"),
            "forecast_travel_time_minutes": forecast_values.values,
        }
    )
    forecast_df["forecast_congestion_level"] = pd.cut(
        forecast_df["forecast_travel_time_minutes"],
        bins=[-float("inf"), 18, 28, float("inf")],
        labels=["Low", "Medium", "High"],
    ).astype(str)
    forecast_df.to_csv(output_path, index=False)
    return forecast_df


def get_processed_data(force_refresh: bool = False) -> pd.DataFrame:
    if force_refresh or not settings.processed_data_path.exists():
        return preprocess_and_save()
    return pd.read_csv(settings.processed_data_path, parse_dates=["timestamp"])


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


def get_forecast(force_refresh: bool = False) -> pd.DataFrame:
    if force_refresh or not settings.forecast_path.exists():
        return forecast_congestion_trend()
    return pd.read_csv(settings.forecast_path, parse_dates=["timestamp"])


def get_realtime_simulation(sample_size: int = 25, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    latest_df = get_processed_data().sort_values("timestamp").tail(sample_size).copy()
    latest_df["traffic_volume"] = (latest_df["traffic_volume"] * rng.uniform(0.9, 1.1, len(latest_df))).round().astype(int)
    latest_df["avg_speed_kmph"] = (latest_df["avg_speed_kmph"] * rng.uniform(0.88, 1.08, len(latest_df))).round(2)
    latest_df["travel_time_minutes"] = (latest_df["travel_time_minutes"] * rng.uniform(0.92, 1.12, len(latest_df))).round(2)
    latest_df["timestamp"] = pd.Timestamp.utcnow().floor("min")
    return latest_df


def ensure_model_bundle(force_refresh: bool = False) -> dict[str, Any]:
    if force_refresh or not settings.model_bundle_path.exists():
        train_and_evaluate(save_artifacts=True)
    return load_joblib(settings.model_bundle_path)


def load_metrics() -> dict[str, Any]:
    if not settings.metrics_path.exists():
        train_and_evaluate(save_artifacts=True)
    return load_json(settings.metrics_path)


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
