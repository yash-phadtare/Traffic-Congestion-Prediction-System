import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, StratifiedKFold
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

from traffic_system.config import (
    settings, NUMERIC_FEATURES, CATEGORICAL_FEATURES, FEATURE_COLUMNS, ensure_directories,
    FORECAST_TRAVEL_LOW_MAX, FORECAST_TRAVEL_MED_MAX,
)
from traffic_system.utils import save_json, load_json, save_joblib, load_joblib
from traffic_system.data import preprocess_and_save, generate_visualizations

def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def _build_regression_models() -> dict[str, Pipeline]:
    models: dict[str, Any] = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=180, max_depth=16, min_samples_split=6, random_state=settings.random_state, n_jobs=1,
        ),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoostRegressor"] = XGBRegressor(
            n_estimators=220, max_depth=6, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9,
            random_state=settings.random_state, objective="reg:squarederror",
        )
    else:
        models["GradientBoostingRegressor"] = GradientBoostingRegressor(random_state=settings.random_state)

    return {name: Pipeline([("preprocessor", build_preprocessor()), ("model", model)]) for name, model in models.items()}


def _build_classification_models() -> dict[str, Pipeline]:
    models: dict[str, Any] = {
        "LogisticRegression": LogisticRegression(max_iter=400),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=220, max_depth=14, min_samples_split=6, random_state=settings.random_state, n_jobs=1,
        ),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoostClassifier"] = XGBClassifier(
            n_estimators=220, max_depth=6, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9,
            random_state=settings.random_state, objective="multi:softprob", eval_metric="mlogloss", num_class=3,
        )
    else:
        models["GradientBoostingClassifier"] = GradientBoostingClassifier(random_state=settings.random_state)

    return {name: Pipeline([("preprocessor", build_preprocessor()), ("model", model)]) for name, model in models.items()}


def _rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    try:
        from sklearn.metrics import root_mean_squared_error
        return float(root_mean_squared_error(y_true, y_pred))
    except ImportError:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _evaluate_regressors(
    models: dict[str, Pipeline], x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series
) -> tuple[dict[str, dict[str, float]], str]:
    metrics: dict[str, dict[str, float]] = {}
    for name, pipeline in models.items():
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        metrics[name] = {
            "rmse": _rmse(y_test, predictions),
            "mae": float(mean_absolute_error(y_test, predictions)),
            "r2": float(r2_score(y_test, predictions)),
        }
    return metrics, min(metrics, key=lambda name: metrics[name]["rmse"])


def _evaluate_classifiers(
    models: dict[str, Pipeline], x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series
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
    pipeline: Pipeline, model_name: str, x_train: pd.DataFrame, y_train: pd.Series, scoring: str,
    is_classification: bool = False,
) -> Pipeline:
    search_space = _get_search_space(model_name)
    if not search_space:
        pipeline.fit(x_train, y_train)
        return pipeline

    # Limit search cost while preserving the ordering assumptions of each CV
    # strategy. The frame index is chronological after train_and_evaluate sorts
    # the source data.
    sampled_rows = min(len(x_train), 8000)
    if is_classification:
        sampled_index = (
            x_train.assign(_y=y_train.values)
            .groupby("_y", group_keys=False)
            .apply(lambda g: g.sample(n=max(1, int(sampled_rows * len(g) / len(x_train))), random_state=settings.random_state),
                   include_groups=False)
            .index
        )
        # Fallback if rounding produced too few/many rows.
        if len(sampled_index) < 10 or len(sampled_index) > len(x_train):
            sampled_index = x_train.sample(n=sampled_rows, random_state=settings.random_state).index
    else:
        # Use one contiguous, ordered window. A random sample sorted by hour/day
        # is not chronological and leaks later observations into earlier folds.
        sampled_index = x_train.index[-sampled_rows:]
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=settings.random_state) if is_classification else TimeSeriesSplit(n_splits=3)
    search = RandomizedSearchCV(
        estimator=pipeline, param_distributions=search_space, n_iter=4, cv=cv,
        scoring=scoring, random_state=settings.random_state, n_jobs=1,
    )
    search.fit(x_train.loc[sampled_index], y_train.loc[sampled_index])

    # RandomizedSearchCV refits only on the sample passed above. Fit the selected
    # hyperparameters on every available training row before persisting it.
    best_pipeline = search.best_estimator_
    best_pipeline.fit(x_train, y_train)
    return best_pipeline


def _extract_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    raw_names = pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
    # Keep transformer provenance to avoid collisions
    # (e.g. num__hour vs cat__hour would both become "hour").
    feature_names = [f.replace("num__", "num:").replace("cat__", "cat:") for f in raw_names]

    if hasattr(model, "feature_importances_"):
        scores = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        scores = np.abs(coef) if coef.ndim == 1 else np.mean(np.abs(coef), axis=0)
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

    # Chronological split: time-ordered data must not leak future hours into train.
    df = df.sort_values("timestamp").reset_index(drop=True)
    x_data = df[FEATURE_COLUMNS]
    y_regression = df["travel_time_minutes"]
    y_classification = df["congestion_level"]

    split_idx = int(len(df) * 0.8)
    x_train, x_test = x_data.iloc[:split_idx], x_data.iloc[split_idx:]
    y_train_reg, y_test_reg = y_regression.iloc[:split_idx], y_regression.iloc[split_idx:]
    y_train_cls, y_test_cls = y_classification.iloc[:split_idx], y_classification.iloc[split_idx:]

    # Choose the model family on a validation window inside the training period.
    # The final test window remains untouched until the selected models are ready.
    validation_idx = int(len(x_train) * 0.8)
    x_model_train, x_validation = x_train.iloc[:validation_idx], x_train.iloc[validation_idx:]
    y_model_train_reg, y_validation_reg = y_train_reg.iloc[:validation_idx], y_train_reg.iloc[validation_idx:]
    y_model_train_cls, y_validation_cls = y_train_cls.iloc[:validation_idx], y_train_cls.iloc[validation_idx:]

    label_encoder = LabelEncoder()
    y_train_cls_encoded = pd.Series(label_encoder.fit_transform(y_train_cls), index=y_train_cls.index)
    y_test_cls_encoded = pd.Series(label_encoder.transform(y_test_cls), index=y_test_cls.index)
    y_model_train_cls_encoded = y_train_cls_encoded.loc[y_model_train_cls.index]
    y_validation_cls_encoded = y_train_cls_encoded.loc[y_validation_cls.index]

    regression_models = _build_regression_models()
    regression_metrics, best_regressor_name = _evaluate_regressors(
        regression_models, x_model_train, x_validation, y_model_train_reg, y_validation_reg,
    )
    regressor = _tune_pipeline(
        regression_models[best_regressor_name], best_regressor_name, x_train, y_train_reg, "neg_root_mean_squared_error",
        is_classification=False,
    )
    regression_predictions = regressor.predict(x_test)
    selected_regressor_name = (
        f"{best_regressor_name}_tuned" if _get_search_space(best_regressor_name) else best_regressor_name
    )
    regression_test_metrics = {
        "rmse": _rmse(y_test_reg, regression_predictions),
        "mae": float(mean_absolute_error(y_test_reg, regression_predictions)),
        "r2": float(r2_score(y_test_reg, regression_predictions)),
    }

    classification_models = _build_classification_models()
    classification_metrics, best_classifier_name = _evaluate_classifiers(
        classification_models,
        x_model_train,
        x_validation,
        y_model_train_cls_encoded,
        y_validation_cls_encoded,
    )
    classifier = _tune_pipeline(
        classification_models[best_classifier_name], best_classifier_name, x_train, y_train_cls_encoded, "f1_weighted",
        is_classification=True,
    )
    class_predictions = classifier.predict(x_test)
    selected_classifier_name = (
        f"{best_classifier_name}_tuned" if _get_search_space(best_classifier_name) else best_classifier_name
    )
    classification_test_metrics = {
        "accuracy": float(accuracy_score(y_test_cls_encoded, class_predictions)),
        "f1_score": float(f1_score(y_test_cls_encoded, class_predictions, average="weighted")),
    }

    metrics = {
        "dataset": {"rows": int(len(df)), "features": len(FEATURE_COLUMNS)},
        "split": {
            "strategy": "chronological_train_validation_test",
            "model_selection_rows": int(len(x_model_train)),
            "validation_rows": int(len(x_validation)),
            "training_rows": int(len(x_train)),
            "test_rows": int(len(x_test)),
        },
        "regression_metrics": regression_metrics,
        "classification_metrics": classification_metrics,
        "final_test_metrics": {
            "regression": {"model": selected_regressor_name, **regression_test_metrics},
            "classification": {"model": selected_classifier_name, **classification_test_metrics},
        },
        "best_regressor": selected_regressor_name,
        "best_classifier": selected_classifier_name,
    }
    feature_importance_df = _extract_feature_importance(regressor)

    if save_artifacts:
        import sklearn
        from datetime import datetime, timezone
        save_joblib(
            {
                "regressor": regressor,
                "classifier": classifier,
                "label_encoder": label_encoder,
                "model_version": "v1.2",
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "sklearn_version": sklearn.__version__,
                "feature_columns": list(FEATURE_COLUMNS),
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
        bins=[-float("inf"), FORECAST_TRAVEL_LOW_MAX, FORECAST_TRAVEL_MED_MAX, float("inf")],
        labels=["Low", "Medium", "High"],
    ).astype(str)
    forecast_df.to_csv(output_path, index=False)
    return forecast_df


def get_forecast(force_refresh: bool = False) -> pd.DataFrame:
    if force_refresh or not settings.forecast_path.exists():
        return forecast_congestion_trend()
    return pd.read_csv(settings.forecast_path, parse_dates=["timestamp"])


def ensure_model_bundle(force_refresh: bool = False) -> dict[str, Any]:
    if force_refresh or not settings.model_bundle_path.exists():
        train_and_evaluate(save_artifacts=True)
    bundle = load_joblib(settings.model_bundle_path)
    for model_name in ["regressor", "classifier"]:
        pipeline = bundle.get(model_name)
        if pipeline is None:
            continue
        estimator = pipeline.named_steps.get("model")
        if estimator is not None and hasattr(estimator, "n_jobs"):
            estimator.n_jobs = 1
    return bundle


def load_metrics() -> dict[str, Any]:
    if not settings.metrics_path.exists():
        train_and_evaluate(save_artifacts=True)
    return load_json(settings.metrics_path)
