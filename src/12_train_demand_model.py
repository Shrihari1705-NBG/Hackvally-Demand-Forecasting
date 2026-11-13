"""Train and evaluate the pooled one-hour-ahead demand forecasting model."""
from __future__ import annotations

import importlib.util
import json
import os
import platform
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.json"


def _load_features_module():
    path = ROOT / "src/10_build_forecasting_features.py"
    spec = importlib.util.spec_from_file_location("forecast_features", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def regression_metrics(actual: pd.Series, predicted: np.ndarray) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    y = np.asarray(actual, dtype=float)
    p = np.maximum(0.0, np.asarray(predicted, dtype=float))
    denominator = float(np.abs(y).sum())
    poisson_terms = p.copy()
    positive = y > 0
    poisson_terms[positive] = y[positive] * np.log(y[positive] / np.maximum(p[positive], 1e-15)) - (y[positive] - p[positive])
    poisson = float(np.mean(2 * poisson_terms))
    return {"MAE": float(mean_absolute_error(y, p)), "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "WAPE": float(np.abs(y - p).sum() / denominator) if denominator else None,
            "Poisson_deviance": poisson, "total_actual": float(y.sum()), "total_predicted": float(p.sum()),
            "mean_actual": float(y.mean()), "mean_predicted": float(p.mean()), "n": int(len(y))}


def grouped_metrics(frame: pd.DataFrame, actual: str, prediction: str, group: str) -> pd.DataFrame:
    rows = []
    for key, group_frame in frame.groupby(group, sort=True):
        metrics = regression_metrics(group_frame[actual], group_frame[prediction].to_numpy())
        metrics[group] = key
        rows.append(metrics)
    return pd.DataFrame(rows)


def train_demand_model(config: dict | None = None) -> dict:
    config = config or json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    forecasting = config.get("forecasting", {})
    if forecasting.get("enabled") is False:
        raise ValueError("forecasting.enabled is false")
    # Keep training deterministic and avoid unrestricted worker creation on
    # constrained Windows environments.
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    try:
        import sklearn
        import joblib
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder
    except ImportError as exc:
        raise RuntimeError("Phase 7A requires scikit-learn and joblib. Install requirements.txt before training.") from exc
    features_module = _load_features_module()
    features, feature_metadata = features_module.build_forecasting_features(config, strict=True)
    target = forecasting.get("target_column", "demand_count")
    feature_columns = feature_metadata["feature_columns"]
    categorical = feature_metadata["categorical_features"]
    numeric = [column for column in feature_columns if column not in categorical]
    x_train = features.loc[features.split == "train", feature_columns]
    x_validation = features.loc[features.split == "validation", feature_columns]
    x_test = features.loc[features.split == "test", feature_columns]
    y_train = features.loc[features.split == "train", target]
    y_validation = features.loc[features.split == "validation", target]
    y_test = features.loc[features.split == "test", target]
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # sklearn versions before sparse_output
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    preprocessor = ColumnTransformer([("categorical", encoder, categorical), ("numeric", "passthrough", numeric)], remainder="drop")
    model = HistGradientBoostingRegressor(loss="poisson", random_state=int(forecasting.get("random_seed", 42)),
        max_iter=int(forecasting.get("max_iter", 160)), learning_rate=float(forecasting.get("learning_rate", .08)),
        max_leaf_nodes=int(forecasting.get("max_leaf_nodes", 31)), l2_regularization=float(forecasting.get("l2_regularization", .1)))
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(x_train, y_train)
    validation_pred = np.maximum(0, pipeline.predict(x_validation))
    test_pred = np.maximum(0, pipeline.predict(x_test))
    test_frame = features.loc[features.split == "test"].copy()
    test_frame["ml_prediction"] = test_pred
    test_frame["baseline_prediction"] = np.maximum(0, test_frame["demand_lag_168"].to_numpy())
    metrics = {"validation_ml": regression_metrics(y_validation, validation_pred), "test_ml": regression_metrics(y_test, test_pred),
               "test_seasonal_naive_168h": regression_metrics(y_test, test_frame["baseline_prediction"]),
               "test_rows": int(len(test_frame)), "validation_rows": int(len(x_validation)), "training_rows": int(len(x_train))}
    region_metrics = pd.concat([grouped_metrics(test_frame.assign(model_prediction=test_frame.ml_prediction), target, "model_prediction", "region_id").assign(model="ml"),
                                grouped_metrics(test_frame.assign(model_prediction=test_frame.baseline_prediction), target, "model_prediction", "region_id").assign(model="seasonal_naive")])
    hourly_metrics = pd.concat([grouped_metrics(test_frame.assign(model_prediction=test_frame.ml_prediction), target, "model_prediction", "hour").assign(model="ml"),
                                grouped_metrics(test_frame.assign(model_prediction=test_frame.baseline_prediction), target, "model_prediction", "hour").assign(model="seasonal_naive")])
    metrics["test_ml_by_region"] = region_metrics[region_metrics.model == "ml"].set_index("region_id").to_dict(orient="index")
    metrics["test_baseline_by_region"] = region_metrics[region_metrics.model == "seasonal_naive"].set_index("region_id").to_dict(orient="index")
    metrics["test_ml_by_hour"] = hourly_metrics[hourly_metrics.model == "ml"].set_index("hour").to_dict(orient="index")
    metrics["test_baseline_by_hour"] = hourly_metrics[hourly_metrics.model == "seasonal_naive"].set_index("hour").to_dict(orient="index")
    test_frame["weekday_group"] = np.where(test_frame.day_of_week >= 5, "weekend", "weekday")
    metrics["test_ml_weekday_weekend"] = grouped_metrics(test_frame.assign(model_prediction=test_frame.ml_prediction), target, "model_prediction", "weekday_group").set_index("weekday_group").to_dict(orient="index")
    metrics["test_baseline_weekday_weekend"] = grouped_metrics(test_frame.assign(model_prediction=test_frame.baseline_prediction), target, "model_prediction", "weekday_group").set_index("weekday_group").to_dict(orient="index")
    output_dir = ROOT / "models/demand_forecasting"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = {"pipeline": pipeline, "feature_columns": feature_columns, "categorical_features": categorical,
                "numeric_features": numeric, "target_column": target, "model_version": "phase7a-v1"}
    joblib.dump(artifact, output_dir / "model.joblib")
    schema = {"feature_columns": feature_columns, "categorical_features": categorical, "numeric_features": numeric,
              "target_column": target, "required_input_columns": ["h3_index", "timestamp"] + feature_columns}
    (output_dir / "feature_schema.json").write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    metadata = {"model_type": "HistGradientBoostingRegressor(loss='poisson')", "model_version": "phase7a-v1",
                "training_timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "python_version": platform.python_version(), "scikit_learn_version": sklearn.__version__,
                "historical_start": feature_metadata["historical_start"], "historical_end": feature_metadata["historical_end"],
                "train_validation_test": feature_metadata["split_timestamps"], "rows_removed_insufficient_history": feature_metadata["rows_removed_insufficient_history"],
                "h3_resolution": 8, "cells": 94, "feature_metadata": feature_metadata, "random_seed": forecasting.get("random_seed", 42),
                "training_configuration": forecasting, "poi_context_status": "unavailable_or_optional",
                "synthetic_warning": "Metrics are on simulated demand data, not observed Belagavi ride demand."}
    (output_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
    (output_dir / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2, default=str) + "\n", encoding="utf-8")
    region_metrics.to_csv(output_dir / "region_metrics.csv", index=False)
    hourly_metrics.to_csv(output_dir / "hourly_metrics.csv", index=False)
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), dpi=160)
    test_frame.groupby("timestamp")[[target, "ml_prediction", "baseline_prediction"]].sum().plot(ax=axes[0, 0], title="Actual vs predictions")
    test_frame.groupby("region_id")[[target, "ml_prediction", "baseline_prediction"]].sum().plot.bar(ax=axes[0, 1], title="By region")
    (test_frame[target] - test_frame.ml_prediction).plot.hist(ax=axes[0, 2], bins=25, title="ML error distribution")
    axes[0, 3].scatter(test_frame[target], test_frame.ml_prediction, s=3, alpha=.25); axes[0, 3].set_title("Actual vs ML scatter")
    region_metrics[region_metrics.model == "ml"].set_index("region_id")["MAE"].plot.bar(ax=axes[1, 0], title="ML MAE by region")
    hourly_metrics[hourly_metrics.model == "ml"].set_index("hour")["MAE"].plot.bar(ax=axes[1, 1], title="ML MAE by hour")
    test_frame.ml_prediction.plot.hist(ax=axes[1, 2], bins=25, title="ML prediction distribution")
    pd.DataFrame({"ML": [metrics["test_ml"]["MAE"]], "Seasonal naive": [metrics["test_seasonal_naive_168h"]["MAE"]]}).plot.bar(ax=axes[1, 3], title="Test MAE comparison")
    fig.suptitle("SYNTHETIC DATA – EXPERIMENTAL FORECASTING RESULTS", fontsize=13); fig.tight_layout(); fig.savefig(output_dir / "diagnostics.png"); plt.close(fig)
    report = ["# Demand Forecasting Report", "", "**SYNTHETIC DATA – EXPERIMENTAL FORECASTING RESULTS**", "",
              f"- Model: `HistGradientBoostingRegressor(loss='poisson')`", f"- Training rows: `{len(x_train):,}`", f"- Validation rows: `{len(x_validation):,}`", f"- Test rows: `{len(x_test):,}`",
              f"- Rows removed for insufficient history: `{feature_metadata['rows_removed_insufficient_history']:,}`", "", "## Test metrics", "",
              f"- ML MAE: `{metrics['test_ml']['MAE']:.4f}`", f"- Seasonal-naive MAE: `{metrics['test_seasonal_naive_168h']['MAE']:.4f}`",
              f"- ML WAPE: `{metrics['test_ml']['WAPE']}`", f"- Seasonal-naive WAPE: `{metrics['test_seasonal_naive_168h']['WAPE']}`", "",
              "The model comparison is an experiment on simulated demand and does not establish performance on actual Belagavi rides.", "No random split, lag leakage, API, or frontend integration was used."]
    (output_dir / "forecasting_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return {"metrics": metrics, "metadata": metadata, "feature_metadata": feature_metadata}


if __name__ == "__main__":
    result = train_demand_model()
    print(json.dumps({"test_ml": result["metrics"]["test_ml"], "test_baseline": result["metrics"]["test_seasonal_naive_168h"]}, indent=2))
