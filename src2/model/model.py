"""Model training and evaluation."""

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)

from .. import config

LOGGER = logging.getLogger(__name__)

N_BOOTSTRAP = 50
RANDOM_STATE = 42


def remove_outliers_iqr(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Remove outliers based on IQR."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


def _build_models() -> dict:
    return {
        "LinearRegression":      LinearRegression(),
        "Ridge":                 Ridge(),
        "Lasso":                 Lasso(),
        "ElasticNet":            ElasticNet(),
        "HuberRegressor":        HuberRegressor(max_iter=200),
        "RandomForest":          RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE),
        "ExtraTrees":            ExtraTreesRegressor(n_estimators=200, random_state=RANDOM_STATE),
        "GradientBoosting":      GradientBoostingRegressor(random_state=RANDOM_STATE),
        "HistGradientBoosting":  HistGradientBoostingRegressor(random_state=RANDOM_STATE),
    }


def _feature_columns(df: pd.DataFrame) -> list:
    feature_columns = config.FEATURE_COLUMNS.copy()
    for col in config.CATEGORICAL_COLUMNS:
        feature_columns.extend([c for c in df.columns if c.startswith(f"{col}_")])
    return feature_columns


def _prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list]:
    feature_columns = _feature_columns(df)
    df = df.dropna(subset=[config.TARGET_COLUMN])
    df = remove_outliers_iqr(df, config.TARGET_COLUMN)
    df = df.fillna(0)
    return df[feature_columns], df[config.TARGET_COLUMN], feature_columns


def _evaluate_models(X_train, X_test, y_train, y_test) -> tuple[dict, str]:
    """Fit every model, return results dict and name of best by test R²."""
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    best_r2 = -float("inf")
    best_name = None

    for name, model in _build_models().items():
        cv_scores = cross_validate(model, X_train, y_train, cv=kf, scoring="r2")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        results[name] = {
            "model": model,
            "metrics": {
                "cv_r2_mean": float(np.mean(cv_scores["test_score"])),
                "rmse":       float(np.sqrt(mean_squared_error(y_test, y_pred))),
                "r2":         float(r2),
                "mae":        float(mean_absolute_error(y_test, y_pred)),
                "mape":       float(mean_absolute_percentage_error(y_test, y_pred)),
                "median_ae":  float(median_absolute_error(y_test, y_pred)),
            },
            "y_pred": y_pred,
        }
        if r2 > best_r2:
            best_r2 = r2
            best_name = name

    return results, best_name


def _bootstrap_prediction_std(model_factory, X_train, y_train, X_test, n_bootstrap: int = N_BOOTSTRAP):
    """Refit on bootstrap samples and return per-test-point prediction mean and std."""
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(X_train)
    predictions = np.zeros((n_bootstrap, len(X_test)))
    X_train_values = X_train.values if hasattr(X_train, "values") else X_train
    y_train_values = y_train.values if hasattr(y_train, "values") else y_train
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        m = model_factory()
        m.fit(X_train_values[idx], y_train_values[idx])
        predictions[i] = m.predict(X_test)
    return predictions.mean(axis=0), predictions.std(axis=0)


def _factory_for(best_name: str):
    """Return a no-arg factory that rebuilds the chosen model with the same settings."""
    return lambda: _build_models()[best_name]


def train_model(df: pd.DataFrame):
    """Train and evaluate models. Returns (best_model, comparison, feature_importance, best_y_data, variants)."""
    LOGGER.info("Training models on full dataset (%d rows)...", len(df))
    X, y, feature_columns = _prepare_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    results, best_name = _evaluate_models(X_train, X_test, y_train, y_test)
    best_model = results[best_name]["model"]
    LOGGER.info("Best model on full dataset: %s with R²=%.4f",
                best_name, results[best_name]["metrics"]["r2"])

    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    elif hasattr(best_model, "coef_"):
        importances = np.atleast_1d(best_model.coef_)
    else:
        importances = np.zeros(len(feature_columns))
    feature_importance = pd.DataFrame({"feature": feature_columns, "importance": np.abs(importances)})

    pred_mean, pred_std = _bootstrap_prediction_std(_factory_for(best_name), X_train, y_train, X_test)

    comparison_metrics = {name: res["metrics"] for name, res in results.items()}
    best_y_data = {
        "y_test":   y_test,
        "y_pred":   results[best_name]["y_pred"],
        "pred_std": pred_std,
        "model":    best_name,
    }

    variants = train_data_age_variants(df)
    return best_model, comparison_metrics, feature_importance, best_y_data, variants


def train_data_age_variants(df: pd.DataFrame) -> dict:
    """Compare 'full' vs 'recent (≤MAX_DATA_AGE_YEARS)' training sets for the same models."""
    cutoff_year = config.CURRENT_YEAR - config.MAX_DATA_AGE_YEARS
    recent = df[df["commissioning_year"] >= cutoff_year].copy()
    LOGGER.info("Variant 'recent' (year >= %d): %d rows", cutoff_year, len(recent))

    out = {"cutoff_year": cutoff_year, "variants": {}}

    for label, subset in (("full", df), ("recent", recent)):
        if len(subset) < 20:
            LOGGER.warning("Variant '%s' has %d rows, skipping", label, len(subset))
            continue
        X, y, _ = _prepare_xy(subset)
        if len(X) < 10:
            LOGGER.warning("Variant '%s' has too few usable rows after cleaning", label)
            continue
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
        results, best_name = _evaluate_models(X_train, X_test, y_train, y_test)
        out["variants"][label] = {
            "n_rows":          len(subset),
            "n_train":         len(X_train),
            "n_test":          len(X_test),
            "best_model":      best_name,
            "metrics_by_model": {name: res["metrics"] for name, res in results.items()},
        }

    return out
