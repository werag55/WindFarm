"""Linear regression model for LCOE prediction."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .. import config
from ..data_preparation.prep_sample import prepare_sample
from ..visualisations.model_dashboard import create_model_dashboard

LOGGER = logging.getLogger(__name__)


@dataclass
class LinearModel:
    """Represents a simple linear regression model."""

    intercept: float
    coefficients: np.ndarray
    feature_names: List[str]

    def predict_one(self, row: Dict[str, float]) -> float:
        """Predict a single value from a dictionary of feature values."""
        vector = np.array([float(row[name]) for name in self.feature_names], dtype=float)
        return float(self.intercept + vector @ self.coefficients)


def _fit_linear_regression(
    frame: pd.DataFrame, features: List[str], target: str
) -> Tuple[LinearModel, Dict[str, float]]:
    """Fit a linear regression model and calculate training metrics."""
    usable = frame[features + [target]].dropna().copy()
    if usable.empty:
        raise ValueError(f"No usable rows available to train model for {target}")

    X = usable[features].to_numpy(dtype=float)
    y = usable[target].to_numpy(dtype=float)
    X_design = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    intercept = float(beta[0])
    coefficients = beta[1:]
    model = LinearModel(
        intercept=intercept, coefficients=coefficients, feature_names=features
    )

    predictions = X_design @ beta
    residuals = y - predictions
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))
    r2_denom = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1.0 - np.sum(residuals**2) / r2_denom) if r2_denom else 0.0

    metrics = {
        "rows_used": int(len(usable)),
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }
    return model, metrics


def _train_test_split(
    frame: pd.DataFrame, test_fraction: float = 0.2, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into training and testing sets."""
    if len(frame) < 5:
        return frame.copy(), frame.copy()
    shuffled = frame.sample(frac=1.0, random_state=seed)
    cutoff = max(1, int(len(shuffled) * (1.0 - test_fraction)))
    return shuffled.iloc[:cutoff].copy(), shuffled.iloc[cutoff:].copy()


def _evaluate_model(
    model: LinearModel, frame: pd.DataFrame, target: str
) -> Dict[str, float]:
    """Evaluate the model on a test set."""
    usable = frame[model.feature_names + [target]].dropna().copy()
    if usable.empty:
        return {
            "rows_used": 0,
            "rmse": float("nan"),
            "mae": float("nan"),
            "r2": float("nan"),
        }
    X = usable[model.feature_names].to_numpy(dtype=float)
    y = usable[target].to_numpy(dtype=float)
    predictions = np.array(
        [model.intercept + row @ model.coefficients for row in X], dtype=float
    )
    residuals = y - predictions
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))
    r2_denom = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1.0 - np.sum(residuals**2) / r2_denom) if r2_denom else 0.0
    return {"rows_used": int(len(usable)), "rmse": rmse, "mae": mae, "r2": r2}


def predict_new_location(
    lcoe_model: LinearModel, sample: Dict[str, Any] | None = None
) -> Dict[str, float]:
    """Predict LCOE for a new location, fetching environmental data if needed."""
    sample_data = prepare_sample(sample)

    has_coords = not pd.isna(sample_data.get("LAT")) and not pd.isna(sample_data.get("LON"))
    if not has_coords:
        LOGGER.warning("No LAT/LON for new location, cannot fetch environmental data.")
        return {
            "mean_wind_speed_mps": float("nan"),
            "mean_wave_height_m": float("nan"),
            "predicted_lcoe_eur_per_mwh": float("nan"),
        }

    try:
        predicted_lcoe = lcoe_model.predict_one(sample_data)
        return {
            "mean_wind_speed_mps": float(sample_data["mean_wind_speed_mps"]),
            "mean_wave_height_m": float(sample_data["mean_wave_height_m"]),
            "predicted_lcoe_eur_per_mwh": float(predicted_lcoe),
        }
    except Exception as exc:
        LOGGER.warning(
            "Prediction failed for new location (%.4f, %.4f): %s. Cannot predict LCOE.",
            sample_data["LAT"],
            sample_data["LON"],
            exc,
        )
        return {
            "mean_wind_speed_mps": float("nan"),
            "mean_wave_height_m": float("nan"),
            "predicted_lcoe_eur_per_mwh": float("nan"),
        }


def train_and_evaluate(frame) -> None:
    """Train the LCOE model, evaluate it, and save the report."""
    LOGGER.info("Starting model training and evaluation.")
    frame = frame.dropna(subset=[config.TARGET_PROFITABILITY_COLUMN]).copy()

    # Ensure all feature columns are present, fill missing with NaN
    for col in config.FEATURE_COLUMNS:
        if col not in frame.columns:
            frame[col] = np.nan

    lcoe_train, lcoe_test = _train_test_split(frame)
    lcoe_model, lcoe_train_metrics = _fit_linear_regression(
        lcoe_train, list(config.FEATURE_COLUMNS), config.TARGET_PROFITABILITY_COLUMN
    )
    lcoe_test_metrics = _evaluate_model(
        lcoe_model, lcoe_test, config.TARGET_PROFITABILITY_COLUMN
    )

    new_location_prediction = predict_new_location(lcoe_model)

    report = {
        "lcoe_model": {
            "features": list(config.FEATURE_COLUMNS),
            "train_metrics": lcoe_train_metrics,
            "test_metrics": lcoe_test_metrics,
            "intercept": lcoe_model.intercept,
            "coefficients": dict(
                zip(list(config.FEATURE_COLUMNS), lcoe_model.coefficients.tolist())
            ),
        },
        "new_location_prediction": new_location_prediction,
    }

    with open(config.MODEL_REPORT_PATH, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    create_model_dashboard(report, config.MODEL_DASHBOARD_PATH)

    LOGGER.info("Model training finished: wrote %s", config.MODEL_REPORT_PATH)
    LOGGER.info("Saved interactive dashboard to %s", config.MODEL_DASHBOARD_PATH)
    LOGGER.info(
        "New location prediction: wind=%.3f m/s, lcoe=%.3f EUR/MWh",
        new_location_prediction["mean_wind_speed_mps"],
        new_location_prediction["predicted_lcoe_eur_per_mwh"],
    )
