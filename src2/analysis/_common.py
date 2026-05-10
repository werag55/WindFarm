"""Shared helpers for analysis modules: reference sample, enrichment, prediction."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

from .. import config
from ..calculations.calculations import calculate


def build_reference_sample() -> pd.DataFrame:
    """Reference 800 MW farm in central North Sea (project-card spec).

    Water depth, distance from shore, wind, and wave fields are intentionally
    left empty here — the enrichment pipeline auto-computes them from LAT/LON.
    """
    sample = deepcopy(config.DEFAULT_NEW_SAMPLE)
    sample.update({
        "LAT": 55.5, "LON": 2.5,
        "country": "UK",
        "commissioning_year": config.CURRENT_YEAR + 2,
        "installed_capacity_MW": 800.0,
        "turbine_power_MW": 15.0,
        "turbine_producer": "Siemens Gamesa",
        "foundation_type": "Monopile",
        "area_sqkm": 800.0 / config.POWER_DENSITY_MW_PER_SQKM,
        "project_lifetime_years": 25,
    })
    return pd.DataFrame([sample])


def enrich_sample(sample: pd.DataFrame, training_df: pd.DataFrame) -> pd.DataFrame:
    """Add environmental + depth + shore distance + port distances + connection
    details + one-hot encoding. Mirrors `prep.prepare_data` for new UI samples."""
    from ..data_preprocessing.enrichment import (
        add_distance_from_construction_port,
        add_distance_from_port,
        add_distance_from_shore_columns,
        add_environmental_columns,
        add_water_depth_columns,
        log_environmental_examples,
    )

    sample = add_environmental_columns(sample)
    sample = add_water_depth_columns(sample)
    sample = add_distance_from_shore_columns(sample)
    sample = add_distance_from_port(sample)
    sample = add_distance_from_construction_port(sample)
    log_environmental_examples(sample, n=min(5, len(sample)))

    conn = config.get_connection_details(
        sample["country"].iloc[0], sample["commissioning_year"].iloc[0]
    )
    for k, v in conn.items():
        sample[k] = v

    for cat in config.CATEGORICAL_COLUMNS:
        if cat not in sample.columns:
            continue
        for col in training_df.columns:
            if col.startswith(f"{cat}_"):
                value_name = col[len(cat) + 1:]
                sample[col] = (sample[cat].astype(str) == value_name).astype(float)
    return sample


def feature_columns_from_pipeline(pipeline, df: pd.DataFrame) -> list[str]:
    """Extract feature names used by the trained pipeline (with fallback)."""
    if hasattr(pipeline, "feature_names_in_"):
        return list(pipeline.feature_names_in_)
    numeric = [c for c in config.FEATURE_COLUMNS if c in df.columns]
    onehot = [
        c for cat in config.CATEGORICAL_COLUMNS
        for c in df.columns if c.startswith(f"{cat}_")
    ]
    return list(dict.fromkeys(numeric + onehot))


def predict_sample(pipeline, sample: pd.DataFrame, feature_columns: list[str]) -> dict[str, float]:
    """Predict CAPEX and propagate through deterministic LCOE calculation."""
    sample_features = sample.reindex(columns=feature_columns)
    pred_capex = float(pipeline.predict(sample_features)[0])

    sample = sample.copy()
    sample["total_project_budget_eur"]         = pred_capex
    sample["total_project_budget_eur_indexed"] = pred_capex

    calc = calculate(sample)
    capacity = float(calc["installed_capacity_MW"].iloc[0])
    return {
        "predicted_capex_eur": pred_capex,
        "capex_eur_per_mw":    pred_capex / capacity if capacity else float("nan"),
        "annual_capex_eur":    float(calc["annual_capex_eur"].iloc[0]),
        "annual_opex_eur":     float(calc["annual_opex_eur"].iloc[0]),
        "annual_energy_mwh":   float(calc["annual_energy_mwh"].iloc[0]),
        "lcoe_eur_per_mwh":    float(calc["lcoe_eur_per_mwh"].iloc[0]),
    }


def safe_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    """Coerce metric dict into JSON-serialisable floats."""
    keys = ("rmse", "mae", "mape", "median_ae", "r2", "cv_r2_mean", "cv_r2_std")
    return {k: float(metrics.get(k, float("nan"))) for k in keys}


def pct_change(a: float, b: float) -> float:
    if a is None or b is None or a == 0 or np.isnan(a) or np.isnan(b):
        return float("nan")
    return float((b - a) / a * 100.0)