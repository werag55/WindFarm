"""One-at-a-time (OAT) sensitivity analysis.

For each feature in `FEATURES_TO_PERTURB` we shift its baseline value by
−20 %, −10 %, +10 %, +20 % (or by ±1, ±2 years for commissioning_year), keep
all other features at their baseline, and re-evaluate:
    - predicted CAPEX (model)
    - CAPEX / MW
    - annual CAPEX
    - annual OPEX  (deterministic formula)
    - annual energy (deterministic formula)
    - LCOE         (deterministic formula)

The "swing" of a feature on a metric is the range
    max(metric across perturbations) − min(metric across perturbations)
expressed both in absolute units and as % of the baseline metric value.
This is the standard input for a tornado chart.
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .. import config
from ..model.model import train_model
from ._common import (
    build_reference_sample,
    enrich_sample,
    feature_columns_from_pipeline,
    predict_sample,
)

LOGGER = logging.getLogger(__name__)

RESULTS_DIR    = Path("results")
SENSITIVITY_CSV  = RESULTS_DIR / "sensitivity_analysis.csv"
SENSITIVITY_JSON = RESULTS_DIR / "sensitivity_summary.json"

FEATURES_TO_PERTURB = (
    "installed_capacity_MW",
    "mean_wind_speed_mps",
    "water_depth_mean_m",
    "distance_from_port_km",
    "distance_from_construction_port_km",
    "commissioning_year",
    "project_lifetime_years",
    "area_sqkm",
)

# Multiplicative perturbations applied to most features
DEFAULT_PERTURBATIONS = (-0.20, -0.10, 0.10, 0.20)

# Special-case: commissioning_year and project_lifetime_years are integers and
# small absolute changes are more meaningful than %.
ADDITIVE_PERTURBATIONS = {
    "commissioning_year":     (-2, -1, +1, +2),       # years
    "project_lifetime_years": (-5, -2, +2, +5),       # years
}

TRACKED_METRICS = (
    "predicted_capex_eur",
    "capex_eur_per_mw",
    "annual_capex_eur",
    "annual_opex_eur",
    "annual_energy_mwh",
    "lcoe_eur_per_mwh",
)


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def run_sensitivity_analysis(
    df: pd.DataFrame,
    sample: pd.DataFrame | None = None,
    features: tuple[str, ...] = FEATURES_TO_PERTURB,
    perturbations: tuple[float, ...] = DEFAULT_PERTURBATIONS,
) -> dict[str, Any]:
    """Run OAT sensitivity around the reference sample.

    Returns a dict with full table, baseline, rankings, and dashboard payload.
    """
    if sample is None:
        sample = build_reference_sample()

    LOGGER.info("=" * 60)
    LOGGER.info("OAT SENSITIVITY ANALYSIS")
    LOGGER.info("=" * 60)
    LOGGER.info("Features to perturb: %d", len(features))
    LOGGER.info("Perturbations: %s", perturbations)

    # Train ONCE on the full dataset — sensitivity is about the function the
    # trained model implements, not about its training stability.
    LOGGER.info("Training reference model on %d rows...", len(df))
    pipeline, _, _, _ = train_model(df)
    feats = feature_columns_from_pipeline(pipeline, df)

    enriched_baseline = enrich_sample(sample.copy(), df)
    baseline_pred = predict_sample(pipeline, enriched_baseline.copy(), feats)
    LOGGER.info(
        "Baseline: CAPEX=%.2e EUR (%.0f EUR/MW), LCOE=%.2f EUR/MWh",
        baseline_pred["predicted_capex_eur"],
        baseline_pred["capex_eur_per_mw"],
        baseline_pred["lcoe_eur_per_mwh"],
    )

    rows: list[dict[str, Any]] = []
    # Baseline row
    rows.append(_make_row(
        feature="<baseline>",
        perturbation_label="baseline",
        perturbation_value=0.0,
        new_value=np.nan,
        baseline_value=np.nan,
        prediction=baseline_pred,
        baseline_pred=baseline_pred,
    ))

    for feature in features:
        if feature not in enriched_baseline.columns:
            LOGGER.warning("Feature %r not in sample columns — skipping", feature)
            continue

        baseline_value = enriched_baseline[feature].iloc[0]
        if pd.isna(baseline_value):
            LOGGER.warning(
                "Feature %r has NaN baseline value — skipping", feature
            )
            continue

        # Choose perturbation type
        if feature in ADDITIVE_PERTURBATIONS:
            perts = ADDITIVE_PERTURBATIONS[feature]
            mode = "additive"
        else:
            perts = perturbations
            mode = "multiplicative"

        for p in perts:
            new_value = (
                baseline_value + p
                if mode == "additive"
                else baseline_value * (1.0 + p)
            )

            # Sanity bounds
            new_value = _clip_to_physical_range(feature, new_value)
            if new_value == baseline_value:
                continue

            perturbed = enriched_baseline.copy()
            perturbed[feature] = new_value

            try:
                pred = predict_sample(pipeline, perturbed, feats)
            except Exception as exc:
                LOGGER.warning(
                    "Prediction failed for %s @ %s: %s", feature, p, exc
                )
                continue

            label = (f"{int(p):+d}y" if mode == "additive"
                     else f"{p*100:+.0f}%")

            rows.append(_make_row(
                feature=feature,
                perturbation_label=label,
                perturbation_value=float(p),
                new_value=float(new_value),
                baseline_value=float(baseline_value),
                prediction=pred,
                baseline_pred=baseline_pred,
            ))

    table = pd.DataFrame(rows)

    rankings = _build_rankings(table, baseline_pred)
    dashboard_data = _build_dashboard_data(table, rankings, baseline_pred)

    summary = {
        "settings": {
            "n_features":      len(features),
            "perturbations":   list(perturbations),
            "additive_overrides": {k: list(v) for k, v in ADDITIVE_PERTURBATIONS.items()},
            "n_training_rows": int(len(df)),
        },
        "baseline":  baseline_pred,
        "rankings":  rankings,
        "dashboard_data": dashboard_data,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(SENSITIVITY_CSV, index=False)
    SENSITIVITY_JSON.write_text(json.dumps(summary, indent=2))
    LOGGER.info("Saved %s", SENSITIVITY_CSV)
    LOGGER.info("Saved %s", SENSITIVITY_JSON)

    _log_top5(rankings, "predicted_capex_eur", "CAPEX")
    _log_top5(rankings, "lcoe_eur_per_mwh",    "LCOE")

    return {"table": table, "summary": summary}


# ---------------------------------------------------------------------------
# Row + ranking helpers
# ---------------------------------------------------------------------------

def _make_row(
    feature: str,
    perturbation_label: str,
    perturbation_value: float,
    new_value: float,
    baseline_value: float,
    prediction: dict[str, float],
    baseline_pred: dict[str, float],
) -> dict[str, Any]:
    """Build one CSV row with metric values + relative deltas vs baseline."""
    row: dict[str, Any] = {
        "feature":            feature,
        "perturbation":       perturbation_label,
        "perturbation_value": perturbation_value,
        "new_value":          new_value,
        "baseline_value":     baseline_value,
    }
    for m in TRACKED_METRICS:
        v = prediction.get(m, float("nan"))
        b = baseline_pred.get(m, float("nan"))
        row[m] = v
        row[f"{m}_delta_pct"] = (
            float((v - b) / b * 100.0)
            if b not in (0, None) and not (np.isnan(v) or np.isnan(b))
            else float("nan")
        )
    return row


def _build_rankings(
    table: pd.DataFrame, baseline_pred: dict[str, float]
) -> dict[str, list[dict[str, Any]]]:
    """For each tracked metric, rank features by max swing (range across perturbations)."""
    perturbed = table[table["feature"] != "<baseline>"]
    rankings: dict[str, list[dict[str, Any]]] = {}

    for metric in TRACKED_METRICS:
        baseline_val = baseline_pred.get(metric, float("nan"))
        records = []
        for feature, group in perturbed.groupby("feature", sort=False):
            vmin = float(group[metric].min())
            vmax = float(group[metric].max())
            swing_abs = vmax - vmin
            swing_pct = (
                swing_abs / abs(baseline_val) * 100.0
                if baseline_val and not np.isnan(baseline_val)
                else float("nan")
            )
            records.append({
                "feature":   feature,
                "min":       vmin,
                "max":       vmax,
                "swing_abs": float(swing_abs),
                "swing_pct": float(swing_pct),
            })
        records.sort(key=lambda r: abs(r["swing_pct"]) if not np.isnan(r["swing_pct"])
                     else -1, reverse=True)
        rankings[metric] = records
    return rankings


def _build_dashboard_data(
    table: pd.DataFrame,
    rankings: dict[str, list[dict[str, Any]]],
    baseline_pred: dict[str, float],
) -> dict[str, Any]:
    """Tornado-chart payload for CAPEX and LCOE."""
    perturbed = table[table["feature"] != "<baseline>"]

    def _tornado(metric: str) -> list[dict[str, Any]]:
        baseline_val = baseline_pred[metric]
        out = []
        for feature, group in perturbed.groupby("feature", sort=False):
            low_row  = group.loc[group[metric].idxmin()]
            high_row = group.loc[group[metric].idxmax()]
            out.append({
                "feature":          feature,
                "low":              float(low_row[metric]),
                "high":             float(high_row[metric]),
                "low_label":        str(low_row["perturbation"]),
                "high_label":       str(high_row["perturbation"]),
                "low_delta_pct":    float(low_row[f"{metric}_delta_pct"]),
                "high_delta_pct":   float(high_row[f"{metric}_delta_pct"]),
                "swing_pct":        float(abs(high_row[metric] - low_row[metric])
                                          / abs(baseline_val) * 100.0)
                                    if baseline_val else float("nan"),
            })
        out.sort(key=lambda r: r["swing_pct"], reverse=True)
        return out

    return {
        "tornado_capex": _tornado("predicted_capex_eur"),
        "tornado_lcoe":  _tornado("lcoe_eur_per_mwh"),
        "ranking_capex": [
            {"feature": r["feature"], "swing_pct": r["swing_pct"]}
            for r in rankings["predicted_capex_eur"]
        ],
        "ranking_lcoe": [
            {"feature": r["feature"], "swing_pct": r["swing_pct"]}
            for r in rankings["lcoe_eur_per_mwh"]
        ],
    }


def _clip_to_physical_range(feature: str, value: float) -> float:
    """Keep perturbed values inside physically meaningful bounds."""
    bounds = {
        "installed_capacity_MW":              (10.0, 5000.0),
        "mean_wind_speed_mps":                (4.0, 14.0),
        "water_depth_mean_m":                 (1.0, 200.0),
        "distance_from_port_km":              (1.0, 500.0),
        "distance_from_construction_port_km": (1.0, 1000.0),
        "commissioning_year":                 (1990, config.CURRENT_YEAR + 20),
        "project_lifetime_years":             (5, 50),
        "area_sqkm":                          (1.0, 2000.0),
    }
    lo, hi = bounds.get(feature, (None, None))
    if lo is None:
        return value
    return float(np.clip(value, lo, hi))


def _log_top5(
    rankings: dict[str, list[dict[str, Any]]],
    metric: str,
    label: str,
) -> None:
    LOGGER.info("-" * 60)
    LOGGER.info("Top 5 most influential features for %s:", label)
    LOGGER.info("-" * 60)
    for i, rec in enumerate(rankings[metric][:5], start=1):
        LOGGER.info(
            "  %d. %-36s swing = %+6.2f %%   (%.2e ↔ %.2e)",
            i, rec["feature"], rec["swing_pct"], rec["min"], rec["max"],
        )