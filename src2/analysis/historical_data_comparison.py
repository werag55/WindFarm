"""Compare model behaviour across data subsets and indexation scenarios.

Two analyses are exposed:
  * compare_full_vs_recent  -- model trained on full history vs only the last 5 years.
  * compare_indexation_scenarios -- model trained on full history under five
    different inflation/CPI indexation strategies.

Both produce JSON reports and structured dicts ready for dashboard consumption.
"""

from __future__ import annotations

import json
import logging
from ._common import (
    build_reference_sample as _build_reference_sample,
    enrich_sample as _enrich_sample,
    feature_columns_from_pipeline as _feature_columns_from_pipeline,
    predict_sample as _predict_sample,
    safe_metrics as _safe_metrics,
    pct_change as _pct,
)
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .. import config
from ..calculations.calculations import calculate
from ..data_preprocessing.indexation import INDEXATION_MODES, build_indexed_dataset
from ..model.model import train_model

LOGGER = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
FULL_VS_RECENT_PATH = RESULTS_DIR / "full_vs_recent_comparison.json"
INDEXATION_PATH     = RESULTS_DIR / "indexation_scenarios_comparison.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _last_n_years(df: pd.DataFrame, years: int) -> pd.DataFrame:
    """Filter to projects commissioned in the last `years` years (inclusive)."""
    cutoff = config.CURRENT_YEAR - years
    out = df[df["commissioning_year"].astype(float) >= cutoff].copy()
    LOGGER.info(
        "Filter last %d years (>= %d): %d / %d rows kept",
        years, cutoff, len(out), len(df),
    )
    return out


def _safe_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    """Coerce metric dict into JSON-serialisable floats."""
    keys = ("rmse", "mae", "mape", "median_ae", "r2", "cv_r2_mean", "cv_r2_std")
    return {k: float(metrics.get(k, float("nan"))) for k in keys}


def _build_reference_sample() -> pd.DataFrame:
    """Build a single new-location sample mirroring the project card reference design.

    800 MW farm, 10 MW turbines, 50 m water depth, 90 km from shore — North Sea coords.
    """
    sample = deepcopy(config.DEFAULT_NEW_SAMPLE)
    # Override with the project-card "standard project" parameters
    sample.update({
        "LAT": 55.5, "LON": 2.5,           # Central North Sea
        "country": "UK",
        "commissioning_year": config.CURRENT_YEAR + 2,
        "installed_capacity_MW": 800.0,
        "turbine_power_MW": 15.0,
        "turbine_producer": "Siemens Gamesa",
        "foundation_type": "Monopile",
        "water_depth_min_m": 30.0, "water_depth_max_m": 50.0, "water_depth_mean_m": 40.0,
        "distance_from_shore_min_km": 80.0, "distance_from_shore_max_km": 100.0,
        "distance_from_shore_mean_km": 90.0,
        "area_sqkm": 800.0 / config.POWER_DENSITY_MW_PER_SQKM,
        "project_lifetime_years": 25,
    })
    return pd.DataFrame([sample])


def _enrich_sample(sample: pd.DataFrame, training_df: pd.DataFrame) -> pd.DataFrame:
    """Add environmental + port distances + connection details + one-hot columns."""
    from ..data_preprocessing.enrichment import (
        add_distance_from_construction_port,
        add_distance_from_port,
        add_environmental_columns,
    )

    sample = add_environmental_columns(sample)
    sample = add_distance_from_port(sample)
    sample = add_distance_from_construction_port(sample)

    conn = config.get_connection_details(
        sample["country"].iloc[0], sample["commissioning_year"].iloc[0]
    )
    for k, v in conn.items():
        sample[k] = v

    # Replicate one-hot columns matching the training frame
    for cat in config.CATEGORICAL_COLUMNS:
        if cat not in sample.columns:
            continue
        for col in training_df.columns:
            if col.startswith(f"{cat}_"):
                value_name = col[len(cat) + 1:]
                sample[col] = (sample[cat].astype(str) == value_name).astype(float)
    return sample


def _predict_sample(
    pipeline,
    sample: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, float]:
    """Predict CAPEX with `pipeline` and propagate through deterministic LCOE calc."""
    sample_features = sample.reindex(columns=feature_columns)
    pred_capex = float(pipeline.predict(sample_features)[0])

    # Inject prediction into both budget columns so downstream formulas work
    sample = sample.copy()
    sample["total_project_budget_eur"]         = pred_capex
    sample["total_project_budget_eur_indexed"] = pred_capex

    calc = calculate(sample)
    capacity = float(calc["installed_capacity_MW"].iloc[0])
    return {
        "predicted_capex_eur":              pred_capex,
        "capex_eur_per_mw":                 pred_capex / capacity if capacity else float("nan"),
        "annual_capex_eur":                 float(calc["annual_capex_eur"].iloc[0]),
        "annual_opex_eur":                  float(calc["annual_opex_eur"].iloc[0]),
        "annual_energy_mwh":                float(calc["annual_energy_mwh"].iloc[0]),
        "lcoe_eur_per_mwh":                 float(calc["lcoe_eur_per_mwh"].iloc[0]),
    }


def _feature_columns_from_pipeline(pipeline, df: pd.DataFrame) -> list[str]:
    """Extract feature names used by the pipeline, with a robust fallback."""
    if hasattr(pipeline, "feature_names_in_"):
        return list(pipeline.feature_names_in_)
    # Fallback: rebuild from config
    numeric = [c for c in config.FEATURE_COLUMNS if c in df.columns]
    onehot = [
        c for cat in config.CATEGORICAL_COLUMNS
        for c in df.columns if c.startswith(f"{cat}_")
    ]
    return list(dict.fromkeys(numeric + onehot))


# ---------------------------------------------------------------------------
# Analysis 1: Full vs Recent
# ---------------------------------------------------------------------------

def compare_full_vs_recent(
    df: pd.DataFrame, recent_years: int = 5
) -> dict[str, Any]:
    """Train two models (full / recent only), report metrics + sample prediction."""
    LOGGER.info("=" * 60)
    LOGGER.info("ANALYSIS 1: full dataset vs last %d years", recent_years)
    LOGGER.info("=" * 60)

    sample = _build_reference_sample()

    # --- Full ---
    LOGGER.info("Training on FULL dataset (%d rows)", len(df))
    pipe_full, metrics_full, _, _ = train_model(df)
    feats_full = _feature_columns_from_pipeline(pipe_full, df)
    sample_full = _enrich_sample(sample.copy(), df)
    pred_full = _predict_sample(pipe_full, sample_full, feats_full)

    # --- Recent ---
    df_recent = _last_n_years(df, recent_years)
    if len(df_recent) < 10:
        LOGGER.warning(
            "Recent subset has only %d rows — model unstable", len(df_recent)
        )
    LOGGER.info("Training on RECENT subset (%d rows)", len(df_recent))
    pipe_rec, metrics_rec, _, _ = train_model(df_recent)
    feats_rec = _feature_columns_from_pipeline(pipe_rec, df_recent)
    sample_rec = _enrich_sample(sample.copy(), df_recent)
    pred_rec = _predict_sample(pipe_rec, sample_rec, feats_rec)

    # Pick best-CV metric per model for the report
    def _best(metrics):
        return max(metrics.items(), key=lambda kv: kv[1].get("cv_r2_mean", -np.inf))

    name_full, m_full = _best(metrics_full)
    name_rec,  m_rec  = _best(metrics_rec)

    report = {
        "settings": {
            "recent_years": recent_years,
            "current_year": config.CURRENT_YEAR,
            "n_full":   int(len(df)),
            "n_recent": int(len(df_recent)),
        },
        "full": {
            "best_model":       name_full,
            "metrics_by_model": {n: _safe_metrics(m) for n, m in metrics_full.items()},
            "best_metrics":     _safe_metrics(m_full),
            "sample_prediction": pred_full,
        },
        "recent": {
            "best_model":       name_rec,
            "metrics_by_model": {n: _safe_metrics(m) for n, m in metrics_rec.items()},
            "best_metrics":     _safe_metrics(m_rec),
            "sample_prediction": pred_rec,
        },
        "delta": {
            "predicted_capex_pct":
                _pct(pred_full["predicted_capex_eur"], pred_rec["predicted_capex_eur"]),
            "lcoe_pct":
                _pct(pred_full["lcoe_eur_per_mwh"], pred_rec["lcoe_eur_per_mwh"]),
            "best_r2_diff":
                m_full["r2"] - m_rec["r2"] if not (np.isnan(m_full["r2"]) or np.isnan(m_rec["r2"])) else float("nan"),
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FULL_VS_RECENT_PATH.write_text(json.dumps(report, indent=2))
    LOGGER.info("Saved %s", FULL_VS_RECENT_PATH)
    _log_full_vs_recent(report)
    return report


def _pct(a: float, b: float) -> float:
    if a is None or b is None or a == 0 or np.isnan(a) or np.isnan(b):
        return float("nan")
    return float((b - a) / a * 100.0)


def _log_full_vs_recent(report: dict[str, Any]) -> None:
    f = report["full"]; r = report["recent"]; d = report["delta"]
    LOGGER.info(
        "FULL  : best=%s  CV-R²=%.3f  test-R²=%.3f  RMSE=%.2e  MAPE=%.3f",
        f["best_model"],
        f["best_metrics"]["cv_r2_mean"], f["best_metrics"]["r2"],
        f["best_metrics"]["rmse"],       f["best_metrics"]["mape"],
    )
    LOGGER.info(
        "RECENT: best=%s  CV-R²=%.3f  test-R²=%.3f  RMSE=%.2e  MAPE=%.3f",
        r["best_model"],
        r["best_metrics"]["cv_r2_mean"], r["best_metrics"]["r2"],
        r["best_metrics"]["rmse"],       r["best_metrics"]["mape"],
    )
    LOGGER.info(
        "Sample pred — FULL: CAPEX=%.2e EUR (%.0f EUR/MW)  LCOE=%.2f EUR/MWh",
        f["sample_prediction"]["predicted_capex_eur"],
        f["sample_prediction"]["capex_eur_per_mw"],
        f["sample_prediction"]["lcoe_eur_per_mwh"],
    )
    LOGGER.info(
        "Sample pred — RECENT: CAPEX=%.2e EUR (%.0f EUR/MW)  LCOE=%.2f EUR/MWh",
        r["sample_prediction"]["predicted_capex_eur"],
        r["sample_prediction"]["capex_eur_per_mw"],
        r["sample_prediction"]["lcoe_eur_per_mwh"],
    )
    LOGGER.info(
        "Δ predicted CAPEX (recent vs full): %+.1f%%  | Δ LCOE: %+.1f%%",
        d["predicted_capex_pct"], d["lcoe_pct"],
    )


# ---------------------------------------------------------------------------
# Analysis 2: Indexation scenarios
# ---------------------------------------------------------------------------

def compare_indexation_scenarios(
    df_unindexed: pd.DataFrame,
) -> dict[str, Any]:
    """Re-run indexation in each mode, train, evaluate, predict on shared sample.

    `df_unindexed` should be the cleaned + enriched dataset BEFORE indexation
    (i.e. it must already contain `total_project_budget_eur` but indexation
    is re-applied per scenario).
    """
    LOGGER.info("=" * 60)
    LOGGER.info("ANALYSIS 2: indexation scenarios")
    LOGGER.info("=" * 60)

    base_sample = _build_reference_sample()
    scenarios: dict[str, dict[str, Any]] = {}

    for mode in INDEXATION_MODES:
        LOGGER.info("--- Scenario: %s ---", mode)
        df_mode = build_indexed_dataset(df_unindexed.copy(), mode=mode)

        try:
            pipe, metrics, _, _ = train_model(df_mode)
        except Exception as exc:
            LOGGER.error("Training failed for %s: %s", mode, exc)
            scenarios[mode] = {"error": str(exc)}
            continue

        feats = _feature_columns_from_pipeline(pipe, df_mode)
        sample = _enrich_sample(base_sample.copy(), df_mode)
        pred = _predict_sample(pipe, sample, feats)

        best_name, best_metrics = max(
            metrics.items(), key=lambda kv: kv[1].get("cv_r2_mean", -np.inf)
        )

        scenarios[mode] = {
            "best_model":        best_name,
            "metrics_by_model":  {n: _safe_metrics(m) for n, m in metrics.items()},
            "best_metrics":      _safe_metrics(best_metrics),
            "sample_prediction": pred,
            "indexed_budget_stats": {
                "mean":   float(df_mode["total_project_budget_eur_indexed"].mean()),
                "median": float(df_mode["total_project_budget_eur_indexed"].median()),
                "std":    float(df_mode["total_project_budget_eur_indexed"].std()),
            },
        }

    report = {
        "settings": {
            "current_year":     config.CURRENT_YEAR,
            "indexed_by_year":  config.INDEXED_BY_YEAR,
            "max_data_age":     config.MAX_DATA_AGE_YEARS,
            "n_rows":           int(len(df_unindexed)),
        },
        "scenarios": scenarios,
        "dashboard_data": _build_dashboard_data(scenarios),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    INDEXATION_PATH.write_text(json.dumps(report, indent=2))
    LOGGER.info("Saved %s", INDEXATION_PATH)
    _log_indexation_summary(report)
    return report


def _build_dashboard_data(scenarios: dict[str, dict]) -> dict[str, Any]:
    """Reshape scenario results into table-ready arrays for the dashboard."""
    rows = []
    for name, s in scenarios.items():
        if "error" in s:
            continue
        bm = s["best_metrics"]
        sp = s["sample_prediction"]
        rows.append({
            "scenario":          name,
            "best_model":        s["best_model"],
            "rmse":              bm["rmse"],
            "mae":               bm["mae"],
            "mape":              bm["mape"],
            "median_ae":         bm["median_ae"],
            "r2":                bm["r2"],
            "cv_r2_mean":        bm["cv_r2_mean"],
            "predicted_capex":   sp["predicted_capex_eur"],
            "capex_per_mw":      sp["capex_eur_per_mw"],
            "lcoe":              sp["lcoe_eur_per_mwh"],
            "indexed_budget_mean": s["indexed_budget_stats"]["mean"],
        })
    return {
        "metrics_table":     rows,
        "predicted_capex":   [(r["scenario"], r["predicted_capex"]) for r in rows],
        "lcoe":              [(r["scenario"], r["lcoe"]) for r in rows],
    }


def _log_indexation_summary(report: dict[str, Any]) -> None:
    LOGGER.info(
        "%-18s  %-16s  %-8s  %-8s  %-12s  %-8s",
        "scenario", "best_model", "CV-R²", "R²", "CAPEX (EUR)", "LCOE",
    )
    for row in report["dashboard_data"]["metrics_table"]:
        LOGGER.info(
            "%-18s  %-16s  %-8.3f  %-8.3f  %-12.2e  %-8.2f",
            row["scenario"], row["best_model"],
            row["cv_r2_mean"], row["r2"],
            row["predicted_capex"], row["lcoe"],
        )