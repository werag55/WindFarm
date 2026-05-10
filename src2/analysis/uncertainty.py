"""Bootstrap-based uncertainty estimation for CAPEX / LCOE predictions.

Approach
--------
1. Sample the cleaned dataset with replacement N times (each draw has the same
   length as the original training set).
2. For each bootstrap draw: train the best pipeline (chosen on the full data),
   predict CAPEX for a fixed reference sample, propagate through the
   deterministic LCOE formula.
3. Aggregate predictions: mean / median / std / 2.5–97.5 percentiles.

The 95 % percentile interval is the standard bootstrap prediction interval and
captures both data variability (resampling) and model variance (refit each time).
"""

from __future__ import annotations

import json
import logging
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

RESULTS_DIR = Path("results")
UNCERTAINTY_PATH = RESULTS_DIR / "prediction_uncertainty.json"

# Numerical fields in `predict_sample` output that we want statistics over
_TRACKED_FIELDS = (
    "predicted_capex_eur",
    "capex_eur_per_mw",
    "annual_capex_eur",
    "annual_opex_eur",
    "annual_energy_mwh",
    "lcoe_eur_per_mwh",
)


def bootstrap_prediction(
    df: pd.DataFrame,
    sample: pd.DataFrame | None = None,
    n_bootstrap: int = 100,
    random_state: int = 42,
    fail_fast: bool = False,
) -> dict[str, Any]:
    """Run bootstrap over training set, predict the same sample each time.

    Parameters
    ----------
    df
        Cleaned + indexed dataset (must contain the target column).
    sample
        Optional one-row DataFrame. If None, uses the project-card 800 MW reference.
    n_bootstrap
        Number of bootstrap draws (default 100).
    random_state
        Seed for reproducibility.
    fail_fast
        If True, exceptions in one draw abort the whole run. If False (default),
        the failed draw is skipped and logged.
    """
    if sample is None:
        sample = build_reference_sample()

    LOGGER.info("=" * 60)
    LOGGER.info("BOOTSTRAP UNCERTAINTY: n=%d draws", n_bootstrap)
    LOGGER.info("=" * 60)

    # Drop rows without target so the resampled DataFrame is always trainable
    df_clean = df.dropna(subset=[config.TARGET_COLUMN]).reset_index(drop=True)
    n = len(df_clean)
    if n < 10:
        raise ValueError(f"Need at least 10 rows to bootstrap, got {n}")

    rng = np.random.default_rng(random_state)
    enriched_sample = enrich_sample(sample.copy(), df_clean)

    results: list[dict[str, float]] = []
    failures = 0

    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        df_boot = df_clean.iloc[idx].reset_index(drop=True)

        try:
            pipeline, _, _, _ = train_model(df_boot)
            feats = feature_columns_from_pipeline(pipeline, df_boot)
            pred = predict_sample(pipeline, enriched_sample.copy(), feats)
            results.append(pred)
        except Exception as exc:
            failures += 1
            LOGGER.warning("Bootstrap draw %d failed: %s", i, exc)
            if fail_fast:
                raise

        if (i + 1) % max(1, n_bootstrap // 10) == 0:
            LOGGER.info("  ... %d / %d draws done (%d failed)", i + 1, n_bootstrap, failures)

    if not results:
        raise RuntimeError("All bootstrap draws failed — see warnings above")

    LOGGER.info("Bootstrap finished: %d successful / %d failed", len(results), failures)

    stats = _aggregate(results, _TRACKED_FIELDS)
    histograms = _histograms(results, ("predicted_capex_eur", "lcoe_eur_per_mwh"), bins=20)

    report = {
        "settings": {
            "n_bootstrap":      int(n_bootstrap),
            "n_successful":     int(len(results)),
            "n_failed":         int(failures),
            "random_state":     int(random_state),
            "n_training_rows":  int(n),
            "current_year":     int(config.CURRENT_YEAR),
        },
        "sample": _sample_to_dict(sample),
        "statistics": stats,
        "intervals": {
            "capex_eur_95ci":   [stats["predicted_capex_eur"]["p2_5"],
                                 stats["predicted_capex_eur"]["p97_5"]],
            "capex_per_mw_95ci":[stats["capex_eur_per_mw"]["p2_5"],
                                 stats["capex_eur_per_mw"]["p97_5"]],
            "lcoe_95ci":        [stats["lcoe_eur_per_mwh"]["p2_5"],
                                 stats["lcoe_eur_per_mwh"]["p97_5"]],
        },
        "dashboard_data": {
            "histograms": histograms,
            "interval_table": [
                {
                    "field":  field,
                    "mean":   stats[field]["mean"],
                    "median": stats[field]["median"],
                    "std":    stats[field]["std"],
                    "p2_5":   stats[field]["p2_5"],
                    "p97_5":  stats[field]["p97_5"],
                }
                for field in _TRACKED_FIELDS
            ],
            "raw_capex":  [r["predicted_capex_eur"] for r in results],
            "raw_lcoe":   [r["lcoe_eur_per_mwh"]    for r in results],
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    UNCERTAINTY_PATH.write_text(json.dumps(report, indent=2))
    LOGGER.info("Saved %s", UNCERTAINTY_PATH)
    _log_summary(report)
    return report


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate(
    results: list[dict[str, float]], fields: tuple[str, ...]
) -> dict[str, dict[str, float]]:
    """Compute mean / median / std / 2.5–97.5 percentiles per field."""
    out: dict[str, dict[str, float]] = {}
    for f in fields:
        values = np.array([r[f] for r in results if np.isfinite(r.get(f, np.nan))])
        if len(values) == 0:
            out[f] = {k: float("nan") for k in
                      ("mean", "median", "std", "p2_5", "p97_5", "min", "max", "n")}
            continue
        out[f] = {
            "mean":   float(np.mean(values)),
            "median": float(np.median(values)),
            "std":    float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "p2_5":   float(np.percentile(values, 2.5)),
            "p97_5":  float(np.percentile(values, 97.5)),
            "min":    float(np.min(values)),
            "max":    float(np.max(values)),
            "n":      int(len(values)),
        }
    return out


def _histograms(
    results: list[dict[str, float]],
    fields: tuple[str, ...],
    bins: int = 20,
) -> dict[str, dict[str, list[float]]]:
    """Build histogram (counts + bin edges) for each field — for the dashboard."""
    out = {}
    for f in fields:
        values = np.array([r[f] for r in results if np.isfinite(r.get(f, np.nan))])
        if len(values) == 0:
            out[f] = {"counts": [], "bin_edges": []}
            continue
        counts, edges = np.histogram(values, bins=bins)
        out[f] = {
            "counts":    counts.tolist(),
            "bin_edges": edges.tolist(),
        }
    return out


def _sample_to_dict(sample: pd.DataFrame) -> dict[str, Any]:
    """Serialise the (single-row) reference sample to plain dict for JSON."""
    row = sample.iloc[0].to_dict()
    out = {}
    for k, v in row.items():
        if isinstance(v, (np.integer, np.floating)):
            v = float(v)
        elif isinstance(v, (np.bool_,)):
            v = bool(v)
        elif pd.isna(v):
            v = None
        out[k] = v
    return out


def _log_summary(report: dict[str, Any]) -> None:
    """Pretty-print the headline numbers."""
    capex = report["statistics"]["predicted_capex_eur"]
    capex_per_mw = report["statistics"]["capex_eur_per_mw"]
    lcoe = report["statistics"]["lcoe_eur_per_mwh"]

    LOGGER.info("-" * 60)
    LOGGER.info("UNCERTAINTY SUMMARY (n=%d successful draws)",
                report["settings"]["n_successful"])
    LOGGER.info("-" * 60)
    LOGGER.info(
        "Predicted CAPEX:   mean=%.2e  median=%.2e  std=%.2e  95%%CI=[%.2e, %.2e]",
        capex["mean"], capex["median"], capex["std"], capex["p2_5"], capex["p97_5"],
    )
    LOGGER.info(
        "CAPEX per MW:      mean=%.2e  median=%.2e  std=%.2e  95%%CI=[%.2e, %.2e]",
        capex_per_mw["mean"], capex_per_mw["median"], capex_per_mw["std"],
        capex_per_mw["p2_5"], capex_per_mw["p97_5"],
    )
    LOGGER.info(
        "LCOE [EUR/MWh]:    mean=%.2f  median=%.2f  std=%.2f  95%%CI=[%.2f, %.2f]",
        lcoe["mean"], lcoe["median"], lcoe["std"], lcoe["p2_5"], lcoe["p97_5"],
    )
    rel_uncertainty = (capex["p97_5"] - capex["p2_5"]) / max(capex["mean"], 1e-9) * 100
    LOGGER.info("Relative CAPEX 95%% CI width: %.1f%% of mean", rel_uncertainty)
    LOGGER.info("-" * 60)