"""Profitability map visualization for offshore wind projects."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

LOGGER = logging.getLogger(__name__)


_HOVER_FORMATS = {
    "country":                            True,
    "lcoe_eur_per_mwh":                  ":.1f",
    "annual_energy_mwh":                 ":,.0f",
    "installed_capacity_MW":             ":,.0f",
    "mean_wind_speed_mps":               ":.2f",
    "distance_from_shore_km":            True,
    "water_depth_m":                     True,
    "distance_from_port_km":             ":.1f",
    "distance_from_construction_port_km":":.1f",
    "nearest_port_name":                 True,
    "nearest_construction_port_name":    True,
}


def profitability_map(df: pd.DataFrame, color_column: str) -> go.Figure:
    """Create a geographic scatter plot of profitability across wind farm locations."""
    df_map = df.copy()

    if color_column not in df_map.columns:
        LOGGER.warning("Color column %r not found — falling back to lcoe_eur_per_mwh",
                       color_column)
        color_column = "lcoe_eur_per_mwh"

    color_min, color_max = _safe_color_range(df_map, color_column)

    hover_data = {k: v for k, v in _HOVER_FORMATS.items() if k in df_map.columns}

    # plotly fails on NaN sizes — clamp to 1 MW to keep the marker visible
    size_col = "installed_capacity_MW"
    if size_col in df_map.columns:
        df_map[size_col] = df_map[size_col].fillna(1.0).clip(lower=1.0)

    fig = px.scatter_geo(
        df_map,
        lat="LAT", lon="LON",
        hover_name="wind_farm_name",
        hover_data=hover_data,
        size=size_col,
        color=color_column,
        range_color=(color_min, color_max),
        color_continuous_scale="RdYlGn_r",
    )
    fig.update_layout(
        title=f"Profitability map  —  color: {color_column}, size: installed capacity",
        margin=dict(l=10, r=10, t=50, b=10),
        geo=dict(
            scope="europe",
            projection_type="natural earth",
            showland=True,
            landcolor="lightgray",
            countrycolor="white",
        ),
    )
    return fig


def _safe_color_range(df: pd.DataFrame, column: str) -> tuple[float, float]:
    """Return (min, max) for color scale, robust to NaN/inf and empty data."""
    series = pd.to_numeric(df[column], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    if series.notna().sum() == 0:
        LOGGER.warning("Column %r has no valid values for color scale", column)
        return 0.0, 1.0
    cmin = float(series.min())
    cmax = float(series.max())
    if cmin == cmax:
        cmax = cmin + 1.0  # avoid a degenerate scale
    LOGGER.info("Color scale for %s: min=%.3f max=%.3f", column, cmin, cmax)

    valid = series.dropna()
    min_idx = valid.idxmin(); max_idx = valid.idxmax()
    LOGGER.info("  min '%s'  at  %s", df.loc[min_idx, "wind_farm_name"], cmin)
    LOGGER.info("  max '%s'  at  %s", df.loc[max_idx, "wind_farm_name"], cmax)
    return cmin, cmax