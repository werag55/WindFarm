"""Profitability map visualization for offshore wind projects."""

import logging

import pandas as pd
import plotly.express as px
import plotly.graph_objs as go


LOGGER = logging.getLogger(__name__)

def profitability_map(df: pd.DataFrame, color_column: str) -> go.Figure:
    """Create a geographic scatter plot of profitability across wind farm locations."""
    df_map = df.copy()

    color_min, color_max = _get_colors_range(df_map, color_column)
    fig = px.scatter_geo(
        df_map,
        lat="LAT",
        lon="LON",
        hover_name="wind_farm_name",
        hover_data={
            "country": True,
            "lcoe_eur_per_mwh": ":.0f",
            "annual_energy_mwh": ":.0f",
            "installed_capacity_MW": True,
            "mean_wind_speed_mps": True,
            "distance_from_shore_km": True,
            "water_depth_m": True,
        },
        size="installed_capacity_MW",
        color=color_column,
        range_color=(color_min, color_max),
        color_continuous_scale="RdYlGn_r", #px.colors.sequential.Viridis,
    )
    fig.update_layout(
        title=f"Profitability Map (color: {color_column}, size: installed capacity)",
        geo=dict(
            scope="europe",
            projection_type="natural earth",
            showland=True,
            landcolor="lightgray",
            countrycolor="white",
        ),
    )
    return fig

def _get_colors_range(df: pd.DataFrame, column: str) -> tuple[float, float]:
    """Helper to compute color range for a given profitability column."""
    color_min = float(df[column].min())
    color_max = float(df[column].max())
    LOGGER.info("Color scale for %s: min=%.3f max=%.3f", column, color_min, color_max)
    min_idx = df[column].idxmin()
    max_idx = df[column].idxmax()
    min_name = str(df.loc[min_idx, "wind_farm_name"])
    max_name = str(df.loc[max_idx, "wind_farm_name"])
    LOGGER.info(
        "%s minimum: %.3f at wind farm '%s'",
        column,
        color_min,
        min_name,
    )
    LOGGER.info(
        "%s maximum: %.3f at wind farm '%s'",
        column,
        color_max,
        max_name,
    )
    return color_min, color_max
