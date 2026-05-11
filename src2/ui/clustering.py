"""Similarity clustering helpers for the Dash UI."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import dash_bootstrap_components as dbc
from dash import dash_table, dcc, html

from .. import config
from ..data_preprocessing.enrichment import add_distance_from_port


CLUSTERING_METRICS = {
    "area_sqkm": "Area (sqkm)",
    "water_depth_mean_m": "Mean water depth (m)",
    "distance_from_shore_mean_km": "Mean distance from shore (km)",
    "installed_capacity_MW": "Installed capacity (MW)",
    "turbine_power_MW": "Turbine power (MW)",
    "distance_from_port_km": "Distance from port (km)",
    "distance_from_construction_port_km": "Distance from construction port (km)",
    "mean_wind_speed_mps": "Mean wind speed (m/s)",
    "mean_wave_height_m": "Mean wave height (m)",
}

DEFAULT_CLUSTERING_METRICS = [
    "area_sqkm",
    "water_depth_mean_m",
    "distance_from_shore_mean_km",
]


def prepare_clustering_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Fill clustering-only metrics that can be recovered from local project data."""
    clustering_df = df.copy()
    clustering_df = _add_turbine_power_from_raw_data(clustering_df)

    if (
        "distance_from_port_km" in clustering_df.columns
        and pd.to_numeric(clustering_df["distance_from_port_km"], errors="coerce").notna().any()
    ):
        return clustering_df

    return add_distance_from_port(clustering_df)


def create_clustering_panel(df: pd.DataFrame) -> dbc.Card:
    """Create controls and output area for farm similarity search."""
    metric_options = []
    for value, label in CLUSTERING_METRICS.items():
        has_historical_data = (
            value in df.columns
            and pd.to_numeric(df[value], errors="coerce").notna().any()
        )
        metric_options.append(
            {
                "label": label if has_historical_data else f"{label} (no historical data)",
                "value": value,
                "disabled": not has_historical_data,
            }
        )

    return dbc.Card(
        [
            dbc.CardHeader("Closest Historical Farms"),
            dbc.CardBody(
                [
                    dbc.Label("Metrics"),
                    dcc.Checklist(
                        id="cluster-metrics-checklist",
                        options=metric_options,
                        value=DEFAULT_CLUSTERING_METRICS,
                        inputStyle={"marginRight": "0.35rem"},
                        labelStyle={"display": "block", "marginBottom": "0.35rem"},
                    ),
                    dbc.Label("Scaling", className="mt-2"),
                    dcc.Dropdown(
                        id="cluster-scaling-dropdown",
                        options=[
                            {
                                "label": "Z-score (recommended)",
                                "value": "zscore",
                            },
                            {"label": "Percentage difference", "value": "percent"},
                            {"label": "Raw values", "value": "raw"},
                        ],
                        value="zscore",
                        clearable=False,
                    ),
                    html.Div(id="cluster-output", className="mt-3"),
                ]
            ),
        ],
        className="mt-3",
    )


def find_closest_farms(
    df: pd.DataFrame,
    sample: pd.Series,
    metrics: list[str],
    scaling: str,
    top_n: int = 5,
) -> pd.DataFrame:
    """Find the closest farms to a proposed sample by selected numeric metrics."""
    available_metrics = [
        metric
        for metric in metrics
        if metric in df.columns and metric in sample.index
    ]
    if not available_metrics:
        return pd.DataFrame()

    sample_values = pd.to_numeric(sample[available_metrics], errors="coerce")
    sample_values = sample_values.dropna()
    available_metrics = [metric for metric in available_metrics if metric in sample_values.index]
    if not available_metrics:
        return pd.DataFrame()

    working_df = df.copy()
    for metric in available_metrics:
        working_df[metric] = pd.to_numeric(working_df[metric], errors="coerce")

    available_metrics = [
        metric
        for metric in available_metrics
        if working_df[metric].notna().any()
    ]
    if not available_metrics:
        return _reference_row(sample, metrics)

    comparison_df = working_df.dropna(subset=available_metrics).copy()
    if comparison_df.empty:
        return _reference_row(sample, metrics)

    distances = _scaled_distances(
        comparison_df[available_metrics],
        sample_values[available_metrics],
        scaling,
    )
    comparison_df["similarity_distance"] = distances
    comparison_df = comparison_df.replace([np.inf, -np.inf], np.nan)
    comparison_df = comparison_df.dropna(subset=["similarity_distance"])
    if comparison_df.empty:
        return _reference_row(sample, metrics)

    comparison_df["similarity_score"] = 1 / (1 + comparison_df["similarity_distance"])

    selected_display_metrics = [
        metric
        for metric in metrics
        if metric in comparison_df.columns and metric in sample.index
    ]
    display_columns = [
        "wind_farm_name",
        "country",
        "similarity_score",
        "similarity_distance",
        *selected_display_metrics,
    ]
    display_columns = [column for column in display_columns if column in comparison_df.columns]

    closest_farms = (
        comparison_df.sort_values("similarity_distance")
        .head(top_n)[display_columns]
        .reset_index(drop=True)
    )
    return pd.concat(
        [_reference_row(sample, display_columns), closest_farms],
        ignore_index=True,
    )


def create_closest_farms_table(
    df: pd.DataFrame,
    sample: pd.Series,
    metrics: list[str] | None,
    scaling: str,
) -> html.Div:
    """Render the closest farms table and short scaling explanation."""
    if not metrics:
        return html.Div("Choose at least one metric to compare farms.")

    closest_farms = find_closest_farms(df, sample, metrics, scaling)
    if closest_farms.empty:
        return html.Div("No comparable farms found for the selected metrics.")

    display_df = _format_closest_farms(closest_farms)
    return html.Div(
        [
            html.P(_scaling_explanation(scaling), className="small text-muted mb-2"),
            dash_table.DataTable(
                columns=[
                    {"name": _column_label(column), "id": column}
                    for column in display_df.columns
                ],
                data=display_df.to_dict("records"),
                page_action="none",
                style_table={"overflowX": "auto"},
                style_cell={
                    "fontFamily": "Arial, sans-serif",
                    "fontSize": 12,
                    "padding": "0.35rem",
                    "textAlign": "left",
                    "whiteSpace": "normal",
                    "height": "auto",
                },
                style_header={"fontWeight": "bold"},
            ),
        ]
    )


def _scaled_distances(
    comparison_values: pd.DataFrame,
    sample_values: pd.Series,
    scaling: str,
) -> pd.Series:
    """Calculate Euclidean distances after applying the selected scaling."""
    if scaling == "raw":
        differences = comparison_values - sample_values
    elif scaling == "percent":
        denominators = sample_values.abs().replace(0, np.nan)
        differences = (comparison_values - sample_values) / denominators
    else:
        means = comparison_values.mean()
        stds = comparison_values.std(ddof=0).replace(0, np.nan)
        comparison_scaled = (comparison_values - means) / stds
        sample_scaled = (sample_values - means) / stds
        differences = comparison_scaled - sample_scaled

    differences = differences.replace([np.inf, -np.inf], np.nan)
    metric_counts = differences.notna().sum(axis=1)
    sum_squared = (differences**2).sum(axis=1, skipna=True)
    return np.sqrt(sum_squared / metric_counts.replace(0, np.nan))


def _reference_row(sample: pd.Series, columns: list[str]) -> pd.DataFrame:
    """Create a reference row containing the proposed farm values."""
    row = {column: np.nan for column in columns}
    row["wind_farm_name"] = "Proposed Wind Farm"
    row["country"] = sample.get("country", "")
    row["similarity_score"] = 1.0
    row["similarity_distance"] = 0.0
    metric_columns = [column for column in columns if column in CLUSTERING_METRICS]
    for column in columns:
        if column in {"wind_farm_name", "country", "similarity_score", "similarity_distance"}:
            continue
        if column in sample.index:
            row[column] = sample[column]
    return pd.DataFrame([row])


def _format_closest_farms(df: pd.DataFrame) -> pd.DataFrame:
    """Format result values for compact display."""
    display_df = df.copy()
    if "similarity_score" in display_df.columns:
        display_df["similarity_score"] = display_df["similarity_score"].map("{:.3f}".format)
    if "similarity_distance" in display_df.columns:
        display_df["similarity_distance"] = display_df["similarity_distance"].map("{:.3f}".format)
    for column in display_df.columns:
        if column in {"wind_farm_name", "country", "similarity_score", "similarity_distance"}:
            continue
        display_df[column] = pd.to_numeric(display_df[column], errors="coerce").map(
            lambda value: "" if pd.isna(value) else f"{value:,.1f}"
        )
    return display_df


def _column_label(column: str) -> str:
    """Return a friendly column label."""
    default_labels = {
        "wind_farm_name": "Farm",
        "country": "Country",
        "similarity_score": "Similarity",
        "similarity_distance": "Distance",
    }
    return default_labels.get(column, CLUSTERING_METRICS.get(column, column))


def _scaling_explanation(scaling: str) -> str:
    """Explain the selected scaling in one compact sentence."""
    if scaling == "raw":
        return "Raw values use original units, so large-scale metrics like area can dominate the distance."
    if scaling == "percent":
        return "Percentage difference compares relative deviation from the proposed farm values."
    return "Z-score scaling gives each selected metric equal statistical weight using the historical fleet distribution."


def _add_turbine_power_from_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Infer historical turbine power from raw turbine model text when possible."""
    if "wind_farm_name" not in df.columns:
        return df

    try:
        raw_df = pd.read_csv(config.RAW_DATASET_PATH, sep=config.CSV_SEPARATOR)
    except Exception:
        return df

    if "wind_farm_name" not in raw_df.columns or "turbine_model" not in raw_df.columns:
        return df

    raw_df["inferred_turbine_power_MW"] = raw_df["turbine_model"].map(
        _infer_turbine_power_from_model
    )
    power_by_farm = (
        raw_df.dropna(subset=["inferred_turbine_power_MW"])
        .drop_duplicates(subset=["wind_farm_name"])
        .set_index("wind_farm_name")["inferred_turbine_power_MW"]
    )
    if power_by_farm.empty:
        return df

    if "turbine_power_MW" not in df.columns:
        df["turbine_power_MW"] = np.nan

    inferred_power = df["wind_farm_name"].map(power_by_farm)
    df["turbine_power_MW"] = pd.to_numeric(
        df["turbine_power_MW"],
        errors="coerce",
    ).fillna(inferred_power)
    return df
def _infer_turbine_power_from_model(model: object) -> float:
    """Infer MW rating from common offshore turbine model naming patterns."""
    if pd.isna(model):
        return np.nan

    model_text = str(model)
    values = [float(value) for value in re.findall(r"(\d+(?:\.\d+)?)\s*MW\b", model_text)]
    values.extend(float(value) for value in re.findall(r"\bSWT-(\d+(?:\.\d+)?)", model_text))
    values.extend(float(value) for value in re.findall(r"\bSG\s*(\d+(?:\.\d+)?)", model_text))
    values.extend(float(value) for value in re.findall(r"\bV\d+-(\d+(?:\.\d+)?)", model_text))

    repower_match = re.search(r"\bREpower\s+(\d+(?:\.\d+)?)M\b", model_text)
    if repower_match:
        values.append(float(repower_match.group(1)))

    plausible_values = [value for value in values if 1.0 <= value <= 20.0]
    if not plausible_values:
        return np.nan
    return float(np.mean(plausible_values))
