"""Main UI application for offshore wind profitability analysis."""

from __future__ import annotations

import logging
from typing import Any

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import dcc, html
from dash.dependencies import Input, Output, State
from waitress import serve

from .. import config
from ..analysis._common import (
    enrich_sample,
    feature_columns_from_pipeline,
    predict_sample,
)
from ..calculations.calculations import calculate
from .components import create_input_form
from .profitability_map import profitability_map

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_eur(v):     return f"{v:,.0f} EUR"           if pd.notna(v) else "—"
def fmt_eur_mw(v):  return f"{v:,.0f} EUR/MW"        if pd.notna(v) else "—"
def fmt_eur_mwh(v): return f"{v:,.2f} EUR/MWh"       if pd.notna(v) else "—"
def fmt_mw(v):      return f"{v:,.1f} MW"            if pd.notna(v) else "—"
def fmt_km(v):      return f"{v:,.1f} km"            if pd.notna(v) else "—"
def fmt_m(v):       return f"{v:,.1f} m"             if pd.notna(v) else "—"
def fmt_mps(v):     return f"{v:,.2f} m/s"           if pd.notna(v) else "—"


def _card(header: str, body: str, *, color: str = "light", width: int = 3) -> dbc.Col:
    return dbc.Col(
        dbc.Card([dbc.CardHeader(header), dbc.CardBody(body)],
                 color=color, inverse=(color == "dark")),
        width=width,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_inputs(
    lat, lon, area, year, turbine_power, lifetime, capacity,
) -> list[str]:
    """Return list of human-readable errors (empty if all valid).

    Water depth and distance from shore are auto-computed and therefore not
    validated here.
    """
    errors = []

    def _missing(name, val):
        if val is None or (isinstance(val, float) and pd.isna(val)):
            errors.append(f"{name} is required")

    _missing("LAT", lat); _missing("LON", lon)
    _missing("Area",  area); _missing("Year", year)
    _missing("Turbine power", turbine_power)
    _missing("Capacity", capacity); _missing("Lifetime", lifetime)

    if errors:
        return errors

    if not (-90 <= lat <= 90):
        errors.append(f"LAT must be in [-90, 90]; got {lat}")
    if not (-180 <= lon <= 180):
        errors.append(f"LON must be in [-180, 180]; got {lon}")
    if capacity <= 0:
        errors.append("Installed capacity must be > 0")
    if turbine_power <= 0:
        errors.append("Turbine power must be > 0")
    if lifetime <= 0:
        errors.append("Project lifetime must be > 0")
    if area is not None and area <= 0:
        errors.append("Area must be > 0")
    if year is not None and not (1990 <= int(year) <= 2050):
        errors.append(f"Commissioning year must be in [1990, 2050]; got {year}")
    return errors


# ---------------------------------------------------------------------------
# Unknown-category detection
# ---------------------------------------------------------------------------

def _detect_unknown_categories(
    sample_row: dict[str, Any], training_df: pd.DataFrame
) -> list[str]:
    """Return human-readable warnings if a categorical value wasn't in training."""
    warnings = []
    for cat in config.CATEGORICAL_COLUMNS:
        value = sample_row.get(cat)
        if value is None or pd.isna(value):
            continue
        if cat not in training_df.columns:
            continue
        seen = set(training_df[cat].dropna().astype(str).unique())
        if str(value) not in seen:
            warnings.append(
                f"'{cat}' = '{value}' was not present in the training data — "
                f"encoded as all-zero one-hot (model fallback)."
            )
    return warnings


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(df: pd.DataFrame, model) -> dash.Dash:
    """Create and configure the Dash application."""
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    color_options = [
        {"label": "LCOE (Indexed) [EUR/MWh]",                   "value": "lcoe_eur_per_mwh_indexed"},
        {"label": "Annual CAPEX (Indexed) [EUR/MWh]",           "value": "annual_capex_eur_per_mwh_indexed"},
        {"label": "Annual OPEX [EUR/MWh]",                      "value": "annual_opex_eur_per_mwh"},
        {"label": "Total Project Budget (Indexed) [EUR]",       "value": "total_project_budget_eur_indexed"},
    ]

    app.layout = dbc.Container(
        [
            dbc.Row(dbc.Col(html.H1("Offshore Wind Farm Profitability Analysis"),
                            width=12), className="my-3"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dcc.Dropdown(
                                id="color-column-dropdown",
                                options=color_options,
                                value=config.TARGET_PROFITABILITY_COLUMN,
                                clearable=False, className="mb-3",
                            ),
                            dcc.Loading(
                                id="map-loading", type="default",
                                children=dcc.Graph(id="map-graph",
                                                   style={"height": "70vh"}),
                            ),
                        ],
                        width=9,
                    ),
                    dbc.Col(create_input_form(df), width=3),
                ]
            ),
            html.Div(id="validation-alert"),
            html.Div(id="warnings-alert"),
            dcc.Loading(
                id="prediction-loading", type="default",
                children=html.Div(id="prediction-output"),
            ),
        ],
        fluid=True,
    )

    # -----------------------------------------------------------------------
    # Callback
    # -----------------------------------------------------------------------
    @app.callback(
        [
            Output("map-graph",          "figure"),
            Output("prediction-output",  "children"),
            Output("validation-alert",   "children"),
            Output("warnings-alert",     "children"),
        ],
        [
            Input("color-column-dropdown", "value"),
            Input("predict-button",        "n_clicks"),
        ],
        [
            State("lat-input", "value"),  State("lon-input", "value"),
            State("area-input", "value"), State("country-dropdown", "value"),
            State("year-input", "value"), State("turbine-power-input", "value"),
            State("producer-dropdown", "value"), State("foundation-dropdown", "value"),
            State("lifetime-input", "value"), State("capacity-input", "value"),
        ],
    )
    def update_map(color_column, n_clicks,
                   lat, lon, area, country, year, turbine_power, producer,
                   foundation, lifetime, capacity):

        # --- Initial render: just show map without prediction ---
        if not n_clicks:
            return profitability_map(df, color_column), html.Div(), None, None

        # --- Validate ---
        errors = _validate_inputs(
            lat, lon, area, year, turbine_power, lifetime, capacity,
        )
        if errors:
            alert = dbc.Alert(
                [html.B("Cannot run prediction:"),
                 html.Ul([html.Li(e) for e in errors])],
                color="danger", className="mt-3",
            )
            return profitability_map(df, color_column), html.Div(), alert, None

        # --- Build sample row ---
        # Water depth / distance from shore / wave height / wind speed are left
        # blank and filled by enrich_sample() from GEBCO + coastline + GWA.
        sample_dict = {
            "wind_farm_name":              "New Sample",
            "LAT":                         float(lat),
            "LON":                         float(lon),
            "area_sqkm":                   float(area),
            "country":                     country,
            "commissioning_year":          int(year),
            "turbine_power_MW":            float(turbine_power),
            "turbine_producer":            producer,
            "foundation_type":             foundation,
            "project_lifetime_years":      int(lifetime),
            "installed_capacity_MW":       float(capacity),
        }

        new_sample = pd.DataFrame([sample_dict])
        LOGGER.info(
            "UI predict: building sample at lat=%.4f lon=%.4f (%s, %s, %d MW)",
            float(lat), float(lon), country, foundation, float(capacity),
        )

        # --- Detect unknown categorical values ---
        category_warnings = _detect_unknown_categories(sample_dict, df)

        # --- Enrich + predict using the same logic as the analysis modules ---
        try:
            enriched = enrich_sample(new_sample, df)
            feats = feature_columns_from_pipeline(model, df)
            pred = predict_sample(model, enriched, feats)
        except Exception as exc:
            LOGGER.exception("Prediction failed")
            err = dbc.Alert(f"Prediction failed: {exc}", color="danger",
                            className="mt-3")
            return profitability_map(df, color_column), html.Div(), err, None

        # --- Run full LCOE calculation for the map row ---
        enriched["total_project_budget_eur"]         = pred["predicted_capex_eur"]
        enriched["total_project_budget_eur_indexed"] = pred["predicted_capex_eur"]
        new_calc = calculate(enriched)

        combined_df = pd.concat([df, new_calc], ignore_index=True, sort=False)

        # --- Build cards ---
        prediction_cards = _build_prediction_cards(pred, new_calc)

        warnings_div = None
        if category_warnings:
            warnings_div = dbc.Alert(
                [html.B("Notes about categorical inputs:"),
                 html.Ul([html.Li(w) for w in category_warnings])],
                color="warning", className="mt-3",
            )

        return (profitability_map(combined_df, color_column),
                prediction_cards, None, warnings_div)

    return app


# ---------------------------------------------------------------------------
# Output cards
# ---------------------------------------------------------------------------

def _build_prediction_cards(pred: dict[str, float], calc_df: pd.DataFrame) -> html.Div:
    row = calc_df.iloc[0]

    return html.Div(
        [
            html.H4("Prediction results", className="mt-4"),

            dbc.Row(
                [
                    _card("Predicted CAPEX",         fmt_eur(pred["predicted_capex_eur"]),         color="primary"),
                    _card("CAPEX per MW",            fmt_eur_mw(pred["capex_eur_per_mw"])),
                    _card("Annual CAPEX",            fmt_eur(pred["annual_capex_eur"])),
                    _card("Annual OPEX",             fmt_eur(pred["annual_opex_eur"])),
                ],
                className="mt-2",
            ),
            dbc.Row(
                [
                    _card("Annual energy",
                          f"{pred['annual_energy_mwh']:,.0f} MWh"),
                    _card("LCOE",
                          fmt_eur_mwh(pred["lcoe_eur_per_mwh"]), color="success"),
                    _card("Annual CAPEX (indexed)",
                          fmt_eur_mwh(row.get("annual_capex_eur_per_mwh_indexed"))),
                    _card("Annual OPEX",
                          fmt_eur_mwh(row.get("annual_opex_eur_per_mwh"))),
                ],
                className="mt-2",
            ),
            dbc.Row(
                [
                    _card("Mean wind speed",
                          fmt_mps(row.get("mean_wind_speed_mps"))),
                    _card("Capacity factor",
                          f"{row.get('capacity_factor', float('nan')):.1%}"
                          if pd.notna(row.get("capacity_factor")) else "—"),
                    _card("Nearest service port",
                          f"{row.get('nearest_port_name', '?')} "
                          f"({fmt_km(row.get('distance_from_port_km'))})"),
                    _card("Nearest construction port",
                          f"{row.get('nearest_construction_port_name', '?')} "
                          f"({fmt_km(row.get('distance_from_construction_port_km'))})"),
                ],
                className="mt-2",
            ),
            dbc.Row(
                [
                    _card(
                        "Water depth (auto)",
                        f"{fmt_m(row.get('water_depth_m'))} "
                        f"[{row.get('water_depth_source', '?')}]",
                    ),
                    _card(
                        "Distance from shore (auto)",
                        f"{fmt_km(row.get('distance_from_shore_km'))} "
                        f"[{row.get('distance_from_shore_source', '?')}]",
                    ),
                    _card(
                        "Mean wave height (auto)",
                        f"{fmt_m(row.get('mean_wave_height_m'))} "
                        f"[{row.get('wave_height_source', '?')}]",
                    ),
                    _card("Data quality flag",
                          str(row.get("water_depth_quality_flag", "—"))),
                ],
                className="mt-2",
            ),
        ]
    )


def start_ui(df: pd.DataFrame, model) -> None:
    """Start the Dash UI via waitress (production WSGI server)."""
    app = create_app(df, model)
    LOGGER.info("Starting UI on http://0.0.0.0:8050")
    serve(app.server, host="0.0.0.0", port=8050)