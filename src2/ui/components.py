"""Form components for the Dash UI."""

from __future__ import annotations

import dash_bootstrap_components as dbc
import pandas as pd
from dash import dcc, html


# ---------------------------------------------------------------------------
# Reusable input row
# ---------------------------------------------------------------------------

def _row(label: str, input_component) -> dbc.Row:
    return dbc.Row(
        [
            dbc.Col(dbc.Label(label), width=5),
            dbc.Col(input_component, width=7),
        ],
        className="mb-2",
    )


def _num(input_id: str, value, *, mn=None, mx=None, step=None):
    kwargs = dict(id=input_id, type="number", value=value, debounce=True,
                  className="form-control form-control-sm")
    if mn is not None: kwargs["min"]  = mn
    if mx is not None: kwargs["max"]  = mx
    if step is not None: kwargs["step"] = step
    return dcc.Input(**kwargs)


# ---------------------------------------------------------------------------
# Public form factory
# ---------------------------------------------------------------------------

def create_input_form(df: pd.DataFrame) -> dbc.Form:
    """Create the input form for new predictions, with bounded numeric inputs."""

    def _opts(col: str, extras: list[str] | None = None) -> list[dict]:
        values = sorted(set(df[col].dropna().astype(str).unique()))
        if extras:
            values = sorted(set(values) | set(extras))
        return [{"label": v, "value": v} for v in values]

    return dbc.Form(
        [
            html.H5("Project parameters", className="mt-2 mb-3"),

            _row("LAT [°]",            _num("lat-input", 56.0, mn=-90,  mx=90,  step=0.1)),
            _row("LON [°]",            _num("lon-input", 10.0, mn=-180, mx=180, step=0.1)),
            _row("Area [km²]",         _num("area-input", 50.0, mn=0.1, step=1)),
            _row("Country",
                 dcc.Dropdown(id="country-dropdown",
                              options=_opts("country",
                                            ["Estonia", "Latvia", "Lithuania",
                                             "Ireland", "Finland"]),
                              value="Denmark", clearable=False)),
            _row("Commissioning year", _num("year-input", 2028, mn=1990, mx=2050, step=1)),
            _row("Turbine power [MW]", _num("turbine-power-input", 10.0, mn=0.1,  step=0.5)),
            _row("Turbine producer",
                 dcc.Dropdown(id="producer-dropdown",
                              options=_opts("turbine_producer", ["Unknown"]),
                              value="Vestas", clearable=False)),
            _row("Foundation type",
                 dcc.Dropdown(id="foundation-dropdown",
                              options=_opts("foundation_type", ["Unknown"]),
                              value="Monopile", clearable=False)),
            # NOTE: water depth and distance from shore are no longer asked for —
            # they're auto-computed from LAT/LON during enrichment (GEBCO bathymetry
            # + Natural Earth coastline). See enrichment.add_water_depth_columns /
            # add_distance_from_shore_columns.
            _row("Lifetime [years]",   _num("lifetime-input", 25, mn=1, mx=60,  step=1)),
            _row("Installed capacity [MW]",
                 _num("capacity-input", 800.0, mn=0.1, step=10)),

            html.Hr(),
            dbc.Button("Predict",
                       id="predict-button",
                       n_clicks=0,
                       color="primary",
                       className="w-100"),
        ]
    )