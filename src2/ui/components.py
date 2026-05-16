"""Components for the Dash UI."""


import dash_bootstrap_components as dbc
from dash import dcc

from .. import config

def create_input_form(df):
    """Create the input form for new predictions."""
    return dbc.Form([
        dbc.Row([
            dbc.Col(dbc.Label("LAT"), width=4),
            dbc.Col(dcc.Input(id='lat-input', type='number', value=config.DEFAULT_NEW_SAMPLE["LAT"]), width=8),
        ], className="mb-2"),
        dbc.Row([
            dbc.Col(dbc.Label("LON"), width=4),
            dbc.Col(dcc.Input(id='lon-input', type='number', value=config.DEFAULT_NEW_SAMPLE["LON"]), width=8),
        ], className="mb-2"),
        dbc.Row([
            dbc.Col(dbc.Label("Area (sqkm)"), width=4),
            dbc.Col(dcc.Input(id='area-input', type='number', value=config.DEFAULT_NEW_SAMPLE["area_sqkm"]), width=8),
        ], className="mb-2"),
        dbc.Row([
            dbc.Col(dbc.Label("Country"), width=4),
            dbc.Col(dcc.Dropdown(id='country-dropdown', options=[{'label': i, 'value': i} for i in df['country'].unique()], value=config.DEFAULT_NEW_SAMPLE["country"]), width=8),
        ], className="mb-2"),
        dbc.Row([
            dbc.Col(dbc.Label("Commissioning Year"), width=4),
            dbc.Col(dcc.Input(id='year-input', type='number', value=config.DEFAULT_NEW_SAMPLE["commissioning_year"]), width=8),
        ], className="mb-2"),
        dbc.Row([
            dbc.Col(dbc.Label("Turbine Power (MW)"), width=4),
            dbc.Col(dcc.Input(id='turbine-power-input', type='number', value=config.DEFAULT_NEW_SAMPLE["turbine_power_MW"]), width=8),
        ], className="mb-2"),
        dbc.Row([
            dbc.Col(dbc.Label("Turbine Producer"), width=4),
            dbc.Col(dcc.Dropdown(id='producer-dropdown', options=[{'label': i, 'value': i} for i in df['turbine_producer'].unique()], value=config.DEFAULT_NEW_SAMPLE["turbine_producer"]), width=8),
        ], className="mb-2"),
        dbc.Row([
            dbc.Col(dbc.Label("Foundation Type"), width=4),
            dbc.Col(dcc.Dropdown(id='foundation-dropdown', options=[{'label': i, 'value': i} for i in df['foundation_type'].unique()], value=config.DEFAULT_NEW_SAMPLE["foundation_type"]), width=8),
        ], className="mb-2"),
        dbc.Row([
            dbc.Col(dbc.Label("Water Depth Min (m)"), width=4),
            dbc.Col(dcc.Input(id='depth-min-input', type='number', value=config.DEFAULT_NEW_SAMPLE["water_depth_min_m"]), width=8),
        ], className="mb-2"),
        dbc.Row([
            dbc.Col(dbc.Label("Water Depth Max (m)"), width=4),
            dbc.Col(dcc.Input(id='depth-max-input', type='number', value=config.DEFAULT_NEW_SAMPLE["water_depth_max_m"]), width=8),
        ], className="mb-2"),
        dbc.Row([
            dbc.Col(dbc.Label("Distance Shore Min (km)"), width=4),
            dbc.Col(dcc.Input(id='shore-min-input', type='number', value=config.DEFAULT_NEW_SAMPLE["distance_from_shore_min_km"]), width=8),
        ], className="mb-2"),
        dbc.Row([
            dbc.Col(dbc.Label("Distance Shore Max (km)"), width=4),
            dbc.Col(dcc.Input(id='shore-max-input', type='number', value=config.DEFAULT_NEW_SAMPLE["distance_from_shore_max_km"]), width=8),
        ], className="mb-2"),
        dbc.Row([
            dbc.Col(dbc.Label("Lifetime (years)"), width=4),
            dbc.Col(dcc.Input(id='lifetime-input', type='number', value=config.DEFAULT_NEW_SAMPLE["project_lifetime_years"]), width=8),
        ], className="mb-2"),
        dbc.Row([
            dbc.Col(dbc.Label("Installed Capacity (MW)"), width=4),
            dbc.Col(dcc.Input(id='capacity-input', type='number', value=config.DEFAULT_NEW_SAMPLE["installed_capacity_MW"]), width=8),
        ], className="mb-2"),
        dbc.Button('Predict', id='predict-button', n_clicks=0, className="w-100")
    ])
