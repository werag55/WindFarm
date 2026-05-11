"""Components for the Dash UI."""


import dash_bootstrap_components as dbc
from dash import dcc

def create_input_form(df):
    """Create the input form for new predictions."""
    return dbc.Form([
        dbc.Row([
            dbc.Col(dbc.Label("LAT"), width=4),
            dbc.Col(dcc.Input(id='lat-input', type='number', value=56.0), width=8),
        ], className="mb-2"),
        dbc.Row([
            dbc.Col(dbc.Label("LON"), width=4),
            dbc.Col(dcc.Input(id='lon-input', type='number', value=10.0), width=8),
        ], className="mb-2"),
        dbc.Row([
            dbc.Col(dbc.Label("Area (sqkm)"), width=4),
            dbc.Col(dcc.Input(id='area-input', type='number', value=50.0), width=8),
        ], className="mb-2"),
        dbc.Row([
            dbc.Col(dbc.Label("Country"), width=4),
            dbc.Col(dcc.Dropdown(id='country-dropdown', options=[{'label': i, 'value': i} for i in df['country'].unique()], value='Denmark'), width=8),
        ], className="mb-2"),
        dbc.Row([
            dbc.Col(dbc.Label("Commissioning Year"), width=4),
            dbc.Col(dcc.Input(id='year-input', type='number', value=2028), width=8),
        ], className="mb-2"),
        dbc.Row([
            dbc.Col(dbc.Label("Turbine Power (MW)"), width=4),
            dbc.Col(dcc.Input(id='turbine-power-input', type='number', value=10.0), width=8),
        ], className="mb-2"),
        dbc.Row([
            dbc.Col(dbc.Label("Turbine Producer"), width=4),
            dbc.Col(dcc.Dropdown(id='producer-dropdown', options=[{'label': i, 'value': i} for i in df['turbine_producer'].unique()], value='Vestas'), width=8),
        ], className="mb-2"),
        dbc.Row([
            dbc.Col(dbc.Label("Foundation Type"), width=4),
            dbc.Col(dcc.Dropdown(id='foundation-dropdown', options=[{'label': i, 'value': i} for i in df['foundation_type'].unique()], value='Monopile'), width=8),
        ], className="mb-2"),
        dbc.Row([
            dbc.Col(dbc.Label("Lifetime (years)"), width=4),
            dbc.Col(dcc.Input(id='lifetime-input', type='number', value=25), width=8),
        ], className="mb-2"),
        dbc.Row([
            dbc.Col(dbc.Label("Installed Capacity (MW)"), width=4),
            dbc.Col(dcc.Input(id='capacity-input', type='number', value=100.0), width=8),
        ], className="mb-2"),
        dbc.Button('Predict', id='predict-button', n_clicks=0, className="w-100")
    ])
