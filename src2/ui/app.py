"""Main UI application for offshore wind profitability analysis."""

import logging

import dash
import pandas as pd
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from waitress import serve

from .. import config
from ..calculations.calculations import budget_eur_from_unit_capex, calculate
from ..data_preprocessing.enrichment import add_environmental_columns, add_distance_from_port, add_distance_from_construction_port
from .components import create_input_form
from .clustering import create_closest_farms_table, create_clustering_panel, prepare_clustering_dataframe
from .profitability_map import profitability_map

LOGGER = logging.getLogger(__name__)

def create_app(df: pd.DataFrame, model):
    """Create and configure the Dash application."""
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    clustering_df = prepare_clustering_dataframe(df)

    app.layout = dbc.Container([
        dbc.Row(dbc.Col(html.H1("Offshore Wind Farm Profitability Analysis"), width=12)),
        dbc.Row([
            dbc.Col([
                dcc.Dropdown(
                    id='color-column-dropdown',
                    options=[
                        {'label': 'LCOE (Indexed) [EUR/MWh]', 'value': 'lcoe_eur_per_mwh_indexed'},
                        {'label': 'Unit CAPEX (Indexed) [EUR/MW]', 'value': 'unit_capex_eur_per_mw_indexed'},
                        {'label': 'Annual OPEX [EUR]', 'value': 'annual_opex_eur'},
                        {'label': 'Annual OPEX [EUR/MWh]', 'value': 'annual_opex_eur_per_mwh'},
                        {'label': 'Total Project Budget (Indexed) [EUR]', 'value': 'total_project_budget_eur_indexed'},
                    ],
                    value=config.TARGET_PROFITABILITY_COLUMN,
                    className="mb-3"
                ),
                dcc.Graph(id='map-graph', style={'height': '70vh'})
            ], width=9),
            dbc.Col(create_input_form(df), width=3)
        ]), 
        dbc.Row(dbc.Col(html.Div(id='prediction-output'))),
        dbc.Row(dbc.Col(create_clustering_panel(clustering_df))),
    ], fluid=True)

    @app.callback(
        [Output('map-graph', 'figure'),
         Output('prediction-output', 'children'),
         Output('cluster-output', 'children')],
        [Input('color-column-dropdown', 'value'),
         Input('predict-button', 'n_clicks'),
         Input('cluster-metrics-checklist', 'value'),
         Input('cluster-scaling-dropdown', 'value')],
        [State('lat-input', 'value'), State('lon-input', 'value'),
         State('area-input', 'value'), State('country-dropdown', 'value'),
         State('year-input', 'value'), State('turbine-power-input', 'value'),
         State('producer-dropdown', 'value'), State('foundation-dropdown', 'value'),
         State('depth-min-input', 'value'), State('depth-max-input', 'value'),
         State('shore-min-input', 'value'), State('shore-max-input', 'value'),
         State('lifetime-input', 'value'), State('capacity-input', 'value')]
    )
    def update_map(color_column, n_clicks, cluster_metrics, cluster_scaling, lat, lon, area, country, year, turbine_power, producer, foundation, depth_min, depth_max, shore_min, shore_max, lifetime, capacity):
        """Update the map based on dropdown and prediction."""

        # Prepare new sample for prediction
        new_sample = pd.DataFrame([{
            "wind_farm_name": "Proposed Wind Farm",
            "LAT": lat, "LON": lon, "area_sqkm": area, "country": country,
            "commissioning_year": year, "turbine_power_MW": turbine_power,
            "turbine_producer": producer, "foundation_type": foundation,
            "water_depth_min_m": depth_min, "water_depth_max_m": depth_max,
            "distance_from_shore_min_km": shore_min, "distance_from_shore_max_km": shore_max,
            "project_lifetime_years": lifetime, "installed_capacity_MW": capacity,
        }])

        # Calculate mean values
        new_sample["water_depth_mean_m"] = new_sample[["water_depth_min_m", "water_depth_max_m"]].mean(axis=1)
        new_sample["distance_from_shore_mean_km"] = new_sample[["distance_from_shore_min_km", "distance_from_shore_max_km"]].mean(axis=1)

        # Enrich grid connection info
        conn_details = config.get_connection_details(country, year)
        for key, value in conn_details.items():
            new_sample[key] = value

        # Enrich with environmental data
        new_sample = add_environmental_columns(new_sample)
        new_sample = add_water_depth(new_sample)
        new_sample = add_distance_from_shore(new_sample)
        new_sample = add_distance_from_port(new_sample)
        new_sample = add_distance_from_construction_port(new_sample)

        # One-hot encode categorical features
        for col in config.CATEGORICAL_COLUMNS:
            for val in df[col].unique():
                new_sample[f"{col}_{val}"] = (new_sample[col] == val).astype(int)

        # Select feature columns for prediction
        feature_columns = config.FEATURE_COLUMNS.copy()
        for col in config.CATEGORICAL_COLUMNS:
            feature_columns.extend([c for c in df.columns if c.startswith(f"{col}_")])

        new_sample_features = new_sample[feature_columns].fillna(0)

        # Predict and calculate profitability
        prediction = model.predict(new_sample_features)[0]
        new_sample[config.TARGET_COLUMN] = prediction

        if (config.TARGET_COLUMN == "unit_capex_eur_per_mw_indexed"
            or config.TARGET_COLUMN == "unit_capex_eur_per_mw"):
            new_sample["total_project_budget_eur_indexed"] = budget_eur_from_unit_capex(prediction, new_sample["installed_capacity_MW"].iloc[0])

        new_sample["total_project_budget_eur"] = new_sample["total_project_budget_eur_indexed"]
        new_sample_calculated = calculate(new_sample)
        clustering_table = create_closest_farms_table(
            clustering_df,
            new_sample_calculated.iloc[0],
            cluster_metrics,
            cluster_scaling,
        )

        # Combine historical and new data
        combined_df = pd.concat([df, new_sample_calculated], ignore_index=True)

        output_div = html.Div([
            html.H4("Prediction Results", className="mt-4"),
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardHeader("Total Project Budget (Indexed)"),
                    dbc.CardBody(f"{new_sample_calculated['total_project_budget_eur_indexed'].iloc[0]:,.2f} EUR")
                ]), width=3),
                dbc.Col(dbc.Card([
                    dbc.CardHeader("Unit CAPEX (Indexed)"),
                    dbc.CardBody(f"{new_sample_calculated['unit_capex_eur_per_mw_indexed'].iloc[0]:,.2f} EUR/MW")
                ]), width=3),
                dbc.Col(dbc.Card([
                    dbc.CardHeader("Annual OPEX"),
                    dbc.CardBody(f"{new_sample_calculated['annual_opex_eur'].iloc[0]:,.2f} EUR")
                ]), width=3),
                dbc.Col(dbc.Card([
                    dbc.CardHeader("LCOE (Indexed)"),
                    dbc.CardBody(f"{new_sample_calculated['lcoe_eur_per_mwh_indexed'].iloc[0]:,.2f} EUR/MWh")
                ]), width=3),
            ]),
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardHeader("Distance from Port (shapefile)"),
                    dbc.CardBody(f"{new_sample_calculated['distance_from_port_km'].iloc[0]:,.1f} km")
                ]), width=3),
                dbc.Col(dbc.Card([
                    dbc.CardHeader("Distance from Construction Port"),
                    dbc.CardBody(f"{new_sample_calculated['distance_from_construction_port_km'].iloc[0]:,.1f} km")
                ]), width=3),
            ], className="mt-2"),
        ])

        return profitability_map(combined_df, color_column), output_div, clustering_table

    return app

def start_ui(df: pd.DataFrame, model):
    """Start the Dash UI."""
    app = create_app(df, model)
    serve(app.server, host="0.0.0.0", port=8050)
