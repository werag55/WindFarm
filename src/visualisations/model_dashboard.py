"""Model result visualization utilities."""

from typing import Dict

import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp


def create_model_dashboard(report: Dict, output_path: str) -> None:
    """Create and save a multi-panel HTML dashboard for the LCOE model."""
    fig = sp.make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "LCOE Model - Feature Importance",
            "LCOE Model - Train vs Test Metrics",
            "LCOE Model Summary",
            "New Location Prediction",
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "table"}, {"type": "table"}],
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.15,
        row_heights=[0.5, 0.5],
    )

    # Row 1, Col 1: Feature Importance
    lcoe_coefs = report["lcoe_model"]["coefficients"]
    lcoe_features = list(lcoe_coefs.keys())
    lcoe_values = list(lcoe_coefs.values())
    fig.add_trace(
        go.Bar(
            y=lcoe_features,
            x=lcoe_values,
            orientation="h",
            marker_color="darkorange",
            name="LCOE",
            showlegend=False,
            hovertemplate="<b>LCOE Model</b><br>Feature: %{y}<br>Coefficient: %{x:.6f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Row 1, Col 2: Train vs Test Metrics
    metrics_names = ["RMSE", "MAE", "R²"]
    lcoe_train = [
        report["lcoe_model"]["train_metrics"]["rmse"],
        report["lcoe_model"]["train_metrics"]["mae"],
        report["lcoe_model"]["train_metrics"]["r2"],
    ]
    lcoe_test = [
        report["lcoe_model"]["test_metrics"]["rmse"],
        report["lcoe_model"]["test_metrics"]["mae"],
        report["lcoe_model"]["test_metrics"]["r2"],
    ]

    x = np.arange(len(metrics_names))
    width = 0.35
    fig.add_trace(
        go.Bar(
            name="Train",
            x=x - width / 2,
            y=lcoe_train,
            width=width,
            marker_color="lightyellow",
            showlegend=True,
            hovertemplate="<b>LCOE Model - Train</b><br>Metric: %{customdata}<br>Value: %{y:.4f}<extra></extra>",
            customdata=metrics_names,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            name="Test",
            x=x + width / 2,
            y=lcoe_test,
            width=width,
            marker_color="darkorange",
            showlegend=True,
            hovertemplate="<b>LCOE Model - Test</b><br>Metric: %{customdata}<br>Value: %{y:.4f}<extra></extra>",
            customdata=metrics_names,
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(tickvals=x, ticktext=metrics_names, row=1, col=2)

    # Row 2, Col 1: LCOE Model Summary
    lcoe_metrics = report["lcoe_model"]["train_metrics"]
    summary_text = f"""
<b>LCOE Prediction Model</b><br>
Rows used: {lcoe_metrics['rows_used']}<br>
Train RMSE: {lcoe_metrics['rmse']:.2f} EUR/MWh<br>
Train R²: {lcoe_metrics['r2']:.4f}<br>
Intercept: {report['lcoe_model']['intercept']:.2f}
"""
    fig.add_trace(
        go.Table(
            cells=dict(
                values=[[summary_text]],
                align="left",
                font=dict(size=12),
            )
        ),
        row=2,
        col=1,
    )

    # Row 2, Col 2: New Location Prediction
    prediction = report["new_location_prediction"]
    prediction_text = f"""
<b>New Location Prediction</b><br>
Mean wind speed: {prediction['mean_wind_speed_mps']:.3f} m/s<br>
Mean wave height: {prediction.get('mean_wave_height_m', 'N/A'):.3f} m<br>
Predicted LCOE: {prediction['predicted_lcoe_eur_per_mwh']:.2f} EUR/MWh
"""
    fig.add_trace(
        go.Table(
            cells=dict(
                values=[[prediction_text]],
                align="left",
                font=dict(size=12),
            )
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        title_text="Offshore Wind LCOE Model Dashboard",
        height=800,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.write_html(output_path)
