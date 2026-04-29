"""Dashboard for model results."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

def create_dashboard(metrics: dict, feature_importance: pd.DataFrame):
    """Create and show a dashboard with model results."""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Model Performance Metrics", "Feature Importance"),
        specs=[[{"type": "table"}], [{"type": "bar"}]]
    )

    # Metrics Table
    fig.add_trace(go.Table(
        header=dict(values=['Metric', 'Value'],
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[list(metrics.keys()), [f"{v:.4f}" for v in metrics.values()]],
                   fill_color='lavender',
                   align='left')),
        row=1, col=1
    )

    # Feature Importance Bar Chart
    sorted_features = feature_importance.sort_values(by='importance', ascending=False)
    fig.add_trace(go.Bar(
        x=sorted_features['feature'],
        y=sorted_features['importance'],
        name='Feature Importance'
    ), row=2, col=1)

    fig.update_layout(
        title_text="Model Results Dashboard",
        height=800,
        showlegend=False
    )
    fig.update_xaxes(title_text="Features", row=2, col=1)
    fig.update_yaxes(title_text="Importance (Coefficient)", row=2, col=1)

    fig.show()
