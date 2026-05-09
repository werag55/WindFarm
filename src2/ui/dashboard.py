"""Dashboard for model results."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

def create_dashboard(metrics_comparison: dict, feature_importance: pd.DataFrame, best_y_data: dict):
    """Create and show a dashboard with model results comparison."""
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=("All Models: Metrics Comparison", "Model Comparison (R2 Score)", "Best Model: Feature Importance", "Best Model: Actual vs Predicted"),
        specs=[[{"type": "table"}], [{"type": "bar"}], [{"type": "bar"}], [{"type": "scatter"}]]
    )

    models = list(metrics_comparison.keys())
    
    # All Metrics Table
    metric_keys = ['r2', 'cv_r2_mean', 'rmse', 'mae', 'mape', 'median_ae']
    header_names = ['Model', 'Test R²', 'CV R² Mean', 'RMSE', 'MAE', 'MAPE', 'Median AE']
    
    cell_values = [models]
    for k in metric_keys:
        cell_values.append([f"{metrics_comparison[m][k]:.4f}" for m in models])

    fig.add_trace(go.Table(
        header=dict(values=header_names, fill_color='paleturquoise', align='left'),
        cells=dict(values=cell_values, fill_color='lavender', align='left')
    ), row=1, col=1)

    # Model Comparison Chart (R2)
    r2_scores = [metrics_comparison[m]['r2'] for m in models]
    fig.add_trace(go.Bar(
        x=models, y=r2_scores, name='Test R² Score', marker_color='rgb(55, 83, 109)'
    ), row=2, col=1)

    # Feature Importance Bar Chart
    sorted_features = feature_importance.sort_values(by='importance', ascending=False).head(20) # Top 20
    fig.add_trace(go.Bar(
        x=sorted_features['feature'],
        y=sorted_features['importance'],
        name='Feature Importance', marker_color='rgb(26, 118, 255)'
    ), row=3, col=1)
    
    # Actual vs Predicted Scatter
    y_test = best_y_data['y_test']
    y_pred = best_y_data['y_pred']
    fig.add_trace(go.Scatter(
        x=y_test, y=y_pred, mode='markers',
        name='Actual vs Predicted',
        marker=dict(color='rgba(135, 206, 250, 0.5)', line=dict(color='MediumPurple', width=1))
    ), row=4, col=1)

    # Ideal line for Actual vs Predicted
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val], mode='lines',
        name='Ideal Prediction', line=dict(color='red', dash='dash')
    ), row=4, col=1)

    fig.update_layout(
        title_text="Model Results & Comparison Dashboard",
        height=1600,
        showlegend=False
    )
    fig.update_xaxes(title_text="Models", row=2, col=1)
    fig.update_yaxes(title_text="R² Score", row=2, col=1)
    fig.update_xaxes(title_text="Features", row=3, col=1)
    fig.update_yaxes(title_text="Importance (Absolute)", row=3, col=1)
    fig.update_xaxes(title_text="Actual Value", row=4, col=1)
    fig.update_yaxes(title_text="Predicted Value", row=4, col=1)

    fig.show()
