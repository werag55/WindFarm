"""Dashboard for model results."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import math

def create_dashboard(metrics_comparison: dict, feature_importance: pd.DataFrame, best_y_data: dict, linear_coefs: dict = None, pdp_data: dict = None):
    """Create and show a dashboard with model results comparison, coefficients, and PDPs."""
    
    # Calculate number of rows needed
    num_base_rows = 4  # Base rows: metrics table, R2 comparison, feature importance, actual vs predicted
    num_linear_coef_plots = len(linear_coefs) if linear_coefs else 0
    
    # For PDPs, calculate rows needed for 2 columns
    num_pdp_plots = len(pdp_data) if pdp_data else 0
    num_pdp_rows = math.ceil(num_pdp_plots / 2) if num_pdp_plots > 0 else 0
    
    total_rows = num_base_rows + num_linear_coef_plots + num_pdp_rows
    
    # Build row weights: allocate more vertical space to coefficient rows when many features are shown
    row_heights = [2.0, 1.2, 1.2, 1.4]
    if linear_coefs:
        for _, coefs_df in linear_coefs.items():
            row_heights.append(1.2 + 0.18 * len(coefs_df))
    row_heights.extend([1.3] * num_pdp_rows)
    
    # Build specs for the subplot grid - all rows need 2 columns for consistency
    specs = [
        [{"type": "table"}, None],       # Metrics table (spanning both columns)
        [{"type": "bar"}, None],         # R2 comparison
        [{"type": "bar"}, None],         # Feature importance
        [{"type": "scatter"}, None]      # Actual vs Predicted
    ]
    
    # Add specs for linear model coefficients (single column each)
    for _ in range(num_linear_coef_plots):
        specs.append([{"type": "bar"}, None])
    
    # Add specs for PDP rows (2 columns each)
    for _ in range(num_pdp_rows):
        specs.append([{"type": "scatter"}, {"type": "scatter"}])
    
    subplot_titles = [
        "All Models: Metrics Comparison",
        "Model Comparison (R2 Score)",
        "Best Model: Feature Importance",
        "Best Model: Actual vs Predicted",
    ]
    
    # Add titles for linear coefficients
    if linear_coefs:
        subplot_titles.extend([f"{name} Model Coefficients" for name in linear_coefs.keys()])
    
    # Add title for PDP
    if pdp_data:
        subplot_titles.append("Partial Dependence Plots (Best Model)")
    
    fig = make_subplots(
        rows=total_rows, 
        cols=2,
        subplot_titles=subplot_titles,
        specs=specs,
        vertical_spacing=0.055,
        horizontal_spacing=0.12,
        row_heights=row_heights
    )

    models = list(metrics_comparison.keys())
    
    # All Metrics Table
    metric_keys = ['r2', 'cv_r2_mean', 'rmse', 'mae', 'mape', 'median_ae']
    header_names = ['Model', 'Test R²', 'CV R² Mean', 'RMSE', 'MAE', 'MAPE', 'Median AE']
    
    cell_values = [models]
    for k in metric_keys:
        cell_values.append([f"{metrics_comparison[m][k]:.4f}" for m in models])

    fig.add_trace(go.Table(
        header=dict(
            values=header_names, 
            fill_color='paleturquoise', 
            align='left',
            font=dict(size=12, color='black')
        ),
        cells=dict(
            values=cell_values, 
            fill_color='lavender', 
            align='left',
            font=dict(size=11)
        )
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

    # Add linear model coefficients
    current_row = num_base_rows + 1
    if linear_coefs:
        colors = {'Lasso': 'rgb(255, 127, 14)', 'Ridge': 'rgb(44, 160, 44)', 'ElasticNet': 'rgb(214, 39, 40)'}
        for model_name, coefs_df in linear_coefs.items():
            # Show all coefficients
            display_coefs = coefs_df
            fig.add_trace(go.Bar(
                x=display_coefs['coefficient'],
                y=display_coefs['feature'],
                orientation='h',
                name=f"{model_name} Coefs",
                marker_color=colors.get(model_name, 'rgb(100, 100, 100)'),
                showlegend=False
            ), row=current_row, col=1)
            
            fig.update_xaxes(title_text="Coefficient Value", row=current_row, col=1)
            fig.update_yaxes(
                title_text="Feature",
                row=current_row,
                col=1,
                tickmode="array",
                tickvals=display_coefs['feature'].tolist(),
                ticktext=display_coefs['feature'].tolist(),
                automargin=True,
                tickfont=dict(size=9)
            )
            current_row += 1

    # Add PDPs
    if pdp_data:
        pdp_row = num_base_rows + num_linear_coef_plots + 1
        pdp_col = 1
        
        for feat_name, pd_values in sorted(pdp_data.items()):
            fig.add_trace(go.Scatter(
                x=pd_values['values'],
                y=pd_values['pd'],
                mode='lines+markers',
                name=f"{feat_name}",
                line=dict(width=2),
                showlegend=False
            ), row=pdp_row, col=pdp_col)
            
            fig.update_xaxes(title_text=feat_name, row=pdp_row, col=pdp_col, title_font=dict(size=10))
            fig.update_yaxes(title_text="Partial Dependence", row=pdp_row, col=pdp_col, title_font=dict(size=10))
            
            pdp_col += 1
            if pdp_col > 2:
                pdp_col = 1
                pdp_row += 1

    # Update layout
    coef_height = 0
    if linear_coefs:
        coef_height = sum(140 + 20 * len(coefs_df) for _, coefs_df in linear_coefs.items())
    pdp_height = num_pdp_rows * 320
    height = 850 + coef_height + pdp_height
    fig.update_layout(
        title_text="Model Results & Comparison Dashboard",
        height=height,
        showlegend=False,
        margin=dict(l=260, r=50, t=80, b=50)
    )
    
    fig.update_xaxes(title_text="Models", row=2, col=1, title_font=dict(size=10))
    fig.update_yaxes(title_text="R² Score", row=2, col=1, title_font=dict(size=10))
    fig.update_xaxes(title_text="Features", row=3, col=1, title_font=dict(size=10))
    fig.update_yaxes(title_text="Importance", row=3, col=1, title_font=dict(size=10))
    fig.update_xaxes(title_text="Actual", row=4, col=1, title_font=dict(size=10))
    fig.update_yaxes(title_text="Predicted", row=4, col=1, title_font=dict(size=10))

    fig.show()
