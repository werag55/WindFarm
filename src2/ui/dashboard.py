"""Dashboard for model results."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


METRIC_KEYS = ["r2", "cv_r2_mean", "rmse", "mae", "mape", "median_ae"]
HEADER_NAMES = ["Model", "Test R²", "CV R² Mean", "RMSE", "MAE", "MAPE", "Median AE"]


def create_dashboard(metrics_comparison: dict, feature_importance: pd.DataFrame,
                     best_y_data: dict, variants: dict | None = None) -> None:
    """Render the full results dashboard."""
    has_variants = bool(variants and variants.get("variants"))
    n_rows = 5 if has_variants else 4
    titles = [
        "All Models: Metrics Comparison",
        "Model Comparison (R² Score)",
        "Best Model: Feature Importance",
        "Best Model: Actual vs Predicted (with bootstrap ±1σ)",
    ]
    specs = [
        [{"type": "table"}],
        [{"type": "bar"}],
        [{"type": "bar"}],
        [{"type": "scatter"}],
    ]
    if has_variants:
        cutoff_year = variants.get("cutoff_year")
        titles.append(f"Data-age variant comparison (full vs recent ≥ {cutoff_year})")
        specs.append([{"type": "table"}])

    fig = make_subplots(rows=n_rows, cols=1, subplot_titles=titles, specs=specs)

    _add_metrics_table(fig, metrics_comparison, row=1)
    _add_r2_bars(fig, metrics_comparison, row=2)
    _add_feature_importance(fig, feature_importance, row=3)
    _add_actual_vs_predicted(fig, best_y_data, row=4)
    if has_variants:
        _add_variants_table(fig, variants, row=5)

    fig.update_layout(
        title_text="Model Results & Comparison Dashboard",
        height=1800 if has_variants else 1600,
        showlegend=False,
    )
    fig.update_xaxes(title_text="Models",   row=2, col=1)
    fig.update_yaxes(title_text="R² Score", row=2, col=1)
    fig.update_xaxes(title_text="Features", row=3, col=1)
    fig.update_yaxes(title_text="Importance (Absolute)", row=3, col=1)
    fig.update_xaxes(title_text="Actual Value",     row=4, col=1)
    fig.update_yaxes(title_text="Predicted Value",  row=4, col=1)

    fig.show()


def _add_metrics_table(fig, metrics_comparison: dict, row: int) -> None:
    models = list(metrics_comparison.keys())
    cell_values = [models]
    for k in METRIC_KEYS:
        cell_values.append([f"{metrics_comparison[m][k]:.4f}" for m in models])
    fig.add_trace(
        go.Table(
            header=dict(values=HEADER_NAMES, fill_color="paleturquoise", align="left"),
            cells=dict(values=cell_values, fill_color="lavender", align="left"),
        ),
        row=row, col=1,
    )


def _add_r2_bars(fig, metrics_comparison: dict, row: int) -> None:
    models = list(metrics_comparison.keys())
    r2_scores = [metrics_comparison[m]["r2"] for m in models]
    fig.add_trace(
        go.Bar(x=models, y=r2_scores, name="Test R² Score", marker_color="rgb(55, 83, 109)"),
        row=row, col=1,
    )


def _add_feature_importance(fig, feature_importance: pd.DataFrame, row: int) -> None:
    sorted_features = feature_importance.sort_values(by="importance", ascending=False).head(20)
    fig.add_trace(
        go.Bar(
            x=sorted_features["feature"],
            y=sorted_features["importance"],
            name="Feature Importance",
            marker_color="rgb(26, 118, 255)",
        ),
        row=row, col=1,
    )


def _add_actual_vs_predicted(fig, best_y_data: dict, row: int) -> None:
    y_test = best_y_data["y_test"]
    y_pred = best_y_data["y_pred"]
    pred_std = best_y_data.get("pred_std")
    err_kwargs = {}
    if pred_std is not None:
        err_kwargs["error_y"] = dict(type="data", array=np.asarray(pred_std), visible=True, thickness=1)
    fig.add_trace(
        go.Scatter(
            x=y_test, y=y_pred, mode="markers",
            name=f"{best_y_data.get('model', 'Best')}: actual vs predicted",
            marker=dict(color="rgba(135, 206, 250, 0.6)", line=dict(color="MediumPurple", width=1)),
            **err_kwargs,
        ),
        row=row, col=1,
    )
    min_val = float(min(min(y_test), min(y_pred)))
    max_val = float(max(max(y_test), max(y_pred)))
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val], mode="lines",
            name="Ideal", line=dict(color="red", dash="dash"),
        ),
        row=row, col=1,
    )


def _add_variants_table(fig, variants: dict, row: int) -> None:
    """Side-by-side metrics for full vs recent training subsets, per model."""
    full = variants["variants"].get("full", {}).get("metrics_by_model", {})
    recent = variants["variants"].get("recent", {}).get("metrics_by_model", {})
    models = sorted(set(full) | set(recent))

    def fmt(metrics: dict, key: str) -> str:
        if not metrics:
            return "-"
        return f"{metrics.get(key, float('nan')):.4f}"

    headers = ["Model",
               "Full R²", "Recent R²",
               "Full RMSE", "Recent RMSE",
               "Full MAPE", "Recent MAPE"]
    rows = [models,
            [fmt(full.get(m, {}), "r2") for m in models],
            [fmt(recent.get(m, {}), "r2") for m in models],
            [fmt(full.get(m, {}), "rmse") for m in models],
            [fmt(recent.get(m, {}), "rmse") for m in models],
            [fmt(full.get(m, {}), "mape") for m in models],
            [fmt(recent.get(m, {}), "mape") for m in models]]
    fig.add_trace(
        go.Table(
            header=dict(values=headers, fill_color="paleturquoise", align="left"),
            cells=dict(values=rows, fill_color="lavender", align="left"),
        ),
        row=row, col=1,
    )
