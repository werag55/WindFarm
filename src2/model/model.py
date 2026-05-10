"""Model training and evaluation."""

import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .. import config

LOGGER = logging.getLogger(__name__)


def remove_outliers_iqr(df: pd.DataFrame, column: str, k: float = 1.5) -> pd.DataFrame:
    """Remove rows whose `column` value lies outside [Q1 - k*IQR, Q3 + k*IQR]."""
    q1, q3 = df[column].quantile([0.25, 0.75])
    iqr = q3 - q1
    return df[(df[column] >= q1 - k * iqr) & (df[column] <= q3 + k * iqr)]


def _build_feature_list(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Return (all_features, numeric_features, onehot_features) that exist in df."""
    numeric = [c for c in config.FEATURE_COLUMNS if c in df.columns]

    onehot = [
        c
        for cat in config.CATEGORICAL_COLUMNS
        for c in df.columns
        if c.startswith(f"{cat}_")
    ]

    # Deduplicate while preserving order
    all_features = list(dict.fromkeys(numeric + onehot))

    missing = set(config.FEATURE_COLUMNS) - set(numeric)
    if missing:
        LOGGER.warning("Missing expected feature columns: %s", sorted(missing))

    return all_features, numeric, onehot


def _make_pipeline(estimator, numeric: list[str], onehot: list[str]) -> Pipeline:
    """Create preprocessing + estimator pipeline."""

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        ("sc", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "oh",
                Pipeline(
                    [
                        (
                            "cast",
                            FunctionTransformer(
                                lambda X: X.astype(float),
                                validate=False,
                            ),
                        ),
                        (
                            "imp",
                            SimpleImputer(
                                strategy="constant",
                                fill_value=0,
                            ),
                        ),
                    ]
                ),
                onehot,
            ),
        ],
        remainder="drop",
    )

    return Pipeline(
        [
            ("pre", preprocessor),
            ("est", estimator),
        ]
    )


def _compute_metrics(y_true, y_pred) -> dict:
    """Compute regression metrics."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        "median_ae": float(median_absolute_error(y_true, y_pred)),
    }


def train_model(df: pd.DataFrame):
    """Train and evaluate multiple regression models."""

    LOGGER.info("Training models on %d rows", len(df))

    df = df.copy()

    # Remove rows with missing target
    df = df.dropna(subset=[config.TARGET_COLUMN])

    LOGGER.info("After dropping NaN target: %d rows", len(df))

    # Build feature lists
    all_features, numeric, onehot = _build_feature_list(df)

    # Convert one-hot bool columns to float
    for col in onehot:
        df[col] = df[col].astype(float)

    LOGGER.info(
        "Using %d features (%d numeric + %d one-hot)",
        len(all_features),
        len(numeric),
        len(onehot),
    )

    LOGGER.info("Feature dtypes:\n%s", df[all_features].dtypes)

    X = df[all_features]
    y = df[config.TARGET_COLUMN]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    # Remove outliers only from training data
    train_df = X_train.copy()
    train_df["__y__"] = y_train.values

    train_df = remove_outliers_iqr(train_df, "__y__")

    LOGGER.info(
        "Train rows after IQR outlier removal: %d (was %d)",
        len(train_df),
        len(X_train),
    )

    y_train = train_df["__y__"]
    X_train = train_df.drop(columns="__y__")

    # Models
    estimators = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=42),
        "Lasso": Lasso(alpha=1.0, max_iter=20000, random_state=42),
        "ElasticNet": ElasticNet(
            alpha=1.0,
            max_iter=20000,
            random_state=42,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
        ),
    }

    # Cross-validation
    kf = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    results = {}

    best_cv_r2 = -np.inf
    best_name = None

    # Train and evaluate
    for name, estimator in estimators.items():

        LOGGER.info("Training %s...", name)

        pipeline = _make_pipeline(
            estimator,
            numeric,
            onehot,
        )

        cv = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=kf,
            scoring="r2",
            return_train_score=True,
            n_jobs=-1,
            error_score="raise",
        )

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        metrics = _compute_metrics(y_test, y_pred)

        metrics["cv_r2_mean"] = float(np.mean(cv["test_score"]))
        metrics["cv_r2_std"] = float(np.std(cv["test_score"]))
        metrics["cv_train_r2_mean"] = float(np.mean(cv["train_score"]))

        results[name] = {
            "model": pipeline,
            "metrics": metrics,
            "y_pred": y_pred,
        }

        LOGGER.info(
            "%s: cv_r2=%.3f ± %.3f | test_r2=%.3f | rmse=%.2e",
            name,
            metrics["cv_r2_mean"],
            metrics["cv_r2_std"],
            metrics["r2"],
            metrics["rmse"],
        )

        if metrics["cv_r2_mean"] > best_cv_r2:
            best_cv_r2 = metrics["cv_r2_mean"]
            best_name = name

    # Best model
    best_pipeline = results[best_name]["model"]

    LOGGER.info(
        "Best model: %s (cv R² = %.4f)",
        best_name,
        best_cv_r2,
    )

    # Permutation importance
    perm = permutation_importance(
        best_pipeline,
        X_test,
        y_test,
        n_repeats=20,
        random_state=42,
        n_jobs=-1,
    )

    feature_importance = pd.DataFrame(
        {
            "feature": all_features,
            "importance": perm.importances_mean,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    # Comparison metrics
    comparison_metrics = {
        name: result["metrics"]
        for name, result in results.items()
    }

    best_y_data = {
        "y_test": y_test,
        "y_pred": results[best_name]["y_pred"],
    }

    return (
        best_pipeline,
        comparison_metrics,
        feature_importance,
        best_y_data,
    )