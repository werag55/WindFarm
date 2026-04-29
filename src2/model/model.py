"""Model training and evaluation."""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import logging

from .. import config

LOGGER = logging.getLogger(__name__)

def train_model(df: pd.DataFrame):
    """Train a linear regression model."""
    LOGGER.info("Training model...")

    # Add one-hot encoded columns to features
    feature_columns = config.FEATURE_COLUMNS.copy()
    for col in config.CATEGORICAL_COLUMNS:
        feature_columns.extend([c for c in df.columns if c.startswith(f"{col}_")])

    # Drop rows with missing target values
    df = df.dropna(subset=[config.TARGET_COLUMN])
    df = df.fillna(0)

    X = df[feature_columns]
    y = df[config.TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    
    metrics = {
        "mse": mean_squared_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred)
    }
    
    feature_importance = pd.DataFrame({'feature': X.columns, 'importance': model.coef_})

    LOGGER.info("Model training complete. Mean Squared Error: %f", metrics["mse"])

    return model, metrics, feature_importance
