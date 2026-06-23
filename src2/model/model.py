"""Model training and evaluation."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.linear_model import LinearRegression, Ridge, LassoCV, ElasticNetCV
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error, median_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.inspection import partial_dependence
import logging

from .. import config

LOGGER = logging.getLogger(__name__)

def remove_outliers_iqr(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Remove outliers based on IQR."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def train_model(df: pd.DataFrame):
    """Train and evaluate multiple regression models."""
    LOGGER.info("Training models...")

    # Add one-hot encoded columns to features
    feature_columns = config.FEATURE_COLUMNS.copy()
    for col in config.CATEGORICAL_COLUMNS:
        feature_columns.extend([c for c in df.columns if c.startswith(f"{col}_")])

    # Drop rows with missing target values and remove outliers
    df = df.dropna(subset=[config.TARGET_COLUMN])
    df = remove_outliers_iqr(df, config.TARGET_COLUMN)
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    df[feature_columns] = imputer.fit_transform(df[feature_columns])
    df = df.fillna(0)

    X = df[feature_columns]
    y = df[config.TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = {
        #"LinearRegression": TransformedTargetRegressor(regressor=LinearRegression(), transformer=StandardScaler()),
        "Ridge": TransformedTargetRegressor(regressor=Ridge(), transformer=StandardScaler()),
        "Lasso": TransformedTargetRegressor(
            regressor=LassoCV(alphas=np.logspace(-3, 3, 10), max_iter=50000, random_state=42),
            transformer=StandardScaler(),
        ),
        "ElasticNet": TransformedTargetRegressor(
                regressor=ElasticNetCV(
                l1_ratio=[.1, .5, .7, .9, .95, .99, 1],
                alphas=np.logspace(-3, 3, 10),
                max_iter=50000,
                random_state=42
            ),
            transformer=StandardScaler()
        ),
        "RandomForest": RandomForestRegressor(random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42)
    }

    results = {}
    best_r2 = -float("inf")
    best_model_name = None
    best_model = None
    linear_models_data = {}  # Track best Lasso, Ridge, ElasticNet

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        # Evaluate using CV
        cv_scores = cross_validate(model, X_train, y_train, cv=kf, scoring='r2')
        
        # Train on full train set and evaluate on test set
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        metrics = {
            "cv_r2_mean": np.mean(cv_scores['test_score']),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "r2": r2,
            "mae": mean_absolute_error(y_test, y_pred),
            "mape": mean_absolute_percentage_error(y_test, y_pred),
            "median_ae": median_absolute_error(y_test, y_pred)
        }
        
        results[name] = {"model": model, "metrics": metrics, "y_pred": y_pred}
        
        # Track best linear models for coefficient extraction
        if name in ["Lasso", "Ridge", "ElasticNet"]:
            linear_models_data[name] = {
                "model": model,
                "r2": r2,
                "metrics": metrics
            }
        
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_model = model

    LOGGER.info("Best model: %s with R2: %f", best_model_name, best_r2)

    # Feature importance for best model
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
    elif hasattr(best_model, 'coef_'):
        importances = best_model.coef_
    elif hasattr(best_model, 'regressor_') and hasattr(best_model.regressor_, 'coef_'):
        importances = best_model.regressor_.coef_
    else:
        importances = np.zeros(len(X.columns))
        
    feature_importance = pd.DataFrame({'feature': X.columns, 'importance': np.abs(importances)})

    # Extract coefficients from best Lasso, Ridge, ElasticNet
    linear_coefs = {}
    for model_name in ["Lasso", "Ridge", "ElasticNet"]:
        if model_name in linear_models_data:
            model = linear_models_data[model_name]["model"]
            if hasattr(model, 'coef_'):
                coefs = model.coef_
            elif hasattr(model, 'regressor_') and hasattr(model.regressor_, 'coef_'):
                coefs = model.regressor_.coef_
            else:
                coefs = np.zeros(len(X.columns))
            
            linear_coefs[model_name] = pd.DataFrame({
                'feature': X.columns,
                'coefficient': coefs
            }).sort_values(by='coefficient', key=abs, ascending=False)

    # Generate partial dependence plots data for best model
    pdp_data = {}
    try:
        # Get top 8 features for PDP (by absolute importance)
        top_features = feature_importance.nlargest(8, 'importance')['feature'].tolist()
        feature_indices = [list(X.columns).index(feat) for feat in top_features]
        
        for feat_idx, feat_name in zip(feature_indices, top_features):
            pd_result = partial_dependence(best_model, X_test, feat_idx, grid_resolution=20)
            pdp_data[feat_name] = {
                'values': pd_result['grid_values'][0],
                'pd': pd_result['average'][0]
            }
    except Exception as e:
        LOGGER.warning("Could not generate PDP: %s", str(e))

    # Package results for dashboard
    comparison_metrics = {name: res["metrics"] for name, res in results.items()}
    best_y_data = {"y_test": y_test, "y_pred": results[best_model_name]["y_pred"]}

    return best_model, comparison_metrics, feature_importance, best_y_data, linear_coefs, pdp_data, X_train, X_test
