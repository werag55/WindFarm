"""Data cleaning utilities for offshore wind project dataset."""

import logging

import numpy as np
import pandas as pd

from .. import config
from .foundations import consolidate_rare_foundation_types, normalize_foundation_type
from .imputation import impute_area_sqkm
from .parsing import parse_budget_to_eur, to_float_range_mean
from .turbines import split_turbine_model

LOGGER = logging.getLogger(__name__)


def prepare_cleaned_dataset(csv_path: str) -> pd.DataFrame:
    """Load, clean, and preprocess the raw dataset."""
    df = _load_raw_dataset(csv_path)
    LOGGER.info("Loaded raw dataset: %d rows, %d columns", len(df), df.shape[1])

    df = _replace_commas_with_periods(df)
    df = parse_budget_to_eur(df)
    df = _fill_default_values(df)
    df = _filter_incomplete_rows(df)
    df = _convert_range_columns(df)
    df = _convert_to_numeric(df)

    # Turbines: split model -> producer + power
    df = split_turbine_model(df)

    # Foundation normalization (BEFORE one-hot, BEFORE rare-collapse)
    df = normalize_foundation_type(df)
    df = consolidate_rare_foundation_types(
        df, min_count=config.FOUNDATION_MIN_COUNT
    )

    # Area imputation from installed capacity
    df = impute_area_sqkm(df)

    # Connection details (country + year -> 4 categorical columns)
    conn_details = df.apply(
        lambda row: pd.Series(
            config.get_connection_details(
                row.get("country"), row.get("commissioning_year")
            )
        ),
        axis=1,
    )
    df = pd.concat([df, conn_details], axis=1)

    df = _one_hot_encode_categoricals(df)

    final_columns = config.FINAL_COLUMNS.copy()
    for col in config.CATEGORICAL_COLUMNS:
        if col in df.columns:
            final_columns.extend(
                [c for c in df.columns if c.startswith(f"{col}_")]
            )

    for col in final_columns:
        if col not in df.columns:
            df[col] = np.nan

    LOGGER.info(
        "prepare_cleaned_dataset: returning %d rows x %d columns",
        len(df),
        len(final_columns),
    )
    return df[final_columns]


def _load_raw_dataset(csv_path: str) -> pd.DataFrame:
    """Load raw CSV while tolerating mixed encodings."""
    try:
        df = pd.read_csv(
            csv_path, sep=config.CSV_SEPARATOR, encoding="utf-8", engine="python"
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            csv_path, sep=config.CSV_SEPARATOR, encoding="latin-1", engine="python"
        )
    return df.replace(list(config.MISSING_VALUE_TOKENS), np.nan)


def _replace_commas_with_periods(df: pd.DataFrame) -> pd.DataFrame:
    """Replace commas with periods in all cells (for European decimal notation)."""
    for column in df.columns:
        df[column] = df[column].astype(str).str.replace(",", ".")
    # Re-introduce NaN after astype(str) turned them into 'nan' strings
    df = df.replace(list(config.MISSING_VALUE_TOKENS), np.nan)
    return df


def _fill_default_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values in critical columns with defaults."""
    for column, default in config.DEFAULT_VALUES.items():
        if column not in df.columns:
            df[column] = default
        df[column] = df[column].fillna(default)
    return df


def _filter_incomplete_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows that are missing critical information."""
    initial_count = len(df)
    cleaned = df.dropna(subset=config.MANDATORY_COLUMNS)

    dropped = initial_count - len(cleaned)
    if dropped:
        dropped_rows = df[~df.index.isin(cleaned.index)]
        for _, row in dropped_rows.iterrows():
            missing_cols = [
                col for col in config.MANDATORY_COLUMNS if pd.isna(row[col])
            ]
            farm_name = row.get("wind_farm_name", "Unknown")
            LOGGER.warning(
                "Dropping row for wind farm '%s' due to missing critical columns: %s",
                farm_name,
                ", ".join(missing_cols),
            )
    LOGGER.info("_filter_incomplete_rows: dropped %d / kept %d", dropped, len(cleaned))
    return cleaned


def _convert_range_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert columns with value ranges (e.g., '10-20') into min, max, and mean."""
    for col_name, suffix in config.RANGE_COLUMNS.items():
        base_name = col_name.replace("_m", "").replace("_km", "")
        df[f"{base_name}_min_{suffix}"] = df[col_name].apply(
            lambda x: to_float_range_mean(x)[0]
        )
        df[f"{base_name}_max_{suffix}"] = df[col_name].apply(
            lambda x: to_float_range_mean(x)[1]
        )
        df[f"{base_name}_mean_{suffix}"] = df[col_name].apply(
            lambda x: to_float_range_mean(x)[2]
        )

    df["installed_capacity_MW"] = df["installed_capacity_MW"].apply(
        lambda x: to_float_range_mean(x)[2] if pd.notna(x) else np.nan
    )
    return df


def _convert_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all specified numeric columns are of a numeric type."""
    for column in config.NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _one_hot_encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Apply one-hot encoding to categorical columns."""
    for column in config.CATEGORICAL_COLUMNS:
        if column in df.columns:
            df[column] = df[column].fillna("Unknown")
            dummies = pd.get_dummies(df[column], prefix=column, dummy_na=False)
            df = pd.concat([df, dummies], axis=1)
    return df