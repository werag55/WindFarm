"""Data cleaning utilities for offshore wind project dataset."""

import numpy as np
import pandas as pd

from .. import config
from .parsing import parse_budget_to_eur, to_float_range_mean

def load_raw_dataset(csv_path: str) -> pd.DataFrame:
    """Load raw CSV while tolerating mixed encodings."""
    try:
        df = pd.read_csv(csv_path, sep=config.CSV_SEPARATOR, encoding="utf-8", engine="python")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, sep=config.CSV_SEPARATOR, encoding="latin-1", engine="python")
    return df.replace(list(config.MISSING_VALUE_TOKENS), np.nan)


def replace_commas_with_periods(df: pd.DataFrame) -> pd.DataFrame:
    """Replace commas with periods in all cells"""
    for column in df.columns:
        df[column] = df[column].astype(str).str.replace(',', '.')
    return df


def fill_default_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values in critical columns with defaults."""
    for column, default in config.DEFAULT_VALUES.items():
        if column not in df.columns:
            df[column] = default
        df[column] = df[column].fillna(default)
    return df


def fill_foundation_scope(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing 'foundation_scope' based on 'country'."""
    missing_scope = df["foundation_scope"].isna() & df["country"].notna()
    df.loc[missing_scope, "foundation_scope"] = \
        df.loc[missing_scope, "country"].map(config.COUNTRY_TO_FOUNDATION_SCOPE)
    return df


def normalize_core_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize numeric project fields used throughout the pipeline."""
    df["commissioning_year"] = df["commissioning_year"].apply(to_float_range_mean)
    df["distance_from_shore_km"] = df["distance_from_shore_km"].apply(to_float_range_mean)
    df["water_depth_m"] = df["water_depth_m"].apply(to_float_range_mean)
    df["project_lifetime_years"] = df["project_lifetime_years"].apply(to_float_range_mean)
    df["total_project_budget_eur"] = df["total_project_budget"].apply(parse_budget_to_eur)
    df["installed_capacity_MW"] = pd.to_numeric(df["installed_capacity_MW"], errors="coerce")
    df['LAT'] = pd.to_numeric(df['LAT'], errors='coerce')
    df['LON'] = pd.to_numeric(df['LON'], errors='coerce')
    return df


def filter_incomplete_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows that are missing critical information."""
    return df.dropna(subset=config.MANDATORY_COLUMNS)


def prepare_cleaned_dataset(raw_csv_path: str) -> pd.DataFrame:
    """Run cleaning pipeline."""
    df = load_raw_dataset(raw_csv_path)
    df = replace_commas_with_periods(df)
    df = fill_default_values(df)
    df = fill_foundation_scope(df)
    df = normalize_core_fields(df)
    df = filter_incomplete_rows(df)
    return df
