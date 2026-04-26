"""Data enrichment functions for offshore wind project dataset."""

from functools import lru_cache
import logging
import math

import numpy as np
import pandas as pd
import wind_stats

from .. import config

LOGGER = logging.getLogger(__name__)

@lru_cache(maxsize=512)
def get_mean_wind_speed_from_gwa(latitude: float, longitude: float) -> float:
    """Fetch mean wind speed from Global Wind Atlas through wind-stats."""
    lat = round(float(latitude), config.GWA_COORDINATE_ROUND_DECIMALS)
    lon = round(float(longitude), config.GWA_COORDINATE_ROUND_DECIMALS)
    LOGGER.debug("GWA request lat=%.4f lon=%.4f", lat, lon)
    ds = wind_stats.get_gwc_data(lat, lon)

    target_height = float(config.HUB_HEIGHT_M)
    height = float(ds.height.sel(height=target_height, method="nearest").item())
    roughness = float(ds.roughness.sel(roughness=config.GWA_TARGET_ROUGHNESS_M, \
        method="nearest").item())

    a_values = ds["A"].sel(roughness=roughness, height=height).values
    k_values = ds["k"].sel(roughness=roughness, height=height).values
    frequency = ds["frequency"].sel(roughness=roughness).values

    sector_mean_speed = a_values * np.vectorize(math.gamma)(1.0 + (1.0 / k_values))
    weights = frequency / np.sum(frequency)
    wind_speed = float(np.sum(sector_mean_speed * weights))
    LOGGER.debug("GWA result lat=%.4f lon=%.4f mean_wind_speed=%.3f", lat, lon, wind_speed)
    return wind_speed


#TODO: Implement actual wave height fetching
def get_mean_wave_height(latitude: float, longitude: float) -> float:
    """Fetch mean wave height from Global Wind Atlas through wind-stats."""
    return 9.0

def add_environmental_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Attach mean wind speed and wave data, using GWA."""
    df["mean_wind_speed_mps"] = df.apply(
        lambda row: get_mean_wind_speed_from_gwa(row["LAT"], row["LON"]) \
            if pd.isna(row["mean_wind_speed_mps"]) \
                and not pd.isna(row["LAT"]) and not pd.isna(row["LON"]) \
            else row["mean_wind_speed_mps"],
        axis=1,
    )
    df["mean_wave_height_m"] = df.apply(
        lambda row: get_mean_wave_height(row["LAT"], row["LON"]) \
            if pd.isna(row["mean_wave_height_m"]) \
                and not pd.isna(row["LAT"]) and not pd.isna(row["LON"]) \
            else row["mean_wave_height_m"],
        axis=1,
    )
    return df
