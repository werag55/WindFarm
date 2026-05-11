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
    """Fetch mean wave height."""
    return 9.0

def add_environmental_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Attach mean wind speed and wave height data.

    Wind speed: Global Wind Atlas via wind-stats.
    Wave height: Open-Meteo Marine API. If unavailable, stays NaN.
    """

    df = df.copy()

    for col in ("mean_wind_speed_mps", "mean_wave_height_m"):
        if col not in df.columns:
            df[col] = np.nan
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    def _fill_wind(row):
        if pd.notna(row["mean_wind_speed_mps"]):
            return row["mean_wind_speed_mps"]

        if pd.isna(row.get("LAT")) or pd.isna(row.get("LON")):
            return np.nan

        try:
            return _get_mean_wind_speed_from_gwa(row["LAT"], row["LON"])
        except Exception as exc:
            LOGGER.warning(
                "Wind speed fetch failed for lat=%s lon=%s: %s",
                row.get("LAT"),
                row.get("LON"),
                exc,
            )
            return np.nan

    def _fill_wave(row):
        if pd.notna(row["mean_wave_height_m"]):
            return row["mean_wave_height_m"]

        if pd.isna(row.get("LAT")) or pd.isna(row.get("LON")):
            return np.nan

        try:
            return _get_mean_wave_height(
                row["LAT"],
                row["LON"],
                row.get("commissioning_year"),
            )
        except Exception as exc:
            LOGGER.warning(
                "Wave height fetch failed for lat=%s lon=%s: %s",
                row.get("LAT"),
                row.get("LON"),
                exc,
            )
            return np.nan

    df["mean_wind_speed_mps"] = df.apply(_fill_wind, axis=1)
    df["mean_wave_height_m"] = df.apply(_fill_wave, axis=1)

    LOGGER.info(
        "Environmental enrichment: wind filled=%d/%d, wave filled=%d/%d",
        int(df["mean_wind_speed_mps"].notna().sum()),
        len(df),
        int(df["mean_wave_height_m"].notna().sum()),
        len(df),
    )

    return df

@lru_cache(maxsize=512)
def _get_mean_wave_height(
    latitude: float,
    longitude: float,
    commissioning_year: int | None = None,
) -> float:
    """Fetch annual mean wave height from Open-Meteo Marine API.

    Returns NaN if no valid data is available.
    """

    lat = round(float(latitude), config.GWA_COORDINATE_ROUND_DECIMALS)
    lon = round(float(longitude), config.GWA_COORDINATE_ROUND_DECIMALS)

    last_year = config.CURRENT_YEAR - 1
    target_year = last_year

    if commissioning_year is not None and pd.notna(commissioning_year):
        try:
            target_year = min(int(commissioning_year), last_year)
        except (ValueError, TypeError):
            target_year = last_year

    start_date = datetime(target_year, 1, 1).date()
    end_date = datetime(target_year, 12, 31).date()

    url = "https://marine-api.open-meteo.com/v1/marine"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "wave_height",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "timezone": "UTC",
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        LOGGER.warning(
            "Open-Meteo wave request failed for lat=%.4f lon=%.4f year=%d: %s",
            lat,
            lon,
            target_year,
            exc,
        )
        return np.nan

    wave_values = data.get("hourly", {}).get("wave_height", [])

    wave_heights = [
        float(h)
        for h in wave_values
        if h is not None and pd.notna(h) and float(h) > 0
    ]

    if not wave_heights:
        LOGGER.warning(
            "No wave data for lat=%.4f lon=%.4f year=%d",
            lat,
            lon,
            target_year,
        )
        return np.nan

    mean_height = float(np.mean(wave_heights))

    LOGGER.debug(
        "Open-Meteo wave result lat=%.4f lon=%.4f year=%d mean_wave_height=%.3f m",
        lat,
        lon,
        target_year,
        mean_height,
    )

    return mean_height
