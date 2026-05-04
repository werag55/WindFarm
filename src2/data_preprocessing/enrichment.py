"""Data enrichment functions for offshore wind project dataset."""

from functools import lru_cache
import logging
import math

import numpy as np
import pandas as pd
import wind_stats

from shapely.geometry import Point
import geopandas as gpd
from pathlib import Path

from .. import config

LOGGER = logging.getLogger(__name__)

#TODO: Implement actual distance calculation - distance from (LAT, LON) to the nearest port, analyse `European Offshore Wind Construction Ports.pdf`
def add_distance_from_port(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the distance from each wind farm to the nearest port."""
    df["distance_from_port_km"] = df["distance_from_shore_mean_km"]
    return df

def add_environmental_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Attach mean wind speed and wave data, using GWA."""
    if "mean_wind_speed_mps" not in df.columns:
        df["mean_wind_speed_mps"] = np.nan
    if "mean_wave_height_m" not in df.columns:
        df["mean_wave_height_m"] = np.nan
        
    df["mean_wind_speed_mps"] = df.apply(
        lambda row: _get_mean_wind_speed_from_gwa(row["LAT"], row["LON"]) \
            if pd.isna(row["mean_wind_speed_mps"]) \
                and not pd.isna(row["LAT"]) and not pd.isna(row["LON"]) \
            else row["mean_wind_speed_mps"],
        axis=1,
    )
    df["mean_wave_height_m"] = df.apply(
        lambda row: _get_mean_wave_height(row["LAT"], row["LON"]) \
            if pd.isna(row["mean_wave_height_m"]) \
                and not pd.isna(row["LAT"]) and not pd.isna(row["LON"]) \
            else row["mean_wave_height_m"],
        axis=1,
    )
    return df

@lru_cache(maxsize=512)
def _get_mean_wind_speed_from_gwa(latitude: float, longitude: float) -> float:
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
def _get_mean_wave_height(latitude: float, longitude: float) -> float:
    """Fetch mean wave height."""
    return 9.0

def _get_distance_from_port(lat: float, lon: float, ports_gdf) -> float:
    """Calculate distance in km from nearest port using preloaded GeoDataFrame."""
    farm_point = Point(lon, lat)
    distances = ports_gdf.geometry.distance(farm_point)
    # distance() returns degrees if CRS is EPSG:4326, convert to km
    min_dist_deg = distances.min()
    # rough conversion: 1 degree = 111 km
    return min_dist_deg * 111.0


def add_distance_from_port(df: pd.DataFrame) -> pd.DataFrame:
    """Add distance_from_port_km column based on wind farm coordinates."""

    PORTS_SHP_PATHS = [
        Path(__file__).parent.parent.parent / "data" / "ports" / "ports1.shp",
        Path(__file__).parent.parent.parent / "data" / "ports" / "ports2.shp",
    ]

    try:
        gdfs = []
        for path in PORTS_SHP_PATHS:
            if path.exists():
                gdf = gpd.read_file(path)
                gdfs.append(gdf)
        if not gdfs:
            LOGGER.warning("No port shapefiles found, skipping distance_from_port_km")
            df["distance_from_port_km"] = np.nan
            return df

        ports_gdf = pd.concat(gdfs, ignore_index=True)
        ports_gdf = gpd.GeoDataFrame(ports_gdf, geometry="geometry")
        if ports_gdf.crs is None:
            ports_gdf = ports_gdf.set_crs("EPSG:4326")
        else:
            ports_gdf = ports_gdf.to_crs("EPSG:4326")

        LOGGER.info(f"Loaded {len(ports_gdf)} ports from shapefiles")

        df["distance_from_port_km"] = df.apply(
            lambda row: _get_distance_from_port(row["LAT"], row["LON"], ports_gdf)
            if pd.notna(row.get("LAT")) and pd.notna(row.get("LON"))
            else np.nan,
            axis=1,
        )

    except Exception as e:
        LOGGER.error(f"Failed to calculate distance from port: {e}")
        df["distance_from_port_km"] = np.nan

    return df