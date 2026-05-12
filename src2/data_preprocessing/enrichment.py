"""Data enrichment functions for offshore wind project dataset."""

from datetime import datetime
from functools import lru_cache
import logging
import math
import os

import numpy as np
import pandas as pd
import requests
import wind_stats

from shapely.geometry import Point
import geopandas as gpd
from pathlib import Path

from .. import config

LOGGER = logging.getLogger(__name__)

CONSTRUCTION_PORTS = [
    {"name": "Esbjerg",             "lon":  8.4469, "lat": 55.4633},
    {"name": "Rønne",               "lon": 14.7014, "lat": 55.0986},
    {"name": "Grenaa",              "lon": 10.9271, "lat": 56.4092},
    {"name": "Køge",                "lon": 12.1950, "lat": 55.4540},
    {"name": "Odense (Lindø)",      "lon": 10.5140, "lat": 55.5370},
    {"name": "Thyborøn",            "lon":  8.2170, "lat": 56.6990},
    {"name": "Cuxhaven",            "lon":  8.7087, "lat": 53.8640},
    {"name": "Bremerhaven",         "lon":  8.5755, "lat": 53.5536},
    {"name": "Sassnitz-Mukran",     "lon": 13.5890, "lat": 54.4820},
    {"name": "Rostock",             "lon": 12.1165, "lat": 54.1370},
    {"name": "Eemshaven",           "lon":  6.8250, "lat": 53.4420},
    {"name": "Rotterdam",           "lon":  4.2867, "lat": 51.8850},
    {"name": "Vlissingen",          "lon":  3.5700, "lat": 51.4400},
    {"name": "Oostende",            "lon":  2.9130, "lat": 51.2300},
    {"name": "Great Yarmouth",      "lon":  1.7300, "lat": 52.6000},
    {"name": "Hull",                "lon": -0.2798, "lat": 53.7424},
    {"name": "Able Seaton",         "lon": -1.1650, "lat": 54.6400},
    {"name": "Nigg Energy Park",    "lon": -3.9200, "lat": 57.7100},
    {"name": "Port of Dundee",      "lon": -2.9630, "lat": 56.4600},
    {"name": "Belfast",             "lon": -5.9100, "lat": 54.6100},
    {"name": "Port of Mostyn",      "lon": -3.2830, "lat": 53.3000},
    {"name": "Stord Base",          "lon":  5.5000, "lat": 59.7800},
    {"name": "Gulen",               "lon":  5.1200, "lat": 61.0242},
    {"name": "Saint-Nazaire",       "lon": -2.2100, "lat": 47.2700},
    {"name": "Brest",               "lon": -4.4900, "lat": 48.3800},
    {"name": "Ferrol",              "lon": -8.2400, "lat": 43.4800},
    {"name": "Taranto",             "lon": 17.2300, "lat": 40.4700},
    {"name": "Gdansk",              "lon": 18.6758, "lat": 54.3781},
    {"name": "Swinoujscie",         "lon": 14.2730, "lat": 53.9100},
]

_CONSTRUCTION_PORTS_GDF: gpd.GeoDataFrame | None = None

def _get_construction_ports_gdf() -> gpd.GeoDataFrame:
    """Return a GeoDataFrame of construction ports."""
    global _CONSTRUCTION_PORTS_GDF
    if _CONSTRUCTION_PORTS_GDF is None:
        ports_df = pd.DataFrame(CONSTRUCTION_PORTS)
        _CONSTRUCTION_PORTS_GDF = gpd.GeoDataFrame(
            ports_df,
            geometry=gpd.points_from_xy(ports_df["lon"], ports_df["lat"]),
            crs="EPSG:4326",
        )
    return _CONSTRUCTION_PORTS_GDF


def add_distance_from_construction_port(df: pd.DataFrame) -> pd.DataFrame:
    """Add distance_from_construction_port_km column"""
    ports_gdf = _get_construction_ports_gdf()
    df["distance_from_construction_port_km"] = df.apply(
        lambda row: _get_distance_from_port(row["LAT"], row["LON"], ports_gdf)
        if pd.notna(row.get("LAT")) and pd.notna(row.get("LON"))
        else np.nan,
        axis=1,
    )
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
        lambda row: _get_mean_wave_height(row["LAT"], row["LON"], row.get("commissioning_year")) \
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


def _get_distance_from_port(lat: float, lon: float, ports_gdf) -> float:
    """Calculate distance in km from nearest port using projected geometries."""
    projected_ports_gdf = ports_gdf
    if projected_ports_gdf.crs is None or projected_ports_gdf.crs.is_geographic:
        projected_ports_gdf = projected_ports_gdf.to_crs("EPSG:3035")

    farm_point = (
        gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326")
        .to_crs(projected_ports_gdf.crs)
        .iloc[0]
    )
    return projected_ports_gdf.geometry.distance(farm_point).min() / 1000.0


def add_distance_from_port(df: pd.DataFrame) -> pd.DataFrame:
    """Add distance_from_port_km column based on wind farm coordinates."""

    PORTS_SHP_PATHS = [
        Path(__file__).parent.parent.parent / "data" / "ports" / "Harbours_EMODnet_OSM_HOLAS3.shp",
        Path(__file__).parent.parent.parent / "data" / "ports" / "Harbours.shp",
    ]

    try:
        # Enable GDAL to restore/create missing .shx files
        os.environ['SHAPE_RESTORE_SHX'] = 'YES'
        
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
            ports_gdf = ports_gdf.set_crs("EPSG:3857")
        else:
            ports_gdf = ports_gdf.to_crs("EPSG:3857")

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

def _get_mean_wave_height(latitude: float, longitude: float, commissioning_year: int = None) -> float:
    """Fetch mean wave height from Open-Meteo Marine API for a specific year."""
    lat = round(float(latitude), config.GWA_COORDINATE_ROUND_DECIMALS)
    lon = round(float(longitude), config.GWA_COORDINATE_ROUND_DECIMALS)
    
    last_year = config.CURRENT_YEAR - 1
    target_year = last_year
    
    if commissioning_year is not None:
        try:
            year_int = int(commissioning_year)
            target_year = min(year_int, last_year)
        except (ValueError, TypeError):
            pass
    
    LOGGER.debug("Open-Meteo request lat=%.4f lon=%.4f year=%d", lat, lon, target_year)
    
    # Fetch full year of wave data based on commissioning year or last year if not available
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
        
        if "hourly" in data and "wave_height" in data["hourly"]:
            wave_heights = [h for h in data["hourly"]["wave_height"] if h is not None and h > 0]
            if wave_heights:
                mean_height = float(np.mean(wave_heights))
                LOGGER.debug("Open-Meteo result lat=%.4f lon=%.4f year=%d mean_wave_height=%.3f m", lat, lon, target_year, mean_height)
                return mean_height
    except Exception as exc:
        LOGGER.warning("Open-Meteo request failed for lat=%.4f lon=%.4f year=%d: %s", lat, lon, target_year, exc)
    
    # Fallback 1: Try current conditions
    LOGGER.debug("Fallback 1: Trying current wave height for lat=%.4f lon=%.4f", lat, lon)
    try:
        current_params = {
            "latitude": lat,
            "longitude": lon,
            "current": "wave_height",
            "timezone": "UTC",
        }
        response = requests.get(url, params=current_params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if "current" in data and "wave_height" in data["current"]:
            wave_height = data["current"]["wave_height"]
            if wave_height is not None and wave_height > 0:
                LOGGER.info("Using current wave height %.3f m for lat=%.4f lon=%.4f", wave_height, lat, lon)
                return float(wave_height)
    except Exception as exc:
        LOGGER.warning("Current wave height fallback failed for lat=%.4f lon=%.4f: %s", lat, lon, exc)
    
    # Return NaN if all attempts fail
    LOGGER.warning("[wave height] All attempts failed for lat=%.4f lon=%.4f", lat, lon)
    return np.nan
