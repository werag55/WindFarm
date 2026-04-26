"""Prepare a sample for prediction"""

import pandas as pd

from .. import config
from .enrichment import get_mean_wind_speed_from_gwa, get_mean_wave_height


def prepare_sample(sample: dict | None = None) -> dict:
    """Fill default values, get country if missing, fill foundation scope, and add environmental columns."""
    sample_data = dict(config.DEFAULT_NEW_SAMPLE)
    if sample:
        sample_data.update(sample)

    if (pd.isna(sample_data.get("foundation_scope")) or sample_data.get("foundation_scope") == "Unknown") \
        and sample_data.get("country") in config.COUNTRY_TO_FOUNDATION_SCOPE:
        sample_data["foundation_scope"] = config.COUNTRY_TO_FOUNDATION_SCOPE[sample_data["country"]]

    if pd.isna(sample_data.get("mean_wind_speed_mps")) \
        and not pd.isna(sample_data.get("LAT")) and not pd.isna(sample_data.get("LON")):
        sample_data["mean_wind_speed_mps"] = get_mean_wind_speed_from_gwa(sample_data["LAT"], sample_data["LON"])

    if pd.isna(sample_data.get("mean_wave_height_m")) \
        and not pd.isna(sample_data.get("LAT")) and not pd.isna(sample_data.get("LON")):
        sample_data["mean_wave_height_m"] = get_mean_wave_height(sample_data["LAT"], sample_data["LON"])

    return sample_data
