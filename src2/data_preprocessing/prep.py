"""Data preparation pipeline: from raw CSV to cleaned and enriched dataset."""

import os

import pandas as pd

from .. import config
from .cleanup import prepare_cleaned_dataset
from .enrichment import (
    add_distance_from_construction_port,
    add_distance_from_port,
    add_distance_from_shore,
    add_environmental_columns,
    add_water_depth,
)
from .indexation import build_indexed_dataset

def prepare_data() -> pd.DataFrame:
    """Run the full data preparation pipeline."""
    if not os.path.isfile(config.CLEANED_DATASET_PATH):
        df = prepare_cleaned_dataset(config.RAW_DATASET_PATH)
        df = add_environmental_columns(df)
        df = add_water_depth(df)
        df = add_distance_from_shore(df)
        df = add_distance_from_port(df)
        df = add_distance_from_construction_port(df)
        df = build_indexed_dataset(df)
        df.to_csv(config.CLEANED_DATASET_PATH, index=False)
    else:
        df = pd.read_csv(config.CLEANED_DATASET_PATH)
    return df
