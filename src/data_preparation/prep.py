"""Data preparation pipeline: from raw CSV to cleaned and enriched dataset."""

import os

import pandas as pd

from .. import config
from .cleanup import prepare_cleaned_dataset
from .enrichment import add_environmental_columns
from .indexation import build_indexed_dataset

def prepare_data() -> pd.DataFrame:
    """Run the full data preparation pipeline."""
    if not os.path.isfile(config.CLEANED_DATA_PATH):
        df = prepare_cleaned_dataset(config.RAW_DATASET_PATH)
        df = add_environmental_columns(df)
        df = build_indexed_dataset(df)
        df.to_csv(config.CLEANED_DATA_PATH, index=False)
    else:
        df = pd.read_csv(config.CLEANED_DATA_PATH)
    return df
