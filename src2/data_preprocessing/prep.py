"""Data preparation pipeline: from raw CSV to cleaned and enriched dataset."""

import logging
import os

import pandas as pd

from .. import config
from .cleanup import prepare_cleaned_dataset
from .enrichment import (
    add_distance_from_construction_port,
    add_distance_from_port,
    add_distance_from_shore_columns,
    add_environmental_columns,
    add_water_depth_columns,
    log_environmental_examples,
    log_port_distance_examples,
)
from .indexation import build_indexed_dataset

LOGGER = logging.getLogger(__name__)


def prepare_data() -> pd.DataFrame:
    """Run the full data preparation pipeline."""
    if not os.path.isfile(config.CLEANED_DATASET_PATH):
        LOGGER.info("No cached dataset found — running full preparation pipeline")
        df = prepare_cleaned_dataset(config.RAW_DATASET_PATH)

        LOGGER.info("--- Step: environmental enrichment (wind + wave) ---")
        df = add_environmental_columns(df)

        LOGGER.info("--- Step: water depth (GEBCO) ---")
        df = add_water_depth_columns(df)

        LOGGER.info("--- Step: distance from shore (coastline) ---")
        df = add_distance_from_shore_columns(df)

        LOGGER.info("--- Step: distances to ports ---")
        df = add_distance_from_port(df)
        df = add_distance_from_construction_port(df)

        LOGGER.info("--- Step: indexation ---")
        df = build_indexed_dataset(df)

        log_port_distance_examples(df, n=5)
        log_environmental_examples(df, n=5)
        df.to_csv(config.CLEANED_DATASET_PATH, index=False)
        LOGGER.info("Saved cleaned dataset to %s", config.CLEANED_DATASET_PATH)
    else:
        LOGGER.info("Loading cached dataset from %s", config.CLEANED_DATASET_PATH)
        df = pd.read_csv(config.CLEANED_DATASET_PATH)

    _log_summary(df)
    return df


def _log_summary(df: pd.DataFrame) -> None:
    """Print a short post-pipeline summary."""
    LOGGER.info("=" * 60)
    LOGGER.info("PREPROCESSING SUMMARY")
    LOGGER.info("=" * 60)
    LOGGER.info("Total rows: %d", len(df))

    if "turbine_producer" in df.columns:
        LOGGER.info(
            "turbine_producer: %d filled / %d missing",
            int(df["turbine_producer"].notna().sum()),
            int(df["turbine_producer"].isna().sum()),
        )
        LOGGER.info(
            "turbine_producer top values: %s",
            df["turbine_producer"].value_counts(dropna=False).head(10).to_dict(),
        )

    if "turbine_power_MW" in df.columns:
        LOGGER.info(
            "turbine_power_MW: %d filled / %d missing (mean=%.2f)",
            int(df["turbine_power_MW"].notna().sum()),
            int(df["turbine_power_MW"].isna().sum()),
            float(df["turbine_power_MW"].mean(skipna=True) or 0.0),
        )

    if "foundation_type" in df.columns:
        LOGGER.info(
            "foundation_type final categories: %s",
            df["foundation_type"].value_counts(dropna=False).to_dict(),
        )

    if "area_sqkm_imputed" in df.columns:
        LOGGER.info(
            "area_sqkm: %d imputed / %d original / %d still missing",
            int(df["area_sqkm_imputed"].sum()),
            int((df["area_sqkm"].notna() & ~df["area_sqkm_imputed"]).sum()),
            int(df["area_sqkm"].isna().sum()),
        )

    # Quality-flag distributions for the marine enrichers.
    for flag in ("wave_height_quality_flag",
                 "water_depth_quality_flag",
                 "distance_from_shore_quality_flag"):
        if flag in df.columns:
            LOGGER.info(
                "%s distribution: %s",
                flag,
                df[flag].value_counts(dropna=False).to_dict(),
            )

    LOGGER.info("=" * 60)