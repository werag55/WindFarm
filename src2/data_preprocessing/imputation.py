"""Simple imputation helpers for fields derivable from other columns."""

import logging

import numpy as np
import pandas as pd

from .. import config

LOGGER = logging.getLogger(__name__)


def impute_area_sqkm(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing `area_sqkm` from installed_capacity / power density.

    Adds boolean column `area_sqkm_imputed` flagging which rows were filled.
    """
    if "area_sqkm" not in df.columns:
        df["area_sqkm"] = np.nan
    if "installed_capacity_MW" not in df.columns:
        LOGGER.warning("impute_area_sqkm: installed_capacity_MW missing, nothing to do")
        df["area_sqkm_imputed"] = False
        return df

    df["area_sqkm"] = pd.to_numeric(df["area_sqkm"], errors="coerce")
    df["installed_capacity_MW"] = pd.to_numeric(
        df["installed_capacity_MW"], errors="coerce"
    )

    mask = df["area_sqkm"].isna() & df["installed_capacity_MW"].notna()
    n_to_fill = int(mask.sum())

    df["area_sqkm_imputed"] = False
    if n_to_fill:
        df.loc[mask, "area_sqkm"] = (
            df.loc[mask, "installed_capacity_MW"] / config.POWER_DENSITY_MW_PER_SQKM
        )
        df.loc[mask, "area_sqkm_imputed"] = True

    LOGGER.info(
        "impute_area_sqkm: imputed %d rows using density %.2f MW/km² "
        "(remaining NaN: %d)",
        n_to_fill,
        config.POWER_DENSITY_MW_PER_SQKM,
        int(df["area_sqkm"].isna().sum()),
    )
    return df