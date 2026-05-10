"""Indexation functions for adjusting project budgets based on age and inflation."""

import logging

import numpy as np
import pandas as pd
import wbgapi as wb

from .. import config

LOGGER = logging.getLogger(__name__)

INDEXATION_MODES = (
    "no_indexation",
    "static_2pct",
    "static_3pct",
    "static_5pct",
    "world_bank_cpi",
)

country_map = {
    "Belgium": "BEL", "Denmark": "DNK", "Finland": "FIN", "France": "FRA",
    "Germany": "DEU", "Greece": "GRC", "Ireland": "IRL", "Italy": "ITA",
    "Netherlands": "NLD", "Norway": "NOR", "Poland": "POL", "Portugal": "PRT",
    "Spain": "ESP", "Sweden": "SWE", "UK": "GBR",
}

_CPI_CACHE: dict[str, pd.Series | None] = {}


def get_inflation_series(country_code: str):
    if country_code in _CPI_CACHE:
        return _CPI_CACHE[country_code]
    try:
        data = wb.data.DataFrame("FP.CPI.TOTL", country_code)
        if data.empty:
            _CPI_CACHE[country_code] = None
            return None
        series = data.iloc[0]
        series.index = [int(str(i).replace("YR", "")) for i in series.index]
        series = series.sort_index()
        _CPI_CACHE[country_code] = series
        return series
    except Exception as e:
        LOGGER.warning("Error fetching CPI for %s: %s", country_code, e)
        _CPI_CACHE[country_code] = None
        return None


def _static_factor(year: int, target_year: int, rate: float) -> float:
    age = target_year - year
    return (1 + rate) ** max(age, 0)


def _index_budget_for_age(
    df: pd.DataFrame, mode: str = "world_bank_cpi"
) -> pd.DataFrame:
    """Index project budgets to `INDEXED_BY_YEAR` according to `mode`."""
    if mode not in INDEXATION_MODES:
        raise ValueError(f"Unknown indexation mode: {mode!r}. "
                         f"Choose from {INDEXATION_MODES}")

    target_year = config.INDEXED_BY_YEAR

    def _row(row):
        budget = row["total_project_budget_eur"]
        year = row["commissioning_year"]
        country = row.get("country")

        if pd.isna(budget) or pd.isna(year):
            return budget, "missing_budget_or_year"

        year = int(year)
        age = config.CURRENT_YEAR - year

        # Recent projects: never indexed
        if age <= config.MAX_DATA_AGE_YEARS:
            return budget, "recent"

        if mode == "no_indexation":
            return budget, "no_indexation"

        if mode == "static_2pct":
            return budget * _static_factor(year, target_year, 0.02), "static_2pct"
        if mode == "static_3pct":
            return budget * _static_factor(year, target_year, 0.03), "static_3pct"
        if mode == "static_5pct":
            return budget * _static_factor(year, target_year, 0.05), "static_5pct"

        # world_bank_cpi
        wb_code = country_map.get(country, country)
        cpi = get_inflation_series(wb_code) if wb_code else None
        if (
            cpi is not None
            and year in cpi.index
            and target_year in cpi.index
        ):
            cpi_start, cpi_end = cpi.get(year), cpi.get(target_year)
            if cpi_start and cpi_end:
                return budget * (cpi_end / cpi_start), f"wb_cpi_{year}"
        # Fallback to static 2%
        LOGGER.debug("CPI fallback to 2%% for %s/%s", wb_code, year)
        return budget * _static_factor(year, target_year, 0.02), f"wb_fallback_static_2pct_{year}"

    indexed_budget, quality = zip(*df.apply(_row, axis=1))
    df["total_project_budget_eur_indexed"] = indexed_budget
    df["data_quality_flag"] = quality
    return df


def build_indexed_dataset(
    df: pd.DataFrame, mode: str = "world_bank_cpi"
) -> pd.DataFrame:
    """Create the indexed dataset (default = World Bank CPI)."""
    return _index_budget_for_age(df.copy(), mode=mode)