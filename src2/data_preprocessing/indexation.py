"""Indexation functions for adjusting project budgets based on age and inflation."""

import logging

import pandas as pd
import wbgapi as wb

from .. import config

LOGGER = logging.getLogger(__name__)

# country names as used in the dataset to ISO3 codes used by World Bank API
country_map = {
    'Belgium': 'BEL',
    'Denmark': 'DNK',
    'Finland': 'FIN',
    'France': 'FRA',
    'Germany': 'DEU',
    'Greece': 'GRC',
    'Ireland': 'IRL',
    'Italy': 'ITA',
    'Netherlands': 'NLD',
    'Norway': 'NOR',
    'Poland': 'POL',
    'Portugal': 'PRT',
    'Spain': 'ESP',
    'Sweden': 'SWE',
    'UK': 'GBR'
}

_CPI_CACHE: dict[str, pd.Series | None] = {}


def get_inflation_series(country_code: str):
    """Fetch historical Consumer Price Index (CPI) from World Bank, cached per country."""
    if country_code in _CPI_CACHE:
        return _CPI_CACHE[country_code]
    try:
        data = wb.data.DataFrame('FP.CPI.TOTL', country_code)
        if data.empty:
            _CPI_CACHE[country_code] = None
            return None
        series = data.iloc[0]
        series.index = [int(str(i).replace('YR', '')) for i in series.index]
        series = series.sort_index()
    except Exception as exc:
        LOGGER.warning("CPI fetch failed for %s: %s", country_code, exc)
        series = None
    _CPI_CACHE[country_code] = series
    return series


def _index_budget_for_age(df: pd.DataFrame) -> pd.DataFrame:
    """Index every budget to INDEXED_BY_YEAR using World Bank CPI, with a static fallback."""

    indexed_by_year = config.INDEXED_BY_YEAR

    def _index_row(row):
        budget = row["total_project_budget_eur"]
        year = row["commissioning_year"]
        country = row["country"]

        if pd.isna(budget) or pd.isna(year):
            return budget, "missing_budget_or_year"

        year = int(year)
        if year == indexed_by_year:
            return budget, "indexed_base"

        wb_code = country_map.get(country, country)
        cpi_series = get_inflation_series(wb_code)

        if cpi_series is not None and year in cpi_series.index and indexed_by_year in cpi_series.index:
            cpi_start = cpi_series.get(year)
            cpi_end = cpi_series.get(indexed_by_year)
            if cpi_start and cpi_end:
                factor = cpi_end / cpi_start
                return budget * factor, f"indexed_wb_{year}"

        # Fallback: compound config.INFLATION_INDEX across the year gap.
        # Works in both directions (year > indexed_by_year => deflate, factor < 1).
        LOGGER.warning("No CPI for %s in year %s, using static fallback", wb_code, year)
        indexed_age = indexed_by_year - year
        factor = (1 + config.INFLATION_INDEX) ** indexed_age
        return budget * factor, f"indexed_static_{indexed_age}y"

    indexed_budget, quality = zip(*df.apply(_index_row, axis=1))
    df["total_project_budget_eur_indexed"] = indexed_budget
    df["data_quality_flag"] = quality
    return df

def build_indexed_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Create the indexed dataset."""
    return _index_budget_for_age(df.copy())