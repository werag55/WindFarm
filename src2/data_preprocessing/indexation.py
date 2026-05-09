"""Indexation functions for adjusting project budgets based on age and inflation."""

import pandas as pd
import wbgapi as wb

from src.data_preparation.cleanup import LOGGER
from .. import config

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

def get_inflation_series(country_code: str):
    """
    Fetches historical Consumer Price Index (CPI) from World Bank.
    """
    try:
        # Returns a pandas DataFrame of CPI values indexed by year
        data = wb.data.DataFrame('FP.CPI.TOTL', country_code)
        if data.empty:
            return None
        series = data.iloc[0]
        series.index = [int(str(i).replace('YR', '')) for i in series.index]
        return series.sort_index()
    except Exception as e:
        print(f"Error fetching data for {country_code}: {e}")
        return None

def _index_budget_for_age(df: pd.DataFrame) -> pd.DataFrame:
    """Index budgets using dynamic World Bank inflation data."""

    def _index_row(row):
        budget = row["total_project_budget_eur"]
        year = row["commissioning_year"]
        country = row["country"]
        
        if pd.isna(budget) or pd.isna(year):
            return budget, "missing_budget_or_year"

        year = int(year)
        indexed_by_year = config.INDEXED_BY_YEAR
        age = config.CURRENT_YEAR - year
        
        if age <= config.MAX_DATA_AGE_YEARS:
            return budget, "recent"

        wb_code = country_map.get(country, country)
        cpi_series = get_inflation_series(wb_code)

        if cpi_series is not None and year in cpi_series.index and indexed_by_year in cpi_series.index:
            # Formula: Budget * (CPI_today / CPI_then)
            cpi_start = cpi_series.get(year)
            cpi_end = cpi_series.get(indexed_by_year)

            if cpi_start and cpi_end:
                factor = cpi_end / cpi_start
                return budget * factor, f"indexed_wb_{year}"
        
        # Fallback to static config if API data is missing
        LOGGER.warning(f"No CPI for {wb_code} in year {year}, using fallback")
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