"""Indexation functions for adjusting project budgets based on age and inflation."""

import pandas as pd

from .. import config

#TODO: Verify / Adjust inflation indexation
def index_budget_for_age(df: pd.DataFrame) -> pd.DataFrame:
    """Index budgets for project age and tag the data quality decision."""

    def _index_row(row):
        budget = row["total_project_budget_eur"]
        year = row["commissioning_year"]
        if pd.isna(budget) or pd.isna(year):
            return budget, "missing_budget_or_year"
        age = config.CURRENT_YEAR - int(year)
        if age <= config.MAX_DATA_AGE_YEARS:
            return budget, "recent"
        factor = (1 + config.INFLATION_INDEX) ** age
        return budget * factor, f"indexed_{age}y"

    indexed_budget, quality = zip(*df.apply(_index_row, axis=1))
    df["total_project_budget_eur_indexed"] = indexed_budget
    df["data_quality_flag"] = quality
    return df


def build_indexed_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Create the indexed dataset."""
    return index_budget_for_age(df.copy())
