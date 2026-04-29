"""Parsing and normalization helpers."""

import logging
import re
from datetime import datetime

import numpy as np
import pandas as pd
from forex_python.converter import RatesNotAvailableError, get_rate

from .. import config


LOGGER = logging.getLogger(__name__)


_CURRENCY_RATE_CACHE = {}


def to_float_range_mean(text: str) -> tuple[float, float, float]:
    """Convert a string representing a range (e.g., '10-20') to a tuple of (min, max, mean)."""
    if pd.isna(text):
        return np.nan, np.nan, np.nan
    numbers = re.findall(r"[0-9]+(?:\.[0-9]+)?", str(text))
    if not numbers:
        return np.nan, np.nan, np.nan
    floats = [float(n) for n in numbers]
    return min(floats), max(floats), np.mean(floats)


def parse_budget_to_eur(df: pd.DataFrame) -> pd.DataFrame:
    """Parse budget strings like 'GBP 4.2 billion' into EUR."""
    df["total_project_budget_eur"] = df.apply(
        lambda row: _parse_single_budget(row["total_project_budget"], row["commissioning_year"]),
        axis=1
    )
    return df


def _get_currency_rate(currency: str, year: int) -> float:
    """Get currency rate for a given year."""
    if currency == "EUR":
        return 1.0
    cache_key = (currency, year)
    if cache_key in _CURRENCY_RATE_CACHE:
        return _CURRENCY_RATE_CACHE[cache_key]
    try:
        rate = get_rate(currency, "EUR", datetime(year, 1, 1))
        _CURRENCY_RATE_CACHE[cache_key] = rate
        return rate
    except RatesNotAvailableError:
        LOGGER.warning("Could not get exchange rate for %s in %s, trying fallback", currency, year)
        try:
            rate = config.CURRENCY_TO_EUR_IN_YEAR[str(year)][currency]
            _CURRENCY_RATE_CACHE[cache_key] = rate
            return rate
        except KeyError:
            LOGGER.error("No fallback rate found for %s in %s", currency, year)
            return np.nan


def _parse_single_budget(text: str, commissioning_year: int) -> float:
    """Parse a single budget string into EUR."""
    if pd.isna(text):
        return np.nan
    raw = str(text)
    year = commissioning_year if pd.notna(commissioning_year) else datetime.now().year

    currency = next((cur for cur in config.SUPPORTED_CURRENCIES if cur in raw), None)

    if currency is None:
        LOGGER.warning("Missing currency in budget value '%s'; defaulting to EUR", raw)
        currency = "EUR"

    rate = _get_currency_rate(currency, int(float(year)))
    if pd.isna(rate):
        return np.nan

    scale_key = next((scale \
        for scale in config.BUDGET_SCALES if scale in raw.lower()), config.DEFAULT_BUDGET_SCALE)
    numbers = re.findall(r"[0-9]+(?:\.[0-9]+)?", raw.replace(",", "."))
    if not numbers:
        return np.nan
    mean_value = float(np.mean([float(number) for number in numbers]))
    return mean_value * config.BUDGET_SCALES[scale_key] * rate
