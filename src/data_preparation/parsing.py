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


def to_float_range_mean(value) -> float:
    """Extract numeric values from a single value or range and return the mean."""
    if pd.isna(value):
        return np.nan
    numbers = re.findall(r"[0-9]+(?:\.[0-9]+)?", str(value).replace(",", "."))
    if not numbers:
        return np.nan
    return float(np.mean([float(number) for number in numbers]))


def get_currency_rate(currency: str, year: int) -> float:
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
        LOGGER.warning("Could not get exchange rate for %s in %s", currency, year)
        return np.nan


def parse_budget_to_eur(text: str, commissioning_year: int) -> float:
    """Parse budget strings like 'GBP 4.2 billion' into EUR."""
    if pd.isna(text):
        return np.nan
    raw = str(text)
    year = commissioning_year if pd.notna(commissioning_year) else datetime.now().year

    currency = next((cur for cur in config.SUPPORTED_CURRENCIES if cur in raw), None)

    if currency is None:
        LOGGER.warning("Missing currency in budget value '%s'; defaulting to EUR", raw)
        currency = "EUR"

    rate = get_currency_rate(currency, int(year))
    if pd.isna(rate):
        return np.nan

    scale_key = next((scale \
        for scale in config.BUDGET_SCALES if scale in raw.lower()), config.DEFAULT_BUDGET_SCALE)
    numbers = re.findall(r"[0-9]+(?:\.[0-9]+)?", raw.replace(",", "."))
    if not numbers:
        return np.nan
    mean_value = float(np.mean([float(number) for number in numbers]))
    return mean_value * config.BUDGET_SCALES[scale_key] * rate
