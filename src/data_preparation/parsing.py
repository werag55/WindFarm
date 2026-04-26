"""Parsing and normalization helpers."""

import logging
import re

import numpy as np
import pandas as pd

from .. import config


LOGGER = logging.getLogger(__name__)


def to_float_range_mean(value) -> float:
    """Extract numeric values from a single value or range and return the mean."""
    if pd.isna(value):
        return np.nan
    numbers = re.findall(r"[0-9]+(?:\.[0-9]+)?", str(value).replace(",", "."))
    if not numbers:
        return np.nan
    return float(np.mean([float(number) for number in numbers]))


#TODO: improve logic to calculate based on exchange range given commisioning year
def parse_budget_to_eur(text: str) -> float:
    """Parse budget strings like 'GBP 4.2 billion' into EUR."""
    if pd.isna(text):
        return np.nan
    raw = str(text)
    currency = next((currency \
        for currency in config.CURRENCY_RATES_TO_EUR if currency in raw), None)
    if currency is None:
        LOGGER.warning("Missing currency in budget value '%s'; defaulting to EUR", raw)
        currency = "EUR"
    elif currency not in config.CURRENCY_RATES_TO_EUR:
        LOGGER.error("Unsupported currency in budget value '%s'; defaulting to EUR", raw)
        return np.nan
    scale_key = next((scale \
        for scale in config.BUDGET_SCALES if scale in raw.lower()), config.DEFAULT_BUDGET_SCALE)
    numbers = re.findall(r"[0-9]+(?:\.[0-9]+)?", raw.replace(",", "."))
    if not numbers:
        return np.nan
    mean_value = float(np.mean([float(number) for number in numbers]))
    return mean_value * config.BUDGET_SCALES[scale_key] * config.CURRENCY_RATES_TO_EUR[currency]
