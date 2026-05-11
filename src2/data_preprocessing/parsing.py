"""Parsing helpers for budget and range columns.

Assumptions baked in (verified against `data/european_offshore_wind_capex.csv`):
  * `_replace_commas_with_periods` runs before this module — incoming strings
    contain no commas, so no thousands-/decimal-separator heuristics are needed.
  * Only two scale tokens ever appear: "million" and "billion".
  * "Combined" / aggregated-budget rows are dropped upstream
    (see `cleanup._drop_combined_budget_rows`), so this parser never sees them.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from forex_python.converter import RatesNotAvailableError, get_rate

from .. import config

LOGGER = logging.getLogger(__name__)

_CURRENCY_RATE_CACHE: dict[tuple[str, int], float] = {}

_SCALES: dict[str, float] = {"billion": 1e9, "million": 1e6}

_NUMBER_RE = re.compile(r"[0-9]+(?:\.[0-9]+)?")


def to_float_range_mean(text: Any) -> tuple[float, float, float]:
    """Convert a string range like '10-20' to (min, max, mean) of all numbers found."""
    if pd.isna(text):
        return np.nan, np.nan, np.nan
    numbers = _NUMBER_RE.findall(str(text))
    if not numbers:
        return np.nan, np.nan, np.nan
    floats = [float(n) for n in numbers]
    return min(floats), max(floats), float(np.mean(floats))


def parse_budget_to_eur(df: pd.DataFrame) -> pd.DataFrame:
    """Parse `total_project_budget` into a numeric EUR column.

    Adds `total_project_budget_eur` (float, NaN if the value could not be parsed).
    """
    df["total_project_budget_eur"] = df.apply(
        lambda row: _parse_single_budget(
            row.get("total_project_budget"),
            row.get("commissioning_year"),
        ),
        axis=1,
    )
    LOGGER.info(
        "Budget parsing: %d / %d rows parsed to EUR",
        int(df["total_project_budget_eur"].notna().sum()),
        len(df),
    )
    return df


def _parse_single_budget(text: Any, commissioning_year: Any) -> float:
    if pd.isna(text):
        return np.nan
    raw = str(text).strip()
    if raw == "":
        return np.nan

    currency = next((c for c in config.SUPPORTED_CURRENCIES if c in raw), None)
    if currency is None:
        LOGGER.warning("No currency token in budget %r", raw)
        return np.nan

    lower = raw.lower()
    scale = next((mult for token, mult in _SCALES.items() if token in lower), None)
    if scale is None:
        LOGGER.warning("No scale (million/billion) in budget %r", raw)
        return np.nan

    numbers = _NUMBER_RE.findall(raw)
    if not numbers:
        return np.nan
    value = float(np.mean([float(n) for n in numbers]))

    rate = _get_currency_rate(currency, commissioning_year)
    if pd.isna(rate):
        return np.nan
    return value * scale * rate


def _get_currency_rate(currency: str, commissioning_year: Any) -> float:
    # Pre-1999 EUR rows are treated as ECU (1:1 with EUR by definition).
    if currency == "EUR":
        return 1.0

    try:
        year = int(float(commissioning_year))
    except (TypeError, ValueError):
        LOGGER.warning("Missing commissioning_year for non-EUR budget — cannot FX")
        return np.nan

    cache_key = (currency, year)
    if cache_key in _CURRENCY_RATE_CACHE:
        return _CURRENCY_RATE_CACHE[cache_key]

    try:
        rate = get_rate(currency, "EUR", datetime(year, 1, 1))
    except RatesNotAvailableError:
        rate = config.CURRENCY_TO_EUR_IN_YEAR.get(str(year), {}).get(currency, np.nan)
        if pd.isna(rate):
            LOGGER.error("No FX rate for %s in %d", currency, year)
    _CURRENCY_RATE_CACHE[cache_key] = rate
    return rate
