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

# Longest prefixes first so multi-word vendors win over their single-word forms.
TURBINE_PRODUCER_PREFIXES = [
    ("Siemens Gamesa", "Siemens Gamesa"),
    ("MHI Vestas", "Vestas"),
    ("Wind World", "Wind World"),
    ("Enron Wind", "Enron Wind"),
    ("Siemens", "Siemens"),
    ("Vestas", "Vestas"),
    ("MingYang", "MingYang"),
    ("Senvion", "Senvion"),
    ("REpower", "Senvion"),
    ("Adwen", "Adwen"),
    ("AREVA", "AREVA"),
    ("BARD", "BARD"),
    ("Bonus", "Bonus"),
    ("Alstom", "Alstom"),
    ("GE", "GE"),
]


def _is_missing(value) -> bool:
    """True if value is NaN or a missing-token string from the raw CSV."""
    if pd.isna(value):
        return True
    return str(value).strip().lower() in {"", "nan", "null", "none"}


def _parse_turbine_producer(model: str):
    """Return canonical producer name parsed from turbine_model, or None."""
    if _is_missing(model):
        return None
    text = str(model).strip()
    for prefix, canonical in TURBINE_PRODUCER_PREFIXES:
        if text.lower().startswith(prefix.lower()):
            return canonical
    return None


def _parse_turbine_power_mw(model: str):
    """Return turbine nameplate power in MW parsed from turbine_model, or NaN."""
    if _is_missing(model):
        return np.nan
    text = str(model)
    # kW form (old turbines): "550kW", "550 kW"
    match = re.search(r"(\d+(?:\.\d+)?)\s*kW", text, re.IGNORECASE)
    if match:
        return float(match.group(1)) / 1000.0
    # Explicit MW: "8.0 MW", "8MW", "10 MW class"
    match = re.search(r"(\d+(?:\.\d+)?)\s*MW", text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    # First decimal number in the MW range: "SWT-7.0-154", "V164-8.0", "BARD 5.0", "MySE3.0-135"
    for token in re.findall(r"\d+\.\d+", text):
        value = float(token)
        if 0.5 <= value <= 25:
            return value
    # Standalone small integer: "Senvion 5M", "AD 5-135"
    for token in re.findall(r"(?<![A-Za-z\d])(\d{1,2})(?=M\b|-\d|\s|$)", text):
        value = float(token)
        if 1 <= value <= 25:
            return value
    return np.nan


def extract_turbine_info(df: pd.DataFrame) -> pd.DataFrame:
    """Fill turbine_producer and turbine_power_MW by parsing turbine_model."""
    if "turbine_model" not in df.columns:
        return df

    producer_from_model = df["turbine_model"].apply(_parse_turbine_producer)
    power_from_model = df["turbine_model"].apply(_parse_turbine_power_mw)

    if "turbine_producer" in df.columns:
        df["turbine_producer"] = df["turbine_producer"].where(
            df["turbine_producer"].apply(lambda v: not _is_missing(v)),
            producer_from_model,
        )
    else:
        df["turbine_producer"] = producer_from_model

    if "turbine_power_MW" in df.columns:
        existing = pd.to_numeric(df["turbine_power_MW"], errors="coerce")
        df["turbine_power_MW"] = existing.where(existing.notna(), power_from_model)
    else:
        df["turbine_power_MW"] = power_from_model

    LOGGER.info(
        "Turbine info: %d/%d producer filled, %d/%d power filled",
        int(df["turbine_producer"].apply(lambda v: not _is_missing(v)).sum()),
        len(df),
        int(pd.to_numeric(df["turbine_power_MW"], errors="coerce").notna().sum()),
        len(df),
    )
    return df


def fill_missing_budget_eur(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing total_project_budget_eur with median EUR/MW × installed_capacity_MW."""
    if "total_project_budget_eur" not in df.columns or "installed_capacity_MW" not in df.columns:
        return df

    budget = pd.to_numeric(df["total_project_budget_eur"], errors="coerce")
    capacity = pd.to_numeric(df["installed_capacity_MW"], errors="coerce")
    known = budget.notna() & capacity.notna() & (capacity > 0)
    if not known.any():
        return df

    median_eur_per_mw = (budget[known] / capacity[known]).median()
    LOGGER.info("Imputing missing budgets using median = %.0f EUR/MW", median_eur_per_mw)

    fill_mask = budget.isna() & capacity.notna()
    df.loc[fill_mask, "total_project_budget_eur"] = capacity[fill_mask] * median_eur_per_mw
    df.loc[fill_mask, "total_project_budget"] = df.loc[fill_mask, "total_project_budget"].fillna("EUR (imputed)")
    LOGGER.info("Imputed budget for %d rows", int(fill_mask.sum()))
    return df


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
