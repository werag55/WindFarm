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


def parse_turbine_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Split raw turbine model text into producer and rated power features."""
    if "turbine_model" not in df.columns:
        return df

    parsed = df["turbine_model"].apply(_parse_turbine_model).apply(pd.Series)

    if "turbine_producer" not in df.columns:
        df["turbine_producer"] = np.nan
    if "turbine_power_MW" not in df.columns:
        df["turbine_power_MW"] = np.nan

    existing_producer = df["turbine_producer"].replace(list(config.MISSING_VALUE_TOKENS), np.nan)
    df["turbine_producer"] = parsed["turbine_producer"].fillna(existing_producer)
    df["turbine_power_MW"] = pd.to_numeric(df["turbine_power_MW"], errors="coerce").fillna(
        parsed["turbine_power_MW"]
    )
    return df


def _parse_turbine_model(model: object) -> dict[str, object]:
    """Infer turbine producer and MW rating from common offshore turbine names."""
    return {
        "turbine_producer": _infer_turbine_producer_from_model(model),
        "turbine_power_MW": _infer_turbine_power_from_model(model),
    }


def _infer_turbine_producer_from_model(model: object) -> str | float:
    """Infer the turbine producer from common model naming patterns."""
    if pd.isna(model):
        return np.nan

    model_text = str(model)
    if model_text.strip().lower() in config.MISSING_VALUE_TOKENS:
        return np.nan

    producer_patterns = [
        ("Siemens Gamesa", r"\bSiemens\s+Gamesa\b"),
        ("MHI Vestas", r"\bMHI\s+Vestas\b"),
        ("Vestas", r"\bVestas\b"),
        ("Siemens", r"\bSiemens\b"),
        ("GE", r"\bGE\b|\bHaliade\b"),
        ("Senvion", r"\bSenvion\b|\bREpower\b"),
        ("Adwen", r"\bAdwen\b"),
        ("BARD", r"\bBARD\b"),
        ("AREVA", r"\bAREVA\b"),
        ("Bonus", r"\bBonus\b"),
        ("MingYang", r"\bMingYang\b|\bMySE\b"),
        ("Wind World", r"\bWind\s+World\b"),
        ("Enron Wind", r"\bEnron\s+Wind\b"),
        ("Alstom", r"\bAlstom\b"),
    ]
    producers = []
    for producer, pattern in producer_patterns:
        if re.search(pattern, model_text, flags=re.IGNORECASE):
            producers.append(producer)
            model_text = re.sub(pattern, "", model_text, flags=re.IGNORECASE)
    if not producers:
        return np.nan
    if len(producers) == 1:
        return producers[0]
    return "Multiple"


def _infer_turbine_power_from_model(model: object) -> float:
    """Infer MW rating from common offshore turbine model naming patterns."""
    if pd.isna(model):
        return np.nan

    model_text = str(model)
    if model_text.strip().lower() in config.MISSING_VALUE_TOKENS:
        return np.nan

    values = [float(value) for value in re.findall(r"(\d+(?:\.\d+)?)\s*MW\b", model_text, flags=re.IGNORECASE)]
    values.extend(
        float(value) / 1000.0
        for value in re.findall(r"(\d+(?:\.\d+)?)\s*kW\b", model_text, flags=re.IGNORECASE)
    )
    values.extend(float(value) for value in re.findall(r"\bSWT-(\d+(?:\.\d+)?)", model_text, flags=re.IGNORECASE))
    values.extend(float(value) for value in re.findall(r"\bSWP-(\d+(?:\.\d+)?)", model_text, flags=re.IGNORECASE))
    values.extend(float(value) for value in re.findall(r"\bSG\s*(\d+(?:\.\d+)?)", model_text, flags=re.IGNORECASE))
    values.extend(float(value) for value in re.findall(r"\bV\d+-(\d+(?:\.\d+)?)", model_text, flags=re.IGNORECASE))
    values.extend(float(value) for value in re.findall(r"\bAD\s*(\d+(?:\.\d+)?)-", model_text, flags=re.IGNORECASE))
    values.extend(float(value) for value in re.findall(r"\bMySE\s*(\d+(?:\.\d+)?)-", model_text, flags=re.IGNORECASE))

    for pattern in [
        r"\bREpower\s+(\d+(?:\.\d+)?)M\b",
        r"\bSenvion\s+(\d+(?:\.\d+)?)M\d+\b",
        r"\bBARD\s+(\d+(?:\.\d+)?)\b",
        r"\bAREVA\s+M(\d{4})\b",
    ]:
        for match in re.finditer(pattern, model_text, flags=re.IGNORECASE):
            value = float(match.group(1))
            values.append(value / 1000.0 if value >= 100 else value)

    plausible_values = [value for value in values if 0.1 <= value <= 20.0]
    if not plausible_values:
        return np.nan
    return float(np.mean(plausible_values))


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
