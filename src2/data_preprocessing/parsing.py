"""Parsing and normalization helpers for budget and range columns."""

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

# Recognised scale tokens → multiplier in EUR units.
# Order matters: longer first so "billion" matches before "bn", "million" before "m".
_SCALE_TOKENS: list[tuple[re.Pattern, float]] = [
    (re.compile(r"\bbillion\b", re.I), 1e9),
    (re.compile(r"\bbn\b",      re.I), 1e9),
    (re.compile(r"\bmilliard\b",re.I), 1e9),  # FR/DE term occasionally seen
    (re.compile(r"\bmillion\b", re.I), 1e6),
    (re.compile(r"\bmln\b",     re.I), 1e6),
    (re.compile(r"\bmio\b",     re.I), 1e6),
    (re.compile(r"\bm\b",       re.I), 1e6),
    # bare "k" / "thousand" intentionally skipped — too ambiguous in budget context
]

# Detect combined / aggregated budgets: "(combined 2+3)", "combined phases I+II", "phases 1+2"
_COMBINED_PATTERNS = [
    re.compile(r"combined", re.I),
    re.compile(r"\bphases?\b.*[+&]", re.I),
    # Number+number INSIDE parentheses: "(2+3)", "(I+II)"
    re.compile(r"\(\s*[\divxIVX]+\s*[+&]\s*[\divxIVX]+", re.I),
]

# Budget parsing notes for the quality flag column
NOTE_OK              = "ok"
NOTE_OK_RANGE        = "ok_range_mean"
NOTE_PRE_1999_EUR    = "pre_1999_eur_assumed"
NOTE_COMBINED        = "combined_budget"
NOTE_NO_VALUE        = "no_numeric_value"
NOTE_NO_CURRENCY     = "no_currency_defaulted_eur"
NOTE_NO_RATE         = "no_fx_rate"
NOTE_EMPTY           = "empty_input"


# ---------------------------------------------------------------------------
# Public range helper (unchanged behaviour, kept for backward compatibility)
# ---------------------------------------------------------------------------

def to_float_range_mean(text: Any) -> tuple[float, float, float]:
    """Convert a string range like '10-20' to (min, max, mean) of all numbers found."""
    if pd.isna(text):
        return np.nan, np.nan, np.nan
    numbers = re.findall(r"[0-9]+(?:\.[0-9]+)?", str(text).replace(",", "."))
    if not numbers:
        return np.nan, np.nan, np.nan
    floats = [float(n) for n in numbers]
    return min(floats), max(floats), float(np.mean(floats))


# ---------------------------------------------------------------------------
# Budget parsing — main entry
# ---------------------------------------------------------------------------

def parse_budget_to_eur(df: pd.DataFrame) -> pd.DataFrame:
    """Parse `total_project_budget` strings into EUR + quality flags.

    Adds columns:
      - total_project_budget_eur          : float | NaN
      - budget_parse_failed               : bool
      - combined_budget_flag              : bool
      - pre_1999_eur_assumed              : bool
      - budget_parse_note                 : short string explaining outcome
    """
    parsed = df.apply(
        lambda row: pd.Series(
            _parse_single_budget(
                row.get("total_project_budget"),
                row.get("commissioning_year"),
            )
        ),
        axis=1,
    )

    df["total_project_budget_eur"] = parsed["value_eur"]
    df["budget_parse_failed"]      = parsed["failed"].astype(bool)
    df["combined_budget_flag"]     = parsed["combined"].astype(bool)
    df["pre_1999_eur_assumed"]     = parsed["pre_1999_eur"].astype(bool)
    df["budget_parse_note"]        = parsed["note"]

    _log_parse_summary(df)
    return df


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _parse_single_budget(text: Any, commissioning_year: Any) -> dict[str, Any]:
    """Parse a single budget cell. Returns a dict with value + flags."""
    result = {
        "value_eur":     np.nan,
        "failed":        False,
        "combined":      False,
        "pre_1999_eur":  False,
        "note":          NOTE_OK,
    }

    # ---- Empty / NaN input ----
    if text is None or (isinstance(text, float) and np.isnan(text)):
        result["failed"] = True
        result["note"] = NOTE_EMPTY
        return result

    raw = str(text).strip()
    if raw == "" or raw.lower() in {"nan", "none", "null"}:
        result["failed"] = True
        result["note"] = NOTE_EMPTY
        return result

    # ---- Combined / aggregated budgets ----
    if any(p.search(raw) for p in _COMBINED_PATTERNS):
        result["combined"] = True
        result["failed"]   = True   # don't trust as a single-project target
        result["note"]     = NOTE_COMBINED
        LOGGER.debug("Combined budget detected: %r", raw)
        return result

    # ---- Year resolution ----
    try:
        year = int(float(commissioning_year)) if pd.notna(commissioning_year) \
            else datetime.now().year
    except (TypeError, ValueError):
        year = datetime.now().year

    # ---- Currency detection ----
    currency = next(
        (cur for cur in config.SUPPORTED_CURRENCIES if cur in raw),
        None,
    )
    if currency is None:
        LOGGER.warning("Missing currency in budget value '%s'; defaulting to EUR", raw)
        currency = "EUR"
        result["note"] = NOTE_NO_CURRENCY

    # ---- pre-1999 EUR special case ----
    if currency == "EUR" and year < 1999:
        result["pre_1999_eur"] = True
        # Treat as ECU which had a 1:1 conversion ratio with EUR by definition.
        # Keep rate = 1.0 and continue parsing.
        rate = 1.0
        if result["note"] == NOTE_OK:
            result["note"] = NOTE_PRE_1999_EUR
    else:
        rate = _get_currency_rate(currency, year)
        if pd.isna(rate):
            result["failed"] = True
            result["note"] = NOTE_NO_RATE
            return result

    # ---- Scale detection ----
    scale_value = _detect_scale(raw)

    # ---- Numeric value extraction ----
    # Normalise European decimal comma → period before extracting numbers.
    # But careful: "1,200" could be a thousands separator. Heuristic:
    #   - if a comma is followed by exactly 3 digits AND no period elsewhere,
    #     treat as thousands separator (drop it).
    #   - otherwise treat as decimal separator (replace with period).
    cleaned = _normalise_decimal_separators(raw)

    numbers = re.findall(r"[0-9]+(?:\.[0-9]+)?", cleaned)
    if not numbers:
        result["failed"] = True
        result["note"] = NOTE_NO_VALUE
        return result

    floats = [float(n) for n in numbers]
    # If multiple numbers were found AND they look like a range (e.g. "1.2-1.6"),
    # use their mean and annotate.
    if len(floats) > 1:
        mean_value = float(np.mean(floats))
        if result["note"] in (NOTE_OK, NOTE_PRE_1999_EUR):
            result["note"] = (
                NOTE_OK_RANGE if result["note"] == NOTE_OK
                else f"{NOTE_PRE_1999_EUR}+{NOTE_OK_RANGE}"
            )
    else:
        mean_value = floats[0]

    result["value_eur"] = mean_value * scale_value * rate
    return result


def _detect_scale(raw: str) -> float:
    """Find scale token (billion/million/bn/m...). Falls back to default."""
    for pat, mult in _SCALE_TOKENS:
        if pat.search(raw):
            return mult
    # Default if no scale specified
    default = config.BUDGET_SCALES.get(config.DEFAULT_BUDGET_SCALE, 1e9)
    return default


def _normalise_decimal_separators(raw: str) -> str:
    """Replace European decimal commas with periods, preserving thousands separators.

    Heuristics:
      - "1,200" (comma followed by exactly 3 digits, no period in token) -> drop comma
      - "2,5"   (comma followed by 1-2 digits)                           -> "2.5"
      - "1.234.567" / "1,234,567" mixed                                  -> drop thousands
    """
    # First handle pure thousands: digit groups separated by commas with 3 digits
    # E.g. "1,200,000" -> "1200000". But only if NO decimal point already present.
    if "." not in raw:
        raw = re.sub(r"(?<=\d),(?=\d{3}\b)", "", raw)
    # Now any remaining commas are decimal separators
    return raw.replace(",", ".")


def _get_currency_rate(currency: str, year: int) -> float:
    """Get currency-to-EUR conversion rate for a given year."""
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
        LOGGER.warning(
            "Could not get FX rate for %s in %d, trying static fallback",
            currency, year,
        )
        try:
            rate = config.CURRENCY_TO_EUR_IN_YEAR[str(year)][currency]
            _CURRENCY_RATE_CACHE[cache_key] = rate
            return rate
        except KeyError:
            LOGGER.error("No fallback FX rate for %s in %d", currency, year)
            return np.nan


def _log_parse_summary(df: pd.DataFrame) -> None:
    """Log a one-shot summary of budget parsing outcomes."""
    n_total    = len(df)
    n_failed   = int(df["budget_parse_failed"].sum())
    n_combined = int(df["combined_budget_flag"].sum())
    n_pre1999  = int(df["pre_1999_eur_assumed"].sum())
    n_ok       = n_total - n_failed

    LOGGER.info(
        "Budget parsing: ok=%d, failed=%d, combined=%d, pre_1999_eur=%d (total=%d)",
        n_ok, n_failed, n_combined, n_pre1999, n_total,
    )

    if n_failed:
        failed = df[df["budget_parse_failed"]]
        for _, row in failed.iterrows():
            name = row.get("wind_farm_name", "?")
            raw = row.get("total_project_budget", "")
            note = row.get("budget_parse_note", "")
            LOGGER.warning(
                "  Unparseable budget: farm=%r raw=%r note=%s",
                name, raw, note,
            )

    note_counts = df["budget_parse_note"].value_counts(dropna=False).to_dict()
    LOGGER.info("Budget parse-note distribution: %s", note_counts)