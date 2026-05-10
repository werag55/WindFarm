"""Parsing turbine_model into producer + power_MW."""

import logging
import re

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

# Order matters: longer / more specific names first to avoid "Vestas" matching
# inside "MHI Vestas".
_PRODUCERS = [
    ("Siemens Gamesa", re.compile(r"\bsiemens\s+gamesa\b", re.I)),
    ("MHI Vestas",     re.compile(r"\bmhi\s+vestas\b", re.I)),
    ("Siemens",        re.compile(r"\bsiemens\b", re.I)),
    ("Vestas",         re.compile(r"\bvestas\b", re.I)),
    ("GE",             re.compile(r"\b(ge|general\s+electric)\b", re.I)),
    ("AREVA",          re.compile(r"\bareva\b", re.I)),
    ("Adwen",          re.compile(r"\badwen\b", re.I)),
    ("Senvion",        re.compile(r"\b(senvion|repower)\b", re.I)),
    ("BARD",           re.compile(r"\bbard\b", re.I)),
    ("Bonus",          re.compile(r"\bbonus\b", re.I)),
    ("Nordex",         re.compile(r"\bnordex\b", re.I)),
    ("Enercon",        re.compile(r"\benercon\b", re.I)),
    ("Goldwind",       re.compile(r"\bgoldwind\b", re.I)),
    ("Mingyang",       re.compile(r"\bmingyang\b", re.I)),
]

# Power patterns, tried in order. All must yield MW (not kW).
_POWER_PATTERNS = [
    # "8.0 MW", "13 MW", "9.525 MW"
    re.compile(r"(\d+(?:\.\d+)?)\s*MW\b", re.I),
    # "SG 14-222", "SG 8.0-167", "V164-9.5", "V164-10.0", "SWT-7.0-154"
    # capture number BEFORE the dash that separates power from rotor diameter
    re.compile(r"[A-Z]{1,4}\s*(\d{1,2}(?:\.\d+)?)-\d{2,3}", re.I),
    # "SWT-3.6-107" -> 3.6
    re.compile(r"-\s*(\d{1,2}\.\d+)\s*-\s*\d{2,3}\b"),
    # "BARD 5.0", "Bonus 2MW" already handled above; "Vestas V39" (kW) skipped intentionally
    # "5M" (Senvion 5M) -> 5
    re.compile(r"\b(\d{1,2})M\b"),
]


def _extract_producer(model: str) -> str | None:
    if not isinstance(model, str):
        return None
    # If the field contains "/" (multi-turbine farm), take the FIRST one.
    head = model.split("/")[0].strip()
    for name, pat in _PRODUCERS:
        if pat.search(head):
            return name
    return None


def _extract_power_mw(model: str) -> float | None:
    if not isinstance(model, str):
        return None
    head = model.split("/")[0].strip()
    for pat in _POWER_PATTERNS:
        m = pat.search(head)
        if m:
            try:
                value = float(m.group(1))
            except ValueError:
                continue
            # Sanity: offshore turbines are 1–25 MW
            if 1.0 <= value <= 25.0:
                return value
    return None


def split_turbine_model(df: pd.DataFrame) -> pd.DataFrame:
    """Populate `turbine_producer` and `turbine_power_MW` from `turbine_model`.

    - Original `turbine_model` is preserved.
    - Existing non-null values in target columns are NOT overwritten.
    - Multi-turbine entries ("model A / model B") use the first listed model.
    """
    if "turbine_model" not in df.columns:
        LOGGER.warning("split_turbine_model: 'turbine_model' column missing, skipping")
        return df

    if "turbine_producer" not in df.columns:
        df["turbine_producer"] = pd.Series(index=df.index, dtype="object")
    else:
        df["turbine_producer"] = df["turbine_producer"].astype("object")

    if "turbine_power_MW" not in df.columns:
        df["turbine_power_MW"] = np.nan
    else:
        df["turbine_power_MW"] = pd.to_numeric(df["turbine_power_MW"], errors="coerce")

    producer_filled = 0
    power_filled = 0

    for idx, model in df["turbine_model"].items():
        if not isinstance(model, str) or model.strip().lower() in ("", "nan", "none"):
            continue

        if pd.isna(df.at[idx, "turbine_producer"]) or \
                str(df.at[idx, "turbine_producer"]).strip() == "":
            producer = _extract_producer(model)
            if producer is not None:
                df.at[idx, "turbine_producer"] = producer
                producer_filled += 1

        existing_power = df.at[idx, "turbine_power_MW"]
        if pd.isna(existing_power) or str(existing_power).strip() == "":
            power = _extract_power_mw(model)
            if power is not None:
                df.at[idx, "turbine_power_MW"] = power
                power_filled += 1

    # Coerce power column to numeric
    df["turbine_power_MW"] = pd.to_numeric(df["turbine_power_MW"], errors="coerce")

    LOGGER.info(
        "split_turbine_model: filled producer=%d, power=%d (out of %d non-null models)",
        producer_filled,
        power_filled,
        df["turbine_model"].notna().sum(),
    )
    return df