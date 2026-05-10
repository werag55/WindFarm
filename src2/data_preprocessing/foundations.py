"""Foundation type normalization + rare-category consolidation."""

import logging
import re

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

# Order matters: more specific patterns first.
# Each pattern is matched (case-insensitive, substring) against the raw value.
_FOUNDATION_RULES = [
    # Floating variants
    ("Floating",        re.compile(r"floating|semi[\s-]?sub|spar|tlp", re.I)),
    # Suction bucket (must come BEFORE jacket because "Suction Bucket Jacket" exists)
    ("Suction bucket",  re.compile(r"suction|bucket", re.I)),
    # Tripile (BARD)
    ("Tripile",         re.compile(r"tripile", re.I)),
    # Tripod (and "Tripod jacket")
    ("Tripod",          re.compile(r"tripod", re.I)),
    # Jacket
    ("Jacket",          re.compile(r"jacket", re.I)),
    # Gravity base / GBS / gravity-based
    ("Gravity-based",   re.compile(r"gravity|\bgbs\b", re.I)),
    # Monopile (last because it's the most generic name)
    ("Monopile",        re.compile(r"mono[\s-]?pile", re.I)),
]


def _normalize_one(value) -> str | float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text == "" or text.lower() in ("nan", "none", "null", "unknown"):
        return np.nan
    for canonical, pat in _FOUNDATION_RULES:
        if pat.search(text):
            return canonical
    LOGGER.debug("Unrecognised foundation_type value: %r -> kept as-is", text)
    return text


def normalize_foundation_type(df: pd.DataFrame) -> pd.DataFrame:
    """Map free-text `foundation_type` values to a canonical vocabulary."""
    if "foundation_type" not in df.columns:
        LOGGER.warning("normalize_foundation_type: column missing")
        return df

    before = df["foundation_type"].astype(str).str.strip().value_counts(dropna=False)
    df["foundation_type"] = df["foundation_type"].apply(_normalize_one)
    after = df["foundation_type"].value_counts(dropna=False)

    LOGGER.info(
        "normalize_foundation_type: %d unique -> %d unique categories",
        before.shape[0],
        after.shape[0],
    )
    return df


def consolidate_rare_foundation_types(
    df: pd.DataFrame, min_count: int = 3, other_label: str = "Other"
) -> pd.DataFrame:
    """Replace foundation types occurring fewer than `min_count` times with `Other`."""
    if "foundation_type" not in df.columns:
        return df

    counts = df["foundation_type"].value_counts(dropna=True)
    rare = counts[counts < min_count].index.tolist()

    if rare:
        mask = df["foundation_type"].isin(rare)
        n_affected = int(mask.sum())
        df.loc[mask, "foundation_type"] = other_label
        LOGGER.info(
            "consolidate_rare_foundation_types: collapsed %d categories "
            "(%d rows) into '%s': %s",
            len(rare),
            n_affected,
            other_label,
            rare,
        )
    else:
        LOGGER.info(
            "consolidate_rare_foundation_types: no categories below threshold (%d)",
            min_count,
        )

    final_counts = df["foundation_type"].value_counts(dropna=False).to_dict()
    LOGGER.info("Final foundation_type distribution: %s", final_counts)
    return df