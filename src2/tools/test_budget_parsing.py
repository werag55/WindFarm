"""Sanity test for the budget parser using values that actually exist in
`data/european_offshore_wind_capex.csv`.

Run: python -m src2.tools.test_budget_parsing
"""

import logging
import math

from src2.data_preprocessing.parsing import _parse_single_budget

logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")

# Real shapes seen in the raw CSV.
# (raw_budget, year, expected_value_eur or None)  — None means "rate-dependent, just check non-NaN".
CASES = [
    ("EUR 2.5 billion",       2020, 2.5e9),
    ("EUR 1.4 billion",       2024, 1.4e9),
    ("EUR 500 million",       2010, 5.0e8),
    ("EUR 57-78.7 million",   2003, 67.85e6),     # range -> mean
    ("EUR 2-2.5 billion",     2021, 2.25e9),      # range -> mean
    ("GBP 2.5 billion",       2020, None),        # FX-rate dependent
    ("USD 3.64 billion",      2022, None),        # FX-rate dependent
    ("DKK 1600 million",      2003, None),        # FX-rate dependent
    ("EUR 10 million",        1991, 1.0e7),       # pre-1999 EUR (ECU 1:1)
]


def main() -> None:
    print(f"{'Input':<28} {'Year':<6} {'Value EUR':>18}  Verdict")
    print("-" * 70)
    failures = 0
    for raw, year, expected in CASES:
        value = _parse_single_budget(raw, year)
        val_str = f"{value:>18,.0f}" if not math.isnan(value) else f"{'NaN':>18}"
        if expected is None:
            verdict = "OK" if not math.isnan(value) else "FAIL: NaN"
        else:
            verdict = "OK" if abs(value - expected) < 1.0 else f"FAIL: expected {expected:,.0f}"
        if verdict.startswith("FAIL"):
            failures += 1
        print(f"{raw:<28} {year:<6} {val_str}  {verdict}")
    print("-" * 70)
    print(f"{len(CASES) - failures}/{len(CASES)} passed")


if __name__ == "__main__":
    main()
