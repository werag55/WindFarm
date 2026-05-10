"""Standalone sanity test for budget parser. Run: python -m src2.tools.test_budget_parsing"""

import logging
import math

from src2.data_preprocessing.parsing import _parse_single_budget

logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")

CASES = [
    # (raw_budget, year, expected_value_eur or None, expected_flags)
    ("2.5 billion EUR",         2020, 2.5e9,  {"failed": False, "combined": False, "pre_1999_eur": False}),
    ("2,5 billion EUR",         2020, 2.5e9,  {"failed": False, "combined": False, "pre_1999_eur": False}),
    ("800 million GBP",         2020, None,   {"failed": False, "combined": False, "pre_1999_eur": False}),
    ("1.2-1.6 billion EUR",     2021, 1.4e9,  {"failed": False, "combined": False, "pre_1999_eur": False}),
    ("1,2-1,6 bn EUR",          2021, 1.4e9,  {"failed": False, "combined": False, "pre_1999_eur": False}),
    ("PLN 30 billion (combined 2+3)", 2024, None, {"failed": True, "combined": True, "pre_1999_eur": False}),
    ("500 million EUR",         1995, 5e8,    {"failed": False, "combined": False, "pre_1999_eur": True}),
    ("",                        2020, None,   {"failed": True}),
    ("USD 4.0 bn",              2018, None,   {"failed": False}),  # fx rate dependent
    ("EUR 1,234,567,890",       2010, 1.234567890e9, {"failed": False}),  # thousands separators
    ("3 bn",                    2020, 3e9,    {"failed": False, "pre_1999_eur": False}),  # default EUR
    ("(2+3) phases combined",   2022, None,   {"failed": True, "combined": True}),
]

def main() -> None:
    print(f"{'Input':<40} {'Year':<6} {'Value EUR':>18}  Flags")
    print("-" * 95)
    for raw, year, expected_val, expected_flags in CASES:
        out = _parse_single_budget(raw, year)
        val_str = f"{out['value_eur']:>18,.0f}" if out['value_eur'] == out['value_eur'] else f"{'NaN':>18}"
        flags_str = ", ".join(
            f"{k}={out[k]}" for k in ("failed", "combined", "pre_1999_eur")
        )
        verdict = "OK"
        if expected_val is not None:
            if math.isnan(out["value_eur"]) or abs(out["value_eur"] - expected_val) > 1.0:
                verdict = f" expected {expected_val:,.0f}"
        for k, v in expected_flags.items():
            if out.get(k) != v:
                verdict = f" flag {k} expected {v} got {out.get(k)}"
                break
        print(f"{raw[:40]:<40} {year:<6} {val_str}  {flags_str}  [{out['note']}]  {verdict}")

if __name__ == "__main__":
    main()