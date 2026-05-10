"""Print connection-model assignments for all (country, year) pairs in the dataset
plus a few synthetic future cases."""

import logging

import pandas as pd

from src2 import config
from src2.data_preprocessing.prep import prepare_data

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def main() -> None:
    df = prepare_data()

    # 1) Distribution from the actual cleaned dataset
    print("\n=== Connection regime distribution in cleaned dataset ===")
    cols = ["country", "model_type", "oss_responsibility", "offshore_cable", "onshore_connection"]
    available = [c for c in cols if c in df.columns]
    print(df[available].value_counts(dropna=False).to_string())

    # 2) Per-country, per-year matrix
    print("\n=== Per (country, commissioning_year) → model_type ===")
    pivot = (
        df.dropna(subset=["country", "commissioning_year", "model_type"])
          .assign(commissioning_year=lambda x: x["commissioning_year"].astype(int))
          .groupby(["country", "commissioning_year"])["model_type"]
          .agg(lambda s: s.mode().iat[0] if not s.mode().empty else "?")
          .unstack(fill_value="-")
    )
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(pivot)

    # 3) Synthetic future cases (sanity check rules)
    print("\n=== Synthetic future cases ===")
    cases = [
        ("UK", 2008), ("UK", 2015), ("UK", 2026),
        ("Germany", 2010), ("Germany", 2026),
        ("Netherlands", 2014), ("Netherlands", 2018), ("Netherlands", 2026),
        ("Denmark", 2018), ("Denmark", 2022), ("Denmark", 2028),
        ("Belgium", 2017), ("Belgium", 2020), ("Belgium", 2026),
        ("France", 2022), ("France", 2027),
        ("Sweden", 2018), ("Sweden", 2024),
        ("Poland", 2026), ("Poland", 2030),
        ("Ireland", 2026), ("Ireland", 2031),
        ("Finland", 2025),
        ("Estonia", 2028), ("Latvia", 2030), ("Lithuania", 2030),
        ("Norway", 2030),
        ("Atlantis", 2030),  # unknown
    ]
    rows = []
    for country, year in cases:
        d = config.get_connection_details(country, year)
        rows.append({"country": country, "year": year, **d})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()