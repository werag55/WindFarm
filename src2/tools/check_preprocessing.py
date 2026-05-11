"""Force a fresh run of the preprocessing pipeline and verify the result.

Removes the cached CSV so the pipeline is exercised end-to-end, then prints
a short summary of the cleaned dataset and runs `src2.utils.checks` to flag
anything missing, stale, or inconsistent.

Run: python -m src2.tools.check_preprocessing
"""

import logging
import os

from src2 import config
from src2.data_preprocessing.prep import prepare_data
from src2.utils.checks import run_all_checks

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

if os.path.exists(config.CLEANED_DATASET_PATH):
    os.remove(config.CLEANED_DATASET_PATH)

df = prepare_data()

print("\n--- Snapshot ---")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
if "turbine_producer" in df.columns:
    print("\nturbine_producer counts (top 10):")
    print(df["turbine_producer"].value_counts(dropna=False).head(10))
if "foundation_type" in df.columns:
    print("\nfoundation_type counts:")
    print(df["foundation_type"].value_counts(dropna=False))

run_all_checks(df)
