"""Quick smoke test for the new preprocessing."""
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

# Force re-run by removing cached file
from src2 import config
if os.path.exists(config.CLEANED_DATASET_PATH):
    os.remove(config.CLEANED_DATASET_PATH)

from src2.data_preprocessing.prep import prepare_data

df = prepare_data()

print("\n--- Sanity checks ---")
print("Rows:", len(df))
print("Columns:", df.columns.tolist())
print("\nturbine_producer counts:")
print(df["turbine_producer"].value_counts(dropna=False).head(15))
print("\nturbine_power_MW describe:")
print(df["turbine_power_MW"].describe())
print("\nfoundation_type counts:")
print(df["foundation_type"].value_counts(dropna=False))
print("\narea_sqkm imputation:")
print(f"  imputed: {df['area_sqkm_imputed'].sum()}")
print(f"  still NaN: {df['area_sqkm'].isna().sum()}")
print(f"  mean area: {df['area_sqkm'].mean():.2f} km²")