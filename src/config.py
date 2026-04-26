"""Configuration constants for the offshore wind project profitability analysis pipeline."""

from datetime import datetime


RAW_DATASET_PATH = "data/european_offshore_wind_capex.csv"
CLEANED_DATA_PATH = "data/cleaned_offshore_wind_data.csv"
CALCULATED_DATA_PATH = "data/calculated_offshore_wind_data.csv"
MAP_HTML_PATH = "results/offshore_wind_profitability_map.html"
MODEL_REPORT_PATH = "results/profitability_model_report.json"
MODEL_DASHBOARD_PATH = "results/profitability_model_dashboard.html"

CSV_SEPARATOR = ";"
MISSING_VALUE_TOKENS = {"null", "None", "", "NaN", "nan"}


MANDATORY_COLUMNS = {
    "LAT",
    "LON",
    "installed_capacity_MW",
    "total_project_budget",
    "distance_from_shore_km",
    "project_lifetime_years",
    "foundation_scope",
    "total_project_budget_eur"
}

DEFAULT_VALUES = {
    "project_lifetime_years": 25,
    "foundation_scope": "Unknown",
    "mean_wind_speed_mps": None,
    "mean_wave_height_m": None,
}

#TODO: prepare country -> foundation scope map based on Modele kosztów przyłączenia morskich farm wiatrowych w Europie (ostatnie ~20 lat).pdf
COUNTRY_TO_FOUNDATION_SCOPE = {
    "UK": "TSO_provided",
    "Germany": "TSO_provided",
    "Netherlands": "TSO_provided",
    "Denmark": "TSO_provided",
    "Belgium": "TSO_provided",
}

SUPPORTED_CURRENCIES = {"EUR", "GBP", "USD", "NOK", "DKK", "SEK", "PLN"}

BUDGET_SCALES = {
	"million": 1e6,
	"billion": 1e9,
}
DEFAULT_BUDGET_SCALE = "billion"

GWA_TARGET_ROUGHNESS_M = 0.03
GWA_COORDINATE_ROUND_DECIMALS = 4
HUB_HEIGHT_M = 140.0

CAPACITY_FACTOR_BY_WIND_SPEED = {
	8.5: 0.45,
	9.0: 0.50,
	9.5: 0.55,
	10.0: 0.59,
}

def hours_per_year(year):
    """Calculate hours per year, accounting for leap years."""
    start = datetime(year, 1, 1)
    end = datetime(year + 1, 1, 1)
    return (end - start).total_seconds() / 3600.0

LOSS_FACTOR = 0.15
DISCOUNT_RATE = 0.07
INFLATION_INDEX = 0.02

OPEX_BASE_KEUR_PER_MW = 65.0
OPEX_DISTANCE_CORRECTIONS = [
	(50.0, 5.0),   # up to 50 km
	(100.0, 15.0), # 50–100 km
	(float("inf"), 25.0),  # beyond 100 km
]

TARGET_PROFITABILITY_COLUMN = "lcoe_eur_per_mwh_indexed"

CURRENT_YEAR = datetime.now().year
MAX_DATA_AGE_YEARS = 5

HOURS_PER_YEAR = hours_per_year(CURRENT_YEAR)

FEATURE_COLUMNS = {
    "LAT",
    "LON",
    "installed_capacity_MW",
    "distance_from_shore_km",
    "project_lifetime_years",
    "commissioning_year",
    "mean_wind_speed_mps",
    "water_depth_m",
}

#TODO: country based on LAT, LON if missing
DEFAULT_NEW_SAMPLE = {
	"LAT": 55.0,
	"LON": 0.0,
	"installed_capacity_MW": 800,
	"distance_from_shore_km": 90,
    "project_lifetime_years": 25,
    "foundation_scope": "TSO_provided",
    "area_sqkm": 100.0,
    "country": "UK",
    "commissioning_year": CURRENT_YEAR,
    "water_depth_m": 50.0,
    "mean_wind_speed_mps": None,
    "mean_wave_height_m": None,
}
