""""""

from datetime import datetime


RAW_DATASET_PATH = "data/european_offshore_wind_capex.csv"
CLEANED_DATASET_PATH = "data/cleaned_european_offshore_wind_capex.csv"
CALCULATED_DATASET_PATH = "data/calculated_offshore_wind_data.csv"

CSV_SEPARATOR = ";"
MISSING_VALUE_TOKENS = {"null", "None", "", "NaN", "nan"}

MANDATORY_COLUMNS = {
    "LAT",
    "LON",
    "installed_capacity_MW",
}

# Project doc, section 4: założona gęstość mocy 8 MW/km²
POWER_DENSITY_MW_PER_SQKM = 8.0

# Merge near-duplicate foundation_type labels before one-hot encoding
FOUNDATION_TYPE_MAPPING = {
    "Gravity base": "Gravity-based",
    "Floating": "Floating",
    "Floating TLP": "Floating",
    "Floating TetraSpar": "Floating",
    "Floating SATH": "Floating",
    "Floating concrete spar buoy": "Floating",
    "Floating damping pool": "Floating",
    "Floating semi-submersible": "Floating",
    "Floating spar buoy": "Floating",
}

NUMERIC_COLUMNS = [
    "LAT",
    "LON",
    "area_sqkm",
    "commissioning_year",
    "installed_capacity_MW",
    "turbine_power_MW",
    "water_depth_min_m",
    "water_depth_max_m",
    "water_depth_mean_m",
    "distance_from_shore_min_km",
    "distance_from_shore_max_km",
    "distance_from_shore_mean_km",
    "distance_from_port_km",
    "total_project_budget_eur",
    "total_project_budget_eur_indexed",
    "project_lifetime_years",
    "mean_wind_speed_mps",
    "mean_wave_height_m",
]

CATEGORICAL_COLUMNS = [
    "country",
    "turbine_producer",
    "foundation_type",
    "oss_responsibility",
    "offshore_cable",
    "onshore_connection",
    "model_type"
]

def get_connection_details(country: str, year: float) -> dict:
    """Determine OSS Responsibility, Offshore Cable, Onshore Connection, and Model Type."""
    unknown = {"oss_responsibility": "Unknown", "offshore_cable": "Unknown", "onshore_connection": "Unknown", "model_type": "Unknown"}
    if not country or str(country).strip().lower() in {"", "nan", "none", "unknown"}:
        return unknown

    c = str(country).strip()
    year_known = year is not None and str(year) != "nan"
    y = int(year) if year_known else None

    # Year-independent countries: resolve immediately so rows without commissioning_year still get mapped.
    if c == "Germany":
        return {"oss_responsibility": "TSO", "offshore_cable": "TSO", "onshore_connection": "TSO", "model_type": "TSO-led"}
    if c == "France":
        return {"oss_responsibility": "TSO", "offshore_cable": "TSO", "onshore_connection": "TSO", "model_type": "TSO-led"}
    if c == "Norway":
        return {"oss_responsibility": "Developer", "offshore_cable": "Developer", "onshore_connection": "Developer", "model_type": "Developer-led"}
    if c == "Poland":
        return {"oss_responsibility": "Developer", "offshore_cable": "Developer", "onshore_connection": "Developer/TSO", "model_type": "Developer-led"}
    if c == "Finland":
        return {"oss_responsibility": "Developer", "offshore_cable": "Developer", "onshore_connection": "Developer", "model_type": "Developer-led"}
    if c == "Portugal":
        return {"oss_responsibility": "Developer", "offshore_cable": "Developer", "onshore_connection": "Developer", "model_type": "Developer-led"}
    if c == "Italy":
        return {"oss_responsibility": "Developer", "offshore_cable": "Developer", "onshore_connection": "Developer", "model_type": "Developer-led"}

    # Year-dependent countries — Unknown if commissioning_year is missing.
    if not year_known:
        return unknown

    if c == "UK":
        if y < 2011: return {"oss_responsibility": "Developer", "offshore_cable": "Developer", "onshore_connection": "Developer", "model_type": "Developer-led"}
        return {"oss_responsibility": "OFTO", "offshore_cable": "OFTO", "onshore_connection": "OFTO", "model_type": "OFTO"}
    if c == "Netherlands":
        if y < 2016: return {"oss_responsibility": "Developer", "offshore_cable": "Developer", "onshore_connection": "TSO", "model_type": "Developer-led"}
        return {"oss_responsibility": "TSO", "offshore_cable": "TSO", "onshore_connection": "TSO", "model_type": "TSO-led"}
    if c == "Denmark":
        if y <= 2020: return {"oss_responsibility": "TSO", "offshore_cable": "TSO", "onshore_connection": "TSO", "model_type": "TSO-led"}
        return {"oss_responsibility": "Developer", "offshore_cable": "Developer", "onshore_connection": "Developer", "model_type": "Developer-led"}
    if c == "Belgium":
        if y < 2019: return {"oss_responsibility": "Developer", "offshore_cable": "Developer", "onshore_connection": "TSO", "model_type": "Developer-led"}
        return {"oss_responsibility": "TSO", "offshore_cable": "TSO", "onshore_connection": "TSO", "model_type": "TSO-led"}
    if c == "Sweden":
        if y < 2021: return {"oss_responsibility": "Developer", "offshore_cable": "Developer", "onshore_connection": "Developer", "model_type": "Developer-led"}
        return {"oss_responsibility": "TSO", "offshore_cable": "TSO", "onshore_connection": "TSO", "model_type": "TSO-led"}
    if c == "Ireland":
        # EirGrid is moving toward TSO-led offshore grid planning; 2031 used as the practical cutoff.
        if y < 2031: return {"oss_responsibility": "Developer", "offshore_cable": "Developer", "onshore_connection": "Developer", "model_type": "Developer-led"}
        return {"oss_responsibility": "TSO", "offshore_cable": "TSO", "onshore_connection": "TSO", "model_type": "TSO-led"}

    return unknown

RANGE_COLUMNS = {
    "water_depth_m": "m",
    "distance_from_shore_km": "km",
}

DEFAULT_VALUES = {
    "project_lifetime_years": 25,
    "mean_wind_speed_mps": None,
    "mean_wave_height_m": None,
    "distance_from_port_km": None,
}

SUPPORTED_CURRENCIES = {"EUR", "GBP", "USD", "NOK", "DKK", "SEK", "PLN"}
#TODO: how to handle it? EUR didn't exist before 1999
CURRENCY_TO_EUR_IN_YEAR = {
    "1995": {
        "GBP": 1.25
    },
    "1998": {
        "GBP": 1.49
    },
}

BUDGET_SCALES = {
	"million": 1e6,
	"billion": 1e9,
}
DEFAULT_BUDGET_SCALE = "billion"

GWA_TARGET_ROUGHNESS_M = 0.03
GWA_COORDINATE_ROUND_DECIMALS = 4
HUB_HEIGHT_M = 140.0

COASTLINE_URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_coastline.geojson"

CAPACITY_FACTOR_BY_WIND_SPEED = {
	8.5: 0.45,
	9.0: 0.50,
	9.5: 0.55,
	10.0: 0.59,
}

LOSS_FACTOR = 0.15
DISCOUNT_RATE = 0.07
INFLATION_INDEX = 0.02

OPEX_BASE_KEUR_PER_MW = 65.0
OPEX_DISTANCE_CORRECTIONS = [
	(50.0, 5.0),   # up to 50 km
	(100.0, 15.0), # 50–100 km
	(float("inf"), 25.0),  # beyond 100 km
]

CURRENT_YEAR = 2026
MAX_DATA_AGE_YEARS = 5
HOURS_PER_YEAR = 8760

# World Bank API does not have 2025/2026 data
INDEXED_BY_YEAR = 2024

FINAL_COLUMNS = [
    "wind_farm_name",
    "LAT",
    "LON",
    "area_sqkm",
    "country",
    "commissioning_year",
    "installed_capacity_MW",
    "turbine_producer",
    "turbine_power_MW",
    "foundation_type",
    "oss_responsibility",
    "offshore_cable",
    "onshore_connection",
    "model_type",
    "water_depth_m",
    "water_depth_min_m",
    "water_depth_max_m",
    "water_depth_mean_m",
    "distance_from_shore_km",
    "distance_from_shore_min_km",
    "distance_from_shore_max_km",
    "distance_from_shore_mean_km",
    "distance_from_port_km",
    "total_project_budget",
    "total_project_budget_eur",
    "total_project_budget_eur_indexed",
    "project_lifetime_years",
    "mean_wind_speed_mps",
    "mean_wave_height_m",
]

FEATURE_COLUMNS = [
    "LAT",
    "LON",
    "area_sqkm",
    "commissioning_year",
    "installed_capacity_MW",
    "turbine_power_MW",
    "water_depth_mean_m",
    "distance_from_shore_mean_km",
    "distance_from_port_km",
    "project_lifetime_years",
    "mean_wind_speed_mps",
    "mean_wave_height_m",
]

TARGET_COLUMN = "total_project_budget_eur_indexed"

DEFAULT_NEW_SAMPLE = {
    "LAT": 56.0,
    "LON": 10.0,
    "area_sqkm": 50.0,
    "country": "Denmark",
    "commissioning_year": CURRENT_YEAR + 2,
    "turbine_power_MW": 10.0,
    "turbine_producer": "Vestas",
    "foundation_type": "Monopile",
    "distance_from_port_km": None, # to be filled based on location
    "project_lifetime_years": 25,
    "installed_capacity_MW": 100.0,
    "mean_wind_speed_mps": None, # to be filled based on location
    "mean_wave_height_m": None, # to be filled based on location
}

TARGET_PROFITABILITY_COLUMN = "lcoe_eur_per_mwh_indexed"
ASSETS_PATH = "src/ui/assets"
