"""Configuration for the offshore wind project."""

import logging
from datetime import datetime

LOGGER = logging.getLogger(__name__)

# Track countries we've already warned about, so we don't spam the log
_UNKNOWN_COUNTRIES_SEEN: set[str] = set()

RAW_DATASET_PATH = "data/european_offshore_wind_capex.csv"
CLEANED_DATASET_PATH = "data/cleaned_european_offshore_wind_capex.csv"
CALCULATED_DATASET_PATH = "data/calculated_offshore_wind_data.csv"

PORTS_DIR = "data/ports"
CONSTRUCTION_PORTS_CSV = "data/ports/european_offshore_wind_construction_ports.csv"
HARBOUR_SHAPEFILES = [
    "data/ports/Harbours_EMODnet_OSM_HOLAS3.shp",
    "data/ports/Harbours.shp",
]

# Local marine data assets (see README — none of these are required;
# missing files trigger logged fallbacks).
BATHYMETRY_DIR = "data/bathymetry"   # GEBCO NetCDF (.nc) or GeoTIFF (.tif)
COASTLINE_DIR  = "data/coastline"    # Natural Earth `ne_10m_coastline.shp`

# Optional Copernicus Marine credentials for live Hs fetch
# (leave as None to force regional fallback for wave heights).
COPERNICUS_MARINE_USERNAME = None
COPERNICUS_MARINE_PASSWORD = None
COPERNICUS_MARINE_PRODUCT  = "GLOBAL_MULTIYEAR_WAV_001_032"

CSV_SEPARATOR = ";"
MISSING_VALUE_TOKENS = {"null", "None", "", "NaN", "nan"}

MANDATORY_COLUMNS = {
    "LAT",
    "LON",
    "installed_capacity_MW",
    "total_project_budget"
}

NUMERIC_COLUMNS = [
    "LAT",
    "LON",
    "area_sqkm",
    "commissioning_year",
    "installed_capacity_MW",
    "turbine_power_MW",
    "water_depth_m",
    "water_depth_min_m",
    "water_depth_max_m",
    "water_depth_mean_m",
    "distance_from_shore_km",
    "distance_from_shore_min_km",
    "distance_from_shore_max_km",
    "distance_from_shore_mean_km",
    "distance_from_port_km",
    "distance_from_construction_port_km",
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
    """Determine connection-infrastructure responsibility for an offshore wind project.

    Returns a dict with four keys:
        - oss_responsibility   : who builds the Offshore Substation
        - offshore_cable       : who builds the offshore export cable
        - onshore_connection   : who builds the onshore connection / converter station
        - model_type           : high-level grid-connection regime label

    Sources:
        WindEurope reports on offshore grid connection regimes,
        national TSO publications (TenneT, 50Hertz, Energinet, Elia, PSE, Svenska Kraftnät,
        EirGrid, Fingrid, Elering, AST, Litgrid),
        Project card "Modele kosztów przyłączenia morskich farm wiatrowych w Europie".
    """
    UNKNOWN = {
        "oss_responsibility": "Unknown",
        "offshore_cable":     "Unknown",
        "onshore_connection": "Unknown",
        "model_type":         "Unknown",
    }

    if country is None or str(country).strip().lower() in {"", "nan", "none", "unknown"}:
        return UNKNOWN
    if year is None or (isinstance(year, float) and str(year) == "nan"):
        return UNKNOWN
    try:
        y = int(float(year))
    except (TypeError, ValueError):
        return UNKNOWN

    c = str(country).strip()

    # ----- helper short-hands -----
    def _all(role: str, model: str) -> dict:
        return {
            "oss_responsibility": role,
            "offshore_cable":     role,
            "onshore_connection": role,
            "model_type":         model,
        }

    def _split(off: str, on: str, model: str) -> dict:
        return {
            "oss_responsibility": off,
            "offshore_cable":     off,
            "onshore_connection": on,
            "model_type":         model,
        }

    # =====================================================================
    # GERMANY — TSO-led from the start (TenneT in North Sea, 50Hertz in Baltic).
    # Since 2013 EnWG amendment formalised TSO build-out (HVDC + AC OSS).
    # =====================================================================
    if c == "Germany":
        return _all("TSO", "TSO-led")

    # =====================================================================
    # UNITED KINGDOM — Developer builds, then transfers to OFTO under Ofgem
    # competitive tender (introduced 2009, first tenders ~2011).
    # Round 1/2 (commissioned ≤2010) developers retained ownership.
    # =====================================================================
    if c in {"UK", "United Kingdom", "Great Britain", "GB"}:
        if y < 2011:
            return _all("Developer", "Developer-led")
        return _all("OFTO", "OFTO")

    # =====================================================================
    # NETHERLANDS — TenneT designated as offshore TSO by the 2016 Offshore Wind
    # Energy Act. Borssele I-II (2020) was first project under new regime.
    # Pre-2016 round (Egmond aan Zee, OWEZ, Princess Amalia) was developer-built.
    # =====================================================================
    if c == "Netherlands":
        if y < 2016:
            return _split("Developer", "TSO", "Developer-led")
        return _all("TSO", "TSO-led")

    # =====================================================================
    # DENMARK — Energinet (TSO) historically built OSS + export cable for
    # state-tendered projects (Horns Rev, Anholt, Kriegers Flak…).
    # 2021 reform: Thor and beyond are developer-led under contract-for-difference.
    # Nearshore Vesterhav (2018) was a transitional case but kept TSO grid.
    # =====================================================================
    if c == "Denmark":
        if y <= 2020:
            return _all("TSO", "TSO-led")
        return _all("Developer", "Developer-led")

    # =====================================================================
    # BELGIUM — Pre-MOG: each developer built own export cable to Stevin onshore
    # substation. Modular Offshore Grid (MOG) by Elia commissioned 2019,
    # serving Rentel, Northwester 2, Mermaid, Seastar. MOG-2 from 2026+.
    # =====================================================================
    if c == "Belgium":
        if y < 2019:
            return _split("Developer", "Developer", "Developer-led")
        return _all("TSO", "TSO-led")

    # =====================================================================
    # FRANCE — RTE (TSO) responsible for OSS and export cable since the first
    # tender round (Saint-Nazaire 2012 / commissioned 2022). Developer builds
    # only the array cables and turbines.
    # =====================================================================
    if c == "France":
        return _all("TSO", "TSO-led")

    # =====================================================================
    # NORWAY — No formal offshore TSO regime yet (Statnett role evolving for
    # Sørlige Nordsjø II / Utsira Nord). Treat all as Developer-led for now.
    # =====================================================================
    if c == "Norway":
        return _all("Developer", "Developer-led")

    # =====================================================================
    # POLAND — Offshore Wind Act 2021. Developer responsible for OSS and offshore
    # cable up to landfall; PSE (TSO) reinforces onshore grid. Phase 1 projects
    # (Baltic Power, MFW Bałtyk II/III, Baltica 2/3) all under this regime.
    # =====================================================================
    if c == "Poland":
        return _split("Developer", "Developer/TSO", "Developer-led")

    # =====================================================================
    # SWEDEN — Pre-2021 small projects (Lillgrund, Kårehamn) were developer-built.
    # 2022 reform: Svenska Kraftnät (TSO) designated to build offshore export
    # infrastructure for projects connecting after ~2021/2022 decision.
    # =====================================================================
    if c == "Sweden":
        if y < 2022:
            return _all("Developer", "Developer-led")
        return _all("TSO", "TSO-led")

    # =====================================================================
    # IRELAND — Phase 1 (ORESS 1, projects 2024-2026) is Developer-led: developer
    # builds OSS, offshore cable and onshore connection. Phase 2+ planned to move
    # towards EirGrid-led "plan-led" model from late 2020s.
    # =====================================================================
    if c == "Ireland":
        if y < 2030:
            return _all("Developer", "Developer-led")
        return _all("TSO", "TSO-led")

    # =====================================================================
    # FINLAND — Tahkoluoto (2017) developer-built. Fingrid not yet operating
    # offshore export grid; future large-scale tenders under discussion.
    # =====================================================================
    if c == "Finland":
        return _all("Developer", "Developer-led")

    # =====================================================================
    # ESTONIA / LATVIA / LITHUANIA — No commissioned projects yet (pre-2026).
    # Estonia (Elering) and Lithuania (Litgrid) plan TSO-led offshore grids
    # for first tenders (Liepāja, Lithuanian 700 MW). Latvia + Estonia
    # joint ELWIND project also TSO-led.
    # =====================================================================
    if c in {"Estonia", "Latvia", "Lithuania"}:
        return _all("TSO", "TSO-led")

    # =====================================================================
    # SPAIN / PORTUGAL / ITALY / GREECE — Mediterranean/Atlantic floating
    # demonstrators so far, all developer-built.
    # =====================================================================
    if c in {"Spain", "Portugal", "Italy", "Greece"}:
        return _all("Developer", "Developer-led")

    # ----- Unknown country -----
    if c not in _UNKNOWN_COUNTRIES_SEEN:
        LOGGER.warning(
            "get_connection_details: no rule for country=%r (year=%d), returning Unknown",
            c, y,
        )
        _UNKNOWN_COUNTRIES_SEEN.add(c)
    return UNKNOWN

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

# --- New constants ---
POWER_DENSITY_MW_PER_SQKM = 8.0   # from project card
FOUNDATION_MIN_COUNT = 3          # rare-category collapse threshold

# --- Update NUMERIC_COLUMNS: add turbine_power_MW already present, add nothing new ---
# (turbine_power_MW is already there)

# --- Update FINAL_COLUMNS: add area_sqkm_imputed, data_quality_flag ---
FINAL_COLUMNS = [
    "wind_farm_name",
    "LAT", "LON",
    "area_sqkm", "area_sqkm_imputed",
    "country",
    "commissioning_year",
    "installed_capacity_MW",
    "turbine_model",          # <- preserve raw
    "turbine_producer",
    "turbine_power_MW",
    "foundation_type",
    "oss_responsibility", "offshore_cable", "onshore_connection", "model_type",
    "water_depth_m",
    "water_depth_min_m", "water_depth_max_m", "water_depth_mean_m",
    "water_depth_source", "water_depth_quality_flag",
    "distance_from_shore_km",
    "distance_from_shore_min_km", "distance_from_shore_max_km", "distance_from_shore_mean_km",
    "distance_from_shore_source", "distance_from_shore_quality_flag",
    "distance_from_port_km",
    "nearest_port_name",
    "distance_from_construction_port_km",
    "nearest_construction_port_name",
    "total_project_budget",
    "total_project_budget_eur",
    "total_project_budget_eur_indexed",
    "project_lifetime_years",
    "mean_wind_speed_mps",
    "mean_wave_height_m",
    "wave_height_source", "wave_height_quality_flag",
]

FEATURE_COLUMNS = [
    "LAT",
    "LON",
    "area_sqkm",
    "commissioning_year",
    "turbine_power_MW",
    "water_depth_min_m",
    "water_depth_max_m",
    "water_depth_mean_m",
    "distance_from_shore_min_km",
    "distance_from_shore_max_km",
    "distance_from_shore_mean_km",
    "distance_from_port_km",
    "distance_from_construction_port_km",
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
    # The four lines below are auto-computed by the enrichment pipeline.
    "water_depth_m": None,                       # GEBCO bathymetry
    "water_depth_min_m": None,                   # filled = water_depth_m
    "water_depth_max_m": None,                   # filled = water_depth_m
    "distance_from_shore_km": None,              # Natural Earth coastline
    "distance_from_shore_min_km": None,          # filled = distance_from_shore_km
    "distance_from_shore_max_km": None,          # filled = distance_from_shore_km
    "distance_from_port_km": None,               # nearest harbour
    "distance_from_construction_port_km": None,  # nearest construction port
    "project_lifetime_years": 25,
    "installed_capacity_MW": 100.0,
    "mean_wind_speed_mps": None,                 # Global Wind Atlas
    "mean_wave_height_m": None,                  # Copernicus Marine / regional
}

TARGET_PROFITABILITY_COLUMN = "lcoe_eur_per_mwh_indexed"
ASSETS_PATH = "src/ui/assets"
