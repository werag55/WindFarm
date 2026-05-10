"""Data enrichment: environmental data + distances to nearest ports.

Public enrichers added by this module:
  - add_environmental_columns        wind speed + wave height (+ source/QF)
  - add_water_depth_columns          GEBCO bathymetry (+ source/QF)
  - add_distance_from_shore_columns  Natural-Earth coastline (+ source/QF)
  - add_distance_from_port           harbour shapefiles
  - add_distance_from_construction_port

Each enricher writes three columns:
  <feature>, <feature>_source, <feature>_quality_flag

Quality flag values:
  "ok"        — value computed from primary data source
  "fallback"  — value taken from regional / heuristic fallback
  "input"     — value retained from raw input
  "missing"   — could not compute and no input available
"""

import logging
import math
import os
from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import wind_stats
from shapely.geometry import Point

from .. import config

LOGGER = logging.getLogger(__name__)

# Allow GDAL to rebuild missing .shx index from .shp
os.environ.setdefault("SHAPE_RESTORE_SHX", "YES")

# ETRS89 / LAEA Europe — equal-area, accurate to ~25 m, no polar singularity.
METRIC_CRS = "EPSG:3035"
GEOGRAPHIC_CRS = "EPSG:4326"

# Bounding box for sanity-checking port coordinates.
EUROPE_BBOX = (-25.0, 30.0, 45.0, 75.0)

# Decimal places used to bin LAT/LON for caching of marine queries.
# 3 decimals ≈ ~110 m at the equator — fine enough for wave/depth fields.
_COORD_CACHE_DECIMALS = 3

# Quality flag constants
QF_OK       = "ok"
QF_FALLBACK = "fallback"
QF_INPUT    = "input"
QF_MISSING  = "missing"

# Source labels
SRC_GEBCO        = "GEBCO"
SRC_NATURAL_EARTH = "NaturalEarth"
SRC_REGIONAL     = "regional_fallback"
SRC_COPERNICUS   = "Copernicus_Marine"
SRC_ERA5         = "ERA5"
SRC_INPUT        = "raw_input"
SRC_UNKNOWN      = "unknown"

# Hardcoded fallback if the CSV is unavailable.
_FALLBACK_CONSTRUCTION_PORTS = [
    {"name": "Esbjerg",       "country": "Denmark", "lat": 55.4633, "lon":  8.4469, "type": "construction", "notes": ""},
    {"name": "Cuxhaven",      "country": "Germany", "lat": 53.8640, "lon":  8.7087, "type": "construction", "notes": ""},
    {"name": "Bremerhaven",   "country": "Germany", "lat": 53.5536, "lon":  8.5755, "type": "construction", "notes": ""},
    {"name": "Eemshaven",     "country": "Netherlands", "lat": 53.4420, "lon": 6.8250, "type": "construction", "notes": ""},
    {"name": "Rotterdam",     "country": "Netherlands", "lat": 51.8850, "lon": 4.2867, "type": "construction", "notes": ""},
    {"name": "Hull",          "country": "UK",      "lat": 53.7424, "lon": -0.2798, "type": "construction", "notes": ""},
    {"name": "Belfast",       "country": "UK",      "lat": 54.6100, "lon": -5.9100, "type": "construction", "notes": ""},
    {"name": "Saint-Nazaire", "country": "France",  "lat": 47.2700, "lon": -2.2100, "type": "construction", "notes": ""},
    {"name": "Ferrol",        "country": "Spain",   "lat": 43.4800, "lon": -8.2400, "type": "construction", "notes": ""},
    {"name": "Gdansk",        "country": "Poland",  "lat": 54.3781, "lon": 18.6758, "type": "construction", "notes": ""},
]

_CONSTRUCTION_PORTS_GDF: gpd.GeoDataFrame | None = None
_HARBOUR_PORTS_GDF:      gpd.GeoDataFrame | None = None


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def add_environmental_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Attach mean wind speed and wave data.

    Wind speed: Global Wind Atlas (wind-stats).
    Wave height: Copernicus Marine if configured, otherwise regional fallback.

    Adds or fills the columns:
        - mean_wind_speed_mps
        - mean_wave_height_m
        - wave_height_source
        - wave_height_quality_flag
    """

    df = df.copy()

    # Numeric columns
    for col in (
        "mean_wind_speed_mps",
        "mean_wave_height_m",
    ):
        if col not in df.columns:
            df[col] = np.nan
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Text columns
    for col in (
        "wave_height_source",
        "wave_height_quality_flag",
    ):
        if col not in df.columns:
            df[col] = pd.Series(index=df.index, dtype="object")
        else:
            df[col] = df[col].astype("object")

    # -- wind speed: Global Wind Atlas ---------------------------------------
    df["mean_wind_speed_mps"] = df.apply(
        lambda row: _get_mean_wind_speed_from_gwa(row["LAT"], row["LON"])
        if pd.isna(row["mean_wind_speed_mps"])
        and pd.notna(row.get("LAT"))
        and pd.notna(row.get("LON"))
        else row["mean_wind_speed_mps"],
        axis=1,
    )

    # -- wave height ----------------------------------------------------------
    n_input = int(df["mean_wave_height_m"].notna().sum())
    n_filled = 0
    n_fallback = 0

    for idx, row in df.iterrows():
        # Already populated -> mark as input-derived, do not overwrite.
        if pd.notna(row["mean_wave_height_m"]):
            if pd.isna(df.at[idx, "wave_height_source"]):
                df.at[idx, "wave_height_source"] = SRC_INPUT
            if pd.isna(df.at[idx, "wave_height_quality_flag"]):
                df.at[idx, "wave_height_quality_flag"] = QF_INPUT
            continue

        lat = row.get("LAT")
        lon = row.get("LON")

        if pd.isna(lat) or pd.isna(lon):
            df.at[idx, "wave_height_source"] = SRC_UNKNOWN
            df.at[idx, "wave_height_quality_flag"] = QF_MISSING
            continue

        value, source, flag = _get_mean_wave_height(float(lat), float(lon))

        df.at[idx, "mean_wave_height_m"] = value
        df.at[idx, "wave_height_source"] = source
        df.at[idx, "wave_height_quality_flag"] = flag

        n_filled += 1

        if flag == QF_FALLBACK:
            n_fallback += 1

    LOGGER.info(
        "Wave height enrichment: %d input-derived, %d filled (%d via fallback), "
        "%d missing (out of %d rows)",
        n_input,
        n_filled,
        n_fallback,
        int(df["mean_wave_height_m"].isna().sum()),
        len(df),
    )

    if n_fallback:
        LOGGER.warning(
            "Wave height: %d rows used regional fallback — values are climatological "
            "averages, NOT site-specific reanalysis. Configure Copernicus Marine "
            "(see config.COPERNICUS_MARINE_*) for accurate Hs.",
            n_fallback,
        )

    return df


def add_distance_from_port(df: pd.DataFrame) -> pd.DataFrame:
    """Add distance and name of nearest harbour (general/service port).

    Uses EMODnet/OSM harbour shapefiles. Adds:
      - distance_from_port_km
      - nearest_port_name
    """
    ports_gdf = _get_harbour_ports_gdf()
    if ports_gdf is None or ports_gdf.empty:
        LOGGER.warning("No harbour ports available — distance_from_port_km set to NaN")
        df["distance_from_port_km"] = np.nan
        df["nearest_port_name"] = np.nan
        return df

    distances, names = _compute_distance_and_name(
        df, ports_gdf, name_col="port_name", label="harbour"
    )
    df["distance_from_port_km"] = distances
    df["nearest_port_name"] = names
    return df


def add_distance_from_construction_port(df: pd.DataFrame) -> pd.DataFrame:
    """Add distance and name of nearest construction port.

    Uses curated CSV `european_offshore_wind_construction_ports.csv`. Adds:
      - distance_from_construction_port_km
      - nearest_construction_port_name
    """
    ports_gdf = _get_construction_ports_gdf()
    distances, names = _compute_distance_and_name(
        df, ports_gdf, name_col="name", label="construction port"
    )
    df["distance_from_construction_port_km"] = distances
    df["nearest_construction_port_name"] = names
    return df


# ---------------------------------------------------------------------------
# Port loaders
# ---------------------------------------------------------------------------

def _get_construction_ports_gdf() -> gpd.GeoDataFrame:
    """Load the curated construction-ports CSV (with hardcoded fallback)."""
    global _CONSTRUCTION_PORTS_GDF
    if _CONSTRUCTION_PORTS_GDF is not None:
        return _CONSTRUCTION_PORTS_GDF

    csv_path = Path(getattr(config, "CONSTRUCTION_PORTS_CSV", ""))
    if csv_path and csv_path.is_file():
        ports_df = pd.read_csv(csv_path)
        # Filter to construction-type ports only (CSV may also contain service-only entries)
        if "type" in ports_df.columns:
            ports_df = ports_df[
                ports_df["type"].astype(str).str.lower().isin(
                    {"construction", "construction+service", "marshalling"}
                )
            ].copy()
        LOGGER.info(
            "Loaded %d construction ports from CSV %s", len(ports_df), csv_path
        )
    else:
        LOGGER.warning(
            "Construction ports CSV not found at %s — using built-in fallback list",
            csv_path,
        )
        ports_df = pd.DataFrame(_FALLBACK_CONSTRUCTION_PORTS)

    if ports_df.empty:
        LOGGER.warning("Construction ports list is empty after filtering")
        _CONSTRUCTION_PORTS_GDF = gpd.GeoDataFrame(geometry=[], crs=METRIC_CRS)
        return _CONSTRUCTION_PORTS_GDF

    gdf = gpd.GeoDataFrame(
        ports_df,
        geometry=gpd.points_from_xy(ports_df["lon"], ports_df["lat"]),
        crs=GEOGRAPHIC_CRS,
    )
    _CONSTRUCTION_PORTS_GDF = _to_metric_and_validate(
        gdf, source="construction ports CSV"
    )
    return _CONSTRUCTION_PORTS_GDF


def _get_harbour_ports_gdf() -> gpd.GeoDataFrame | None:
    """Load EMODnet/OSM harbour shapefiles, with name-column normalisation."""
    global _HARBOUR_PORTS_GDF
    if _HARBOUR_PORTS_GDF is not None:
        return _HARBOUR_PORTS_GDF

    paths = [Path(p) for p in getattr(config, "HARBOUR_SHAPEFILES", [])]
    if not paths:
        # Backwards compatibility: previous hardcoded paths
        base = Path(__file__).parent.parent.parent / "data" / "ports"
        paths = [
            base / "Harbours_EMODnet_OSM_HOLAS3.shp",
            base / "Harbours.shp",
        ]

    gdfs = []
    for path in paths:
        if not path.exists():
            LOGGER.info("Port shapefile not found: %s", path)
            continue
        try:
            gdf = gpd.read_file(path)
            LOGGER.info(
                "Loaded %s: %d records, crs=%s",
                path.name, len(gdf), gdf.crs,
            )
            gdfs.append(gdf)
        except Exception as exc:
            LOGGER.warning("Could not read %s: %s", path, exc)

    if not gdfs:
        _HARBOUR_PORTS_GDF = gpd.GeoDataFrame()
        return None

    merged = pd.concat(gdfs, ignore_index=True)
    merged = gpd.GeoDataFrame(merged, geometry="geometry")

    if merged.crs is None:
        LOGGER.warning("Harbour shapefile has no CRS — assuming EPSG:4326")
        merged = merged.set_crs(GEOGRAPHIC_CRS)

    # Normalise name column (different shapefiles use different column names)
    name_candidates = ["NAME", "Name", "name", "harbour", "PORT_NAME", "PORTNAME", "HARBOR"]
    found = next((c for c in name_candidates if c in merged.columns), None)
    if found:
        merged["port_name"] = merged[found].astype(str)
    else:
        merged["port_name"] = "Unknown harbour"

    _HARBOUR_PORTS_GDF = _to_metric_and_validate(
        merged, source="harbour shapefiles"
    )
    return _HARBOUR_PORTS_GDF


def _to_metric_and_validate(
    gdf: gpd.GeoDataFrame, source: str
) -> gpd.GeoDataFrame:
    """Drop bad geometries, reproject to metric CRS, log statistics."""
    n0 = len(gdf)

    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    n_geom = len(gdf)

    if gdf.crs != GEOGRAPHIC_CRS:
        try:
            gdf_geo = gdf.to_crs(GEOGRAPHIC_CRS)
        except Exception as exc:
            LOGGER.error("Reprojection of %s to geographic failed: %s", source, exc)
            return gpd.GeoDataFrame(geometry=[], crs=METRIC_CRS)
    else:
        gdf_geo = gdf

    xs = gdf_geo.geometry.x
    ys = gdf_geo.geometry.y
    in_europe = (
        (xs >= EUROPE_BBOX[0]) & (xs <= EUROPE_BBOX[2])
        & (ys >= EUROPE_BBOX[1]) & (ys <= EUROPE_BBOX[3])
        & xs.notna() & ys.notna()
        & np.isfinite(xs) & np.isfinite(ys)
    )
    gdf = gdf.loc[in_europe.values].copy()
    n_in_bbox = len(gdf)

    try:
        gdf = gdf.to_crs(METRIC_CRS)
    except Exception as exc:
        LOGGER.error("Reprojection of %s to %s failed: %s", source, METRIC_CRS, exc)
        return gpd.GeoDataFrame(geometry=[], crs=METRIC_CRS)

    LOGGER.info(
        "Port set %r: %d raw -> %d valid geom -> %d in Europe bbox",
        source, n0, n_geom, n_in_bbox,
    )
    return gdf


# ---------------------------------------------------------------------------
# Distance computation
# ---------------------------------------------------------------------------

def _compute_distance_and_name(
    df: pd.DataFrame,
    ports_metric: gpd.GeoDataFrame,
    name_col: str,
    label: str,
) -> tuple[pd.Series, pd.Series]:
    """Return (distances_km, nearest_port_names) Series aligned with df.index."""
    distances = pd.Series(np.nan, index=df.index, dtype=float)
    names     = pd.Series(np.nan, index=df.index, dtype=object)

    if ports_metric is None or ports_metric.empty:
        LOGGER.warning("No %s ports available — returning NaN", label)
        return distances, names

    if name_col not in ports_metric.columns:
        LOGGER.warning(
            "Name column %r not found in %s ports; using fallback 'Unknown'",
            name_col, label,
        )
        ports_metric = ports_metric.copy()
        ports_metric[name_col] = "Unknown"

    valid = df["LAT"].notna() & df["LON"].notna()
    if not valid.any():
        return distances, names

    farm_pts = gpd.GeoSeries(
        [Point(lon, lat) for lat, lon in zip(df.loc[valid, "LAT"], df.loc[valid, "LON"])],
        crs=GEOGRAPHIC_CRS,
        index=df.index[valid],
    ).to_crs(METRIC_CRS)

    port_geoms = ports_metric.geometry.reset_index(drop=True)
    port_names = ports_metric[name_col].reset_index(drop=True)

    for idx, pt in farm_pts.items():
        dist_m = port_geoms.distance(pt)
        if dist_m.empty or not np.isfinite(dist_m.min()):
            continue
        nearest_pos = int(dist_m.idxmin())
        distances.at[idx] = float(dist_m.iat[nearest_pos]) / 1000.0
        names.at[idx]     = str(port_names.iat[nearest_pos])

    distances = distances.replace([np.inf, -np.inf], np.nan)

    if distances.notna().any():
        LOGGER.info(
            "%s distances: count=%d, min=%.1f km, mean=%.1f km, max=%.1f km",
            label,
            int(distances.notna().sum()),
            float(distances.min(skipna=True)),
            float(distances.mean(skipna=True)),
            float(distances.max(skipna=True)),
        )
    else:
        LOGGER.warning("%s distances: all NaN", label)

    return distances, names


# ---------------------------------------------------------------------------
# Sample logging
# ---------------------------------------------------------------------------

def log_port_distance_examples(df: pd.DataFrame, n: int = 5) -> None:
    """Log a few sample rows with farm name + nearest ports + distances."""
    cols = [
        "wind_farm_name",
        "nearest_port_name",
        "distance_from_port_km",
        "nearest_construction_port_name",
        "distance_from_construction_port_km",
    ]
    available = [c for c in cols if c in df.columns]
    if not available:
        LOGGER.info("No port distance columns available to log")
        return

    sample = df[available].dropna(subset=["distance_from_port_km"]).head(n)
    LOGGER.info("=" * 60)
    LOGGER.info("PORT DISTANCE EXAMPLES (first %d rows)", n)
    LOGGER.info("=" * 60)
    for _, row in sample.iterrows():
        LOGGER.info(
            "%-32s  port=%-25s (%.1f km)  | construction=%-22s (%.1f km)",
            str(row.get("wind_farm_name", "?"))[:32],
            str(row.get("nearest_port_name", "?"))[:25],
            float(row.get("distance_from_port_km", float("nan"))),
            str(row.get("nearest_construction_port_name", "?"))[:22],
            float(row.get("distance_from_construction_port_km", float("nan"))),
        )
    LOGGER.info("=" * 60)


# ---------------------------------------------------------------------------
# Environmental data fetchers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=512)
def _get_mean_wind_speed_from_gwa(latitude: float, longitude: float) -> float:
    """Fetch mean wind speed from Global Wind Atlas through wind-stats."""
    lat = round(float(latitude), config.GWA_COORDINATE_ROUND_DECIMALS)
    lon = round(float(longitude), config.GWA_COORDINATE_ROUND_DECIMALS)
    LOGGER.debug("GWA request lat=%.4f lon=%.4f", lat, lon)
    ds = wind_stats.get_gwc_data(lat, lon)

    target_height = float(config.HUB_HEIGHT_M)
    height = float(ds.height.sel(height=target_height, method="nearest").item())
    roughness = float(
        ds.roughness.sel(roughness=config.GWA_TARGET_ROUGHNESS_M, method="nearest").item()
    )

    a_values = ds["A"].sel(roughness=roughness, height=height).values
    k_values = ds["k"].sel(roughness=roughness, height=height).values
    frequency = ds["frequency"].sel(roughness=roughness).values

    sector_mean_speed = a_values * np.vectorize(math.gamma)(1.0 + (1.0 / k_values))
    weights = frequency / np.sum(frequency)
    wind_speed = float(np.sum(sector_mean_speed * weights))
    LOGGER.debug("GWA result lat=%.4f lon=%.4f mean_wind_speed=%.3f", lat, lon, wind_speed)
    return wind_speed


# ---------------------------------------------------------------------------
# Wave height — Copernicus Marine (optional) + regional fallback
# ---------------------------------------------------------------------------

# Climatological mean significant wave height (Hs) for European seas, in metres.
# Sources: Copernicus Marine WAVERYS / ERA5 multi-year climatology summaries.
# These are intentionally conservative averages; replace with site-specific
# reanalysis when API access is configured.
_REGIONAL_WAVE_HEIGHT_M = {
    "North Sea":      1.7,
    "Baltic Sea":     0.8,
    "Irish Sea":      1.5,
    "Atlantic":       2.6,
    "Mediterranean":  1.0,
    "Unknown":        1.5,
}


def _classify_marine_region(lat: float, lon: float) -> str:
    """Bucket a (lat, lon) into one of the predefined European seas."""
    # Order matters: smaller seas first, then the open Atlantic as a catch-all.
    if 53.0 <= lat <= 66.0 and 10.0 <= lon <= 30.5:
        return "Baltic Sea"
    if 51.0 <= lat <= 62.0 and -4.0 <= lon <= 9.5:
        return "North Sea"
    if 51.0 <= lat <= 56.0 and -7.0 <= lon <= -2.5:
        return "Irish Sea"
    if 30.0 <= lat <= 46.5 and -6.0 <= lon <= 36.5:
        return "Mediterranean"
    if 35.0 <= lat <= 72.0 and -35.0 <= lon <= 5.0:
        return "Atlantic"
    return "Unknown"


def _fetch_wave_height_from_copernicus(lat: float, lon: float) -> float | None:
    """Optional hook for Copernicus Marine WAVERYS data.

    Returns None if not configured. To enable, set the following in config.py
    or via env-vars and implement the actual fetch (e.g. with `copernicusmarine`):
        - COPERNICUS_MARINE_USERNAME
        - COPERNICUS_MARINE_PASSWORD
        - COPERNICUS_MARINE_PRODUCT (default: GLOBAL_MULTIYEAR_WAV_001_032)
    """
    user = os.environ.get("COPERNICUS_MARINE_USERNAME") or getattr(
        config, "COPERNICUS_MARINE_USERNAME", None
    )
    if not user:
        return None
    # Real fetch deliberately not implemented to avoid network/credential
    # surprises during a deterministic preprocessing run. Drop the actual
    # `copernicusmarine` call here when you have access configured.
    LOGGER.debug(
        "Copernicus Marine credentials present but client not wired in; "
        "skipping live fetch for lat=%.4f lon=%.4f", lat, lon,
    )
    return None


@lru_cache(maxsize=2048)
def _get_mean_wave_height(latitude: float, longitude: float) -> tuple[float, str, str]:
    """Return (mean_wave_height_m, source, quality_flag) for a LAT/LON.

    Tries Copernicus Marine (if configured), then regional climatology.
    Caches per-(lat, lon) bucketed to _COORD_CACHE_DECIMALS.
    """
    lat = round(float(latitude), _COORD_CACHE_DECIMALS)
    lon = round(float(longitude), _COORD_CACHE_DECIMALS)

    # 1. Try Copernicus Marine
    try:
        value = _fetch_wave_height_from_copernicus(lat, lon)
    except Exception as exc:  # noqa: BLE001 — never block pipeline on this
        LOGGER.warning(
            "Copernicus Marine fetch failed for lat=%.4f lon=%.4f: %s — "
            "falling back to regional climatology", lat, lon, exc,
        )
        value = None

    if value is not None and np.isfinite(value):
        LOGGER.debug(
            "Wave height %.2f m from Copernicus for lat=%.4f lon=%.4f",
            value, lat, lon,
        )
        return float(value), SRC_COPERNICUS, QF_OK

    # 2. Regional climatology fallback
    region = _classify_marine_region(lat, lon)
    value = _REGIONAL_WAVE_HEIGHT_M[region]
    LOGGER.debug(
        "Wave height fallback %.2f m (%s) for lat=%.4f lon=%.4f",
        value, region, lat, lon,
    )
    return float(value), f"{SRC_REGIONAL}:{region}", QF_FALLBACK


# ---------------------------------------------------------------------------
# Water depth — GEBCO bathymetry (local NetCDF) + input fallback
# ---------------------------------------------------------------------------

_GEBCO_DATASET = None      # cached xarray.Dataset
_GEBCO_LOAD_FAILED = False  # avoid re-trying every call once the file is missing


def _gebco_dataset():
    """Lazy-load the GEBCO bathymetry NetCDF, if available.

    Returns the xarray.Dataset or None when not available.
    """
    global _GEBCO_DATASET, _GEBCO_LOAD_FAILED
    if _GEBCO_DATASET is not None or _GEBCO_LOAD_FAILED:
        return _GEBCO_DATASET

    nc_dir = Path(getattr(config, "BATHYMETRY_DIR", "data/bathymetry"))
    if not nc_dir.exists():
        LOGGER.warning(
            "Bathymetry directory %s not found — water depth will fall back to "
            "raw input values when available", nc_dir,
        )
        _GEBCO_LOAD_FAILED = True
        return None

    candidates = sorted(
        list(nc_dir.glob("*.nc")) + list(nc_dir.glob("*.tif"))
    )
    if not candidates:
        LOGGER.warning(
            "No GEBCO file (.nc/.tif) found in %s — water depth will fall back "
            "to raw input values when available", nc_dir,
        )
        _GEBCO_LOAD_FAILED = True
        return None

    try:
        import xarray as xr  # noqa: WPS433 — optional dep, imported lazily
    except ImportError:
        LOGGER.warning(
            "xarray not installed — cannot read GEBCO NetCDF. Install xarray + "
            "netCDF4 to enable GEBCO-based water depth"
        )
        _GEBCO_LOAD_FAILED = True
        return None

    nc_path = candidates[0]
    try:
        ds = xr.open_dataset(nc_path)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Could not open GEBCO file %s: %s", nc_path, exc)
        _GEBCO_LOAD_FAILED = True
        return None

    LOGGER.info("Loaded GEBCO bathymetry from %s (vars=%s)", nc_path, list(ds.data_vars))
    _GEBCO_DATASET = ds
    return _GEBCO_DATASET


@lru_cache(maxsize=4096)
def get_water_depth_from_location(
    lat: float, lon: float
) -> tuple[float | None, str, str]:
    """Return (water_depth_m, source, quality_flag) at (lat, lon).

    Positive metres of depth (sea floor below surface). Land returns None.
    Cached per (lat, lon) bucketed to _COORD_CACHE_DECIMALS.
    """
    lat = round(float(lat), _COORD_CACHE_DECIMALS)
    lon = round(float(lon), _COORD_CACHE_DECIMALS)

    ds = _gebco_dataset()
    if ds is None:
        return None, SRC_UNKNOWN, QF_MISSING

    # GEBCO variable is typically 'elevation' (negative = below sea level)
    elev_var = None
    for candidate in ("elevation", "z", "depth", "bathymetry"):
        if candidate in ds.data_vars:
            elev_var = candidate
            break
    if elev_var is None:
        LOGGER.warning(
            "GEBCO dataset has no recognised elevation variable (vars=%s)",
            list(ds.data_vars),
        )
        return None, SRC_UNKNOWN, QF_MISSING

    # GEBCO typically uses 'lat' / 'lon' coordinates
    try:
        value = ds[elev_var].sel(lat=lat, lon=lon, method="nearest").item()
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug(
            "GEBCO lookup failed for lat=%.4f lon=%.4f: %s", lat, lon, exc,
        )
        return None, SRC_UNKNOWN, QF_MISSING

    if not np.isfinite(value):
        return None, SRC_UNKNOWN, QF_MISSING

    if value >= 0:
        # Above sea level — not a valid offshore depth.
        LOGGER.debug(
            "GEBCO elevation %.1f m at lat=%.4f lon=%.4f is on land",
            value, lat, lon,
        )
        return None, SRC_GEBCO, QF_MISSING

    depth_m = float(-value)
    LOGGER.debug(
        "GEBCO depth=%.1f m at lat=%.4f lon=%.4f", depth_m, lat, lon,
    )
    return depth_m, SRC_GEBCO, QF_OK


def add_water_depth_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Fill `water_depth_m` from GEBCO bathymetry where missing.

    Adds the following columns (create-if-missing):
      - water_depth_m
      - water_depth_source
      - water_depth_quality_flag

    For backward compatibility, if water_depth_min_m / max_m / mean_m are NaN
    they are set equal to water_depth_m.
    """
    # Numeric columns
    for col in (
        "water_depth_m",
        "water_depth_min_m",
        "water_depth_max_m",
        "water_depth_mean_m",
    ):
        if col not in df.columns:
            df[col] = np.nan
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Text columns — must be object dtype so string assignments don't raise
    # LossySetitemError under newer pandas versions.
    for col in ("water_depth_source", "water_depth_quality_flag"):
        if col not in df.columns:
            df[col] = pd.Series(index=df.index, dtype="object")
        else:
            df[col] = df[col].astype("object")

    n_input    = 0
    n_gebco    = 0
    n_missing  = 0

    for idx, row in df.iterrows():
        depth = row.get("water_depth_m")

        # Honour existing input — mark its provenance and keep the value.
        if pd.notna(depth):
            if pd.isna(df.at[idx, "water_depth_source"]):
                df.at[idx, "water_depth_source"] = SRC_INPUT
                df.at[idx, "water_depth_quality_flag"] = QF_INPUT
            n_input += 1
        else:
            lat, lon = row.get("LAT"), row.get("LON")
            if pd.isna(lat) or pd.isna(lon):
                df.at[idx, "water_depth_source"] = SRC_UNKNOWN
                df.at[idx, "water_depth_quality_flag"] = QF_MISSING
                n_missing += 1
            else:
                value, source, flag = get_water_depth_from_location(
                    float(lat), float(lon)
                )
                df.at[idx, "water_depth_m"]              = value
                df.at[idx, "water_depth_source"]         = source
                df.at[idx, "water_depth_quality_flag"]   = flag
                if value is not None and flag == QF_OK:
                    n_gebco += 1
                else:
                    n_missing += 1

        # Backward-compat: fill min/max/mean from the canonical value
        depth = df.at[idx, "water_depth_m"]
        if pd.notna(depth):
            if pd.isna(df.at[idx, "water_depth_min_m"]):
                df.at[idx, "water_depth_min_m"] = depth
            if pd.isna(df.at[idx, "water_depth_max_m"]):
                df.at[idx, "water_depth_max_m"] = depth
            if pd.isna(df.at[idx, "water_depth_mean_m"]):
                df.at[idx, "water_depth_mean_m"] = depth

    LOGGER.info(
        "Water depth enrichment: input=%d, gebco=%d, missing=%d (total=%d)",
        n_input, n_gebco, n_missing, len(df),
    )
    if n_missing and _GEBCO_LOAD_FAILED:
        LOGGER.warning(
            "Water depth: GEBCO unavailable and %d rows had no input value — "
            "they remain NaN. Place a GEBCO NetCDF in %s to fix.",
            n_missing, getattr(config, "BATHYMETRY_DIR", "data/bathymetry"),
        )
    return df


# ---------------------------------------------------------------------------
# Distance from shore — Natural Earth coastline + input fallback
# ---------------------------------------------------------------------------

_COASTLINE_GDF: gpd.GeoDataFrame | None = None
_COASTLINE_LOAD_FAILED = False


def _get_coastline_gdf() -> gpd.GeoDataFrame | None:
    """Lazy-load the Natural Earth (or equivalent) coastline shapefile."""
    global _COASTLINE_GDF, _COASTLINE_LOAD_FAILED
    if _COASTLINE_GDF is not None or _COASTLINE_LOAD_FAILED:
        return _COASTLINE_GDF

    coast_dir = Path(getattr(config, "COASTLINE_DIR", "data/coastline"))
    if not coast_dir.exists():
        LOGGER.warning(
            "Coastline directory %s not found — distance from shore will fall "
            "back to raw input values when available", coast_dir,
        )
        _COASTLINE_LOAD_FAILED = True
        return None

    shapefiles = sorted(coast_dir.glob("*.shp"))
    if not shapefiles:
        LOGGER.warning(
            "No .shp coastline file found in %s — distance from shore will fall "
            "back to raw input values when available", coast_dir,
        )
        _COASTLINE_LOAD_FAILED = True
        return None

    try:
        gdf = gpd.read_file(shapefiles[0])
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Could not read coastline %s: %s", shapefiles[0], exc)
        _COASTLINE_LOAD_FAILED = True
        return None

    if gdf.crs is None:
        gdf = gdf.set_crs(GEOGRAPHIC_CRS)
    try:
        gdf = gdf.to_crs(METRIC_CRS)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Reprojection of coastline to %s failed: %s", METRIC_CRS, exc)
        _COASTLINE_LOAD_FAILED = True
        return None

    LOGGER.info(
        "Loaded coastline from %s: %d features, projected to %s",
        shapefiles[0], len(gdf), METRIC_CRS,
    )
    _COASTLINE_GDF = gdf
    return _COASTLINE_GDF


@lru_cache(maxsize=4096)
def get_distance_from_shore_km(
    lat: float, lon: float
) -> tuple[float | None, str, str]:
    """Geodesic distance (km) from (lat, lon) to the nearest coastline.

    Returns (distance_km, source, quality_flag). Cached per binned LAT/LON.
    """
    lat = round(float(lat), _COORD_CACHE_DECIMALS)
    lon = round(float(lon), _COORD_CACHE_DECIMALS)

    coast = _get_coastline_gdf()
    if coast is None or coast.empty:
        return None, SRC_UNKNOWN, QF_MISSING

    try:
        point = gpd.GeoSeries(
            [Point(lon, lat)], crs=GEOGRAPHIC_CRS,
        ).to_crs(METRIC_CRS).iloc[0]
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("Point projection failed for (%s, %s): %s", lat, lon, exc)
        return None, SRC_UNKNOWN, QF_MISSING

    dist_m = coast.geometry.distance(point)
    if dist_m.empty or not np.isfinite(dist_m.min()):
        return None, SRC_UNKNOWN, QF_MISSING

    distance_km = float(dist_m.min()) / 1000.0
    LOGGER.debug(
        "Distance from shore %.2f km for lat=%.4f lon=%.4f",
        distance_km, lat, lon,
    )
    return distance_km, SRC_NATURAL_EARTH, QF_OK


def add_distance_from_shore_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Fill `distance_from_shore_km` from the local coastline shapefile.

    Adds the following columns (create-if-missing):
      - distance_from_shore_km
      - distance_from_shore_source
      - distance_from_shore_quality_flag

    For backward compatibility, if distance_from_shore_min/max/mean_km are NaN
    they are set equal to distance_from_shore_km.
    """
    # Numeric columns
    for col in (
        "distance_from_shore_km",
        "distance_from_shore_min_km",
        "distance_from_shore_max_km",
        "distance_from_shore_mean_km",
    ):
        if col not in df.columns:
            df[col] = np.nan
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Text columns — must be object dtype so string assignments don't raise
    # LossySetitemError under newer pandas versions.
    for col in ("distance_from_shore_source", "distance_from_shore_quality_flag"):
        if col not in df.columns:
            df[col] = pd.Series(index=df.index, dtype="object")
        else:
            df[col] = df[col].astype("object")

    n_input   = 0
    n_coast   = 0
    n_missing = 0

    for idx, row in df.iterrows():
        existing = row.get("distance_from_shore_km")

        if pd.notna(existing):
            if pd.isna(df.at[idx, "distance_from_shore_source"]):
                df.at[idx, "distance_from_shore_source"] = SRC_INPUT
                df.at[idx, "distance_from_shore_quality_flag"] = QF_INPUT
            n_input += 1
        else:
            lat, lon = row.get("LAT"), row.get("LON")
            if pd.isna(lat) or pd.isna(lon):
                df.at[idx, "distance_from_shore_source"] = SRC_UNKNOWN
                df.at[idx, "distance_from_shore_quality_flag"] = QF_MISSING
                n_missing += 1
            else:
                value, source, flag = get_distance_from_shore_km(
                    float(lat), float(lon)
                )
                df.at[idx, "distance_from_shore_km"]              = value
                df.at[idx, "distance_from_shore_source"]         = source
                df.at[idx, "distance_from_shore_quality_flag"]   = flag
                if value is not None and flag == QF_OK:
                    n_coast += 1
                else:
                    n_missing += 1

        # Backward-compat: fill min/max/mean from the canonical value
        dist = df.at[idx, "distance_from_shore_km"]
        if pd.notna(dist):
            if pd.isna(df.at[idx, "distance_from_shore_min_km"]):
                df.at[idx, "distance_from_shore_min_km"] = dist
            if pd.isna(df.at[idx, "distance_from_shore_max_km"]):
                df.at[idx, "distance_from_shore_max_km"] = dist
            if pd.isna(df.at[idx, "distance_from_shore_mean_km"]):
                df.at[idx, "distance_from_shore_mean_km"] = dist

    LOGGER.info(
        "Distance-from-shore enrichment: input=%d, coastline=%d, missing=%d (total=%d)",
        n_input, n_coast, n_missing, len(df),
    )
    if n_missing and _COASTLINE_LOAD_FAILED:
        LOGGER.warning(
            "Distance from shore: coastline unavailable and %d rows had no "
            "input value — they remain NaN. Place a Natural-Earth coastline "
            "shapefile in %s to fix.",
            n_missing, getattr(config, "COASTLINE_DIR", "data/coastline"),
        )
    return df


# ---------------------------------------------------------------------------
# Combined sample logging
# ---------------------------------------------------------------------------

def log_environmental_examples(df: pd.DataFrame, n: int = 5) -> None:
    """Log n example rows showing wave/depth/distance values + provenance.

    Helps a reviewer eyeball whether enrichment is doing the right thing
    without having to load the cleaned CSV.
    """
    cols = [
        "wind_farm_name", "LAT", "LON",
        "mean_wave_height_m",       "wave_height_source",
        "water_depth_m",            "water_depth_source",
        "distance_from_shore_km",   "distance_from_shore_source",
    ]
    available = [c for c in cols if c in df.columns]
    if not available:
        LOGGER.info("log_environmental_examples: no env columns to display")
        return

    sample = df[available].head(n)
    LOGGER.info("=" * 72)
    LOGGER.info("ENVIRONMENTAL ENRICHMENT EXAMPLES (first %d rows)", n)
    LOGGER.info("=" * 72)
    for _, row in sample.iterrows():
        LOGGER.info(
            "%-28s lat=%6.2f lon=%6.2f | Hs=%.2f m [%s] | depth=%s [%s] | "
            "shore=%s km [%s]",
            str(row.get("wind_farm_name", "?"))[:28],
            float(row.get("LAT", float("nan"))),
            float(row.get("LON", float("nan"))),
            float(row.get("mean_wave_height_m", float("nan"))),
            str(row.get("wave_height_source", "?")),
            _fmt_opt_num(row.get("water_depth_m"), "%.1f"),
            str(row.get("water_depth_source", "?")),
            _fmt_opt_num(row.get("distance_from_shore_km"), "%.1f"),
            str(row.get("distance_from_shore_source", "?")),
        )
    LOGGER.info("=" * 72)


def _fmt_opt_num(value, fmt: str) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    try:
        return fmt % float(value)
    except (TypeError, ValueError):
        return str(value)