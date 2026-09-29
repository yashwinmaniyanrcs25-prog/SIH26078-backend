"""
SIH26078 — Normalized Atmospheric Data Schema & Unit Normalizer
Provides 4D/5D spatio-temporal multidimensional data representations,
strict physical unit tracking, and ensemble specification.
"""
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field

# ==============================================================================
# UNIT CONVERSION CONSTANTS & FUNCTIONS
# ==============================================================================

def celsius_to_kelvin(c: Optional[float]) -> Optional[float]:
    return round(c + 273.15, 2) if c is not None else None

def kelvin_to_celsius(k: Optional[float]) -> Optional[float]:
    return round(k - 273.15, 2) if k is not None else None

def kmh_to_ms(kmh: Optional[float]) -> Optional[float]:
    return round(kmh / 3.6, 2) if kmh is not None else None

def ms_to_kmh(ms: Optional[float]) -> Optional[float]:
    return round(ms * 3.6, 2) if ms is not None else None

def pa_to_hpa(pa: Optional[float]) -> Optional[float]:
    return round(pa / 100.0, 2) if pa is not None else None

def hpa_to_pa(hpa: Optional[float]) -> Optional[float]:
    return round(hpa * 100.0, 2) if hpa is not None else None


# ==============================================================================
# CANONICAL METEOROLOGICAL VARIABLE DEFINITIONS
# ==============================================================================

VARIABLE_METADATA = {
    "temperature_2m": {
        "standard_name": "air_temperature",
        "canonical_unit": "K",
        "display_unit": "°C",
        "description": "Temperature of air at 2 meters above surface"
    },
    "precipitation": {
        "standard_name": "precipitation_flux",
        "canonical_unit": "mm/h",
        "display_unit": "mm/h",
        "description": "Hourly precipitation rate"
    },
    "pressure_msl": {
        "standard_name": "air_pressure_at_mean_sea_level",
        "canonical_unit": "hPa",
        "display_unit": "hPa",
        "description": "Mean sea level pressure"
    },
    "relative_humidity_2m": {
        "standard_name": "relative_humidity",
        "canonical_unit": "%",
        "display_unit": "%",
        "description": "Relative humidity at 2 meters [0-100]"
    },
    "wind_speed_10m": {
        "standard_name": "wind_speed",
        "canonical_unit": "m/s",
        "display_unit": "km/h",
        "description": "Magnitude of horizontal wind velocity at 10 meters"
    },
    "wind_direction_10m": {
        "standard_name": "wind_to_direction",
        "canonical_unit": "degrees",
        "display_unit": "°",
        "description": "Meteorological wind direction clockwise from north"
    },
    "wind_u_10m": {
        "standard_name": "eastward_wind",
        "canonical_unit": "m/s",
        "display_unit": "m/s",
        "description": "Zonal (east-west) wind component"
    },
    "wind_v_10m": {
        "standard_name": "northward_wind",
        "canonical_unit": "m/s",
        "display_unit": "m/s",
        "description": "Meridional (north-south) wind component"
    },
    "wind_gusts_10m": {
        "standard_name": "wind_speed_of_gust",
        "canonical_unit": "m/s",
        "display_unit": "km/h",
        "description": "Peak wind gust velocity at 10 meters"
    },
    "cape": {
        "standard_name": "convective_available_potential_energy",
        "canonical_unit": "J/kg",
        "display_unit": "J/kg",
        "description": "Convective available potential energy"
    },
    "cloud_cover": {
        "standard_name": "cloud_area_fraction",
        "canonical_unit": "%",
        "display_unit": "%",
        "description": "Total cloud area fraction"
    }
}


# ==============================================================================
# PYDANTIC ATMOSPHERIC MODELS
# ==============================================================================

class AtmosphericObservationPoint(BaseModel):
    """Point observation or single cell in an atmospheric grid."""
    time: str = Field(..., description="ISO 8601 UTC timestamp")
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    temperature_c: Optional[float] = None
    temperature_k: Optional[float] = None
    precipitation_mm_hr: Optional[float] = None
    pressure_hpa: Optional[float] = None
    relative_humidity_pct: Optional[float] = None
    wind_speed_kmh: Optional[float] = None
    wind_speed_ms: Optional[float] = None
    wind_direction_deg: Optional[float] = None
    wind_u_ms: Optional[float] = None
    wind_v_ms: Optional[float] = None
    wind_gusts_kmh: Optional[float] = None
    cape_jkg: Optional[float] = None
    cloud_cover_pct: Optional[float] = None
    weather_code: Optional[int] = None


class SpatioTemporalDataCube(BaseModel):
    """
    Standardized 4D / 5D Multivariable Atmospheric Representation:
    Canonical Shape: [T, MEMBER, LAT, LON, VARIABLES]
    Where:
      - T: forecast valid timestamps
      - MEMBER: ensemble member indices (1 for deterministic)
      - LAT: latitude coordinates in degrees
      - LON: longitude coordinates in degrees
      - VARIABLES: canonical meteorological variable indices
    """
    dataset_name: str
    source_model: str
    spatial_resolution_km: float
    time_steps: List[str]
    latitudes: List[float]
    longitudes: List[float]
    variables: List[str]
    ensemble_available: bool = False
    ensemble_status: str = "not_available"  # "online" | "not_available" | "partial"
    ensemble_members_count: int = 1
    ensemble_member_ids: List[str] = Field(default_factory=lambda: ["control_deterministic"])
    units: Dict[str, str] = Field(default_factory=dict)
    grid_type: str = "regular_lat_lon"  # regular_lat_lon | spherical_mesh | icosahedral_grid
    # Canonical 5D shape: [T, MEMBER, LAT, LON, VARIABLES]
    data_shape: List[int] = Field(default_factory=list)
    status: str = "online"
    # Optional raw/normalized numpy array or tensor metadata
    tensor_format: str = "[T, MEMBER, LAT, LON, VARIABLES]"


class ScientificDataProvenance(BaseModel):
    """Explicit scientific data provenance contract (Section 20)."""
    source: str
    model: str
    resolution: str
    forecast_initialization: Optional[str] = None
    valid_time: Optional[str] = None
    variables: List[str] = Field(default_factory=list)
    dataset: str = "operational_nwp"
    processing_method: str = "canonical_normalization"
    model_checkpoint_status: str = "checkpoint_not_loaded"
    baseline_status: str = "not_loaded"
    ensemble_status: str = "not_available"
    ensemble: bool = False
    gnn: str = "checkpoint_not_loaded"
    diffusion: str = "checkpoint_not_loaded"


class AnomalyDetectionResult(BaseModel):
    """Detected anomaly candidate region."""
    anomaly_id: str
    variable: str
    detection_method: str = "BASELINE ANOMALY (Z-Score / Quantile Deviation)"
    start_time: str
    end_time: str
    peak_time: str
    centroid_lat: float
    centroid_lon: float
    area_km2: float
    peak_anomaly_value: float
    mean_anomaly_value: float
    baseline_value: Optional[float] = None
    z_score: float
    anomaly_score: float = 0.0
    confidence: float = 1.0
    source: str = "Open-Meteo / ECMWF IFS HRES"
    severity: str  # LOW | MODERATE | SEVERE
    is_gnn_tracked: bool = False
    status: str = "DERIVED"


class TrajectoryPoint(BaseModel):
    """Temporal trajectory point for extreme anomaly propagation."""
    timestamp: str
    centroid_lat: float
    centroid_lon: float
    area_km2: float
    intensity: float
    lead_time_hours: int
    anomaly_score: Optional[float] = None
    movement_speed_kmh: Optional[float] = None
    movement_direction_deg: Optional[float] = None
    track_confidence: float = 1.0
    velocity_kmh: Optional[float] = None
    heading_deg: Optional[float] = None


class BoundingBox4D(BaseModel):
    """Spatio-temporal 4D dynamic bounding box (Section 10)."""
    anomaly_id: str
    variable: Optional[str] = None
    severity: Optional[str] = None
    spatial: Dict[str, float] = Field(..., description="min_lat, max_lat, min_lon, max_lon")
    temporal: Dict[str, str] = Field(..., description="start_time, end_time")
    centroid: Dict[str, float]
    area_km2: float
    crop_resolution_km: float = 12.0
    crop_shape: Optional[List[int]] = None
    trajectory_envelope_points: Optional[int] = None
    status: str = "calculated"


class ThreatFootprint(BaseModel):
    """Final unified scientific threat footprint object."""
    id: str
    type: str
    severity: str  # LOW | MODERATE | SEVERE
    centroid: Dict[str, float]
    bbox: BoundingBox4D
    trajectory: List[TrajectoryPoint]
    efi: Optional[float] = None
    efi_status: str = "not_available"
    lead_time_hours: int
    spatial_resolution_km: float
    source: str
    status: str
    triggering_variables: Dict[str, Any]
    affected_districts: List[str] = Field(default_factory=list)
    affected_blocks: List[str] = Field(default_factory=list)
    affected_panchayats: List[str] = Field(default_factory=list)
    administrative_status: str = "intersection_available_for_target_region"

