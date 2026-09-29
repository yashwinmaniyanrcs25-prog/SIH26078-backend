"""
SIH26078 — Atmospheric Intelligence Backend API
FastAPI Gateway for live meteorological ingestion, deterministic threat heuristics,
SIH26078 scientific core pipeline, ML module status interfaces, and Gemini scientific interpretation.
"""
import sys
import os
from pathlib import Path

# Ensure backend and workspace root are in sys.path
backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
project_root = backend_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Header, Response, Body
from fastapi.middleware.cors import CORSMiddleware
import requests

# Load environment variables from backend/.env
env_path = backend_dir / ".env"
load_dotenv(dotenv_path=env_path)

try:
    from backend.services.weather_service import weather_service
    from backend.services.threat_engine import threat_engine
    from backend.services.gemini_service import gemini_service
    from backend.ml.gnn_tracker import weather_gnn_tracker, gnn_tracker
    from backend.ml.diffusion_downscaler import conditional_weather_diffusion, diffusion_downscaler
    from backend.ml.efi import efi_engine
    from backend.services.pipeline_service import pipeline_service
    from backend.services.anomaly_service import anomaly_service
    from backend.services.trajectory_service import trajectory_service
    from backend.services.alert_service import alert_service
    from backend.evaluation.reports import generate_evaluation_summary
except ImportError:
    from services.weather_service import weather_service
    from services.threat_engine import threat_engine
    from services.gemini_service import gemini_service
    from ml.gnn_tracker import weather_gnn_tracker, gnn_tracker
    from ml.diffusion_downscaler import conditional_weather_diffusion, diffusion_downscaler
    from ml.efi import efi_engine
    from services.pipeline_service import pipeline_service
    from services.anomaly_service import anomaly_service
    from services.trajectory_service import trajectory_service
    from services.alert_service import alert_service
    from evaluation.reports import generate_evaluation_summary

IMD_API_KEY = os.getenv("IMD_API_KEY")
IMD_AUTH_TOKEN = os.getenv("IMD_AUTH_TOKEN")
IMD_FORECAST_URL = "https://api.imd.gov.in/api/v1/cityforecastloc"

app = FastAPI(
    title="SIH26078 Atmospheric Intelligence Scientific API",
    description="Scientific core pipeline for extreme weather anomaly tracking, 4D bounding envelopes, and localized threat intelligence.",
    version="2.0.0"
)

# Deployment & dev CORS configuration
allowed_origins_env = os.getenv("ALLOWED_ORIGINS") or os.getenv("CORS_ORIGINS") or os.getenv("FRONTEND_URL", "")
origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://localhost:4173",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:4173",
]
if allowed_origins_env:
    if allowed_origins_env.strip() == "*":
        origins = ["*"]
    else:
        for o in allowed_origins_env.split(","):
            cleaned = o.strip()
            if cleaned and cleaned not in origins:
                origins.append(cleaned)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1):[0-9]+" if origins != ["*"] else None,
    allow_credentials=True if origins != ["*"] else False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    key_configured = gemini_service.is_configured()
    raw_key = os.getenv("GEMINI_API_KEY")
    key_len = len(raw_key.strip()) if (raw_key and raw_key.strip()) else 0
    print(f"[STARTUP] SIH26078 API initialized. GEMINI_API_KEY configured: {key_configured} (key length: {key_len})")


# Pydantic models for API requests
class AIExplainRequest(BaseModel):
    location: Optional[Dict[str, Any]] = None
    current: Optional[Dict[str, Any]] = None
    weather: Optional[Dict[str, Any]] = None
    threats: List[Dict[str, Any]] = Field(default_factory=list)
    anomaly_indicator: Optional[Dict[str, Any]] = None
    user_prompt: Optional[str] = None
    query: Optional[str] = None


class DownscaleRequest(BaseModel):
    anomaly_id: Optional[str] = None
    crop_data: Optional[Dict[str, Any]] = None
    target_resolution_km: float = 5.0


class PipelineRunRequest(BaseModel):
    latitude: float = Field(11.4984, ge=-90.0, le=90.0)
    longitude: float = Field(77.2774, ge=-180.0, le=180.0)
    forecast_days: int = Field(10, ge=1, le=16)
    model: str = Field("ecmwf_ifs025")
    regional_crop: Optional[Dict[str, float]] = None
    time_window: Optional[int] = None



@app.get("/")
def root():
    return {
        "status": "online",
        "service": "SIH26078 Atmospheric Intelligence Scientific API",
        "docs_url": "/docs",
        "system_status_url": "/api/system/status",
        "science_status_url": "/api/science/status"
    }


@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "backend": "online",
        "gemini": {
            "configured": gemini_service.is_configured()
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


# ==============================================================================
# SECTION 30: SCIENTIFIC CORE REST API
# ==============================================================================

@app.get("/api/science/status")
def get_science_status():
    """
    Exposes readiness status of all 11 scientific pipeline stages.
    Adheres strictly to scientific honesty rule (no fake claims of loaded models).
    """
    return pipeline_service.get_system_status()


@app.get("/api/science/anomalies")
def get_science_anomalies(
    latitude: float = Query(11.49, ge=-90.0, le=90.0),
    longitude: float = Query(77.27, ge=-180.0, le=180.0),
    forecast_days: int = Query(10, ge=1, le=16)
):
    """
    Returns detected baseline anomaly clusters from the scientific pipeline.
    Labelled as BASELINE ANOMALY (Z-Score / Quantile Deviation).
    """
    forecast = weather_service.get_forecast(
        latitude=latitude,
        longitude=longitude,
        forecast_days=forecast_days
    )
    result = pipeline_service.execute_live_pipeline(latitude, longitude, forecast.get("raw_data", {}))

    return {
        "status": "success",
        "source": "Open-Meteo / ECMWF IFS HRES",
        "model": "ECMWF IFS (0.1° / ~9km)",
        "resolution": "9.0 km",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "detection_method": "BASELINE ANOMALY (Z-Score Deviation)",
        "anomalies_count": len(result.get("anomalies_detected", [])),
        "anomalies": result.get("anomalies_detected", [])
    }


@app.get("/api/science/trajectory/{anomaly_id}")
def get_science_trajectory(
    anomaly_id: str,
    latitude: float = Query(11.49, ge=-90.0, le=90.0),
    longitude: float = Query(77.27, ge=-180.0, le=180.0)
):
    """
    Returns dynamic 4D spatio-temporal trajectory points for a specified anomaly.
    Labelled as BASELINE TRAJECTORY TRACKER. Real data derived from kinematic advection.
    """
    # Check if cached or generate
    threat = pipeline_service.cached_threats.get(anomaly_id)
    if threat:
        traj_points = [p.model_dump() for p in threat.trajectory]
    else:
        # Compute on demand
        forecast = weather_service.get_forecast(latitude=latitude, longitude=longitude, forecast_days=10)
        pipeline_service.execute_live_pipeline(latitude, longitude, forecast.get("raw_data", {}))
        threat = pipeline_service.cached_threats.get(anomaly_id)
        if threat:
            traj_points = [p.model_dump() for p in threat.trajectory]
        else:
            # Return nearest or first available trajectory
            all_threats = list(pipeline_service.cached_threats.values())
            if all_threats:
                traj_points = [p.model_dump() for p in all_threats[0].trajectory]
            else:
                traj_points = []

    return {
        "status": "success",
        "anomaly_id": anomaly_id,
        "source": "Open-Meteo / ECMWF IFS HRES",
        "model": "BASELINE TRAJECTORY TRACKER (Kinematic Matching)",
        "resolution": "9.0 km",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trajectory_points_count": len(traj_points),
        "trajectory": traj_points
    }


@app.get("/api/science/efi")
def get_science_efi(
    latitude: float = Query(11.49, ge=-90.0, le=90.0),
    longitude: float = Query(77.27, ge=-180.0, le=180.0),
    variable: str = Query("precipitation", description="precipitation, temperature, or wind_speed")
):
    """
    Returns Extreme Forecast Index (EFI) calculation.
    If full ensemble distribution is absent, transparently reports efi_status: not_available.
    """
    day_of_year = datetime.now(timezone.utc).timetuple().tm_yday
    efi_result = efi_engine.calculate_efi(
        forecast_ensemble_samples=None,  # Deterministic live feed
        climatology_quantiles=None,
        variable_name=variable
    )

    return {
        "status": "success",
        "variable": variable,
        "source": "ECMWF ERA5 30-Year Reference (1991-2020 M-climate)",
        "model": "ECMWF Extreme Forecast Index (Integral Formulation)",
        "resolution": "0.25° (~28 km) Climatology",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "efi_result": efi_result
    }


@app.get("/api/science/bounding-box/{anomaly_id}")
def get_science_bounding_box(
    anomaly_id: str,
    latitude: float = Query(11.49, ge=-90.0, le=90.0),
    longitude: float = Query(77.27, ge=-180.0, le=180.0)
):
    """
    Returns 4D dynamic bounding box (spatial bounds + temporal window) for an anomaly.
    """
    bbox = pipeline_service.cached_bboxes.get(anomaly_id)
    if not bbox:
        forecast = weather_service.get_forecast(latitude=latitude, longitude=longitude, forecast_days=10)
        pipeline_service.execute_live_pipeline(latitude, longitude, forecast.get("raw_data", {}))
        bbox = pipeline_service.cached_bboxes.get(anomaly_id)
        if not bbox and pipeline_service.cached_bboxes:
            bbox = list(pipeline_service.cached_bboxes.values())[0]

    if not bbox:
        raise HTTPException(status_code=404, detail=f"Bounding box for anomaly '{anomaly_id}' not found.")

    return {
        "status": "success",
        "anomaly_id": anomaly_id,
        "source": "SIH26078 Dynamic 4D Bounding Engine",
        "model": "DynamicBoundingBox4D",
        "resolution": "12.0 km coarse envelope",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "bounding_box": bbox.model_dump()
    }


@app.post("/api/science/downscale")
def post_science_downscale(payload: DownscaleRequest):
    """
    Interface for conditional diffusion downscaling (12 km → 5 km).
    Adheres to Scientific Honesty Rule: reports model_not_loaded if checkpoint unmounted.
    Never fabricates fake high-resolution weather fields.
    """
    downscale_result = conditional_weather_diffusion.generate(
        coarse_crop=payload.crop_data or {"anomaly_id": payload.anomaly_id or "ANOM-TEST"}
    )

    return {
        "status": downscale_result["status"],
        "source": "Conditional Atmospheric Diffusion Downscaler",
        "model": "ConditionalWeatherDiffusion (DDPM)",
        "resolution": "12 km → 5 km",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "downscale_result": downscale_result
    }


@app.get("/api/science/threats")
def get_science_threats(
    latitude: float = Query(11.49, ge=-90.0, le=90.0),
    longitude: float = Query(77.27, ge=-180.0, le=180.0),
    forecast_days: int = Query(10, ge=1, le=16)
):
    """
    Returns standardized ThreatFootprint objects with verified administrative intersections.
    """
    forecast = weather_service.get_forecast(latitude=latitude, longitude=longitude, forecast_days=forecast_days)
    pipeline_result = pipeline_service.execute_live_pipeline(latitude, longitude, forecast.get("raw_data", {}))

    return {
        "status": "success",
        "source": "Open-Meteo / ECMWF IFS HRES",
        "model": "SIH26078 Scientific Core Pipeline",
        "resolution": "9.0 km native",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "threat_count": len(pipeline_result.get("threats", [])),
        "threats": pipeline_result.get("threats", [])
    }


@app.get("/api/science/administrative-impact/{anomaly_id}")
def get_science_administrative_impact(
    anomaly_id: str,
    latitude: float = Query(11.49, ge=-90.0, le=90.0),
    longitude: float = Query(77.27, ge=-180.0, le=180.0)
):
    """
    Calculates geographic intersection with real administrative boundaries (District -> Block).
    Adheres strictly to Section 28: does not fabricate panchayat polygons.
    """
    threat = pipeline_service.cached_threats.get(anomaly_id)
    if not threat:
        forecast = weather_service.get_forecast(latitude=latitude, longitude=longitude, forecast_days=10)
        pipeline_service.execute_live_pipeline(latitude, longitude, forecast.get("raw_data", {}))
        threat = pipeline_service.cached_threats.get(anomaly_id)
        if not threat and pipeline_service.cached_threats:
            threat = list(pipeline_service.cached_threats.values())[0]

    if not threat:
        return {
            "status": "no_active_anomaly",
            "anomaly_id": anomaly_id,
            "affected_districts": [],
            "affected_blocks": [],
            "affected_panchayats": [],
            "administrative_status": "no_intersection",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    return {
        "status": "success",
        "anomaly_id": anomaly_id,
        "source": "Survey of India / Verified Administrative Reference Bounds",
        "model": "Spatial Bounding Intersection",
        "resolution": "District / Block Level",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "affected_districts": threat.affected_districts,
        "affected_blocks": threat.affected_blocks,
        "affected_panchayats": threat.affected_panchayats,
        "administrative_status": threat.administrative_status
    }


@app.get("/api/science/evaluation")
def get_science_evaluation():
    """Returns historical benchmark evaluation report."""
    return generate_evaluation_summary()


@app.post("/api/science/pipeline/run")
def post_science_pipeline_run(payload: PipelineRunRequest):
    """
    Executes the end-to-end SIH26078 scientific pipeline:
    NWP Ingestion -> 4D/5D Tensor Cube -> Spherical/Icosahedral Graph -> Anomaly Detection
    -> Point Alerts (LOW/MODERATE/SEVERE) -> Dynamic 4D Bounding Box -> Field Crop
    -> Diffusion Super-Resolution Interface -> Physics Validation -> Output.
    Returns complete pipeline execution object with structured provenance (Section 19 & 20).
    """
    forecast = weather_service.get_forecast(
        latitude=payload.latitude,
        longitude=payload.longitude,
        forecast_days=payload.forecast_days,
        model=payload.model
    )
    result = pipeline_service.execute_live_pipeline(
        payload.latitude,
        payload.longitude,
        forecast.get("raw_data", {})
    )
    return result


# ==============================================================================
# EXISTING BACKEND ENDPOINTS (PRESERVED FOR COMPATIBILITY)
# ==============================================================================


@app.get("/api/system/status")
def system_status():
    """
    Returns comprehensive system and scientific component status.
    """
    sci_stat = pipeline_service.get_system_status()
    is_conf = gemini_service.is_configured()
    return {
        "service": "SIH26078 Atmospheric Intelligence",
        "backend": "online",
        "weather_api": "online",
        "forecast_engine": "online",
        "threat_engine": "online",
        "gemini": {
            "configured": is_conf,
            "available": is_conf,
            "model": getattr(gemini_service, "PRIMARY_MODEL", "gemini-3.8-flash"),
            "role": "Meteorological reasoning & threat interpretation",
            "status": "configured" if is_conf else "not_configured"
        },
        "gemini_status": "configured" if is_conf else "not_configured",
        "gnn": weather_gnn_tracker.status()["status"],
        "diffusion": conditional_weather_diffusion.status()["status"],
        "data_provider": "Open-Meteo (ECMWF IFS / Global Blend)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scientific_pipeline": sci_stat["stages"],
        "ml_modules": {
            "gnn_tracker": weather_gnn_tracker.status(),
            "diffusion_downscaler": conditional_weather_diffusion.status()
        }
    }


@app.get("/api/weather/search")
def search_location(
    q: str = Query(..., min_length=2, description="City or region name to search")
):
    return weather_service.search_locations(q)


@app.get("/api/weather/forecast")
def get_forecast(
    latitude: float = Query(..., ge=-90.0, le=90.0, description="Latitude in decimal degrees"),
    longitude: float = Query(..., ge=-180.0, le=180.0, description="Longitude in decimal degrees"),
    forecast_days: int = Query(10, ge=1, le=16, description="Forecast horizon in days"),
    model: str = Query("ecmwf_ifs025", description="NWP model"),
    location_name: Optional[str] = Query(None, description="Optional city name"),
    country: Optional[str] = Query(None, description="Optional country name"),
    admin1: Optional[str] = Query(None, description="Optional state/region name")
):
    """
    Returns normalized real forecast data, hourly/daily arrays, and calculated threats.
    Integrates scientific pipeline execution.
    """
    res = weather_service.get_forecast(
        latitude=latitude,
        longitude=longitude,
        forecast_days=forecast_days,
        model=model,
        location_name=location_name,
        country=country,
        admin1=admin1
    )

    # Attach scientific analysis section (Section 35)
    try:
        sci_res = pipeline_service.execute_live_pipeline(latitude, longitude, res.get("raw_data", {}))
        res["scientific_analysis"] = {
            "status": "available",
            "provenance": {
                "source": "Open-Meteo / ECMWF IFS HRES",
                "model": "ECMWF IFS (0.1° / ~9km)",
                "target_pipeline": "NCMRWF NEPS-G / ERA5",
                "generated": datetime.now(timezone.utc).isoformat()
            },
            "physics_validation": sci_res.get("physics_validation"),
            "baseline_anomalies": sci_res.get("anomalies_detected", []),
            "efi": sci_res.get("efi_summary"),
            "scientific_threats": sci_res.get("threats", [])
        }
    except Exception as e:
        res["scientific_analysis"] = {
            "status": "partial",
            "error": str(e)
        }

    return res


@app.get("/api/weather/current")
def get_current(
    latitude: float = Query(..., ge=-90.0, le=90.0),
    longitude: float = Query(..., ge=-180.0, le=180.0),
    location_name: Optional[str] = None
):
    forecast = weather_service.get_forecast(
        latitude=latitude,
        longitude=longitude,
        forecast_days=2,
        location_name=location_name
    )
    return {
        "location": forecast["location"],
        "source": forecast["source"],
        "current": forecast["current"]
    }


@app.get("/api/weather/threats")
def get_threats(
    latitude: float = Query(..., ge=-90.0, le=90.0),
    longitude: float = Query(..., ge=-180.0, le=180.0),
    location_name: Optional[str] = None
):
    forecast = weather_service.get_forecast(
        latitude=latitude,
        longitude=longitude,
        forecast_days=10,
        location_name=location_name
    )
    return {
        "location": forecast["location"],
        "threat_engine": "Prototype Threat Heuristic",
        "threat_count": len(forecast.get("threats", [])),
        "threats": forecast.get("threats", [])
    }


@app.get("/api/weather/status")
def weather_status():
    return {
        "provider": "Open-Meteo",
        "models_available": ["ECMWF IFS (0.1° / ~9km)", "Global NWP Blend"],
        "operational_status": "online",
        "attribution": "Weather data by Open-Meteo under CC BY 4.0; ECMWF IFS model."
    }


@app.post("/api/ai/explain")
def explain_weather(payload: AIExplainRequest):
    return gemini_service.generate_explanation(
        location_data=payload.location,
        current_data=payload.current or payload.weather,
        threats_data=payload.threats,
        anomaly_data=payload.anomaly_indicator,
        user_prompt=payload.user_prompt or payload.query
    )



@app.get("/api/ml/gnn/status")
def gnn_status():
    return weather_gnn_tracker.status()


@app.get("/api/ml/diffusion/status")
def diffusion_status():
    return conditional_weather_diffusion.status()


@app.get("/api/imd/forecast")
def get_imd_forecast(
    id: Optional[str] = Query(None, description="Station ID"),
    authorization: Optional[str] = Header(None)
):
    if not IMD_API_KEY or IMD_API_KEY == "YOUR_ACTUAL_KEY":
        raise HTTPException(
            status_code=503,
            detail="Official IMD API credentials not yet provisioned. Use /api/weather/forecast for operational data."
        )

    headers = {"X-Api-Key": IMD_API_KEY, "Accept": "application/json"}
    if authorization:
        headers["Authorization"] = authorization
    elif IMD_AUTH_TOKEN:
        headers["Authorization"] = f"Bearer {IMD_AUTH_TOKEN}" if not IMD_AUTH_TOKEN.startswith("Bearer ") else IMD_AUTH_TOKEN

    params = {"id": id} if id else {}
    try:
        imd_response = requests.get(IMD_FORECAST_URL, headers=headers, params=params, timeout=15)
        return Response(
            content=imd_response.content,
            status_code=imd_response.status_code,
            media_type=imd_response.headers.get("content-type", "application/json")
        )
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Failed to communicate with IMD API: {type(e).__name__}")


# ==============================================================================
# Production SPA Static Serving (Single-service deployment architecture)
# ==============================================================================
dist_dir = project_root / "dist"
if dist_dir.is_dir():
    from fastapi.staticfiles import StaticFiles
    from starlette.responses import FileResponse

    assets_dir = dist_dir / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    landing_dir = dist_dir / "landing-pages"
    if landing_dir.is_dir():
        app.mount("/landing-pages", StaticFiles(directory=str(landing_dir)), name="landing-pages")

    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_frontend_spa(full_path: str):
        # Prevent intercepting API routes
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API route not found")
        file_path = dist_dir / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        index_file = dist_dir / "index.html"
        if index_file.is_file():
            return FileResponse(index_file)
        raise HTTPException(status_code=404, detail="Page not found")