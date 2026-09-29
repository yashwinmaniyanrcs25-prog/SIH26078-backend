"""
Weather Service Gateway for SIH26078.
Fetches real meteorological observations and NWP forecast fields from Open-Meteo (ECMWF IFS).
Implements caching, coordinate validation, error handling, and data normalization.
NEVER fabricates fallback weather data.
"""
from typing import Any, Dict, List, Optional
import time
import requests
from fastapi import HTTPException

try:
    from services.data_normalizer import normalize_open_meteo_forecast
    from services.threat_engine import threat_engine
except ImportError:
    from backend.services.data_normalizer import normalize_open_meteo_forecast
    from backend.services.threat_engine import threat_engine

OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"

# In-memory cache: (cache_key -> (timestamp, data))
_CACHE: Dict[str, tuple[float, Dict[str, Any]]] = {}
CACHE_TTL_SECONDS = 300  # 5 minutes


def _get_cache_key(lat: float, lon: float, days: int, model: str) -> str:
    return f"{round(lat, 4)}:{round(lon, 4)}:{days}:{model}"


class WeatherService:
    def __init__(self):
        self.session = requests.Session()

    def search_locations(self, query: str) -> List[Dict[str, Any]]:
        """Resolves city/region queries to real coordinates."""
        if not query or len(query.strip()) < 2:
            return []

        try:
            resp = self.session.get(
                OPEN_METEO_GEOCODING_URL,
                params={
                    "name": query.strip(),
                    "count": 6,
                    "language": "en",
                    "format": "json"
                },
                timeout=10
            )
            if not resp.ok:
                raise HTTPException(
                    status_code=502,
                    detail=f"Geocoding service error (HTTP {resp.status_code})"
                )

            data = resp.json()
            results = data.get("results", [])
            output = []
            for item in results:
                output.append({
                    "id": item.get("id"),
                    "name": item.get("name"),
                    "latitude": item.get("latitude"),
                    "longitude": item.get("longitude"),
                    "elevation": item.get("elevation"),
                    "country": item.get("country"),
                    "country_code": item.get("country_code"),
                    "admin1": item.get("admin1"),
                    "timezone": item.get("timezone")
                })
            return output
        except requests.RequestException as e:
            raise HTTPException(
                status_code=502,
                detail=f"Failed to connect to geocoding provider: {type(e).__name__}"
            )

    def get_forecast(
        self,
        latitude: float,
        longitude: float,
        forecast_days: int = 10,
        model: str = "ecmwf_ifs025",
        location_name: Optional[str] = None,
        country: Optional[str] = None,
        admin1: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Retrieves real NWP forecast data and executes deterministic threat heuristics.
        """
        # 1. Coordinate Validation
        if not (-90.0 <= latitude <= 90.0):
            raise HTTPException(status_code=400, detail="Latitude must be between -90 and 90 degrees.")
        if not (-180.0 <= longitude <= 180.0):
            raise HTTPException(status_code=400, detail="Longitude must be between -180 and 180 degrees.")
        if not (1 <= forecast_days <= 16):
            raise HTTPException(status_code=400, detail="forecast_days must be between 1 and 16.")

        cache_key = _get_cache_key(latitude, longitude, forecast_days, model)
        now = time.time()

        # Check Cache
        if cache_key in _CACHE:
            ts, cached_data = _CACHE[cache_key]
            if now - ts < CACHE_TTL_SECONDS:
                return cached_data

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": (
                "temperature_2m,relative_humidity_2m,dew_point_2m,precipitation,rain,"
                "weather_code,pressure_msl,surface_pressure,cloud_cover,wind_speed_10m,"
                "wind_direction_10m,wind_gusts_10m"
            ),
            "hourly": (
                "temperature_2m,relative_humidity_2m,dew_point_2m,precipitation,rain,showers,"
                "weather_code,pressure_msl,surface_pressure,cloud_cover,wind_speed_10m,"
                "wind_direction_10m,wind_gusts_10m,cape"
            ),
            "daily": (
                "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,"
                "precipitation_probability_max,wind_speed_10m_max,wind_gusts_10m_max"
            ),
            "forecast_days": forecast_days,
            "timezone": "auto"
        }

        # Attempt with specified ECMWF model, fall back gracefully to blend if model is unavailable
        if model and model != "best_match":
            params["models"] = model

        try:
            resp = self.session.get(OPEN_METEO_FORECAST_URL, params=params, timeout=15)
            if not resp.ok:
                # If specific model rejected, retry with standard high-res ensemble blend
                if "models" in params:
                    params.pop("models")
                    resp = self.session.get(OPEN_METEO_FORECAST_URL, params=params, timeout=15)

            if not resp.ok:
                raise HTTPException(
                    status_code=502,
                    detail=f"Weather API returned error: HTTP {resp.status_code} - {resp.text}"
                )

            raw_json = resp.json()
        except requests.Timeout:
            raise HTTPException(status_code=504, detail="Weather API request timed out.")
        except requests.RequestException as e:
            raise HTTPException(
                status_code=502,
                detail=f"Failed to communicate with weather provider: {type(e).__name__}"
            )

        loc_meta = {
            "name": location_name or f"{latitude:.2f}°, {longitude:.2f}°",
            "latitude": latitude,
            "longitude": longitude,
            "country": country,
            "admin1": admin1
        }

        model_label = "ECMWF IFS (0.1° / ~9km)" if "models" in params else "Open-Meteo Global NWP Blend"
        normalized = normalize_open_meteo_forecast(raw_json, location_meta=loc_meta, model_name=model_label)

        # Run threat detection engine on actual forecast data
        threats = threat_engine.analyze_forecast(normalized)
        normalized["threats"] = threats

        # Cache result
        _CACHE[cache_key] = (now, normalized)
        return normalized


weather_service = WeatherService()
