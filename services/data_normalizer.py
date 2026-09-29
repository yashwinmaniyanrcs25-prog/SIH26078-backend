"""
Data Normalization Service for SIH26078.
Translates raw meteorological API responses into the unified SIH26078 schema.
Strictly preserves real data and sets missing values to None. Never fabricates.
"""
from typing import Any, Dict, List, Optional
from datetime import datetime

WMO_CODE_MAP: Dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail"
}


def get_weather_condition(code: Optional[int]) -> str:
    if code is None:
        return "Unknown"
    return WMO_CODE_MAP.get(code, f"Weather code {code}")


def calculate_anomaly_indicators(current: Dict[str, Any], hourly: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes prototype anomaly indices using statistical deviation of forecast horizon variables.
    Clearly labeled as 'EXTREME ANOMALY INDICATOR' (prototype heuristic, not official ECMWF EFI).
    """
    precip_values = [h["precipitation"] for h in hourly if h.get("precipitation") is not None]
    wind_values = [h["wind_speed"] for h in hourly if h.get("wind_speed") is not None]
    temp_values = [h["temperature"] for h in hourly if h.get("temperature") is not None]
    cape_values = [h["cape"] for h in hourly if h.get("cape") is not None]

    max_hourly_precip = max(precip_values) if precip_values else 0.0
    max_wind = max(wind_values) if wind_values else 0.0
    max_temp = max(temp_values) if temp_values else 0.0
    min_temp = min(temp_values) if temp_values else 0.0
    max_cape = max(cape_values) if cape_values else 0.0

    # Normalized indicator scores [0.0 - 1.0] based on meteorological severity scales
    # Precip: 0-50mm/hr scale
    precip_score = min(1.0, max_hourly_precip / 50.0)
    # Wind: 0-100 km/h scale
    wind_score = min(1.0, max_wind / 90.0)
    # Heat: >35°C up to 45°C
    heat_score = min(1.0, max(0.0, (max_temp - 30.0) / 15.0)) if max_temp > 30 else 0.0
    # Convective: 0-3000 J/kg CAPE
    convective_score = min(1.0, max_cape / 3000.0) if max_cape > 0 else 0.0

    composite_score = round(max(precip_score, wind_score, heat_score, convective_score), 2)

    return {
        "title": "EXTREME ANOMALY INDICATOR",
        "composite_score": composite_score,
        "disclaimer": "Prototype indicator derived from available forecast variables; not official ECMWF EFI.",
        "components": {
            "precipitation_intensity_score": round(precip_score, 2),
            "wind_extremity_score": round(wind_score, 2),
            "thermal_extremity_score": round(heat_score, 2),
            "convective_instability_score": round(convective_score, 2)
        },
        "peak_forecast_metrics": {
            "max_hourly_precipitation_mm": max_hourly_precip,
            "max_wind_speed_kmh": max_wind,
            "max_temperature_c": max_temp,
            "min_temperature_c": min_temp,
            "max_cape_jkg": max_cape
        }
    }


def normalize_open_meteo_forecast(
    raw_data: Dict[str, Any],
    location_meta: Optional[Dict[str, Any]] = None,
    model_name: str = "ECMWF IFS (0.1° / ~9km)"
) -> Dict[str, Any]:
    """
    Normalizes Open-Meteo response into the SIH26078 schema.
    """
    lat = raw_data.get("latitude")
    lon = raw_data.get("longitude")
    elevation = raw_data.get("elevation")
    timezone = raw_data.get("timezone", "UTC")
    utc_offset_seconds = raw_data.get("utc_offset_seconds", 0)

    # Location info
    target_lat = location_meta.get("latitude") if location_meta and location_meta.get("latitude") is not None else lat
    target_lon = location_meta.get("longitude") if location_meta and location_meta.get("longitude") is not None else lon

    location = {
        "name": (location_meta.get("name") if location_meta and location_meta.get("name") else f"{target_lat:.2f}°, {target_lon:.2f}°"),
        "latitude": target_lat,
        "longitude": target_lon,
        "grid_latitude": lat,
        "grid_longitude": lon,
        "country": location_meta.get("country") if location_meta else None,
        "admin1": location_meta.get("admin1") if location_meta else None,
        "elevation_m": elevation,
        "timezone": timezone
    }

    # Source metadata
    source = {
        "provider": "Open-Meteo",
        "model": model_name,
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "resolution": "~9-11 km native grid (ECMWF IFS HRES)",
        "utc_offset_seconds": utc_offset_seconds
    }

    # Hourly structure
    raw_hourly = raw_data.get("hourly", {})
    hourly_times = raw_hourly.get("time", [])
    hourly_records: List[Dict[str, Any]] = []

    h_temp = raw_hourly.get("temperature_2m", [])
    h_rh = raw_hourly.get("relative_humidity_2m", [])
    h_precip = raw_hourly.get("precipitation", [])
    h_rain = raw_hourly.get("rain", [])
    h_showers = raw_hourly.get("showers", [])
    h_weather_code = raw_hourly.get("weather_code", [])
    h_pressure = raw_hourly.get("pressure_msl", [])
    h_cloud = raw_hourly.get("cloud_cover", [])
    h_wind_speed = raw_hourly.get("wind_speed_10m", [])
    h_wind_dir = raw_hourly.get("wind_direction_10m", [])
    h_wind_gusts = raw_hourly.get("wind_gusts_10m", [])
    h_dew_point = raw_hourly.get("dew_point_2m", [])
    h_surface_pressure = raw_hourly.get("surface_pressure", [])
    h_cape = raw_hourly.get("cape", [])

    for i in range(len(hourly_times)):
        code = h_weather_code[i] if i < len(h_weather_code) else None
        hourly_records.append({
            "time": hourly_times[i],
            "temperature": h_temp[i] if i < len(h_temp) else None,
            "relative_humidity": h_rh[i] if i < len(h_rh) else None,
            "precipitation": h_precip[i] if i < len(h_precip) else None,
            "rain": h_rain[i] if i < len(h_rain) else None,
            "showers": h_showers[i] if i < len(h_showers) else None,
            "weather_code": code,
            "weather_condition": get_weather_condition(code),
            "pressure": h_pressure[i] if i < len(h_pressure) else None,
            "cloud_cover": h_cloud[i] if i < len(h_cloud) else None,
            "wind_speed": h_wind_speed[i] if i < len(h_wind_speed) else None,
            "wind_direction": h_wind_dir[i] if i < len(h_wind_dir) else None,
            "wind_gusts": h_wind_gusts[i] if i < len(h_wind_gusts) else None,
            "dew_point": h_dew_point[i] if i < len(h_dew_point) else None,
            "surface_pressure": h_surface_pressure[i] if i < len(h_surface_pressure) else None,
            "cape": h_cape[i] if i < len(h_cape) else None
        })

    # Current conditions: prefer raw_data.get('current') if present, else first hour
    raw_current = raw_data.get("current")
    if raw_current:
        cur_code = raw_current.get("weather_code")
        current = {
            "time": raw_current.get("time"),
            "temperature": raw_current.get("temperature_2m"),
            "relative_humidity": raw_current.get("relative_humidity_2m"),
            "precipitation": raw_current.get("precipitation"),
            "rain": raw_current.get("rain"),
            "weather_code": cur_code,
            "weather_condition": get_weather_condition(cur_code),
            "pressure": raw_current.get("pressure_msl"),
            "cloud_cover": raw_current.get("cloud_cover"),
            "wind_speed": raw_current.get("wind_speed_10m"),
            "wind_direction": raw_current.get("wind_direction_10m"),
            "wind_gusts": raw_current.get("wind_gusts_10m"),
            "dew_point": raw_current.get("dew_point_2m"),
            "surface_pressure": raw_current.get("surface_pressure"),
            "cape": raw_current.get("cape") if "cape" in raw_current else (hourly_records[0]["cape"] if hourly_records else None)
        }
    elif hourly_records:
        h0 = hourly_records[0]
        current = {
            "time": h0["time"],
            "temperature": h0["temperature"],
            "relative_humidity": h0["relative_humidity"],
            "precipitation": h0["precipitation"],
            "rain": h0["rain"],
            "weather_code": h0["weather_code"],
            "weather_condition": h0["weather_condition"],
            "pressure": h0["pressure"],
            "cloud_cover": h0["cloud_cover"],
            "wind_speed": h0["wind_speed"],
            "wind_direction": h0["wind_direction"],
            "wind_gusts": h0["wind_gusts"],
            "dew_point": h0["dew_point"],
            "surface_pressure": h0["surface_pressure"],
            "cape": h0["cape"]
        }
    else:
        current = {}

    # Daily structure
    raw_daily = raw_data.get("daily", {})
    daily_times = raw_daily.get("time", [])
    daily_records: List[Dict[str, Any]] = []

    d_temp_max = raw_daily.get("temperature_2m_max", [])
    d_temp_min = raw_daily.get("temperature_2m_min", [])
    d_precip_sum = raw_daily.get("precipitation_sum", [])
    d_precip_prob = raw_daily.get("precipitation_probability_max", [])
    d_wind_max = raw_daily.get("wind_speed_10m_max", [])
    d_gust_max = raw_daily.get("wind_gusts_10m_max", [])
    d_weather_code = raw_daily.get("weather_code", [])

    for i in range(len(daily_times)):
        d_code = d_weather_code[i] if i < len(d_weather_code) else None
        daily_records.append({
            "date": daily_times[i],
            "temperature_max": d_temp_max[i] if i < len(d_temp_max) else None,
            "temperature_min": d_temp_min[i] if i < len(d_temp_min) else None,
            "precipitation_sum": d_precip_sum[i] if i < len(d_precip_sum) else None,
            "precipitation_probability_max": d_precip_prob[i] if i < len(d_precip_prob) else None,
            "wind_speed_max": d_wind_max[i] if i < len(d_wind_max) else None,
            "wind_gusts_max": d_gust_max[i] if i < len(d_gust_max) else None,
            "weather_code": d_code,
            "weather_condition": get_weather_condition(d_code)
        })

    anomaly_indicator = calculate_anomaly_indicators(current, hourly_records)

    return {
        "location": location,
        "source": source,
        "current": current,
        "hourly": hourly_records,
        "daily": daily_records,
        "anomaly_indicator": anomaly_indicator
    }
