"""
SIH26078 — Deterministic Prototype Threat Analysis Engine
Calculates derived threat indicators from real NWP forecast arrays.
Clearly labeled as 'Prototype Threat Heuristic' / 'Derived Threat Indicator'.
Never fabricates threats or claims to be official IMD warnings without official feeds.
"""
from typing import Any, Dict, List, Optional
from datetime import datetime

# Configurable heuristic thresholds
THRESHOLDS = {
    # Rainfall
    "EXTREME_RAIN_HOURLY_MM": 35.0,     # Extreme localized downpour (hourly)
    "HEAVY_RAIN_HOURLY_MM": 15.0,       # Heavy rain rate (hourly)
    "HEAVY_RAIN_DAILY_MM": 64.5,        # IMD-like standard heavy rainfall threshold (24h)
    "VERY_HEAVY_RAIN_DAILY_MM": 115.6,  # IMD-like very heavy rain threshold (24h)

    # Wind
    "GALE_WIND_GUST_KMH": 75.0,         # Severe gale / cyclone gusts
    "STRONG_WIND_GUST_KMH": 50.0,       # Strong wind gusts
    "HIGH_SUSTAINED_WIND_KMH": 40.0,    # High sustained 10m wind

    # Temperature
    "EXTREME_HEAT_MAX_C": 42.0,         # Severe heatwave condition
    "HIGH_HEAT_MAX_C": 38.0,            # Heat warning
    "COLD_WAVE_MIN_C": 5.0,             # Cold wave condition

    # Atmospheric Instability
    "HIGH_CAPE_JKG": 1800.0,            # High convective available potential energy
    "MODERATE_CAPE_JKG": 1000.0         # Moderate convective instability
}


class ThreatEngine:
    """
    Evaluates real hourly and daily forecasts against meteorological thresholds.
    Produces structured threat indicators.
    """

    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        self.thresholds = thresholds or THRESHOLDS
        self.engine_type = "Prototype Threat Heuristic"

    def analyze_forecast(
        self,
        normalized_forecast: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Inspects hourly and daily arrays to find any matching extreme conditions.
        Returns a list of detected threats, or [] if none are present.
        """
        threats: List[Dict[str, Any]] = []
        loc = normalized_forecast.get("location", {})
        lat = loc.get("latitude")
        lon = loc.get("longitude")
        loc_name = loc.get("name", "Unknown Location")

        hourly = normalized_forecast.get("hourly", [])
        daily = normalized_forecast.get("daily", [])

        # 1. Hourly Rainfall Threat Search
        max_hourly_rain = 0.0
        max_hourly_rain_time = None
        for h in hourly:
            precip = h.get("precipitation") or 0.0
            if precip > max_hourly_rain:
                max_hourly_rain = precip
                max_hourly_rain_time = h.get("time")

        if max_hourly_rain >= self.thresholds["EXTREME_RAIN_HOURLY_MM"]:
            threats.append({
                "id": "THREAT-EX-RAIN",
                "engine_type": self.engine_type,
                "threat_type": "EXTREME RAINFALL",
                "severity": "CRITICAL",
                "indicator": "Severe Flash Flood / Cloudburst Heuristic",
                "location_name": loc_name,
                "latitude": lat,
                "longitude": lon,
                "forecast_window": max_hourly_rain_time,
                "triggering_variables": {
                    "peak_rate_mm_hr": round(max_hourly_rain, 1),
                    "threshold_mm_hr": self.thresholds["EXTREME_RAIN_HOURLY_MM"]
                },
                "explanation": (
                    f"Forecasted hourly precipitation of {max_hourly_rain:.1f} mm/h exceeds "
                    f"the extreme threshold of {self.thresholds['EXTREME_RAIN_HOURLY_MM']} mm/h, "
                    "indicating rapid water accumulation and high runoff risk."
                )
            })
        elif max_hourly_rain >= self.thresholds["HEAVY_RAIN_HOURLY_MM"]:
            threats.append({
                "id": "THREAT-HVY-RAIN",
                "engine_type": self.engine_type,
                "threat_type": "HEAVY RAINFALL",
                "severity": "HIGH",
                "indicator": "Intense Precipitation Heuristic",
                "location_name": loc_name,
                "latitude": lat,
                "longitude": lon,
                "forecast_window": max_hourly_rain_time,
                "triggering_variables": {
                    "peak_rate_mm_hr": round(max_hourly_rain, 1),
                    "threshold_mm_hr": self.thresholds["HEAVY_RAIN_HOURLY_MM"]
                },
                "explanation": (
                    f"Forecasted precipitation of {max_hourly_rain:.1f} mm/h exceeds "
                    f"the heavy rain threshold of {self.thresholds['HEAVY_RAIN_HOURLY_MM']} mm/h."
                )
            })

        # 2. Daily Accumulated Rainfall
        for d in daily:
            psum = d.get("precipitation_sum") or 0.0
            d_date = d.get("date")
            if psum >= self.thresholds["VERY_HEAVY_RAIN_DAILY_MM"]:
                threats.append({
                    "id": f"THREAT-VHVY-DAILY-{d_date}",
                    "engine_type": self.engine_type,
                    "threat_type": "VERY HEAVY ACCUMULATION",
                    "severity": "CRITICAL",
                    "indicator": "24h Cumulative Precipitation Heuristic",
                    "location_name": loc_name,
                    "latitude": lat,
                    "longitude": lon,
                    "forecast_window": d_date,
                    "triggering_variables": {
                        "daily_accumulation_mm": round(psum, 1),
                        "threshold_mm": self.thresholds["VERY_HEAVY_RAIN_DAILY_MM"]
                    },
                    "explanation": (
                        f"Projected 24h total rainfall of {psum:.1f} mm on {d_date} exceeds "
                        f"{self.thresholds['VERY_HEAVY_RAIN_DAILY_MM']} mm threshold."
                    )
                })
                break  # Record highest/first daily event

        # 3. Wind and Gusts
        max_gust = 0.0
        max_gust_time = None
        max_sustained_wind = 0.0
        for h in hourly:
            gust = h.get("wind_gusts") or 0.0
            wspd = h.get("wind_speed") or 0.0
            if gust > max_gust:
                max_gust = gust
                max_gust_time = h.get("time")
            if wspd > max_sustained_wind:
                max_sustained_wind = wspd

        if max_gust >= self.thresholds["GALE_WIND_GUST_KMH"]:
            threats.append({
                "id": "THREAT-SEV-WIND",
                "engine_type": self.engine_type,
                "threat_type": "SEVERE WIND",
                "severity": "CRITICAL",
                "indicator": "Severe Gale / Cyclonic Gust Heuristic",
                "location_name": loc_name,
                "latitude": lat,
                "longitude": lon,
                "forecast_window": max_gust_time,
                "triggering_variables": {
                    "peak_gust_kmh": round(max_gust, 1),
                    "max_sustained_kmh": round(max_sustained_wind, 1),
                    "threshold_kmh": self.thresholds["GALE_WIND_GUST_KMH"]
                },
                "explanation": (
                    f"Peak wind gusts of {max_gust:.1f} km/h forecasted at {max_gust_time}. "
                    "Associated with structural threat and debris displacement."
                )
            })
        elif max_gust >= self.thresholds["STRONG_WIND_GUST_KMH"] or max_sustained_wind >= self.thresholds["HIGH_SUSTAINED_WIND_KMH"]:
            threats.append({
                "id": "THREAT-STR-WIND",
                "engine_type": self.engine_type,
                "threat_type": "STRONG WIND",
                "severity": "MODERATE",
                "indicator": "Elevated Wind Velocity Heuristic",
                "location_name": loc_name,
                "latitude": lat,
                "longitude": lon,
                "forecast_window": max_gust_time,
                "triggering_variables": {
                    "peak_gust_kmh": round(max_gust, 1),
                    "sustained_kmh": round(max_sustained_wind, 1),
                    "threshold_kmh": self.thresholds["STRONG_WIND_GUST_KMH"]
                },
                "explanation": (
                    f"Peak wind gusts reach {max_gust:.1f} km/h (sustained {max_sustained_wind:.1f} km/h)."
                )
            })

        # 4. Thermal Extremes (Heat Wave & Cold Wave)
        max_temp = -999.0
        max_temp_time = None
        min_temp = 999.0
        min_temp_time = None
        for h in hourly:
            t = h.get("temperature")
            if t is not None:
                if t > max_temp:
                    max_temp = t
                    max_temp_time = h.get("time")
                if t < min_temp:
                    min_temp = t
                    min_temp_time = h.get("time")

        if max_temp >= self.thresholds["EXTREME_HEAT_MAX_C"]:
            threats.append({
                "id": "THREAT-EX-HEAT",
                "engine_type": self.engine_type,
                "threat_type": "EXTREME HEAT",
                "severity": "HIGH",
                "indicator": "Severe Thermal Heatwave Heuristic",
                "location_name": loc_name,
                "latitude": lat,
                "longitude": lon,
                "forecast_window": max_temp_time,
                "triggering_variables": {
                    "maximum_temperature_c": round(max_temp, 1),
                    "threshold_c": self.thresholds["EXTREME_HEAT_MAX_C"]
                },
                "explanation": (
                    f"Forecasted temperature reaches {max_temp:.1f} °C on {max_temp_time}. "
                    "Presents severe heat stress risk."
                )
            })

        if min_temp <= self.thresholds["COLD_WAVE_MIN_C"] and min_temp > -900:
            threats.append({
                "id": "THREAT-COLD-WAVE",
                "engine_type": self.engine_type,
                "threat_type": "COLD CONDITIONS",
                "severity": "MODERATE",
                "indicator": "Low Temperature Thermal Heuristic",
                "location_name": loc_name,
                "latitude": lat,
                "longitude": lon,
                "forecast_window": min_temp_time,
                "triggering_variables": {
                    "minimum_temperature_c": round(min_temp, 1),
                    "threshold_c": self.thresholds["COLD_WAVE_MIN_C"]
                },
                "explanation": (
                    f"Minimum forecasted temperature drops to {min_temp:.1f} °C on {min_temp_time}."
                )
            })

        # 5. Thunderstorm Potential & Convective Instability
        for h in hourly:
            wcode = h.get("weather_code")
            cape = h.get("cape") or 0.0
            if wcode in (95, 96, 99) or (cape >= self.thresholds["HIGH_CAPE_JKG"] and (h.get("precipitation") or 0) > 0.5):
                threats.append({
                    "id": f"THREAT-THUNDERSTORM-{h.get('time')}",
                    "engine_type": self.engine_type,
                    "threat_type": "THUNDERSTORM POTENTIAL",
                    "severity": "HIGH" if wcode in (96, 99) else "MODERATE",
                    "indicator": "Convective Instability & Lightning Heuristic",
                    "location_name": loc_name,
                    "latitude": lat,
                    "longitude": lon,
                    "forecast_window": h.get("time"),
                    "triggering_variables": {
                        "weather_code": wcode,
                        "cape_jkg": round(cape, 1),
                        "hourly_rain_mm": h.get("precipitation")
                    },
                    "explanation": (
                        f"Atmospheric convective energy (CAPE: {cape:.0f} J/kg) and convective "
                        f"weather code ({wcode}) at {h.get('time')} indicate strong storm formation potential."
                    )
                })
                break  # Record first storm event

        return threats


threat_engine = ThreatEngine()
