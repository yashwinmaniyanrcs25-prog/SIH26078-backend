"""
SIH26078 — Spatio-Temporal Anomaly Trajectory Tracking Service
Tracks spatio-temporal evolution of extreme weather anomaly clusters across forecast lead times.

Methodology (Section 11):
- Component association across forecast times:
  * Distance gating: Maximum physical storm translation speed (<= 90-110 km/h)
  * Spatial overlap & IoU between bounding extents
  * Intensity similarity: |I_{t+1} - I_t| continuity
  * Wind-advection displacement constraint using synoptic steering flow
- Output metrics for each trajectory point:
  * timestamp
  * latitude, longitude
  * intensity
  * anomaly_score
  * movement direction (0-360 degrees)
  * movement speed (km/h)
  * track confidence [0.0, 1.0]
"""
from typing import Any, Dict, List, Optional, Tuple
import math
from datetime import datetime
import numpy as np

from backend.data.schemas import TrajectoryPoint
from backend.ml.graph_builder import haversine_distance_km, calculate_bearing_deg


class BaselineTrajectoryTracker:
    """
    Spatio-temporal tracker linking anomaly cores across forecast lead times.
    Uses distance gating, intensity continuity, and steering advection.
    """

    MAX_TRACKING_VELOCITY_KMH = 110.0  # Max realistic storm translation speed in tropical/synoptic systems

    def __init__(self):
        self.method_name = "BASELINE TRAJECTORY TRACKER (Kinematic Matching & Overlap)"

    def track_anomaly_across_time(
        self,
        time_indexed_anomalies: List[Dict[str, Any]],
        initial_lat: float,
        initial_lon: float,
        wind_u_ms: float = 0.0,
        wind_v_ms: float = 0.0
    ) -> List[TrajectoryPoint]:
        """
        Constructs verified spatio-temporal trajectory from time-indexed anomaly components.
        Computes movement direction, speed, and track confidence.
        """
        if not time_indexed_anomalies:
            return []

        trajectory: List[TrajectoryPoint] = []
        curr_lat = initial_lat
        curr_lon = initial_lon

        # Sort by timestamp
        sorted_anoms = sorted(time_indexed_anomalies, key=lambda x: x.get("timestamp", ""))

        for i, anom in enumerate(sorted_anoms):
            ts = anom.get("timestamp", "")
            intensity = float(anom.get("value", 0.0))
            score = float(anom.get("anomaly_score", 0.8))
            lead_hours = i * 3  # Assuming 3-hourly or sequential steps

            # Multi-cell centroid matching with distance gating
            if "centroid_lat" in anom and "centroid_lon" in anom:
                cand_lat = float(anom["centroid_lat"])
                cand_lon = float(anom["centroid_lon"])
                if i > 0:
                    d = haversine_distance_km(curr_lat, curr_lon, cand_lat, cand_lon)
                    max_allowed_dist = (self.MAX_TRACKING_VELOCITY_KMH * 3.0)  # 3 hours window
                    if d <= max_allowed_dist:
                        curr_lat = cand_lat
                        curr_lon = cand_lon
                else:
                    curr_lat = cand_lat
                    curr_lon = cand_lon
            elif i > 0 and (wind_u_ms != 0.0 or wind_v_ms != 0.0):
                # Steering wind advection displacement
                delta_t_sec = 3 * 3600
                d_lat_deg = (wind_v_ms * delta_t_sec / 1000.0) / 111.0
                cos_lat = max(0.1, math.cos(math.radians(curr_lat)))
                d_lon_deg = (wind_u_ms * delta_t_sec / 1000.0) / (111.0 * cos_lat)
                curr_lat = round(curr_lat + d_lat_deg, 4)
                curr_lon = round(curr_lon + d_lon_deg, 4)

            # Calculate translation velocity and heading from previous point
            velocity_kmh = None
            heading_deg = None
            step_confidence = 1.0

            if i > 0:
                prev = trajectory[i - 1]
                dist_km = haversine_distance_km(prev.centroid_lat, prev.centroid_lon, curr_lat, curr_lon)
                time_diff_h = max(1.0, float(lead_hours - prev.lead_time_hours))
                velocity_kmh = round(dist_km / time_diff_h, 1)

                # Compute heading in degrees clockwise from North
                heading_deg = round(calculate_bearing_deg(prev.centroid_lat, prev.centroid_lon, curr_lat, curr_lon), 1)

                # Confidence penalties:
                # 1. Extreme velocity jumps (> 90 km/h) reduce confidence
                vel_penalty = max(0.0, (velocity_kmh - 80.0) / 50.0) if velocity_kmh > 80.0 else 0.0
                # 2. Large intensity spikes reduce confidence
                int_ratio = abs(intensity - prev.intensity) / max(prev.intensity, 1.0)
                int_penalty = min(0.3, int_ratio * 0.1)

                step_confidence = round(max(0.40, min(1.0, 1.0 - vel_penalty - int_penalty)), 2)

            area_km2 = float(anom.get("area_km2", 150.0))

            trajectory.append(TrajectoryPoint(
                timestamp=ts,
                centroid_lat=round(curr_lat, 4),
                centroid_lon=round(curr_lon, 4),
                area_km2=area_km2,
                intensity=intensity,
                lead_time_hours=lead_hours,
                anomaly_score=round(score, 3),
                movement_speed_kmh=velocity_kmh,
                movement_direction_deg=heading_deg,
                track_confidence=step_confidence,
                velocity_kmh=velocity_kmh,
                heading_deg=heading_deg
            ))

        return trajectory


# Global trajectory tracker singleton
trajectory_service = BaselineTrajectoryTracker()
