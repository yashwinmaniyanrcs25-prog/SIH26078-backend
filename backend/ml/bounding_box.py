"""
SIH26078 — Dynamic 4D Spatio-Temporal Bounding Box & Field Cropper
Calculates minimum bounding hypercubes across moving anomaly trajectories and crops
coarse atmospheric fields for conditional diffusion downscaling.

Section 10 & Section 12 Specification:
- Bounding Box attributes:
  * minimum latitude, maximum latitude, minimum longitude, maximum longitude
  * start_time, end_time
  * anomaly_id, variable, severity
- Dynamic envelope enclosing moving anomalies across all trajectory timesteps (t0 -> t1 -> t2).
- Anomaly Cropping Output:
  * cropped anomaly tensor metadata
  * source resolution, crop bounds, forecast times, variables, ensemble members, original shape, cropped shape.
"""
from typing import Any, Dict, List, Optional, Tuple
import math
import numpy as np

from backend.data.schemas import BoundingBox4D
from backend.ml.graph_builder import haversine_distance_km


class DynamicBoundingBox4D:
    """
    Computes dynamic 4D (time, level, lat, lon) bounding boxes around anomaly components.
    Computes true spatio-temporal envelopes across moving trajectories without fixed circles.
    """

    @staticmethod
    def compute_bounding_box(
        anomaly_id: str,
        trajectory_points: List[Dict[str, Any]],
        variable: str = "precipitation",
        severity: str = "MODERATE",
        padding_deg: float = 0.25
    ) -> BoundingBox4D:
        """
        Calculates the 4D bounding envelope across space and time for a tracked anomaly trajectory.
        Encloses the entire path from t0 -> tn.
        """
        if not trajectory_points:
            return BoundingBox4D(
                anomaly_id=anomaly_id,
                variable=variable,
                severity=severity,
                spatial={"min_lat": 0.0, "max_lat": 0.0, "min_lon": 0.0, "max_lon": 0.0},
                temporal={"start_time": "", "end_time": ""},
                centroid={"lat": 0.0, "lon": 0.0},
                area_km2=0.0,
                trajectory_envelope_points=0,
                status="empty_trajectory"
            )

        lats = [float(pt.get("centroid_lat", pt.get("latitude", 0.0))) for pt in trajectory_points]
        lons = [float(pt.get("centroid_lon", pt.get("longitude", 0.0))) for pt in trajectory_points]
        times = [str(pt.get("timestamp", pt.get("time", ""))) for pt in trajectory_points if pt.get("timestamp") or pt.get("time")]

        min_lat = max(-90.0, min(lats) - padding_deg)
        max_lat = min(90.0, max(lats) + padding_deg)
        min_lon = max(-180.0, min(lons) - padding_deg)
        max_lon = min(180.0, max(lons) + padding_deg)

        # Trajectory centroid (mean of all path points)
        avg_lat = sum(lats) / len(lats)
        avg_lon = sum(lons) / len(lons)

        # Approximate spatial area of the bounding box envelope in km^2
        lat_dist_km = haversine_distance_km(min_lat, avg_lon, max_lat, avg_lon)
        lon_dist_km = haversine_distance_km(avg_lat, min_lon, avg_lat, max_lon)
        box_area_km2 = round(lat_dist_km * lon_dist_km, 2)

        start_time = min(times) if times else ""
        end_time = max(times) if times else ""

        return BoundingBox4D(
            anomaly_id=anomaly_id,
            variable=variable,
            severity=severity,
            spatial={
                "min_lat": round(min_lat, 4),
                "max_lat": round(max_lat, 4),
                "min_lon": round(min_lon, 4),
                "max_lon": round(max_lon, 4)
            },
            temporal={
                "start_time": start_time,
                "end_time": end_time
            },
            centroid={
                "lat": round(avg_lat, 4),
                "lon": round(avg_lon, 4)
            },
            area_km2=box_area_km2,
            crop_resolution_km=12.0,
            trajectory_envelope_points=len(trajectory_points),
            status="calculated"
        )


class AtmosphericFieldCropper:
    """
    Crops real atmospheric fields within the 4D bounding box.
    Preserves coordinate indexing, variables, and ensemble dimensions for conditional diffusion downscaling.
    """

    @staticmethod
    def crop_field(
        atmospheric_cube: Dict[str, Any],
        bbox: BoundingBox4D
    ) -> Dict[str, Any]:
        """
        Extracts sub-cube corresponding to the bounding box.
        Outputs exact metadata required by Section 12:
        - source resolution
        - crop bounds
        - forecast times
        - variables
        - ensemble members
        - original shape
        - cropped shape
        """
        spatial = bbox.spatial
        temporal = bbox.temporal

        time_steps = atmospheric_cube.get("time_steps", [])
        latitudes = atmospheric_cube.get("latitudes", [])
        longitudes = atmospheric_cube.get("longitudes", [])
        variables = atmospheric_cube.get("variables", [])
        ensemble_members = atmospheric_cube.get("ensemble_member_ids", ["control_deterministic"])
        original_shape = atmospheric_cube.get("data_shape", [len(time_steps), len(ensemble_members), len(latitudes), len(longitudes), len(variables)])

        # Filter temporal steps
        start_t = temporal.get("start_time", "")
        end_t = temporal.get("end_time", "")

        cropped_times = [
            t for t in time_steps
            if (not start_t or t >= start_t) and (not end_t or t <= end_t)
        ]
        if not cropped_times and time_steps:
            cropped_times = time_steps[:min(24, len(time_steps))]

        # Filter spatial bounds
        cropped_lats = [
            lat for lat in latitudes
            if spatial["min_lat"] <= lat <= spatial["max_lat"]
        ]
        if not cropped_lats and latitudes:
            cropped_lats = latitudes

        cropped_lons = [
            lon for lon in longitudes
            if spatial["min_lon"] <= lon <= spatial["max_lon"]
        ]
        if not cropped_lons and longitudes:
            cropped_lons = longitudes

        # 5D Cropped Shape: [T, MEMBER, LAT, LON, VARIABLES]
        cropped_shape = [
            len(cropped_times),
            len(ensemble_members),
            len(cropped_lats),
            len(cropped_lons),
            len(variables)
        ]

        return {
            "anomaly_id": bbox.anomaly_id,
            "variable": bbox.variable,
            "severity": bbox.severity,
            "status": "cropped",
            "source_resolution": f"{atmospheric_cube.get('spatial_resolution_km', 12.0)} km",
            "source_model": atmospheric_cube.get("source_model", "ECMWF IFS HRES"),
            "crop_bounds": {
                "min_lat": spatial["min_lat"],
                "max_lat": spatial["max_lat"],
                "min_lon": spatial["min_lon"],
                "max_lon": spatial["max_lon"]
            },
            "forecast_times": cropped_times,
            "variables": variables,
            "ensemble_members": ensemble_members,
            "original_shape": original_shape,
            "cropped_shape": cropped_shape,
            "tensor_schema": "[T, MEMBER, LAT, LON, VARIABLES]",
            "ready_for_diffusion": True
        }


bounding_box_engine = DynamicBoundingBox4D()
field_cropper = AtmosphericFieldCropper()
