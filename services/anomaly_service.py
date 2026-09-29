"""
SIH26078 — Anomaly Detection & Spatial Connected Components Service
Identifies extreme weather cores and groups adjacent cells into candidate anomaly clusters.

Scientific Methodology:
- Deterministic Baseline Anomaly: Evaluates standard deviations (Z-score), empirical percentiles,
  and threshold exceedances relative to reference climatology:
    z = (forecast - baseline_mean) / baseline_std
- Standardized severity categories: LOW / MODERATE / SEVERE
- Every anomaly contains:
    anomaly_id, variable, severity, timestamp, latitude, longitude,
    value, baseline_value, anomaly_score, confidence, source
"""
from typing import Any, Dict, List, Optional, Tuple
import math
import uuid
import numpy as np

from backend.data.schemas import AnomalyDetectionResult
from backend.ml.graph_builder import haversine_distance_km


class SpatialConnectedComponentsDetector:
    """
    Groups neighboring grid cells where anomaly exceeds threshold into discrete spatial clusters.
    """

    def __init__(self, adjacency_distance_km: float = 60.0):
        self.adjacency_distance_km = adjacency_distance_km

    def group_cells(
        self,
        candidate_cells: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """
        Groups points within adjacency_distance_km into connected clusters using Breadth-First Search (BFS).
        """
        N = len(candidate_cells)
        if N == 0:
            return []

        visited = [False] * N
        clusters = []

        for i in range(N):
            if visited[i]:
                continue

            cluster = [candidate_cells[i]]
            visited[i] = True
            queue = [i]

            while queue:
                curr_idx = queue.pop(0)
                lat_c = candidate_cells[curr_idx]["latitude"]
                lon_c = candidate_cells[curr_idx]["longitude"]

                for j in range(N):
                    if not visited[j]:
                        lat_j = candidate_cells[j]["latitude"]
                        lon_j = candidate_cells[j]["longitude"]
                        dist = haversine_distance_km(lat_c, lon_c, lat_j, lon_j)
                        if dist <= self.adjacency_distance_km:
                            visited[j] = True
                            queue.append(j)
                            cluster.append(candidate_cells[j])

            clusters.append(cluster)

        return clusters


class BaselineAnomalyDetector:
    """
    Detects meteorological anomalies using statistical z-scores, quantile departures,
    and absolute meteorological threshold exceedance.
    Alert categories strictly follow: LOW / MODERATE / SEVERE.
    """

    # Threshold exceedance triggers for India / Tropics
    ABSOLUTE_THRESHOLDS = {
        "precipitation": {"LOW": 10.0, "MODERATE": 25.0, "SEVERE": 50.0},     # mm/h
        "wind_speed": {"LOW": 40.0, "MODERATE": 60.0, "SEVERE": 85.0},          # km/h
        "temperature": {"LOW": 38.0, "MODERATE": 42.0, "SEVERE": 45.0}          # °C
    }

    def __init__(self):
        self.cluster_engine = SpatialConnectedComponentsDetector()

    def classify_severity(self, z_score: float, abs_val: float, variable: str) -> str:
        """Categorizes severity strictly into LOW, MODERATE, or SEVERE."""
        thresholds = self.ABSOLUTE_THRESHOLDS.get(variable, {})
        # Z-score based or absolute threshold based
        if z_score >= 3.0 or (thresholds and abs_val >= thresholds.get("SEVERE", 999)):
            return "SEVERE"
        elif z_score >= 2.0 or (thresholds and abs_val >= thresholds.get("MODERATE", 999)):
            return "MODERATE"
        elif z_score >= 1.5 or (thresholds and abs_val >= thresholds.get("LOW", 999)):
            return "LOW"
        return "LOW"

    def detect_point_anomalies(
        self,
        time_series: List[Dict[str, Any]],
        variable_name: str = "precipitation",
        baseline_stats: Optional[Dict[str, float]] = None,
        source: str = "Open-Meteo / ECMWF IFS HRES"
    ) -> List[Dict[str, Any]]:
        """
        Detects point anomalies containing complete provenance:
        anomaly_id, variable, severity, timestamp, latitude, longitude,
        value, baseline_value, anomaly_score, confidence, source.
        """
        if not time_series:
            return []

        values = []
        for pt in time_series:
            v = pt.get(variable_name) or pt.get("precipitation_mm_hr") or pt.get("wind_speed_kmh")
            if v is not None:
                values.append(float(v))
            else:
                values.append(0.0)

        arr = np.array(values, dtype=np.float64)
        if len(arr) == 0:
            return []

        if baseline_stats and "mean" in baseline_stats and "std" in baseline_stats:
            b_mean = baseline_stats["mean"]
            b_std = max(baseline_stats["std"], 0.01)
            method = "BASELINE ANOMALY (ERA5 Climatology Z-Score)"
        else:
            b_mean = float(np.mean(arr))
            b_std = max(float(np.std(arr)), 0.01)
            method = "BASELINE ANOMALY (Lead-Time Deviation)"

        anomalies = []
        for idx, pt in enumerate(time_series):
            val = values[idx]
            z = (val - b_mean) / b_std
            abs_thresh_mod = self.ABSOLUTE_THRESHOLDS.get(variable_name, {}).get("LOW", 999.0)

            # Trigger anomaly if z-score >= 1.5 or absolute meteorological threshold exceeded
            if z >= 1.5 or val >= abs_thresh_mod:
                sev = self.classify_severity(z, val, variable_name)
                # Normalized anomaly score in [0.0, 1.0]
                norm_score = round(min(1.0, max(0.0, (z - 1.5) / 3.0)), 3)
                conf = round(min(1.0, 0.80 + 0.05 * min(4.0, z)), 2)

                anom_id = f"ANOM-{variable_name[:4].upper()}-{idx:03d}-{uuid.uuid4().hex[:6]}"
                anomalies.append({
                    "anomaly_id": anom_id,
                    "variable": variable_name,
                    "severity": sev,
                    "timestamp": pt.get("time", ""),
                    "latitude": pt.get("latitude", 0.0),
                    "longitude": pt.get("longitude", 0.0),
                    "value": round(val, 2),
                    "baseline_value": round(b_mean, 2),
                    "anomaly_score": norm_score,
                    "z_score": round(float(z), 2),
                    "confidence": conf,
                    "source": source,
                    "detection_method": method
                })

        return anomalies

    def cluster_spatial_anomalies(
        self,
        anomalous_points: List[Dict[str, Any]],
        variable_name: str = "precipitation"
    ) -> List[AnomalyDetectionResult]:
        """
        Groups anomalous points into spatial connected components.
        Computes centroids, bounding areas, and peak values.
        """
        if not anomalous_points:
            return []

        clusters = self.cluster_engine.group_cells(anomalous_points)
        results = []

        for cluster in clusters:
            lats = [p["latitude"] for p in cluster]
            lons = [p["longitude"] for p in cluster]
            times = [p["timestamp"] for p in cluster]
            vals = [p["value"] for p in cluster]
            z_scores = [p["z_score"] for p in cluster]
            scores = [p.get("anomaly_score", 0.0) for p in cluster]
            confidences = [p.get("confidence", 1.0) for p in cluster]

            centroid_lat = round(sum(lats) / len(lats), 4)
            centroid_lon = round(sum(lons) / len(lons), 4)

            # Spatial spread: bounding rectangle area
            min_lat, max_lat = min(lats), max(lats)
            min_lon, max_lon = min(lons), max(lons)
            lat_span_km = max(10.0, haversine_distance_km(min_lat, centroid_lon, max_lat, centroid_lon))
            lon_span_km = max(10.0, haversine_distance_km(centroid_lat, min_lon, centroid_lat, max_lon))
            area_km2 = round(lat_span_km * lon_span_km, 2)

            peak_idx = int(np.argmax(vals))
            peak_val = round(vals[peak_idx], 2)
            peak_time = times[peak_idx]
            max_z = round(max(z_scores), 2)
            max_score = round(max(scores), 3)
            mean_conf = round(sum(confidences) / len(confidences), 2)

            # Overall cluster severity
            var_type = cluster[0].get("variable", variable_name)
            cluster_sev = self.classify_severity(max_z, peak_val, var_type)
            anom_id = f"CLUST-{var_type[:4].upper()}-{uuid.uuid4().hex[:6]}"

            results.append(AnomalyDetectionResult(
                anomaly_id=anom_id,
                variable=var_type,
                detection_method=cluster[0].get("detection_method", "BASELINE ANOMALY (Z-Score)"),
                start_time=min(times),
                end_time=max(times),
                peak_time=peak_time,
                centroid_lat=centroid_lat,
                centroid_lon=centroid_lon,
                area_km2=area_km2,
                peak_anomaly_value=peak_val,
                mean_anomaly_value=round(float(np.mean(vals)), 2),
                baseline_value=cluster[0].get("baseline_value"),
                z_score=max_z,
                anomaly_score=max_score,
                confidence=mean_conf,
                source=cluster[0].get("source", "Open-Meteo / ECMWF IFS HRES"),
                severity=cluster_sev,
                is_gnn_tracked=False,
                status="DERIVED"
            ))

        return results


# Global singleton
anomaly_service = BaselineAnomalyDetector()
