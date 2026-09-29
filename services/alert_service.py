"""
SIH26078 — Scientific Threat Footprint & Alert Engine
Generates standardized ThreatFootprint objects and point/coordinate-based alerts.

Methodology (Section 17):
- Alert Categories strictly:
  * LOW
  * MODERATE
  * SEVERE
- Every alert derived from actual anomaly metrics:
  * alert_id
  * severity
  * latitude, longitude
  * valid_time
  * variable
  * intensity
  * confidence
  * source
  * model
  * geometry (GeoJSON Polygon or Point; strictly NO circular fake radius from a single point)
"""
from typing import Any, Dict, List, Optional, Tuple
import uuid
from backend.data.schemas import BoundingBox4D, ThreatFootprint, TrajectoryPoint
from backend.ml.bounding_box import DynamicBoundingBox4D

# Verified Administrative Reference Bounds (Districts and Taluks/Blocks in Southern / Central / Coastal India)
VERIFIED_ADMIN_BOUNDARIES = [
    {
        "district": "Erode",
        "state": "Tamil Nadu",
        "bbox": {"min_lat": 11.0, "max_lat": 11.85, "min_lon": 76.8, "max_lon": 77.8},
        "blocks": ["Sathyamangalam", "Bhavani", "Gobichettipalayam", "Anthiyur", "Erode", "Perundurai"]
    },
    {
        "district": "Coimbatore",
        "state": "Tamil Nadu",
        "bbox": {"min_lat": 10.6, "max_lat": 11.35, "min_lon": 76.6, "max_lon": 77.3},
        "blocks": ["Pollachi", "Mettupalayam", "Sulur", "Coimbatore North", "Coimbatore South", "Annur"]
    },
    {
        "district": "Nilgiris",
        "state": "Tamil Nadu",
        "bbox": {"min_lat": 11.2, "max_lat": 11.7, "min_lon": 76.3, "max_lon": 77.0},
        "blocks": ["Udhagamandalam", "Coonoor", "Kotagiri", "Gudalur", "Pandalur", "Kundah"]
    },
    {
        "district": "Chennai",
        "state": "Tamil Nadu",
        "bbox": {"min_lat": 12.9, "max_lat": 13.25, "min_lon": 80.1, "max_lon": 80.35},
        "blocks": ["Tondiarpet", "Royapuram", "Anna Nagar", "Teynampet", "Alandur", "Sholinganallur"]
    },
    {
        "district": "Wayanad",
        "state": "Kerala",
        "bbox": {"min_lat": 11.5, "max_lat": 11.95, "min_lon": 75.8, "max_lon": 76.45},
        "blocks": ["Vythiri", "Sulthan Bathery", "Mananthavady"]
    },
    {
        "district": "Ernakulam",
        "state": "Kerala",
        "bbox": {"min_lat": 9.75, "max_lat": 10.25, "min_lon": 76.15, "max_lon": 76.75},
        "blocks": ["Kochi", "Aluva", "Paravur", "Kunnathunad", "Kanayannur", "Muvattupuzha"]
    },
    {
        "district": "Bengaluru Urban",
        "state": "Karnataka",
        "bbox": {"min_lat": 12.8, "max_lat": 13.15, "min_lon": 77.4, "max_lon": 77.75},
        "blocks": ["Bengaluru North", "Bengaluru South", "Bengaluru East", "Anekal"]
    }
]


class ScientificAlertService:
    """
    Alert Engine conforming strictly to Section 17:
    Categories: LOW, MODERATE, SEVERE.
    """

    @staticmethod
    def classify_severity(
        hazard_type: str,
        peak_value: float,
        efi_value: Optional[float] = None
    ) -> str:
        """
        Classifies hazard severity strictly into LOW, MODERATE, SEVERE.
        """
        if "RAIN" in hazard_type or hazard_type == "precipitation":
            if peak_value >= 35.0 or (efi_value is not None and efi_value >= 0.80):
                return "SEVERE"
            elif peak_value >= 15.0 or (efi_value is not None and efi_value >= 0.50):
                return "MODERATE"
            else:
                return "LOW"

        elif "WIND" in hazard_type or hazard_type == "wind_speed":
            if peak_value >= 75.0 or (efi_value is not None and efi_value >= 0.80):
                return "SEVERE"
            elif peak_value >= 50.0 or (efi_value is not None and efi_value >= 0.50):
                return "MODERATE"
            else:
                return "LOW"

        elif "HEAT" in hazard_type or hazard_type == "temperature":
            if peak_value >= 44.0 or (efi_value is not None and efi_value >= 0.80):
                return "SEVERE"
            elif peak_value >= 40.0:
                return "MODERATE"
            else:
                return "LOW"

        return "LOW"

    @staticmethod
    def calculate_administrative_intersection(
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float
    ) -> Tuple[List[str], List[str], List[str], str]:
        """
        Computes intersection between anomaly bounding box and administrative boundaries.
        Returns: (affected_districts, affected_blocks, affected_panchayats, status)
        """
        affected_districts = []
        affected_blocks = []

        for admin in VERIFIED_ADMIN_BOUNDARIES:
            b = admin["bbox"]
            disjoint = (
                max_lat < b["min_lat"] or
                min_lat > b["max_lat"] or
                max_lon < b["min_lon"] or
                min_lon > b["max_lon"]
            )
            if not disjoint:
                affected_districts.append(f"{admin['district']} ({admin['state']})")
                affected_blocks.extend(admin["blocks"])

        if affected_districts:
            status = "district_and_block_intersection_active"
            panchayats_note = []
        else:
            status = "outside_target_reference_states"
            panchayats_note = []

        return affected_districts, affected_blocks, panchayats_note, status

    def create_point_alert(
        self,
        variable: str,
        intensity: float,
        latitude: float,
        longitude: float,
        valid_time: str,
        confidence: float = 1.0,
        efi_value: Optional[float] = None,
        source: str = "Open-Meteo / ECMWF IFS HRES",
        model: str = "ECMWF IFS (0.1° / ~9km)"
    ) -> Dict[str, Any]:
        """
        Generates individual point alert adhering strictly to Section 17.
        Does NOT create a circular impact radius. Shows exact coordinates.
        """
        alert_id = f"ALT-{variable[:4].upper()}-{uuid.uuid4().hex[:8].upper()}"
        severity = self.classify_severity(variable, intensity, efi_value)

        return {
            "alert_id": alert_id,
            "severity": severity,  # LOW | MODERATE | SEVERE
            "latitude": round(latitude, 4),
            "longitude": round(longitude, 4),
            "valid_time": valid_time,
            "variable": variable,
            "intensity": round(intensity, 2),
            "confidence": round(confidence, 2),
            "source": source,
            "model": model,
            # Point geometry: no circular radius fabrication
            "geometry": {
                "type": "Point",
                "coordinates": [round(longitude, 4), round(latitude, 4)]
            }
        }

    def generate_threat_footprint(
        self,
        hazard_type: str,
        peak_value: float,
        trajectory: List[TrajectoryPoint],
        efi_value: Optional[float] = None,
        source_model: str = "ECMWF IFS HRES (0.1° / ~9km)",
        triggering_variables: Optional[Dict[str, Any]] = None
    ) -> ThreatFootprint:
        """
        Creates a verified ThreatFootprint object with dynamic 4D bounding box
        and administrative intersection.
        """
        threat_id = f"THREAT-{uuid.uuid4().hex[:8].upper()}"
        severity = self.classify_severity(hazard_type, peak_value, efi_value)

        # Convert trajectory to dicts for bounding box computation
        traj_dicts = [p.model_dump() for p in trajectory]
        bbox = DynamicBoundingBox4D.compute_bounding_box(
            anomaly_id=threat_id,
            trajectory_points=traj_dicts,
            variable=hazard_type,
            severity=severity
        )

        districts, blocks, panchayats, admin_status = self.calculate_administrative_intersection(
            bbox.spatial["min_lat"],
            bbox.spatial["max_lat"],
            bbox.spatial["min_lon"],
            bbox.spatial["max_lon"]
        )

        lead_time = trajectory[0].lead_time_hours if trajectory else 0

        return ThreatFootprint(
            id=threat_id,
            type=hazard_type,
            severity=severity,
            centroid=bbox.centroid,
            bbox=bbox,
            trajectory=trajectory,
            efi=efi_value,
            efi_status="available" if efi_value is not None else "not_available",
            lead_time_hours=lead_time,
            spatial_resolution_km=9.0,
            source=source_model,
            status="DERIVED_OPERATIONAL_ANOMALY",
            triggering_variables=triggering_variables or {"peak_value": peak_value},
            affected_districts=districts,
            affected_blocks=blocks,
            affected_panchayats=panchayats,
            administrative_status=admin_status
        )


# Global alert service singleton
alert_service = ScientificAlertService()
