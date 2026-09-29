"""
SIH26078 — ERA5 & IMDAA Historical Baseline Builder
Offline preprocessing workflow for generating 30-year reference climatology distributions.
Produces reference quantiles for Extreme Forecast Index (EFI) and baseline anomaly detection.

Standard Reference:
- ECMWF Model Climate (M-climate): 30-year reanalysis baseline (configurable period, e.g. 1991-2020)
  with a moving window (+/- 15 days) centered on forecast valid day-of-year.
- Regional Reference: NCMRWF IMDAA 12 km Regional Reanalysis over South Asia.

Scientific Rules:
- If baseline archive is missing:
  return baseline_status = 'not_loaded', message = 'ERA5 climatology unavailable'.
- Absolutely no fabricated 30-year baselines.
"""
import os
import json
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class RegionalClimatologyBuilder:
    """
    Builds and manages regional ERA5 and IMDAA historical reference distributions.
    Extracts mean, standard deviation, and quantiles (p10, p25, p50, p75, p90, p95, p98, p99)
    by variable, geographic cell, and day-of-year.
    """

    DEFAULT_QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.98, 0.99]

    def __init__(
        self,
        output_path: Optional[str] = None,
        baseline_period: Tuple[int, int] = (1991, 2020),
        moving_window_days: int = 15
    ):
        self.output_path = output_path or os.path.join(
            os.path.dirname(__file__), "..", "storage", "regional_baseline.json"
        )
        self.baseline_period = baseline_period
        self.moving_window_days = moving_window_days
        self._cached_distributions = {}
        self._load_existing_baseline()

    def _load_existing_baseline(self):
        """Loads precomputed baseline if available on disk."""
        if os.path.exists(self.output_path):
            try:
                with open(self.output_path, "r") as f:
                    self._cached_distributions = json.load(f)
            except Exception:
                self._cached_distributions = {}

    def get_status(self) -> Dict[str, Any]:
        """Returns baseline availability adhering to Section 8."""
        exists = os.path.exists(self.output_path) and len(self._cached_distributions) > 0
        return {
            "status": "ready" if exists else "not_loaded",
            "baseline_status": "ready" if exists else "not_loaded",
            "baseline_path": self.output_path,
            "period": f"{self.baseline_period[0]}-{self.baseline_period[1]} (30-Year)",
            "moving_window_days": self.moving_window_days,
            "variables_stored": list(self._cached_distributions.keys()),
            "description": "ERA5 / IMDAA 30-Year Regional Climatological Baseline",
            "message": (
                "Regional historical climatology loaded."
                if exists
                else "ERA5 climatology unavailable (30-year reanalysis archive not mounted)."
            )
        }

    def get_climatology(
        self,
        lat: float,
        lon: float,
        variable: str,
        day_of_year: int
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieves reference distribution for a specific coordinate and day-of-year.
        Returns None if not precomputed (honest scientific behavior).
        """
        # Snap coordinates to nearest 0.25 deg ERA5 grid cell
        lat_grid = round(round(lat / 0.25) * 0.25, 2)
        lon_grid = round(round(lon / 0.25) * 0.25, 2)
        key = f"{lat_grid}_{lon_grid}_{variable}"

        var_data = self._cached_distributions.get(key)
        if not var_data:
            return None

        doy_str = str(day_of_year)
        if doy_str in var_data:
            return var_data[doy_str]
        return var_data.get("annual_reference")

    def process_era5_sample_series(
        self,
        variable: str,
        lat: float,
        lon: float,
        values_series: List[float],
        day_of_year: int
    ) -> Dict[str, Any]:
        """Calculates empirical reference distribution from historical time series."""
        arr = np.array(values_series, dtype=np.float64)
        if len(arr) == 0:
            return {}

        quantiles_dict = {}
        for q in self.DEFAULT_QUANTILES:
            quantiles_dict[f"p{int(q*100)}"] = round(float(np.quantile(arr, q)), 2)

        dist = {
            "mean": round(float(np.mean(arr)), 2),
            "std": round(float(np.std(arr)), 2),
            "min": round(float(np.min(arr)), 2),
            "max": round(float(np.max(arr)), 2),
            "quantiles": quantiles_dict,
            "sample_count": len(arr),
            "day_of_year": day_of_year,
            "period": f"{self.baseline_period[0]}-{self.baseline_period[1]}"
        }

        lat_grid = round(round(lat / 0.25) * 0.25, 2)
        lon_grid = round(round(lon / 0.25) * 0.25, 2)
        key = f"{lat_grid}_{lon_grid}_{variable}"

        if key not in self._cached_distributions:
            self._cached_distributions[key] = {}
        self._cached_distributions[key][str(day_of_year)] = dist

        return dist

    def save_baseline(self):
        """Saves precomputed climatological distributions to disk."""
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "w") as f:
            json.dump(self._cached_distributions, f, indent=2)


baseline_builder = RegionalClimatologyBuilder()
