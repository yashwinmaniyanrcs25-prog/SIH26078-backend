"""
SIH26078 — Extreme Forecast Index (EFI) & Shift of Tail (SOT) Implementation
Official ECMWF Extreme Forecast Index formulation comparing forecast ensemble distribution
against historical climatological reference distribution (M-climate).

Mathematical References:
1. Extreme Forecast Index (EFI):
   EFI = (2 / pi) * integral_0^1 [ (p - F_f(Q_c(p))) / sqrt(p * (1 - p)) ] dp
   Where:
   - p in (0, 1): probability level
   - Q_c(p): Quantile function of climatological reference distribution (e.g. ERA5 30-year baseline)
   - F_f(x): Empirical CDF of the forecast ensemble evaluated at x
   - Denominator sqrt(p * (1 - p)): Weights the extreme tails of the distribution
   - EFI range: [-1.0, +1.0]

2. Shift of Tail (SOT):
   Measures how far into the climatological tail the extreme forecast distribution extends:
   SOT = (Q_f(0.90) - Q_c(0.99)) / max(Q_c(0.99) - Q_c(0.90), 1e-4)
   - SOT > 0 indicates that more than 10% of ensemble members exceed the 99th climatological percentile.

Scientific Rules:
- If forecast ensemble is absent or single-member deterministic:
  return status = 'not_available', reason = 'Ensemble forecast distribution unavailable'.
- If reference climatology is absent:
  return status = 'not_available', reason = 'Climatological reference distribution unavailable'.
- Absolutely NO fabricated numbers.
"""
from typing import Any, Dict, List, Optional, Tuple, Union
import math
import numpy as np

# Compatible trapezoid integration for both NumPy 1.x and 2.x
def _numerical_trapezoid(y, x):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    elif hasattr(np, "trapz"):
        return np.trapz(y, x)
    else:
        from scipy.integrate import trapezoid
        return trapezoid(y, x)


class ExtremeForecastIndexEngine:
    """
    Computes ECMWF-standard Extreme Forecast Index and Shift of Tail (SOT) across meteorological variables.
    Supported variables: precipitation, temperature, wind_speed.
    """

    SUPPORTED_VARIABLES = ["precipitation", "temperature", "wind_speed"]

    def __init__(self):
        self.version = "ECMWF-Standard-EFI-SOT-v2.0"

    def calculate_efi(
        self,
        forecast_ensemble_samples: Optional[List[float]],
        climatology_quantiles: Optional[Dict[str, float]],
        variable_name: str = "precipitation",
        num_integration_steps: int = 100
    ) -> Dict[str, Any]:
        """
        Calculates official tail-weighted Extreme Forecast Index and Shift of Tail.
        Args:
            forecast_ensemble_samples: Array of values from all ensemble members for this valid time.
            climatology_quantiles: Dict mapping quantile probabilities (e.g. '0.1', '0.5', '0.9', '0.99') to values.
            variable_name: Name of variable ('precipitation', 'temperature', 'wind_speed').
        Returns:
            Dictionary with efi, sot, reference_quantiles, forecast_distribution, metadata.
        """
        # Section 9 Scientific Honesty Rule: Verify required ensemble distribution exists
        if not forecast_ensemble_samples or len(forecast_ensemble_samples) < 2:
            return {
                "variable": variable_name,
                "efi": None,
                "sot": None,
                "status": "not_available",
                "efi_status": "not_available",
                "reason": "Ensemble forecast distribution unavailable (operational feed is deterministic single-member).",
                "sample_count": len(forecast_ensemble_samples) if forecast_ensemble_samples else 0,
                "baseline_metadata": {
                    "source": "ERA5 30-Year Climatology (1991-2020)",
                    "status": "loaded" if climatology_quantiles else "not_loaded"
                },
                "forecast_metadata": {
                    "ensemble_available": False,
                    "members_count": len(forecast_ensemble_samples) if forecast_ensemble_samples else 0,
                    "model": "ECMWF IFS HRES (Deterministic Operational)"
                },
                "display_label": "NOT AVAILABLE"
            }

        # Check climatological distribution
        if not climatology_quantiles or len(climatology_quantiles) < 3:
            return {
                "variable": variable_name,
                "efi": None,
                "sot": None,
                "status": "not_available",
                "efi_status": "not_available",
                "reason": "Climatological reference distribution (M-climate) unavailable for this coordinate.",
                "sample_count": len(forecast_ensemble_samples),
                "baseline_metadata": {
                    "source": "ERA5 30-Year Climatology (1991-2020)",
                    "status": "not_loaded"
                },
                "forecast_metadata": {
                    "ensemble_available": True,
                    "members_count": len(forecast_ensemble_samples)
                },
                "display_label": "NOT AVAILABLE"
            }

        forecast_arr = np.sort(np.array(forecast_ensemble_samples, dtype=np.float64))
        n_members = len(forecast_arr)

        # Parse climatology quantiles
        p_keys = []
        q_vals = []
        for k, v in climatology_quantiles.items():
            try:
                # Support "p95" or "0.95"
                p_float = float(k.replace("p", "")) / 100.0 if "p" in k else float(k)
                p_keys.append(p_float)
                q_vals.append(float(v))
            except ValueError:
                continue

        if len(p_keys) < 2:
            return {
                "variable": variable_name,
                "efi": None,
                "sot": None,
                "status": "not_available",
                "efi_status": "not_available",
                "reason": "Insufficient quantile levels in climatology.",
                "sample_count": n_members,
                "display_label": "NOT AVAILABLE"
            }

        # Sort quantiles
        sort_idx = np.argsort(p_keys)
        p_sorted = np.array(p_keys)[sort_idx]
        q_sorted = np.array(q_vals)[sort_idx]

        # Numerical integration over p in [epsilon, 1 - epsilon]
        eps = 0.01
        p_grid = np.linspace(eps, 1.0 - eps, num_integration_steps)

        # Interpolate Q_c(p) for each p in p_grid
        qc_values = np.interp(p_grid, p_sorted, q_sorted)

        # Compute empirical CDF F_f(Q_c(p))
        # F_f(x) = fraction of ensemble members <= x
        integrand = []
        for p_val, qc in zip(p_grid, qc_values):
            f_f = np.searchsorted(forecast_arr, qc, side="right") / float(n_members)
            denom = math.sqrt(p_val * (1.0 - p_val))
            if denom > 1e-6:
                integrand.append((p_val - f_f) / denom)
            else:
                integrand.append(0.0)

        # Trapezoidal numerical integration (NumPy 2.x and 1.x safe)
        integral_val = _numerical_trapezoid(integrand, p_grid)
        efi_raw = (2.0 / math.pi) * integral_val
        efi_clamped = max(-1.0, min(1.0, float(efi_raw)))

        # Shift of Tail (SOT) Calculation
        # SOT = (Q_f(0.90) - Q_c(0.99)) / (Q_c(0.99) - Q_c(0.90))
        qf_90 = float(np.percentile(forecast_arr, 90))
        qc_90 = float(np.interp(0.90, p_sorted, q_sorted))
        qc_99 = float(np.interp(0.99, p_sorted, q_sorted))

        denom_sot = max(qc_99 - qc_90, 1e-3)
        sot_val = round((qf_90 - qc_99) / denom_sot, 3)

        return {
            "variable": variable_name,
            "efi": round(efi_clamped, 4),
            "sot": sot_val,
            "status": "available",
            "efi_status": "available",
            "display_label": f"{'+' if efi_clamped > 0 else ''}{efi_clamped:.2f}",
            "sample_count": n_members,
            "reference_quantiles": climatology_quantiles,
            "forecast_ensemble_summary": {
                "member_count": n_members,
                "min": round(float(forecast_arr[0]), 2),
                "median": round(float(np.median(forecast_arr)), 2),
                "max": round(float(forecast_arr[-1]), 2),
                "p90": qf_90
            },
            "baseline_metadata": {
                "source": "ERA5 30-Year Climatology (1991-2020)",
                "qc_90": round(qc_90, 2),
                "qc_99": round(qc_99, 2),
                "status": "loaded"
            },
            "forecast_metadata": {
                "ensemble_available": True,
                "member_count": n_members,
                "qf_90": qf_90
            },
            "formula": "EFI = (2/pi) * integral_0^1 [ (p - F_f(Q_c(p))) / sqrt(p*(1-p)) ] dp | SOT = (Q_f(0.90) - Q_c(0.99)) / (Q_c(0.99) - Q_c(0.90))",
            "source_provenance": "ECMWF Extreme Forecast Index formulation against ERA5 M-climate"
        }

    def compute_multivariable_efi(
        self,
        forecast_ensemble_data: Dict[str, List[float]],
        climatology_data: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """Calculates EFI and SOT across all supported variables."""
        results = {}
        for var in self.SUPPORTED_VARIABLES:
            f_samples = forecast_ensemble_data.get(var)
            c_quantiles = climatology_data.get(var)
            results[var] = self.calculate_efi(f_samples, c_quantiles, variable_name=var)
        return results


# Global EFI engine singleton
efi_engine = ExtremeForecastIndexEngine()
