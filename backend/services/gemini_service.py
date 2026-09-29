"""
SIH26078 — Gemini AI Meteorological Interpretation Service
Translates structured meteorological analysis into concise, scientific briefings.
NEVER invents weather observations or predictions.
Only interprets supplied structured weather & threat data.
"""
import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, List
import requests
from dotenv import load_dotenv

# Configure logger
logger = logging.getLogger("sih26078.gemini")

# Explicitly load environment variables from backend/.env and project root .env
_current_dir = Path(__file__).resolve().parent
_backend_dir = _current_dir.parent
_project_root = _backend_dir.parent

load_dotenv(_backend_dir / ".env", override=True)
load_dotenv(_project_root / ".env", override=False)

# Model configuration: Gemini 3.8 Flash as primary target; no obsolete fallbacks
PRIMARY_MODEL = "gemini-3.8-flash"
FALLBACK_MODELS: List[str] = []
CANDIDATE_MODELS = [PRIMARY_MODEL] + FALLBACK_MODELS

PLACEHOLDER_KEYS = {
    "YOUR_GEMINI_API_KEY",
    "your_gemini_api_key_here",
    "your_gemini_api_key",
    "None",
    "null",
    ""
}


class GeminiService:
    """
    Atmospheric Reasoning & Meteorological Briefing Generator powered by Gemini.
    Strictly constrained as an interpretation layer.
    """

    def __init__(self):
        self._check_and_log_status()

    def _get_api_key(self) -> Optional[str]:
        # Always check environment first
        raw_key = os.getenv("GEMINI_API_KEY")
        if raw_key is not None and raw_key.strip() in PLACEHOLDER_KEYS:
            return None

        if not raw_key:
            load_dotenv(_backend_dir / ".env", override=False)
            load_dotenv(_project_root / ".env", override=False)
            raw_key = os.getenv("GEMINI_API_KEY")

        if raw_key and raw_key.strip() and raw_key.strip() not in PLACEHOLDER_KEYS:
            return raw_key.strip()
        return None

    def _check_and_log_status(self):
        key = self._get_api_key()
        is_conf = bool(key)
        key_len = len(key) if key else 0
        logger.info(f"GEMINI_API_KEY configured: {is_conf} (key length: {key_len})")

    def is_configured(self) -> bool:
        """Returns True only if a genuine, non-placeholder API key is set."""
        return self._get_api_key() is not None

    def generate_explanation(
        self,
        location_data: Optional[Dict[str, Any]] = None,
        current_data: Optional[Dict[str, Any]] = None,
        threats_data: Optional[List[Dict[str, Any]]] = None,
        anomaly_data: Optional[Dict[str, Any]] = None,
        user_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Sends structured meteorological data to Gemini for scientific explanation.
        Adheres to strict Scientific Safety instructions.
        """
        key = self._get_api_key()
        if not key:
            return {
                "status": "not_configured",
                "explanation": None,
                "model": None,
                "message": "AI INTERPRETATION UNAVAILABLE: Gemini API key is not configured on the backend."
            }

        # Strict Scientific Safety Instruction (Section 10)
        system_instruction = (
            "You are an atmospheric intelligence AI assistant for Smart India Hackathon project SIH26078. "
            "You are an interpretation layer. Do not generate or invent meteorological observations or forecast values. "
            "Use only the supplied structured weather, anomaly, trajectory, EFI, and threat data. "
            "If data is missing, say that the data is unavailable. Do not fill missing values with guesses. "
            "Provide a concise, professional 2-3 paragraph atmospheric intelligence briefing tailored to the user inquiry. "
            "Structure your response with: "
            "(1) Current Atmospheric Analysis: Interpret the observed and forecast thermodynamic conditions (temperature, pressure, precipitation flux, wind velocity). "
            "(2) Threat & Instability Evaluation: Address the user's specific inquiry (e.g. flood potential, convective storm risk, or extreme anomaly tracks) based strictly on supplied threat indicators and anomaly scores. "
            "(3) Horizon Trends & Recommendations: Outline medium-range developments and operational precautions indicated by the data."
        )

        inquiry = user_prompt or "Evaluate atmospheric conditions and flood potential."

        structured_context = {
            "project": "SIH26078 Atmospheric Intelligence",
            "target_location": location_data or {"note": "Location coordinates unavailable"},
            "current_weather": current_data or {"note": "Current atmospheric observations unavailable"},
            "detected_threats": threats_data or [],
            "anomaly_indices": anomaly_data or {"note": "No active baseline anomaly"},
            "user_inquiry": inquiry
        }

        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": (
                                f"{system_instruction}\n\n"
                                f"SUPPLIED METEOROLOGICAL CONTEXT (JSON):\n"
                                f"{json.dumps(structured_context, indent=2)}\n\n"
                                f"USER INQUIRY: {inquiry}\n\n"
                                f"Please produce the scientific atmospheric briefing:"
                            )
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 800
            }
        }

        # Request Gemini model (Primary: gemini-3.8-flash)
        last_error = None
        for model_name in CANDIDATE_MODELS:
            # Prefer v1 stable production endpoint, with v1beta fallback
            endpoints = [
                f"https://generativelanguage.googleapis.com/v1/models/{model_name}:generateContent?key={key}",
                f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={key}"
            ]
            for url in endpoints:
                try:
                    resp = requests.post(
                        url,
                        headers={"Content-Type": "application/json"},
                        json=payload,
                        timeout=28
                    )

                    # If 404 or 503 on v1, allow next endpoint (v1beta) to try
                    if resp.status_code in (404, 503) and url == endpoints[0]:
                        continue

                    if not resp.ok:
                        try:
                            err_json = resp.json()
                            err_text = err_json.get("error", {}).get("message") or resp.text
                        except Exception:
                            err_text = resp.text

                        sanitized_err = err_text.replace(key, "[REDACTED_API_KEY]") if key else err_text
                        logger.warning(f"Gemini API returned HTTP {resp.status_code} for '{model_name}': {sanitized_err}")
                        last_error = f"Google API returned HTTP {resp.status_code}: {sanitized_err}"

                        if url != endpoints[-1] and resp.status_code != 429:
                            continue

                        return {
                            "status": "error",
                            "explanation": None,
                            "model": model_name,
                            "message": f"Gemini could not generate the briefing. Google API returned HTTP {resp.status_code}: {sanitized_err}"
                        }

                    result = resp.json()
                    candidates = result.get("candidates", [])
                    if not candidates:
                        return {
                            "status": "error",
                            "explanation": None,
                            "model": model_name,
                            "message": "Gemini could not generate the briefing. No text candidates returned."
                        }

                    parts = candidates[0].get("content", {}).get("parts", [])
                    text = "".join(p.get("text", "") for p in parts if "text" in p).strip()

                    if not text:
                        return {
                            "status": "error",
                            "explanation": None,
                            "model": model_name,
                            "message": "Gemini returned an empty briefing text."
                        }

                    return {
                        "status": "online",
                        "explanation": text,
                        "model": model_name,
                        "message": "Atmospheric briefing generated from verified forecast data."
                    }

                except requests.Timeout:
                    if url != endpoints[-1]:
                        continue
                    return {
                        "status": "timeout",
                        "explanation": None,
                        "model": model_name,
                        "message": "Gemini could not generate the briefing. Upstream request timed out (28s limit)."
                    }
                except Exception as e:
                    sanitized_e = str(e).replace(key, "[REDACTED_API_KEY]") if key else str(e)
                    if url != endpoints[-1]:
                        continue
                    return {
                        "status": "error",
                        "explanation": None,
                        "model": model_name,
                        "message": f"Gemini could not generate the briefing. {type(e).__name__}: {sanitized_e}"
                    }

        return {
            "status": "error",
            "explanation": None,
            "model": PRIMARY_MODEL,
            "message": f"Gemini could not generate the briefing. {last_error}" if last_error else "Gemini could not generate the briefing. No candidate models configured."
        }


# Global singleton instance
gemini_service = GeminiService()
