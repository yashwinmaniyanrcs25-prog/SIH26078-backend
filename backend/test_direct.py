"""
Direct Python Test of SIH26078 Backend Architecture & Real Data Calls
"""
import json
from services.weather_service import weather_service
from services.threat_engine import threat_engine
from services.gemini_service import gemini_service
from ml.gnn_tracker import gnn_tracker
from ml.diffusion_downscaler import diffusion_downscaler

def run_tests():
    print("=== SIH26078 DIRECT VERIFICATION TEST ===")

    print("\n1. Testing Location Search (Geocoding):")
    cities = weather_service.search_locations("Coimbatore")
    assert len(cities) > 0, "No cities found"
    c0 = cities[0]
    print(f"   Found: {c0['name']}, {c0.get('admin1')}, {c0['country']} ({c0['latitude']}°N, {c0['longitude']}°E)")

    print("\n2. Testing Live Open-Meteo ECMWF Forecast for Bannari Amman Institute (11.4984°N, 77.2774°E):")
    fc = weather_service.get_forecast(
        latitude=11.4984,
        longitude=77.2774,
        forecast_days=3,
        location_name="Bannari Amman Institute of Technology",
        country="India",
        admin1="Tamil Nadu"
    )
    print(f"   Provider: {fc['source']['provider']}")
    print(f"   Model: {fc['source']['model']}")
    print(f"   Updated At: {fc['source']['updated_at']}")
    print(f"   Current Temp: {fc['current']['temperature']} °C")
    print(f"   Current Humidity: {fc['current']['relative_humidity']} %")
    print(f"   Current Pressure: {fc['current']['pressure']} hPa")
    print(f"   Current Condition: {fc['current']['weather_condition']}")
    print(f"   Wind Speed: {fc['current']['wind_speed']} km/h (Direction: {fc['current']['wind_direction']}°)")
    print(f"   Total Hourly Slots: {len(fc['hourly'])}")
    print(f"   Total Daily Slots: {len(fc['daily'])}")
    print(f"   Calculated Threats: {len(fc['threats'])}")
    print(f"   Extreme Anomaly Indicator: {fc['anomaly_indicator']['composite_score']} / 1.00")
    print(f"   Disclaimer: {fc['anomaly_indicator']['disclaimer']}")

    assert fc['current']['temperature'] is not None, "Temperature must be a real number"
    assert len(fc['hourly']) == 72, "3 days should yield 72 hourly records"

    print("\n3. Testing GNN Architecture Interface:")
    gnn_status = gnn_tracker.get_status()
    print(f"   GNN Status: {gnn_status['status']}")
    print(f"   GNN Message: {gnn_status['message']}")
    assert gnn_status['status'] == "model_not_loaded"

    print("\n4. Testing Diffusion Downscaling Interface:")
    diff_status = diffusion_downscaler.get_status()
    print(f"   Diffusion Status: {diff_status['status']}")
    print(f"   Diffusion Message: {diff_status['message']}")
    assert diff_status['status'] == "model_not_loaded"

    print("\n5. Testing Threat Engine Heuristics:")
    threats = threat_engine.analyze_forecast(fc)
    print(f"   Threat count evaluated: {len(threats)}")
    print(f"   Engine type: {threat_engine.engine_type}")

    print("\n6. Testing Gemini AI Service Fallback:")
    ai_res = gemini_service.generate_explanation(
        fc["location"],
        fc["current"],
        fc["threats"],
        fc["anomaly_indicator"]
    )
    print(f"   AI Status: {ai_res['status']}")
    print(f"   AI Message: {ai_res['message']}")
    assert ai_res['status'] in ("online", "not_configured")

    print("\n=======================================================")
    print("ALL VERIFICATIONS COMPLETED SUCCESSFULLY WITH REAL DATA!")
    print("=======================================================")

if __name__ == "__main__":
    run_tests()
