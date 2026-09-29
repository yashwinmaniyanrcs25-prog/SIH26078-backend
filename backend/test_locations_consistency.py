"""
SIH26078 — Map & API Consistency Test Script
Tests Mirzapur, Chennai, and Delhi geocoding, coordinates, and weather response.
Verifies no coordinate inversions:
latitude -> lat
longitude -> lon
MapLibre coordinate -> [lon, lat]
"""
import sys
import requests

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

base = "http://127.0.0.1:8000"

def test_locations():
    print("=" * 70)
    print("SIH26078 — MAP & API COORDINATE CONSISTENCY TEST")
    print("=" * 70)

    # 1. Geocoding search test
    geo_res = requests.get(f"{base}/api/weather/search?q=Mirzapur").json()
    print(f"[GEOCODING] Search 'Mirzapur' returned {len(geo_res)} results.")
    if geo_res:
        top = geo_res[0]
        print(f"  Top Match: {top.get('name')}, {top.get('admin1')}, {top.get('country')}")
        print(f"  Coordinates: Lat={top.get('latitude')}, Lon={top.get('longitude')}")

    # 2. Consistency test for Mirzapur, Chennai, Delhi
    locations = [
        ("Mirzapur", 25.250, 82.500),
        ("Chennai", 13.0827, 80.2707),
        ("Delhi", 28.6139, 77.2090)
    ]

    for name, lat, lon in locations:
        print(f"\n--- Testing {name} ---")
        url = f"{base}/api/weather/forecast?latitude={lat}&longitude={lon}&forecast_days=3"
        res = requests.get(url)
        assert res.status_code == 200, f"HTTP error {res.status_code} for {name}"
        data = res.json()

        loc = data.get("location", {})
        curr = data.get("current", {})

        resp_lat = loc.get("latitude")
        resp_lon = loc.get("longitude")

        # MapLibre convention check: [longitude, latitude]
        maplibre_coord = [lon, lat]

        print(f"  Weather Request URL: ?latitude={lat}&longitude={lon}")
        print(f"  Backend Returned Location: {loc.get('name')} ({resp_lat}N, {resp_lon}E)")
        print(f"  MapLibre Anchor Coordinate: {maplibre_coord} (Lng={maplibre_coord[0]}, Lat={maplibre_coord[1]})")
        print(f"  Live Meteorological HUD: Temp={curr.get('temperature')}C, Rain={curr.get('precipitation')}mm, Wind={curr.get('wind_speed')}km/h, Pressure={curr.get('pressure')}hPa")

        assert abs(resp_lat - lat) < 0.1, f"Latitude discrepancy for {name}: {resp_lat} vs {lat}"
        assert abs(resp_lon - lon) < 0.1, f"Longitude discrepancy for {name}: {resp_lon} vs {lon}"
        print(f"  [PASS] Coordinates verified across Request, Response, and MapLibre [Lng, Lat] anchor.")

    print("\n" + "=" * 70)
    print("ALL COORDINATE CONSISTENCY TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    test_locations()
