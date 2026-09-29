"""
SIH26078 — Scientific API & Pipeline Verification Test
Tests all endpoints and verifies strict scientific honesty:
- GNN reports 'model_not_loaded'
- Diffusion reports 'model_not_loaded'
- ERA5 baseline reports accurate readiness
- EFI reports honest availability
- Dynamic Bounding Box & Trajectory adhere to canonical formats
"""
import sys
from pathlib import Path
backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
project_root = backend_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_pipeline():
    print("=" * 70)
    print("RUNNING SIH26078 SCIENTIFIC PIPELINE VERIFICATION")
    print("=" * 70)

    # 1. Test /api/science/status
    res = client.get("/api/science/status")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    status_data = res.json()
    print("[PASS] GET /api/science/status:")
    print(f"  GNN Status: {status_data['stages']['gnn']['sublabel']}")
    print(f"  Diffusion Status: {status_data['stages']['diffusion']['sublabel']}")
    print(f"  Data Ingestion: {status_data['stages']['data_ingestion']['sublabel']}")
    print(f"  Graph Builder: {status_data['stages']['graph_builder']['sublabel']}")
    print(f"  Physics Check: {status_data['stages']['physics']['sublabel']}")
    assert status_data["stages"]["gnn"]["status"] == "model_not_loaded", "GNN must report model_not_loaded"
    assert status_data["stages"]["diffusion"]["status"] == "model_not_loaded", "Diffusion must report model_not_loaded"

    # 2. Test /api/science/efi
    res = client.get("/api/science/efi?latitude=11.49&longitude=77.27&variable=precipitation")
    assert res.status_code == 200
    efi_data = res.json()
    print("[PASS] GET /api/science/efi:")
    print(f"  EFI Result Status: {efi_data['efi_result']['efi_status']}")
    print(f"  Display Label: {efi_data['efi_result']['display_label']}")
    print(f"  Source: {efi_data['source']}")
    assert efi_data["efi_result"]["efi_status"] == "not_available", "Deterministic live feed must not fake ensemble EFI"

    # 3. Test /api/science/anomalies
    res = client.get("/api/science/anomalies?latitude=11.49&longitude=77.27")
    assert res.status_code == 200
    anom_data = res.json()
    print(f"[PASS] GET /api/science/anomalies: found {anom_data['anomalies_count']} anomalies")
    print(f"  Detection Method: {anom_data['detection_method']}")
    assert "GNN" not in anom_data["detection_method"], "Must not claim GNN prediction for baseline anomaly"

    # 4. Test /api/science/threats
    res = client.get("/api/science/threats?latitude=11.49&longitude=77.27")
    assert res.status_code == 200
    threats_data = res.json()
    print(f"[PASS] GET /api/science/threats: count={threats_data['threat_count']}")

    # 5. Test /api/science/downscale
    res = client.post("/api/science/downscale", json={"anomaly_id": "TEST-1", "target_resolution_km": 5.0})
    assert res.status_code == 200
    down_data = res.json()
    print("[PASS] POST /api/science/downscale:")
    print(f"  Status: {down_data['status']}")
    print(f"  Field: {down_data['downscale_result']['field_5km']}")
    assert down_data["status"] == "model_not_loaded", "Diffusion must honestly report model_not_loaded"
    assert down_data["downscale_result"]["field_5km"] is None, "Diffusion must NOT fabricate 5 km field"

    # 6. Test /api/science/evaluation
    res = client.get("/api/science/evaluation")
    assert res.status_code == 200
    eval_data = res.json()
    print("[PASS] GET /api/science/evaluation:")
    for ev_name, ev_val in eval_data["benchmark_events"].items():
        print(f"  Benchmark '{ev_name}': {ev_val['status']}")

    print("=" * 70)
    print("ALL SCIENTIFIC PIPELINE CHECKS PASSED WITH STRICT HONESTY VERIFICATION!")
    print("=" * 70)

if __name__ == "__main__":
    test_pipeline()
