"""
Rig 2 LAN Compute Worker & Async Fallback Tests (`tests/test_lan_worker.py`)
============================================================================
Tests:
  • FastAPI `/health` and `/crunch` endpoints end-to-end via `fastapi.testclient.TestClient`
    executing real `wgpu` Vulkan 3D knot + 175-galaxy SPARC + Gaia DR4 sweeps!
  • `AsyncLANWorkerDispatcher` graceful non-blocking fallback to local Vulkan worker thread
    when Rig 2 (`http://127.0.0.1:59999`) is unreachable/powered off!
"""

from __future__ import annotations

import time
from fastapi.testclient import TestClient

from weaveforge.compute.lan_worker import AsyncLANWorkerDispatcher, create_worker_app
from weaveforge.core.frameworks import get_mods_by_id


def test_rig2_fastapi_worker_health_and_crunch() -> None:
    app = create_worker_app(prefer_discrete=True)
    client = TestClient(app)

    r_health = client.get("/health")
    assert r_health.status_code == 200
    h_json = r_health.json()
    assert h_json["status"] == "online"
    assert "OpenGL" not in h_json["gpu_adapter"]

    tnap = get_mods_by_id()["tnap_weave"]
    payload = {
        "toml_source": tnap.to_toml_string(),
        "knot_grid_n": 32,
        "knot_steps": 12,
        "optimize_sparc_ml": False,
        "gaia_mock_pairs": 400,
    }
    r_crunch = client.post("/crunch", json=payload)
    assert r_crunch.status_code == 200
    data = r_crunch.json()
    assert data["mod_id"] == "tnap_weave"
    assert data["sparc_175"]["n_galaxies"] == 175
    assert data["knot_3d"]["retention_ratio"] > 0.5
    assert data["gaia_dr4_mock"]["halo_minus_disk_pct"] > 5.0


def test_async_dispatcher_falls_back_cleanly_when_rig2_offline() -> None:
    # Point at an unused closed localhost port so LAN probe fails immediately and triggers local Vulkan fallback
    disp = AsyncLANWorkerDispatcher(worker_url="http://127.0.0.1:59997")
    tnap = get_mods_by_id()["tnap_weave"]
    disp.dispatch_async(tnap, use_lan_if_online=True)

    deadline = time.perf_counter() + 25.0
    while disp.is_running and time.perf_counter() < deadline:
        time.sleep(0.1)

    assert disp.is_running is False
    assert disp.last_result is not None
    assert disp.last_result["sparc_175"]["n_galaxies"] == 175
    assert "Complete" in disp.last_status
