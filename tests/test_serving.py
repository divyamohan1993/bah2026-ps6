"""Tests for the AgriStress serving layer (FastAPI API + cache + tiler).

All tests run fully offline with no cloud credentials: the app factory seeds a
deterministic synthetic feature store.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="serving extra (fastapi) not installed")
from fastapi.testclient import TestClient

from agristress.serving.api import create_app
from agristress.serving.cache import Cache, cache_key
from agristress.serving.store import seed_demo_store
from agristress.serving.tiler import h3_to_tile, render_demo_tile


@pytest.fixture(scope="module")
def client() -> TestClient:
    app = create_app(seed=True)
    return TestClient(app)


@pytest.fixture(scope="module")
def first_cell() -> str:
    return seed_demo_store().cells()[0]


# -- meta -------------------------------------------------------------------
def test_health_ok(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["cells"] > 0
    assert body["aois"] >= 1
    assert body["cache_backend"] in {"memory", "redis"}


def test_aoi_lists_command_areas(client: TestClient) -> None:
    resp = client.get("/aoi")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] >= 1
    ids = {a["id"] for a in body["aois"]}
    assert "CMD-001" in ids
    aoi0 = body["aois"][0]
    assert aoi0["command_area_ha"] > 0
    assert len(aoi0["h3_cells"]) > 0


# -- keyed-lookup layer endpoints ------------------------------------------
def test_crop_returns_seeded_json(client: TestClient, first_cell: str) -> None:
    resp = client.get("/crop", params={"h3": first_cell})
    assert resp.status_code == 200
    body = resp.json()
    assert body["h3"] == first_cell
    assert 0 <= body["crop_class"] < 6
    assert isinstance(body["crop_label"], str)
    assert 0.0 <= body["confidence"] <= 1.0


def test_stress_returns_seeded_json(client: TestClient, first_cell: str) -> None:
    resp = client.get("/stress", params={"h3": first_cell})
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["stress_index"] <= 1.0
    assert body["stress_label"] in {"none", "mild", "moderate", "severe"}
    assert body["phenology_stage"] in {"sowing", "vegetative", "reproductive", "maturity"}


def test_advisory_returns_seeded_json(client: TestClient, first_cell: str) -> None:
    resp = client.get("/advisory", params={"h3": first_cell})
    assert resp.status_code == 200
    body = resp.json()
    assert body["deficit_mm"] >= 0.0
    assert body["status"] in {"adequate", "mild_deficit", "moderate_deficit", "severe_deficit"}
    assert "recommended_action" in body


def test_unknown_cell_404(client: TestClient) -> None:
    resp = client.get("/crop", params={"h3": "doesnotexist"})
    assert resp.status_code == 404


# -- timeseries -------------------------------------------------------------
def test_timeseries_returns_series(client: TestClient, first_cell: str) -> None:
    resp = client.get("/timeseries", params={"h3": first_cell, "var": "stress_index"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["var"] == "stress_index"
    assert body["count"] >= 1
    assert all({"date", "value"} <= set(p) for p in body["series"])


# -- command rollup ---------------------------------------------------------
def test_command_rollup(client: TestClient) -> None:
    resp = client.get("/command/CMD-001/rollup")
    assert resp.status_code == 200
    body = resp.json()
    assert body["command_id"] == "CMD-001"
    assert body["n_cells"] >= 1
    assert 0.0 <= body["mean_stress_index"] <= 1.0
    assert body["total_irrigation_demand_m3"] >= 0.0
    assert body["dominant_action"]


def test_command_rollup_unknown_404(client: TestClient) -> None:
    resp = client.get("/command/NOPE/rollup")
    assert resp.status_code == 404


# -- tiles ------------------------------------------------------------------
@pytest.mark.parametrize("layer", ["crop", "stress", "advisory"])
def test_tile_endpoint_returns_png(client: TestClient, layer: str) -> None:
    resp = client.get(f"/tiles/{layer}/8/181/110.png")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert resp.content[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic
    assert len(resp.content) > 100


def test_tile_unknown_layer_404(client: TestClient) -> None:
    resp = client.get("/tiles/bogus/8/181/110.png")
    assert resp.status_code == 404


# -- tiler unit -------------------------------------------------------------
def test_render_demo_tile_bytes() -> None:
    png = render_demo_tile("stress", 7, 90, 55)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_h3_to_tile_helper() -> None:
    out = h3_to_tile("87283472bffffff", z=9)
    assert out["z"] == 9
    assert 0 <= out["x"] < 2**9
    assert 0 <= out["y"] < 2**9


# -- cache ------------------------------------------------------------------
def test_inmemory_cache_roundtrip() -> None:
    c = Cache()
    assert c.backend == "memory"
    key = cache_key("crop", h3="abc", date="2025-06-01")
    assert c.get(key) is None
    c.set(key, {"v": 1})
    assert c.get(key) == {"v": 1}


def test_cache_get_or_set_computes_once() -> None:
    c = Cache()
    calls = {"n": 0}

    def producer() -> int:
        calls["n"] += 1
        return 42

    assert c.get_or_set("k", producer) == 42
    assert c.get_or_set("k", producer) == 42
    assert calls["n"] == 1  # producer ran only once


def test_cache_lru_eviction() -> None:
    c = Cache(maxsize=2)
    c.set("a", 1)
    c.set("b", 2)
    c.get("a")  # touch a → b becomes LRU
    c.set("c", 3)  # evicts b
    assert c.get("a") == 1
    assert c.get("c") == 3
    assert c.get("b") is None
