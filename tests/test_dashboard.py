"""Smoke tests for the AgriStress web dashboard.

These assert that the dashboard's key files exist, that index.html wires up
MapLibre GL JS and the app/config scripts, and that the bundled demo data is
valid JSON with the GeoJSON shapes the dashboard depends on. They are
intentionally lightweight (no browser / JS execution) so they run in CI without
a Node toolchain.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = REPO_ROOT / "dashboard"
DEMO = DASHBOARD / "demo"

REQUIRED_FILES = [
    "index.html",
    "app.js",
    "style.css",
    "config.js",
    "README.md",
]

DEMO_GEOJSON_FILES = [
    "command_area.geojson",
    "fields.geojson",
]


def test_dashboard_dir_exists() -> None:
    assert DASHBOARD.is_dir(), f"missing dashboard dir: {DASHBOARD}"


@pytest.mark.parametrize("name", REQUIRED_FILES)
def test_required_file_exists_and_nonempty(name: str) -> None:
    path = DASHBOARD / name
    assert path.is_file(), f"missing dashboard file: {path}"
    assert path.stat().st_size > 0, f"empty dashboard file: {path}"


def test_index_html_references_maplibre() -> None:
    html = (DASHBOARD / "index.html").read_text(encoding="utf-8").lower()
    assert "maplibre-gl" in html, "index.html must load MapLibre GL JS"
    # both the JS and CSS assets should be referenced
    assert "maplibre-gl.js" in html
    assert "maplibre-gl.css" in html


def test_index_html_references_app_and_config() -> None:
    html = (DASHBOARD / "index.html").read_text(encoding="utf-8")
    assert "app.js" in html, "index.html must reference app.js"
    assert "config.js" in html, "index.html must reference config.js"
    assert "style.css" in html, "index.html must reference style.css"
    assert 'id="map"' in html, "index.html must contain a #map container"


def test_index_html_has_core_ui_hooks() -> None:
    """The slider, layer toggles, legend and roll-up containers must exist."""
    html = (DASHBOARD / "index.html").read_text(encoding="utf-8")
    for hook in ("time-slider", "layer-toggles", "legend-body", "rollup-body"):
        assert hook in html, f"index.html missing UI hook: {hook}"


def test_config_defines_global() -> None:
    cfg = (DASHBOARD / "config.js").read_text(encoding="utf-8")
    assert "AGRI_CONFIG" in cfg, "config.js must expose window.AGRI_CONFIG"
    assert "API_BASE" in cfg, "config.js must define an API base URL"


@pytest.mark.parametrize("name", DEMO_GEOJSON_FILES)
def test_demo_geojson_valid_featurecollection(name: str) -> None:
    path = DEMO / name
    assert path.is_file(), f"missing demo geojson: {path}"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data.get("type") == "FeatureCollection", f"{name} must be a FeatureCollection"
    feats = data.get("features")
    assert isinstance(feats, list) and len(feats) >= 1, f"{name} has no features"
    first = feats[0]
    assert first.get("type") == "Feature"
    assert "geometry" in first and "properties" in first


def test_demo_fields_have_expected_properties() -> None:
    """Fields must carry the season-series props the dashboard renders."""
    data = json.loads((DEMO / "fields.geojson").read_text(encoding="utf-8"))
    feats = data["features"]
    assert len(feats) >= 30, "expected at least ~30 demo field polygons"
    props = feats[0]["properties"]
    for key in (
        "crop",
        "stage",
        "stress_class",
        "advisory_class",
        "dr",
        "raw",
        "ndvi_series",
        "advisory_series",
    ):
        assert key in props, f"field feature missing property: {key}"
    # series arrays should be aligned and non-trivial
    n = len(props["ndvi_series"])
    assert n >= 2
    assert len(props["advisory_series"]) == n
    # advisory values must come from the 5 mandated classes
    allowed = {
        "no_irrigation",
        "watch",
        "irrigate_soon",
        "irrigate_now",
        "critical",
    }
    assert set(props["advisory_series"]).issubset(allowed)


def test_demo_rollup_json_valid() -> None:
    path = DEMO / "rollup.json"
    assert path.is_file(), f"missing demo rollup: {path}"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert "by_date" in data and isinstance(data["by_date"], list) and data["by_date"]
    row = data["by_date"][0]
    assert "advisory_pct" in row and "date" in row
