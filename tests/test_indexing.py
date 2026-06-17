"""Tests for the indexing subpackage (H3 wrappers, cube→H3 table, feature store)."""

from __future__ import annotations

import numpy as np
import pytest

from agristress.fusion import build_datacube
from agristress.indexing import (
    H3_AVAILABLE,
    InMemoryFeatureStore,
    ParquetFeatureStore,
    cell_to_children,
    cell_to_latlng,
    cell_to_parent,
    cells_for_polygon,
    cube_to_h3_table,
    grid_disk,
    latlng_to_cell,
)


# ---------------------------------------------------------------------------
# H3 wrapper round-trips (work with real h3 OR the fallback)
# ---------------------------------------------------------------------------
def test_latlng_cell_roundtrip():
    lat, lng, res = 28.6139, 77.2090, 9
    cell = latlng_to_cell(lat, lng, res)
    assert isinstance(cell, str) and cell

    back_lat, back_lng = cell_to_latlng(cell)
    # Centroid of the containing cell is close to the original point.
    # res-9 H3 edge ~174 m (~0.002°); fallback step is coarser, so allow margin.
    tol = 0.01 if H3_AVAILABLE else _fallback_tol(res)
    assert abs(back_lat - lat) < tol
    assert abs(back_lng - lng) < tol


def test_cell_is_stable_and_deterministic():
    a = latlng_to_cell(12.97, 77.59, 8)
    b = latlng_to_cell(12.97, 77.59, 8)
    assert a == b
    # Nearby points fall in the same cell; far points do not.
    near = latlng_to_cell(12.9701, 77.5901, 8)
    far = latlng_to_cell(20.0, 80.0, 8)
    assert near == a
    assert far != a


def test_parent_children_relationship():
    cell = latlng_to_cell(19.07, 72.87, 9)
    parent = cell_to_parent(cell)
    assert isinstance(parent, str) and parent != cell

    children = cell_to_children(parent)
    assert isinstance(children, list) and len(children) >= 1
    if H3_AVAILABLE:
        # The original cell must be among the parent's children.
        assert cell in children


def test_grid_disk_contains_centre_and_grows():
    cell = latlng_to_cell(22.57, 88.36, 7)
    d0 = grid_disk(cell, 0)
    d1 = grid_disk(cell, 1)
    d2 = grid_disk(cell, 2)
    assert cell in d0 and cell in d1
    assert len(d1) > len(d0)
    assert len(d2) >= len(d1)


def test_cells_for_polygon_covers_interior_point():
    # Small square around (77.0, 29.0) lng/lat.
    poly = {
        "type": "Polygon",
        "coordinates": [[[76.9, 28.9], [77.1, 28.9], [77.1, 29.1], [76.9, 29.1], [76.9, 28.9]]],
    }
    cells = cells_for_polygon(poly, res=7)
    assert len(cells) >= 1
    assert len(set(cells)) == len(cells), "covering cells must be unique"
    # The cell of the polygon centroid should appear in the covering.
    centre = latlng_to_cell(29.0, 77.0, 7)
    assert centre in set(cells)


# ---------------------------------------------------------------------------
# cube_to_h3_table
# ---------------------------------------------------------------------------
def test_cube_to_h3_table_keys_unique():
    ds = build_datacube(height=12, width=12, n_times=4, seed=2)
    table = cube_to_h3_table(ds, res=7, variables=["ndvi", "soil_moisture"])

    assert list(table.columns) == ["h3_cell", "date", "variable", "value"]
    assert len(table) > 0
    # The (h3_cell, date, variable) key must be unique (O(1) lookup invariant).
    key = table[["h3_cell", "date", "variable"]]
    assert not key.duplicated().any(), "feature-table keys must be unique"
    # Only requested variables are present.
    assert set(table["variable"].unique()) <= {"ndvi", "soil_moisture"}
    # Values are finite (NaNs dropped).
    assert np.isfinite(table["value"].to_numpy()).all()


def test_cube_to_h3_table_roundtrips_into_store():
    ds = build_datacube(height=8, width=8, n_times=3, seed=4)
    table = cube_to_h3_table(ds, res=7, variables=["ndvi"])
    store = InMemoryFeatureStore.from_h3_table(table)
    assert len(store) > 0
    # Every (cell, date) in the table is retrievable.
    sample = table.iloc[0]
    feats = store.get(sample["h3_cell"], sample["date"])
    assert feats is not None and "ndvi" in feats


# ---------------------------------------------------------------------------
# Feature store: get == put
# ---------------------------------------------------------------------------
def test_inmemory_store_get_equals_put():
    store = InMemoryFeatureStore()
    feats = {"ndvi": 0.71, "soil_moisture": 0.24, "phenophase": 3}
    store.put("89283082837ffff", "2024-07-01", feats)

    got = store.get("89283082837ffff", "2024-07-01")
    assert got == feats
    # Stored copy is independent (mutating the input does not corrupt the store).
    feats["ndvi"] = -1.0
    assert store.get("89283082837ffff", "2024-07-01")["ndvi"] == 0.71
    # Miss returns None.
    assert store.get("89283082837ffff", "1999-01-01") is None
    assert ("89283082837ffff", "2024-07-01") in store


def test_inmemory_store_date_normalisation():
    store = InMemoryFeatureStore()
    store.put("cellA", "2024-07-01T00:00:00", {"x": 1})
    # ISO datetime and date forms map to the same key.
    assert store.get("cellA", "2024-07-01") == {"x": 1}


def test_parquet_store_get_equals_put_and_persists(tmp_path):
    pytest.importorskip("pyarrow")
    path = tmp_path / "features.parquet"

    store = ParquetFeatureStore(path)
    feats = {"ndvi": 0.66, "stress": "mild", "deficit_mm": 12.5}
    store.put("cellX", "2024-08-09", feats)
    assert store.get("cellX", "2024-08-09") == feats

    store.flush()
    assert path.exists()

    # Reload into a fresh instance — durability + get==put across the boundary.
    reloaded = ParquetFeatureStore(path)
    assert len(reloaded) == 1
    assert reloaded.get("cellX", "2024-08-09") == feats


def test_parquet_store_context_manager_flushes(tmp_path):
    pytest.importorskip("pyarrow")
    path = tmp_path / "ctx.parquet"
    with ParquetFeatureStore(path) as store:
        store.put("c1", "2024-06-01", {"a": 1})
    assert path.exists()
    assert ParquetFeatureStore(path).get("c1", "2024-06-01") == {"a": 1}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _fallback_tol(res: int) -> float:
    """Half-step tolerance for the fallback quantisation grid at `res`."""
    from agristress.indexing.h3_index import _fallback_step_deg

    return _fallback_step_deg(res)
