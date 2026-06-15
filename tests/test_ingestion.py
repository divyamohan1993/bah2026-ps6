"""Tests for :mod:`agristress.ingestion` — offline DEMO path.

All tests run with ``demo=True`` (or rely on the auto-fallback) so they pass with no
Earth Engine / STAC client and no cloud credentials. They assert the loaders return
well-shaped arrays with the right bands, value ranges and provenance.
"""

from __future__ import annotations

import datetime as _dt

import numpy as np
import pytest

from agristress.catalog.sensors import get_sensor
from agristress.ingestion import (
    SyntheticStack,
    ee_available,
    load_collection,
    load_dem,
    load_embeddings,
    load_family,
    load_items,
    load_optical_sr,
    load_precip,
    load_sar_grd,
    load_sensor,
    load_soil_moisture,
    load_thermal_et,
    search,
    search_sensor,
    stac_available,
    synth_stack_for_sensor,
)
from agristress.ingestion.synthetic import DEFAULT_BBOX, make_time_axis, normalize_bbox

AOI = (74.0, 18.4, 74.4, 18.8)
START = "2024-06-01"
END = "2024-09-01"


def _as_array(obj):
    """Return the underlying ndarray for a SyntheticStack or xarray DataArray."""
    return obj.data if isinstance(obj, SyntheticStack) else np.asarray(obj.values)


# ---------------------------------------------------------------------------
# Synthetic core
# ---------------------------------------------------------------------------
def test_synth_stack_shape_and_bands():
    spec = get_sensor("sentinel2")
    stack = synth_stack_for_sensor(spec, aoi=AOI, start=START, end=END, n_time=5, size=16)
    assert stack.shape == (5, 16, 16, len(spec.bands))
    assert stack.bands == spec.bands
    assert stack.dims == ("time", "y", "x", "band")
    assert len(stack.times) == 5
    assert stack.provenance["demo"] is True
    assert stack.provenance["sensor_id"] == "sentinel2"


def test_synth_stack_is_deterministic():
    spec = get_sensor("sentinel1")
    a = synth_stack_for_sensor(spec, aoi=AOI, start=START, end=END)
    b = synth_stack_for_sensor(spec, aoi=AOI, start=START, end=END)
    assert np.array_equal(a.data, b.data)


def test_synth_stack_no_nans():
    for sid in ("sentinel2", "sentinel1", "smap", "chirps", "ecostress", "alphaearth"):
        stack = synth_stack_for_sensor(get_sensor(sid), aoi=AOI, start=START, end=END)
        assert np.isfinite(stack.data).all(), f"{sid} produced non-finite values"


def test_normalize_bbox_variants():
    assert normalize_bbox(None) == DEFAULT_BBOX
    assert normalize_bbox([1, 2, 3, 4]) == (1.0, 2.0, 3.0, 4.0)

    class _Geom:
        bounds = (10.0, 20.0, 30.0, 40.0)

    assert normalize_bbox(_Geom()) == (10.0, 20.0, 30.0, 40.0)


def test_normalize_bbox_bad_input_raises():
    with pytest.raises(ValueError):
        normalize_bbox((1, 2, 3))


def test_make_time_axis_spans_range_inclusive():
    times = make_time_axis("2024-06-01", "2024-06-11", 6)
    assert len(times) == 6
    assert times[0] == _dt.date(2024, 6, 1)
    assert times[-1] == _dt.date(2024, 6, 11)
    assert times == sorted(times)


# ---------------------------------------------------------------------------
# to_xarray upgrade (xarray is installed in the test env)
# ---------------------------------------------------------------------------
def test_to_xarray_labels_and_coords():
    xr = pytest.importorskip("xarray")
    stack = synth_stack_for_sensor(get_sensor("sentinel2"), aoi=AOI, n_time=4, size=8)
    da = stack.to_xarray()
    assert isinstance(da, xr.DataArray)
    assert da.dims == ("time", "y", "x", "band")
    assert da.shape == (4, 8, 8, len(stack.bands))
    assert list(da.coords["band"].values) == list(stack.bands)
    assert da.sizes["time"] == 4


# ---------------------------------------------------------------------------
# Family loaders (demo)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "loader,expected_bands",
    [
        (load_optical_sr, len(get_sensor("sentinel2").bands)),
        (load_sar_grd, len(get_sensor("sentinel1").bands)),
        (load_soil_moisture, len(get_sensor("smap").bands)),
        (load_precip, len(get_sensor("chirps").bands)),
        (load_thermal_et, len(get_sensor("ecostress").bands)),
        (load_embeddings, len(get_sensor("alphaearth").bands)),
    ],
)
def test_family_loaders_demo_shapes(loader, expected_bands):
    out = loader(aoi=AOI, start=START, end=END, demo=True, n_time=4, size=8)
    arr = _as_array(out)
    assert arr.shape == (4, 8, 8, expected_bands)
    assert np.isfinite(arr).all()


def test_load_dem_is_static_single_timestep():
    out = load_dem(aoi=AOI, demo=True, size=8)
    arr = _as_array(out)
    assert arr.shape[0] == 1  # DEM has no time dimension
    assert arr.shape[1:3] == (8, 8)


def test_loaders_attach_provenance():
    out = load_optical_sr(aoi=AOI, start=START, end=END, demo=True, as_xarray=False)
    assert isinstance(out, SyntheticStack)
    assert out.provenance["family"] == "optical_sr"
    assert out.provenance["route"] == "demo"
    assert out.provenance["sensor_id"] == "sentinel2"


def test_loader_returns_xarray_when_requested():
    xr = pytest.importorskip("xarray")
    out = load_sar_grd(aoi=AOI, start=START, end=END, demo=True, as_xarray=True)
    assert isinstance(out, xr.DataArray)


def test_loader_returns_stack_when_xarray_disabled():
    out = load_sar_grd(aoi=AOI, start=START, end=END, demo=True, as_xarray=False)
    assert isinstance(out, SyntheticStack)


# ---------------------------------------------------------------------------
# Physical plausibility of demo values per family
# ---------------------------------------------------------------------------
def test_optical_values_in_reflectance_range():
    arr = _as_array(load_optical_sr(demo=True, as_xarray=False))
    assert arr.min() >= -0.2
    assert arr.max() <= 1.0


def test_sar_values_are_negative_db():
    arr = _as_array(load_sar_grd(demo=True, as_xarray=False))
    assert arr.max() <= 0.0  # backscatter in dB
    assert arr.min() >= -30.0


def test_soil_moisture_in_physical_range():
    arr = _as_array(load_soil_moisture(demo=True, as_xarray=False))
    assert arr.min() >= 0.0
    assert arr.max() <= 0.6  # volumetric soil moisture m3/m3


def test_precip_nonnegative():
    arr = _as_array(load_precip(demo=True, as_xarray=False))
    assert arr.min() >= 0.0


def test_thermal_lst_in_kelvin_range():
    arr = _as_array(load_thermal_et(demo=True, as_xarray=False))
    # ECOSTRESS bands include LST (Kelvin) and emissivity; LST band must look like K.
    assert arr.max() <= 330.0
    assert arr.min() >= 0.0


# ---------------------------------------------------------------------------
# load_sensor dispatcher
# ---------------------------------------------------------------------------
def test_load_sensor_by_id_demo():
    out = load_sensor("eos04", aoi=AOI, start=START, end=END, demo=True, as_xarray=False)
    assert isinstance(out, SyntheticStack)
    assert out.bands == get_sensor("eos04").bands


def test_load_sensor_accepts_date_objects():
    out = load_sensor(
        "sentinel2",
        aoi=AOI,
        start=_dt.date(2024, 6, 1),
        end=_dt.date(2024, 9, 1),
        demo=True,
        as_xarray=False,
    )
    assert isinstance(out, SyntheticStack)
    assert out.times[0] >= _dt.date(2024, 6, 1)


def test_load_family_unknown_raises():
    with pytest.raises(ValueError):
        load_family("not_a_family", demo=True)


def test_load_family_rejects_wrong_sensor_type():
    # Sentinel-1 (SAR) is invalid for the optical family.
    with pytest.raises(ValueError):
        load_family("optical_sr", sensor="sentinel1", demo=True)


# ---------------------------------------------------------------------------
# GEE client demo fallback
# ---------------------------------------------------------------------------
def test_load_collection_demo_returns_stack():
    out = load_collection("sentinel2", AOI, START, END, demo=True, n_time=3, size=8)
    assert isinstance(out, SyntheticStack)
    assert out.shape == (3, 8, 8, len(get_sensor("sentinel2").bands))


def test_load_collection_auto_fallback_without_ee():
    # With no earthengine-api installed, even demo=False must not raise.
    if ee_available():
        pytest.skip("earthengine-api is installed; auto-fallback path not exercised")
    out = load_collection("sentinel1", AOI, START, END, n_time=2, size=4)
    assert isinstance(out, SyntheticStack)
    assert out.shape[0] == 2


# ---------------------------------------------------------------------------
# STAC client demo fallback
# ---------------------------------------------------------------------------
def test_search_demo_returns_items():
    items = search("sentinel-2-l2a", AOI, (START, END), demo=True, limit=5)
    assert items
    assert all(it["collection"] == "sentinel-2-l2a" for it in items)
    assert all(it["demo"] for it in items)


def test_search_sensor_demo():
    items = search_sensor("ecostress", AOI, START, END, demo=True)
    assert items
    assert items[0]["collection"] == get_sensor("ecostress").stac_collection


def test_search_sensor_non_stac_raises():
    # MODIS-Terra is GEE-native only (no stac_collection) -> should error.
    with pytest.raises(ValueError):
        search_sensor("modis_terra", AOI, START, END, demo=True)


def test_load_items_from_demo_search():
    items = search("sentinel-1-grd", AOI, (START, END), demo=True, limit=4)
    out = load_items(items_from(items), bbox=AOI, sensor="sentinel1", demo=True)
    assert isinstance(out, SyntheticStack)
    assert out.bands == get_sensor("sentinel1").bands


def items_from(items):
    """Identity helper (keeps the test readable / future-proofs item adaptation)."""
    return items


# ---------------------------------------------------------------------------
# Availability flags are booleans (don't require the libs)
# ---------------------------------------------------------------------------
def test_availability_flags_are_bool():
    assert isinstance(ee_available(), bool)
    assert isinstance(stac_available(), bool)
