"""Tests for :mod:`agristress.catalog` — sensor registry + asset resolution.

These tests are intentionally dependency-light (stdlib only) and require no network or
cloud credentials.
"""

from __future__ import annotations

from dataclasses import fields

import pytest

from agristress.catalog import (
    STAC_ENDPOINTS,
    AccessMethod,
    Cost,
    SensorSpec,
    SensorType,
    StacEndpoint,
    by_tier,
    by_type,
    gap_fillers_for,
    gee_native,
    get_sensor,
    registry_summary,
    resolve_asset,
    stac_native,
)
from agristress.catalog.sensors import FAILURE_MODES, REQUIRED_FIELDS, SENSOR_REGISTRY

# The 12 Tier-1 sensors the operational core depends on.
TIER1_IDS = {
    "sentinel2",
    "sentinel1",
    "landsat8",
    "landsat9",
    "modis_terra",
    "viirs",
    "smap",
    "gpm_imerg",
    "chirps",
    "ecostress",
    "nisar",
    "copernicus_dem",
}


# ---------------------------------------------------------------------------
# Registry size & coverage
# ---------------------------------------------------------------------------
def test_registry_has_at_least_40_sensors():
    assert len(SENSOR_REGISTRY) >= 40


def test_registry_keys_match_spec_ids():
    for key, spec in SENSOR_REGISTRY.items():
        assert key == spec.id, f"registry key {key!r} != spec.id {spec.id!r}"


def test_no_duplicate_ids():
    ids = [s.id for s in SENSOR_REGISTRY.values()]
    assert len(ids) == len(set(ids))


def test_all_tier1_sensors_present():
    missing = TIER1_IDS - set(SENSOR_REGISTRY)
    assert not missing, f"missing Tier-1 sensors: {sorted(missing)}"
    for sid in TIER1_IDS:
        assert get_sensor(sid).tier == 1, f"{sid} is not declared tier 1"


def test_tier1_count_matches_registry():
    assert {s.id for s in by_tier(1)} == TIER1_IDS


@pytest.mark.parametrize("tier", [1, 2, 3])
def test_every_tier_nonempty(tier):
    assert by_tier(tier), f"tier {tier} is empty"


# ---------------------------------------------------------------------------
# Required fields on every spec
# ---------------------------------------------------------------------------
def test_every_spec_has_required_fields_populated():
    field_names = {f.name for f in fields(SensorSpec)}
    for required in REQUIRED_FIELDS:
        assert required in field_names
    for spec in SENSOR_REGISTRY.values():
        for fname in REQUIRED_FIELDS:
            value = getattr(spec, fname)
            assert value is not None, f"{spec.id}.{fname} is None"
            if isinstance(value, str):
                assert value.strip(), f"{spec.id}.{fname} is blank"


def test_every_spec_has_valid_enums_and_tier():
    for spec in SENSOR_REGISTRY.values():
        assert isinstance(spec.sensor_type, SensorType)
        assert isinstance(spec.cost, Cost)
        assert spec.tier in (1, 2, 3)
        # roles / fills_gaps are tuples (immutable, JSON-friendly).
        assert isinstance(spec.roles, tuple)
        assert isinstance(spec.fills_gaps, tuple)


def test_fills_gaps_use_canonical_vocabulary():
    for spec in SENSOR_REGISTRY.values():
        unknown = set(spec.fills_gaps) - set(FAILURE_MODES)
        assert not unknown, f"{spec.id} declares unknown failure modes {sorted(unknown)}"


def test_sar_specs_expose_polarizations():
    sars = by_type(SensorType.SAR)
    assert sars
    for spec in sars:
        assert spec.polarizations == spec.bands
        assert spec.polarizations, f"{spec.id} (SAR) has no polarizations"


# ---------------------------------------------------------------------------
# Native-asset queries
# ---------------------------------------------------------------------------
def test_gee_native_nonempty_and_consistent():
    gee = gee_native()
    assert gee, "no GEE-native sensors found"
    assert all(s.gee_asset_id for s in gee)
    # Core collections must be present with their real asset ids.
    assert get_sensor("sentinel2").gee_asset_id == "COPERNICUS/S2_SR_HARMONIZED"
    assert get_sensor("sentinel1").gee_asset_id == "COPERNICUS/S1_GRD"
    assert get_sensor("chirps").gee_asset_id == "UCSB-CHG/CHIRPS/DAILY"


def test_stac_native_nonempty():
    stac = stac_native()
    assert stac
    assert all(s.stac_collection for s in stac)
    assert get_sensor("ecostress").stac_collection == "eco-l2t-lste"


# ---------------------------------------------------------------------------
# Gap-filler routing — the heart of the multi-sensor fusion design
# ---------------------------------------------------------------------------
def test_gap_fillers_monsoon_cloud_includes_key_sar():
    ids = {s.id for s in gap_fillers_for("monsoon_cloud")}
    # SAR sees through monsoon cloud — Sentinel-1 / NISAR / EOS-04 must compensate.
    for expected in ("sentinel1", "nisar", "eos04"):
        assert expected in ids, f"{expected} should fill the monsoon_cloud gap"


def test_gap_fillers_coarse_soil_moisture_includes_lband():
    ids = {s.id for s in gap_fillers_for("coarse_soil_moisture")}
    assert "smap" in ids
    # L-band SAR downscales coarse passive-MW soil moisture.
    assert ids & {"nisar", "saocom", "eos04"}


def test_gap_fillers_missing_thermal_has_thermal_sources():
    fillers = gap_fillers_for("missing_thermal")
    assert fillers
    assert {s.id for s in fillers} & {"landsat8", "landsat9", "mod11a2", "ecostress"}


@pytest.mark.parametrize("mode", FAILURE_MODES)
def test_every_failure_mode_has_at_least_one_filler(mode):
    assert gap_fillers_for(mode), f"no sensor fills failure mode {mode!r}"


def test_gap_fillers_sorted_by_priority():
    # Tier-1 / FREE sensors should come before Tier-3 / COMMERCIAL ones.
    fillers = gap_fillers_for("monsoon_cloud")
    tiers = [s.tier for s in fillers]
    assert tiers == sorted(tiers), "gap_fillers_for should return cheapest/priority first"


def test_gap_fillers_for_rejects_unknown_mode():
    with pytest.raises(ValueError):
        gap_fillers_for("not_a_real_mode")


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------
def test_get_sensor_unknown_raises_keyerror():
    with pytest.raises(KeyError):
        get_sensor("does_not_exist")


def test_by_type_accepts_enum_and_str():
    by_enum = by_type(SensorType.SAR)
    by_str = by_type("SAR")
    assert by_enum == by_str
    assert all(s.sensor_type is SensorType.SAR for s in by_enum)


def test_registry_summary_structure():
    summary = registry_summary()
    assert summary["total"] == len(SENSOR_REGISTRY)
    assert sum(summary["by_tier"].values()) == len(SENSOR_REGISTRY)
    assert summary["gee_native"] == len(gee_native())
    assert set(summary["failure_modes"]) == set(FAILURE_MODES)


# ---------------------------------------------------------------------------
# Asset resolution
# ---------------------------------------------------------------------------
def test_resolve_gee_native_to_gee():
    ref = resolve_asset("sentinel2")
    assert ref.method is AccessMethod.GEE
    assert ref.gee_id == "COPERNICUS/S2_SR_HARMONIZED"


def test_resolve_stac_only_to_stac():
    ref = resolve_asset("ecostress")
    assert ref.method is AccessMethod.STAC
    assert ref.stac_collection == "eco-l2t-lste"
    assert ref.stac_endpoint == STAC_ENDPOINTS[StacEndpoint.PLANETARY_COMPUTER]


def test_resolve_nisar_routes_to_asf():
    ref = resolve_asset("nisar")
    assert ref.method is AccessMethod.STAC
    assert ref.stac_endpoint == STAC_ENDPOINTS[StacEndpoint.ASF]


def test_resolve_prefer_stac_over_gee():
    # copernicus_dem is both GEE- and STAC-native; honour the preference.
    gee_ref = resolve_asset("copernicus_dem", prefer=AccessMethod.GEE)
    stac_ref = resolve_asset("copernicus_dem", prefer=AccessMethod.STAC)
    assert gee_ref.method is AccessMethod.GEE
    assert stac_ref.method is AccessMethod.STAC


def test_resolve_commercial_sensor_is_portal_only():
    ref = resolve_asset("planetscope")
    assert ref.method is AccessMethod.PORTAL
    assert ref.gee_id is None and ref.stac_collection is None
    assert ref.portal


def test_all_stac_endpoints_are_urls():
    for url in STAC_ENDPOINTS.values():
        assert url.startswith("https://")
