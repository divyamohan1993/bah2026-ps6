"""Per-family analysis-ready loaders for AgriStress.

Each loader hides *how* a sensor family is fetched (Earth Engine vs STAC) behind one
``demo``-aware call that returns a standardized object:

* a :class:`xarray.DataArray` when xarray is installed, or
* a :class:`~agristress.ingestion.synthetic.SyntheticStack` otherwise,

always carrying ``provenance`` metadata (source sensor, family, bbox, time range, units).
With ``demo=True`` (or when no cloud client/credentials are present) every loader
produces deterministic synthetic data so the pipeline and tests run fully offline.

Families
--------
``optical_sr`` · ``sar_grd`` · ``soil_moisture`` · ``precip`` · ``thermal_et`` ·
``dem`` · ``embeddings`` — plus the generic dispatcher :func:`load_family` and the
sensor-driven :func:`load_sensor`.
"""

from __future__ import annotations

import datetime as _dt
from typing import TYPE_CHECKING, Any

from agristress.catalog.sensors import SENSOR_REGISTRY, SensorSpec, SensorType, get_sensor
from agristress.ingestion.gee_client import ee_available, load_collection
from agristress.ingestion.stac_client import load_items, search_sensor, stac_available
from agristress.ingestion.synthetic import BBox, SyntheticStack, synth_stack_for_sensor

if TYPE_CHECKING:  # pragma: no cover - typing only
    import xarray as xr  # noqa: F401

__all__ = [
    "DEFAULT_SENSORS",
    "LoadResult",
    "load_dem",
    "load_embeddings",
    "load_family",
    "load_optical_sr",
    "load_precip",
    "load_sar_grd",
    "load_sensor",
    "load_soil_moisture",
    "load_thermal_et",
]

#: A loaded family's data is one of these (labelled cube, or lightweight fallback).
LoadResult = "xr.DataArray | SyntheticStack"

#: Default representative sensor per family (used when the caller doesn't pick one).
DEFAULT_SENSORS: dict[str, str] = {
    "optical_sr": "sentinel2",
    "sar_grd": "sentinel1",
    "soil_moisture": "smap",
    "precip": "chirps",
    "thermal_et": "ecostress",
    "dem": "copernicus_dem",
    "embeddings": "alphaearth",
}

# Which SensorType(s) each family accepts, for validation.
_FAMILY_TYPES: dict[str, tuple[SensorType, ...]] = {
    "optical_sr": (SensorType.OPTICAL, SensorType.HYPERSPECTRAL),
    "sar_grd": (SensorType.SAR,),
    "soil_moisture": (SensorType.RADIOMETER, SensorType.SAR),
    "precip": (SensorType.PRECIP,),
    "thermal_et": (SensorType.THERMAL,),
    "dem": (SensorType.DEM,),
    "embeddings": (SensorType.ANCILLARY,),
}


def _default_dates() -> tuple[str, str]:
    """A kharif-season-ish default window (used when caller omits start/end)."""
    return ("2024-06-01", "2024-10-01")


def _provenance_for(spec: SensorSpec, family: str, route: str, demo: bool) -> dict[str, Any]:
    return {
        "sensor_id": spec.id,
        "sensor_name": spec.name,
        "family": family,
        "sensor_type": spec.sensor_type.value,
        "route": route,  # "gee" | "stac" | "demo"
        "demo": demo,
        "gee_asset_id": spec.gee_asset_id,
        "stac_collection": spec.stac_collection,
        "native_resolution_m": spec.native_resolution_m,
    }


def _attach_provenance(result: Any, provenance: dict[str, Any]) -> Any:
    """Merge provenance into the result's metadata, whatever its concrete type."""
    if isinstance(result, SyntheticStack):
        result.provenance.update(provenance)
        return result
    # xarray DataArray / Dataset both have a mutable ``.attrs`` mapping.
    attrs = getattr(result, "attrs", None)
    if isinstance(attrs, dict):
        attrs.update(provenance)
    return result


def load_sensor(
    sensor: SensorSpec | str,
    aoi: BBox | Any = None,
    start: str | _dt.date | _dt.datetime | None = None,
    end: str | _dt.date | _dt.datetime | None = None,
    *,
    demo: bool = False,
    prefer_stac: bool = False,
    n_time: int = 6,
    size: int = 32,
    as_xarray: bool = True,
    project: str | None = None,
) -> LoadResult:
    """Load a single sensor over an AOI/date range as a standardized cube.

    Routing:

    * ``demo=True`` or no cloud client available  -> synthetic stack.
    * GEE-native sensor (and not ``prefer_stac``)  -> :func:`gee_client.load_collection`.
    * STAC-native sensor (or ``prefer_stac``)      -> search + :func:`stac_client.load_items`.
    * otherwise                                    -> synthetic stack (portal-only sensor).

    The result is upgraded to an :class:`xarray.DataArray` when ``as_xarray`` is True and
    xarray is installed; otherwise a :class:`SyntheticStack` is returned. ``provenance``
    metadata is always attached.
    """
    spec = sensor if isinstance(sensor, SensorSpec) else get_sensor(sensor)
    s, e = (start, end)
    if s is None or e is None:
        ds, de = _default_dates()
        s = s if s is not None else ds
        e = e if e is not None else de

    use_demo = demo or not (ee_available() or stac_available())

    # ---- offline / synthetic --------------------------------------------
    if use_demo:
        stack = synth_stack_for_sensor(
            spec,
            aoi=aoi,
            start=str(_iso(s)),
            end=str(_iso(e)),
            n_time=n_time,
            size=size,
            source="loader_demo",
        )
        stack.provenance.update(_provenance_for(spec, _family_of(spec), "demo", True))
        return _maybe_xarray(stack, as_xarray)

    # ---- live: STAC route -----------------------------------------------
    if (prefer_stac or not spec.gee_asset_id) and spec.stac_collection:
        items = search_sensor(spec, aoi, s, e, demo=False)
        result = load_items(items, bbox=aoi, sensor=spec, demo=False)
        return _attach_provenance(result, _provenance_for(spec, _family_of(spec), "stac", False))

    # ---- live: GEE route -------------------------------------------------
    if spec.gee_asset_id:
        result = load_collection(
            spec, aoi, s, e, demo=False, project=project, n_time=n_time, size=size
        )
        # A live ee.ImageCollection has no ``.attrs``; wrap provenance only if possible.
        return _attach_provenance(result, _provenance_for(spec, _family_of(spec), "gee", False))

    # ---- portal-only sensor: synthesise ---------------------------------
    stack = synth_stack_for_sensor(
        spec,
        aoi=aoi,
        start=str(_iso(s)),
        end=str(_iso(e)),
        n_time=n_time,
        size=size,
        source="loader_portal_demo",
    )
    stack.provenance.update(_provenance_for(spec, _family_of(spec), "demo", True))
    return _maybe_xarray(stack, as_xarray)


def load_family(
    family: str,
    aoi: BBox | Any = None,
    start: str | _dt.date | _dt.datetime | None = None,
    end: str | _dt.date | _dt.datetime | None = None,
    *,
    sensor: SensorSpec | str | None = None,
    demo: bool = False,
    **kwargs: Any,
) -> LoadResult:
    """Load a family's default (or chosen) sensor. See :func:`load_sensor` for kwargs."""
    if family not in DEFAULT_SENSORS:
        raise ValueError(
            f"Unknown family {family!r}; valid families are {sorted(DEFAULT_SENSORS)}."
        )
    spec = _resolve_family_sensor(family, sensor)
    return load_sensor(spec, aoi, start, end, demo=demo, **kwargs)


# ---------------------------------------------------------------------------
# Thin family-named convenience wrappers (stable public API)
# ---------------------------------------------------------------------------
def load_optical_sr(
    aoi: BBox | Any = None,
    start: Any = None,
    end: Any = None,
    *,
    sensor: SensorSpec | str | None = None,
    demo: bool = False,
    **kwargs: Any,
) -> LoadResult:
    """Optical surface-reflectance family (Sentinel-2 default)."""
    return load_family("optical_sr", aoi, start, end, sensor=sensor, demo=demo, **kwargs)


def load_sar_grd(
    aoi: BBox | Any = None,
    start: Any = None,
    end: Any = None,
    *,
    sensor: SensorSpec | str | None = None,
    demo: bool = False,
    **kwargs: Any,
) -> LoadResult:
    """SAR ground-range-detected family (Sentinel-1 default)."""
    return load_family("sar_grd", aoi, start, end, sensor=sensor, demo=demo, **kwargs)


def load_soil_moisture(
    aoi: BBox | Any = None,
    start: Any = None,
    end: Any = None,
    *,
    sensor: SensorSpec | str | None = None,
    demo: bool = False,
    **kwargs: Any,
) -> LoadResult:
    """Soil-moisture family (SMAP default; also accepts L-band SAR)."""
    return load_family("soil_moisture", aoi, start, end, sensor=sensor, demo=demo, **kwargs)


def load_precip(
    aoi: BBox | Any = None,
    start: Any = None,
    end: Any = None,
    *,
    sensor: SensorSpec | str | None = None,
    demo: bool = False,
    **kwargs: Any,
) -> LoadResult:
    """Precipitation family (CHIRPS default)."""
    return load_family("precip", aoi, start, end, sensor=sensor, demo=demo, **kwargs)


def load_thermal_et(
    aoi: BBox | Any = None,
    start: Any = None,
    end: Any = None,
    *,
    sensor: SensorSpec | str | None = None,
    demo: bool = False,
    **kwargs: Any,
) -> LoadResult:
    """Thermal / evapotranspiration family (ECOSTRESS default)."""
    return load_family("thermal_et", aoi, start, end, sensor=sensor, demo=demo, **kwargs)


def load_dem(
    aoi: BBox | Any = None,
    *,
    sensor: SensorSpec | str | None = None,
    demo: bool = False,
    **kwargs: Any,
) -> LoadResult:
    """Digital-elevation family (Copernicus DEM GLO-30 default). Static: no date range."""
    kwargs.setdefault("n_time", 1)
    return load_family("dem", aoi, None, None, sensor=sensor, demo=demo, **kwargs)


def load_embeddings(
    aoi: BBox | Any = None,
    start: Any = None,
    end: Any = None,
    *,
    sensor: SensorSpec | str | None = None,
    demo: bool = False,
    **kwargs: Any,
) -> LoadResult:
    """Learned satellite-embedding family (AlphaEarth annual default)."""
    return load_family("embeddings", aoi, start, end, sensor=sensor, demo=demo, **kwargs)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _family_of(spec: SensorSpec) -> str:
    """Map a sensor back to its loader family name (best match by type)."""
    for fam, default_id in DEFAULT_SENSORS.items():
        if spec.id == default_id:
            return fam
    for fam, types in _FAMILY_TYPES.items():
        if spec.sensor_type in types:
            return fam
    return "optical_sr"


def _resolve_family_sensor(family: str, sensor: SensorSpec | str | None) -> SensorSpec:
    """Pick + validate the sensor used for a family."""
    if sensor is None:
        return get_sensor(DEFAULT_SENSORS[family])
    spec = sensor if isinstance(sensor, SensorSpec) else get_sensor(sensor)
    allowed = _FAMILY_TYPES[family]
    if spec.sensor_type not in allowed:
        raise ValueError(
            f"Sensor {spec.id!r} ({spec.sensor_type.value}) is not valid for family "
            f"{family!r}; expected one of {[t.value for t in allowed]}."
        )
    return spec


def _iso(value: Any) -> str:
    if isinstance(value, (_dt.date, _dt.datetime)):
        return value.strftime("%Y-%m-%d")
    return str(value)


def _maybe_xarray(stack: SyntheticStack, as_xarray: bool) -> LoadResult:
    """Upgrade a SyntheticStack to xarray when requested and available."""
    if not as_xarray:
        return stack
    try:
        return stack.to_xarray()
    except ImportError:
        return stack


# Sanity: every default sensor must exist in the registry (import-time guard).
_missing = [sid for sid in DEFAULT_SENSORS.values() if sid not in SENSOR_REGISTRY]
if _missing:  # pragma: no cover - guards against registry drift
    raise RuntimeError(f"DEFAULT_SENSORS references unknown sensor ids: {_missing}")
