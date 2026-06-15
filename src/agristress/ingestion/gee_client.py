"""Thin Google Earth Engine client wrapper with an offline DEMO fallback.

Earth Engine is an *optional* runtime dependency. This module imports it lazily so that
``import agristress.ingestion.gee_client`` always succeeds, even with no
``earthengine-api`` installed and no cloud credentials. When EE is unavailable (or the
caller passes ``demo=True``), the loaders fall back to deterministic synthetic stacks so
that the rest of the pipeline — and the test suite — runs fully offline.
"""

from __future__ import annotations

import datetime as _dt
from typing import TYPE_CHECKING, Any

from agristress.catalog.sensors import SensorSpec, get_sensor
from agristress.ingestion.synthetic import (
    BBox,
    SyntheticStack,
    normalize_bbox,
    synth_stack_for_sensor,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import ee  # noqa: F401

__all__ = [
    "EarthEngineUnavailable",
    "ee_available",
    "init_ee",
    "load_collection",
]


class EarthEngineUnavailable(RuntimeError):
    """Raised when an Earth Engine operation is requested but EE cannot be used."""


# Module-level cache of the imported ``ee`` module + init flag, to avoid re-importing.
_EE_MODULE: Any | None = None
_EE_INITIALIZED: bool = False


def _try_import_ee() -> Any | None:
    """Import ``earthengine-api`` lazily; return the module or ``None`` if absent."""
    global _EE_MODULE
    if _EE_MODULE is not None:
        return _EE_MODULE
    try:
        import ee  # type: ignore
    except Exception:  # noqa: BLE001 - any import error means EE is unusable here
        return None
    _EE_MODULE = ee
    return ee


def ee_available() -> bool:
    """Return ``True`` if the ``earthengine-api`` package can be imported."""
    return _try_import_ee() is not None


def init_ee(project: str | None = None, *, high_volume: bool = False) -> Any:
    """Initialise Earth Engine and return the ``ee`` module.

    Parameters
    ----------
    project:
        GCP project id to bill EE quota against (``ee.Initialize(project=...)``).
    high_volume:
        Route through the high-volume endpoint (batch/tile workloads).

    Raises
    ------
    EarthEngineUnavailable
        If ``earthengine-api`` is not installed, or initialisation fails (e.g. no
        credentials). The message explains how to proceed (authenticate, or use
        ``demo=True``).
    """
    global _EE_INITIALIZED
    ee = _try_import_ee()
    if ee is None:
        raise EarthEngineUnavailable(
            "earthengine-api is not installed. Install it (`pip install earthengine-api`) "
            "and run `earthengine authenticate`, or call the loaders with demo=True to use "
            "the offline synthetic fallback."
        )
    if _EE_INITIALIZED:
        return ee
    try:
        kwargs: dict[str, Any] = {}
        if project:
            kwargs["project"] = project
        if high_volume:
            # The high-volume endpoint is opt-in; guard for older ee versions.
            opt_url = getattr(ee.data, "HIGH_VOLUME_ENDPOINT", None)
            if opt_url:
                kwargs["opt_url"] = opt_url
        ee.Initialize(**kwargs)
        _EE_INITIALIZED = True
    except Exception as exc:  # noqa: BLE001 - surface a single clear error
        raise EarthEngineUnavailable(
            "Earth Engine failed to initialise (missing/invalid credentials?). Run "
            "`earthengine authenticate` or pass demo=True to use the offline fallback. "
            f"Underlying error: {exc}"
        ) from exc
    return ee


def _to_ee_date(value: str | _dt.date | _dt.datetime) -> str:
    """Coerce a date-like value to an ISO ``YYYY-MM-DD`` string EE accepts."""
    if isinstance(value, (_dt.date, _dt.datetime)):
        return value.strftime("%Y-%m-%d")
    return str(value)


def load_collection(
    sensor: SensorSpec | str,
    aoi: BBox | Any,
    start: str | _dt.date | _dt.datetime,
    end: str | _dt.date | _dt.datetime,
    *,
    demo: bool = False,
    project: str | None = None,
    n_time: int = 6,
    size: int = 32,
) -> "Any | SyntheticStack":
    """Load a sensor's collection over an AOI/date range.

    When Earth Engine is available and ``demo`` is ``False`` this returns a real,
    AOI/date-filtered ``ee.ImageCollection``. Otherwise it returns a deterministic
    :class:`~agristress.ingestion.synthetic.SyntheticStack` of shape
    ``(n_time, size, size, n_bands)`` so callers work offline.

    Parameters
    ----------
    sensor:
        A :class:`SensorSpec` or registry id. Must expose a ``gee_asset_id`` for the
        live path (otherwise a ``ValueError`` is raised unless ``demo`` is requested).
    aoi:
        Bounding box ``(min_lon, min_lat, max_lon, max_lat)`` (or any object with a
        ``.bounds`` 4-tuple, e.g. a shapely geometry) for the synthetic/live filter.
        An ``ee.Geometry`` is also accepted on the live path.
    start, end:
        Inclusive date range as ISO strings or ``date``/``datetime`` objects.
    demo:
        Force the offline synthetic path.
    project:
        GCP project for ``init_ee`` (live path only).
    n_time, size:
        Shape controls for the synthetic stack (ignored on the live path).
    """
    spec = sensor if isinstance(sensor, SensorSpec) else get_sensor(sensor)

    # ---- offline / synthetic path -----------------------------------------
    if demo or not ee_available():
        return synth_stack_for_sensor(
            spec,
            aoi=aoi,
            start=_to_ee_date(start),
            end=_to_ee_date(end),
            n_time=n_time,
            size=size,
            source="gee_demo",
        )

    # ---- live Earth Engine path -------------------------------------------
    if not spec.gee_asset_id:
        raise ValueError(
            f"Sensor {spec.id!r} has no gee_asset_id; it is not GEE-native. "
            f"Use the STAC client, or call with demo=True."
        )
    ee = init_ee(project=project)

    geom = aoi
    if not _is_ee_geometry(ee, aoi):
        min_lon, min_lat, max_lon, max_lat = normalize_bbox(aoi)
        geom = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])

    collection = (
        ee.ImageCollection(spec.gee_asset_id)
        .filterBounds(geom)
        .filterDate(_to_ee_date(start), _to_ee_date(end))
    )
    return collection


def _is_ee_geometry(ee: Any, obj: Any) -> bool:
    """Best-effort check that ``obj`` is already an ``ee.Geometry``/``ee.Feature``."""
    for attr in ("Geometry", "Feature", "FeatureCollection"):
        cls = getattr(ee, attr, None)
        if cls is not None and isinstance(obj, cls):
            return True
    return False
