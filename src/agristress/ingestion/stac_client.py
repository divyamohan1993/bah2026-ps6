"""Thin STAC client wrapper with graceful offline fallback.

Wraps ``pystac-client`` for catalog search and, when available, ``odc-stac`` /
``stackstac`` for loading matched items into an :class:`xarray.DataArray`. All third-party
imports are lazy, so ``import agristress.ingestion.stac_client`` succeeds with none of
them installed; in that case (or with ``demo=True``) the helpers return deterministic
synthetic data instead.
"""

from __future__ import annotations

import datetime as _dt
from typing import TYPE_CHECKING, Any

from agristress.catalog.assets import STAC_ENDPOINTS, StacEndpoint, default_stac_endpoint_for
from agristress.catalog.sensors import SensorSpec, get_sensor
from agristress.ingestion.synthetic import (
    BBox,
    SyntheticStack,
    make_time_axis,
    normalize_bbox,
    synth_stack_for_sensor,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import xarray as xr  # noqa: F401

__all__ = [
    "StacUnavailable",
    "stac_available",
    "resolve_endpoint",
    "open_client",
    "search",
    "load_items",
    "search_sensor",
]


class StacUnavailable(RuntimeError):
    """Raised when a STAC operation needs ``pystac-client`` but it is not installed."""


def stac_available() -> bool:
    """Return ``True`` if ``pystac-client`` can be imported."""
    try:
        import pystac_client  # type: ignore # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def resolve_endpoint(endpoint: str | StacEndpoint) -> str:
    """Resolve a logical endpoint name (or :class:`StacEndpoint`) to a URL.

    A full ``http(s)://`` URL is returned unchanged so callers can point at any catalog.
    """
    if isinstance(endpoint, StacEndpoint):
        return STAC_ENDPOINTS[endpoint]
    text = str(endpoint)
    if text.startswith(("http://", "https://")):
        return text
    try:
        return STAC_ENDPOINTS[StacEndpoint(text)]
    except ValueError:
        raise ValueError(
            f"Unknown STAC endpoint {endpoint!r}; pass a URL or one of "
            f"{[e.value for e in StacEndpoint]}."
        ) from None


def open_client(endpoint: str | StacEndpoint = StacEndpoint.PLANETARY_COMPUTER) -> Any:
    """Open a ``pystac_client.Client`` against ``endpoint``.

    Raises
    ------
    StacUnavailable
        If ``pystac-client`` is not installed.
    """
    try:
        from pystac_client import Client  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise StacUnavailable(
            "pystac-client is not installed. Install it (`pip install pystac-client`) or "
            "call search()/load_items() with demo=True for the offline fallback."
        ) from exc

    url = resolve_endpoint(endpoint)
    modifier = _planetary_computer_modifier(url)
    return Client.open(url, modifier=modifier)


def _planetary_computer_modifier(url: str) -> Any | None:
    """Return the PC ``sign_inplace`` modifier when targeting Planetary Computer."""
    if "planetarycomputer" not in url:
        return None
    try:  # planetary-computer is optional; only needed to sign PC asset hrefs.
        import planetary_computer as pc  # type: ignore

        return pc.sign_inplace
    except Exception:  # noqa: BLE001 - unsigned access still works for searching
        return None


def _fmt_datetime(start: str | _dt.date | _dt.datetime, end: str | _dt.date | _dt.datetime) -> str:
    """Format an inclusive date range as the ``start/end`` string STAC expects."""

    def _iso(v: str | _dt.date | _dt.datetime) -> str:
        if isinstance(v, (_dt.date, _dt.datetime)):
            return v.strftime("%Y-%m-%d")
        return str(v)

    return f"{_iso(start)}/{_iso(end)}"


def search(
    collection: str,
    bbox: BBox | Any,
    datetime: str | tuple[Any, Any],
    *,
    endpoint: str | StacEndpoint | None = None,
    limit: int = 100,
    query: dict[str, Any] | None = None,
    demo: bool = False,
) -> list[Any]:
    """Search a STAC collection and return a list of matched items.

    On the live path (``pystac-client`` present, ``demo`` False) returns real STAC
    ``Item`` objects. On the offline path returns a list of lightweight ``dict`` items
    (``{"id", "collection", "datetime", "bbox", "demo"}``) so callers/tests still get a
    non-empty, well-shaped result.

    Parameters
    ----------
    collection:
        STAC collection id (e.g. ``"sentinel-2-l2a"``).
    bbox:
        AOI as a 4-tuple or any object with ``.bounds`` (normalised internally).
    datetime:
        Either a ``"start/end"`` string or a ``(start, end)`` tuple of date-likes.
    endpoint:
        STAC endpoint name/URL; defaults to the best endpoint for ``collection``.
    limit:
        Max items to request / synthesise.
    query:
        Optional STAC ``query`` extension filter (e.g. cloud cover) — live path only.
    demo:
        Force the offline path.
    """
    box = normalize_bbox(bbox)
    dt_str = datetime if isinstance(datetime, str) else _fmt_datetime(datetime[0], datetime[1])

    if demo or not stac_available():
        return _synthetic_items(collection, box, dt_str, limit=limit)

    if endpoint is None:
        endpoint = default_stac_endpoint_for(collection)
    client = open_client(endpoint)
    result = client.search(
        collections=[collection],
        bbox=list(box),
        datetime=dt_str,
        limit=limit,
        query=query,
    )
    return list(result.items())


def _synthetic_items(collection: str, box: BBox, dt_str: str, *, limit: int) -> list[dict[str, Any]]:
    """Build a handful of placeholder STAC item dicts spanning the date range."""
    start_s, _, end_s = dt_str.partition("/")
    n = max(1, min(limit, 6))
    times = make_time_axis(start_s or "2024-06-01", end_s or start_s or "2024-10-01", n)
    return [
        {
            "id": f"{collection}-demo-{i:03d}",
            "collection": collection,
            "datetime": t.isoformat(),
            "bbox": list(box),
            "demo": True,
            "assets": {},
        }
        for i, t in enumerate(times)
    ]


def load_items(
    items: list[Any],
    *,
    bands: list[str] | None = None,
    bbox: BBox | Any = None,
    resolution: float | None = None,
    sensor: SensorSpec | str | None = None,
    demo: bool = False,
) -> "xr.DataArray | SyntheticStack":
    """Open matched STAC items into an :class:`xarray.DataArray` (or synthetic fallback).

    Tries ``odc.stac.load`` first, then ``stackstac.stack``. If neither library is
    available, or ``demo`` is True, or ``items`` are the synthetic dicts from
    :func:`search`, returns a :class:`~agristress.ingestion.synthetic.SyntheticStack`
    (a ``sensor`` is required in that case to pick bands/value ranges).
    """
    synthetic_input = bool(items) and isinstance(items[0], dict) and items[0].get("demo")

    if demo or synthetic_input or not items:
        spec = _coerce_sensor(sensor, items)
        return synth_stack_for_sensor(spec, aoi=bbox, source="stac_demo")

    # ---- odc-stac -------------------------------------------------------
    try:
        import odc.stac as odc_stac  # type: ignore

        ds = odc_stac.load(
            items,
            bands=bands,
            bbox=list(normalize_bbox(bbox)) if bbox is not None else None,
            resolution=resolution,
            chunks={},
        )
        return ds
    except Exception:  # noqa: BLE001 - fall through to stackstac
        pass

    # ---- stackstac ------------------------------------------------------
    try:
        import stackstac  # type: ignore

        return stackstac.stack(
            items,
            assets=bands,
            bounds_latlon=tuple(normalize_bbox(bbox)) if bbox is not None else None,
            resolution=resolution,
        )
    except Exception as exc:  # noqa: BLE001
        # Last resort: synthetic, so the pipeline never hard-fails offline.
        spec = _coerce_sensor(sensor, items)
        stack = synth_stack_for_sensor(spec, aoi=bbox, source="stac_demo_fallback")
        stack.provenance["fallback_reason"] = f"{type(exc).__name__}: {exc}"
        return stack


def _coerce_sensor(sensor: SensorSpec | str | None, items: list[Any]) -> SensorSpec:
    """Resolve a SensorSpec from an explicit arg or by matching item collection ids."""
    if isinstance(sensor, SensorSpec):
        return sensor
    if isinstance(sensor, str):
        return get_sensor(sensor)
    # Try to map the item's collection id back to a registered sensor.
    if items:
        coll = items[0].get("collection") if isinstance(items[0], dict) else getattr(
            items[0], "collection_id", None
        )
        if coll:
            from agristress.catalog.sensors import SENSOR_REGISTRY

            for spec in SENSOR_REGISTRY.values():
                if spec.stac_collection == coll:
                    return spec
    raise ValueError(
        "load_items() needs a `sensor` (SensorSpec or id) to build the offline fallback "
        "when the source collection cannot be matched to the registry."
    )


def search_sensor(
    sensor: SensorSpec | str,
    bbox: BBox | Any,
    start: str | _dt.date | _dt.datetime,
    end: str | _dt.date | _dt.datetime,
    *,
    demo: bool = False,
    limit: int = 100,
) -> list[Any]:
    """Convenience: search by *sensor* (must be STAC-native) instead of a raw collection."""
    spec = sensor if isinstance(sensor, SensorSpec) else get_sensor(sensor)
    if not spec.stac_collection:
        raise ValueError(
            f"Sensor {spec.id!r} has no stac_collection; it is not STAC-native. "
            f"Use the GEE client, or call with demo=True."
        )
    return search(
        spec.stac_collection,
        bbox,
        (start, end),
        endpoint=default_stac_endpoint_for(spec.stac_collection),
        limit=limit,
        demo=demo,
    )
