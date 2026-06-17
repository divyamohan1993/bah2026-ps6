"""Asset resolution for AgriStress.

Maps a :class:`~agristress.catalog.sensors.SensorSpec` to a *concrete* asset reference
the ingestion layer can act on:

* a Google Earth Engine collection id (``GEE``), or
* a STAC ``{endpoint, collection}`` pair (``STAC``).

This module is pure-stdlib and offline-safe: it only resolves *identifiers*, it never
opens a network connection. Actual data access lives in :mod:`agristress.ingestion`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from agristress.catalog.sensors import SENSOR_REGISTRY, SensorSpec, get_sensor

__all__ = [
    "STAC_ENDPOINTS",
    "AccessMethod",
    "AssetRef",
    "StacEndpoint",
    "default_stac_endpoint_for",
    "resolve_asset",
]


class AccessMethod(StrEnum):
    """How a resolved asset should be opened."""

    GEE = "GEE"
    STAC = "STAC"
    PORTAL = "PORTAL"  # no cloud-native handle; manual portal download (e.g. tasking)


class StacEndpoint(StrEnum):
    """Logical names for the STAC APIs AgriStress knows about."""

    PLANETARY_COMPUTER = "planetary_computer"
    ASF = "asf"
    BHOONIDHI = "bhoonidhi"
    EARTH_SEARCH = "earth_search"


#: Logical endpoint name -> STAC API root URL.
STAC_ENDPOINTS: dict[StacEndpoint, str] = {
    StacEndpoint.PLANETARY_COMPUTER: "https://planetarycomputer.microsoft.com/api/stac/v1",
    StacEndpoint.ASF: "https://stac.asf.alaska.edu",
    StacEndpoint.BHOONIDHI: "https://bhoonidhi.nrsc.gov.in/stac",
    StacEndpoint.EARTH_SEARCH: "https://earth-search.aws.element84.com/v1",
}

# Which STAC collections are served by which endpoint (best-effort routing). A given
# collection id may exist on more than one catalog; the first match wins.
_COLLECTION_ENDPOINT: dict[str, StacEndpoint] = {
    # Microsoft Planetary Computer
    "sentinel-2-l2a": StacEndpoint.PLANETARY_COMPUTER,
    "sentinel-1-grd": StacEndpoint.PLANETARY_COMPUTER,
    "landsat-c2-l2": StacEndpoint.PLANETARY_COMPUTER,
    "cop-dem-glo-30": StacEndpoint.PLANETARY_COMPUTER,
    "nasadem": StacEndpoint.PLANETARY_COMPUTER,
    "esa-worldcover": StacEndpoint.PLANETARY_COMPUTER,
    "eco-l2t-lste": StacEndpoint.PLANETARY_COMPUTER,
    "emit-l2a-rfl": StacEndpoint.PLANETARY_COMPUTER,
    # ASF (NASA Alaska Satellite Facility) — SAR
    "nisar-l-band": StacEndpoint.ASF,
}


@dataclass(frozen=True)
class AssetRef:
    """A concrete, resolvable reference to a sensor's data.

    Exactly one access path is populated depending on :attr:`method`:

    * ``GEE``    -> :attr:`gee_id` is set.
    * ``STAC``   -> :attr:`stac_endpoint` (URL) and :attr:`stac_collection` are set.
    * ``PORTAL`` -> neither cloud handle is set; :attr:`portal` describes manual access.
    """

    sensor_id: str
    method: AccessMethod
    gee_id: str | None = None
    stac_endpoint: str | None = None
    stac_collection: str | None = None
    portal: str = ""

    def as_dict(self) -> dict[str, str | None]:
        """JSON-friendly representation."""
        return {
            "sensor_id": self.sensor_id,
            "method": self.method.value,
            "gee_id": self.gee_id,
            "stac_endpoint": self.stac_endpoint,
            "stac_collection": self.stac_collection,
            "portal": self.portal or None,
        }


def default_stac_endpoint_for(collection: str) -> StacEndpoint:
    """Best-effort STAC endpoint for a collection id (defaults to Planetary Computer)."""
    return _COLLECTION_ENDPOINT.get(collection, StacEndpoint.PLANETARY_COMPUTER)


def resolve_asset(
    sensor: SensorSpec | str,
    *,
    prefer: AccessMethod = AccessMethod.GEE,
) -> AssetRef:
    """Resolve a sensor (spec or id) to a concrete :class:`AssetRef`.

    Parameters
    ----------
    sensor:
        A :class:`SensorSpec` or a registry id string.
    prefer:
        Which access method to try first when a sensor supports more than one. When
        the preferred method is unavailable the function falls back to the other
        cloud-native option, then to ``PORTAL``.
    """
    spec = sensor if isinstance(sensor, SensorSpec) else get_sensor(sensor)

    can_gee = spec.gee_asset_id is not None
    can_stac = spec.stac_collection is not None

    # Honour the caller's preference, then fall back to whatever is available.
    order = (
        [AccessMethod.GEE, AccessMethod.STAC]
        if prefer is AccessMethod.GEE
        else [AccessMethod.STAC, AccessMethod.GEE]
    )

    for method in order:
        if method is AccessMethod.GEE and can_gee:
            return AssetRef(
                sensor_id=spec.id,
                method=AccessMethod.GEE,
                gee_id=spec.gee_asset_id,
                portal=spec.portal,
            )
        if method is AccessMethod.STAC and can_stac:
            endpoint = default_stac_endpoint_for(spec.stac_collection)  # type: ignore[arg-type]
            return AssetRef(
                sensor_id=spec.id,
                method=AccessMethod.STAC,
                stac_endpoint=STAC_ENDPOINTS[endpoint],
                stac_collection=spec.stac_collection,
                portal=spec.portal,
            )

    # No cloud-native handle (typically Tier-3 tasking sensors): portal-only.
    return AssetRef(sensor_id=spec.id, method=AccessMethod.PORTAL, portal=spec.portal)


def _resolve_all() -> dict[str, AssetRef]:  # pragma: no cover - convenience for debugging
    """Resolve every registered sensor (used by CLIs / smoke checks)."""
    return {sid: resolve_asset(sid) for sid in SENSOR_REGISTRY}
