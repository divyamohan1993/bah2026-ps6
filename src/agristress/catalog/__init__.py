"""AgriStress data catalog.

A dependency-light, offline-safe description of the Earth-observation sensors the
pipeline can fuse, plus helpers to resolve each sensor to a concrete cloud asset.

Public API
----------
* :class:`SensorSpec`, :class:`SensorType`, :class:`Cost` — typed sensor model.
* :data:`SENSOR_REGISTRY` — id -> :class:`SensorSpec` for 40+ sensors.
* Query helpers: :func:`get_sensor`, :func:`by_tier`, :func:`by_type`,
  :func:`gee_native`, :func:`stac_native`, :func:`gap_fillers_for`.
* Asset resolution: :class:`AssetRef`, :func:`resolve_asset`, :data:`STAC_ENDPOINTS`.
"""

from __future__ import annotations

from agristress.catalog.assets import (
    AccessMethod,
    AssetRef,
    STAC_ENDPOINTS,
    StacEndpoint,
    default_stac_endpoint_for,
    resolve_asset,
)
from agristress.catalog.sensors import (
    FAILURE_MODES,
    SENSOR_REGISTRY,
    Cost,
    SensorSpec,
    SensorType,
    by_tier,
    by_type,
    gap_fillers_for,
    gee_native,
    get_sensor,
    registry_summary,
    stac_native,
)

__all__ = [
    # sensors
    "SensorSpec",
    "SensorType",
    "Cost",
    "SENSOR_REGISTRY",
    "FAILURE_MODES",
    "get_sensor",
    "by_tier",
    "by_type",
    "gee_native",
    "stac_native",
    "gap_fillers_for",
    "registry_summary",
    # assets
    "AccessMethod",
    "StacEndpoint",
    "STAC_ENDPOINTS",
    "AssetRef",
    "resolve_asset",
    "default_stac_endpoint_for",
]
