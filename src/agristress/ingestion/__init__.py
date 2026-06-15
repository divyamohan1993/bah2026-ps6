"""AgriStress ingestion layer.

Cloud-data access for the sensors described in :mod:`agristress.catalog`, with a
first-class **offline DEMO path**: every entry point accepts ``demo=True`` (and auto-falls
back when no Earth Engine / STAC client or credentials are present) so the pipeline and
test suite run with zero cloud setup.

Public API
----------
* Earth Engine: :func:`init_ee`, :func:`load_collection`, :func:`ee_available`.
* STAC: :func:`search`, :func:`load_items`, :func:`search_sensor`, :func:`stac_available`.
* Family loaders: :func:`load_optical_sr`, :func:`load_sar_grd`, :func:`load_soil_moisture`,
  :func:`load_precip`, :func:`load_thermal_et`, :func:`load_dem`, :func:`load_embeddings`,
  plus the dispatchers :func:`load_family` / :func:`load_sensor`.
* Synthetic core: :class:`SyntheticStack`, :func:`synth_stack_for_sensor`.
"""

from __future__ import annotations

from agristress.ingestion.gee_client import (
    EarthEngineUnavailable,
    ee_available,
    init_ee,
    load_collection,
)
from agristress.ingestion.loaders import (
    DEFAULT_SENSORS,
    load_dem,
    load_embeddings,
    load_family,
    load_optical_sr,
    load_precip,
    load_sar_grd,
    load_sensor,
    load_soil_moisture,
    load_thermal_et,
)
from agristress.ingestion.stac_client import (
    StacUnavailable,
    load_items,
    search,
    search_sensor,
    stac_available,
)
from agristress.ingestion.synthetic import (
    BBox,
    SyntheticStack,
    normalize_bbox,
    synth_stack_for_sensor,
)

__all__ = [
    # gee
    "init_ee",
    "load_collection",
    "ee_available",
    "EarthEngineUnavailable",
    # stac
    "search",
    "load_items",
    "search_sensor",
    "stac_available",
    "StacUnavailable",
    # loaders
    "load_sensor",
    "load_family",
    "load_optical_sr",
    "load_sar_grd",
    "load_soil_moisture",
    "load_precip",
    "load_thermal_et",
    "load_dem",
    "load_embeddings",
    "DEFAULT_SENSORS",
    # synthetic
    "SyntheticStack",
    "synth_stack_for_sensor",
    "normalize_bbox",
    "BBox",
]
