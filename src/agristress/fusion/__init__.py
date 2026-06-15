"""Fusion: spatiotemporal blending, SAR-optical gap-filling, soil-moisture, datacube.

This subpackage turns harmonised single-sensor observations into the analysis-ready
multi-sensor datacube that drives crop-type, phenology and moisture-stress models.

Public surface:

* :mod:`~agristress.fusion.spatiotemporal` — STARFM / ESTARFM image fusion.
* :mod:`~agristress.fusion.sar_optical`    — temporal gap-filling (Whittaker /
  Savitzky-Golay), SAR→NDVI translation, multi-sensor consensus.
* :mod:`~agristress.fusion.soil_moisture`  — SWI exponential filter, SMAP
  downscaling, triple-collocation error variances.
* :mod:`~agristress.fusion.datacube`       — build / read the (time, y, x) cube.

Everything runs offline on synthetic arrays (DEMO mode).
"""

from __future__ import annotations

from agristress.fusion.datacube import build_datacube, open_zarr, to_zarr
from agristress.fusion.sar_optical import (
    gap_fill_temporal,
    multi_sensor_consensus,
    sar_to_ndvi,
    savitzky_golay,
    whittaker_smooth,
)
from agristress.fusion.soil_moisture import (
    downscale_smap,
    swi_exponential_filter,
    triple_collocation,
)
from agristress.fusion.spatiotemporal import estarfm, starfm

__all__ = [
    # spatiotemporal
    "starfm",
    "estarfm",
    # sar_optical
    "gap_fill_temporal",
    "whittaker_smooth",
    "savitzky_golay",
    "sar_to_ndvi",
    "multi_sensor_consensus",
    # soil_moisture
    "swi_exponential_filter",
    "downscale_smap",
    "triple_collocation",
    # datacube
    "build_datacube",
    "to_zarr",
    "open_zarr",
]
