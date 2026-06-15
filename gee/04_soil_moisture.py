"""AgriStress · GEE step 04 — soil moisture, rainfall & ET0 forcing on Earth Engine.

Assembles the hydrological / meteorological layers that drive the moisture-stress
and irrigation-advisory heads (DATA_FUSION Stage 4 + met forcing for Stage-3
head), all on Google Earth Engine.

Layers
------
* **Soil moisture** — SMAP L4 Global ``NASA/SMAP/SPL4SMGP/008`` surface &
  root-zone soil moisture (3-hourly, 9 km), composited to 8-day means.
* **Active–passive downscaling sketch** — disaggregate the coarse SMAP field to
  Sentinel-1 resolution using the σ⁰_VV–soil-moisture sensitivity (a linear
  ``β``-model stand-in for SPL2SMAP_S / Das 2019, DATA_FUSION §4.1). This is a
  *sketch*: the high-res spatial pattern comes from S1, the magnitude/mean from
  SMAP.
* **Rainfall** — GPM IMERG ``NASA/GPM_L3/IMERG_V07`` (half-hourly) and/or CHIRPS
  ``UCSB-CHG/CHIRPS/DAILY``, accumulated to 8-day totals (mm).
* **ET0 forcing** — ERA5-Land ``ECMWF/ERA5_LAND/HOURLY`` fields (2 m temperature,
  dewpoint, wind, net radiation, surface pressure) reduced to 8-day means — the
  raw inputs the FAO-56 Penman-Monteith ET0 in step 06 consumes.

Import-safe without credentials. Run :func:`main` after ``init_ee``.
"""

from __future__ import annotations

from typing import Any

from gee._auth import EarthEngineUnavailable, init_ee

SMAP_L4_COLLECTION = "NASA/SMAP/SPL4SMGP/008"
IMERG_COLLECTION = "NASA/GPM_L3/IMERG_V07"
CHIRPS_COLLECTION = "UCSB-CHG/CHIRPS/DAILY"
ERA5_LAND_COLLECTION = "ECMWF/ERA5_LAND/HOURLY"

# ERA5-Land bands needed for FAO-56 ET0 (consumed by step 06).
ERA5_ET0_BANDS = [
    "temperature_2m",
    "dewpoint_temperature_2m",
    "u_component_of_wind_10m",
    "v_component_of_wind_10m",
    "surface_net_solar_radiation",
    "surface_pressure",
]


def _aoi_geometry(ee: Any, aoi: Any) -> Any:
    if isinstance(aoi, (list, tuple)) and len(aoi) == 4:
        return ee.Geometry.Rectangle(list(aoi))
    if isinstance(aoi, dict):
        return ee.Geometry(aoi)
    return aoi


def _eight_day_reduce(ee: Any, collection: Any, start: str, end: str, reducer: Any, prefix: str) -> Any:
    """Reduce a collection to one image per 8-day window with the given reducer."""
    start_d = ee.Date(start)
    n_days = ee.Date(end).difference(start_d, "day")
    n_steps = n_days.divide(8).ceil()
    steps = ee.List.sequence(0, n_steps.subtract(1))

    def _step(i: Any) -> Any:
        w_start = start_d.advance(ee.Number(i).multiply(8), "day")
        w_end = w_start.advance(8, "day")
        window = collection.filterDate(w_start, w_end)
        return window.reduce(reducer).set(
            {"system:time_start": w_start.millis(), "n_images": window.size(), "layer": prefix}
        )

    out = ee.ImageCollection.fromImages(steps.map(_step))
    return out.filter(ee.Filter.gt("n_images", 0))


def smap_soil_moisture(ee: Any, aoi: Any, start: str, end: str) -> Any:
    """8-day mean SMAP L4 surface + root-zone soil moisture over the AOI."""
    geom = _aoi_geometry(ee, aoi)
    smap = (
        ee.ImageCollection(SMAP_L4_COLLECTION)
        .filterBounds(geom)
        .filterDate(start, end)
        .select(["sm_surface", "sm_rootzone"])
    )
    return _eight_day_reduce(ee, smap, start, end, ee.Reducer.mean(), "SMAP")


def downscale_smap_with_s1(ee: Any, smap_img: Any, s1_img: Any, aoi: Any, scale: int = 100) -> Any:
    """Downscale a coarse SMAP soil-moisture image to S1 resolution (β-model sketch).

    Stand-in for SPL2SMAP_S (DATA_FUSION §4.1): the **mean** soil-moisture level
    is taken from SMAP (preserves the radiometer's calibrated magnitude) and the
    **spatial texture** from the S1 VV backscatter anomaly (sensitive to
    dielectric/moisture). Concretely::

        sm_high = sm_smap_mean + β · (VV − VV_mean)

    with β a small positive sensitivity (dB→volumetric). Real SPL2SMAP_S solves
    β per-pixel against the brightness temperature; this linear sketch gives a
    physically-motivated high-resolution pattern for the demo.
    """
    geom = _aoi_geometry(ee, aoi)
    sm_surface = smap_img.select("sm_surface")
    vv = s1_img.select("VV")

    # Coarse references: SMAP supplies the calibrated SM magnitude (kept at its
    # native resolution); S1 supplies the fine spatial texture relative to its
    # local mean backscatter.
    sm_mean = sm_surface  # SMAP magnitude carried as the level term
    vv_mean = vv.reduceNeighborhood(ee.Reducer.mean(), ee.Kernel.square(500, "meters"))

    beta = 0.02  # volumetric-SM per dB sensitivity (illustrative)
    anomaly = vv.subtract(vv_mean).multiply(beta)
    sm_high = sm_mean.add(anomaly).rename("sm_surface_downscaled").clamp(0.0, 0.6)
    return sm_high.clip(geom).set("scale_m", scale)


def imerg_rainfall(ee: Any, aoi: Any, start: str, end: str) -> Any:
    """8-day accumulated rainfall (mm) from GPM IMERG V07.

    IMERG ``precipitation`` is mm/hr at half-hourly steps; sum over each window
    and multiply by 0.5 h to get mm per 8-day window.
    """
    geom = _aoi_geometry(ee, aoi)
    imerg = (
        ee.ImageCollection(IMERG_COLLECTION)
        .filterBounds(geom)
        .filterDate(start, end)
        .select("precipitation")
    )
    # 0.5 h per half-hourly slice → mm.
    acc = _eight_day_reduce(ee, imerg, start, end, ee.Reducer.sum(), "IMERG")
    return acc.map(lambda img: img.multiply(0.5).rename("rain_mm").copyProperties(img, ["system:time_start"]))


def chirps_rainfall(ee: Any, aoi: Any, start: str, end: str) -> Any:
    """8-day accumulated rainfall (mm) from CHIRPS daily (already mm/day)."""
    geom = _aoi_geometry(ee, aoi)
    chirps = (
        ee.ImageCollection(CHIRPS_COLLECTION)
        .filterBounds(geom)
        .filterDate(start, end)
        .select("precipitation")
    )
    acc = _eight_day_reduce(ee, chirps, start, end, ee.Reducer.sum(), "CHIRPS")
    return acc.map(lambda img: img.rename("rain_mm").copyProperties(img, ["system:time_start"]))


def era5_et0_inputs(ee: Any, aoi: Any, start: str, end: str) -> Any:
    """8-day mean ERA5-Land fields required for FAO-56 ET0 (consumed by step 06)."""
    geom = _aoi_geometry(ee, aoi)
    era5 = (
        ee.ImageCollection(ERA5_LAND_COLLECTION)
        .filterBounds(geom)
        .filterDate(start, end)
        .select(ERA5_ET0_BANDS)
    )
    return _eight_day_reduce(ee, era5, start, end, ee.Reducer.mean(), "ERA5_LAND")


def main(aoi: Any = None, start: str = "2023-06-01", end: str = "2023-11-30", project: str | None = None) -> Any:
    """Build SM, rainfall and ET0-forcing collections. Requires EE auth."""
    ee = init_ee(project)
    if aoi is None:
        aoi = [76.30, 30.60, 76.55, 30.80]

    sm = smap_soil_moisture(ee, aoi, start, end)
    rain_imerg = imerg_rainfall(ee, aoi, start, end)
    rain_chirps = chirps_rainfall(ee, aoi, start, end)
    et0_inputs = era5_et0_inputs(ee, aoi, start, end)
    try:
        print("[gee/04] SMAP SM (surface+rootzone), IMERG+CHIRPS rainfall, ERA5-Land ET0 forcing built")
        print(f"[gee/04] SMAP composites: {sm.size().getInfo()}; ERA5 fields: {ERA5_ET0_BANDS}")
    except Exception as exc:  # pragma: no cover
        print(f"[gee/04] built SM/rain/ET0 (getInfo skipped: {exc})")
    return {
        "soil_moisture": sm,
        "rain_imerg": rain_imerg,
        "rain_chirps": rain_chirps,
        "et0_inputs": et0_inputs,
    }


if __name__ == "__main__":  # pragma: no cover
    import os
    import sys

    try:
        main(project=os.environ.get("EE_PROJECT"))
    except EarthEngineUnavailable as exc:
        print(str(exc))
        sys.exit(1)
