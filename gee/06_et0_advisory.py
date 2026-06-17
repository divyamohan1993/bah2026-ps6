"""AgriStress · GEE step 06 — FAO-56 ET0, ETc, 8-day water deficit & irrigation advisory.

DATA_FUSION Head 3 on Google Earth Engine. Turns the ERA5-Land forcing (step 04),
the NDVI series (step 03) and rainfall (step 04) into an **8-day crop-water-deficit**
layer and a colour-coded **irrigation advisory** map for canal command areas.

Method (physically-grounded, FAO-56)
------------------------------------
1. **ET0 (reference ET)** — FAO-56 Penman-Monteith from ERA5-Land 8-day fields
   (air temp, dewpoint, wind, net radiation, pressure). Implemented server-side
   in :func:`fao56_penman_monteith`.
2. **Kc from NDVI** — linear ``Kc = a·NDVI + b`` (Kc–NDVI relationship; a≈1.45,
   b≈−0.1 clamped to [0.15, 1.2]). Avoids hard-coded growth-stage tables and is
   naturally phenology-aware.
3. **ETc (crop ET)** — ``ETc = Kc · ET0`` (mm / 8-day).
4. **Root-zone water balance** — ``ΔS = Rain + Irr − ETc``; track depletion vs.
   the readily-available water (RAW = p·TAW). The **8-day deficit** is the unmet
   crop demand ``max(ETc − Rain − SM_supply, 0)``.
5. **Advisory classes** — bucket the deficit (and root-zone SM, if supplied) into
   ``{no_irrigation, watch, deficit, severe_deficit}`` with explanatory codes.

Import-safe without credentials. Run :func:`main` after ``init_ee``.
"""

from __future__ import annotations

from typing import Any

from gee._auth import EarthEngineUnavailable, init_ee

# Kc–NDVI linear relationship (clamped); see FAO-56 / Kc-NDVI literature.
KC_NDVI_SLOPE = 1.45
KC_NDVI_INTERCEPT = -0.10
KC_MIN, KC_MAX = 0.15, 1.20

# Advisory deficit thresholds (mm per 8-day window).
ADVISORY_THRESHOLDS = {
    "no_irrigation": 5.0,    # deficit < 5 mm  → 0
    "watch": 15.0,           # 5–15 mm         → 1
    "deficit": 30.0,         # 15–30 mm        → 2
    # > 30 mm                                  → 3 severe_deficit
}
ADVISORY_CLASSES = {
    0: "no_irrigation",
    1: "watch",
    2: "deficit",
    3: "severe_deficit",
}


def _aoi_geometry(ee: Any, aoi: Any) -> Any:
    if isinstance(aoi, (list, tuple)) and len(aoi) == 4:
        return ee.Geometry.Rectangle(list(aoi))
    if isinstance(aoi, dict):
        return ee.Geometry(aoi)
    return aoi


def fao56_penman_monteith(ee: Any, era5_img: Any, day_count: int = 8) -> Any:
    """FAO-56 Penman-Monteith reference ET (mm over ``day_count`` days) from ERA5-Land.

    Expects an 8-day-mean ERA5-Land image with bands:
    ``temperature_2m`` (K), ``dewpoint_temperature_2m`` (K),
    ``u_component_of_wind_10m`` / ``v_component_of_wind_10m`` (m/s),
    ``surface_net_solar_radiation`` (J/m² accumulated → converted to MJ/m²/day),
    ``surface_pressure`` (Pa).
    """
    t_k = era5_img.select("temperature_2m")
    t_c = t_k.subtract(273.15)
    td_c = era5_img.select("dewpoint_temperature_2m").subtract(273.15)
    u10 = era5_img.select("u_component_of_wind_10m")
    v10 = era5_img.select("v_component_of_wind_10m")
    # 10 m → 2 m wind (FAO-56 log-law factor 0.748).
    u2 = u10.hypot(v10).multiply(0.748).rename("u2")
    # Net solar radiation J/m²/day-mean → MJ/m²/day (≈ net radiation proxy Rn).
    rn = era5_img.select("surface_net_solar_radiation").divide(1e6).max(0)
    p_kpa = era5_img.select("surface_pressure").divide(1000.0)

    # Saturation & actual vapour pressure (kPa).
    es = t_c.expression("0.6108 * exp(17.27 * T / (T + 237.3))", {"T": t_c})
    ea = td_c.expression("0.6108 * exp(17.27 * Td / (Td + 237.3))", {"Td": td_c})
    # Slope of vapour-pressure curve Δ (kPa/°C) and psychrometric γ (kPa/°C).
    delta = t_c.expression(
        "4098 * (0.6108 * exp(17.27 * T / (T + 237.3))) / pow(T + 237.3, 2)", {"T": t_c}
    )
    gamma = p_kpa.multiply(0.000665)

    # FAO-56 PM (G≈0 for 8-day step). Tmean in K for the (900/T) term.
    et0_day = t_c.expression(
        "(0.408 * D * Rn + g * (900 / (T + 273)) * U * (es - ea)) / (D + g * (1 + 0.34 * U))",
        {"D": delta, "Rn": rn, "g": gamma, "T": t_c, "U": u2, "es": es, "ea": ea},
    ).max(0)
    return et0_day.multiply(day_count).rename("ET0")


def kc_from_ndvi(ee: Any, ndvi_img: Any) -> Any:
    """Crop coefficient Kc from NDVI (linear, clamped) — phenology-aware."""
    return (
        ndvi_img.multiply(KC_NDVI_SLOPE)
        .add(KC_NDVI_INTERCEPT)
        .clamp(KC_MIN, KC_MAX)
        .rename("Kc")
    )


def etc_from_kc_et0(ee: Any, kc_img: Any, et0_img: Any) -> Any:
    """Crop ET ``ETc = Kc · ET0`` (mm / 8-day)."""
    return kc_img.multiply(et0_img).rename("ETc")


def water_deficit(ee: Any, etc_img: Any, rain_img: Any, sm_supply_img: Any | None = None) -> Any:
    """8-day crop-water deficit ``max(ETc − Rain − SM_supply, 0)`` (mm).

    ``sm_supply_img`` (optional) is an effective soil-moisture contribution in mm
    over the window (e.g. derived from the root-zone SWI change in step 04).
    """
    supply = rain_img.rename("supply")
    if sm_supply_img is not None:
        supply = supply.add(sm_supply_img)
    deficit = etc_img.subtract(supply).max(0).rename("deficit_mm")
    return deficit


def advisory_classes(ee: Any, deficit_img: Any) -> Any:
    """Bucket the 8-day deficit into ordinal advisory classes (0..3)."""
    t = ADVISORY_THRESHOLDS
    cls = (
        deficit_img.gte(t["no_irrigation"]).int()
        .add(deficit_img.gte(t["watch"]).int())
        .add(deficit_img.gte(t["deficit"]).int())
        .rename("advisory")
    )
    return cls


def advisory_for_step(
    ee: Any,
    era5_img: Any,
    ndvi_img: Any,
    rain_img: Any,
    sm_supply_img: Any | None = None,
    day_count: int = 8,
) -> dict[str, Any]:
    """Full 8-day advisory for one step: ET0 → Kc → ETc → deficit → class."""
    et0 = fao56_penman_monteith(ee, era5_img, day_count)
    kc = kc_from_ndvi(ee, ndvi_img)
    etc = etc_from_kc_et0(ee, kc, et0)
    deficit = water_deficit(ee, etc, rain_img, sm_supply_img)
    advisory = advisory_classes(ee, deficit)
    bundle = ee.Image.cat([et0, kc, etc, deficit, advisory])
    return {"et0": et0, "kc": kc, "etc": etc, "deficit": deficit, "advisory": advisory, "bundle": bundle}


def export_advisory(ee: Any, advisory_img: Any, asset_id: str, aoi: Any, scale: int = 30) -> Any:
    """Export the advisory bundle to an EE asset (gold artefact for tiles)."""
    geom = _aoi_geometry(ee, aoi)
    return ee.batch.Export.image.toAsset(
        image=advisory_img.clip(geom),
        description="agristress_advisory",
        assetId=asset_id,
        region=geom,
        scale=scale,
        maxPixels=int(1e13),
    )


def main(aoi: Any = None, start: str = "2023-06-01", end: str = "2023-11-30", project: str | None = None) -> Any:
    """Build the latest-step ET0/ETc/deficit/advisory layers. Requires EE auth."""
    ee = init_ee(project)
    if aoi is None:
        aoi = [76.30, 30.60, 76.55, 30.80]

    import importlib.util
    import os

    here = os.path.dirname(os.path.abspath(__file__))

    def _load(fname: str, name: str) -> Any:
        spec = importlib.util.spec_from_file_location(name, os.path.join(here, fname))
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return mod

    opt = _load("01_optical_harmonize.py", "gee_01")
    idx_mod = _load("03_indices_phenology.py", "gee_03")
    sm_mod = _load("04_soil_moisture.py", "gee_04")

    merged = opt.build_harmonized_collection(ee, aoi, start, end)
    comps = opt.to_8day_composites(ee, merged, start, end)
    index_ts = idx_mod.index_time_series(ee, comps)
    et0_inputs = sm_mod.era5_et0_inputs(ee, aoi, start, end)
    rain = sm_mod.chirps_rainfall(ee, aoi, start, end)

    # Use the most recent 8-day step for the headline advisory.
    era5_last = ee.Image(et0_inputs.sort("system:time_start", False).first())
    ndvi_last = ee.Image(index_ts.sort("system:time_start", False).first()).select("NDVI")
    rain_last = ee.Image(rain.sort("system:time_start", False).first()).select("rain_mm")

    out = advisory_for_step(ee, era5_last, ndvi_last, rain_last)
    try:
        print("[gee/06] FAO-56 ET0 → Kc(NDVI) → ETc → 8-day deficit → advisory built.")
        print(f"[gee/06] advisory classes: {ADVISORY_CLASSES}")
    except Exception as exc:  # pragma: no cover
        print(f"[gee/06] built advisory (getInfo skipped: {exc})")
    return out


if __name__ == "__main__":  # pragma: no cover
    import os
    import sys

    try:
        main(project=os.environ.get("EE_PROJECT"))
    except EarthEngineUnavailable as exc:
        print(str(exc))
        sys.exit(1)
