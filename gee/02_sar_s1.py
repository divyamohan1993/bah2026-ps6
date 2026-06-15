"""AgriStress · GEE step 02 — Sentinel-1 GRD SAR preparation & 8-day composites.

Builds the all-weather microwave layer (DATA_FUSION Stage 3 inputs) on Google
Earth Engine: clean Sentinel-1 GRD backscatter, derive vegetation-sensitive SAR
features, and reduce to evenly-spaced **8-day composites** with explicit
ascending / descending handling.

Processing chain
----------------
1. **Filtering** — ``COPERNICUS/S1_GRD`` to IW mode, VV+VH dual-pol, 10 m, over
   the AOI/date range; split by ``orbitProperties_pass`` (ASCENDING /
   DESCENDING) since their geometry/backscatter are not directly comparable.
2. **Border-noise handling** — GEE's S1_GRD is already thermal-noise-removed and
   border-noise-corrected by the IPF; we additionally trim residual low-signal
   scene edges by masking VV/VH below a noise floor (default −30 dB).
3. **Speckle handling** — a focal-mean (boxcar) multi-look in the **linear**
   domain (a pragmatic GEE stand-in for the Refined-Lee filter named in the PS),
   converted back to dB. Window size is configurable.
4. **Features** — ``VV``, ``VH`` (dB), ``VH/VV`` ratio (dB difference), and the
   **Radar Vegetation Index** ``RVI = 4·VH / (VV + VH)`` computed in linear
   power (canopy-density proxy, 0–~1).
5. **8-day composites** — per-orbit median composites on a regular 8-day grid.

Import-safe without credentials (``import ee`` guarded; all EE calls in
functions). Run :func:`main` after ``init_ee``.
"""

from __future__ import annotations

from typing import Any

from gee._auth import EarthEngineUnavailable, init_ee

S1_GRD_COLLECTION = "COPERNICUS/S1_GRD"

# SAR feature band names produced by this module.
SAR_BANDS: list[str] = ["VV", "VH", "VH_VV_ratio", "RVI"]

# Residual edge-noise floor (dB); pixels below this in VV or VH are masked.
NOISE_FLOOR_DB = -30.0
# Default speckle multi-look window (pixels) for the boxcar filter.
SPECKLE_WINDOW = 30


def _aoi_geometry(ee: Any, aoi: Any) -> Any:
    if isinstance(aoi, (list, tuple)) and len(aoi) == 4:
        return ee.Geometry.Rectangle(list(aoi))
    if isinstance(aoi, dict):
        return ee.Geometry(aoi)
    return aoi


def _to_linear(ee: Any, db_img: Any) -> Any:
    """dB → linear power: 10^(dB/10)."""
    return ee.Image(10.0).pow(db_img.divide(10.0))


def _to_db(ee: Any, lin_img: Any) -> Any:
    """linear power → dB: 10·log10(power)."""
    return lin_img.log10().multiply(10.0)


def mask_edge_noise(ee: Any, img: Any, floor_db: float = NOISE_FLOOR_DB) -> Any:
    """Mask residual scene-edge / low-signal pixels below a dB noise floor."""
    keep = img.select("VV").gt(floor_db).And(img.select("VH").gt(floor_db))
    return img.updateMask(keep)


def refined_lee_boxcar(ee: Any, img: Any, window: int = SPECKLE_WINDOW) -> Any:
    """Speckle-reduce VV/VH via a linear-domain boxcar multi-look.

    Pragmatic GEE substitute for the Refined-Lee filter named in PS6: convert to
    linear power, focal-mean over a square window, convert back to dB. Operating
    in linear (not dB) is important — averaging dB biases the estimate.
    """
    kernel = ee.Kernel.square(radius=window, units="meters")
    out = img
    for band in ("VV", "VH"):
        lin = _to_linear(ee, img.select(band))
        smoothed = lin.reduceNeighborhood(reducer=ee.Reducer.mean(), kernel=kernel)
        out = out.addBands(_to_db(ee, smoothed).rename(band), overwrite=True)
    return out


def add_sar_features(ee: Any, img: Any) -> Any:
    """Add VH/VV ratio (dB difference) and RVI (computed in linear power)."""
    vv_db = img.select("VV")
    vh_db = img.select("VH")
    ratio = vh_db.subtract(vv_db).rename("VH_VV_ratio")  # dB difference == log ratio

    vv_lin = _to_linear(ee, vv_db)
    vh_lin = _to_linear(ee, vh_db)
    # RVI = 4*VH / (VV + VH) in linear power; canopy-density / vegetation proxy.
    rvi = vh_lin.multiply(4).divide(vv_lin.add(vh_lin)).rename("RVI")
    return img.addBands([ratio, rvi])


def load_sentinel1(
    ee: Any,
    aoi: Any,
    start: str,
    end: str,
    orbit: str | None = None,
    speckle_window: int = SPECKLE_WINDOW,
) -> Any:
    """Prepared Sentinel-1 GRD collection (masked, despeckled, with SAR features).

    Parameters
    ----------
    orbit:
        ``"ASCENDING"``, ``"DESCENDING"``, or ``None`` to keep both passes (a
        ``relative_pass`` property is carried for downstream per-orbit grouping).
    """
    geom = _aoi_geometry(ee, aoi)
    coll = (
        ee.ImageCollection(S1_GRD_COLLECTION)
        .filterBounds(geom)
        .filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .select(["VV", "VH"])
    )
    if orbit:
        coll = coll.filter(ee.Filter.eq("orbitProperties_pass", orbit))

    def _prep(img: Any) -> Any:
        img = mask_edge_noise(ee, img)
        img = refined_lee_boxcar(ee, img, speckle_window)
        img = add_sar_features(ee, img)
        return img.select(SAR_BANDS).copyProperties(
            img, ["system:time_start", "orbitProperties_pass", "relativeOrbitNumber_start"]
        )

    return coll.map(_prep)


def to_8day_composites(ee: Any, collection: Any, start: str, end: str, orbit_aware: bool = True) -> Any:
    """Reduce S1 features to 8-day median composites (optionally per orbit pass).

    With ``orbit_aware`` the ascending and descending sub-collections are
    composited separately and merged, so each 8-day step can carry both
    geometries (tagged via the ``pass`` property) rather than blending them.
    """
    if orbit_aware:
        asc = collection.filter(ee.Filter.eq("orbitProperties_pass", "ASCENDING"))
        desc = collection.filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
        comp_asc = _composite_window(ee, asc, start, end, "ASCENDING")
        comp_desc = _composite_window(ee, desc, start, end, "DESCENDING")
        return comp_asc.merge(comp_desc).sort("system:time_start")
    return _composite_window(ee, collection, start, end, "BOTH")


def _composite_window(ee: Any, collection: Any, start: str, end: str, pass_label: str) -> Any:
    start_d = ee.Date(start)
    n_days = ee.Date(end).difference(start_d, "day")
    n_steps = n_days.divide(8).ceil()
    steps = ee.List.sequence(0, n_steps.subtract(1))

    def _comp(i: Any) -> Any:
        w_start = start_d.advance(ee.Number(i).multiply(8), "day")
        w_end = w_start.advance(8, "day")
        window = collection.filterDate(w_start, w_end)
        return window.median().set(
            {
                "system:time_start": w_start.millis(),
                "n_images": window.size(),
                "pass": pass_label,
                "composite_days": 8,
            }
        )

    comps = ee.ImageCollection.fromImages(steps.map(_comp))
    return comps.filter(ee.Filter.gt("n_images", 0))


def main(aoi: Any = None, start: str = "2023-06-01", end: str = "2023-11-30", project: str | None = None) -> Any:
    """Build the 8-day Sentinel-1 SAR composites. Requires EE auth."""
    ee = init_ee(project)
    if aoi is None:
        aoi = [76.30, 30.60, 76.55, 30.80]

    prepared = load_sentinel1(ee, aoi, start, end)
    comps = to_8day_composites(ee, prepared, start, end, orbit_aware=True)
    try:
        n = comps.size().getInfo()
        print(f"[gee/02] {n} eight-day Sentinel-1 composites (ASC+DESC) for {start}..{end}")
        print(f"[gee/02] SAR features: {SAR_BANDS}")
    except Exception as exc:  # pragma: no cover
        print(f"[gee/02] built S1 collection (getInfo skipped: {exc})")
    return comps


if __name__ == "__main__":  # pragma: no cover
    import os
    import sys

    try:
        main(project=os.environ.get("EE_PROJECT"))
    except EarthEngineUnavailable as exc:
        print(str(exc))
        sys.exit(1)
