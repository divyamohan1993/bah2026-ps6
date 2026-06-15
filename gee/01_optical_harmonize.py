"""AgriStress · GEE step 01 — HLS-style optical harmonisation (Sentinel-2 + Landsat-8/9).

Implements **Stage 1** of ``docs/DATA_FUSION.md`` on Google Earth Engine: force
Sentinel-2 MSI and Landsat 8/9 OLI surface reflectance onto a single radiometric
reference, mask clouds/shadows, and emit a regular **8-day median composite**
collection over an AOI / date range.

Processing chain
----------------
1. **Cloud / shadow masking**
   * Sentinel-2 — Cloud Score+ ``cs_cdf >= 0.6`` (band ``cs_cdf`` of
     ``GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED``; the terrain-robust score
     recommended for masking, see DATA_FUSION §1.2).
   * Landsat 8/9 — ``QA_PIXEL`` bitmask (cloud, cloud-shadow, cirrus, dilated).
2. **Scaling** — apply each product's SR scale/offset to physical reflectance
   in [0, 1].
3. **S2 → OLI bandpass adjustment** — per-band linear transform
   ``ρ_OLI = a·ρ_S2 + b`` using the HLS v2.0 S2A→OLI table (DATA_FUSION §1.4),
   so Sentinel-2 and Landsat are radiometrically interoperable.
4. **8-day median composites** — group the merged, harmonised collection into
   contiguous 8-day windows and reduce each to a per-band median, yielding a
   gap-tolerant, evenly spaced ImageCollection with common band names
   ``[blue, green, red, nir, swir1, swir2]``.
5. **Export** — optional batch export of each composite to an EE asset
   ImageCollection (``export_collection``), the "offline factory → gold asset"
   pattern (see gee/README.md).

Everything is import-safe without credentials: ``import ee`` is guarded and all
EE calls live inside functions. Run :func:`main` after ``init_ee``.
"""

from __future__ import annotations

from typing import Any

from gee._auth import EarthEngineUnavailable, get_ee, init_ee

# Common, sensor-agnostic band names used downstream (indices, classifier).
COMMON_BANDS: list[str] = ["blue", "green", "red", "nir", "swir1", "swir2"]

# Asset IDs (kept in one place; mirrored in gee/README.md).
S2_SR_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
S2_CLOUDSCORE_COLLECTION = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
L8_SR_COLLECTION = "LANDSAT/LC08/C02/T1_L2"
L9_SR_COLLECTION = "LANDSAT/LC09/C02/T1_L2"

# Cloud Score+ threshold on cs_cdf (DATA_FUSION §1.2: permissive monsoon default).
CS_CDF_THRESHOLD = 0.6

# HLS v2.0 Sentinel-2A → Landsat-OLI bandpass coefficients (DATA_FUSION §1.4):
#   ρ_OLI = slope · ρ_S2 + intercept   (per common band)
S2_TO_OLI_BANDPASS: dict[str, tuple[float, float]] = {
    "blue": (0.9778, -0.0040),
    "green": (1.0053, -0.0009),
    "red": (0.9765, 0.0009),
    "nir": (0.9983, -0.0001),
    "swir1": (0.9987, -0.0011),
    "swir2": (1.0030, -0.0012),
}

# Native band name → common band name maps.
_S2_BAND_MAP = {
    "B2": "blue",
    "B3": "green",
    "B4": "red",
    "B8": "nir",
    "B11": "swir1",
    "B12": "swir2",
}
_LANDSAT_BAND_MAP = {
    "SR_B2": "blue",
    "SR_B3": "green",
    "SR_B4": "red",
    "SR_B5": "nir",
    "SR_B6": "swir1",
    "SR_B7": "swir2",
}


def _aoi_geometry(ee: Any, aoi: Any) -> Any:
    """Coerce an AOI (ee.Geometry, GeoJSON dict, or [w, s, e, n] bbox) to ee.Geometry."""
    if isinstance(aoi, (list, tuple)) and len(aoi) == 4:
        return ee.Geometry.Rectangle(list(aoi))
    if isinstance(aoi, dict):
        return ee.Geometry(aoi)
    return aoi  # assume already an ee.Geometry / ee.Feature geometry


def mask_s2_cloudscore(ee: Any, img: Any, threshold: float = CS_CDF_THRESHOLD) -> Any:
    """Mask a Sentinel-2 SR image using its linked Cloud Score+ ``cs_cdf`` band.

    Cloud Score+ images are joined by system:index; the image is expected to
    already carry a ``cs_cdf`` band (see :func:`load_sentinel2`). Keeps pixels
    with ``cs_cdf >= threshold``.
    """
    return img.updateMask(img.select("cs_cdf").gte(threshold))


def _scale_s2(ee: Any, img: Any) -> Any:
    """Scale S2 SR (DN×1e4) → reflectance, rename to common bands, keep cs_cdf."""
    scaled = img.select(list(_S2_BAND_MAP)).multiply(0.0001).rename(
        list(_S2_BAND_MAP.values())
    )
    return scaled.copyProperties(img, ["system:time_start"]).set(
        "SENSOR", "S2"
    )


def _apply_s2_to_oli(ee: Any, img: Any) -> Any:
    """Apply the per-band S2A→OLI linear bandpass adjustment (DATA_FUSION §1.4)."""
    out = img
    for band, (slope, intercept) in S2_TO_OLI_BANDPASS.items():
        adj = img.select(band).multiply(slope).add(intercept).rename(band)
        out = out.addBands(adj, overwrite=True)
    return out.set("HARMONIZED_TO", "OLI")


def load_sentinel2(ee: Any, aoi: Any, start: str, end: str, threshold: float = CS_CDF_THRESHOLD) -> Any:
    """Cloud-masked, scaled, OLI-harmonised Sentinel-2 SR collection over AOI/date."""
    geom = _aoi_geometry(ee, aoi)
    s2 = (
        ee.ImageCollection(S2_SR_COLLECTION)
        .filterBounds(geom)
        .filterDate(start, end)
    )
    cs = ee.ImageCollection(S2_CLOUDSCORE_COLLECTION).filterBounds(geom).filterDate(start, end)

    # Link each S2 scene to its Cloud Score+ image (shared system:index).
    join = ee.Join.saveFirst("cs")
    cond = ee.Filter.equals(leftField="system:index", rightField="system:index")
    joined = ee.ImageCollection(join.apply(s2, cs, cond))

    def _attach_and_mask(img: Any) -> Any:
        img = ee.Image(img)
        cs_img = ee.Image(img.get("cs"))
        img = img.addBands(cs_img.select("cs_cdf"))
        img = mask_s2_cloudscore(ee, img, threshold)
        img = _scale_s2(ee, img)
        return _apply_s2_to_oli(ee, img)

    return joined.map(_attach_and_mask).select(COMMON_BANDS)


def mask_landsat_qa(ee: Any, img: Any) -> Any:
    """Mask Landsat C2 L2 using the ``QA_PIXEL`` bitmask.

    Bits cleared: dilated-cloud (1), cirrus (2), cloud (3), cloud-shadow (4).
    """
    qa = img.select("QA_PIXEL")
    dilated = 1 << 1
    cirrus = 1 << 2
    cloud = 1 << 3
    shadow = 1 << 4
    mask = (
        qa.bitwiseAnd(dilated).eq(0)
        .And(qa.bitwiseAnd(cirrus).eq(0))
        .And(qa.bitwiseAnd(cloud).eq(0))
        .And(qa.bitwiseAnd(shadow).eq(0))
    )
    return img.updateMask(mask)


def _scale_landsat(ee: Any, img: Any) -> Any:
    """Apply Landsat C2 L2 SR scale/offset (×2.75e-5 − 0.2) → reflectance, rename."""
    scaled = (
        img.select(list(_LANDSAT_BAND_MAP))
        .multiply(0.0000275)
        .add(-0.2)
        .rename(list(_LANDSAT_BAND_MAP.values()))
    )
    return scaled.copyProperties(img, ["system:time_start"]).set("SENSOR", "Landsat")


def load_landsat(ee: Any, aoi: Any, start: str, end: str) -> Any:
    """Cloud-masked, scaled Landsat-8 + Landsat-9 SR collection (already OLI reference)."""
    geom = _aoi_geometry(ee, aoi)

    def _prep(img: Any) -> Any:
        img = mask_landsat_qa(ee, img)
        return _scale_landsat(ee, img)

    l8 = ee.ImageCollection(L8_SR_COLLECTION).filterBounds(geom).filterDate(start, end).map(_prep)
    l9 = ee.ImageCollection(L9_SR_COLLECTION).filterBounds(geom).filterDate(start, end).map(_prep)
    return l8.merge(l9).select(COMMON_BANDS)


def build_harmonized_collection(
    ee: Any, aoi: Any, start: str, end: str, threshold: float = CS_CDF_THRESHOLD
) -> Any:
    """Merge OLI-harmonised S2 + Landsat into one surface-reflectance collection."""
    s2 = load_sentinel2(ee, aoi, start, end, threshold)
    landsat = load_landsat(ee, aoi, start, end)
    return s2.merge(landsat).sort("system:time_start")


def to_8day_composites(ee: Any, collection: Any, start: str, end: str) -> Any:
    """Reduce a merged SR collection to evenly-spaced 8-day median composites.

    Returns an ImageCollection with one median image per 8-day window that
    contains data, each tagged with ``system:time_start`` at the window start.
    """
    start_d = ee.Date(start)
    n_days = ee.Date(end).difference(start_d, "day")
    n_steps = n_days.divide(8).ceil()
    steps = ee.List.sequence(0, n_steps.subtract(1))

    def _composite(i: Any) -> Any:
        w_start = start_d.advance(ee.Number(i).multiply(8), "day")
        w_end = w_start.advance(8, "day")
        window = collection.filterDate(w_start, w_end)
        med = window.median().set(
            {
                "system:time_start": w_start.millis(),
                "n_images": window.size(),
                "composite_days": 8,
            }
        )
        return med

    comps = ee.ImageCollection.fromImages(steps.map(_composite))
    # Drop empty windows (bands absent → no data).
    return comps.filter(ee.Filter.gt("n_images", 0))


def export_collection(
    ee: Any,
    collection: Any,
    asset_prefix: str,
    aoi: Any,
    scale: int = 30,
    max_images: int = 64,
) -> list[Any]:
    """Batch-export each composite image to ``{asset_prefix}/comp_{i}`` assets.

    Returns the list of started ``ee.batch.Task`` handles. The 30 m scale and
    per-image asset layout match the HLS L30/S30 grid and feed the downstream
    "gold COG → tiles" serving path (see gee/README.md). Tasks must be started
    in a credentialed session; call ``task.start()`` on each.
    """
    geom = _aoi_geometry(ee, aoi)
    img_list = collection.toList(max_images)
    n = min(int(collection.size().getInfo()), max_images)
    tasks = []
    for i in range(n):
        img = ee.Image(img_list.get(i))
        task = ee.batch.Export.image.toAsset(
            image=img.clip(geom),
            description=f"agristress_optical_comp_{i:03d}",
            assetId=f"{asset_prefix}/comp_{i:03d}",
            region=geom,
            scale=scale,
            maxPixels=int(1e13),
        )
        tasks.append(task)
    return tasks


def main(aoi: Any = None, start: str = "2023-06-01", end: str = "2023-11-30", project: str | None = None) -> Any:
    """Build and (lazily) describe the 8-day harmonised optical composites.

    Returns the composite ImageCollection. Prints a short summary using a single
    ``getInfo`` round-trip. Requires ``earthengine authenticate`` — raises
    :class:`EarthEngineUnavailable` (with setup recipe) otherwise.
    """
    ee = init_ee(project)
    if aoi is None:
        # Default pilot: a small canal-command-area bbox (lon/lat, W,S,E,N).
        aoi = [76.30, 30.60, 76.55, 30.80]  # ~Patiala/Bhakra command demo box

    merged = build_harmonized_collection(ee, aoi, start, end)
    comps = to_8day_composites(ee, merged, start, end)
    try:
        n = comps.size().getInfo()
        print(f"[gee/01] {n} eight-day harmonised optical composites for {start}..{end}")
        print(f"[gee/01] common bands: {COMMON_BANDS} (S2 bandpass-adjusted to OLI reference)")
    except Exception as exc:  # pragma: no cover - network/quota
        print(f"[gee/01] built collection (getInfo skipped: {exc})")
    return comps


if __name__ == "__main__":  # pragma: no cover - manual entry point
    import os
    import sys

    try:
        main(project=os.environ.get("EE_PROJECT"))
    except EarthEngineUnavailable as exc:
        print(str(exc))
        sys.exit(1)
