"""AgriStress · GEE step 05 — crop-type classification (feature-stack + AlphaEarth).

DATA_FUSION Head 1 on Google Earth Engine. Two interchangeable feature paths are
provided, both classified with ``ee.Classifier.smileRandomForest`` and validated
with an ``errorMatrix`` (Overall Accuracy + Cohen's Kappa):

A. **Multi-temporal feature stack** — stack the per-composite optical indices
   (NDVI/EVI/NDWI/NDMI/STR from step 03) and Sentinel-1 SAR features
   (VV/VH/ratio/RVI from step 02) across the season into one wide image, the
   classic multi-temporal classifier input named in PS6.

B. **AlphaEarth Satellite Embedding** — Google's annual 64-D learned embedding
   ``GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`` (bands ``A00``..``A63``), a single
   analysis-ready image per year that already fuses optical+SAR+context. Often
   matches or beats hand-built stacks with far fewer inputs.

Both paths:
  * sample training/validation points (``sampleRegions``),
  * train ``smileRandomForest`` (default 100 trees),
  * classify the image into a crop-type map,
  * compute accuracy via ``errorMatrix`` →
    ``.accuracy()`` (Overall Accuracy) and ``.kappa()`` (Cohen's Kappa).

Import-safe without credentials. Run :func:`main` after ``init_ee``.
"""

from __future__ import annotations

from typing import Any

from gee._auth import EarthEngineUnavailable, init_ee

# AlphaEarth annual satellite embedding (64-D); the parallel feature path.
SATELLITE_EMBEDDING_COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
EMBEDDING_BANDS = [f"A{i:02d}" for i in range(64)]  # A00..A63

DEFAULT_N_TREES = 100
DEFAULT_CLASS_PROPERTY = "crop_class"


def _aoi_geometry(ee: Any, aoi: Any) -> Any:
    if isinstance(aoi, (list, tuple)) and len(aoi) == 4:
        return ee.Geometry.Rectangle(list(aoi))
    if isinstance(aoi, dict):
        return ee.Geometry(aoi)
    return aoi


# --------------------------------------------------------------------------
# Path A — multi-temporal optical + SAR feature stack
# --------------------------------------------------------------------------
def stack_time_series(ee: Any, collection: Any, band_prefix: str) -> Any:
    """Flatten an ImageCollection into one multiband image (``{prefix}_t{idx}_{band}``).

    Each composite contributes its bands suffixed with the time index, giving a
    wide per-pixel feature vector spanning the season.
    """
    img_list = collection.toList(collection.size())
    n = collection.size()

    def _one(i: Any) -> Any:
        idx = ee.Number(i).int()
        img = ee.Image(img_list.get(idx))
        bands = img.bandNames()
        renamed = bands.map(
            lambda b: ee.String(band_prefix)
            .cat("_t")
            .cat(idx.format("%02d"))
            .cat("_")
            .cat(ee.String(b))
        )
        return img.rename(renamed)

    stacked = ee.ImageCollection(ee.List.sequence(0, n.subtract(1)).map(_one)).toBands()
    return stacked


def build_feature_stack(ee: Any, index_collection: Any, sar_collection: Any) -> Any:
    """Concatenate the multi-temporal optical-index stack and the SAR-feature stack."""
    optical = stack_time_series(ee, index_collection, "OPT")
    sar = stack_time_series(ee, sar_collection, "SAR")
    return optical.addBands(sar)


# --------------------------------------------------------------------------
# Path B — AlphaEarth Satellite Embedding (64-D)
# --------------------------------------------------------------------------
def satellite_embedding_image(ee: Any, aoi: Any, year: int) -> Any:
    """Return the annual AlphaEarth 64-D embedding image for the AOI/year.

    The collection has one global image per calendar year; we mosaic the tiles
    intersecting the AOI for the requested year.
    """
    geom = _aoi_geometry(ee, aoi)
    start = ee.Date.fromYMD(year, 1, 1)
    end = start.advance(1, "year")
    emb = (
        ee.ImageCollection(SATELLITE_EMBEDDING_COLLECTION)
        .filterBounds(geom)
        .filterDate(start, end)
    )
    return emb.mosaic().select(EMBEDDING_BANDS).clip(geom)


# --------------------------------------------------------------------------
# Train / classify / evaluate (shared by both paths)
# --------------------------------------------------------------------------
def sample_features(
    ee: Any,
    features_img: Any,
    points: Any,
    class_property: str = DEFAULT_CLASS_PROPERTY,
    scale: int = 10,
) -> Any:
    """Sample the feature image at labelled training/validation points."""
    return features_img.sampleRegions(
        collection=points,
        properties=[class_property],
        scale=scale,
        tileScale=4,
        geometries=False,
    )


def train_random_forest(
    ee: Any,
    training: Any,
    feature_img: Any,
    class_property: str = DEFAULT_CLASS_PROPERTY,
    n_trees: int = DEFAULT_N_TREES,
) -> Any:
    """Train a ``smileRandomForest`` classifier on the sampled training table."""
    bands = feature_img.bandNames()
    classifier = ee.Classifier.smileRandomForest(numberOfTrees=n_trees).train(
        features=training,
        classProperty=class_property,
        inputProperties=bands,
    )
    return classifier


def classify_image(ee: Any, feature_img: Any, classifier: Any) -> Any:
    """Apply a trained classifier to the feature image → crop-type map."""
    return feature_img.classify(classifier).rename("crop_class")


def accuracy_assessment(
    ee: Any,
    classifier: Any,
    validation: Any,
    class_property: str = DEFAULT_CLASS_PROPERTY,
) -> dict[str, Any]:
    """Validate via a confusion ``errorMatrix`` → Overall Accuracy + Kappa.

    Returns server-side objects (``.getInfo()`` to realise): the error matrix,
    overall accuracy, Cohen's kappa, and per-class producer/consumer accuracy —
    the PS6-mandated metrics (target OA > 85%).
    """
    validated = validation.classify(classifier, "predicted")
    matrix = validated.errorMatrix(class_property, "predicted")
    return {
        "error_matrix": matrix,
        "overall_accuracy": matrix.accuracy(),
        "kappa": matrix.kappa(),
        "producers_accuracy": matrix.producersAccuracy(),
        "consumers_accuracy": matrix.consumersAccuracy(),
    }


def classify_with_features(
    ee: Any,
    feature_img: Any,
    points: Any,
    class_property: str = DEFAULT_CLASS_PROPERTY,
    n_trees: int = DEFAULT_N_TREES,
    train_fraction: float = 0.7,
    scale: int = 10,
    seed: int = 42,
) -> dict[str, Any]:
    """End-to-end: sample → 70/30 split → RF train → classify → errorMatrix/Kappa.

    Works identically for the multi-temporal stack (Path A) and the AlphaEarth
    embedding (Path B) — just pass the corresponding ``feature_img``.
    """
    samples = sample_features(ee, feature_img, points, class_property, scale)
    samples = samples.randomColumn("rand", seed)
    training = samples.filter(ee.Filter.lt("rand", train_fraction))
    validation = samples.filter(ee.Filter.gte("rand", train_fraction))

    classifier = train_random_forest(ee, training, feature_img, class_property, n_trees)
    crop_map = classify_image(ee, feature_img, classifier)
    metrics = accuracy_assessment(ee, classifier, validation, class_property)
    return {"classifier": classifier, "crop_map": crop_map, **metrics}


def export_crop_map(ee: Any, crop_map: Any, asset_id: str, aoi: Any, scale: int = 10) -> Any:
    """Export the crop-type map to an EE asset (gold artefact for the tile server)."""
    geom = _aoi_geometry(ee, aoi)
    return ee.batch.Export.image.toAsset(
        image=crop_map.clip(geom),
        description="agristress_crop_map",
        assetId=asset_id,
        region=geom,
        scale=scale,
        maxPixels=int(1e13),
    )


def main(aoi: Any = None, start: str = "2022-06-01", end: str = "2022-11-30", project: str | None = None, points: Any = None) -> Any:
    """Run both classification paths and report OA/Kappa. Requires EE auth + labels.

    ``points`` must be an ``ee.FeatureCollection`` of labelled samples with a
    ``crop_class`` integer property. When absent, the feature images are still
    built (Path A stack and Path B embedding) and described, but training is
    skipped with a clear note.
    """
    ee = init_ee(project)
    if aoi is None:
        aoi = [76.30, 30.60, 76.55, 30.80]
    year = int(start[:4])

    # Path B feature image (single AlphaEarth call — cheap to build/describe).
    embedding = satellite_embedding_image(ee, aoi, year)

    # Path A feature stack from steps 02/03 (lazy file-path imports of numbered modules).
    import importlib.util
    import os

    here = os.path.dirname(os.path.abspath(__file__))

    def _load(fname: str, name: str) -> Any:
        spec = importlib.util.spec_from_file_location(name, os.path.join(here, fname))
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return mod

    opt = _load("01_optical_harmonize.py", "gee_01")
    sar = _load("02_sar_s1.py", "gee_02")
    idx_mod = _load("03_indices_phenology.py", "gee_03")

    merged = opt.build_harmonized_collection(ee, aoi, start, end)
    comps = opt.to_8day_composites(ee, merged, start, end)
    index_ts = idx_mod.index_time_series(ee, comps)
    s1 = sar.load_sentinel1(ee, aoi, start, end)
    s1_comps = sar.to_8day_composites(ee, s1, start, end, orbit_aware=False)
    stack = build_feature_stack(ee, index_ts, s1_comps)

    result: dict[str, Any] = {"embedding": embedding, "feature_stack": stack}

    if points is not None:
        # Path A
        a = classify_with_features(ee, stack, points)
        # Path B
        b = classify_with_features(ee, embedding, points)
        result["stack_metrics"] = a
        result["embedding_metrics"] = b
        try:
            print("[gee/05] Path A (multi-temporal stack) OA =", a["overall_accuracy"].getInfo(),
                  "Kappa =", a["kappa"].getInfo())
            print("[gee/05] Path B (AlphaEarth 64-D)       OA =", b["overall_accuracy"].getInfo(),
                  "Kappa =", b["kappa"].getInfo())
        except Exception as exc:  # pragma: no cover
            print(f"[gee/05] trained both paths (getInfo skipped: {exc})")
    else:
        print("[gee/05] feature stack + AlphaEarth embedding built.")
        print("[gee/05] pass points=ee.FeatureCollection(...) with 'crop_class' to train RF + get OA/Kappa.")
        print(f"[gee/05] AlphaEarth asset: {SATELLITE_EMBEDDING_COLLECTION} ({len(EMBEDDING_BANDS)}-D).")
    return result


if __name__ == "__main__":  # pragma: no cover
    import os
    import sys

    try:
        main(project=os.environ.get("EE_PROJECT"))
    except EarthEngineUnavailable as exc:
        print(str(exc))
        sys.exit(1)
