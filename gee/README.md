# `gee/` — Google Earth Engine pipeline (the offline "factory")

Runnable Earth Engine scripts that execute the AgriStress multi-satellite
pipeline server-side on Google's planetary-scale compute, then **export gold
artefacts** (crop map, stress, advisory, 8-day composites) for the tile server.

These scripts implement the methodology in [`../docs/DATA_FUSION.md`](../docs/DATA_FUSION.md)
for ISRO BAH 2026 Problem Statement 6 (AI-driven crop-type classification,
phenology-aware moisture-stress detection, and 8-day irrigation advisory for
canal command areas).

---

## The O(1) "precompute-then-serve" philosophy

> **Earth Engine is the factory, not the storefront.**

The expensive, planetary-scale work — pulling 30+ satellite collections, cloud
masking, harmonising, fusing, classifying — runs **once, offline**, inside GEE.
The outputs are exported as **gold Cloud-Optimized GeoTIFFs (COGs) / EE assets**.

At request time the dashboard never touches a satellite: it reads a pre-baked
tile from object storage. That is the **O(1) serve** — map-pan latency is a CDN
fetch, independent of AOI size or the number of fused sensors:

```
  RAW (30+ sats)  ──►  GEE factory (this dir)  ──►  gold COG/asset  ──►  XYZ tiles  ──►  dashboard
   O(everything)        run once, batch              precomputed         O(1) read       instant
```

Each script therefore has an `export_*` helper that writes an EE asset (or COG)
— the boundary between the heavy factory and the instant storefront.

---

## Scripts

| # | File | Stage(s) | What it does |
|---|------|----------|--------------|
| 00 | `00_auth.py` | — | `init_ee(project)` + `ee_available()`; one place for EE auth. Importing it never crashes without credentials. (`_auth.py` re-exports it under an importable name, since module names can't start with a digit.) |
| 01 | `01_optical_harmonize.py` | 1 | HLS-style **Sentinel-2 + Landsat-8/9** surface-reflectance: Cloud Score+ `cs_cdf ≥ 0.6` (S2) + `QA_PIXEL` (Landsat) masking, SR scaling, **S2→OLI bandpass** adjustment, **8-day median composites**, asset export. |
| 02 | `02_sar_s1.py` | 3 | **Sentinel-1 GRD**: edge/border-noise masking, boxcar (Refined-Lee-style) speckle reduction, `VV`/`VH`/`VH⁄VV`/**RVI**, **ascending/descending** handling, 8-day composites. |
| 03 | `03_indices_phenology.py` | 5 | **NDVI/EVI/NDWI/NDMI/STR** time-series + **harmonic** regression and **double-logistic** (TIMESAT-style) phenology → **SOS/EOS/LGP**. |
| 04 | `04_soil_moisture.py` | 4 | **SMAP L4** soil moisture + **Sentinel-1 downscaling sketch** (SPL2SMAP_S-style β-model) + **IMERG/CHIRPS** rainfall + **ERA5-Land** fields for ET0. |
| 05 | `05_crop_classification.py` | Head 1 | Multi-temporal **feature stack** (optical indices + S1) **and** a parallel **AlphaEarth Satellite Embedding** (64-D) path; trains `smileRandomForest`; exports crop map + **Overall Accuracy & Kappa** via `errorMatrix`. |
| 06 | `06_et0_advisory.py` | Head 3 | **FAO-56 Penman-Monteith ET0** (ERA5-Land) → **Kc-from-NDVI** → **ETc** → rainfall (IMERG/CHIRPS) → root-zone balance → **8-day deficit** → **advisory classes**; export. |

Every numbered script exposes `main(aoi, start, end, project=...)` plus an
`if __name__ == "__main__"` guard that prints a clear *"requires
`earthengine authenticate`"* message (and exits 1) when EE is unavailable.

---

## Asset / collection IDs used

| Purpose | Earth Engine ID |
|---|---|
| Sentinel-2 SR (harmonised) | `COPERNICUS/S2_SR_HARMONIZED` |
| Cloud Score+ (S2 masking, band `cs_cdf`) | `GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED` |
| Landsat 8 / 9 SR (Collection 2 L2) | `LANDSAT/LC08/C02/T1_L2`, `LANDSAT/LC09/C02/T1_L2` |
| Sentinel-1 GRD | `COPERNICUS/S1_GRD` |
| SMAP L4 soil moisture | `NASA/SMAP/SPL4SMGP/008` |
| GPM IMERG rainfall | `NASA/GPM_L3/IMERG_V07` |
| CHIRPS daily rainfall | `UCSB-CHG/CHIRPS/DAILY` |
| ERA5-Land hourly (ET0 forcing) | `ECMWF/ERA5_LAND/HOURLY` |
| **AlphaEarth** Satellite Embedding (64-D, annual) | `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` |

---

## Auth setup

```bash
pip install earthengine-api geemap          # client libraries
earthengine authenticate                    # interactive OAuth (laptops/Colab)

# …or a service account for headless / CI:
export EE_PROJECT=your-gcp-project
export GEE_SERVICE_ACCOUNT=ee-runner@your-gcp-project.iam.gserviceaccount.com
export GEE_SERVICE_ACCOUNT_KEY=/secrets/gee-sa.json
```

Run any step:

```bash
export EE_PROJECT=your-gcp-project
python gee/01_optical_harmonize.py          # builds 8-day harmonised optical composites
python gee/05_crop_classification.py         # builds feature stack + AlphaEarth embedding
```

From Python:

```python
from importlib import import_module
auth = import_module("gee._auth")
ee = auth.init_ee(project="your-gcp-project")   # raises with setup recipe if EE absent

opt = import_module("gee.01_optical_harmonize")  # numbered name → import_module
# or by file path; see notebooks/02_gee_pipeline.ipynb for a guided tour.
```

---

## Graceful degradation (no credentials)

* `import` of every script is **safe without `earthengine-api`** — `import ee`
  is guarded and all `ee.*` calls live inside functions.
* Calling `init_ee()` (or any `main`) without auth raises
  `EarthEngineUnavailable` carrying the full setup recipe — never an opaque
  traceback.
* This is what lets `tests/test_gee.py` compile + import the whole directory
  offline, and what keeps `notebooks/02_gee_pipeline.ipynb` honest about which
  cells need credentials.

The fully **offline** end-to-end demo (synthetic data, no EE) lives in
[`../notebooks/01_quickstart_demo.ipynb`](../notebooks/01_quickstart_demo.ipynb).
