# AgriStress

**AI-driven crop-type classification, phenology-aware moisture-stress detection, and 8-day irrigation advisory for canal command areas.**

> ISRO **Bharatiya Antariksh Hackathon (BAH) 2026 — Problem Statement 6**
> Multi-sensor **optical + microwave** fusion across **40+ satellites** (LISS-IV/III, AWiFS, Sentinel-2, Landsat, MODIS, EOS-04, Sentinel-1, NISAR-SAR, …) that cross-verify and gap-fill each other, served on an **O(1) read-hot-path** platform.

AgriStress ingests multi-temporal optical and SAR observations, fuses them into an analysis-ready
datacube, runs three analytic heads (crop type, growth-stage phenology, moisture stress), and turns
crop water deficit into pixel/field-level irrigation advisory maps for irrigation command areas. The
whole pipeline runs **fully offline on synthetic data** (no cloud credentials required) so it can be
demoed and tested anywhere.

## Documentation

The authoritative blueprint and its companion deep-dives live in [`docs/`](./docs):

- [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md) — master system architecture (start here)
- [`docs/SATELLITE_CATALOG.md`](./docs/SATELLITE_CATALOG.md) — the 40+ satellite fleet & gap-filling matrix
- [`docs/DATA_FUSION.md`](./docs/DATA_FUSION.md) — the 6-stage optical/SAR data-fusion pipeline
- [`docs/PLATFORM_O1.md`](./docs/PLATFORM_O1.md) — O(1) platform, H3 indexing & serving
- [`docs/MODELS.md`](./docs/MODELS.md) — crop-type, phenology & moisture-stress models
- [`docs/IRRIGATION_ADVISORY.md`](./docs/IRRIGATION_ADVISORY.md) — FAO-56 ET0 / water-balance advisory engine

## Quickstart

Requires Python **3.11+**. Install the package with the development/test extras (pure-wheel science
stack — no system GDAL required):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Run the test suite and linters:

```bash
pytest -q
ruff check src tests
ruff format --check src tests
```

Run the end-to-end demo on synthetic data (writes crop / stress / advisory maps to `outputs/`):

```bash
agristress catalog     # print the satellite/sensor registry
agristress demo        # run the full pipeline offline (no credentials)
agristress serve       # launch the FastAPI serving app (needs the [serving] extra)
```

## Optional extras

Heavier or environment-specific stacks are opt-in so the core install always succeeds:

| Extra        | Purpose |
|--------------|---------|
| `[ml]`       | scikit-learn / scikit-image / joblib model stack (xgboost/lightgbm optional, auto-fallback) |
| `[serving]`  | FastAPI + Uvicorn + Typer CLI + Pillow tiling (Redis optional) |
| `[viz]`      | matplotlib / plotly / folium |
| `[geo]`      | native-geo stack (rasterio, rioxarray, geopandas, shapely, pyproj) — needs system GDAL/PROJ |
| `[cloud]`    | Earth Engine / STAC access (earthengine-api, geemap, pystac-client, odc-stac, stackstac, planetary-computer) |
| `[dl]`       | optional deep-learning backbones (torch, timm) |
| `[all]`      | everything above |

```bash
pip install -e ".[all]"   # full install (requires system GDAL for the [geo] stack)
```

## Repository layout

```
src/agristress/      # library: catalog, ingestion, preprocessing, fusion, indexing,
                     #          features, models, irrigation, serving, pipeline
tests/               # pytest suite (offline; heavy-optional tests skip gracefully)
docs/                # architecture + companion design docs
gee/                 # Google Earth Engine scripts (optical/SAR/indices/advisory)
notebooks/           # quickstart & GEE-pipeline notebooks
dashboard/           # front-end serving artifacts
configs/             # pipeline / AOI configuration
```

## License

Apache-2.0.
