# `notebooks/` — AgriStress demo notebooks

Two notebooks for ISRO BAH 2026 Problem Statement 6: one that runs **offline with
no credentials**, and one **guided tour** of the real Earth Engine pipeline.

| Notebook | Credentials | What it shows |
|---|---|---|
| [`01_quickstart_demo.ipynb`](01_quickstart_demo.ipynb) | **None** — fully offline | End-to-end demo on **synthetic** data via `agristress.pipeline.orchestrator.run_demo()` (with an embedded synthetic fallback if the package isn't importable yet): crop map, moisture-stress map, irrigation-advisory map, and an NDVI time-series chart. |
| [`02_gee_pipeline.ipynb`](02_gee_pipeline.ipynb) | **Earth Engine auth** | Guided tour calling the `gee/` scripts — Sentinel-2 + Landsat optical harmonisation, Sentinel-1 SAR, indices/phenology, SMAP/IMERG/CHIRPS/ERA5-Land, crop classification (multi-temporal stack **and** AlphaEarth 64-D embedding, OA + Kappa), FAO-56 advisory — with `geemap` interactive maps and gold-artefact export. |

## Running

```bash
# install Jupyter + the project (editable) and viz extras
pip install -e ".[viz]" jupyter          # adds matplotlib/geemap; agristress importable
jupyter lab    # or: jupyter notebook
```

### `01_quickstart_demo.ipynb` — start here
* Requires only **`numpy`**. `matplotlib` is optional: if absent, charts render
  as compact text/ASCII summaries so the notebook **always completes**.
* Tries `agristress.pipeline.orchestrator.run_demo()` first; if that module is
  not yet importable, it transparently falls back to an embedded synthetic
  pipeline returning the same result contract.
* No Earth Engine, no network, no API keys.

### `02_gee_pipeline.ipynb` — needs Earth Engine
* First run **`earthengine authenticate`** (or set a service account) — see
  [`../gee/README.md`](../gee/README.md). Set `EE_PROJECT` (and optionally
  `EE_ASSET_PREFIX`).
* Every EE-touching cell is marked **[needs EE]** and is guarded: if EE is not
  initialised, the cell prints the setup recipe and is skipped rather than
  crashing — so you can read the whole flow without credentials.
* `geemap` is optional for the interactive-map cell.

> Notebooks under `notebooks/` are intentionally excluded from `ruff`/`black` in
> `pyproject.toml`; they are demonstration artefacts, not library code.
