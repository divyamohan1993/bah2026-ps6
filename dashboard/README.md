# AgriStress Dashboard

Web dashboard for **ISRO BAH 2026 · Problem Statement 6** — *AI-Driven Crop-type,
Moisture-Stress Detection and Irrigation Advisory across growth stages*.

A framework-light **MapLibre GL JS** app that visualises crop-type,
phenology-aware moisture-stress and 8-day irrigation-advisory layers for a canal
command area (pilot: **Mula Command Area, Ahmednagar, Maharashtra**), with a
season time slider, click-to-inspect field popups (with an NDVI / Dr-RAW
time-series chart) and a command-area roll-up card.

No build step. Vanilla HTML / CSS / JS + MapLibre GL from CDN. The UI is fully
interactive **offline** using bundled demo data, and automatically upgrades to a
live backend when the AgriStress API is reachable.

---

## Run

### Option A — just open it
Open `index.html` in a modern browser. The dashboard loads the bundled demo data
and is fully interactive.

> Note: some browsers block `fetch()` of local `demo/*.geojson` files under the
> `file://` protocol (CORS). If layers don't appear, use Option B.

### Option B — serve over HTTP (recommended)
```bash
cd dashboard
python -m http.server 8080
# then open http://localhost:8080
```

### Point at the live API
The dashboard tries the AgriStress API first and falls back to demo data if it is
unreachable. Set the API base URL via the query string:
```
http://localhost:8080/?api=http://localhost:8000
```
Other handy query params: `?command=<id>`, `?lat=&lng=&zoom=`.
Defaults live in `config.js` (`API_BASE` defaults to `http://localhost:8000`).

A **● DEMO DATA** / **● LIVE API** badge in the top-right shows the active source.

---

## API contract (consumed by `app.js`)

| Endpoint | Returns |
|---|---|
| `GET /aoi?command={id}` | Command-area boundary `FeatureCollection` |
| `GET /advisory?command={id}` | Per-field `FeatureCollection` (props below) |
| `GET /command/{id}/rollup` | Roll-up stats JSON (`by_date[]`, see `demo/rollup.json`) |
| `GET /timeseries?field={fid}` | Optional per-field NDVI / stress series |
| `GET /tiles/{layer}/{z}/{x}/{y}.png` | Optional raster XYZ tiles (`crop`/`stress`/`advisory`) |

**Field feature properties** (see `demo/fields.geojson`):
`field_id, village, area_ha, crop, stage, ndvi, stress_class, advisory_class,
dr, raw` plus full-season arrays `ndvi_series, stress_series, advisory_series,
stage_series, dr_series, raw_series` (length = number of 8-day steps). The arrays
drive the time slider and the popup chart; the scalar fields are the snapshot for
the currently selected date.

---

## Layers & legend

| Layer | What it shows | Legend |
|---|---|---|
| **Base satellite** | ESRI World Imagery basemap (no key) | — |
| **Crop type** | Supervised crop-type classification | per-crop colours |
| **Moisture stress** | Phenology-aware stress severity (8-day) | green → red ramp |
| **Irrigation advisory** | 8-day advisory from root-zone depletion | 5-class scale |

**Crop classes:** Sugarcane · Cotton · Soybean · Maize · Wheat · Onion · Fallow.

**Stress severity ramp (green→red):** No stress → Mild → Moderate → Severe → Extreme.

**Irrigation advisory (5 mandated classes):**

| Class | Meaning |
|---|---|
| No irrigation | Soil water adequate (`Dr ≤ 0.5·RAW`). |
| Watch | Depletion approaching RAW; monitor next pass. |
| Irrigate soon | `Dr` near `RAW`; schedule within the warabandi turn. |
| Irrigate now | `Dr > RAW`; apply net irrigation this turn. |
| Critical | Severe deficit at a sensitive stage; yield-loss risk. |

The advisory is derived from the FAO-56 root-zone water balance: **readily
available water** `RAW = p · TAW` and **root-zone depletion** `Dr`. The class is a
function of `Dr/RAW` escalated by growth-stage sensitivity (mid-season is the most
sensitive). The roll-up card reports the % area per advisory class, the **gross
irrigation demand** (net depth ÷ 65 % application efficiency, summed over fields)
and an indicative **warabandi turn-time**.

---

## Files

```
dashboard/
├── index.html          # layout, CDN MapLibre, panels, time bar
├── app.js              # map init, layer manager, API+demo fallback, popups, chart, legend, roll-up
├── config.js           # API base URL, AOI centre/zoom, colour ramps, season date list
├── style.css           # dark, responsive map UI
├── demo/
│   ├── command_area.geojson   # command-area boundary
│   ├── fields.geojson         # ~60 field polygons with season series props
│   ├── rollup.json            # per-date roll-up stats
│   └── _generate.py           # (dev) regenerates the demo data
└── README.md
```

Regenerate demo data (optional): `cd dashboard/demo && python _generate.py`.

---

## Screenshots

_TODO: add screenshots once captured —_
- `docs/dashboard-crop.png` — crop-type layer + legend
- `docs/dashboard-stress.png` — mid-season moisture-stress peak
- `docs/dashboard-advisory.png` — advisory layer + field popup with NDVI/Dr-RAW chart
- `docs/dashboard-rollup.png` — command-area roll-up card

---

## Notes
- Pure client-side; no secrets. Configure the API endpoint at runtime via `?api=`.
- Demo boundary and field data are **illustrative** (synthetic but physically
  plausible) and are for offline UI demonstration only.
