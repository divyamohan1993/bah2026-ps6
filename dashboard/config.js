/* =============================================================================
 * AgriStress Dashboard — configuration
 * ISRO BAH 2026 PS6 · AI-Driven Crop-type, Moisture-Stress & Irrigation Advisory
 * -----------------------------------------------------------------------------
 * Single source of truth for: API endpoints, default Area-of-Interest (AOI),
 * the seasonal 8-day date list, layer definitions, colour ramps and legends.
 *
 * This file is plain ES (no modules / no build step). It attaches a global
 * `AGRI_CONFIG` object consumed by app.js. Everything is overridable at runtime
 * via the URL query string, e.g.  index.html?api=http://localhost:8000
 * ===========================================================================*/
(function () {
  "use strict";

  // --- Runtime overrides via query string ----------------------------------
  const qs = new URLSearchParams(window.location.search);

  // -------------------------------------------------------------------------
  // API — AgriStress FastAPI backend. The dashboard degrades gracefully to the
  // bundled demo/*.geojson files when this base URL is unreachable.
  // Endpoints (see README):
  //   GET /aoi                          -> command-area boundary FeatureCollection
  //   GET /advisory?date=YYYY-MM-DD     -> per-field advisory FeatureCollection
  //   GET /command/{id}/rollup?date=... -> roll-up stats JSON
  //   GET /timeseries?field={fid}       -> NDVI / stress time-series JSON
  //   GET /tiles/{layer}/{z}/{x}/{y}.png-> raster XYZ tiles (crop/stress/advisory)
  // -------------------------------------------------------------------------
  const API_BASE = (qs.get("api") || "http://localhost:8000").replace(/\/+$/, "");

  // Pilot command area (matches AGRISTRESS_DEFAULT_AOI=mula_nira_command).
  // Mula command area, Ahmednagar district, Maharashtra, India.
  const COMMAND_ID = qs.get("command") || "mula_nira_command";

  // -------------------------------------------------------------------------
  // Map defaults — centred on the Mula canal command pilot AOI.
  // -------------------------------------------------------------------------
  const MAP = {
    center: [
      parseFloat(qs.get("lng")) || 74.595, // longitude
      parseFloat(qs.get("lat")) || 19.355, // latitude
    ],
    zoom: parseFloat(qs.get("zoom")) || 12.2,
    minZoom: 4,
    maxZoom: 18,
    pitch: 0,
    bearing: 0,
  };

  // -------------------------------------------------------------------------
  // Season — 8-day compositing steps (FAO-56 / MODIS-style). Kharif 2025.
  // app.js can also adopt a date list returned by the API /timeseries call.
  // -------------------------------------------------------------------------
  function buildSeason(startISO, steps, stepDays) {
    const out = [];
    const d0 = new Date(startISO + "T00:00:00Z");
    for (let i = 0; i < steps; i++) {
      const d = new Date(d0.getTime() + i * stepDays * 86400000);
      out.push(d.toISOString().slice(0, 10));
    }
    return out;
  }
  const SEASON = {
    start: "2025-06-10",
    stepDays: 8,
    steps: 17, // 2025-06-10 .. 2025-10-16  (kharif window)
    label: "Kharif 2025",
    get dates() {
      return buildSeason(this.start, this.steps, this.stepDays);
    },
  };

  // -------------------------------------------------------------------------
  // Crop classes — colour-coded crop-type map.
  // -------------------------------------------------------------------------
  const CROP_CLASSES = [
    { id: "sugarcane", label: "Sugarcane", color: "#7e57c2" },
    { id: "cotton", label: "Cotton", color: "#26c6da" },
    { id: "soybean", label: "Soybean", color: "#9ccc65" },
    { id: "maize", label: "Maize", color: "#ffca28" },
    { id: "wheat", label: "Wheat", color: "#d4a017" },
    { id: "onion", label: "Onion", color: "#ef5350" },
    { id: "fallow", label: "Fallow / bare", color: "#8d8576" },
  ];

  // -------------------------------------------------------------------------
  // Growth stages — phenology (FAO-56 stages).
  // -------------------------------------------------------------------------
  const GROWTH_STAGES = [
    { id: "initial", label: "Initial" },
    { id: "development", label: "Development" },
    { id: "mid", label: "Mid-season" },
    { id: "late", label: "Late-season" },
  ];

  // -------------------------------------------------------------------------
  // Moisture-stress severity ramp (green -> red). 5 ordered classes.
  // -------------------------------------------------------------------------
  const STRESS_CLASSES = [
    { id: "none", label: "No stress", color: "#1a9850" },
    { id: "mild", label: "Mild", color: "#a6d96a" },
    { id: "moderate", label: "Moderate", color: "#fee08b" },
    { id: "severe", label: "Severe", color: "#fc8d59" },
    { id: "extreme", label: "Extreme", color: "#d73027" },
  ];

  // -------------------------------------------------------------------------
  // Irrigation advisory — exactly the 5 mandated classes.
  // Driven by root-zone depletion ratio Dr/RAW and growth-stage sensitivity.
  // -------------------------------------------------------------------------
  const ADVISORY_CLASSES = [
    {
      id: "no_irrigation",
      label: "No irrigation",
      color: "#2c7bb6",
      hint: "Soil water adequate (Dr ≤ 0.5·RAW). No action.",
    },
    {
      id: "watch",
      label: "Watch",
      color: "#abd9e9",
      hint: "Depletion approaching RAW. Monitor next pass.",
    },
    {
      id: "irrigate_soon",
      label: "Irrigate soon",
      color: "#ffffbf",
      hint: "Dr near RAW. Schedule within warabandi turn.",
    },
    {
      id: "irrigate_now",
      label: "Irrigate now",
      color: "#fdae61",
      hint: "Dr > RAW. Apply net irrigation this turn.",
    },
    {
      id: "critical",
      label: "Critical",
      color: "#d7191c",
      hint: "Severe deficit at sensitive stage. Yield loss risk.",
    },
  ];

  // -------------------------------------------------------------------------
  // Layers — toggleable. `kind` drives how app.js renders them.
  //   base      -> raster basemap (satellite)
  //   choropleth-> demo fallback paints the fields GeoJSON by `paintProp`
  //   raster    -> XYZ tiles from the API /tiles/{layer}/...
  // -------------------------------------------------------------------------
  const LAYERS = [
    {
      id: "satellite",
      label: "Base satellite",
      kind: "base",
      legend: "none",
      defaultOn: true,
      // ESRI World Imagery — public XYZ basemap (no key required).
      tiles: [
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      ],
      attribution: "Imagery © Esri, Maxar, Earthstar Geographics",
    },
    {
      id: "crop",
      label: "Crop type",
      kind: "choropleth",
      paintProp: "crop",
      legend: "crop",
      defaultOn: true,
      apiTiles: "/tiles/crop/{z}/{x}/{y}.png",
    },
    {
      id: "stress",
      label: "Moisture stress",
      kind: "choropleth",
      paintProp: "stress_class",
      legend: "stress",
      defaultOn: false,
      timeAware: true,
      apiTiles: "/tiles/stress/{z}/{x}/{y}.png",
    },
    {
      id: "advisory",
      label: "Irrigation advisory",
      kind: "choropleth",
      paintProp: "advisory_class",
      legend: "advisory",
      defaultOn: false,
      timeAware: true,
      apiTiles: "/tiles/advisory/{z}/{x}/{y}.png",
    },
  ];

  // -------------------------------------------------------------------------
  // Helpers shared by app.js — colour lookups + MapLibre match expressions.
  // -------------------------------------------------------------------------
  function asMap(list) {
    const m = {};
    list.forEach((x) => (m[x.id] = x));
    return m;
  }
  const CROP_MAP = asMap(CROP_CLASSES);
  const STRESS_MAP = asMap(STRESS_CLASSES);
  const ADVISORY_MAP = asMap(ADVISORY_CLASSES);
  const STAGE_MAP = asMap(GROWTH_STAGES);

  function matchExpression(prop, list, fallback) {
    const expr = ["match", ["get", prop]];
    list.forEach((x) => expr.push(x.id, x.color));
    expr.push(fallback || "#555555");
    return expr;
  }

  window.AGRI_CONFIG = {
    API_BASE,
    COMMAND_ID,
    MAP,
    SEASON,
    LAYERS,
    CROP_CLASSES,
    GROWTH_STAGES,
    STRESS_CLASSES,
    ADVISORY_CLASSES,
    CROP_MAP,
    STRESS_MAP,
    ADVISORY_MAP,
    STAGE_MAP,
    matchExpression,
    // demo file paths (relative to index.html)
    DEMO: {
      aoi: "demo/command_area.geojson",
      fields: "demo/fields.geojson",
      rollup: "demo/rollup.json",
    },
  };
})();
