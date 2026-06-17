/* =============================================================================
 * AgriStress Dashboard — application logic
 * ISRO BAH 2026 PS6
 * -----------------------------------------------------------------------------
 * Framework-light MapLibre GL JS app. Responsibilities:
 *   - map init (satellite basemap + command AOI + field polygons)
 *   - layer manager: fetch from AgriStress API, FALL BACK to bundled demo data
 *   - 8-day time slider that re-paints time-aware layers & the roll-up card
 *   - click popup with crop / stage / stress / Dr-RAW / recommendation
 *   - a small canvas time-series chart of NDVI + root-zone depletion
 *   - dynamic, per-layer legend
 * No build step; relies on globals: maplibregl, AGRI_CONFIG.
 * ===========================================================================*/
(function () {
  "use strict";

  const CFG = window.AGRI_CONFIG;
  const DATES = CFG.SEASON.dates;
  const $ = (sel) => document.querySelector(sel);
  const el = (tag, cls, html) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html != null) e.innerHTML = html;
    return e;
  };

  // ---- Application state ---------------------------------------------------
  const state = {
    map: null,
    fields: null, // GeoJSON FeatureCollection (with *_series props)
    aoi: null, // GeoJSON FeatureCollection
    rollup: null, // { by_date:[...], dates:[...] }
    timeIndex: DATES.length - 1, // current 8-day step
    activeChoropleth: "crop", // which choropleth drives the legend / paint
    dataSource: "demo", // "api" | "demo"
    layerOn: {}, // id -> bool
    selectedField: null,
    playTimer: null,
  };
  CFG.LAYERS.forEach((l) => (state.layerOn[l.id] = !!l.defaultOn));

  // =========================================================================
  // Networking — try the API, fall back to bundled demo files.
  // =========================================================================
  async function fetchJSON(url, opts) {
    const ctl = new AbortController();
    const to = setTimeout(() => ctl.abort(), (opts && opts.timeout) || 3500);
    try {
      const res = await fetch(url, { signal: ctl.signal });
      if (!res.ok) throw new Error("HTTP " + res.status);
      return await res.json();
    } finally {
      clearTimeout(to);
    }
  }

  async function loadData() {
    const base = CFG.API_BASE;
    // Try the live API first; any failure -> demo bundle.
    try {
      const [aoi, fields] = await Promise.all([
        fetchJSON(base + "/aoi?command=" + encodeURIComponent(CFG.COMMAND_ID)),
        fetchJSON(base + "/advisory?command=" + encodeURIComponent(CFG.COMMAND_ID)),
      ]);
      // Basic shape validation.
      if (!fields || fields.type !== "FeatureCollection") throw new Error("bad /advisory");
      state.aoi = aoi;
      state.fields = fields;
      state.dataSource = "api";
      // roll-up is optional; compute locally if endpoint missing.
      try {
        state.rollup = await fetchJSON(
          base + "/command/" + encodeURIComponent(CFG.COMMAND_ID) + "/rollup"
        );
      } catch (_) {
        state.rollup = null;
      }
    } catch (apiErr) {
      // ---- graceful degradation to bundled demo data ----
      const [aoi, fields, rollup] = await Promise.all([
        fetchJSON(CFG.DEMO.aoi),
        fetchJSON(CFG.DEMO.fields),
        fetchJSON(CFG.DEMO.rollup).catch(() => null),
      ]);
      state.aoi = aoi;
      state.fields = fields;
      state.rollup = rollup;
      state.dataSource = "demo";
    }

    // Adopt the API/demo season date list when present (keeps slider honest).
    if (state.fields && Array.isArray(state.fields.dates) && state.fields.dates.length) {
      DATES.length = 0;
      state.fields.dates.forEach((d) => DATES.push(d));
      state.timeIndex = DATES.length - 1;
    }
  }

  // =========================================================================
  // Time-awareness — copy the per-field series value at the current step into
  // the live properties the map paints + popups read.
  // =========================================================================
  function applyTimeStep(idx) {
    state.timeIndex = idx;
    const feats = (state.fields && state.fields.features) || [];
    for (const f of feats) {
      const p = f.properties;
      if (Array.isArray(p.advisory_series) && idx < p.advisory_series.length)
        p.advisory_class = p.advisory_series[idx];
      if (Array.isArray(p.stress_series) && idx < p.stress_series.length)
        p.stress_class = p.stress_series[idx];
      if (Array.isArray(p.stage_series) && idx < p.stage_series.length)
        p.stage = p.stage_series[idx];
      if (Array.isArray(p.ndvi_series) && idx < p.ndvi_series.length)
        p.ndvi = p.ndvi_series[idx];
      if (Array.isArray(p.dr_series) && idx < p.dr_series.length) p.dr = p.dr_series[idx];
      if (Array.isArray(p.raw_series) && idx < p.raw_series.length) p.raw = p.raw_series[idx];
    }
    const src = state.map && state.map.getSource("fields");
    if (src) src.setData(state.fields);
    updateDateLabel();
    updateRollup();
    if (state.selectedField) renderPopupChart(state.selectedField);
  }

  // =========================================================================
  // Roll-up computation (local fallback) + card rendering.
  // =========================================================================
  function computeRollupAt(idx) {
    // Prefer server/demo precomputed roll-up if available for this date.
    if (state.rollup && Array.isArray(state.rollup.by_date)) {
      const byDate = state.rollup.by_date;
      const target = DATES[idx];
      const hit = byDate.find((r) => r.date === target) || byDate[Math.min(idx, byDate.length - 1)];
      if (hit) return hit;
    }
    // Compute from field features.
    const feats = (state.fields && state.fields.features) || [];
    const counts = {};
    const areaBy = {};
    CFG.ADVISORY_CLASSES.forEach((a) => {
      counts[a.id] = 0;
      areaBy[a.id] = 0;
    });
    let totalArea = 0;
    let grossM3 = 0;
    for (const f of feats) {
      const p = f.properties;
      const adv = p.advisory_class || "no_irrigation";
      const ha = +p.area_ha || 1;
      counts[adv] = (counts[adv] || 0) + 1;
      areaBy[adv] = (areaBy[adv] || 0) + ha;
      totalArea += ha;
      const dr = +p.dr || 0;
      const raw = +p.raw || 1;
      let netMm = 0;
      if (adv === "irrigate_now" || adv === "critical") netMm = dr;
      else if (adv === "irrigate_soon") netMm = Math.max(0, dr - raw * 0.6);
      grossM3 += (netMm / 0.65) * 1e-3 * ha * 1e4;
    }
    const n = feats.length || 1;
    const pct = {};
    Object.keys(counts).forEach((k) => (pct[k] = Math.round((1000 * counts[k]) / n) / 10));
    return {
      date: DATES[idx],
      n_fields: feats.length,
      total_area_ha: Math.round(totalArea * 10) / 10,
      advisory_counts: counts,
      advisory_pct: pct,
      advisory_area_ha: areaBy,
      gross_demand_m3: Math.round(grossM3),
      gross_demand_ham: Math.round((grossM3 / 1e4) * 10) / 10,
      warabandi_cycle_days: (state.rollup && state.rollup.warabandi_cycle_days) || 7,
      warabandi_turn_min_per_ha: (state.rollup && state.rollup.warabandi_turn_min_per_ha) || 38,
    };
  }

  function fmt(n, d) {
    if (n == null || isNaN(n)) return "–";
    return Number(n).toLocaleString("en-IN", { maximumFractionDigits: d == null ? 0 : d });
  }

  function updateRollup() {
    const r = computeRollupAt(state.timeIndex);
    const card = $("#rollup-body");
    if (!card) return;

    // Stacked advisory composition bar.
    const bar = el("div", "stackbar");
    CFG.ADVISORY_CLASSES.forEach((a) => {
      const pct = (r.advisory_pct && r.advisory_pct[a.id]) || 0;
      if (pct <= 0) return;
      const seg = el("span", "stackbar-seg");
      seg.style.width = pct + "%";
      seg.style.background = a.color;
      seg.title = `${a.label}: ${pct}%`;
      bar.appendChild(seg);
    });

    // Per-class rows.
    const rows = el("div", "rollup-rows");
    CFG.ADVISORY_CLASSES.forEach((a) => {
      const pct = (r.advisory_pct && r.advisory_pct[a.id]) || 0;
      const ha = (r.advisory_area_ha && r.advisory_area_ha[a.id]) || 0;
      const row = el("div", "rollup-row");
      row.innerHTML =
        `<span class="sw" style="background:${a.color}"></span>` +
        `<span class="rl-label">${a.label}</span>` +
        `<span class="rl-pct">${fmt(pct, 1)}%</span>` +
        `<span class="rl-ha">${fmt(ha, 0)} ha</span>`;
      rows.appendChild(row);
    });

    // Demand + warabandi metrics.
    const ham = r.gross_demand_ham != null ? r.gross_demand_ham : r.gross_demand_m3 / 1e4;
    const metrics = el("div", "rollup-metrics");
    metrics.innerHTML =
      metricHTML("Gross irrigation demand", fmt(ham, 1) + " ha·m", `${fmt(r.gross_demand_m3)} m³`) +
      metricHTML(
        "Warabandi turn-time",
        fmt(r.warabandi_turn_min_per_ha) + " min/ha",
        `${fmt(r.warabandi_cycle_days)}-day rotation`
      ) +
      metricHTML("Command extent", fmt(r.total_area_ha) + " ha", `${fmt(r.n_fields)} fields`);

    card.innerHTML = "";
    card.appendChild(bar);
    card.appendChild(rows);
    card.appendChild(metrics);
  }

  function metricHTML(label, big, sub) {
    return (
      `<div class="metric"><div class="metric-label">${label}</div>` +
      `<div class="metric-val">${big}</div><div class="metric-sub">${sub}</div></div>`
    );
  }

  // =========================================================================
  // Map + layers.
  // =========================================================================
  function buildBaseStyle() {
    const baseLayer = CFG.LAYERS.find((l) => l.kind === "base");
    return {
      version: 8,
      glyphs: "https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf",
      sources: {
        satellite: {
          type: "raster",
          tiles: baseLayer.tiles,
          tileSize: 256,
          attribution: baseLayer.attribution,
          maxzoom: 19,
        },
      },
      layers: [
        { id: "bg", type: "background", paint: { "background-color": "#0b1622" } },
        {
          id: "satellite",
          type: "raster",
          source: "satellite",
          layout: { visibility: state.layerOn.satellite ? "visible" : "none" },
          paint: { "raster-opacity": 0.92 },
        },
      ],
    };
  }

  function choroplethPaint(layerId) {
    const layer = CFG.LAYERS.find((l) => l.id === layerId);
    let list;
    if (layer.legend === "crop") list = CFG.CROP_CLASSES;
    else if (layer.legend === "stress") list = CFG.STRESS_CLASSES;
    else list = CFG.ADVISORY_CLASSES;
    return CFG.matchExpression(layer.paintProp, list, "#666");
  }

  function addDataLayers() {
    const map = state.map;

    // AOI source + outline + faint label halo.
    if (state.aoi) {
      map.addSource("aoi", { type: "geojson", data: state.aoi });
      map.addLayer({
        id: "aoi-fill",
        type: "fill",
        source: "aoi",
        paint: { "fill-color": "#34d3ff", "fill-opacity": 0.04 },
      });
      map.addLayer({
        id: "aoi-line",
        type: "line",
        source: "aoi",
        paint: {
          "line-color": "#46e0ff",
          "line-width": 2.2,
          "line-dasharray": [2, 1.5],
        },
      });
    }

    // Fields source.
    map.addSource("fields", { type: "geojson", data: state.fields });

    // One fill layer per choropleth layer; visibility toggled by the manager.
    CFG.LAYERS.filter((l) => l.kind === "choropleth").forEach((layer) => {
      map.addLayer({
        id: "fill-" + layer.id,
        type: "fill",
        source: "fields",
        layout: { visibility: state.layerOn[layer.id] ? "visible" : "none" },
        paint: {
          "fill-color": choroplethPaint(layer.id),
          "fill-opacity": layer.id === "crop" ? 0.62 : 0.75,
          "fill-outline-color": "rgba(7,16,26,0.55)",
        },
      });
    });

    // Field outlines (always on, subtle) + selection highlight.
    map.addLayer({
      id: "fields-line",
      type: "line",
      source: "fields",
      paint: { "line-color": "rgba(10,18,28,0.7)", "line-width": 0.6 },
    });
    map.addLayer({
      id: "fields-selected",
      type: "line",
      source: "fields",
      paint: { "line-color": "#ffffff", "line-width": 2.6 },
      filter: ["==", ["get", "field_id"], "___none___"],
    });

    // Interactions: pointer + click popup on the top-most visible choropleth.
    const clickable = CFG.LAYERS.filter((l) => l.kind === "choropleth").map((l) => "fill-" + l.id);
    map.on("mousemove", (e) => {
      const hits = map.queryRenderedFeatures(e.point, { layers: clickable });
      map.getCanvas().style.cursor = hits.length ? "pointer" : "";
    });
    map.on("click", (e) => {
      const hits = map.queryRenderedFeatures(e.point, { layers: clickable });
      if (!hits.length) return;
      onFieldClick(hits[0], e.lngLat);
    });

    applyChoroplethVisibility();
    applyTimeStep(state.timeIndex);
  }

  function applyChoroplethVisibility() {
    const map = state.map;
    // Determine the active legend layer = top-most enabled choropleth.
    const order = ["advisory", "stress", "crop"]; // top to bottom priority
    let active = null;
    CFG.LAYERS.filter((l) => l.kind === "choropleth").forEach((layer) => {
      const v = state.layerOn[layer.id] ? "visible" : "none";
      if (map.getLayer("fill-" + layer.id)) map.setLayoutProperty("fill-" + layer.id, "visibility", v);
    });
    for (const id of order) {
      if (state.layerOn[id]) {
        active = id;
        break;
      }
    }
    if (active) state.activeChoropleth = active;
    if (map.getLayer("satellite"))
      map.setLayoutProperty(
        "satellite",
        "visibility",
        state.layerOn.satellite ? "visible" : "none"
      );
    renderLegend();
  }

  // =========================================================================
  // Click -> popup with details + chart.
  // =========================================================================
  function onFieldClick(feature, lngLat) {
    const p = feature.properties;
    state.selectedField = p.field_id;
    state.map.setFilter("fields-selected", ["==", ["get", "field_id"], p.field_id]);

    const crop = CFG.CROP_MAP[p.crop] || { label: p.crop, color: "#999" };
    const stress = CFG.STRESS_MAP[p.stress_class] || { label: p.stress_class, color: "#999" };
    const adv = CFG.ADVISORY_MAP[p.advisory_class] || { label: p.advisory_class, color: "#999" };
    const stage = CFG.STAGE_MAP[p.stage] || { label: p.stage };
    const dr = +p.dr,
      raw = +p.raw;
    const depl = raw > 0 ? Math.min(1.5, dr / raw) : 0;

    const html =
      `<div class="popup">` +
      `<div class="popup-head">` +
      `<div class="popup-id">${p.field_id || "Field"}</div>` +
      `<div class="popup-sub">${p.village || ""} · ${fmt(p.area_ha, 2)} ha</div>` +
      `</div>` +
      `<div class="popup-grid">` +
      pill("Crop", crop.label, crop.color) +
      pill("Growth stage", stage.label, "#5b8") +
      pill("Stress", stress.label, stress.color) +
      pill("Advisory", adv.label, adv.color) +
      `</div>` +
      `<div class="popup-soil">` +
      `<div class="soil-row"><span>Root-zone depletion Dr</span><b>${fmt(dr, 1)} mm</b></div>` +
      `<div class="soil-row"><span>Readily avail. water RAW</span><b>${fmt(raw, 1)} mm</b></div>` +
      `<div class="depl-bar"><span style="width:${Math.min(100, depl * 66.6).toFixed(
        0
      )}%;background:${adv.color}"></span><i style="left:66.6%"></i></div>` +
      `<div class="depl-cap">Dr/RAW = ${depl.toFixed(2)} ${
        depl >= 1 ? "→ deficit (irrigate)" : "→ within readily-available water"
      }</div>` +
      `</div>` +
      `<div class="popup-reco" style="border-color:${adv.color}">` +
      `<b style="color:${adv.color}">${adv.label}</b> — ${recommendation(p, adv)}` +
      `</div>` +
      `<div class="popup-chart"><div class="chart-title">NDVI &amp; root-zone depletion · ${CFG.SEASON.label}</div>` +
      `<canvas id="ts-canvas" width="300" height="120"></canvas>` +
      `<div class="chart-legend"><span class="ck ndvi">NDVI</span><span class="ck dr">Dr/RAW</span><span class="ck now">selected date</span></div>` +
      `</div>` +
      `</div>`;

    if (state._popup) state._popup.remove();
    state._popup = new maplibregl.Popup({
      closeButton: true,
      maxWidth: "340px",
      className: "agri-popup",
    })
      .setLngLat(lngLat)
      .setHTML(html)
      .addTo(state.map);
    state._popup.on("close", () => {
      state.selectedField = null;
      state.map.setFilter("fields-selected", ["==", ["get", "field_id"], "___none___"]);
    });
    renderPopupChart(p.field_id);
  }

  function recommendation(p, adv) {
    const reco = {
      no_irrigation: "Soil moisture adequate; defer irrigation and reassess next 8-day pass.",
      watch: "Depletion approaching RAW; keep on watch and confirm with next SAR/optical pass.",
      irrigate_soon: `Schedule irrigation within the current warabandi turn (~${fmt(
        Math.max(10, p.dr - p.raw * 0.6),
        0
      )} mm net).`,
      irrigate_now: `Apply ~${fmt(p.dr, 0)} mm net (${fmt(
        p.dr / 0.65,
        0
      )} mm gross @65% eff.) this turn to refill the root zone.`,
      critical: `Severe deficit at a sensitive stage — prioritise this field; apply ~${fmt(
        p.dr,
        0
      )} mm net immediately to limit yield loss.`,
    };
    return reco[adv.id] || "";
  }

  function pill(label, value, color) {
    return (
      `<div class="pp"><div class="pp-l">${label}</div>` +
      `<div class="pp-v"><span class="sw" style="background:${color}"></span>${value}</div></div>`
    );
  }

  // ---- tiny canvas time-series chart (no chart lib dependency) -------------
  function renderPopupChart(fieldId) {
    const canvas = document.getElementById("ts-canvas");
    if (!canvas) return;
    const feat = (state.fields.features || []).find((f) => f.properties.field_id === fieldId);
    if (!feat) return;
    const p = feat.properties;
    const ndvi = p.ndvi_series || [];
    const dr = p.dr_series || [];
    const raw = p.raw_series || [];
    const n = ndvi.length || (dr.length ? dr.length : DATES.length);
    if (!n) return;

    const ctx = canvas.getContext("2d");
    const W = canvas.width,
      H = canvas.height;
    const padL = 6,
      padR = 6,
      padT = 8,
      padB = 14;
    const plotW = W - padL - padR,
      plotH = H - padT - padB;
    ctx.clearRect(0, 0, W, H);

    const x = (i) => padL + (plotW * i) / Math.max(1, n - 1);
    const yN = (v) => padT + plotH * (1 - Math.max(0, Math.min(1, v))); // NDVI 0..1
    const ratio = dr.map((d, i) => (raw[i] > 0 ? d / raw[i] : 0));
    const rMax = Math.max(1.2, ...ratio);
    const yR = (v) => padT + plotH * (1 - Math.max(0, Math.min(rMax, v)) / rMax);

    // gridlines
    ctx.strokeStyle = "rgba(255,255,255,0.08)";
    ctx.lineWidth = 1;
    for (let g = 0; g <= 4; g++) {
      const yy = padT + (plotH * g) / 4;
      ctx.beginPath();
      ctx.moveTo(padL, yy);
      ctx.lineTo(W - padR, yy);
      ctx.stroke();
    }
    // RAW = 1.0 threshold line (Dr/RAW)
    ctx.strokeStyle = "rgba(255,120,120,0.45)";
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(padL, yR(1));
    ctx.lineTo(W - padR, yR(1));
    ctx.stroke();
    ctx.setLineDash([]);

    // current date marker
    const cx = x(Math.min(state.timeIndex, n - 1));
    ctx.strokeStyle = "rgba(255,255,255,0.55)";
    ctx.beginPath();
    ctx.moveTo(cx, padT);
    ctx.lineTo(cx, H - padB);
    ctx.stroke();

    // Dr/RAW area + line (amber)
    ctx.beginPath();
    ratio.forEach((v, i) => (i ? ctx.lineTo(x(i), yR(v)) : ctx.moveTo(x(i), yR(v))));
    ctx.strokeStyle = "#fdae61";
    ctx.lineWidth = 1.8;
    ctx.stroke();

    // NDVI line (green)
    ctx.beginPath();
    ndvi.forEach((v, i) => (i ? ctx.lineTo(x(i), yN(v)) : ctx.moveTo(x(i), yN(v))));
    ctx.strokeStyle = "#5fd17a";
    ctx.lineWidth = 2;
    ctx.stroke();

    // current-date dots
    if (ndvi[state.timeIndex] != null) {
      ctx.fillStyle = "#5fd17a";
      ctx.beginPath();
      ctx.arc(cx, yN(ndvi[state.timeIndex]), 2.6, 0, 7);
      ctx.fill();
    }
    if (ratio[state.timeIndex] != null) {
      ctx.fillStyle = "#fdae61";
      ctx.beginPath();
      ctx.arc(cx, yR(ratio[state.timeIndex]), 2.6, 0, 7);
      ctx.fill();
    }
  }

  // =========================================================================
  // Legend — re-rendered when the active layer changes.
  // =========================================================================
  function renderLegend() {
    const box = $("#legend-body");
    const title = $("#legend-title");
    if (!box) return;
    const layer = CFG.LAYERS.find((l) => l.id === state.activeChoropleth) || CFG.LAYERS[1];
    let list, sub;
    if (layer.legend === "crop") {
      list = CFG.CROP_CLASSES;
      sub = "Supervised crop-type classification";
    } else if (layer.legend === "stress") {
      list = CFG.STRESS_CLASSES;
      sub = "Phenology-aware moisture-stress severity";
    } else {
      list = CFG.ADVISORY_CLASSES;
      sub = "8-day irrigation advisory (Dr vs RAW)";
    }
    if (title) title.textContent = layer.label + " legend";
    box.innerHTML = "";
    const subEl = el("div", "legend-sub", sub);
    box.appendChild(subEl);
    list.forEach((item) => {
      const row = el("div", "legend-row");
      row.innerHTML =
        `<span class="sw" style="background:${item.color}"></span>` +
        `<span class="lg-label">${item.label}</span>` +
        (item.hint ? `<span class="lg-hint" title="${item.hint}">?</span>` : "");
      box.appendChild(row);
    });
  }

  // =========================================================================
  // UI wiring — layer toggles, time slider, play button.
  // =========================================================================
  function buildLayerToggles() {
    const box = $("#layer-toggles");
    box.innerHTML = "";
    CFG.LAYERS.forEach((layer) => {
      const row = el("label", "toggle");
      const cb = el("input");
      cb.type = "checkbox";
      cb.checked = !!state.layerOn[layer.id];
      cb.addEventListener("change", () => {
        state.layerOn[layer.id] = cb.checked;
        applyChoroplethVisibility();
      });
      const sw = el("span", "toggle-sw");
      const txt = el("span", "toggle-label", layer.label);
      const tag = el("span", "toggle-tag", layer.timeAware ? "8-day" : layer.kind === "base" ? "base" : "");
      row.appendChild(cb);
      row.appendChild(sw);
      row.appendChild(txt);
      if (tag.textContent) row.appendChild(tag);
      box.appendChild(row);
    });
  }

  function updateDateLabel() {
    const lbl = $("#date-label");
    const idxLbl = $("#step-label");
    if (lbl) lbl.textContent = DATES[state.timeIndex] || "—";
    if (idxLbl) idxLbl.textContent = `step ${state.timeIndex + 1}/${DATES.length}`;
  }

  function buildTimeSlider() {
    const slider = $("#time-slider");
    slider.min = 0;
    slider.max = DATES.length - 1;
    slider.step = 1;
    slider.value = state.timeIndex;
    slider.addEventListener("input", () => applyTimeStep(+slider.value));

    const play = $("#play-btn");
    play.addEventListener("click", () => {
      if (state.playTimer) {
        clearInterval(state.playTimer);
        state.playTimer = null;
        play.textContent = "▶";
        play.classList.remove("playing");
        return;
      }
      play.textContent = "⏸";
      play.classList.add("playing");
      state.playTimer = setInterval(() => {
        let next = state.timeIndex + 1;
        if (next >= DATES.length) next = 0;
        slider.value = next;
        applyTimeStep(next);
      }, 900);
    });
  }

  function setStatus() {
    const badge = $("#data-badge");
    if (!badge) return;
    if (state.dataSource === "api") {
      badge.textContent = "● LIVE API";
      badge.className = "badge live";
      badge.title = "Connected to AgriStress API at " + CFG.API_BASE;
    } else {
      badge.textContent = "● DEMO DATA";
      badge.className = "badge demo";
      badge.title =
        "API unreachable at " + CFG.API_BASE + " — showing bundled offline demo data.";
    }
    const apiLbl = $("#api-base");
    if (apiLbl) apiLbl.textContent = CFG.API_BASE;
  }

  // =========================================================================
  // Boot.
  // =========================================================================
  async function init() {
    buildLayerToggles();
    buildTimeSlider();
    updateDateLabel();

    // Load data (API or demo) BEFORE adding map layers.
    try {
      await loadData();
    } catch (e) {
      console.error("Data load failed entirely:", e);
      const card = $("#rollup-body");
      if (card) card.innerHTML = `<div class="err">Could not load demo data. Serve the dashboard over HTTP (see README).</div>`;
    }
    setStatus();
    // Re-sync slider bounds in case the API changed the date list.
    const slider = $("#time-slider");
    slider.max = DATES.length - 1;
    slider.value = state.timeIndex;

    const map = new maplibregl.Map({
      container: "map",
      style: buildBaseStyle(),
      center: CFG.MAP.center,
      zoom: CFG.MAP.zoom,
      minZoom: CFG.MAP.minZoom,
      maxZoom: CFG.MAP.maxZoom,
      attributionControl: false,
    });
    state.map = map;
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "bottom-right");
    map.addControl(
      new maplibregl.AttributionControl({ compact: true }),
      "bottom-right"
    );
    map.addControl(new maplibregl.ScaleControl({ maxWidth: 120, unit: "metric" }), "bottom-left");

    map.on("load", () => {
      if (state.fields) addDataLayers();
      updateRollup();
      renderLegend();
      // Fit to AOI if present.
      try {
        if (state.aoi && state.aoi.features && state.aoi.features.length) {
          const b = geojsonBounds(state.aoi);
          if (b) map.fitBounds(b, { padding: 60, duration: 0 });
        }
      } catch (_) {}
    });
  }

  function geojsonBounds(fc) {
    let minX = 180,
      minY = 90,
      maxX = -180,
      maxY = -90,
      seen = false;
    const eat = (coords) => {
      if (typeof coords[0] === "number") {
        seen = true;
        minX = Math.min(minX, coords[0]);
        maxX = Math.max(maxX, coords[0]);
        minY = Math.min(minY, coords[1]);
        maxY = Math.max(maxY, coords[1]);
      } else coords.forEach(eat);
    };
    (fc.features || []).forEach((f) => f.geometry && eat(f.geometry.coordinates));
    return seen ? [[minX, minY], [maxX, maxY]] : null;
  }

  if (document.readyState === "loading")
    document.addEventListener("DOMContentLoaded", init);
  else init();
})();
