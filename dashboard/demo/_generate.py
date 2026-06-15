#!/usr/bin/env python3
"""One-shot generator for AgriStress dashboard demo GeoJSON / JSON.

Produces physically-plausible, phenology-aware demo data for ~48 fields in the
Mula canal command pilot AOI so the dashboard is fully interactive offline.
This script is a build helper; the dashboard ships only its JSON outputs.
"""
from __future__ import annotations

import json
import math
import random
from datetime import date, timedelta

random.seed(42)

# ---- Season (must mirror config.js SEASON) ---------------------------------
START = date(2025, 6, 10)
STEP_DAYS = 8
STEPS = 17
DATES = [(START + timedelta(days=STEP_DAYS * i)).isoformat() for i in range(STEPS)]

# ---- AOI geometry (Mula command, Ahmednagar, Maharashtra) ------------------
# Irregular command-area polygon around the canal network.
AOI_CENTER = (74.595, 19.355)
AOI_RING = [
    [74.515, 19.300],
    [74.560, 19.275],
    [74.640, 19.285],
    [74.700, 19.320],
    [74.715, 19.370],
    [74.690, 19.420],
    [74.625, 19.440],
    [74.555, 19.430],
    [74.510, 19.395],
    [74.500, 19.345],
    [74.515, 19.300],
]

# ---- Class vocabularies (mirror config.js) ---------------------------------
CROPS = ["sugarcane", "cotton", "soybean", "maize", "wheat", "onion", "fallow"]
CROP_WEIGHTS = [0.26, 0.16, 0.20, 0.12, 0.08, 0.08, 0.10]
STRESS = ["none", "mild", "moderate", "severe", "extreme"]
ADVISORY = ["no_irrigation", "watch", "irrigate_soon", "irrigate_now", "critical"]
STAGES = ["initial", "development", "mid", "late"]

# Per-crop agronomy: total available water (TAW, mm), depletion fraction p,
# peak NDVI, season length (in 8-day steps), planting offset (steps).
CROP_PARAMS = {
    "sugarcane": dict(taw=160, p=0.65, ndvi_peak=0.86, length=17, plant=0),
    "cotton":    dict(taw=140, p=0.65, ndvi_peak=0.80, length=15, plant=1),
    "soybean":   dict(taw=110, p=0.50, ndvi_peak=0.84, length=12, plant=2),
    "maize":     dict(taw=120, p=0.55, ndvi_peak=0.82, length=13, plant=2),
    "wheat":     dict(taw=120, p=0.55, ndvi_peak=0.78, length=14, plant=3),
    "onion":     dict(taw=90,  p=0.30, ndvi_peak=0.74, length=12, plant=2),
    "fallow":    dict(taw=80,  p=0.50, ndvi_peak=0.18, length=17, plant=0),
}

# Stage sensitivity (Ky-like): mid-season is most sensitive to deficit.
STAGE_SENSITIVITY = {"initial": 0.4, "development": 0.8, "mid": 1.2, "late": 0.7}


def stage_for(crop: str, t: int) -> str:
    """Map an 8-day index to an FAO-56 growth stage for a crop."""
    pr = CROP_PARAMS[crop]
    rel = (t - pr["plant"]) / max(1, pr["length"])
    if crop == "fallow":
        return "initial"
    if rel < 0.0:
        return "initial"
    if rel < 0.2:
        return "initial"
    if rel < 0.45:
        return "development"
    if rel < 0.75:
        return "mid"
    return "late"


def ndvi_for(crop: str, t: int, vigour: float) -> float:
    """Double-logistic-ish NDVI curve across the season."""
    pr = CROP_PARAMS[crop]
    rel = (t - pr["plant"]) / max(1, pr["length"])
    base = 0.14
    if crop == "fallow":
        return round(base + 0.05 * math.sin(t / 2.0) + random.uniform(-0.02, 0.02), 3)
    if rel < 0:
        val = base
    else:
        # green-up then senescence
        green = 1.0 / (1.0 + math.exp(-9 * (rel - 0.22)))
        senesce = 1.0 / (1.0 + math.exp(-9 * (rel - 0.85)))
        val = base + (pr["ndvi_peak"] - base) * vigour * (green - senesce)
    val += random.uniform(-0.025, 0.025)
    return round(max(0.05, min(0.95, val)), 3)


def advisory_from_ratio(ratio: float, stage: str) -> tuple[str, str]:
    """Map Dr/RAW ratio + stage sensitivity to advisory + stress class."""
    sens = STAGE_SENSITIVITY[stage]
    eff = ratio * (0.85 + 0.25 * (sens - 0.4))  # sensitive stages escalate
    if eff < 0.5:
        adv, st = "no_irrigation", "none"
    elif eff < 0.85:
        adv, st = "watch", "mild"
    elif eff < 1.0:
        adv, st = "irrigate_soon", "moderate"
    elif eff < 1.25:
        adv, st = "irrigate_now", "severe"
    else:
        adv, st = "critical", "extreme"
    return adv, st


def grid_polygons(rows: int, cols: int):
    """Square-ish field polygons covering the AOI bbox, clipped roughly."""
    lons = [p[0] for p in AOI_RING]
    lats = [p[1] for p in AOI_RING]
    minx, maxx = min(lons), max(lons)
    miny, maxy = min(lats), max(lats)
    # inset so cells fall inside the irregular boundary
    minx += 0.012; maxx -= 0.012; miny += 0.012; maxy -= 0.012
    dx = (maxx - minx) / cols
    dy = (maxy - miny) / rows
    cells = []
    for r in range(rows):
        for c in range(cols):
            x0 = minx + c * dx + dx * 0.06
            x1 = minx + (c + 1) * dx - dx * 0.06
            y0 = miny + r * dy + dy * 0.06
            y1 = miny + (r + 1) * dy - dy * 0.06
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            # keep cells roughly within the command's circular extent
            if math.hypot(cx - AOI_CENTER[0], (cy - AOI_CENTER[1]) * 1.05) > 0.105:
                continue
            ring = [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]
            cells.append((ring, (cx, cy)))
    return cells


def build_fields():
    cells = grid_polygons(8, 8)
    feats = []
    fid = 0
    for ring, (cx, cy) in cells:
        fid += 1
        crop = random.choices(CROPS, weights=CROP_WEIGHTS)[0]
        pr = CROP_PARAMS[crop]
        vigour = round(random.uniform(0.82, 1.0), 3)
        # head-reach vs tail-reach water availability gradient (east = tail)
        reach = (cx - 74.50) / 0.215  # 0 head .. 1 tail
        soil_aw_frac = 0.62 - 0.18 * reach + random.uniform(-0.05, 0.05)

        ndvi_series = []
        stress_series = []
        advisory_series = []
        dr_series = []
        raw_series = []
        stage_series = []

        # simple running root-zone depletion bucket
        taw = pr["taw"] * (0.85 + 0.3 * soil_aw_frac)
        dr = taw * 0.30  # start moderately wet
        # one mid-season dry spell per field (peak-demand window) — drives the
        # agronomically-important moisture-stress signal the dashboard showcases.
        dry_start = random.randint(6, 9)
        dry_len = random.randint(2, 4)
        for t in range(STEPS):
            stage = stage_for(crop, t)
            stage_series.append(stage)
            n = ndvi_for(crop, t, vigour)
            ndvi_series.append(n)

            raw = round(pr["p"] * taw, 1)
            # ET demand scales with canopy (NDVI) and stage (Kc-like, ~8-day mm)
            kc = 0.35 + 1.20 * max(0.0, n - 0.12)
            etc = kc * 8.6 * (0.9 + 0.25 * (STAGE_SENSITIVITY[stage] - 0.4))
            # effective rainfall — sparse late-monsoon, tail-reach gets less,
            # and a dry spell suppresses it during the peak-demand window.
            in_dry = dry_start <= t < dry_start + dry_len
            rain_mu = (16 if t < 5 else 6) - 9 * reach
            rain = max(0.0, random.gauss(rain_mu, 6))
            if in_dry:
                rain *= 0.10
            dr = dr + etc - rain
            if dr < 0:
                dr = 0.0
            if dr > taw:
                dr = taw
            ratio = dr / max(1e-6, raw)
            adv, st = advisory_from_ratio(ratio, stage)
            # fallow never needs irrigation advisory beyond watch
            if crop == "fallow":
                adv, st = ("no_irrigation", "none") if dr < raw else ("watch", "mild")
            # If advised to irrigate, canal water partially refills the profile
            # next step. Tail-reach fields receive less (warabandi inequity) so
            # they linger in irrigate-now / critical — a key story for judges.
            if adv in ("irrigate_now", "critical"):
                refill_frac = 0.85 - 0.45 * reach
                dr = max(0.0, dr - raw * refill_frac)

            dr_series.append(round(dr, 1))
            raw_series.append(raw)
            stress_series.append(st)
            advisory_series.append(adv)

        feats.append(
            {
                "type": "Feature",
                "id": fid,
                "properties": {
                    "field_id": f"MUL-{fid:03d}",
                    "command_id": "mula_nira_command",
                    "village": random.choice(
                        ["Belapur", "Rahuri", "Newasa", "Shevgaon", "Sonai", "Pathardi"]
                    ),
                    "area_ha": round(random.uniform(0.8, 3.4), 2),
                    "crop": crop,
                    "vigour": vigour,
                    "reach": round(reach, 2),
                    # snapshot at the LAST step (so static map looks meaningful);
                    # app.js overwrites these per the time slider from *_series.
                    "stage": stage_series[-1],
                    "ndvi": ndvi_series[-1],
                    "stress_class": stress_series[-1],
                    "advisory_class": advisory_series[-1],
                    "dr": dr_series[-1],
                    "raw": raw_series[-1],
                    # full season arrays (length == STEPS) for the time slider/chart
                    "stage_series": stage_series,
                    "ndvi_series": ndvi_series,
                    "stress_series": stress_series,
                    "advisory_series": advisory_series,
                    "dr_series": dr_series,
                    "raw_series": raw_series,
                },
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            }
        )
    return feats


def rollup_for(feats, t):
    """Roll up advisory composition + gross irrigation demand at step t."""
    n = len(feats)
    counts = {a: 0 for a in ADVISORY}
    total_area = 0.0
    advisory_area = {a: 0.0 for a in ADVISORY}
    gross_demand_m3 = 0.0
    for f in feats:
        p = f["properties"]
        adv = p["advisory_series"][t]
        counts[adv] += 1
        a_ha = p["area_ha"]
        total_area += a_ha
        advisory_area[adv] += a_ha
        dr = p["dr_series"][t]
        raw = p["raw_series"][t]
        # net irrigation depth (mm) needed where Dr exceeds RAW; gross @ 0.65 eff
        net_mm = max(0.0, dr - 0.0) if adv in ("irrigate_now", "critical") else 0.0
        if adv == "irrigate_soon":
            net_mm = max(0.0, dr - raw * 0.6)
        gross_mm = net_mm / 0.65
        gross_demand_m3 += gross_mm * 1e-3 * a_ha * 1e4  # mm->m * ha->m2
    pct = {a: round(100.0 * counts[a] / n, 1) for a in ADVISORY}
    return {
        "command_id": "mula_nira_command",
        "command_name": "Mula Command Area",
        "date": DATES[t],
        "n_fields": n,
        "total_area_ha": round(total_area, 1),
        "advisory_counts": counts,
        "advisory_pct": pct,
        "advisory_area_ha": {a: round(advisory_area[a], 1) for a in ADVISORY},
        "gross_demand_m3": round(gross_demand_m3, 0),
        "gross_demand_ham": round(gross_demand_m3 / 1e4, 1),  # hectare-metres
        # warabandi: fixed 7-day rotation; turn-time proportional to area share
        "warabandi_cycle_days": 7,
        "warabandi_turn_min_per_ha": 38,
    }


def main():
    aoi = {
        "type": "FeatureCollection",
        "name": "mula_command_area",
        "features": [
            {
                "type": "Feature",
                "id": "mula_nira_command",
                "properties": {
                    "command_id": "mula_nira_command",
                    "name": "Mula Command Area",
                    "state": "Maharashtra",
                    "district": "Ahmednagar",
                    "source": "Demo boundary (illustrative)",
                    "gca_ha": 80900,
                },
                "geometry": {"type": "Polygon", "coordinates": [AOI_RING]},
            }
        ],
    }

    feats = build_fields()
    fields = {
        "type": "FeatureCollection",
        "name": "mula_fields",
        "season": {"start": START.isoformat(), "step_days": STEP_DAYS, "steps": STEPS},
        "dates": DATES,
        "features": feats,
    }

    # rollup.json holds the full per-step series + a default snapshot.
    rollups = [rollup_for(feats, t) for t in range(STEPS)]
    rollup = {
        "command_id": "mula_nira_command",
        "command_name": "Mula Command Area",
        "dates": DATES,
        "default_index": STEPS - 1,
        "by_date": rollups,
    }

    base = "."
    with open(f"{base}/command_area.geojson", "w") as fh:
        json.dump(aoi, fh, indent=1)
    with open(f"{base}/fields.geojson", "w") as fh:
        json.dump(fields, fh, indent=1)
    with open(f"{base}/rollup.json", "w") as fh:
        json.dump(rollup, fh, indent=1)
    print(f"fields={len(feats)} steps={STEPS} dates={DATES[0]}..{DATES[-1]}")


if __name__ == "__main__":
    main()
