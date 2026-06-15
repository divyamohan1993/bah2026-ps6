"""Materialised H3 feature store backing the serving API.

The serving layer never recomputes science on the request path — it performs an
O(1) keyed lookup against a *materialised* store indexed by H3 cell. In
production this is populated by :mod:`agristress.indexing` (e.g. a parquet /
key-value table of per-cell, per-date crop / stress / advisory records).

This module provides:

* :class:`FeatureStore` — an in-memory dict-of-records store with O(1)
  ``(h3, date)`` lookups and helpers the API depends on.
* :func:`seed_demo_store` — fills a store with deterministic synthetic data for
  a set of H3 cells + dates so the API serves meaningful demo JSON with **zero**
  credentials. The pipeline orchestrator reuses this to populate the store.

Keeping the demo store here (not in ``api.py``) lets the pipeline and the API
share one source of truth.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import math
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "CROP_LABELS",
    "STRESS_LABELS",
    "FeatureStore",
    "demo_dates",
    "demo_h3_cells",
    "seed_demo_store",
]

CROP_LABELS = ["fallow", "rice", "wheat", "maize", "cotton", "sugarcane"]
STRESS_LABELS = ["none", "mild", "moderate", "severe"]
# Phenology stages used by the stage-aware stress model.
PHENOLOGY_STAGES = ["sowing", "vegetative", "reproductive", "maturity"]


def _h(*parts: Any) -> int:
    """Stable integer hash of the given parts (deterministic across runs)."""

    raw = "/".join(str(p) for p in parts).encode()
    return int(hashlib.sha256(raw).hexdigest()[:12], 16)


def _unit(*parts: Any) -> float:
    """Deterministic pseudo-random float in [0, 1)."""

    return (_h(*parts) % 10_000) / 10_000.0


@dataclass
class FeatureStore:
    """In-memory, O(1) materialised feature store keyed by H3 cell.

    Records are nested ``{h3: {date: {layer: payload}}}``. Lookups are plain dict
    indexing — constant time — which is the whole point of the read-hot-path.
    """

    records: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)
    aois: list[dict[str, Any]] = field(default_factory=list)

    # -- writes ------------------------------------------------------------
    def put(self, h3: str, date: str, layer: str, payload: dict[str, Any]) -> None:
        self.records.setdefault(h3, {}).setdefault(date, {})[layer] = payload

    # -- reads (O(1)) ------------------------------------------------------
    def get(self, h3: str, date: str, layer: str) -> dict[str, Any] | None:
        return self.records.get(h3, {}).get(date, {}).get(layer)

    def latest_date(self, h3: str) -> str | None:
        dates = self.records.get(h3)
        return max(dates) if dates else None

    def has_cell(self, h3: str) -> bool:
        return h3 in self.records

    def cells(self) -> list[str]:
        return list(self.records.keys())

    def dates(self, h3: str | None = None) -> list[str]:
        if h3 is not None:
            return sorted(self.records.get(h3, {}).keys())
        out: set[str] = set()
        for per_date in self.records.values():
            out.update(per_date.keys())
        return sorted(out)

    def timeseries(
        self, h3: str, var: str, start: str | None, end: str | None
    ) -> list[dict[str, Any]]:
        """Return ``[{date, value}]`` for a variable across the date range.

        ``var`` may be a stress metric (``vci``/``smi``/``stress_index``/``ndvi``/
        ``ndwi``/``vh_vv``) or an advisory metric (``etc``/``deficit_mm``).
        """

        series: list[dict[str, Any]] = []
        for date in self.dates(h3):
            if start and date < start:
                continue
            if end and date > end:
                continue
            rec = self.records[h3][date]
            value = _extract_var(rec, var)
            if value is not None:
                series.append({"date": date, "value": value})
        return series

    def __len__(self) -> int:
        return sum(len(d) for d in self.records.values())


def _extract_var(rec: dict[str, dict[str, Any]], var: str) -> float | None:
    """Pull a named scalar from a per-date record across its layers."""

    for layer in ("stress", "advisory", "crop"):
        payload = rec.get(layer)
        if isinstance(payload, dict) and var in payload:
            val = payload[var]
            if isinstance(val, (int, float)):
                return float(val)
    return None


# --------------------------------------------------------------------------
# Demo data generation
# --------------------------------------------------------------------------
def demo_h3_cells(n: int = 16, seed: str = "agristress") -> list[str]:
    """Return ``n`` synthetic H3-like cell ids.

    Uses real :mod:`h3` cells around a pilot canal command area when available,
    otherwise deterministic placeholder ids of the right shape.
    """

    try:
        import h3  # type: ignore

        latlng_to_cell = getattr(h3, "latlng_to_cell", None) or getattr(h3, "geo_to_h3", None)
        grid_disk = getattr(h3, "grid_disk", None) or getattr(h3, "k_ring", None)
        if latlng_to_cell and grid_disk:
            # Pilot AOI ~ Bhakra canal command (Punjab), res 7.
            center = latlng_to_cell(30.74, 76.79, 7)
            ring = list(grid_disk(center, 3))
            return ring[:n]
    except Exception:
        pass
    return [f"87{_h(seed, i):010x}ffff"[:15] for i in range(n)]


def demo_dates(season: str = "kharif", count: int = 8, step_days: int = 8) -> list[str]:
    """Return ``count`` 8-day composite dates for a season (ISO ``YYYY-MM-DD``)."""

    starts = {"kharif": (2025, 6, 1), "rabi": (2025, 11, 1), "zaid": (2025, 3, 1)}
    y, m, d = starts.get(season, (2025, 6, 1))
    base = _dt.date(y, m, d)
    return [(base + _dt.timedelta(days=step_days * i)).isoformat() for i in range(count)]


def _crop_record(h3: str, date: str) -> dict[str, Any]:
    cls = _h(h3, "crop") % len(CROP_LABELS)
    conf = round(0.7 + 0.29 * _unit(h3, "cropconf"), 3)
    return {
        "crop_class": cls,
        "crop_label": CROP_LABELS[cls],
        "confidence": conf,
        "ndvi": round(0.2 + 0.6 * _unit(h3, date, "ndvi"), 3),
    }


def _stress_record(h3: str, date: str, t: int) -> dict[str, Any]:
    # Seasonal arc: stress rises mid-season then eases (phenology-aware feel).
    phase = math.sin(math.pi * (t + 1) / 9.0)
    base = 0.25 + 0.5 * phase * (0.6 + 0.4 * _unit(h3, "stressbias"))
    stress_index = round(min(max(base + 0.1 * (_unit(h3, date) - 0.5), 0.0), 1.0), 3)
    level = min(int(stress_index * len(STRESS_LABELS)), len(STRESS_LABELS) - 1)
    stage = PHENOLOGY_STAGES[min(t // 2, len(PHENOLOGY_STAGES) - 1)]
    vci = round(max(0.0, 1.0 - stress_index + 0.05 * (_unit(h3, date, "vci") - 0.5)), 3)
    return {
        "stress_index": stress_index,
        "stress_level": level,
        "stress_label": STRESS_LABELS[level],
        "phenology_stage": stage,
        "vci": vci,
        "smi": round(max(0.0, 0.8 - 0.6 * stress_index), 3),
        "ndvi": round(0.2 + 0.6 * (1.0 - stress_index), 3),
        "ndwi": round(0.1 + 0.4 * (1.0 - stress_index), 3),
        "vh_vv": round(0.3 + 0.4 * _unit(h3, date, "sar"), 3),
    }


def _advisory_record(h3: str, date: str, stress: dict[str, Any]) -> dict[str, Any]:
    si = float(stress["stress_index"])
    etc = round(3.0 + 4.0 * (0.5 + 0.5 * si), 2)  # mm/day crop ET demand
    rain = round(6.0 * _unit(h3, date, "rain"), 2)
    eta = round(etc * (1.0 - 0.6 * si), 2)
    deficit = round(max(0.0, (etc - eta) * 8.0 - rain), 2)  # mm over 8-day window
    if deficit <= 2:
        status, action = "adequate", "no_irrigation"
    elif deficit <= 10:
        status, action = "mild_deficit", "monitor"
    elif deficit <= 25:
        status, action = "moderate_deficit", "irrigate_soon"
    else:
        status, action = "severe_deficit", "irrigate_now"
    return {
        "etc_mm": etc,
        "eta_mm": eta,
        "rainfall_mm": rain,
        "deficit_mm": deficit,
        "status": status,
        "recommended_action": action,
        "irrigation_mm": round(deficit, 2) if deficit > 2 else 0.0,
    }


def seed_demo_store(
    store: FeatureStore | None = None,
    season: str = "kharif",
    n_cells: int = 16,
    n_dates: int = 8,
) -> FeatureStore:
    """Populate (or create) a :class:`FeatureStore` with synthetic demo data.

    Generates crop / stress / advisory records for every ``(cell, date)`` and a
    couple of canal-command AOIs grouping the cells. Deterministic: same inputs →
    same store, so tests and demos are reproducible.
    """

    # NB: use an explicit ``is None`` check — an empty FeatureStore is falsy
    # (``__len__`` == 0), so ``store or FeatureStore()`` would silently discard
    # a caller-supplied empty store and populate a throwaway instead.
    store = FeatureStore() if store is None else store
    cells = demo_h3_cells(n_cells)
    dates = demo_dates(season, n_dates)

    for h3 in cells:
        crop = _crop_record(h3, dates[0])
        for t, date in enumerate(dates):
            store.put(h3, date, "crop", {**crop, "date": date})
            stress = _stress_record(h3, date, t)
            store.put(h3, date, "stress", {**stress, "date": date})
            advisory = _advisory_record(h3, date, stress)
            store.put(h3, date, "advisory", {**advisory, "date": date})

    # Two demo canal-command AOIs splitting the cells between head/tail reaches.
    half = max(1, len(cells) // 2)
    store.aois = [
        {
            "id": "CMD-001",
            "name": "Bhakra Main Canal — Head Reach",
            "command_area_ha": 12500,
            "h3_cells": cells[:half],
            "centroid": {"lat": 30.74, "lon": 76.79},
        },
        {
            "id": "CMD-002",
            "name": "Bhakra Main Canal — Tail Reach",
            "command_area_ha": 9800,
            "h3_cells": cells[half:],
            "centroid": {"lat": 30.55, "lon": 76.60},
        },
    ]
    return store
