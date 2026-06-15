"""FastAPI serving layer for AgriStress (the O(1) read-hot-path).

Exposes the dashboard / field-app API over a materialised H3 feature store:

* ``GET /health``                          — liveness + backend info
* ``GET /crop?h3=&date=``                  — crop-type record for a cell
* ``GET /stress?h3=&date=``                — phenology-aware moisture-stress record
* ``GET /advisory?h3=&date=``              — 8-day irrigation advisory record
* ``GET /timeseries?h3=&var=&start=&end=`` — per-variable time series for a cell
* ``GET /tiles/{layer}/{z}/{x}/{y}.png``   — dynamic map tile (demo PNG / TiTiler)
* ``GET /aoi``                             — configured canal-command AOIs
* ``GET /command/{id}/rollup``             — advisory aggregated to a command area

Responses are Pydantic-validated, CORS is enabled for the dashboard, and reads
are served through the :mod:`~agristress.serving.cache` (Redis if ``REDIS_URL``
is set, else in-memory LRU). Everything runs offline with no cloud credentials:
the feature store is seeded with deterministic synthetic data at app creation.
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .cache import Cache, cache_key
from .store import FeatureStore, seed_demo_store
from .tiler import LAYER_PALETTES, h3_to_tile, render_demo_tile, titiler_router

__all__ = ["app", "create_app"]

# Layers exposed by the keyed-lookup endpoints.
_VALID_LAYERS = {"crop", "stress", "advisory"}


# --------------------------------------------------------------------------
# Pydantic response models
# --------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "agristress-serving"
    version: str
    cache_backend: str
    cells: int
    aois: int


class CropResponse(BaseModel):
    h3: str
    date: str
    crop_class: int
    crop_label: str
    confidence: float
    ndvi: float | None = None


class StressResponse(BaseModel):
    h3: str
    date: str
    stress_index: float
    stress_level: int
    stress_label: str
    phenology_stage: str
    vci: float | None = None
    smi: float | None = None
    ndvi: float | None = None
    ndwi: float | None = None
    vh_vv: float | None = None


class AdvisoryResponse(BaseModel):
    h3: str
    date: str
    etc_mm: float
    eta_mm: float
    rainfall_mm: float
    deficit_mm: float
    status: str
    recommended_action: str
    irrigation_mm: float


class TimeSeriesPoint(BaseModel):
    date: str
    value: float


class TimeSeriesResponse(BaseModel):
    h3: str
    var: str
    start: str | None = None
    end: str | None = None
    count: int
    series: list[TimeSeriesPoint]


class AOI(BaseModel):
    id: str
    name: str
    command_area_ha: float
    h3_cells: list[str]
    centroid: dict[str, float]


class AOIListResponse(BaseModel):
    count: int
    aois: list[AOI]


class CommandRollup(BaseModel):
    command_id: str
    name: str
    date: str
    n_cells: int
    command_area_ha: float
    mean_stress_index: float = Field(..., description="Area-mean moisture-stress index")
    mean_deficit_mm: float
    total_irrigation_demand_m3: float = Field(
        ..., description="Aggregate 8-day irrigation demand over the command area"
    )
    action_breakdown: dict[str, int]
    dominant_action: str


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _resolve_date(store: FeatureStore, h3: str, date: str | None) -> str:
    """Return the requested date or fall back to the cell's latest available."""

    if date:
        return date
    latest = store.latest_date(h3)
    if latest is None:
        raise HTTPException(status_code=404, detail=f"No data for h3={h3}")
    return latest


def _lookup(store: FeatureStore, cache: Cache, layer: str, h3: str, date: str) -> dict[str, Any]:
    """O(1) cached materialised lookup of a single ``(layer, h3, date)`` record."""

    key = cache_key(layer, h3=h3, date=date)
    record = cache.get(key)
    if record is None:
        record = store.get(h3, date, layer)
        if record is None:
            raise HTTPException(
                status_code=404,
                detail=f"No {layer} record for h3={h3} date={date}",
            )
        cache.set(key, record, ttl=300)
    return record


# --------------------------------------------------------------------------
# App factory
# --------------------------------------------------------------------------
def create_app(
    store: FeatureStore | None = None,
    cache: Cache | None = None,
    seed: bool = True,
    season: str = "kharif",
    cors_origins: list[str] | None = None,
) -> FastAPI:
    """Build and return the AgriStress serving :class:`FastAPI` app.

    Parameters
    ----------
    store:
        Pre-populated feature store. When ``None`` a fresh one is created and (if
        ``seed``) filled with deterministic synthetic demo data so the API serves
        meaningful JSON offline.
    cache:
        Cache instance. Defaults to a new :class:`Cache` (Redis via ``REDIS_URL``
        else in-memory LRU).
    seed:
        Seed the store with demo data when it is empty.
    season / cors_origins:
        Demo season and allowed CORS origins (``["*"]`` by default for the dash).
    """

    try:
        from .. import __version__ as version
    except Exception:  # pragma: no cover - defensive
        version = "0.0.0"

    if store is None:
        store = FeatureStore()
    if seed and len(store) == 0:
        seed_demo_store(store, season=season)
    if cache is None:
        cache = Cache()

    app = FastAPI(
        title="AgriStress Serving API",
        version=version,
        description=(
            "Crop-type, moisture-stress and 8-day irrigation-advisory serving "
            "layer for canal command areas (ISRO BAH 2026 PS6)."
        ),
    )

    origins = (
        cors_origins
        if cors_origins is not None
        else os.environ.get("AGRISTRESS_CORS_ORIGINS", "*").split(",")
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in origins] or ["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Make collaborators reachable from handlers / tests.
    app.state.store = store
    app.state.cache = cache

    # Optionally mount TiTiler for real COG tiling when the lib is installed.
    _titiler = titiler_router()
    if _titiler is not None:  # pragma: no cover - only when titiler installed
        app.include_router(_titiler, prefix="/cog", tags=["tiles"])

    # ---- endpoints -------------------------------------------------------
    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    def health() -> HealthResponse:
        return HealthResponse(
            version=version,
            cache_backend=cache.backend,
            cells=len(store.cells()),
            aois=len(store.aois),
        )

    @app.get("/crop", response_model=CropResponse, tags=["layers"])
    def crop(
        h3: str = Query(..., description="H3 cell index"),
        date: str | None = Query(None, description="ISO date; defaults to latest"),
    ) -> CropResponse:
        d = _resolve_date(store, h3, date)
        rec = _lookup(store, cache, "crop", h3, d)
        return CropResponse(
            h3=h3,
            date=d,
            **{
                k: rec[k] for k in rec if k in CropResponse.model_fields and k not in ("h3", "date")
            },
        )

    @app.get("/stress", response_model=StressResponse, tags=["layers"])
    def stress(
        h3: str = Query(..., description="H3 cell index"),
        date: str | None = Query(None, description="ISO date; defaults to latest"),
    ) -> StressResponse:
        d = _resolve_date(store, h3, date)
        rec = _lookup(store, cache, "stress", h3, d)
        return StressResponse(
            h3=h3,
            date=d,
            **{
                k: rec[k]
                for k in rec
                if k in StressResponse.model_fields and k not in ("h3", "date")
            },
        )

    @app.get("/advisory", response_model=AdvisoryResponse, tags=["layers"])
    def advisory(
        h3: str = Query(..., description="H3 cell index"),
        date: str | None = Query(None, description="ISO date; defaults to latest"),
    ) -> AdvisoryResponse:
        d = _resolve_date(store, h3, date)
        rec = _lookup(store, cache, "advisory", h3, d)
        return AdvisoryResponse(
            h3=h3,
            date=d,
            **{
                k: rec[k]
                for k in rec
                if k in AdvisoryResponse.model_fields and k not in ("h3", "date")
            },
        )

    @app.get("/timeseries", response_model=TimeSeriesResponse, tags=["layers"])
    def timeseries(
        h3: str = Query(..., description="H3 cell index"),
        var: str = Query(
            "stress_index", description="Variable: stress_index/vci/ndvi/deficit_mm/..."
        ),
        start: str | None = Query(None, description="ISO start date (inclusive)"),
        end: str | None = Query(None, description="ISO end date (inclusive)"),
    ) -> TimeSeriesResponse:
        if not store.has_cell(h3):
            raise HTTPException(status_code=404, detail=f"No data for h3={h3}")
        key = cache_key("timeseries", h3=h3, var=var, start=start or "", end=end or "")
        series = cache.get(key)
        if series is None:
            series = store.timeseries(h3, var, start, end)
            cache.set(key, series, ttl=120)
        return TimeSeriesResponse(
            h3=h3,
            var=var,
            start=start,
            end=end,
            count=len(series),
            series=[TimeSeriesPoint(**p) for p in series],
        )

    @app.get("/tiles/{layer}/{z}/{x}/{y}.png", tags=["tiles"])
    def tiles(layer: str, z: int, x: int, y: int) -> Response:
        if layer not in LAYER_PALETTES:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown layer '{layer}'. Valid: {sorted(LAYER_PALETTES)}",
            )
        key = cache_key("tile", h3=f"{layer}/{z}/{x}/{y}")
        png = cache.get(key)
        if png is None:
            png = render_demo_tile(layer, z, x, y)
            cache.set(key, png, ttl=600)
        return Response(content=png, media_type="image/png")

    @app.get("/aoi", response_model=AOIListResponse, tags=["meta"])
    def aoi() -> AOIListResponse:
        aois = _load_aois(store)
        return AOIListResponse(count=len(aois), aois=[AOI(**a) for a in aois])

    @app.get("/command/{command_id}/rollup", response_model=CommandRollup, tags=["layers"])
    def command_rollup(
        command_id: str,
        date: str | None = Query(None, description="ISO date; defaults to latest"),
    ) -> CommandRollup:
        aois = _load_aois(store)
        match = next((a for a in aois if a["id"] == command_id), None)
        if match is None:
            raise HTTPException(status_code=404, detail=f"Unknown command id '{command_id}'")
        return _rollup_command(store, cache, match, date)

    @app.get("/h3/tile", tags=["tiles"])
    def h3_tile(h3: str = Query(...), z: int = Query(10, ge=0, le=20)) -> dict[str, Any]:
        """Resolve the XYZ tile (and centroid) containing an H3 cell."""

        return h3_to_tile(h3, z)

    return app


# --------------------------------------------------------------------------
# Rollup + AOI helpers (kept module-level for reuse / testing)
# --------------------------------------------------------------------------
def _load_aois(store: FeatureStore) -> list[dict[str, Any]]:
    """Return AOIs from the store, else from configs, else seeded demo AOIs."""

    if store.aois:
        return store.aois
    configured = _aois_from_config()
    if configured:
        store.aois = configured
        return configured
    # Last resort: synthesise from whatever cells exist.
    cells = store.cells()
    if cells:
        store.aois = [
            {
                "id": "CMD-001",
                "name": "Demo Command Area",
                "command_area_ha": 10000.0,
                "h3_cells": cells,
                "centroid": {"lat": 30.74, "lon": 76.79},
            }
        ]
    return store.aois


def _aois_from_config() -> list[dict[str, Any]]:
    """Best-effort load of canal-command AOIs from package config (defensive).

    The config subpackage may not exist yet; any failure yields an empty list so
    the API still serves seeded demo AOIs.
    """

    try:  # pragma: no cover - config package optional during early dev
        from ..config import settings as _settings  # type: ignore

        getter = getattr(_settings, "get_aois", None) or getattr(_settings, "aois", None)
        data = getter() if callable(getter) else getter
        if isinstance(data, list) and data:
            return [dict(a) for a in data]
    except Exception:
        pass
    return []


def _rollup_command(
    store: FeatureStore, cache: Cache, aoi: dict[str, Any], date: str | None
) -> CommandRollup:
    """Aggregate per-cell advisory + stress to a command-area rollup."""

    cells = aoi.get("h3_cells", [])
    # Resolve a common date (latest shared if not given).
    if date is None:
        date = next(
            (store.latest_date(c) for c in cells if store.latest_date(c)),
            None,
        )
    if date is None:
        raise HTTPException(status_code=404, detail="No data for command area")

    key = cache_key("rollup", h3=aoi["id"], date=date)
    cached = cache.get(key)
    if cached is not None:
        return CommandRollup(**cached)

    stress_vals: list[float] = []
    deficit_vals: list[float] = []
    actions: dict[str, int] = {}
    n = 0
    for c in cells:
        s = store.get(c, date, "stress")
        a = store.get(c, date, "advisory")
        if s is None or a is None:
            continue
        n += 1
        stress_vals.append(float(s.get("stress_index", 0.0)))
        deficit_vals.append(float(a.get("deficit_mm", 0.0)))
        act = a.get("recommended_action", "unknown")
        actions[act] = actions.get(act, 0) + 1

    area_ha = float(aoi.get("command_area_ha", 0.0))
    mean_def = sum(deficit_vals) / n if n else 0.0
    # mm over command area -> m^3: mm * ha * 10.
    total_demand = round(mean_def * area_ha * 10.0, 1)
    dominant = max(actions, key=lambda k: actions[k]) if actions else "no_data"

    result = CommandRollup(
        command_id=aoi["id"],
        name=aoi.get("name", aoi["id"]),
        date=date,
        n_cells=n,
        command_area_ha=area_ha,
        mean_stress_index=round(sum(stress_vals) / n, 4) if n else 0.0,
        mean_deficit_mm=round(mean_def, 2),
        total_irrigation_demand_m3=total_demand,
        action_breakdown=actions,
        dominant_action=dominant,
    )
    cache.set(key, result.model_dump(), ttl=120)
    return result


# Default module-level app for ``uvicorn agristress.serving.api:app`` and tooling.
# Built lazily-ish but eagerly enough for ASGI servers that import the attribute.
app = create_app()
