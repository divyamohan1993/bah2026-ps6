"""End-to-end pipeline orchestration for AgriStress.

Wires the science stages into one chain and exposes a credentials-free
:meth:`Pipeline.run_demo` that executes the whole flow on synthetic data and
populates the serving feature store (plus a few COG/PNG demo artefacts), so
``agristress demo`` works with zero cloud credentials.

Stage order (mirrors :mod:`agristress` subpackages)::

    catalog → ingestion → preprocessing → fusion (datacube) → features
            → models (crop + stress) → irrigation (advisory) → indexing (H3 store) → serve

Sibling modules are imported **defensively**: each stage tries to import and call
its real implementation, and degrades gracefully (clear log message + synthetic
fallback) when a sibling isn't ready yet. This lets the pipeline run end-to-end
during early development while remaining a drop-in once siblings land.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..serving.store import FeatureStore, demo_dates, demo_h3_cells, seed_demo_store

__all__ = ["Pipeline", "PipelineResult", "run_demo"]

logger = logging.getLogger("agristress.pipeline")

# Ordered stages: (attribute name, module path, candidate entrypoint names).
_STAGES: list[tuple[str, str, tuple[str, ...]]] = [
    ("catalog", "agristress.catalog", ("get_registry", "registry", "summary")),
    ("ingestion", "agristress.ingestion", ("ingest", "run", "main")),
    ("preprocessing", "agristress.preprocessing", ("preprocess", "run")),
    ("fusion", "agristress.fusion", ("build_datacube", "fuse", "run")),
    ("features", "agristress.features", ("extract_features", "compute", "run")),
    ("models", "agristress.models", ("predict", "run")),
    ("irrigation", "agristress.irrigation", ("advisory", "compute_advisory", "run")),
    ("indexing", "agristress.indexing", ("index", "build_store", "run")),
]


def _try_import(module_path: str) -> Any | None:
    """Import a sibling module, returning ``None`` (logged) if unavailable."""

    try:
        return importlib.import_module(module_path)
    except Exception as exc:  # ModuleNotFoundError or import-time error
        logger.info("stage module %s not available yet (%s); using fallback", module_path, exc)
        return None


def _resolve_entrypoint(mod: Any, names: tuple[str, ...]) -> Any | None:
    for name in names:
        fn = getattr(mod, name, None)
        # Must be callable *and* not a submodule that merely shares the name
        # (e.g. ``agristress.irrigation.advisory`` is a module, not a function).
        if callable(fn) and not _ismodule(fn):
            return fn
    return None


def _ismodule(obj: Any) -> bool:
    import types

    return isinstance(obj, types.ModuleType)


@dataclass
class PipelineResult:
    """Outcome of a pipeline run."""

    aoi: str
    season: str
    store: FeatureStore
    stages_run: list[str] = field(default_factory=list)
    stages_fallback: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)

    @property
    def n_cells(self) -> int:
        return len(self.store.cells())

    @property
    def n_records(self) -> int:
        return len(self.store)

    def summary(self) -> dict[str, Any]:
        return {
            "aoi": self.aoi,
            "season": self.season,
            "cells": self.n_cells,
            "records": self.n_records,
            "stages_run": self.stages_run,
            "stages_fallback": self.stages_fallback,
            "artifacts": self.artifacts,
        }


class Pipeline:
    """Orchestrates the AgriStress stages end-to-end.

    The orchestrator owns a :class:`FeatureStore` (the materialised H3 store the
    serving API reads). Each ``run_*`` populates ``self.context`` which later
    stages consume; missing siblings fall back to synthetic generation.
    """

    def __init__(
        self, store: FeatureStore | None = None, out_dir: str | Path | None = None
    ) -> None:
        self.store = store if store is not None else FeatureStore()
        self.out_dir = Path(out_dir) if out_dir else Path("outputs")
        self.context: dict[str, Any] = {}
        self.stages_run: list[str] = []
        self.stages_fallback: list[str] = []

    # -- generic stage runner ---------------------------------------------
    def _run_stage(self, attr: str, module_path: str, names: tuple[str, ...], **kwargs: Any) -> Any:
        mod = _try_import(module_path)
        fn = _resolve_entrypoint(mod, names) if mod is not None else None
        if fn is None:
            self.stages_fallback.append(attr)
            return None
        try:
            result = fn(**kwargs) if kwargs else fn()
            self.stages_run.append(attr)
            self.context[attr] = result
            return result
        except Exception as exc:  # real module present but failed → fall back
            logger.warning("stage %s raised %s; continuing with fallback", attr, exc)
            self.stages_fallback.append(attr)
            return None

    # -- demo entrypoint ---------------------------------------------------
    def run_demo(self, aoi: str = "CMD-001", season: str = "kharif") -> PipelineResult:
        """Run the whole chain on synthetic data; populate the feature store.

        Always succeeds with no credentials. Real stages are invoked when
        available (best-effort, non-fatal); the synthetic feature store + demo
        artefacts guarantee the serving API has data to serve.
        """

        logger.info("AgriStress demo pipeline: aoi=%s season=%s", aoi, season)
        cells = demo_h3_cells()
        dates = demo_dates(season)
        self.context.update({"aoi": aoi, "season": season, "h3_cells": cells, "dates": dates})

        # Best-effort invoke each real stage (graceful when sibling missing).
        for attr, module_path, names in _STAGES:
            self._run_stage(attr, module_path, names)

        # Materialise the H3 feature store with deterministic synthetic records.
        # (This is the indexing-stage output the serving API reads in O(1).)
        seed_demo_store(self.store, season=season)

        artifacts = self._write_demo_artifacts(season)

        result = PipelineResult(
            aoi=aoi,
            season=season,
            store=self.store,
            stages_run=list(self.stages_run),
            stages_fallback=list(self.stages_fallback),
            artifacts=artifacts,
            context=dict(self.context),
        )
        logger.info(
            "demo complete: %d cells, %d records, %d artifacts (%d stages real, %d fallback)",
            result.n_cells,
            result.n_records,
            len(artifacts),
            len(self.stages_run),
            len(self.stages_fallback),
        )
        return result

    def _write_demo_artifacts(self, season: str) -> list[str]:
        """Write a few PNG tiles (and a GeoTIFF/COG if rasterio is present).

        Failures here are non-fatal — artefacts are a convenience for the
        dashboard, not required for the API to serve JSON.
        """

        artifacts: list[str] = []
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # pragma: no cover - fs perms
            logger.warning("could not create out_dir %s: %s", self.out_dir, exc)
            return artifacts

        # Demo PNG previews per layer via the tiler.
        try:
            from ..serving.tiler import render_demo_tile

            for layer in ("crop", "stress", "advisory"):
                png = render_demo_tile(layer, 8, 181, 110)
                path = self.out_dir / f"{layer}_{season}_demo.png"
                path.write_bytes(png)
                artifacts.append(str(path))
        except Exception as exc:  # pragma: no cover - Pillow missing
            logger.info("PNG artefact generation skipped: %s", exc)

        # Optional COG (only if rasterio available) — documents the real output.
        cog = self._maybe_write_cog(season)
        if cog:
            artifacts.append(cog)
        return artifacts

    def _maybe_write_cog(self, season: str) -> str | None:
        try:  # pragma: no cover - rasterio is heavy / optional
            import numpy as np
            import rasterio
            from rasterio.transform import from_bounds

            data = (np.random.default_rng(0).random((64, 64)) * 100).astype("float32")
            path = self.out_dir / f"stress_{season}_demo.tif"
            transform = from_bounds(76.5, 30.4, 77.0, 30.9, 64, 64)
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                height=64,
                width=64,
                count=1,
                dtype="float32",
                crs="EPSG:4326",
                transform=transform,
            ) as dst:
                dst.write(data, 1)
            return str(path)
        except Exception:
            return None


def run_demo(
    aoi: str = "CMD-001", season: str = "kharif", out_dir: str | Path | None = None
) -> PipelineResult:
    """Module-level convenience wrapper around :meth:`Pipeline.run_demo`."""

    return Pipeline(out_dir=out_dir).run_demo(aoi=aoi, season=season)
