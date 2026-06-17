"""AgriStress pipeline — orchestrator + CLI.

Ties the science stages (catalog → ingestion → preprocessing → fusion → features
→ models → irrigation → indexing) into the serving feature store, and exposes the
``agristress`` command-line interface.
"""

from __future__ import annotations

__all__ = ["Pipeline", "PipelineResult", "app", "main", "run_demo"]


def __getattr__(name: str):  # PEP 562 lazy access keeps imports light
    if name in ("Pipeline", "PipelineResult", "run_demo"):
        from .orchestrator import Pipeline, PipelineResult, run_demo

        return {"Pipeline": Pipeline, "PipelineResult": PipelineResult, "run_demo": run_demo}[name]
    if name in ("main", "app"):
        from .cli import app, main

        return {"main": main, "app": app}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
