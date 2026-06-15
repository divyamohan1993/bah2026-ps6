"""AgriStress serving layer — FastAPI API + tiler + cache (O(1) read-hot-path).

Public surface kept import-light: ``create_app`` and ``Cache`` are imported
eagerly (they pull in FastAPI / Pillow only when actually used at call time via
local imports where heavy), while the rest is available on submodules.
"""

from __future__ import annotations

__all__ = ["Cache", "FeatureStore", "app", "create_app", "seed_demo_store"]


def __getattr__(name: str):  # PEP 562 lazy attribute access
    # Lazy so ``import agristress.serving`` doesn't require fastapi unless used.
    if name in ("create_app", "app"):
        from .api import app, create_app

        return {"create_app": create_app, "app": app}[name]
    if name == "Cache":
        from .cache import Cache

        return Cache
    if name in ("FeatureStore", "seed_demo_store"):
        from .store import FeatureStore, seed_demo_store

        return {"FeatureStore": FeatureStore, "seed_demo_store": seed_demo_store}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
