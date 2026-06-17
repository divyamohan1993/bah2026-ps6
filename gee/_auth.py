"""Importable alias for ``gee/00_auth.py``.

Python module names cannot start with a digit, so the numbered step script
``00_auth.py`` cannot be imported with a normal ``import`` statement. The
numbered files are the human-facing, ordered pipeline steps; this thin shim
re-exports the auth helpers under an importable name so that steps 01–06 (and
the test-suite / notebooks) can simply do::

    from gee._auth import init_ee, ee_available, EarthEngineUnavailable

It loads ``00_auth.py`` by file path and forwards its public API. Import never
crashes when ``earthengine-api`` is missing (the underlying module guards it).
"""

from __future__ import annotations

import importlib.util
import os
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
_AUTH_PATH = os.path.join(_HERE, "00_auth.py")

_spec = importlib.util.spec_from_file_location("gee_00_auth", _AUTH_PATH)
if _spec is None or _spec.loader is None:  # pragma: no cover - defensive
    raise ImportError(f"Could not load auth helper from {_AUTH_PATH}")
_auth = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_auth)

# Re-export the public surface.
init_ee = _auth.init_ee
require_ee = _auth.require_ee
ee_available = _auth.ee_available
EarthEngineUnavailable = _auth.EarthEngineUnavailable


def get_ee() -> Any:
    """Return the (possibly ``None``) guarded ``ee`` handle from the auth module."""
    return _auth.ee


__all__ = [
    "init_ee",
    "require_ee",
    "ee_available",
    "EarthEngineUnavailable",
    "get_ee",
]
