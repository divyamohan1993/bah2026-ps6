"""AgriStress · GEE step 00 — Earth Engine authentication & initialisation helper.

ISRO BAH 2026 · Problem Statement 6 (AI-driven crop-type, moisture-stress &
irrigation advisory). This module is the single entry point every other
``gee/*.py`` script uses to obtain an *initialised* Earth Engine session.

Design rules (shared by all gee/ scripts)
------------------------------------------
1. ``import ee`` is **guarded**. Importing this module never crashes, even when
   the ``earthengine-api`` package is absent or the user has not authenticated.
   That lets the offline test-suite (``tests/test_gee.py``) compile and import
   every script without credentials.
2. The *only* place Earth Engine is actually contacted is inside
   :func:`init_ee`. Call it explicitly before doing any ``ee.*`` work.
3. If EE is unavailable we raise a single, actionable
   :class:`EarthEngineUnavailable` error with copy-paste setup instructions,
   rather than letting an opaque ``ImportError``/``EEException`` leak out.

Auth setup (one-time)
---------------------
    pip install earthengine-api
    earthengine authenticate                 # interactive OAuth (dev laptops)
    # …or service account for headless / CI runs:
    #   export EE_PROJECT=your-gcp-project
    #   export GEE_SERVICE_ACCOUNT=ee-runner@your-gcp-project.iam.gserviceaccount.com
    #   export GEE_SERVICE_ACCOUNT_KEY=/secrets/gee-sa.json

Then, in any script::

    from importlib import import_module
    auth = import_module("gee.00_auth")   # numeric module name → import_module
    auth.init_ee(project="your-gcp-project")
"""

from __future__ import annotations

import os
from typing import Any

# --- guarded Earth Engine import -------------------------------------------
# Never let a missing dependency break module import. ``ee is None`` signals
# "not installed"; the real check (installed *and* initialised) is in init_ee.
try:  # pragma: no cover - depends on environment
    import ee  # type: ignore
except Exception:  # ImportError or any transitive failure
    ee = None  # type: ignore


_AUTH_INSTRUCTIONS = (
    "Earth Engine is not available.\n"
    "  1) pip install earthengine-api\n"
    "  2) earthengine authenticate      # interactive OAuth, or set a service account:\n"
    "       export EE_PROJECT=your-gcp-project\n"
    "       export GEE_SERVICE_ACCOUNT=ee-runner@your-gcp-project.iam.gserviceaccount.com\n"
    "       export GEE_SERVICE_ACCOUNT_KEY=/secrets/gee-sa.json\n"
    "  3) re-run, passing project=... (or set EE_PROJECT)."
)


class EarthEngineUnavailable(RuntimeError):
    """Raised when an EE-dependent action is requested but EE cannot be used."""


def ee_available() -> bool:
    """Return ``True`` iff the ``earthengine-api`` package imported successfully.

    This does *not* guarantee the session is authenticated — only that calling
    :func:`init_ee` is worth attempting. Cheap and safe to call offline.
    """
    return ee is not None


def init_ee(
    project: str | None = None,
    *,
    service_account: str | None = None,
    key_file: str | None = None,
    high_volume: bool | None = None,
    quiet: bool = False,
) -> Any:
    """Initialise Earth Engine and return the live ``ee`` module.

    Resolution order for credentials:

    1. **Service account** — if ``service_account`` + ``key_file`` are given (or
       the ``GEE_SERVICE_ACCOUNT`` / ``GEE_SERVICE_ACCOUNT_KEY`` env vars are
       set). Best for CI / servers.
    2. **Stored user credentials** — whatever ``earthengine authenticate`` wrote
       to the persistent token store (best for laptops / Colab).

    Parameters
    ----------
    project:
        GCP project that EE quota/billing attaches to. Falls back to the
        ``EE_PROJECT`` environment variable.
    service_account, key_file:
        Service-account email + path to its JSON key (headless auth).
    high_volume:
        Route through the high-volume endpoint (batch/tile workloads). Falls
        back to the ``EE_HIGH_VOLUME`` env var.
    quiet:
        Suppress the success print.

    Returns
    -------
    The initialised ``ee`` module, so callers can do ``ee = init_ee(...)``.

    Raises
    ------
    EarthEngineUnavailable
        If the package is missing or initialisation fails. The message contains
        the full setup recipe.
    """
    if ee is None:
        raise EarthEngineUnavailable(_AUTH_INSTRUCTIONS)

    project = project or os.environ.get("EE_PROJECT")
    service_account = service_account or os.environ.get("GEE_SERVICE_ACCOUNT")
    key_file = key_file or os.environ.get("GEE_SERVICE_ACCOUNT_KEY")
    if high_volume is None:
        high_volume = os.environ.get("EE_HIGH_VOLUME", "false").lower() in {
            "1",
            "true",
            "yes",
        }

    init_kwargs: dict[str, Any] = {}
    if project:
        init_kwargs["project"] = project
    if high_volume:
        # High-volume endpoint for large batch / tile serving workloads.
        init_kwargs["opt_url"] = "https://earthengine-highvolume.googleapis.com"

    try:
        if service_account and key_file and os.path.exists(key_file):
            credentials = ee.ServiceAccountCredentials(service_account, key_file)
            ee.Initialize(credentials, **init_kwargs)
        else:
            # Use whatever `earthengine authenticate` persisted.
            ee.Initialize(**init_kwargs)
    except Exception as exc:  # EEException, auth failures, network, …
        raise EarthEngineUnavailable(
            f"ee.Initialize failed ({exc}).\n{_AUTH_INSTRUCTIONS}"
        ) from exc

    if not quiet:
        where = "service account" if (service_account and key_file) else "user credentials"
        print(f"[gee] Earth Engine initialised via {where}" + (f" (project={project})" if project else ""))
    return ee


def require_ee(project: str | None = None, **kwargs: Any) -> Any:
    """Convenience wrapper: initialise EE or raise. Alias of :func:`init_ee`.

    Provided so downstream scripts can write ``ee = require_ee()`` to express
    intent ("I need a working EE here") clearly.
    """
    return init_ee(project, **kwargs)


def main(project: str | None = None) -> int:
    """CLI guard: try to initialise EE and report status. Never raises."""
    if not ee_available():
        print("[gee] earthengine-api is NOT installed.\n" + _AUTH_INSTRUCTIONS)
        return 1
    try:
        init_ee(project)
        # Tiny round-trip to prove the session really works.
        print("[gee] sanity check:", ee.Number(1).add(1).getInfo(), "== 2")
        return 0
    except EarthEngineUnavailable as exc:
        print(str(exc))
        return 1


if __name__ == "__main__":  # pragma: no cover - manual entry point
    import sys

    raise SystemExit(main(project=os.environ.get("EE_PROJECT")))
