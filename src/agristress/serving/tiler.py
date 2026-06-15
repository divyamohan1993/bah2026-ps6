"""Tile serving for the AgriStress dashboard.

Production pattern (documented, optional deps)
----------------------------------------------
The map layers (crop type, moisture stress, irrigation advisory) are produced as
Cloud-Optimized GeoTIFFs (COGs) by the pipeline. Two complementary serving
strategies are supported:

* **Dynamic XYZ / WMTS via TiTiler + rio-tiler** — a tile is rendered on demand
  from a COG for a given ``z/x/y``. Mounting ``titiler.core``'s ``TilerFactory``
  exposes ``/cog/tiles/{z}/{x}/{y}.png?url=...`` with rescaling + colormaps. See
  :func:`titiler_router` for the wiring (a no-op stub when titiler is absent).
* **Static PMTiles** — pre-tiled pyramids packaged as a single ``.pmtiles`` file,
  served by range requests. Good for fully-offline / CDN distribution. See
  :func:`describe_pmtiles_pattern`.

Demo / offline path (always works)
----------------------------------
For a credentials-free demo we don't need a COG: :func:`render_demo_tile` renders
a deterministic, colour-coded 256x256 PNG per ``(layer, z, x, y)`` using NumPy +
Pillow so the dashboard renders meaningful tiles offline. Colours follow the same
legends the real layers use (a discrete class palette for *crop*, a green→red
ramp for *stress* / *advisory*).
"""

from __future__ import annotations

import hashlib
import io
import math
from typing import Any

import numpy as np

__all__ = [
    "TILE_SIZE",
    "LAYER_PALETTES",
    "render_demo_tile",
    "h3_to_tile",
    "tile_bounds",
    "describe_pmtiles_pattern",
    "titiler_router",
]

TILE_SIZE = 256

# Discrete crop-class palette (RGB). Indices align with a typical kharif legend.
_CROP_CLASSES: list[tuple[int, int, int]] = [
    (198, 224, 180),  # 0 fallow / other
    (33, 145, 56),    # 1 rice / paddy
    (255, 211, 0),    # 2 wheat
    (242, 142, 43),   # 3 maize
    (140, 86, 199),   # 4 cotton
    (78, 121, 167),   # 5 sugarcane
]

# Sequential ramps (low → high) used for continuous stress / advisory rasters.
LAYER_PALETTES: dict[str, str] = {
    "crop": "discrete",
    "stress": "green_red",   # low stress (green) → high stress (red)
    "advisory": "blue_red",  # adequate (blue) → deficit, irrigate (red)
}


def _ramp(value: np.ndarray, stops: list[tuple[int, int, int]]) -> np.ndarray:
    """Linearly interpolate ``value`` in [0, 1] across colour ``stops``."""

    value = np.clip(value, 0.0, 1.0)
    n = len(stops) - 1
    scaled = value * n
    idx = np.clip(np.floor(scaled).astype(int), 0, n - 1)
    frac = (scaled - idx)[..., None]
    lo = np.array(stops, dtype=float)[idx]
    hi = np.array(stops, dtype=float)[idx + 1]
    return (lo * (1.0 - frac) + hi * frac).astype(np.uint8)


_GREEN_RED = [(26, 152, 80), (255, 255, 191), (215, 48, 39)]
_BLUE_RED = [(49, 54, 149), (255, 255, 191), (165, 0, 38)]


def _field_for_tile(layer: str, z: int, x: int, y: int) -> np.ndarray:
    """Deterministic synthetic scalar field in [0, 1] for a tile.

    Combines smooth sinusoidal gradients with a per-tile phase seeded from the
    tile address, so adjacent tiles look continuous yet every tile differs.
    """

    seed = int(hashlib.sha256(f"{layer}/{z}/{x}/{y}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    phase_x, phase_y = rng.uniform(0, 2 * math.pi, size=2)
    freq = 1.5 + (z % 4) * 0.5

    u = np.linspace(0, 1, TILE_SIZE, dtype=np.float32)
    gx, gy = np.meshgrid(u + x * 0.13, u + y * 0.13)
    field = (
        0.5
        + 0.25 * np.sin(2 * math.pi * freq * gx + phase_x)
        + 0.25 * np.cos(2 * math.pi * freq * gy + phase_y)
    )
    field += rng.normal(0, 0.04, size=field.shape)
    return np.clip(field, 0.0, 1.0)


def render_demo_tile(layer: str, z: int, x: int, y: int, fmt: str = "PNG") -> bytes:
    """Render a colour-coded demo tile as encoded image bytes (default PNG).

    Works fully offline. ``layer`` is one of ``crop`` / ``stress`` / ``advisory``
    (anything else falls back to a neutral grey ramp).
    """

    from PIL import Image  # local import keeps Pillow optional at import time

    field = _field_for_tile(layer, z, x, y)
    kind = LAYER_PALETTES.get(layer, "grey")

    if kind == "discrete":
        n = len(_CROP_CLASSES)
        classes = np.clip((field * n).astype(int), 0, n - 1)
        rgb = np.array(_CROP_CLASSES, dtype=np.uint8)[classes]
    elif kind == "green_red":
        rgb = _ramp(field, _GREEN_RED)
    elif kind == "blue_red":
        rgb = _ramp(field, _BLUE_RED)
    else:
        g = (field * 255).astype(np.uint8)
        rgb = np.stack([g, g, g], axis=-1)

    # Soft alpha so basemap shows through (dashboard overlays these tiles).
    alpha = np.full((TILE_SIZE, TILE_SIZE, 1), 200, dtype=np.uint8)
    rgba = np.concatenate([rgb, alpha], axis=-1)

    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def tile_bounds(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    """Return the ``(west, south, east, north)`` lon/lat bounds of an XYZ tile."""

    n = 2.0**z
    west = x / n * 360.0 - 180.0
    east = (x + 1) / n * 360.0 - 180.0

    def _lat(yy: float) -> float:
        return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * yy / n))))

    north = _lat(y)
    south = _lat(y + 1)
    return (west, south, east, north)


def _lonlat_to_tile(lon: float, lat: float, z: int) -> tuple[int, int]:
    n = 2.0**z
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    x = min(max(x, 0), int(n) - 1)
    y = min(max(y, 0), int(n) - 1)
    return x, y


def h3_to_tile(h3_index: str, z: int = 10) -> dict[str, Any]:
    """Map an H3 cell to the XYZ tile containing its centroid.

    Uses :mod:`h3` when available; otherwise hashes the index to a deterministic
    pseudo-location so the helper still returns a usable tile offline.
    """

    lat = lon = None
    try:
        import h3  # type: ignore

        # h3 v4 API; fall back to legacy name if needed.
        to_latlng = getattr(h3, "cell_to_latlng", None) or getattr(h3, "h3_to_geo", None)
        if to_latlng is not None:
            lat, lon = to_latlng(h3_index)
    except Exception:
        lat = lon = None

    if lat is None or lon is None:
        h = int(hashlib.sha256(str(h3_index).encode()).hexdigest()[:12], 16)
        lon = (h % 3600) / 10.0 - 180.0
        lat = ((h >> 16) % 1700) / 10.0 - 85.0

    x, y = _lonlat_to_tile(lon, lat, z)
    return {"h3": h3_index, "z": z, "x": x, "y": y, "lat": lat, "lon": lon}


def describe_pmtiles_pattern() -> dict[str, str]:
    """Document the PMTiles serving pattern (for the dashboard / docs)."""

    return {
        "format": "PMTiles v3 single-file archive",
        "build": "rio pmtiles / pmtiles convert from a COG or MBTiles pyramid",
        "serve": "Range-request capable static host or CDN; pmtiles JS reads the header",
        "url": "pmtiles://{base}/{layer}.pmtiles -> {z}/{x}/{y}",
        "use_case": "fully offline / edge distribution of crop/stress/advisory layers",
    }


def titiler_router(prefix: str = "/cog") -> Any | None:
    """Return a TiTiler ``TilerFactory`` router for dynamic COG tiling.

    Returns ``None`` when ``titiler.core`` is unavailable so the app factory can
    skip mounting it without failing. When present, the caller can
    ``app.include_router(router, prefix=prefix)`` to expose
    ``{prefix}/tiles/{z}/{x}/{y}.png?url=<cog>``.
    """

    try:  # pragma: no cover - exercised only when titiler is installed
        from titiler.core.factory import TilerFactory  # type: ignore

        return TilerFactory().router
    except Exception:
        return None
