"""H3 spatial indexing wrappers (+ a pure-python fallback) and cube → H3 flattening.

Thin, working wrappers over Uber's `h3` v4 library so the rest of AgriStress depends
on a small stable surface (``latlng_to_cell``, ``cell_to_latlng``, parent/children,
``grid_disk``, ``cells_for_polygon``). When `h3` is **not installed** a deterministic
lat/lng-quantisation fallback is used instead so spatial indexing, the feature store
and the tests keep working offline. The fallback cells are *not* real H3 indices but
preserve the contract the pipeline relies on: a stable many-to-one
``(lat, lng, res) → cell`` mapping with an (approximate) inverse.

:func:`cube_to_h3_table` flattens an xarray datacube into a tidy
``(h3_cell, date, variable) → value`` DataFrame — the O(1) feature-store source.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

try:
    import h3 as _h3

    H3_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only without h3
    _h3 = None
    H3_AVAILABLE = False


# ---------------------------------------------------------------------------
# Fallback cell scheme: quantise lat/lng onto a per-resolution grid and pack the
# row/col into a hex string tagged with the resolution. Approximate edge lengths
# loosely track H3 so a higher `res` means a finer grid.
# ---------------------------------------------------------------------------
def _fallback_step_deg(res: int) -> float:
    """Grid step in degrees for the fallback scheme (finer as res grows)."""
    # H3 res 0 ≈ 1000 km cells; ~halving per level. Map to a degree step.
    return 60.0 / (2.0 ** max(res, 0))


def _fallback_latlng_to_cell(lat: float, lng: float, res: int) -> str:
    step = _fallback_step_deg(res)
    row = int(math.floor((lat + 90.0) / step))
    col = int(math.floor((lng + 180.0) / step))
    return f"f{res:02d}{row:08x}{col:08x}"


def _fallback_cell_to_latlng(cell: str) -> tuple[float, float]:
    res = int(cell[1:3])
    row = int(cell[3:11], 16)
    col = int(cell[11:19], 16)
    step = _fallback_step_deg(res)
    lat = (row + 0.5) * step - 90.0
    lng = (col + 0.5) * step - 180.0
    return lat, lng


def _fallback_resolution(cell: str) -> int:
    return int(cell[1:3])


# ---------------------------------------------------------------------------
# Public wrappers
# ---------------------------------------------------------------------------
def latlng_to_cell(lat: float, lng: float, res: int = 9) -> str:
    """Index a ``(lat, lng)`` point to the H3 cell at resolution ``res``.

    Parameters
    ----------
    lat, lng
        Latitude / longitude in degrees (WGS84).
    res
        H3 resolution (0 coarse … 15 fine). Default 9 (~174 m edge, field scale).

    Returns
    -------
    str
        H3 cell token (or a fallback token when ``h3`` is unavailable).
    """
    if H3_AVAILABLE:
        return _h3.latlng_to_cell(float(lat), float(lng), int(res))
    return _fallback_latlng_to_cell(float(lat), float(lng), int(res))


def cell_to_latlng(cell: str) -> tuple[float, float]:
    """Return the ``(lat, lng)`` centroid of an H3 cell."""
    if H3_AVAILABLE:
        return tuple(_h3.cell_to_latlng(cell))
    return _fallback_cell_to_latlng(cell)


def cell_to_parent(cell: str, res: int | None = None) -> str:
    """Return the parent cell at coarser resolution ``res`` (default: one level up)."""
    if H3_AVAILABLE:
        if res is None:
            res = _h3.get_resolution(cell) - 1
        return _h3.cell_to_parent(cell, int(res))
    cur = _fallback_resolution(cell)
    target = (cur - 1) if res is None else int(res)
    lat, lng = _fallback_cell_to_latlng(cell)
    return _fallback_latlng_to_cell(lat, lng, max(target, 0))


def cell_to_children(cell: str, res: int | None = None) -> list[str]:
    """Return child cells at finer resolution ``res`` (default: one level down).

    The fallback emits the 4 quadrant children of the quantisation grid (a real H3
    cell has 7 children); this is sufficient for the pipeline's hierarchical roll-up.
    """
    if H3_AVAILABLE:
        if res is None:
            res = _h3.get_resolution(cell) + 1
        return list(_h3.cell_to_children(cell, int(res)))
    cur = _fallback_resolution(cell)
    target = (cur + 1) if res is None else int(res)
    lat, lng = _fallback_cell_to_latlng(cell)
    step = _fallback_step_deg(target)
    out = []
    for dlat in (-0.5, 0.5):
        for dlng in (-0.5, 0.5):
            out.append(_fallback_latlng_to_cell(lat + dlat * step, lng + dlng * step, target))
    return sorted(set(out))


def grid_disk(cell: str, k: int = 1) -> list[str]:
    """Return all cells within grid distance ``k`` of ``cell`` (inclusive).

    Parameters
    ----------
    cell
        Centre cell.
    k
        Ring radius in cells (``k=1`` ⇒ the cell + its immediate neighbours).
    """
    if H3_AVAILABLE:
        return list(_h3.grid_disk(cell, int(k)))
    # Fallback: walk the quantisation grid in a (2k+1)^2 block.
    res = _fallback_resolution(cell)
    lat, lng = _fallback_cell_to_latlng(cell)
    step = _fallback_step_deg(res)
    out = set()
    for dr in range(-k, k + 1):
        for dc in range(-k, k + 1):
            out.add(_fallback_latlng_to_cell(lat + dr * step, lng + dc * step, res))
    return sorted(out)


def cells_for_polygon(geojson, res: int = 9) -> list[str]:
    """Return the H3 cells whose centroid falls inside a GeoJSON Polygon.

    Parameters
    ----------
    geojson
        A GeoJSON-like Polygon ``dict`` (``{"type": "Polygon", "coordinates":
        [[[lng, lat], ...]]}``) or a ``(lng, lat)`` ring list.
    res
        H3 resolution.

    Returns
    -------
    list[str]
        Sorted, de-duplicated covering cells.
    """
    ring = _extract_ring(geojson)
    if H3_AVAILABLE:
        # h3 v4: build an LatLngPoly (expects (lat, lng) tuples) then fill.
        latlng_ring = [(lat, lng) for lng, lat in ring]
        poly = _h3.LatLngPoly(latlng_ring)
        return sorted(_h3.h3shape_to_cells(poly, int(res)))

    # Fallback: rasterise the polygon bbox onto the quantisation grid and keep
    # cells whose centre is inside (ray-casting point-in-polygon).
    lngs = [p[0] for p in ring]
    lats = [p[1] for p in ring]
    step = _fallback_step_deg(res)
    cells = set()
    lat = min(lats)
    while lat <= max(lats) + step:
        lng = min(lngs)
        while lng <= max(lngs) + step:
            if _point_in_ring(lng, lat, ring):
                cells.add(_fallback_latlng_to_cell(lat, lng, res))
            lng += step
        lat += step
    if not cells:  # tiny polygon — at least index its centroid
        cells.add(_fallback_latlng_to_cell(float(np.mean(lats)), float(np.mean(lngs)), res))
    return sorted(cells)


def _extract_ring(geojson) -> list[tuple[float, float]]:
    """Pull the outer ring ``[(lng, lat), ...]`` from a GeoJSON Polygon-ish input."""
    if isinstance(geojson, dict):
        if geojson.get("type") == "Feature":
            geojson = geojson["geometry"]
        coords = geojson["coordinates"]
        ring = coords[0]
    else:
        ring = list(geojson)
    return [(float(p[0]), float(p[1])) for p in ring]


def _point_in_ring(x: float, y: float, ring: list[tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon for a ``(lng, lat)`` ring."""
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-15) + xi):
            inside = not inside
        j = i
    return inside


# ---------------------------------------------------------------------------
# Cube -> H3 table
# ---------------------------------------------------------------------------
def cube_to_h3_table(
    ds,
    res: int = 9,
    *,
    variables: list[str] | None = None,
    aggfunc: str = "mean",
    dropna: bool = True,
) -> "pd.DataFrame":
    """Flatten a ``(time, y, x)`` datacube into a tidy H3 feature table.

    Every pixel is indexed to its H3 cell; pixels sharing a cell at a given
    ``(date, variable)`` are aggregated (``aggfunc``). The result is keyed uniquely
    on ``(h3_cell, date, variable)`` — the source for the O(1) feature store.

    Parameters
    ----------
    ds
        ``xarray.Dataset`` with dims ``(time, y, x)`` and 1-D ``y`` (lat) / ``x``
        (lng) coordinates in degrees.
    res
        H3 resolution for the cell keys (default 9).
    variables
        Subset of data variables to include (default: all numeric data vars).
    aggfunc
        Pandas aggregation for pixels collapsing into the same cell (``"mean"``,
        ``"median"``, ``"max"`` …).
    dropna
        Drop rows whose aggregated value is NaN (default ``True``).

    Returns
    -------
    pandas.DataFrame
        Columns ``["h3_cell", "date", "variable", "value"]``, unique on the first
        three.
    """
    import pandas as pd

    if not hasattr(ds, "data_vars"):
        raise TypeError("cube_to_h3_table expects an xarray.Dataset")

    lats = np.asarray(ds["y"].values, dtype=float)
    lngs = np.asarray(ds["x"].values, dtype=float)
    times = np.asarray(ds["time"].values)
    dates = np.datetime_as_string(times.astype("datetime64[D]"), unit="D")

    # Precompute the H3 cell for every (y, x) pixel once (shared across time/vars).
    lat_grid, lng_grid = np.meshgrid(lats, lngs, indexing="ij")
    flat_lat = lat_grid.ravel()
    flat_lng = lng_grid.ravel()
    cells = np.array(
        [latlng_to_cell(la, lo, res) for la, lo in zip(flat_lat, flat_lng)],
        dtype=object,
    )

    if variables is None:
        variables = [
            v for v in ds.data_vars
            if np.issubdtype(np.asarray(ds[v].dtype), np.number)
        ]

    frames = []
    n_pix = flat_lat.size
    for var in variables:
        arr = np.asarray(ds[var].values)  # (time, y, x)
        if arr.ndim != 3:
            continue
        for ti in range(arr.shape[0]):
            vals = arr[ti].ravel().astype(float)
            frames.append(
                pd.DataFrame(
                    {
                        "h3_cell": cells,
                        "date": np.repeat(dates[ti], n_pix),
                        "variable": var,
                        "value": vals,
                    }
                )
            )

    if not frames:
        return pd.DataFrame(columns=["h3_cell", "date", "variable", "value"])

    long = pd.concat(frames, ignore_index=True)
    if dropna:
        long = long.dropna(subset=["value"])

    table = (
        long.groupby(["h3_cell", "date", "variable"], as_index=False)["value"].agg(aggfunc)
    )
    return table.reset_index(drop=True)
