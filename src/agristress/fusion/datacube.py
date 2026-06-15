"""Analysis-ready datacube assembly: (time, y, x) xarray.Dataset + zarr I/O.

The datacube is the hand-off product between *fusion* and the downstream crop-type /
phenology / moisture-stress / irrigation-advisory models. It stacks every harmonised
layer on a single grid and cadence:

variables
    Surface-reflectance bands (``blue, green, red, nir, swir1, swir2``),
    vegetation indices (``ndvi, evi, ndwi``), ``soil_moisture``, ``lst_et``,
    ``phenophase`` (integer growth-stage code), ``qa`` (clear/cloud flag) and
    ``sigma`` (per-pixel uncertainty / σ for downstream weighting).
dims
    ``(time, y, x)``.

:func:`build_datacube` accepts real harmonised ``layers`` *or*, with
``demo=True`` (the default when no layers are supplied), synthesises a realistic
seasonal cube — sinusoidal NDVI phenology + spatial field structure + noise + cloud
gaps — so the whole downstream stack is testable completely offline.

:func:`to_zarr` / :func:`open_zarr` persist / reload the cube (chunked Zarr).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

try:  # xarray is a core dependency; import defensively for clearer errors.
    import xarray as xr
except Exception as exc:  # pragma: no cover - xarray missing
    xr = None  # type: ignore[assignment]
    _XR_ERR = exc
else:
    _XR_ERR = None

if TYPE_CHECKING:  # pragma: no cover
    import xarray as xr  # noqa: F811

# Canonical variable groups.
SR_BANDS: tuple[str, ...] = ("blue", "green", "red", "nir", "swir1", "swir2")
VI_VARS: tuple[str, ...] = ("ndvi", "evi", "ndwi")
EXTRA_VARS: tuple[str, ...] = ("soil_moisture", "lst_et", "phenophase", "qa", "sigma")
CUBE_VARS: tuple[str, ...] = SR_BANDS + VI_VARS + EXTRA_VARS

# Phenophase codes (stage-wise stress interpretation downstream).
PHENOPHASE: dict[str, int] = {
    "bare": 0,
    "emergence": 1,
    "vegetative": 2,
    "reproductive": 3,
    "maturity": 4,
    "senescence": 5,
}


def _require_xarray() -> None:
    if xr is None:  # pragma: no cover - exercised only without xarray
        raise ImportError(
            "agristress.fusion.datacube requires xarray; install it via "
            "`pip install xarray`."
        ) from _XR_ERR


# ---------------------------------------------------------------------------
# DEMO synthetic cube
# ---------------------------------------------------------------------------
def _demo_cube(
    aoi: tuple[float, float, float, float],
    times,
    *,
    target_res_m: float,
    height: int,
    width: int,
    seed: int,
    cloud_fraction: float,
):
    """Synthesise a realistic seasonal datacube (sinusoidal phenology + cloud gaps)."""
    rng = np.random.default_rng(seed)
    minx, miny, maxx, maxy = aoi
    nt = len(times)

    # Geographic coordinates (degrees) for the demo grid.
    xs = np.linspace(minx, maxx, width)
    ys = np.linspace(maxy, miny, height)  # north-up

    # Day-of-year drives the seasonal NDVI phenology.
    t64 = np.asarray(times, dtype="datetime64[D]")
    doy = ((t64 - t64.astype("datetime64[Y]")).astype("timedelta64[D]")).astype(int) + 1

    # Per-field spatial structure: a few "fields" with different peak timing/amplitude.
    fy, fx = np.mgrid[0:height, 0:width]
    field_id = ((fy // max(height // 4, 1)) * 3 + (fx // max(width // 4, 1))) % 5
    peak_doy = 150 + 25 * (field_id - 2)  # staggered peak greenness per field block
    amplitude = 0.45 + 0.08 * (field_id - 2)
    baseline = 0.18 + 0.02 * field_id

    ndvi = np.empty((nt, height, width), dtype="float32")
    for i in range(nt):
        # Gaussian-in-time greenness curve centred on each field's peak DOY.
        season = np.exp(-0.5 * ((doy[i] - peak_doy) / 45.0) ** 2)
        frame = baseline + amplitude * season
        frame = frame + rng.normal(0.0, 0.015, size=frame.shape)
        ndvi[i] = np.clip(frame, 0.0, 0.95)

    # Cloud gaps: knock out random pixels per time (more in the monsoon middle).
    qa = np.ones((nt, height, width), dtype="int16")  # 1 = clear, 0 = cloud
    for i in range(nt):
        frac = cloud_fraction * (0.6 + 0.8 * np.exp(-0.5 * ((doy[i] - 200) / 40.0) ** 2))
        mask = rng.random((height, width)) < frac
        qa[i, mask] = 0

    ndvi_obs = np.where(qa == 1, ndvi, np.nan).astype("float32")

    # Derive plausible SR bands from NDVI (red/nir consistent with the index).
    nir = np.clip(0.20 + 0.45 * ndvi, 0.0, 1.0).astype("float32")
    red = np.clip(nir * (1.0 - ndvi) / (1.0 + ndvi + 1e-6), 0.0, 1.0).astype("float32")
    green = np.clip(red * 1.15 + 0.02, 0.0, 1.0).astype("float32")
    blue = np.clip(red * 0.95 + 0.01, 0.0, 1.0).astype("float32")
    swir1 = np.clip(0.18 + 0.10 * (1 - ndvi), 0.0, 1.0).astype("float32")
    swir2 = np.clip(0.12 + 0.08 * (1 - ndvi), 0.0, 1.0).astype("float32")

    evi = (2.5 * (nir - red) / (nir + 6.0 * red - 7.5 * blue + 1.0 + 1e-6)).astype("float32")
    evi = np.clip(evi, -0.2, 1.0)
    ndwi = ((green - nir) / (green + nir + 1e-6)).astype("float32")

    # Soil moisture: anti-correlated-ish with greenness + seasonal monsoon pulse.
    sm = np.empty((nt, height, width), dtype="float32")
    for i in range(nt):
        monsoon = 0.12 * np.exp(-0.5 * ((doy[i] - 210) / 35.0) ** 2)
        sm[i] = np.clip(0.15 + monsoon + 0.05 * (1 - ndvi[i]) + rng.normal(0, 0.01, (height, width)), 0.02, 0.55)

    # LST/ET composite proxy (deg C-ish), warmer over bare / stressed pixels.
    lst_et = np.clip(28.0 + 8.0 * (1 - ndvi) - 20.0 * sm + rng.normal(0, 0.5, ndvi.shape), 15.0, 50.0).astype("float32")

    # Phenophase from NDVI level + rising/falling limb.
    pheno = np.zeros((nt, height, width), dtype="int16")
    dndvi = np.gradient(ndvi, axis=0)
    pheno[(ndvi >= 0.25) & (ndvi < 0.5) & (dndvi >= 0)] = PHENOPHASE["emergence"]
    pheno[(ndvi >= 0.5) & (dndvi >= 0)] = PHENOPHASE["vegetative"]
    pheno[(ndvi >= 0.6) & (np.abs(dndvi) < 0.01)] = PHENOPHASE["reproductive"]
    pheno[(ndvi >= 0.5) & (dndvi < 0)] = PHENOPHASE["maturity"]
    pheno[(ndvi >= 0.25) & (ndvi < 0.5) & (dndvi < 0)] = PHENOPHASE["senescence"]

    # Per-pixel uncertainty: larger where cloudy (gap-filled) and at low NDVI.
    sigma = (0.01 + 0.04 * (qa == 0) + 0.02 * (1 - ndvi)).astype("float32")

    data = {
        "blue": blue, "green": green, "red": red, "nir": nir, "swir1": swir1, "swir2": swir2,
        "ndvi": ndvi_obs, "evi": evi, "ndwi": ndwi,
        "soil_moisture": sm, "lst_et": lst_et, "phenophase": pheno,
        "qa": qa, "sigma": sigma,
    }
    coords = {"time": np.asarray(times, dtype="datetime64[ns]"), "y": ys, "x": xs}
    ds = xr.Dataset(
        {k: (("time", "y", "x"), v) for k, v in data.items()},
        coords=coords,
        attrs={
            "title": "AgriStress DEMO datacube",
            "mode": "demo",
            "crs": "EPSG:4326",
            "target_res_m": float(target_res_m),
            "aoi_bbox": list(aoi),
            "phenophase_codes": ";".join(f"{k}={v}" for k, v in PHENOPHASE.items()),
        },
    )
    return ds


def build_datacube(
    layers: dict | None = None,
    aoi: tuple[float, float, float, float] = (76.0, 29.0, 76.5, 29.5),
    times=None,
    *,
    target_res_m: float = 10.0,
    cadence_days: int = 8,
    demo: bool | None = None,
    height: int = 32,
    width: int = 32,
    n_times: int | None = None,
    seed: int = 42,
    cloud_fraction: float = 0.25,
):
    """Assemble the analysis-ready (time, y, x) datacube.

    Parameters
    ----------
    layers
        Mapping ``{variable_name: array}`` of pre-harmonised layers, each shaped
        ``(time, y, x)`` (or 2-D, broadcast over time). When ``None`` (or ``demo``
        is truthy) a synthetic seasonal cube is generated instead.
    aoi
        Area of interest as ``(minx, miny, maxx, maxy)`` in EPSG:4326.
    times
        Sequence of timestamps (``datetime64`` / ISO strings). When ``None`` a
        season of ``n_times`` steps spaced ``cadence_days`` apart is created
        (kharif season default, starting 2024-06-01).
    target_res_m
        Nominal target resolution (m); recorded in attrs (the demo grid uses a
        fixed ``height x width`` for speed).
    cadence_days
        Temporal cadence (days) of the cube (8-day default to match the advisory).
    demo
        Force demo (``True``) / real (``False``) mode. Default: demo when ``layers``
        is ``None``.
    height, width
        Demo grid size.
    n_times
        Number of demo time steps (default: one kharif season ≈ ``184/cadence``).
    seed
        RNG seed for reproducible demo data.
    cloud_fraction
        Base fraction of cloud-masked pixels per scene in demo mode.

    Returns
    -------
    xarray.Dataset
        Cube with dims ``(time, y, x)`` and variables :data:`CUBE_VARS`.
    """
    _require_xarray()
    use_demo = demo if demo is not None else (layers is None)

    if times is None:
        if n_times is None:
            n_times = max(int(round(184 / cadence_days)), 4)  # ~kharif season
        start = np.datetime64("2024-06-01")
        times = start + np.arange(n_times) * np.timedelta64(int(cadence_days), "D")
    times = np.asarray(times, dtype="datetime64[ns]")

    if use_demo:
        return _demo_cube(
            aoi, times,
            target_res_m=target_res_m, height=height, width=width,
            seed=seed, cloud_fraction=cloud_fraction,
        )

    # ---- real-layers assembly ----
    if not layers:
        raise ValueError("build_datacube(demo=False) requires a non-empty `layers` mapping.")

    nt = len(times)
    # Infer the grid from the first 3-D layer.
    ref = next((np.asarray(v) for v in layers.values() if np.asarray(v).ndim == 3), None)
    if ref is None:
        ref2 = np.asarray(next(iter(layers.values())))
        if ref2.ndim != 2:
            raise ValueError("layers must contain at least one (time,y,x) or (y,x) array")
        h, w = ref2.shape
    else:
        _, h, w = ref.shape

    minx, miny, maxx, maxy = aoi
    coords = {
        "time": times,
        "y": np.linspace(maxy, miny, h),
        "x": np.linspace(minx, maxx, w),
    }

    data_vars = {}
    for name, arr in layers.items():
        a = np.asarray(arr)
        if a.ndim == 2:  # broadcast a static layer across time
            a = np.broadcast_to(a, (nt, h, w))
        if a.shape != (nt, h, w):
            raise ValueError(f"layer {name!r} has shape {a.shape}, expected {(nt, h, w)}")
        data_vars[name] = (("time", "y", "x"), a)

    # Ensure a qa / sigma layer exists (downstream weighting relies on them).
    if "qa" not in data_vars:
        data_vars["qa"] = (("time", "y", "x"), np.ones((nt, h, w), dtype="int16"))
    if "sigma" not in data_vars:
        data_vars["sigma"] = (("time", "y", "x"), np.full((nt, h, w), 0.02, dtype="float32"))

    return xr.Dataset(
        data_vars,
        coords=coords,
        attrs={
            "title": "AgriStress datacube",
            "mode": "real",
            "crs": "EPSG:4326",
            "target_res_m": float(target_res_m),
            "cadence_days": int(cadence_days),
            "aoi_bbox": list(aoi),
        },
    )


# ---------------------------------------------------------------------------
# Zarr I/O
# ---------------------------------------------------------------------------
def to_zarr(ds, path, *, mode: str = "w", chunks: dict | None = None):
    """Persist a datacube to a chunked Zarr store.

    Parameters
    ----------
    ds
        The datacube (``xarray.Dataset``).
    path
        Destination Zarr store path / URL.
    mode
        Write mode (``"w"`` overwrite by default).
    chunks
        Optional rechunking before write, e.g. ``{"time": 1, "y": 256, "x": 256}``.

    Returns
    -------
    The store handle returned by ``Dataset.to_zarr`` (backend-dependent).
    """
    _require_xarray()
    if chunks:
        ds = ds.chunk(chunks)
    return ds.to_zarr(path, mode=mode)


def open_zarr(path, **kwargs):
    """Open a datacube previously written with :func:`to_zarr`.

    Parameters
    ----------
    path
        Zarr store path / URL.
    **kwargs
        Forwarded to ``xarray.open_zarr``.

    Returns
    -------
    xarray.Dataset
    """
    _require_xarray()
    return xr.open_zarr(path, **kwargs)
