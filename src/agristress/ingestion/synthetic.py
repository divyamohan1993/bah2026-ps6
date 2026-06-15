"""Offline synthetic data generation for AgriStress ingestion.

Everything here is deterministic (seeded) and depends only on ``numpy`` (and optionally
``xarray`` when the caller asks for a labelled object). It lets the whole ingestion stack
— and the test suite — run with **no cloud credentials and no network**.

The two public products are:

* :class:`SyntheticStack` — a lightweight dataclass wrapping a numpy array plus
  ``time``/``band`` coordinates and provenance metadata. It has ``.to_xarray()`` so a
  caller can upgrade to a labelled :class:`xarray.DataArray` when xarray is installed.
* :func:`synth_stack_for_sensor` — builds a physically-plausible stack for a given
  :class:`~agristress.catalog.sensors.SensorSpec` (right bands, sensible value ranges).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from agristress.catalog.sensors import SensorSpec, SensorType

if TYPE_CHECKING:  # pragma: no cover - typing only
    import xarray as xr

__all__ = [
    "BBox",
    "SyntheticStack",
    "make_time_axis",
    "normalize_bbox",
    "synth_stack_for_sensor",
]

#: A bounding box as ``(min_lon, min_lat, max_lon, max_lat)``.
BBox = tuple[float, float, float, float]

# A small default AOI: a slice of the Mula-Nira canal command area (Maharashtra, India),
# matching ``AGRISTRESS_DEFAULT_AOI=mula_nira_command`` in .env.example.
DEFAULT_BBOX: BBox = (74.0, 18.4, 74.4, 18.8)


def normalize_bbox(aoi: BBox | Any) -> BBox:
    """Coerce an AOI into a ``(min_lon, min_lat, max_lon, max_lat)`` tuple.

    Accepts a 4-tuple/list, anything exposing a ``.bounds`` 4-tuple (e.g. a shapely
    geometry), or ``None`` (-> :data:`DEFAULT_BBOX`).
    """
    if aoi is None:
        return DEFAULT_BBOX
    bounds = getattr(aoi, "bounds", None)
    if bounds is not None and len(tuple(bounds)) == 4:
        b = tuple(float(x) for x in bounds)
        return (b[0], b[1], b[2], b[3])
    seq = tuple(aoi)
    if len(seq) != 4:
        raise ValueError(
            f"AOI must be a (min_lon, min_lat, max_lon, max_lat) 4-tuple or have a "
            f".bounds attribute; got {aoi!r}"
        )
    return (float(seq[0]), float(seq[1]), float(seq[2]), float(seq[3]))


def make_time_axis(start: str, end: str, n_time: int) -> list[_dt.date]:
    """Return ``n_time`` evenly spaced dates spanning ``[start, end]`` inclusive."""
    s = _parse_date(start)
    e = _parse_date(end)
    if e < s:
        s, e = e, s
    if n_time <= 1:
        return [s]
    total = (e - s).days
    step = total / (n_time - 1) if total > 0 else 0
    return [s + _dt.timedelta(days=round(step * i)) for i in range(n_time)]


def _parse_date(value: str) -> _dt.date:
    if isinstance(value, _dt.datetime):
        return value.date()
    if isinstance(value, _dt.date):
        return value
    return _dt.date.fromisoformat(str(value)[:10])


def _seed_from(*parts: object) -> int:
    """Stable 32-bit seed derived from the given parts (so output is reproducible)."""
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(digest[:8], 16)


@dataclass
class SyntheticStack:
    """A lightweight, labelled cube returned by the offline ingestion path.

    Attributes
    ----------
    data:
        ``float32`` array of shape ``(time, y, x, band)``.
    sensor_id:
        Source sensor registry id.
    bands:
        Band / polarization names, length == ``data.shape[-1]``.
    times:
        Acquisition dates, length == ``data.shape[0]``.
    bbox:
        AOI bounding box used to generate the stack.
    provenance:
        Free-form metadata dict (source, family, units, demo flag, ...).
    """

    data: np.ndarray
    sensor_id: str
    bands: tuple[str, ...]
    times: list[_dt.date]
    bbox: BBox
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def dims(self) -> tuple[str, str, str, str]:
        return ("time", "y", "x", "band")

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return tuple(self.data.shape)  # type: ignore[return-value]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"SyntheticStack(sensor_id={self.sensor_id!r}, shape={self.shape}, "
            f"bands={self.bands}, times=[{self.times[0]}..{self.times[-1]}])"
        )

    def to_xarray(self) -> xr.DataArray:
        """Return an :class:`xarray.DataArray` view (requires ``xarray``).

        Raises
        ------
        ImportError
            If ``xarray`` is not installed.
        """
        try:
            import xarray as xr
        except ImportError as exc:  # pragma: no cover - exercised only without xarray
            raise ImportError(
                "xarray is required for SyntheticStack.to_xarray(); install xarray or "
                "use the .data ndarray directly."
            ) from exc

        ny, nx = self.data.shape[1], self.data.shape[2]
        min_lon, min_lat, max_lon, max_lat = self.bbox
        ys = np.linspace(max_lat, min_lat, ny)
        xs = np.linspace(min_lon, max_lon, nx)
        return xr.DataArray(
            self.data,
            dims=self.dims,
            coords={
                "time": [np.datetime64(d) for d in self.times],
                "y": ys,
                "x": xs,
                "band": list(self.bands),
            },
            name=self.sensor_id,
            attrs=dict(self.provenance),
        )


# ---------------------------------------------------------------------------
# Per-family value generation
# ---------------------------------------------------------------------------
def _band_field(
    rng: np.random.Generator,
    base: float,
    spread: float,
    season: float,
    size: int,
    lo: float,
    hi: float,
) -> np.ndarray:
    """A smooth-ish 2-D field around ``base`` with a seasonal offset, clipped to [lo, hi]."""
    # Low-frequency spatial structure + light noise -> looks like a field, not TV static.
    yy, xx = np.mgrid[0:size, 0:size] / max(size - 1, 1)
    gradient = 0.5 * (yy + xx) - 0.5
    speckle = rng.standard_normal((size, size))
    field2d = base + season + spread * (0.6 * gradient + 0.4 * speckle)
    return np.clip(field2d, lo, hi).astype(np.float32)


def _family_profile(sensor_type: SensorType, band: str) -> tuple[float, float, float, float]:
    """Return ``(base, spread, lo, hi)`` value parameters for a band of a sensor family."""
    b = band.lower()
    if sensor_type is SensorType.SAR:
        # Backscatter in dB; VH typically lower than VV.
        base = -16.0 if "vh" in b or "hv" in b else -10.0
        return (base, 3.0, -30.0, 0.0)
    if sensor_type is SensorType.OPTICAL:
        if "ndvi" in b:
            return (0.55, 0.25, -0.2, 1.0)
        if "evi" in b:
            return (0.45, 0.2, -0.2, 1.0)
        if b.endswith(("b8", "b08", "nir", "b5", "sr_b5")) or "nir" in b:
            return (0.35, 0.15, 0.0, 1.0)  # high NIR over vegetation
        return (0.12, 0.08, 0.0, 1.0)  # visible reflectance
    if sensor_type is SensorType.RADIOMETER:
        return (0.28, 0.08, 0.0, 0.6)  # volumetric soil moisture m3/m3
    if sensor_type is SensorType.THERMAL:
        if "et" in b or "pet" in b:
            return (4.0, 1.5, 0.0, 12.0)  # mm/day
        if "emis" in b:
            return (0.97, 0.01, 0.9, 1.0)
        return (305.0, 6.0, 270.0, 330.0)  # LST in Kelvin
    if sensor_type is SensorType.PRECIP:
        return (3.0, 4.0, 0.0, 60.0)  # mm
    if sensor_type is SensorType.DEM:
        return (520.0, 60.0, 0.0, 3000.0)  # metres elevation
    if sensor_type is SensorType.HYPERSPECTRAL:
        return (0.2, 0.12, 0.0, 1.0)
    # ANCILLARY (embeddings, land-cover codes, reanalysis): generic normalised range.
    return (0.0, 1.0, -3.0, 3.0)


def _seasonal_offset(sensor_type: SensorType, t_idx: int, n_time: int) -> float:
    """A crop-phenology-like seasonal modulation across the time axis."""
    if n_time <= 1:
        return 0.0
    phase = np.sin(np.pi * t_idx / (n_time - 1))  # 0 -> 1 -> 0 (green-up then senescence)
    if sensor_type in (SensorType.OPTICAL, SensorType.HYPERSPECTRAL):
        return 0.15 * phase
    if sensor_type is SensorType.SAR:
        return 2.0 * phase  # dB rises with canopy/biomass
    if sensor_type is SensorType.RADIOMETER:
        return -0.05 * phase  # soil dries through the season
    if sensor_type is SensorType.THERMAL:
        return 5.0 * phase
    if sensor_type is SensorType.PRECIP:
        return 4.0 * phase  # monsoon hump
    return 0.0


def synth_stack_for_sensor(
    sensor: SensorSpec,
    *,
    aoi: BBox | Any = None,
    start: str = "2024-06-01",
    end: str = "2024-10-01",
    n_time: int = 6,
    size: int = 32,
    source: str = "synthetic",
) -> SyntheticStack:
    """Build a deterministic, physically-plausible :class:`SyntheticStack` for a sensor.

    The band set comes from the sensor's :attr:`SensorSpec.bands` (falling back to a
    single ``value`` band). Value ranges and the seasonal profile follow the sensor's
    :class:`~agristress.catalog.sensors.SensorType`, so e.g. SAR yields dB backscatter,
    optical yields reflectance/NDVI, precipitation yields mm, etc.
    """
    bbox = normalize_bbox(aoi)
    bands: tuple[str, ...] = tuple(sensor.bands) or ("value",)
    times = make_time_axis(start, end, n_time)
    rng = np.random.default_rng(_seed_from(sensor.id, bbox, start, end, n_time, size))

    cube = np.empty((len(times), size, size, len(bands)), dtype=np.float32)
    for ti in range(len(times)):
        for bi, band in enumerate(bands):
            base, spread, lo, hi = _family_profile(sensor.sensor_type, band)
            season = _seasonal_offset(sensor.sensor_type, ti, len(times))
            cube[ti, :, :, bi] = _band_field(rng, base, spread, season, size, lo, hi)

    provenance: dict[str, Any] = {
        "source": source,
        "demo": True,
        "sensor_id": sensor.id,
        "sensor_name": sensor.name,
        "sensor_type": sensor.sensor_type.value,
        "family": sensor.sensor_type.value.lower(),
        "gee_asset_id": sensor.gee_asset_id,
        "stac_collection": sensor.stac_collection,
        "native_resolution_m": sensor.native_resolution_m,
        "bbox": bbox,
        "start": start,
        "end": end,
        "synthetic": True,
        "note": "Offline synthetic stack — NOT real observations.",
    }
    return SyntheticStack(
        data=cube,
        sensor_id=sensor.id,
        bands=bands,
        times=times,
        bbox=bbox,
        provenance=provenance,
    )
