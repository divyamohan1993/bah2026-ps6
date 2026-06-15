"""Optical preprocessing: cloud / shadow masking and surface-reflectance harmonisation.

The functions here cover the *optical* leg of the AgriStress fusion pipeline
(Sentinel-2 / Landsat-8/9 / LISS / AWiFS):

* **Cloud & shadow masking** — three interchangeable strategies are exposed:
  - :func:`mask_s2_scl`      — Sen2Cor Scene-Classification-Layer (SCL) based mask.
  - :func:`cloud_score_plus` — Google CloudScore+ (``cs`` / ``cs_cdf`` band, 0..1).
  - :func:`fmask`            — CFMask / Fmask 4.x wrapper.
* :func:`apply_cloud_mask`  — pure-numpy masker that works in DEMO mode.
* :func:`scale_surface_reflectance` — DN → reflectance scaling.
* :func:`harmonize_bandpass` — HLS Sentinel-2 → Landsat-8 OLI bandpass adjustment
  (Claverie et al. 2018), real and simple linear math.

Everything is import-clean and runs without Earth Engine / network access; the
service-backed strategies fall back to a documented no-op / heuristic so the rest
of the pipeline stays testable offline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Sentinel-2 Scene Classification Layer (SCL) class codes (Sen2Cor L2A).
# https://sentinels.copernicus.eu/web/sentinel/technical-guides/sentinel-2-msi/level-2a/algorithm
# ---------------------------------------------------------------------------
SCL_NO_DATA = 0
SCL_SATURATED = 1
SCL_DARK_AREA = 2
SCL_CLOUD_SHADOW = 3
SCL_VEGETATION = 4
SCL_BARE_SOIL = 5
SCL_WATER = 6
SCL_CLOUD_LOW = 7
SCL_CLOUD_MEDIUM = 8
SCL_CLOUD_HIGH = 9
SCL_THIN_CIRRUS = 10
SCL_SNOW = 11

#: SCL classes treated as *invalid* (cloud / shadow / saturated / no-data) by default.
SCL_INVALID_CLASSES: tuple[int, ...] = (
    SCL_NO_DATA,
    SCL_SATURATED,
    SCL_DARK_AREA,
    SCL_CLOUD_SHADOW,
    SCL_CLOUD_MEDIUM,
    SCL_CLOUD_HIGH,
    SCL_THIN_CIRRUS,
)

#: Default CloudScore+ clear-probability threshold. Pixels with ``cs`` >= this are
#: kept. Google recommends 0.50–0.65 depending on aggressiveness (0.60 conservative).
CLOUD_SCORE_PLUS_THRESHOLD = 0.60


# ---------------------------------------------------------------------------
# Masking strategies
# ---------------------------------------------------------------------------
def mask_s2_scl(
    scl: "NDArray",
    invalid_classes: tuple[int, ...] = SCL_INVALID_CLASSES,
) -> "NDArray":
    """Boolean *valid* mask from a Sentinel-2 SCL band.

    Parameters
    ----------
    scl
        Scene-Classification-Layer array (integer class codes 0..11), any shape.
    invalid_classes
        SCL codes to treat as cloud / shadow / invalid. Defaults to
        :data:`SCL_INVALID_CLASSES`.

    Returns
    -------
    ndarray of bool
        ``True`` where the pixel is *clear / usable*, ``False`` otherwise. Same
        shape as ``scl``.
    """
    scl = np.asarray(scl)
    invalid = np.isin(scl, np.asarray(invalid_classes))
    return ~invalid


def cloud_score_plus(
    cs: "NDArray",
    threshold: float = CLOUD_SCORE_PLUS_THRESHOLD,
) -> "NDArray":
    """Boolean *valid* mask from a Google CloudScore+ quality band.

    CloudScore+ (Pasquarella et al. 2023) emits a continuous clear-probability in
    ``[0, 1]`` (``cs`` = spectral distance; ``cs_cdf`` = cumulative form). Pixels at
    or above ``threshold`` are considered clear.

    Parameters
    ----------
    cs
        CloudScore+ band, values in ``[0, 1]``. NaNs are treated as cloudy.
    threshold
        Clear-probability cut-off (~0.5–0.6). Default 0.60.

    Returns
    -------
    ndarray of bool
        ``True`` where ``cs >= threshold`` (and not NaN).
    """
    cs = np.asarray(cs, dtype=float)
    return np.nan_to_num(cs, nan=0.0) >= float(threshold)


def fmask(
    *,
    qa: "NDArray | None" = None,
    cloud_bit: int = 3,
    shadow_bit: int = 4,
    backend: str | None = None,
) -> "NDArray":
    """Fmask / CFMask cloud-shadow mask wrapper (interface + bit-field fallback).

    In production this dispatches to the Fmask 4.x executable or the Landsat
    Collection-2 ``QA_PIXEL`` bit field. With no backend available it decodes the
    standard Landsat C2 ``QA_PIXEL`` bits (cloud=bit 3, cloud-shadow=bit 4) so the
    function stays usable offline.

    Parameters
    ----------
    qa
        Packed QA bit-field array (e.g. Landsat ``QA_PIXEL``). Required for the
        fallback path.
    cloud_bit, shadow_bit
        Bit positions for cloud / cloud-shadow flags.
    backend
        Reserved for selecting an external Fmask implementation; ``None`` uses the
        bit-field fallback.

    Returns
    -------
    ndarray of bool
        ``True`` where clear.

    Raises
    ------
    ValueError
        If neither a backend nor a ``qa`` bit-field is supplied.
    """
    if backend is not None:  # pragma: no cover - external tool not available offline
        raise NotImplementedError(
            f"External Fmask backend {backend!r} requires the fmask package / executable; "
            "pass a QA_PIXEL bit-field for the offline fallback instead."
        )
    if qa is None:
        raise ValueError("fmask fallback needs a `qa` bit-field array (e.g. Landsat QA_PIXEL).")
    qa = np.asarray(qa).astype(np.int64)
    cloud = (qa >> cloud_bit) & 1
    shadow = (qa >> shadow_bit) & 1
    return (cloud == 0) & (shadow == 0)


# ---------------------------------------------------------------------------
# Apply a mask to a band stack (pure numpy, DEMO-ready)
# ---------------------------------------------------------------------------
def apply_cloud_mask(
    stack: "NDArray",
    qa: "NDArray",
    *,
    fill: float = np.nan,
    qa_kind: str = "scl",
    threshold: float = CLOUD_SCORE_PLUS_THRESHOLD,
    invalid_classes: tuple[int, ...] = SCL_INVALID_CLASSES,
) -> "NDArray":
    """Mask cloudy / shadowed pixels in a reflectance stack with a fill value.

    Pure-numpy and broadcast-aware so it works in DEMO mode on synthetic arrays.

    Parameters
    ----------
    stack
        Reflectance array. Either ``(bands, H, W)`` or ``(H, W)``. A leading band
        axis is detected when ``stack.ndim == qa.ndim + 1``.
    qa
        Quality array shaped like a single band ``(..., H, W)``.
    fill
        Value written where invalid (default ``np.nan``). The output is upcast to
        float when ``fill`` is NaN.
    qa_kind
        ``"scl"``  → interpret ``qa`` with :func:`mask_s2_scl`;
        ``"cs"`` / ``"cloudscore"`` → :func:`cloud_score_plus`;
        ``"bool"`` → ``qa`` already a boolean *valid* mask.
    threshold
        Threshold forwarded to :func:`cloud_score_plus` when ``qa_kind`` is a score.
    invalid_classes
        SCL codes forwarded to :func:`mask_s2_scl`.

    Returns
    -------
    ndarray
        Copy of ``stack`` with invalid pixels set to ``fill``.
    """
    stack = np.asarray(stack)
    qa = np.asarray(qa)

    kind = qa_kind.lower()
    if kind == "scl":
        valid = mask_s2_scl(qa, invalid_classes=invalid_classes)
    elif kind in {"cs", "cloudscore", "cloud_score_plus"}:
        valid = cloud_score_plus(qa, threshold=threshold)
    elif kind == "bool":
        valid = qa.astype(bool)
    else:  # pragma: no cover - defensive
        raise ValueError(f"unknown qa_kind {qa_kind!r}; expected scl|cs|bool")

    # Promote dtype if we are going to write NaN into an integer stack.
    out = stack.astype(float) if (np.isnan(fill) and not np.issubdtype(stack.dtype, np.floating)) else stack.copy()

    if out.ndim == valid.ndim + 1:
        # leading band axis -> broadcast the mask across bands
        out[:, ~valid] = fill
    else:
        out[~valid] = fill
    return out


# ---------------------------------------------------------------------------
# Surface reflectance scaling
# ---------------------------------------------------------------------------
def scale_surface_reflectance(
    dn: "NDArray",
    *,
    scale: float = 1e-4,
    offset: float = 0.0,
    clip: bool = True,
) -> "NDArray":
    """Convert integer DN to physical surface reflectance ``[0, 1]``.

    Defaults match Sentinel-2 L2A / HLS (``scale=1e-4``). Landsat Collection-2
    surface reflectance uses ``scale=2.75e-5, offset=-0.2``.

    Parameters
    ----------
    dn
        Raw digital numbers.
    scale, offset
        Linear scaling: ``reflectance = dn * scale + offset``.
    clip
        Clip the result to ``[0, 1]`` (default ``True``).

    Returns
    -------
    ndarray of float
    """
    refl = np.asarray(dn, dtype=float) * float(scale) + float(offset)
    if clip:
        refl = np.clip(refl, 0.0, 1.0)
    return refl


# ---------------------------------------------------------------------------
# HLS Sentinel-2 -> Landsat-8 OLI bandpass adjustment (Claverie et al. 2018).
# reflectance_OLI = slope * reflectance_S2 + intercept
# Keys are canonical band names; aliases are resolved case-insensitively.
# ---------------------------------------------------------------------------
HLS_S2_TO_OLI: dict[str, tuple[float, float]] = {
    "blue": (0.9778, -0.0040),
    "green": (1.0053, -0.0009),
    "red": (0.9765, 0.0009),
    "nir": (0.9983, -0.0001),
    "swir1": (0.9987, -0.0011),
    "swir2": (1.0030, -0.0012),
}

#: Common band-name aliases → canonical key used by :data:`HLS_S2_TO_OLI`.
_BAND_ALIASES: dict[str, str] = {
    "b": "blue",
    "b2": "blue",
    "g": "green",
    "b3": "green",
    "r": "red",
    "b4": "red",
    "n": "nir",
    "nir8": "nir",
    "b8": "nir",
    "b8a": "nir",
    "swir": "swir1",
    "swir16": "swir1",
    "b11": "swir1",
    "swir22": "swir2",
    "b12": "swir2",
}


def _canonical_band(sensor: str) -> str:
    key = sensor.strip().lower()
    if key in HLS_S2_TO_OLI:
        return key
    if key in _BAND_ALIASES:
        return _BAND_ALIASES[key]
    raise KeyError(
        f"no HLS bandpass coefficient for band {sensor!r}; "
        f"known bands: {sorted(HLS_S2_TO_OLI)}"
    )


def harmonize_bandpass(
    reflectance: "NDArray",
    sensor: str,
    *,
    clip: bool = True,
) -> "NDArray":
    """Apply the HLS Sentinel-2 → Landsat-8 OLI bandpass adjustment.

    Implements the per-band linear transform of Claverie et al. (2018,
    *Remote Sensing of Environment* 219:145-161, the HLS v1.4 product)::

        reflectance_OLI = slope * reflectance_S2 + intercept

    so that Sentinel-2 MSI and Landsat OLI surface reflectance can be used
    interchangeably in a single harmonised time series.

    Coefficients (slope, intercept):

    ====== ======== =========
    band   slope    intercept
    ====== ======== =========
    blue   0.9778   -0.0040
    green  1.0053   -0.0009
    red    0.9765   +0.0009
    nir    0.9983   -0.0001
    swir1  0.9987   -0.0011
    swir2  1.0030   -0.0012
    ====== ======== =========

    Parameters
    ----------
    reflectance
        Sentinel-2 surface reflectance for a **single** band (any shape), already
        scaled to ``[0, 1]``.
    sensor
        Band identifier. Canonical names (``blue``/``green``/``red``/``nir``/
        ``swir1``/``swir2``) or aliases (``B2``, ``B8A``, ``SWIR16`` …) are accepted.
    clip
        Clip the harmonised output to ``[0, 1]`` (default ``True``).

    Returns
    -------
    ndarray of float
        Harmonised OLI-equivalent reflectance, same shape as ``reflectance``.
    """
    slope, intercept = HLS_S2_TO_OLI[_canonical_band(sensor)]
    out = np.asarray(reflectance, dtype=float) * slope + intercept
    if clip:
        out = np.clip(out, 0.0, 1.0)
    return out
