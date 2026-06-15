"""Spectral & SAR index calculators for AgriStress.

All functions are **vectorised** (operate elementwise on numpy arrays or
broadcastable scalars) and **NaN-safe**: divisions guard against a zero (or
near-zero) denominator by returning ``nan`` rather than raising or emitting a
``RuntimeWarning``.  Inputs are assumed to be surface-reflectance values
(typically in ``[0, 1]``); the functions do not rescale, so pass reflectance,
not raw DN.

Band naming follows the Sentinel-2 convention used throughout AgriStress:

==========  =======================  ============
Symbol      Sentinel-2 band          ~Wavelength
==========  =======================  ============
``blue``    B2                       490 nm
``green``   B3                       560 nm
``red``     B4                       665 nm
``red_edge``B5 / B6 / B7             705-783 nm
``nir``     B8                       842 nm
``nir_n``   B8A (narrow NIR)         865 nm
``swir1``   B11                      1610 nm
``swir2``   B12                      2190 nm
==========  =======================  ============

References
----------
* Rouse et al. 1974 (NDVI); Huete et al. 2002 (EVI); Jiang et al. 2008 (EVI2)
* Gao 1996 (NDWI, NIR-SWIR); McFeeters 1996 (NDWI, water)
* Wilson & Sader 2002 (NDMI); Gitelson 2005 (GCVI); Merzlyak 1999 (PSRI)
* Wang & Qu 2007 (NMDI); Huete 1988 (SAVI)
* Kim & Zerbe 1990 / Trudel 2012 (RVI for SAR)
* Sadeghi et al. 2017 (STR / OPTRAM transform)
"""

from __future__ import annotations

from typing import Union

import numpy as np

ArrayLike = Union[np.ndarray, float, int]

__all__ = [
    "safe_divide",
    "ndvi",
    "evi",
    "evi2",
    "ndwi_gao",
    "ndwi_mcfeeters",
    "ndmi",
    "ndre",
    "gcvi",
    "psri",
    "nmdi",
    "savi",
    "msavi",
    "normalized_difference",
    "sar_rvi",
    "sar_cross_ratio",
    "str_index",
    "stack_indices",
]

# Numerical floor for denominators.  Anything with |denominator| below this is
# treated as undefined and mapped to NaN.
_EPS = 1e-12


def _as_float(*arrays: ArrayLike) -> tuple[np.ndarray, ...]:
    """Convert inputs to float64 ndarrays (copy-free where possible)."""
    return tuple(np.asarray(a, dtype=np.float64) for a in arrays)


def safe_divide(
    numerator: ArrayLike,
    denominator: ArrayLike,
    *,
    fill: float = np.nan,
    eps: float = _EPS,
) -> np.ndarray:
    """Elementwise ``numerator / denominator`` returning ``fill`` where invalid.

    A result is invalid where ``|denominator| < eps`` or where either operand is
    NaN.  No divide-by-zero or invalid-value warnings are emitted.
    """
    num, den = _as_float(numerator, denominator)
    num, den = np.broadcast_arrays(num, den)
    out = np.full(np.broadcast_shapes(num.shape, den.shape), fill, dtype=np.float64)
    valid = (np.abs(den) >= eps) & np.isfinite(num) & np.isfinite(den)
    np.divide(num, den, out=out, where=valid)
    # Ensure positions that were excluded keep the fill value (np.divide leaves
    # `out` untouched where `where` is False, which is exactly what we want).
    out[~valid] = fill
    return out


def normalized_difference(a: ArrayLike, b: ArrayLike, *, fill: float = np.nan) -> np.ndarray:
    """Generic normalized difference ``(a - b) / (a + b)``.

    Most two-band indices (NDVI, NDWI, NDMI, NDRE, ...) are special cases of
    this helper; they are provided as named wrappers for clarity and validation.
    """
    a_arr, b_arr = _as_float(a, b)
    return safe_divide(a_arr - b_arr, a_arr + b_arr, fill=fill)


# --------------------------------------------------------------------------- #
# Greenness / vigour
# --------------------------------------------------------------------------- #
def ndvi(nir: ArrayLike, red: ArrayLike) -> np.ndarray:
    """Normalized Difference Vegetation Index ``(NIR - Red)/(NIR + Red)``.

    Range ``[-1, 1]``; healthy dense vegetation ~0.6-0.9.
    """
    return normalized_difference(nir, red)


def evi(
    nir: ArrayLike,
    red: ArrayLike,
    blue: ArrayLike,
    *,
    g: float = 2.5,
    c1: float = 6.0,
    c2: float = 7.5,
    l: float = 1.0,
) -> np.ndarray:
    """Enhanced Vegetation Index (Huete et al. 2002).

    ``G * (NIR - Red) / (NIR + C1*Red - C2*Blue + L)``.  Reduces canopy-background
    and atmospheric influence relative to NDVI; less prone to saturation.
    """
    nir_a, red_a, blue_a = _as_float(nir, red, blue)
    num = g * (nir_a - red_a)
    den = nir_a + c1 * red_a - c2 * blue_a + l
    return safe_divide(num, den)


def evi2(nir: ArrayLike, red: ArrayLike, *, g: float = 2.5, l: float = 1.0, c: float = 2.4) -> np.ndarray:
    """Two-band EVI (Jiang et al. 2008) — EVI without the blue band.

    ``G * (NIR - Red) / (NIR + C*Red + L)``.  Useful for sensors lacking a
    reliable blue band (e.g. some SAR-optical fusions / MODIS-style products).
    """
    nir_a, red_a = _as_float(nir, red)
    num = g * (nir_a - red_a)
    den = nir_a + c * red_a + l
    return safe_divide(num, den)


def savi(nir: ArrayLike, red: ArrayLike, *, l: float = 0.5) -> np.ndarray:
    """Soil-Adjusted Vegetation Index (Huete 1988).

    ``(1 + L) * (NIR - Red) / (NIR + Red + L)``.  ``L`` (soil brightness factor)
    is 0.5 for intermediate cover; ``L -> 0`` recovers NDVI.
    """
    nir_a, red_a = _as_float(nir, red)
    num = (1.0 + l) * (nir_a - red_a)
    den = nir_a + red_a + l
    return safe_divide(num, den)


def msavi(nir: ArrayLike, red: ArrayLike) -> np.ndarray:
    """Modified SAVI (Qi et al. 1994) — self-adjusting soil factor.

    ``(2*NIR + 1 - sqrt((2*NIR+1)^2 - 8*(NIR-Red))) / 2``.
    """
    nir_a, red_a = _as_float(nir, red)
    term = (2.0 * nir_a + 1.0) ** 2 - 8.0 * (nir_a - red_a)
    # Clamp negative radicands (numerically impossible region) to NaN.
    term = np.where(term < 0, np.nan, term)
    return (2.0 * nir_a + 1.0 - np.sqrt(term)) / 2.0


def gcvi(nir: ArrayLike, green: ArrayLike) -> np.ndarray:
    """Green Chlorophyll Vegetation Index (Gitelson 2005) ``NIR/Green - 1``.

    Sensitive to canopy chlorophyll content; unbounded above.
    """
    nir_a, green_a = _as_float(nir, green)
    return safe_divide(nir_a, green_a) - 1.0


def ndre(nir: ArrayLike, red_edge: ArrayLike) -> np.ndarray:
    """Normalized Difference Red-Edge ``(NIR - RedEdge)/(NIR + RedEdge)``.

    Tracks chlorophyll / nitrogen status; saturates later than NDVI.
    """
    return normalized_difference(nir, red_edge)


# --------------------------------------------------------------------------- #
# Water / moisture
# --------------------------------------------------------------------------- #
def ndwi_gao(nir: ArrayLike, swir1: ArrayLike) -> np.ndarray:
    """Gao (1996) NDWI — **vegetation liquid water** ``(NIR - SWIR1)/(NIR + SWIR1)``.

    This is the canopy-water flavour (a.k.a. NDII / LSWI), sensitive to leaf
    water content.  Distinct from the McFeeters open-water NDWI.
    """
    return normalized_difference(nir, swir1)


def ndwi_mcfeeters(green: ArrayLike, nir: ArrayLike) -> np.ndarray:
    """McFeeters (1996) NDWI — **open water** ``(Green - NIR)/(Green + NIR)``.

    Positive over water bodies; used for delineating canals / ponded fields.
    """
    return normalized_difference(green, nir)


def ndmi(nir_narrow: ArrayLike, swir1: ArrayLike) -> np.ndarray:
    """Normalized Difference Moisture Index ``(B8A - B11)/(B8A + B11)``.

    Uses the *narrow* NIR (Sentinel-2 B8A, 865 nm) with SWIR1 (B11) for canopy
    moisture (Wilson & Sader 2002).
    """
    return normalized_difference(nir_narrow, swir1)


def nmdi(nir: ArrayLike, swir1: ArrayLike, swir2: ArrayLike) -> np.ndarray:
    """Normalized Multi-band Drought Index (Wang & Qu 2007).

    ``(NIR - (SWIR1 - SWIR2)) / (NIR + (SWIR1 - SWIR2))``.  Combines the two SWIR
    bands to be jointly sensitive to soil and vegetation water content.
    """
    nir_a, s1, s2 = _as_float(nir, swir1, swir2)
    diff = s1 - s2
    return safe_divide(nir_a - diff, nir_a + diff)


def psri(red: ArrayLike, blue: ArrayLike, red_edge: ArrayLike) -> np.ndarray:
    """Plant Senescence Reflectance Index (Merzlyak et al. 1999).

    ``(Red - Blue) / RedEdge``.  Increases as canopy senesces (carotenoid /
    chlorophyll ratio); a useful end-of-season / stress marker.
    """
    red_a, blue_a, re_a = _as_float(red, blue, red_edge)
    return safe_divide(red_a - blue_a, re_a)


# --------------------------------------------------------------------------- #
# SAR (microwave)
# --------------------------------------------------------------------------- #
def sar_rvi(vv: ArrayLike, vh: ArrayLike) -> np.ndarray:
    """Radar Vegetation Index for dual-pol (VV/VH) SAR.

    ``RVI = 4*VH / (VV + VH)`` (linear-power inputs).  Ranges ~``[0, ~1+]``;
    increases with volume scattering from vegetation.  **Pass linear power**, not
    dB.  If you have dB, convert with ``10**(db/10)`` first.
    """
    vv_a, vh_a = _as_float(vv, vh)
    return safe_divide(4.0 * vh_a, vv_a + vh_a)


def sar_cross_ratio(vv: ArrayLike, vh: ArrayLike) -> np.ndarray:
    """Cross-polarisation ratio ``VH / VV`` (linear power).

    A simple, widely used crop-structure / biomass proxy.
    """
    vv_a, vh_a = _as_float(vv, vh)
    return safe_divide(vh_a, vv_a)


# --------------------------------------------------------------------------- #
# OPTRAM transform
# --------------------------------------------------------------------------- #
def str_index(swir: ArrayLike) -> np.ndarray:
    """Swir Transformed Reflectance for the OPTRAM model (Sadeghi et al. 2017).

    ``STR = (1 - SWIR)^2 / (2 * SWIR)``.

    STR is monotonically related to surface soil moisture; together with NDVI it
    forms the OPTRAM trapezoid used by :func:`agristress.models.stress.optram_soil_moisture`.
    ``SWIR`` must be surface reflectance in ``[0, 1]``.
    """
    swir_a = _as_float(swir)[0]
    num = (1.0 - swir_a) ** 2
    den = 2.0 * swir_a
    return safe_divide(num, den)


# --------------------------------------------------------------------------- #
# Convenience: compute a dict / stack of indices from a band mapping
# --------------------------------------------------------------------------- #
def stack_indices(bands: dict[str, ArrayLike], which: list[str] | None = None) -> dict[str, np.ndarray]:
    """Compute a set of indices from a ``{band_name: array}`` mapping.

    Parameters
    ----------
    bands
        Mapping with any of: ``blue, green, red, red_edge, nir, nir_n,
        swir1, swir2, vv, vh``.  Missing bands simply skip the indices that
        need them.
    which
        Subset of index names to compute.  ``None`` computes every index whose
        required bands are available.

    Returns
    -------
    dict
        ``{index_name: ndarray}`` for every successfully computed index.

    Notes
    -----
    Intended for building a multi-temporal feature stack; safe to call per
    time-step and concatenate.  Never raises for missing bands — it just omits
    the corresponding index.
    """
    b = bands

    def has(*names: str) -> bool:
        return all(n in b and b[n] is not None for n in names)

    recipes: dict[str, tuple[tuple[str, ...], object]] = {
        "ndvi": (("nir", "red"), lambda: ndvi(b["nir"], b["red"])),
        "evi": (("nir", "red", "blue"), lambda: evi(b["nir"], b["red"], b["blue"])),
        "evi2": (("nir", "red"), lambda: evi2(b["nir"], b["red"])),
        "savi": (("nir", "red"), lambda: savi(b["nir"], b["red"])),
        "msavi": (("nir", "red"), lambda: msavi(b["nir"], b["red"])),
        "gcvi": (("nir", "green"), lambda: gcvi(b["nir"], b["green"])),
        "ndre": (("nir", "red_edge"), lambda: ndre(b["nir"], b["red_edge"])),
        "ndwi_gao": (("nir", "swir1"), lambda: ndwi_gao(b["nir"], b["swir1"])),
        "ndwi_mcfeeters": (("green", "nir"), lambda: ndwi_mcfeeters(b["green"], b["nir"])),
        "ndmi": (("nir_n", "swir1"), lambda: ndmi(b["nir_n"], b["swir1"])),
        "nmdi": (("nir", "swir1", "swir2"), lambda: nmdi(b["nir"], b["swir1"], b["swir2"])),
        "psri": (("red", "blue", "red_edge"), lambda: psri(b["red"], b["blue"], b["red_edge"])),
        "sar_rvi": (("vv", "vh"), lambda: sar_rvi(b["vv"], b["vh"])),
        "sar_cross_ratio": (("vv", "vh"), lambda: sar_cross_ratio(b["vv"], b["vh"])),
        "str": (("swir1",), lambda: str_index(b["swir1"])),
    }

    names = which if which is not None else list(recipes)
    out: dict[str, np.ndarray] = {}
    for name in names:
        if name not in recipes:
            continue
        required, fn = recipes[name]
        if has(*required):
            out[name] = fn()  # type: ignore[operator]
    return out
