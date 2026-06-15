"""SAR preprocessing: speckle filtering, dB/linear conversion, backscatter features.

Covers the microwave leg of AgriStress (Sentinel-1 / EOS-04 / NISAR):

* :func:`refined_lee`    — working numpy Refined-Lee adaptive speckle filter
  (Lee et al. 1999) using the local-statistics MMSE form.
* :func:`to_db` / :func:`to_linear` — power <-> decibel conversion.
* :func:`terrain_flatten` — radiometric terrain flattening interface (RTC).
* :func:`compute_sar_features` — VV, VH, VH/VV ratio and the Radar Vegetation
  Index ``RVI = 4*VH / (VV + VH)``.

All numeric routines are pure numpy and run in DEMO mode on synthetic arrays
(no SNAP / GEE dependency).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

_EPS = 1e-12


# ---------------------------------------------------------------------------
# dB <-> linear
# ---------------------------------------------------------------------------
def to_db(linear: "NDArray", *, eps: float = _EPS) -> "NDArray":
    r"""Convert linear backscatter power to decibels: ``10 * log10(x)``.

    Non-positive inputs are floored at ``eps`` to avoid ``-inf`` / NaN.
    """
    x = np.asarray(linear, dtype=float)
    return 10.0 * np.log10(np.maximum(x, eps))


def to_linear(db: "NDArray") -> "NDArray":
    r"""Convert decibel backscatter to linear power: ``10 ** (dB / 10)``."""
    return np.power(10.0, np.asarray(db, dtype=float) / 10.0)


# ---------------------------------------------------------------------------
# Refined Lee speckle filter (Lee 1999, MMSE local-statistics form)
# ---------------------------------------------------------------------------
def _boxcar_mean(img: "NDArray", size: int) -> "NDArray":
    """Local mean over a ``size x size`` window with edge replication (numpy only)."""
    r = size // 2
    padded = np.pad(img, r, mode="edge")
    # Summed-area table for an O(N) sliding-window mean.
    csum = padded.cumsum(axis=0).cumsum(axis=1)
    csum = np.pad(csum, ((1, 0), (1, 0)), mode="constant")
    h, w = img.shape
    win = size
    total = (
        csum[win : win + h, win : win + w]
        - csum[0:h, win : win + w]
        - csum[win : win + h, 0:w]
        + csum[0:h, 0:w]
    )
    return total / float(win * win)


def refined_lee(img: "NDArray", size: int = 7, *, looks: float = 1.0) -> "NDArray":
    r"""Refined-Lee adaptive speckle filter for SAR intensity imagery.

    Implements the minimum-mean-square-error (MMSE) local-statistics estimator of
    Lee et al. (1999, *IEEE TGRS*). For each pixel the despeckled value is::

        R = mean + W * (pixel - mean)
        W = max(0, var_signal) / (var_signal + mean**2 * Cu**2)
        var_signal = (var_local - mean**2 * Cu**2) / (1 + Cu**2)

    where ``Cu = 1 / sqrt(looks)`` is the noise coefficient of variation for an
    ``looks``-look intensity image. In homogeneous regions ``W -> 0`` (heavy
    smoothing); over strong edges / point targets ``W -> 1`` (detail preserved).

    Parameters
    ----------
    img
        2-D SAR backscatter in **linear** power (not dB). NaNs are filled with the
        global mean before filtering.
    size
        Odd window size (default 7).
    looks
        Equivalent number of looks of the input (Sentinel-1 GRD ≈ 4.4). Sets the
        speckle noise level ``Cu``.

    Returns
    -------
    ndarray of float
        Despeckled image, same shape as ``img``.

    Notes
    -----
    This is the practical single-window MMSE Refined-Lee variant. The full
    Lee-1999 algorithm additionally selects one of eight edge-aligned sub-windows;
    that directional refinement is omitted here for speed/clarity but the adaptive
    weighting already suppresses speckle while retaining edges, which is sufficient
    for the AgriStress feature stack.
    """
    img = np.asarray(img, dtype=float)
    if img.ndim != 2:
        raise ValueError(f"refined_lee expects a 2-D image, got shape {img.shape}")
    if size < 3 or size % 2 == 0:
        raise ValueError(f"`size` must be an odd integer >= 3, got {size}")

    work = np.where(np.isfinite(img), img, np.nan)
    if np.isnan(work).any():
        work = np.nan_to_num(work, nan=float(np.nanmean(work)))

    cu2 = 1.0 / float(looks)  # (1/sqrt(L))**2

    mean = _boxcar_mean(work, size)
    mean_sq = _boxcar_mean(work * work, size)
    var_local = np.maximum(mean_sq - mean * mean, 0.0)

    var_signal = (var_local - (mean * mean) * cu2) / (1.0 + cu2)
    var_signal = np.maximum(var_signal, 0.0)

    denom = var_signal + (mean * mean) * cu2
    weight = np.where(denom > _EPS, var_signal / denom, 0.0)

    return mean + weight * (work - mean)


# ---------------------------------------------------------------------------
# Radiometric terrain flattening (interface)
# ---------------------------------------------------------------------------
def terrain_flatten(
    sigma0: "NDArray",
    *,
    local_incidence_angle_deg: "NDArray | None" = None,
    dem: "NDArray | None" = None,
    reference_angle_deg: float = 40.0,
) -> "NDArray":
    r"""Radiometric terrain flattening: sigma0 → terrain-flattened gamma0 (interface).

    Production deployments perform full Radiometric-Terrain-Correction (gamma-naught
    flattening, Small 2011) from a DEM and the SAR orbit geometry (e.g. via SNAP /
    GEE ``ee.Algorithms.Sentinel1.terrainCorrection`` or the ESA RTC processor).
    This entry point keeps that contract while providing a first-order cosine
    correction usable offline when only the *local incidence angle* (LIA) raster is
    available::

        gamma0 = sigma0 * cos(reference) / cos(LIA)

    Parameters
    ----------
    sigma0
        Backscatter in **linear** power.
    local_incidence_angle_deg
        Per-pixel local incidence angle in degrees. If ``None`` the input is
        returned unchanged (documented no-op) so the call site stays valid offline.
    dem
        Optional DEM placeholder for the production geometric path (unused in the
        offline cosine fallback).
    reference_angle_deg
        Reference incidence angle to normalise toward (default 40°).

    Returns
    -------
    ndarray of float
        Terrain-flattened backscatter (linear power).
    """
    sigma0 = np.asarray(sigma0, dtype=float)
    if local_incidence_angle_deg is None:
        # No geometry available offline — return unchanged rather than guess.
        return sigma0.copy()
    lia = np.deg2rad(np.asarray(local_incidence_angle_deg, dtype=float))
    ref = np.deg2rad(float(reference_angle_deg))
    cos_lia = np.maximum(np.cos(lia), _EPS)
    return sigma0 * (np.cos(ref) / cos_lia)


# ---------------------------------------------------------------------------
# SAR backscatter features
# ---------------------------------------------------------------------------
def compute_sar_features(
    vv: "NDArray",
    vh: "NDArray",
    *,
    input_in_db: bool = False,
    eps: float = _EPS,
) -> dict[str, "NDArray"]:
    r"""Derive the standard Sentinel-1 dual-pol backscatter feature set.

    Produces:

    * ``vv``, ``vh``      — co-/cross-pol backscatter in **dB**.
    * ``vh_vv_ratio``     — cross-to-co ratio ``VH/VV`` in **linear** power
      (crop-structure sensitive).
    * ``ratio_db``        — the same ratio expressed as ``VH_dB - VV_dB``.
    * ``rvi``             — Radar Vegetation Index ``4*VH / (VV + VH)`` (linear),
      bounded ~[0, 1+] and increasing with vegetation volume scattering.
    * ``cross_ratio_db``  — alias of ``ratio_db`` (common naming).

    Parameters
    ----------
    vv, vh
        Co-pol / cross-pol backscatter arrays (same shape).
    input_in_db
        Set ``True`` if ``vv``/``vh`` are already in dB; they are converted to
        linear power internally for the ratio / RVI math.
    eps
        Small floor to keep ratios finite.

    Returns
    -------
    dict[str, ndarray]
        Mapping of feature name → array (all same shape as the inputs).
    """
    vv = np.asarray(vv, dtype=float)
    vh = np.asarray(vh, dtype=float)
    if vv.shape != vh.shape:
        raise ValueError(f"VV {vv.shape} and VH {vh.shape} must share a shape")

    vv_lin = to_linear(vv) if input_in_db else np.maximum(vv, 0.0)
    vh_lin = to_linear(vh) if input_in_db else np.maximum(vh, 0.0)

    ratio = vh_lin / np.maximum(vv_lin, eps)
    rvi = (4.0 * vh_lin) / np.maximum(vv_lin + vh_lin, eps)

    vv_db = vv if input_in_db else to_db(vv_lin, eps=eps)
    vh_db = vh if input_in_db else to_db(vh_lin, eps=eps)
    ratio_db = vh_db - vv_db

    return {
        "vv": vv_db,
        "vh": vh_db,
        "vh_vv_ratio": ratio,
        "ratio_db": ratio_db,
        "cross_ratio_db": ratio_db,
        "rvi": rvi,
    }
