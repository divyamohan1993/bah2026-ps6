"""Soil-moisture fusion: SWI exponential filter, SMAP downscaling, triple collocation.

* :func:`swi_exponential_filter` — recursive Soil Water Index (Wagner et al. 1999)
  propagating surface soil moisture into a root-zone estimate.
* :func:`downscale_smap`         — disaggregate coarse SMAP soil moisture to field
  scale using S1/optical predictors (regression interface, demo fallback).
* :func:`triple_collocation`     — classical triple-collocation error-variance
  estimator (Stoffelen 1998; McColl et al. 2014) for three collocated SM products.

All numpy based and offline-runnable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Soil Water Index — recursive exponential filter (Wagner 1999)
# ---------------------------------------------------------------------------
def swi_exponential_filter(
    ssm: NDArray,
    time: NDArray,
    T: float = 10.0,
) -> NDArray:
    r"""Soil Water Index from surface soil moisture via a recursive exponential filter.

    Propagates surface soil moisture (SSM) into a root-zone Soil Water Index using
    the recursive formulation of Wagner et al. (1999, *RSE* 70:191) as implemented
    by Albergel et al. (2008)::

        K_n = K_{n-1} / ( K_{n-1} + exp(-(t_n - t_{n-1}) / T) )
        SWI_n = SWI_{n-1} + K_n * ( SSM_n - SWI_{n-1} )

    with ``SWI_0 = SSM_0`` and ``K_0 = 1``. ``T`` is the characteristic time length
    of the soil layer (days): larger ``T`` ⇒ slower, smoother root-zone response.

    Parameters
    ----------
    ssm
        Surface soil-moisture observations (1-D, time-ordered). ``NaN`` gaps are
        carried over (SWI persists, gain decays) until the next valid observation.
    time
        Observation times (1-D, same length). Either ``datetime64`` or numeric days;
        only consecutive **differences** in days are used.
    T
        Characteristic time length in days (default 10).

    Returns
    -------
    ndarray of float
        SWI series, same length as ``ssm``.
    """
    ssm = np.asarray(ssm, dtype=float)
    time = np.asarray(time)
    if ssm.ndim != 1:
        raise ValueError("swi_exponential_filter expects a 1-D SSM series")
    if ssm.size != time.size:
        raise ValueError("ssm and time must have equal length")
    n = ssm.size
    if n == 0:
        return ssm.copy()

    # Convert times to days-from-start.
    if np.issubdtype(time.dtype, np.datetime64):
        t_days = (time - time[0]) / np.timedelta64(1, "D")
        t_days = t_days.astype(float)
    else:
        t_days = time.astype(float) - float(time[0])

    swi = np.full(n, np.nan, dtype=float)
    gain = 1.0
    last_swi: float | None = None
    last_t: float | None = None

    for i in range(n):
        if not np.isfinite(ssm[i]):
            swi[i] = last_swi if last_swi is not None else np.nan
            continue
        if last_swi is None:  # first valid observation seeds the filter
            swi[i] = ssm[i]
            last_swi = ssm[i]
            last_t = t_days[i]
            gain = 1.0
            continue
        dt = max(t_days[i] - last_t, 0.0)
        gain = gain / (gain + np.exp(-dt / float(T)))
        last_swi = last_swi + gain * (ssm[i] - last_swi)
        swi[i] = last_swi
        last_t = t_days[i]

    return swi


# ---------------------------------------------------------------------------
# SMAP downscaling (interface + demo regression)
# ---------------------------------------------------------------------------
def downscale_smap(
    smap: NDArray,
    s1: NDArray | None = None,
    optical: NDArray | None = None,
    *,
    model=None,
) -> NDArray:
    r"""Disaggregate coarse SMAP soil moisture to fine resolution (interface).

    Operational disaggregation (e.g. SMAP/Sentinel-1 active-passive, DISPATCH)
    learns ``SM_fine = f(sigma0_VV, sigma0_VH, NDVI, LST, ...)`` from coarse
    co-located samples and applies it at fine scale, conserving the coarse mean.
    Provide that fitted regressor as ``model`` (``.predict`` on stacked predictors).

    The offline demo path applies a transparent linear adjustment that nudges the
    (broadcast) SMAP field by mean-removed S1 and optical predictors::

        SM_fine = SMAP + 0.10 * z(VV) - 0.05 * z(NDVI)

    where ``z`` is a robust standardisation. With no predictors it returns ``smap``
    unchanged. The result is clipped to a physical ``[0, 0.6] m^3/m^3`` range.

    Parameters
    ----------
    smap
        Coarse soil-moisture field (already resampled to the fine grid).
    s1
        Fine-scale SAR predictor (e.g. VV in dB), same shape as ``smap``.
    optical
        Fine-scale optical predictor (e.g. NDVI), same shape as ``smap``.
    model
        Optional fitted regressor; if given, predicts from stacked ``[s1, optical]``.

    Returns
    -------
    ndarray of float
        Downscaled soil moisture.
    """
    smap = np.asarray(smap, dtype=float)

    if model is not None:
        feats = [smap]
        if s1 is not None:
            feats.append(np.asarray(s1, dtype=float))
        if optical is not None:
            feats.append(np.asarray(optical, dtype=float))
        stack = np.stack(feats, axis=-1)
        flat = stack.reshape(-1, stack.shape[-1])
        pred = np.asarray(model.predict(flat)).reshape(smap.shape)
        return np.clip(pred, 0.0, 0.6)

    def _z(arr):
        arr = np.asarray(arr, dtype=float)
        med = np.nanmedian(arr)
        mad = np.nanmedian(np.abs(arr - med)) + _EPS
        return (arr - med) / (1.4826 * mad)

    out = smap.copy()
    if s1 is not None:
        out = out + 0.10 * _z(s1) * np.nanstd(smap)
    if optical is not None:
        out = out - 0.05 * _z(optical) * np.nanstd(smap)
    return np.clip(out, 0.0, 0.6)


# ---------------------------------------------------------------------------
# Triple collocation
# ---------------------------------------------------------------------------
def triple_collocation(
    x: NDArray,
    y: NDArray,
    z: NDArray,
    *,
    ddof: int = 1,
) -> dict[str, object]:
    r"""Triple-collocation error-variance estimation for three SM products.

    Given three collocated estimates of the same geophysical variable with mutually
    uncorrelated errors, the covariance method (Stoffelen 1998; McColl et al. 2014,
    *GRL* 41:6229) recovers each product's random-error variance::

        var_e_x = Cxx - (Cxy * Cxz) / Cyz
        var_e_y = Cyy - (Cxy * Cyz) / Cxz
        var_e_z = Czz - (Cxz * Cyz) / Cxy

    and the correlation of each product with the unknown truth (TC ``R``)::

        Rx^2 = (Cxy * Cxz) / (Cxx * Cyz)   (etc.)

    Parameters
    ----------
    x, y, z
        Three collocated time series (1-D, equal length). Rows with any ``NaN`` are
        dropped pairwise-consistently before the covariances are formed.
    ddof
        Delta-degrees-of-freedom for the covariance (default 1, unbiased).

    Returns
    -------
    dict
        ``{"var_err": {"x","y","z"}, "rho": {...}, "snr_db": {...}, "n": int}``
        where ``var_err`` are the random-error variances (negative values, a sign of
        violated assumptions / small samples, are clipped to 0), ``rho`` the TC
        correlation-with-truth, and ``snr_db`` the signal-to-noise ratio in dB.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    z = np.asarray(z, dtype=float).ravel()
    if not (x.size == y.size == z.size):
        raise ValueError("triple_collocation requires three equal-length series")

    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if m.sum() < 3:
        raise ValueError("need >= 3 jointly-finite samples for triple collocation")
    x, y, z = x[m], y[m], z[m]

    cxx = np.cov(x, x, ddof=ddof)[0, 1]
    cyy = np.cov(y, y, ddof=ddof)[0, 1]
    czz = np.cov(z, z, ddof=ddof)[0, 1]
    cxy = np.cov(x, y, ddof=ddof)[0, 1]
    cxz = np.cov(x, z, ddof=ddof)[0, 1]
    cyz = np.cov(y, z, ddof=ddof)[0, 1]

    def _safe(a, b):
        return a / b if abs(b) > _EPS else np.nan

    var_ex = cxx - _safe(cxy * cxz, cyz)
    var_ey = cyy - _safe(cxy * cyz, cxz)
    var_ez = czz - _safe(cxz * cyz, cxy)

    # Signal variances (product-resolution wrt truth).
    sig_x = _safe(cxy * cxz, cyz)
    sig_y = _safe(cxy * cyz, cxz)
    sig_z = _safe(cxz * cyz, cxy)

    rho_x2 = _safe(cxy * cxz, cxx * cyz)
    rho_y2 = _safe(cxy * cyz, cyy * cxz)
    rho_z2 = _safe(cxz * cyz, czz * cxy)

    def _clip0(v):
        return float(max(v, 0.0)) if np.isfinite(v) else float("nan")

    def _rho(v):
        return float(np.sqrt(v)) if np.isfinite(v) and v >= 0 else float("nan")

    def _snr_db(sig, err):
        if np.isfinite(sig) and np.isfinite(err) and err > _EPS and sig > 0:
            return float(10.0 * np.log10(sig / err))
        return float("nan")

    return {
        "var_err": {"x": _clip0(var_ex), "y": _clip0(var_ey), "z": _clip0(var_ez)},
        "rho": {"x": _rho(rho_x2), "y": _rho(rho_y2), "z": _rho(rho_z2)},
        "snr_db": {
            "x": _snr_db(sig_x, var_ex),
            "y": _snr_db(sig_y, var_ey),
            "z": _snr_db(sig_z, var_ez),
        },
        "n": int(m.sum()),
    }
