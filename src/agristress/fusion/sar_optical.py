"""SAR-optical fusion & temporal reconstruction.

* :func:`whittaker_smooth`      — Whittaker smoother (Eilers 2003) with a sparse
  2nd-difference roughness penalty; fills missing values and denoises a series.
* :func:`savitzky_golay`        — Savitzky-Golay polynomial smoother (gap-aware).
* :func:`gap_fill_temporal`     — dispatcher over the two smoothers for an NDVI /
  index time series with cloud gaps (NaNs).
* :func:`sar_to_ndvi`           — translate SAR features → NDVI; RF-regression
  interface with a fitted-linear demo fallback (all-weather NDVI proxy).
* :func:`multi_sensor_consensus`— robust per-pixel fusion of several NDVI estimates
  via weighted median + MAD outlier rejection.

All routines are numpy/scipy based and run offline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

_EPS = 1e-9


# ---------------------------------------------------------------------------
# Whittaker smoother (Eilers 2003)
# ---------------------------------------------------------------------------
def whittaker_smooth(
    y: "NDArray",
    *,
    lambda_: float = 10.0,
    d: int = 2,
    weights: "NDArray | None" = None,
) -> "NDArray":
    r"""Whittaker–Henderson smoother with a discrete roughness penalty.

    Minimises ``||W^{1/2}(y - z)||^2 + lambda * ||D_d z||^2`` where ``D_d`` is the
    ``d``-th order finite-difference operator (Eilers 2003, *Anal. Chem.* 75:3631).
    Missing samples are handled by giving them zero weight, so the smoother also
    **interpolates gaps** by extrapolating the smooth trend.

    Solved as the banded linear system ``(W + lambda * D'D) z = W y``. Uses
    ``scipy.sparse`` when available, otherwise a dense numpy solve.

    Parameters
    ----------
    y
        1-D series. ``NaN`` entries are treated as missing (weight 0).
    lambda_
        Smoothing strength (larger ⇒ smoother). Typical 1–1e4.
    d
        Order of the difference penalty (2 ⇒ penalise curvature; default).
    weights
        Optional per-sample weights in ``[0, 1]``. Multiplied with the
        finite-mask weights derived from ``y``.

    Returns
    -------
    ndarray of float
        Smoothed/gap-filled series, same length as ``y``.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("whittaker_smooth expects a 1-D series")
    n = y.size
    if n <= d + 1:
        return y.copy()

    finite = np.isfinite(y)
    w = finite.astype(float)
    if weights is not None:
        w = w * np.asarray(weights, dtype=float)
    y_filled = np.where(finite, y, 0.0)

    try:  # sparse path (preferred)
        from scipy import sparse
        from scipy.sparse.linalg import spsolve

        ident = sparse.eye(n, format="csc")
        diff = ident.copy()
        for _ in range(d):
            diff = diff[1:] - diff[:-1]
        wmat = sparse.diags(w)
        lhs = (wmat + lambda_ * (diff.T @ diff)).tocsc()
        z = spsolve(lhs, w * y_filled)
        return np.asarray(z, dtype=float)
    except Exception:  # pragma: no cover - dense fallback (no scipy)
        diff = np.diff(np.eye(n), n=d, axis=0)
        lhs = np.diag(w) + lambda_ * (diff.T @ diff)
        return np.linalg.solve(lhs, w * y_filled)


# ---------------------------------------------------------------------------
# Savitzky-Golay smoother (gap-aware)
# ---------------------------------------------------------------------------
def savitzky_golay(
    y: "NDArray",
    *,
    window: int = 7,
    polyorder: int = 2,
) -> "NDArray":
    """Savitzky-Golay polynomial smoother, NaN-aware.

    Gaps are linearly interpolated first (so the convolution is well-defined), then
    a least-squares polynomial of order ``polyorder`` is fitted in a sliding
    ``window``. Uses ``scipy.signal.savgol_filter`` when available, else a numpy
    fallback.

    Parameters
    ----------
    y
        1-D series (NaN = missing).
    window
        Odd window length (default 7).
    polyorder
        Polynomial order (< ``window``).

    Returns
    -------
    ndarray of float
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    if window % 2 == 0:
        window += 1
    if window > n:
        window = n if n % 2 == 1 else n - 1
    if window < 3 or polyorder >= window:
        return y.copy()

    finite = np.isfinite(y)
    if not finite.any():
        return y.copy()
    idx = np.arange(n)
    filled = np.interp(idx, idx[finite], y[finite])

    try:
        from scipy.signal import savgol_filter

        return np.asarray(savgol_filter(filled, window, polyorder), dtype=float)
    except Exception:  # pragma: no cover - numpy fallback
        r = window // 2
        out = filled.copy()
        for i in range(n):
            lo, hi = max(0, i - r), min(n, i + r + 1)
            xs = np.arange(lo, hi) - i
            coef = np.polyfit(xs, filled[lo:hi], min(polyorder, hi - lo - 1))
            out[i] = np.polyval(coef, 0)
        return out


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
def gap_fill_temporal(series: "NDArray", method: str = "whittaker", **kwargs) -> "NDArray":
    """Reconstruct a gappy index time series (e.g. cloud-masked NDVI).

    Parameters
    ----------
    series
        1-D series with ``NaN`` gaps, **or** an ``(T, ...)`` stack — when more than
        1-D the smoother is applied independently along the leading (time) axis.
    method
        ``"whittaker"`` (default) or ``"savgol"`` / ``"savitzky_golay"``.
    **kwargs
        Forwarded to the chosen smoother.

    Returns
    -------
    ndarray of float
        Gap-filled series with the same shape as ``series``.
    """
    method = method.lower()
    if method in {"whittaker", "whit"}:
        fn = whittaker_smooth
    elif method in {"savgol", "savitzky_golay", "sg"}:
        fn = savitzky_golay
    else:  # pragma: no cover - defensive
        raise ValueError(f"unknown method {method!r}; expected whittaker|savgol")

    series = np.asarray(series, dtype=float)
    if series.ndim == 1:
        return fn(series, **kwargs)
    return np.apply_along_axis(lambda v: fn(v, **kwargs), 0, series)


# ---------------------------------------------------------------------------
# SAR -> NDVI translation
# ---------------------------------------------------------------------------
def sar_to_ndvi(sar_features, model=None, *, feature_order=None) -> "NDArray":
    """Estimate optical NDVI from SAR features (all-weather NDVI proxy).

    Production uses a trained Random-Forest / gradient-boosting regressor (or a
    small temporal-CNN) mapping ``{VV, VH, VH/VV, RVI, ...}`` → NDVI, fitted against
    cloud-free coincident optical NDVI. Pass that fitted estimator as ``model`` (any
    object exposing ``.predict`` on an ``(n_samples, n_features)`` matrix).

    With ``model=None`` a transparent **linear demo fallback** is used::

        NDVI ≈ 0.55 + 0.45 * tanh( 1.2 * RVI - 0.6 )

    which reproduces the empirical monotonic increase of NDVI with the Radar
    Vegetation Index while staying in ``[0, 1]`` — enough to keep the cloudy-period
    pipeline running offline.

    Parameters
    ----------
    sar_features
        ``dict`` of named feature arrays (e.g. the output of
        :func:`agristress.preprocessing.sar.compute_sar_features`) **or** an
        ``(n_samples, n_features)`` matrix paired with ``feature_order`` (for a
        fitted ``model``).
    model
        Optional fitted regressor with a ``.predict`` method.
    feature_order
        Column order when ``sar_features`` is a raw matrix / for assembling the
        design matrix for ``model``.

    Returns
    -------
    ndarray of float
        Estimated NDVI, clipped to ``[0, 1]``, shaped like an input feature array.
    """
    if model is not None:
        if isinstance(sar_features, dict):
            order = feature_order or list(sar_features.keys())
            stack = np.stack([np.asarray(sar_features[k], dtype=float) for k in order], axis=-1)
            flat = stack.reshape(-1, stack.shape[-1])
            pred = np.asarray(model.predict(flat)).reshape(stack.shape[:-1])
        else:
            mat = np.asarray(sar_features, dtype=float)
            pred = np.asarray(model.predict(mat))
        return np.clip(pred, 0.0, 1.0)

    # ---- linear/empirical demo fallback ----
    if isinstance(sar_features, dict):
        if "rvi" in sar_features:
            rvi = np.asarray(sar_features["rvi"], dtype=float)
        elif "vv" in sar_features and "vh" in sar_features:
            from agristress.preprocessing.sar import compute_sar_features

            rvi = compute_sar_features(
                sar_features["vv"], sar_features["vh"], input_in_db=True
            )["rvi"]
        else:  # pragma: no cover - defensive
            raise KeyError("sar_features dict needs 'rvi' or both 'vv' and 'vh'")
    else:
        rvi = np.asarray(sar_features, dtype=float)

    ndvi = 0.55 + 0.45 * np.tanh(1.2 * rvi - 0.6)
    return np.clip(ndvi, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Multi-sensor robust consensus
# ---------------------------------------------------------------------------
def multi_sensor_consensus(
    estimates,
    weights=None,
    *,
    mad_threshold: float = 3.5,
    return_mask: bool = False,
):
    r"""Robustly fuse several per-pixel estimates with MAD outlier rejection.

    Combines ``k`` co-registered estimates (e.g. NDVI from S2, Landsat and a
    SAR→NDVI proxy) into one consensus layer. Outliers are detected with the
    modified Z-score on the **median absolute deviation** (Iglewicz & Hoaglin)::

        Mi = 0.6745 * |xi - median| / MAD     ;  reject where Mi > mad_threshold

    Surviving values are combined by a (weighted) mean; if all members at a pixel
    are rejected the plain median is used as a safe fallback.

    Parameters
    ----------
    estimates
        Sequence/array of ``k`` arrays (same shape), or an array stacked along
        ``axis=0`` with length ``k``. ``NaN`` entries are ignored.
    weights
        Optional per-source weights (length ``k``). Default: equal weights.
    mad_threshold
        Modified Z-score cut-off (default 3.5).
    return_mask
        If ``True`` also return the boolean *kept* mask (shape ``(k, ...)``).

    Returns
    -------
    ndarray  (and optionally the kept-mask)
        Consensus estimate, shape = element shape.
    """
    stack = np.asarray(estimates, dtype=float)
    if stack.ndim == 1:  # k scalars -> (k, 1)
        stack = stack[:, None]
        squeeze = True
    else:
        squeeze = False
    k = stack.shape[0]

    median = np.nanmedian(stack, axis=0)
    abs_dev = np.abs(stack - median[None, ...])
    mad = np.nanmedian(abs_dev, axis=0)

    safe_mad = np.where(mad > _EPS, mad, np.nan)
    mod_z = 0.6745 * abs_dev / safe_mad[None, ...]
    # Keep finite, non-outlier samples. Where MAD==0 every finite value is kept.
    kept = np.isfinite(stack) & (~(mod_z > mad_threshold) | ~np.isfinite(safe_mad)[None, ...])

    if weights is None:
        w = np.ones(k, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape[0] != k:
            raise ValueError(f"weights length {w.shape[0]} != number of sources {k}")
    w_b = w.reshape((k,) + (1,) * (stack.ndim - 1))

    eff_w = np.where(kept, w_b, 0.0)
    wsum = eff_w.sum(axis=0)
    weighted = np.where(kept, stack * w_b, 0.0).sum(axis=0)
    consensus = np.where(wsum > 0, weighted / np.where(wsum > 0, wsum, 1.0), median)

    if squeeze:
        consensus = consensus[0]
        kept = kept[:, 0]
    if return_mask:
        return consensus, kept
    return consensus
