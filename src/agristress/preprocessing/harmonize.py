"""Cross-sensor harmonisation: common-grid resampling and linear cross-calibration.

* :func:`to_common_grid` — resample a raster onto a target grid (interface; uses
  ``rioxarray`` reprojection when available, else a pure-numpy nearest/bilinear
  resampler so DEMO mode works without GDAL).
* :func:`resample_nearest` — dependency-free nearest-neighbour block resampler.
* :func:`cross_calibrate`  — apply a per-band linear cross-calibration
  ``out = gain * src + bias`` to inter-calibrate one sensor onto a reference.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------
def resample_nearest(arr: "NDArray", out_shape: tuple[int, int]) -> "NDArray":
    """Nearest-neighbour resample a 2-D (or ``(band, H, W)``) array to ``out_shape``.

    Pure numpy (index remap) — no SciPy / GDAL. Handles both up- and down-sampling.

    Parameters
    ----------
    arr
        ``(H, W)`` or ``(bands, H, W)`` array.
    out_shape
        Target ``(H_out, W_out)``.

    Returns
    -------
    ndarray
        Resampled array with the same number of bands / dtype as ``arr``.
    """
    arr = np.asarray(arr)
    out_h, out_w = out_shape
    if out_h <= 0 or out_w <= 0:
        raise ValueError(f"out_shape must be positive, got {out_shape}")

    *lead, in_h, in_w = arr.shape
    row_idx = (np.linspace(0, in_h - 1, out_h)).round().astype(int)
    col_idx = (np.linspace(0, in_w - 1, out_w)).round().astype(int)
    row_idx = np.clip(row_idx, 0, in_h - 1)
    col_idx = np.clip(col_idx, 0, in_w - 1)

    if lead:
        return arr[..., row_idx, :][..., :, col_idx]
    return arr[np.ix_(row_idx, col_idx)]


def _resample_bilinear(arr: "NDArray", out_shape: tuple[int, int]) -> "NDArray":
    """Bilinear resample a 2-D array to ``out_shape`` (numpy only)."""
    arr = np.asarray(arr, dtype=float)
    in_h, in_w = arr.shape
    out_h, out_w = out_shape
    if in_h == 1 or in_w == 1:
        return resample_nearest(arr, out_shape).astype(float)

    ys = np.linspace(0, in_h - 1, out_h)
    xs = np.linspace(0, in_w - 1, out_w)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = np.clip(y0 + 1, 0, in_h - 1)
    x1 = np.clip(x0 + 1, 0, in_w - 1)
    wy = (ys - y0)[:, None]
    wx = (xs - x0)[None, :]

    top = arr[y0][:, x0] * (1 - wx) + arr[y0][:, x1] * wx
    bot = arr[y1][:, x0] * (1 - wx) + arr[y1][:, x1] * wx
    return top * (1 - wy) + bot * wy


def to_common_grid(
    data: "NDArray",
    out_shape: tuple[int, int] | None = None,
    *,
    method: str = "nearest",
    target=None,
):
    """Resample a raster / DataArray onto a common analysis grid (interface).

    If ``data`` is an ``xarray.DataArray`` with a CRS and a ``target`` grid is
    provided, this delegates to ``rioxarray.reproject_match`` (the production path).
    Otherwise it runs a pure-numpy resampler to ``out_shape`` so the call works in
    DEMO mode without GDAL/PROJ.

    Parameters
    ----------
    data
        ``numpy.ndarray`` (``(H, W)`` / ``(band, H, W)``) or ``xarray.DataArray``.
    out_shape
        Target ``(H, W)`` for the numpy path. Ignored when reprojecting to a
        ``target`` DataArray.
    method
        ``"nearest"`` or ``"bilinear"`` (numpy path).
    target
        Optional reference ``xarray.DataArray`` defining the destination grid/CRS.

    Returns
    -------
    Same type as ``data`` (ndarray or DataArray), resampled.
    """
    # Production path: georeferenced reprojection via rioxarray, if both available.
    if target is not None and hasattr(data, "rio"):
        try:  # pragma: no cover - requires rioxarray + GDAL
            return data.rio.reproject_match(target)
        except Exception:  # pragma: no cover - fall through to numpy
            pass

    values = np.asarray(getattr(data, "values", data))
    if out_shape is None:
        raise ValueError(
            "to_common_grid needs `out_shape` for the numpy resampling path "
            "(or an xarray `target` with a CRS for georeferenced reprojection)."
        )

    if method == "nearest" or values.ndim != 2:
        out = resample_nearest(values, out_shape)
    elif method == "bilinear":
        out = _resample_bilinear(values, out_shape)
    else:  # pragma: no cover - defensive
        raise ValueError(f"unknown method {method!r}; expected nearest|bilinear")
    return out


# ---------------------------------------------------------------------------
# Linear cross-calibration
# ---------------------------------------------------------------------------
def cross_calibrate(
    src: "NDArray",
    ref=None,
    coeffs: tuple[float, float] | dict[str, tuple[float, float]] | None = None,
    *,
    band: str | None = None,
    fit: bool = False,
) -> "NDArray":
    """Inter-calibrate ``src`` onto a reference sensor via a linear transform.

    Two modes:

    * **Apply** (default) — ``out = gain * src + bias`` using ``coeffs``. ``coeffs``
      may be a ``(gain, bias)`` pair, or a ``{band: (gain, bias)}`` mapping selected
      with ``band``.
    * **Fit** (``fit=True``) — least-squares-fit ``(gain, bias)`` from co-located
      ``src`` / ``ref`` samples (e.g. a cross-over region), then apply it. NaNs in
      either array are dropped before fitting.

    Parameters
    ----------
    src
        Source-sensor values to be calibrated.
    ref
        Reference-sensor values (required when ``fit=True``).
    coeffs
        ``(gain, bias)`` or ``{band: (gain, bias)}``. Required when ``fit=False``.
    band
        Key into ``coeffs`` when it is a mapping.
    fit
        Estimate the coefficients from ``src``/``ref`` instead of using ``coeffs``.

    Returns
    -------
    ndarray of float
        Calibrated ``src``.
    """
    src = np.asarray(src, dtype=float)

    if fit:
        if ref is None:
            raise ValueError("cross_calibrate(fit=True) requires `ref` samples.")
        ref_arr = np.asarray(ref, dtype=float)
        m = np.isfinite(src) & np.isfinite(ref_arr)
        if m.sum() < 2:
            raise ValueError("need >= 2 finite co-located samples to fit a line.")
        gain, bias = np.polyfit(src[m].ravel(), ref_arr[m].ravel(), 1)
        return gain * src + bias

    if coeffs is None:
        raise ValueError("cross_calibrate needs `coeffs` when fit=False.")
    if isinstance(coeffs, dict):
        if band is None:
            raise ValueError("`band` is required when `coeffs` is a mapping.")
        gain, bias = coeffs[band]
    else:
        gain, bias = coeffs
    return float(gain) * src + float(bias)
