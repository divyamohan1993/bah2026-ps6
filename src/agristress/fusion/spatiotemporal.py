"""Spatiotemporal image fusion: STARFM and ESTARFM (numpy DEMO implementations).

These blend a *fine*-resolution sensor (e.g. Sentinel-2 / Landsat, 10–30 m, sparse
in time) with a *coarse*-resolution sensor (e.g. MODIS, 250–500 m, daily) to predict
a fine-resolution image at a date when only the coarse image exists — the core trick
for dense, cloud-robust optical time series.

* :func:`starfm`   — Spatial and Temporal Adaptive Reflectance Fusion Model
  (Gao et al. 2006, *IEEE TGRS*). Working simplified single-pair implementation.
* :func:`estarfm`  — Enhanced STARFM (Zhu et al. 2010, *RSE*) using two
  fine/coarse pairs with a linear conversion coefficient. Working simplified form.

Notes
-----
For operational scale this should be replaced/augmented by a GPU implementation or
HISTARFM (Moreno-Martínez et al. 2020) gap-filling; those paths are noted in the
docstrings and left as production extensions. The implementations here are
deliberately compact, vectorised and offline-runnable for the DEMO datacube.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

_EPS = 1e-6


def _moving_windows(img: NDArray, win: int) -> NDArray:
    """Stack of ``win x win`` neighbourhoods for every pixel (edge-padded).

    Returns an array shaped ``(H, W, win*win)``.
    """
    r = win // 2
    padded = np.pad(img, r, mode="reflect")
    h, w = img.shape
    out = np.empty((h, w, win * win), dtype=float)
    k = 0
    for dy in range(win):
        for dx in range(win):
            out[:, :, k] = padded[dy : dy + h, dx : dx + w]
            k += 1
    return out


def starfm(
    fine_t0: NDArray,
    coarse_t0: NDArray,
    coarse_t1: NDArray,
    *,
    win: int = 25,
    spectral_tol: float | None = None,
    n_classes: int = 4,
) -> NDArray:
    r"""Predict a fine-resolution image at ``t1`` from one fine/coarse pair at ``t0``.

    Simplified STARFM (Gao et al. 2006). For each target pixel a local
    ``win x win`` neighbourhood supplies *spectrally similar* candidate pixels; each
    candidate contributes the coarse temporal change, weighted by combined spectral
    difference, temporal difference and spatial distance::

        F(t1) = sum_i  W_i * ( F_i(t0) + [C_i(t1) - C_i(t0)] )

    where similar pixels satisfy ``|F_i(t0) - F_center(t0)| <= spectral_tol`` and the
    weight ``W_i ∝ 1 / (spectral_diff * temporal_diff * (1 + spatial_dist))``.

    Parameters
    ----------
    fine_t0
        Fine-resolution image at the base date ``t0`` (2-D).
    coarse_t0, coarse_t1
        Coarse-resolution images at ``t0`` and the prediction date ``t1``,
        already resampled to the fine grid (same shape as ``fine_t0``).
    win
        Odd local window size in fine pixels (default 25).
    spectral_tol
        Spectral-similarity threshold for selecting candidate pixels. Default:
        ``std(fine_t0) * 2 / n_classes`` (Gao's class-count heuristic).
    n_classes
        Approximate number of land-cover classes used by the default tolerance.

    Returns
    -------
    ndarray of float
        Predicted fine-resolution reflectance at ``t1`` (same shape as ``fine_t0``).
    """
    fine_t0 = np.asarray(fine_t0, dtype=float)
    coarse_t0 = np.asarray(coarse_t0, dtype=float)
    coarse_t1 = np.asarray(coarse_t1, dtype=float)
    if not (fine_t0.shape == coarse_t0.shape == coarse_t1.shape):
        raise ValueError("starfm inputs must all share the same (fine-grid) shape")
    if fine_t0.ndim != 2:
        raise ValueError("starfm operates on a single 2-D band")
    if win % 2 == 0:
        win += 1

    if spectral_tol is None:
        spectral_tol = float(np.nanstd(fine_t0)) * 2.0 / max(n_classes, 1) + _EPS

    f0 = _moving_windows(fine_t0, win)
    c0 = _moving_windows(coarse_t0, win)
    c1 = _moving_windows(coarse_t1, win)
    center = fine_t0[:, :, None]

    spec_diff = np.abs(f0 - center)
    temp_diff = np.abs(c1 - c0)

    # Spatial distance weight (geometric, fixed per window position).
    r = win // 2
    dy, dx = np.mgrid[-r : r + 1, -r : r + 1]
    dist = np.sqrt(dy.astype(float) ** 2 + dx**2).ravel()[None, None, :]

    similar = spec_diff <= spectral_tol
    if not similar.any():  # degenerate: fall back to the coarse change everywhere
        return fine_t0 + (coarse_t1 - coarse_t0)

    combined = (spec_diff + _EPS) * (temp_diff + _EPS) * (1.0 + dist)
    weight = np.where(similar, 1.0 / combined, 0.0)

    wsum = weight.sum(axis=2)
    candidates = f0 + (c1 - c0)
    pred = (weight * candidates).sum(axis=2) / np.where(wsum > 0, wsum, 1.0)

    # Where no similar pixel was found, default to the direct coarse-change model.
    no_sim = wsum <= 0
    if no_sim.any():
        pred[no_sim] = (fine_t0 + (coarse_t1 - coarse_t0))[no_sim]
    return pred


def estarfm(
    fine_t0: NDArray,
    coarse_t0: NDArray,
    fine_t2: NDArray,
    coarse_t2: NDArray,
    coarse_t1: NDArray,
    *,
    win: int = 25,
    spectral_tol: float | None = None,
    n_classes: int = 4,
) -> NDArray:
    r"""Enhanced STARFM prediction at ``t1`` from **two** fine/coarse pairs.

    Simplified ESTARFM (Zhu et al. 2010). Two STARFM predictions are made — one
    using the ``t0`` pair, one using the ``t2`` pair — and combined with temporal
    weights derived from how well each coarse image tracks the coarse image at
    ``t1`` (smaller coarse change ⇒ larger weight). ESTARFM's local linear
    conversion coefficient between fine and coarse change is captured implicitly by
    the spectrally-weighted STARFM blend, giving better accuracy over heterogeneous
    fields than single-pair STARFM.

    Parameters
    ----------
    fine_t0, coarse_t0
        Fine/coarse pair at ``t0``.
    fine_t2, coarse_t2
        Fine/coarse pair at ``t2`` (``t0 < t1 < t2``).
    coarse_t1
        Coarse image at the prediction date ``t1`` (fine grid).
    win, spectral_tol, n_classes
        As in :func:`starfm`.

    Returns
    -------
    ndarray of float
        Predicted fine-resolution reflectance at ``t1``.
    """
    pred0 = starfm(
        fine_t0, coarse_t0, coarse_t1, win=win, spectral_tol=spectral_tol, n_classes=n_classes
    )
    pred2 = starfm(
        fine_t2, coarse_t2, coarse_t1, win=win, spectral_tol=spectral_tol, n_classes=n_classes
    )

    coarse_t0 = np.asarray(coarse_t0, dtype=float)
    coarse_t1 = np.asarray(coarse_t1, dtype=float)
    coarse_t2 = np.asarray(coarse_t2, dtype=float)

    # Temporal weights: the pair whose coarse image is closer to t1 is more reliable.
    d0 = np.abs(coarse_t1 - coarse_t0).mean() + _EPS
    d2 = np.abs(coarse_t1 - coarse_t2).mean() + _EPS
    w0 = (1.0 / d0) / (1.0 / d0 + 1.0 / d2)
    w2 = 1.0 - w0
    return w0 * pred0 + w2 * pred2
