"""Phenology & growth-stage feature extraction.

Tools to turn a noisy vegetation-index (VI) time series into compact,
physically meaningful descriptors used by the crop classifier and the
stage-aware stress model:

* :func:`whittaker_smooth` -- robust temporal smoother (Eilers 2003).
* :func:`double_logistic_fit` -- fit the Beck et al. (2006) double-logistic
  growth curve and derive SOS / POS / EOS / LGP / amplitude / integral.
* :func:`harmonic_features` -- least-squares Fourier / HANTS-style harmonics.
* :func:`phenometrics` -- convenience wrapper returning a flat metric dict.
* :func:`gdd_accumulate` / :func:`assign_growth_stage` -- growing-degree-day
  accumulation and FAO-56-style per-crop stage assignment.

All functions accept plain numpy arrays.  ``scipy`` is required for the
non-linear curve fit (``scipy.optimize.curve_fit``) and the banded solve used
by the Whittaker smoother; both are core AgriStress dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

__all__ = [
    "whittaker_smooth",
    "double_logistic",
    "double_logistic_fit",
    "DoubleLogisticResult",
    "harmonic_features",
    "phenometrics",
    "gdd_accumulate",
    "assign_growth_stage",
    "CROP_GDD_TABLE",
    "growth_stage_fractions",
]


# --------------------------------------------------------------------------- #
# Whittaker smoother
# --------------------------------------------------------------------------- #
def whittaker_smooth(
    y: np.ndarray,
    lmbda: float = 10.0,
    d: int = 2,
    w: np.ndarray | None = None,
) -> np.ndarray:
    """Whittaker-Eilers smoother (penalised least squares).

    Minimises ``||W^{1/2}(y - z)||^2 + lambda * ||D^d z||^2`` where ``D^d`` is
    the ``d``-th order difference operator.  Larger ``lambda`` -> smoother.

    Parameters
    ----------
    y
        1-D series.  NaNs are handled by zero-weighting (gap filling).
    lmbda
        Smoothing strength (>= 0).
    d
        Order of the difference penalty (2 is the usual choice).
    w
        Optional per-sample weights in ``[0, 1]``.  ``0`` ignores a sample.

    Returns
    -------
    np.ndarray
        Smoothed series, same length as ``y``.
    """
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    if n == 0:
        return y.copy()
    if w is None:
        w = np.ones(n, dtype=np.float64)
    else:
        w = np.asarray(w, dtype=np.float64).copy()
    # Treat NaNs as missing (weight 0, value 0 so they don't poison the solve).
    nan_mask = ~np.isfinite(y)
    y = np.where(nan_mask, 0.0, y)
    w = np.where(nan_mask, 0.0, w)

    if n <= d:
        # Not enough points to apply the difference penalty meaningfully.
        return y.copy()

    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    eye = sparse.eye(n, format="csc")
    diff = eye.copy()
    for _ in range(d):
        diff = diff[1:] - diff[:-1]
    diff = diff.tocsc()
    weights = sparse.diags(w, 0, format="csc")
    a = weights + lmbda * (diff.T @ diff)
    z = spsolve(a.tocsc(), weights @ y)
    return np.asarray(z, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Double-logistic curve
# --------------------------------------------------------------------------- #
def double_logistic(
    t: np.ndarray,
    base: float,
    amp: float,
    sos: float,
    r_sp: float,
    eos: float,
    r_au: float,
) -> np.ndarray:
    """Beck et al. (2006) double-logistic seasonal VI model.

    ``f(t) = base + amp * [ 1/(1+exp(-r_sp*(t-sos))) - 1/(1+exp(-r_au*(t-eos))) ]``

    Parameters
    ----------
    base
        Dormant-season baseline VI.
    amp
        Seasonal amplitude (peak above baseline).
    sos, eos
        Inflection (mid-point) times of the spring rise and autumn fall.
    r_sp, r_au
        Rise / fall rates (slope steepness) at the inflection points.
    """
    t = np.asarray(t, dtype=np.float64)
    spring = 1.0 / (1.0 + np.exp(-r_sp * (t - sos)))
    autumn = 1.0 / (1.0 + np.exp(-r_au * (t - eos)))
    return base + amp * (spring - autumn)


@dataclass
class DoubleLogisticResult:
    """Container for a fitted double-logistic curve and its phenometrics."""

    params: dict[str, float]
    sos: float
    pos: float
    eos: float
    lgp: float
    amplitude: float
    peak_value: float
    base_value: float
    integral: float
    success: bool
    rmse: float
    t_grid: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))
    fitted: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))

    def as_dict(self) -> dict[str, float]:
        """Flat metric dict (for feature tables / reports)."""
        return {
            "sos": self.sos,
            "pos": self.pos,
            "eos": self.eos,
            "lgp": self.lgp,
            "amplitude": self.amplitude,
            "peak_value": self.peak_value,
            "base_value": self.base_value,
            "integral": self.integral,
            "rmse": self.rmse,
            **{f"dl_{k}": v for k, v in self.params.items()},
        }


def double_logistic_fit(
    t: np.ndarray,
    vi: np.ndarray,
    *,
    amp_threshold: float = 0.5,
    maxfev: int = 20000,
    smooth_lambda: float | None = None,
) -> DoubleLogisticResult:
    """Fit a double-logistic curve to a VI time series and derive phenometrics.

    Parameters
    ----------
    t
        Time stamps (e.g. day-of-year). Strictly increasing not required but
        recommended.
    vi
        Vegetation-index values aligned with ``t``.  NaNs are dropped.
    amp_threshold
        Fraction of the seasonal amplitude (0-1) defining SOS/EOS on the fitted
        curve.  ``0.5`` -> half-maximum crossing (default, robust).
    maxfev
        Maximum function evaluations for ``curve_fit``.
    smooth_lambda
        If given, pre-smooth ``vi`` with :func:`whittaker_smooth` before fitting.

    Returns
    -------
    DoubleLogisticResult
        Fitted parameters + derived metrics:

        * ``sos`` / ``eos`` : start / end of season (amplitude-threshold
          crossings on the rise / fall limbs).
        * ``pos`` : peak-of-season time (argmax of the fitted curve).
        * ``lgp`` : length of growing period (``eos - sos``).
        * ``amplitude`` : peak minus baseline of the fitted curve.
        * ``integral`` : VI integrated over ``[sos, eos]`` above the baseline
          (a greenness / productivity proxy).

    Notes
    -----
    Falls back gracefully: if the non-linear fit fails to converge, metrics are
    derived from the (optionally smoothed) observations and ``success=False``.
    """
    t = np.asarray(t, dtype=np.float64)
    vi = np.asarray(vi, dtype=np.float64)
    mask = np.isfinite(t) & np.isfinite(vi)
    t_f, vi_f = t[mask], vi[mask]
    if t_f.size < 6:
        return _fallback_metrics(t_f, vi_f, amp_threshold, reason="insufficient_points")

    if smooth_lambda is not None:
        vi_f = whittaker_smooth(vi_f, lmbda=smooth_lambda)

    # --- initial parameter guesses from the data ---
    vmin, vmax = float(np.min(vi_f)), float(np.max(vi_f))
    base0 = vmin
    amp0 = max(vmax - vmin, 1e-3)
    tmin, tmax = float(np.min(t_f)), float(np.max(t_f))
    span = max(tmax - tmin, 1.0)
    sos0 = tmin + 0.25 * span
    eos0 = tmin + 0.75 * span
    rate0 = 6.0 / span  # gentle slope spanning the season

    p0 = [base0, amp0, sos0, rate0, eos0, rate0]
    lower = [vmin - amp0, 1e-4, tmin - span, 1e-4, tmin - span, 1e-4]
    upper = [vmax + amp0, 3.0 * amp0 + 1e-3, tmax + span, 50.0, tmax + span, 50.0]

    success = True
    rmse = float("nan")
    try:
        from scipy.optimize import curve_fit

        popt, _ = curve_fit(
            double_logistic,
            t_f,
            vi_f,
            p0=p0,
            bounds=(lower, upper),
            maxfev=maxfev,
        )
        base, amp, sos_p, r_sp, eos_p, r_au = popt
        resid = vi_f - double_logistic(t_f, *popt)
        rmse = float(np.sqrt(np.mean(resid**2)))
    except Exception:
        return _fallback_metrics(t_f, vi_f, amp_threshold, reason="fit_failed")

    # Dense grid for metric extraction.
    grid = np.linspace(tmin, tmax, 512)
    curve = double_logistic(grid, base, amp, sos_p, r_sp, eos_p, r_au)
    peak_idx = int(np.argmax(curve))
    pos = float(grid[peak_idx])
    peak_value = float(curve[peak_idx])
    base_value = float(np.min(curve))
    amplitude = float(peak_value - base_value)

    # Amplitude-threshold crossings (rise limb before peak, fall limb after).
    thr = base_value + amp_threshold * amplitude
    sos = _first_crossing(grid[: peak_idx + 1], curve[: peak_idx + 1], thr, rising=True)
    eos = _first_crossing(grid[peak_idx:], curve[peak_idx:], thr, rising=False)
    if not np.isfinite(sos):
        sos = float(sos_p)
    if not np.isfinite(eos):
        eos = float(eos_p)
    lgp = float(max(eos - sos, 0.0))

    # Integral of (curve - base) over the growing period.
    integ_mask = (grid >= sos) & (grid <= eos)
    if integ_mask.sum() >= 2:
        integral = float(np.trapezoid(np.clip(curve[integ_mask] - base_value, 0, None), grid[integ_mask]))
    else:
        integral = 0.0

    params = {
        "base": float(base),
        "amp": float(amp),
        "sos": float(sos_p),
        "r_sp": float(r_sp),
        "eos": float(eos_p),
        "r_au": float(r_au),
    }
    return DoubleLogisticResult(
        params=params,
        sos=float(sos),
        pos=pos,
        eos=float(eos),
        lgp=lgp,
        amplitude=amplitude,
        peak_value=peak_value,
        base_value=base_value,
        integral=integral,
        success=success,
        rmse=rmse,
        t_grid=grid,
        fitted=curve,
    )


def _first_crossing(t: np.ndarray, y: np.ndarray, level: float, *, rising: bool) -> float:
    """Linear-interpolated time at which ``y`` first crosses ``level``.

    ``rising`` selects an upward (SOS) or downward (EOS) crossing.  Returns NaN
    if no crossing is found.
    """
    if t.size < 2:
        return float("nan")
    above = y >= level
    for i in range(1, t.size):
        if rising and (not above[i - 1]) and above[i]:
            pass
        elif (not rising) and above[i - 1] and (not above[i]):
            pass
        else:
            continue
        y0, y1 = y[i - 1], y[i]
        if y1 == y0:
            return float(t[i])
        frac = (level - y0) / (y1 - y0)
        return float(t[i - 1] + frac * (t[i] - t[i - 1]))
    return float("nan")


def _fallback_metrics(t: np.ndarray, vi: np.ndarray, amp_threshold: float, *, reason: str) -> DoubleLogisticResult:
    """Derive coarse phenometrics directly from observations when the fit fails."""
    if t.size == 0:
        return DoubleLogisticResult(
            params={}, sos=float("nan"), pos=float("nan"), eos=float("nan"),
            lgp=float("nan"), amplitude=float("nan"), peak_value=float("nan"),
            base_value=float("nan"), integral=float("nan"), success=False, rmse=float("nan"),
        )
    order = np.argsort(t)
    t, vi = t[order], vi[order]
    peak_idx = int(np.argmax(vi))
    pos = float(t[peak_idx])
    peak_value = float(vi[peak_idx])
    base_value = float(np.min(vi))
    amplitude = float(peak_value - base_value)
    thr = base_value + amp_threshold * amplitude
    sos = _first_crossing(t[: peak_idx + 1], vi[: peak_idx + 1], thr, rising=True)
    eos = _first_crossing(t[peak_idx:], vi[peak_idx:], thr, rising=False)
    sos = float(sos) if np.isfinite(sos) else float(t[0])
    eos = float(eos) if np.isfinite(eos) else float(t[-1])
    lgp = float(max(eos - sos, 0.0))
    integral = float(np.trapezoid(np.clip(vi - base_value, 0, None), t)) if t.size >= 2 else 0.0
    return DoubleLogisticResult(
        params={}, sos=sos, pos=pos, eos=eos, lgp=lgp, amplitude=amplitude,
        peak_value=peak_value, base_value=base_value, integral=integral,
        success=False, rmse=float("nan"), t_grid=t, fitted=vi,
    )


# --------------------------------------------------------------------------- #
# Harmonic (Fourier / HANTS) features
# --------------------------------------------------------------------------- #
def harmonic_features(
    t: np.ndarray,
    vi: np.ndarray,
    n: int = 3,
    period: float | None = None,
) -> dict[str, float]:
    """Least-squares harmonic (Fourier / HANTS) decomposition of a VI series.

    Fits ``vi(t) ~ a0 + sum_{k=1..n} [ a_k cos(2*pi*k*t/T) + b_k sin(...) ]`` by
    ordinary least squares and returns the coefficients plus, per harmonic, the
    amplitude ``sqrt(a^2+b^2)`` and phase ``atan2(-b, a)``.

    Parameters
    ----------
    t
        Time stamps.
    vi
        VI values aligned with ``t``.  NaNs dropped.
    n
        Number of harmonics.
    period
        Fundamental period ``T``.  Defaults to the observed time span.

    Returns
    -------
    dict
        ``{'harm_mean', 'harm_a1','harm_b1','harm_amp1','harm_phase1', ...,
        'harm_rmse'}``.  Robust to short series (returns NaNs for unfittable
        harmonics rather than raising).
    """
    t = np.asarray(t, dtype=np.float64)
    vi = np.asarray(vi, dtype=np.float64)
    mask = np.isfinite(t) & np.isfinite(vi)
    t_f, vi_f = t[mask], vi[mask]

    out: dict[str, float] = {"harm_mean": float("nan"), "harm_rmse": float("nan")}
    for k in range(1, n + 1):
        out[f"harm_a{k}"] = float("nan")
        out[f"harm_b{k}"] = float("nan")
        out[f"harm_amp{k}"] = float("nan")
        out[f"harm_phase{k}"] = float("nan")

    if t_f.size == 0:
        return out
    if period is None:
        span = float(np.max(t_f) - np.min(t_f))
        period = span if span > 0 else 1.0

    # Design matrix: [1, cos1, sin1, cos2, sin2, ...].
    cols = [np.ones_like(t_f)]
    names = ["mean"]
    omega = 2.0 * np.pi / period
    for k in range(1, n + 1):
        cols.append(np.cos(k * omega * t_f))
        cols.append(np.sin(k * omega * t_f))
        names.extend([f"a{k}", f"b{k}"])
    design = np.column_stack(cols)

    # Only fit as many params as we have independent observations for.
    n_params = min(design.shape[1], t_f.size)
    design = design[:, :n_params]
    coef, *_ = np.linalg.lstsq(design, vi_f, rcond=None)

    fitted = design @ coef
    rmse = float(np.sqrt(np.mean((vi_f - fitted) ** 2)))
    out["harm_rmse"] = rmse
    out["harm_mean"] = float(coef[0]) if n_params >= 1 else float("nan")

    ci = 1
    for k in range(1, n + 1):
        if ci + 1 < n_params:
            a = float(coef[ci])
            b = float(coef[ci + 1])
            out[f"harm_a{k}"] = a
            out[f"harm_b{k}"] = b
            out[f"harm_amp{k}"] = float(np.hypot(a, b))
            out[f"harm_phase{k}"] = float(np.arctan2(-b, a))
        ci += 2
    return out


# --------------------------------------------------------------------------- #
# Convenience: combined phenometrics
# --------------------------------------------------------------------------- #
def phenometrics(
    t: np.ndarray,
    vi: np.ndarray,
    *,
    n_harmonics: int = 3,
    smooth_lambda: float | None = 5.0,
    amp_threshold: float = 0.5,
) -> dict[str, float]:
    """One-call phenology feature extractor.

    Combines double-logistic phenometrics (SOS/POS/EOS/LGP/amplitude/integral)
    with harmonic descriptors and simple statistics into a single flat dict
    suitable for a classifier feature row.
    """
    dl = double_logistic_fit(t, vi, amp_threshold=amp_threshold, smooth_lambda=smooth_lambda)
    feats = dl.as_dict()
    feats.update(harmonic_features(t, vi, n=n_harmonics))

    vi_arr = np.asarray(vi, dtype=np.float64)
    finite = vi_arr[np.isfinite(vi_arr)]
    if finite.size:
        feats["vi_mean"] = float(np.mean(finite))
        feats["vi_std"] = float(np.std(finite))
        feats["vi_min"] = float(np.min(finite))
        feats["vi_max"] = float(np.max(finite))
        feats["vi_range"] = float(np.max(finite) - np.min(finite))
    else:
        for k in ("vi_mean", "vi_std", "vi_min", "vi_max", "vi_range"):
            feats[k] = float("nan")
    return feats


# --------------------------------------------------------------------------- #
# Growing Degree Days & growth-stage assignment
# --------------------------------------------------------------------------- #
def gdd_accumulate(
    tmin: np.ndarray,
    tmax: np.ndarray,
    tbase: float = 10.0,
    tcap: float | None = 30.0,
) -> np.ndarray:
    """Cumulative Growing Degree Days from daily min/max temperature.

    Daily ``GDD = max(((Tmax_c + Tmin_c)/2) - Tbase, 0)`` where ``Tmax_c`` /
    ``Tmin_c`` are optionally capped at ``tcap`` (the standard upper-threshold
    method).  Returns the running cumulative sum.

    Parameters
    ----------
    tmin, tmax
        Daily minimum / maximum temperature (deg C), aligned 1-D arrays.
    tbase
        Base temperature below which no development occurs.
    tcap
        Upper cap applied to Tmax (and Tmin) before averaging.  ``None`` disables.

    Returns
    -------
    np.ndarray
        Cumulative GDD, same length as inputs.
    """
    tmin = np.asarray(tmin, dtype=np.float64)
    tmax = np.asarray(tmax, dtype=np.float64)
    if tmin.shape != tmax.shape:
        raise ValueError("tmin and tmax must have the same shape")
    tmx = tmax.copy()
    tmn = tmin.copy()
    if tcap is not None:
        tmx = np.minimum(tmx, tcap)
        tmn = np.minimum(tmn, tcap)
    # Also floor at tbase per the standard method (avoids negative contributions
    # from very cold days dragging the mean below base).
    tmx = np.maximum(tmx, tbase)
    tmn = np.maximum(tmn, tbase)
    daily = (tmx + tmn) / 2.0 - tbase
    daily = np.clip(daily, 0.0, None)
    daily = np.where(np.isfinite(daily), daily, 0.0)
    return np.cumsum(daily)


# Per-crop cumulative-GDD boundaries for the four FAO-56 development stages.
# Values are cumulative GDD (base temp in parentheses) at the END of each stage:
# (initial, development, mid-season, late-season-end).  Approximate, literature
# -informed defaults suitable for canal-command kharif/rabi crops.
CROP_GDD_TABLE: dict[str, dict[str, object]] = {
    "wheat": {"tbase": 4.5, "stages": (200.0, 700.0, 1400.0, 1900.0)},
    "rice": {"tbase": 10.0, "stages": (250.0, 750.0, 1500.0, 2000.0)},
    "maize": {"tbase": 10.0, "stages": (250.0, 800.0, 1500.0, 2100.0)},
    "cotton": {"tbase": 15.0, "stages": (300.0, 1000.0, 1900.0, 2600.0)},
    "sugarcane": {"tbase": 12.0, "stages": (400.0, 1500.0, 3500.0, 4500.0)},
    "soybean": {"tbase": 10.0, "stages": (200.0, 650.0, 1300.0, 1700.0)},
    "mustard": {"tbase": 5.0, "stages": (180.0, 550.0, 1100.0, 1500.0)},
    "default": {"tbase": 10.0, "stages": (250.0, 800.0, 1500.0, 2000.0)},
}

# Canonical stage labels (FAO-56 development stages) in order.
STAGE_LABELS = ("initial", "development", "mid-season", "late-season", "mature")


def assign_growth_stage(gdd: float | np.ndarray, crop: str = "default") -> "np.ndarray | str":
    """Map cumulative GDD to an FAO-56 growth stage for a given crop.

    Uses the per-crop boundaries in :data:`CROP_GDD_TABLE`.  Stages, in order:
    ``initial -> development -> mid-season -> late-season -> mature`` (the last
    label applies once cumulative GDD exceeds the late-season boundary, i.e.
    post-harvest / senesced).

    Parameters
    ----------
    gdd
        Cumulative GDD (scalar or array).
    crop
        Crop key; unknown crops fall back to ``"default"``.

    Returns
    -------
    str or np.ndarray
        Stage label(s).  Scalar input -> ``str``; array input -> array of str.
    """
    entry = CROP_GDD_TABLE.get(crop.lower(), CROP_GDD_TABLE["default"])
    bounds = np.asarray(entry["stages"], dtype=np.float64)  # type: ignore[index]
    labels = np.asarray(STAGE_LABELS)

    scalar = np.isscalar(gdd) or (isinstance(gdd, np.ndarray) and gdd.ndim == 0)
    g = np.atleast_1d(np.asarray(gdd, dtype=np.float64))
    # np.searchsorted with 'right': index = number of bounds strictly < g.
    idx = np.searchsorted(bounds, g, side="right")
    idx = np.clip(idx, 0, len(labels) - 1)
    out = labels[idx]
    return str(out[0]) if scalar else out


def growth_stage_fractions(crop: str = "default") -> dict[str, tuple[float, float]]:
    """Return the cumulative-GDD interval for each stage of ``crop``.

    Useful for plotting / reporting stage windows.  Intervals are
    ``[lower, upper)`` in cumulative GDD; the final stage is open-ended
    (upper = ``inf``).
    """
    entry = CROP_GDD_TABLE.get(crop.lower(), CROP_GDD_TABLE["default"])
    bounds = list(entry["stages"])  # type: ignore[index]
    edges = [0.0, *bounds, float("inf")]
    return {STAGE_LABELS[i]: (edges[i], edges[i + 1]) for i in range(len(STAGE_LABELS))}
