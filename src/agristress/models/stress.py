"""Stage-aware moisture-stress detection.

Building blocks:

* :func:`optram_soil_moisture` -- OPTRAM (Sadeghi et al. 2017) soil-moisture
  proxy ``W = (STR - STR_dry) / (STR_wet - STR_dry)`` with NDVI-dependent wet /
  dry edges.
* Condition indices: :func:`vci`, :func:`tci`, :func:`vhi`, :func:`tvdi`,
  :func:`cwsi`.
* :func:`anomaly_vs_baseline` -- z-score of a value / series against a baseline
  distribution (climatological anomaly).
* :class:`StageAwareStress` -- combines an index with the crop growth stage to
  pick **stage-specific thresholds**, returning a 0-4 severity class plus
  interpretive flags.

Everything is numpy-vectorised and NaN-safe.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

__all__ = [
    "DEFAULT_STAGE_THRESHOLDS",
    "SEVERITY_LABELS",
    "StageAwareStress",
    "anomaly_vs_baseline",
    "cwsi",
    "optram_soil_moisture",
    "tci",
    "tvdi",
    "vci",
    "vhi",
]

_EPS = 1e-12

# Severity classes (0 = none, 4 = severe).
SEVERITY_LABELS = {
    0: "none",
    1: "mild",
    2: "moderate",
    3: "high",
    4: "severe",
}


def _safe_norm(
    num: np.ndarray, den: np.ndarray, *, clip: tuple[float, float] | None = (0.0, 1.0)
) -> np.ndarray:
    out = np.full(np.broadcast_shapes(num.shape, den.shape), np.nan, dtype=np.float64)
    valid = (np.abs(den) >= _EPS) & np.isfinite(num) & np.isfinite(den)
    np.divide(num, den, out=out, where=valid)
    out[~valid] = np.nan
    if clip is not None:
        out = np.clip(out, clip[0], clip[1])
    return out


# --------------------------------------------------------------------------- #
# OPTRAM soil moisture
# --------------------------------------------------------------------------- #
def optram_soil_moisture(
    ndvi: np.ndarray,
    str_: np.ndarray,
    edges: Mapping[str, float],
) -> np.ndarray:
    """OPTRAM surface soil-moisture content ``W`` in ``[0, 1]``.

    ``W = (STR - STR_dry) / (STR_wet - STR_dry)`` where the wet and dry edges of
    the NDVI-STR feature space are linear in NDVI:

    * ``STR_dry = i_dry + s_dry * NDVI``
    * ``STR_wet = i_wet + s_wet * NDVI``

    Parameters
    ----------
    ndvi
        NDVI array.
    str_
        STR array (see :func:`agristress.features.indices.str_index`).
    edges
        Dict with keys ``i_dry, s_dry, i_wet, s_wet`` (the OPTRAM trapezoid
        parameters, typically fitted from a scatter of NDVI vs STR).

    Returns
    -------
    np.ndarray
        Soil-moisture proxy clipped to ``[0, 1]`` (0 = dry edge, 1 = wet edge).
    """
    ndvi = np.asarray(ndvi, dtype=np.float64)
    str_ = np.asarray(str_, dtype=np.float64)
    try:
        i_dry = float(edges["i_dry"])
        s_dry = float(edges["s_dry"])
        i_wet = float(edges["i_wet"])
        s_wet = float(edges["s_wet"])
    except KeyError as exc:  # pragma: no cover - guard for caller mistakes
        raise KeyError("edges must contain i_dry, s_dry, i_wet, s_wet") from exc

    str_dry = i_dry + s_dry * ndvi
    str_wet = i_wet + s_wet * ndvi
    return _safe_norm(str_ - str_dry, str_wet - str_dry, clip=(0.0, 1.0))


# --------------------------------------------------------------------------- #
# Condition indices
# --------------------------------------------------------------------------- #
def vci(ndvi: np.ndarray, ndvi_min: np.ndarray, ndvi_max: np.ndarray) -> np.ndarray:
    """Vegetation Condition Index (Kogan 1995), ``0-1``.

    ``VCI = (NDVI - NDVI_min) / (NDVI_max - NDVI_min)`` using the historical
    per-pixel NDVI extremes.  Low VCI -> vegetation stress.
    """
    ndvi = np.asarray(ndvi, dtype=np.float64)
    lo = np.asarray(ndvi_min, dtype=np.float64)
    hi = np.asarray(ndvi_max, dtype=np.float64)
    return _safe_norm(ndvi - lo, hi - lo, clip=(0.0, 1.0))


def tci(lst: np.ndarray, lst_min: np.ndarray, lst_max: np.ndarray) -> np.ndarray:
    """Temperature Condition Index (Kogan 1995), ``0-1``.

    ``TCI = (LST_max - LST) / (LST_max - LST_min)`` — inverted so that *low* TCI
    (high temperature) indicates thermal stress, consistent with VCI.
    """
    lst = np.asarray(lst, dtype=np.float64)
    lo = np.asarray(lst_min, dtype=np.float64)
    hi = np.asarray(lst_max, dtype=np.float64)
    return _safe_norm(hi - lst, hi - lo, clip=(0.0, 1.0))


def vhi(vci_val: np.ndarray, tci_val: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Vegetation Health Index ``alpha*VCI + (1-alpha)*TCI`` (Kogan 1997)."""
    vci_val = np.asarray(vci_val, dtype=np.float64)
    tci_val = np.asarray(tci_val, dtype=np.float64)
    return alpha * vci_val + (1.0 - alpha) * tci_val


def tvdi(
    ndvi: np.ndarray,
    lst: np.ndarray,
    *,
    dry_edge: tuple[float, float],
    wet_edge: float | tuple[float, float],
) -> np.ndarray:
    """Temperature-Vegetation Dryness Index (Sandholt et al. 2002), ``0-1``.

    ``TVDI = (LST - LST_wet) / (LST_dry - LST_wet)`` where the **dry edge** is a
    line in NDVI ``LST_dry = a + b*NDVI`` and the **wet edge** is either a
    constant or a line.

    Parameters
    ----------
    ndvi, lst
        NDVI and land-surface-temperature arrays.
    dry_edge
        ``(a, b)`` intercept/slope of the dry edge ``LST_dry = a + b*NDVI``.
    wet_edge
        Constant ``LST_wet`` or ``(a_w, b_w)`` for a sloped wet edge.

    Returns
    -------
    np.ndarray
        Dryness in ``[0, 1]`` (1 = dry edge -> maximum stress).
    """
    ndvi = np.asarray(ndvi, dtype=np.float64)
    lst = np.asarray(lst, dtype=np.float64)
    a, b = dry_edge
    lst_dry = a + b * ndvi
    if isinstance(wet_edge, (tuple, list)):
        aw, bw = wet_edge
        lst_wet = aw + bw * ndvi
    else:
        lst_wet = np.full_like(ndvi, float(wet_edge))
    return _safe_norm(lst - lst_wet, lst_dry - lst_wet, clip=(0.0, 1.0))


def cwsi(
    canopy_temp: np.ndarray,
    air_temp: np.ndarray,
    *,
    lower_limit: float,
    upper_limit: float,
) -> np.ndarray:
    """Crop Water Stress Index (Idso/Jackson 1981), empirical form, ``0-1``.

    ``CWSI = ((Tc - Ta) - LL) / (UL - LL)`` where ``LL`` is the non-water-stressed
    baseline (well-watered) canopy-air temperature difference and ``UL`` the
    fully stressed (non-transpiring) difference.

    Returns
    -------
    np.ndarray
        Stress in ``[0, 1]`` (1 = maximal water stress).
    """
    tc = np.asarray(canopy_temp, dtype=np.float64)
    ta = np.asarray(air_temp, dtype=np.float64)
    dt = tc - ta
    num = dt - lower_limit
    den = np.asarray(upper_limit - lower_limit, dtype=np.float64)
    return _safe_norm(
        num, np.full_like(dt, float(den)) if np.ndim(den) == 0 else den, clip=(0.0, 1.0)
    )


# --------------------------------------------------------------------------- #
# Anomaly vs baseline
# --------------------------------------------------------------------------- #
def anomaly_vs_baseline(
    series: np.ndarray,
    baseline: np.ndarray,
    *,
    robust: bool = False,
) -> np.ndarray:
    """Standardised anomaly (z-score) of ``series`` against a ``baseline`` sample.

    ``z = (x - mu) / sigma`` using the mean / std of ``baseline`` (a multi-year
    climatology for the same period).  With ``robust=True`` uses the median and
    a MAD-based scale (1.4826 * MAD) for outlier resistance.

    Negative z indicates a below-normal value (e.g. NDVI deficit -> stress).
    NaNs in ``baseline`` are ignored; an all-NaN or zero-variance baseline yields
    NaNs.
    """
    series = np.asarray(series, dtype=np.float64)
    base = np.asarray(baseline, dtype=np.float64)
    base = base[np.isfinite(base)]
    if base.size == 0:
        return np.full_like(series, np.nan)
    if robust:
        center = float(np.median(base))
        mad = float(np.median(np.abs(base - center)))
        scale = 1.4826 * mad
    else:
        center = float(np.mean(base))
        scale = float(np.std(base))
    if scale < _EPS:
        return np.full_like(series, np.nan)
    return (series - center) / scale


# --------------------------------------------------------------------------- #
# Stage-aware stress classifier
# --------------------------------------------------------------------------- #
# Default per-stage thresholds on a *condition* index (higher = healthier,
# 0-1 scale, e.g. VCI / VHI / OPTRAM-W).  Boundaries give the lower edge of each
# severity band; crops are more sensitive during mid-season (flowering/grain
# fill), so its thresholds are stricter (higher) than initial/late stages.
#
# Mapping: condition >= b[0] -> none(0); >= b[1] -> mild(1); >= b[2] ->
# moderate(2); >= b[3] -> high(3); else severe(4).
DEFAULT_STAGE_THRESHOLDS: dict[str, tuple[float, float, float, float]] = {
    "initial": (0.55, 0.40, 0.28, 0.18),
    "development": (0.60, 0.45, 0.32, 0.20),
    "mid-season": (0.70, 0.55, 0.40, 0.25),  # most sensitive
    "late-season": (0.55, 0.40, 0.28, 0.18),
    "mature": (0.45, 0.32, 0.22, 0.12),  # senescence expected, least sensitive
    "default": (0.60, 0.45, 0.32, 0.20),
}


@dataclass
class StressResult:
    """Per-sample stage-aware stress output."""

    severity: np.ndarray  # int 0-4
    labels: np.ndarray  # str
    condition: np.ndarray  # the (clipped) condition index used
    stressed: np.ndarray  # bool flag (severity >= 1)
    stage: np.ndarray  # the stage label(s) applied


class StageAwareStress:
    """Classify moisture stress with thresholds that depend on growth stage.

    The classifier consumes a **condition index** where higher means healthier
    (e.g. VCI, VHI, or OPTRAM soil-moisture ``W``).  For each sample it looks up
    the four-element threshold vector for that sample's growth stage and returns
    a 0-4 severity class.  If an index where *higher = more stress* is supplied
    (e.g. TVDI, CWSI), set ``higher_is_stress=True`` and it is internally
    inverted to a condition score.

    Parameters
    ----------
    thresholds
        Mapping ``stage -> (b0, b1, b2, b3)`` of descending condition cut-offs.
        Defaults to :data:`DEFAULT_STAGE_THRESHOLDS`.
    higher_is_stress
        If ``True``, the input index is treated as a stress index in ``[0, 1]``
        and converted to a condition score via ``1 - index`` before thresholding.
    """

    def __init__(
        self,
        thresholds: Mapping[str, tuple[float, float, float, float]] | None = None,
        *,
        higher_is_stress: bool = False,
    ) -> None:
        self.thresholds = (
            dict(thresholds) if thresholds is not None else dict(DEFAULT_STAGE_THRESHOLDS)
        )
        if "default" not in self.thresholds:
            self.thresholds["default"] = DEFAULT_STAGE_THRESHOLDS["default"]
        self.higher_is_stress = higher_is_stress

    def _thr_for(self, stage: str) -> np.ndarray:
        return np.asarray(self.thresholds.get(stage, self.thresholds["default"]), dtype=np.float64)

    def classify(
        self,
        index_value: np.ndarray,
        stage: str | np.ndarray,
    ) -> StressResult:
        """Classify stress severity for one or many samples.

        Parameters
        ----------
        index_value
            Condition (or stress, if ``higher_is_stress``) index value(s) in
            ``[0, 1]``.
        stage
            Growth stage label, scalar (applied to all) or array aligned with
            ``index_value``.

        Returns
        -------
        StressResult
            Severity ints (0-4), text labels, the condition score used, a
            boolean stressed flag and the applied stage.
        """
        idx = np.atleast_1d(np.asarray(index_value, dtype=np.float64))
        condition = 1.0 - idx if self.higher_is_stress else idx
        condition = np.clip(condition, 0.0, 1.0)

        if isinstance(stage, str):
            stages = np.array([stage] * idx.size, dtype=object)
        else:
            stages = np.asarray(stage, dtype=object)
            if stages.size == 1:
                stages = np.array([stages.item()] * idx.size, dtype=object)
            elif stages.size != idx.size:
                raise ValueError("stage array must match index_value length or be scalar")

        severity = np.zeros(idx.size, dtype=np.int64)
        for i in range(idx.size):
            c = condition[i]
            if not np.isfinite(c):
                severity[i] = -1  # sentinel for missing
                continue
            b = self._thr_for(str(stages[i]))
            if c >= b[0]:
                severity[i] = 0
            elif c >= b[1]:
                severity[i] = 1
            elif c >= b[2]:
                severity[i] = 2
            elif c >= b[3]:
                severity[i] = 3
            else:
                severity[i] = 4

        labels = np.array(
            [SEVERITY_LABELS.get(int(s), "unknown") if s >= 0 else "no_data" for s in severity],
            dtype=object,
        )
        stressed = severity >= 1
        return StressResult(
            severity=severity,
            labels=labels,
            condition=condition,
            stressed=stressed,
            stage=stages,
        )

    # Convenience scalar wrapper.
    def severity(self, index_value: float, stage: str) -> int:
        """Return the single integer severity class for one sample."""
        return int(self.classify(index_value, stage).severity[0])
