"""Crop coefficient (Kc) handling — FAO-56 single crop coefficient curves.

Loads the stage-length / Kc table from ``configs/crops/kc_curves.yaml`` and exposes:

* :func:`kc_for_stage`  — Kc for a named growth stage.
* :func:`kc_curve`      — Kc as a function of days-after-sowing (4-stage interp).
* :func:`kc_from_ndvi`  — empirical Kc from NDVI (canopy-greenness proxy).
* :func:`adjust_kc_mid_for_climate` — FAO-56 Eq. 62 climate correction.
* :func:`growth_stage_for_das` — stage name for a given day-after-sowing.

The configuration is cached after first load; pass an explicit ``config_path`` (or
set ``AGRISTRESS_KC_CONFIG``) to override the default search.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

__all__ = [
    "STAGES",
    "CropCoefficients",
    "adjust_kc_mid_for_climate",
    "available_crops",
    "get_crop",
    "growth_stage_for_das",
    "kc_curve",
    "kc_for_stage",
    "kc_from_ndvi",
    "load_kc_config",
]

#: Canonical FAO-56 growth-stage names, in chronological order.
STAGES = ("ini", "dev", "mid", "late")

# Friendly aliases accepted by ``kc_for_stage`` / advisory stage handling.
_STAGE_ALIASES = {
    "initial": "ini",
    "ini": "ini",
    "development": "dev",
    "dev": "dev",
    "crop_development": "dev",
    "mid": "mid",
    "mid-season": "mid",
    "mid_season": "mid",
    "midseason": "mid",
    "flowering": "mid",
    "reproductive": "mid",
    "late": "late",
    "late-season": "late",
    "late_season": "late",
    "maturity": "late",
    "senescence": "late",
}


def _default_config_path() -> Path:
    """Locate ``configs/crops/kc_curves.yaml``.

    Search order: ``AGRISTRESS_KC_CONFIG`` env var, then walk up from this module
    until a ``configs/crops/kc_curves.yaml`` is found (repo root), then CWD.
    """
    env = os.environ.get("AGRISTRESS_KC_CONFIG")
    if env:
        return Path(env)

    rel = Path("configs") / "crops" / "kc_curves.yaml"
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / rel
        if candidate.is_file():
            return candidate
    return Path.cwd() / rel


@lru_cache(maxsize=8)
def _load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict) or "crops" not in data:
        raise ValueError(f"Malformed kc_curves config (missing 'crops'): {path}")
    return data


def load_kc_config(config_path: os.PathLike | str | None = None) -> dict:
    """Return the parsed Kc configuration dict (cached)."""
    path = Path(config_path) if config_path is not None else _default_config_path()
    if not path.is_file():
        raise FileNotFoundError(f"Kc curve config not found: {path}")
    return _load_yaml(str(path))


@dataclass(frozen=True)
class CropCoefficients:
    """Typed view of a single crop's Kc parameters."""

    name: str
    stage_lengths_days: tuple[int, int, int, int]
    kc_ini: float
    kc_mid: float
    kc_end: float
    zr_m: float
    depletion_p: float
    ponded: bool = False

    @property
    def season_length_days(self) -> int:
        return int(sum(self.stage_lengths_days))

    @property
    def stage_boundaries(self) -> tuple[int, int, int, int]:
        """Cumulative day-after-sowing at the END of each of the 4 stages."""
        l_ini, l_dev, l_mid, l_late = self.stage_lengths_days
        return (l_ini, l_ini + l_dev, l_ini + l_dev + l_mid, l_ini + l_dev + l_mid + l_late)


def available_crops(config_path: os.PathLike | str | None = None) -> list[str]:
    """List crop keys present in the configuration."""
    return sorted(load_kc_config(config_path)["crops"].keys())


def get_crop(crop: str, config_path: os.PathLike | str | None = None) -> CropCoefficients:
    """Return :class:`CropCoefficients` for *crop* (case-insensitive)."""
    cfg = load_kc_config(config_path)["crops"]
    key = crop.lower()
    if key not in cfg:
        raise KeyError(f"Unknown crop '{crop}'. Available: {sorted(cfg.keys())}")
    c = cfg[key]
    sl = tuple(int(x) for x in c["stage_lengths_days"])
    if len(sl) != 4:
        raise ValueError(f"{crop}: stage_lengths_days must have 4 entries, got {sl}")
    return CropCoefficients(
        name=key,
        stage_lengths_days=sl,  # type: ignore[arg-type]
        kc_ini=float(c["kc_ini"]),
        kc_mid=float(c["kc_mid"]),
        kc_end=float(c["kc_end"]),
        zr_m=float(c["zr_m"]),
        depletion_p=float(c["depletion_p"]),
        ponded=bool(c.get("ponded", False)),
    )


def _normalise_stage(stage: str) -> str:
    key = str(stage).strip().lower()
    if key in _STAGE_ALIASES:
        return _STAGE_ALIASES[key]
    raise KeyError(f"Unknown growth stage '{stage}'. Use one of {STAGES} (or aliases).")


def kc_for_stage(crop: str, stage: str, config_path: os.PathLike | str | None = None) -> float:
    """Return the representative Kc for a named growth *stage* of *crop*.

    The development ("dev") stage is transitional; its representative value is the
    mean of ``Kc_ini`` and ``Kc_mid``. Initial/mid/late map directly to the table.
    """
    c = get_crop(crop, config_path)
    s = _normalise_stage(stage)
    if s == "ini":
        return c.kc_ini
    if s == "dev":
        return 0.5 * (c.kc_ini + c.kc_mid)
    if s == "mid":
        return c.kc_mid
    return c.kc_end  # late


def growth_stage_for_das(
    crop: str, days_after_sowing: float, config_path: os.PathLike | str | None = None
) -> str:
    """Return the stage name ('ini'|'dev'|'mid'|'late') for a day-after-sowing."""
    c = get_crop(crop, config_path)
    b_ini, b_dev, b_mid, _b_late = c.stage_boundaries
    das = float(days_after_sowing)
    if das <= b_ini:
        return "ini"
    if das <= b_dev:
        return "dev"
    if das <= b_mid:
        return "mid"
    return "late"


def kc_curve(
    crop: str,
    days_after_sowing: float | Sequence[float] | np.ndarray,
    config_path: os.PathLike | str | None = None,
) -> np.ndarray:
    """Single crop coefficient Kc(t) from the 4-stage FAO-56 curve.

    Piecewise construction (FAO-56 Fig. 25):

    * ``ini``  : constant ``Kc_ini``
    * ``dev``  : linear ramp ``Kc_ini -> Kc_mid``
    * ``mid``  : constant ``Kc_mid``
    * ``late`` : linear ramp ``Kc_mid -> Kc_end``

    Days beyond the season are clamped to ``Kc_end`` (post-harvest bare soil should
    use ``Kc_ini``-type values, handled elsewhere).

    Returns a float for scalar input, else a numpy array.
    """
    c = get_crop(crop, config_path)
    l_ini, l_dev, l_mid, l_late = c.stage_lengths_days
    b_ini = l_ini
    b_dev = l_ini + l_dev
    b_mid = l_ini + l_dev + l_mid
    b_late = b_mid + l_late

    das = np.asarray(days_after_sowing, dtype=float)
    kc = np.empty_like(das, dtype=float)

    # initial (and any negative / pre-sowing days -> Kc_ini)
    m_ini = das <= b_ini
    kc[m_ini] = c.kc_ini

    # development : linear ini -> mid
    m_dev = (das > b_ini) & (das <= b_dev)
    frac = (das[m_dev] - b_ini) / float(l_dev) if l_dev > 0 else np.zeros(np.count_nonzero(m_dev))
    kc[m_dev] = c.kc_ini + frac * (c.kc_mid - c.kc_ini)

    # mid season : flat
    m_mid = (das > b_dev) & (das <= b_mid)
    kc[m_mid] = c.kc_mid

    # late season : linear mid -> end
    m_late = (das > b_mid) & (das <= b_late)
    if l_late > 0:
        frac = (das[m_late] - b_mid) / float(l_late)
    else:
        frac = np.zeros(np.count_nonzero(m_late))
    kc[m_late] = c.kc_mid + frac * (c.kc_end - c.kc_mid)

    # beyond season -> clamp to Kc_end
    kc[das > b_late] = c.kc_end

    return float(kc) if kc.ndim == 0 else kc


def kc_from_ndvi(
    ndvi: float | np.ndarray,
    a: float = 1.457,
    b: float = -0.1725,
    kc_min: float = 0.0,
    kc_max: float = 1.3,
) -> np.ndarray:
    """Empirical Kc from NDVI: ``Kc = a * NDVI + b`` (linear), clamped.

    Default coefficients (a=1.457, b=-0.1725) give Kc ~= 1.15 at NDVI ~= 0.86
    (closed canopy) and Kc ~= 0 at NDVI ~= 0.12 (bare soil), consistent with the
    FAO-56 mid-season reference. Result is clipped to ``[kc_min, kc_max]``.
    """
    ndvi = np.asarray(ndvi, dtype=float)
    kc = a * ndvi + b
    kc = np.clip(kc, kc_min, kc_max)
    return float(kc) if kc.ndim == 0 else kc


def adjust_kc_mid_for_climate(
    kc_mid: float,
    u2: float = 2.0,
    rh_min: float = 45.0,
    plant_height_m: float = 0.3,
) -> float:
    """FAO-56 Eq. 62 climate adjustment of ``Kc_mid`` (and Kc_end).

    ::

        Kc = Kc(table) + [0.04*(u2 - 2) - 0.004*(RHmin - 45)] * (h/3)^0.3

    Adjusts the tabulated (sub-humid, u2=2 m/s) value for the local wind / humidity
    regime. ``RHmin`` is clipped to the valid 20-80 % range and ``h`` to 0.1-10 m.
    """
    rh_min = float(np.clip(rh_min, 20.0, 80.0))
    h = float(np.clip(plant_height_m, 0.1, 10.0))
    adj = (0.04 * (float(u2) - 2.0) - 0.004 * (rh_min - 45.0)) * (h / 3.0) ** 0.3
    return float(kc_mid + adj)
