"""Preprocessing: optical (cloud/shadow + bandpass), SAR (speckle + features), harmonisation.

All public helpers are import-clean and runnable **without credentials** on synthetic
numpy/xarray arrays (DEMO mode). Heavy / vendor-specific steps (CloudScore+, Fmask,
RTC terrain flattening) are exposed as thin interfaces that degrade gracefully to a
pure-numpy implementation when the backing service / library is unavailable.
"""

from __future__ import annotations

from agristress.preprocessing.harmonize import (
    cross_calibrate,
    resample_nearest,
    to_common_grid,
)
from agristress.preprocessing.optical import (
    apply_cloud_mask,
    cloud_score_plus,
    fmask,
    harmonize_bandpass,
    mask_s2_scl,
    scale_surface_reflectance,
)
from agristress.preprocessing.sar import (
    compute_sar_features,
    refined_lee,
    terrain_flatten,
    to_db,
    to_linear,
)

__all__ = [
    # optical
    "mask_s2_scl",
    "cloud_score_plus",
    "fmask",
    "apply_cloud_mask",
    "scale_surface_reflectance",
    "harmonize_bandpass",
    # sar
    "refined_lee",
    "to_db",
    "to_linear",
    "terrain_flatten",
    "compute_sar_features",
    # harmonize
    "to_common_grid",
    "resample_nearest",
    "cross_calibrate",
]
