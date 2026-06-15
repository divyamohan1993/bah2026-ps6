"""AgriStress feature-engineering subpackage.

Spectral indices (optical + SAR), GLCM texture, and phenology / growth-stage
features used by the crop-type classifier and the stage-aware stress models.

The module is intentionally dependency-light: everything works on plain
``numpy`` arrays.  Optional accelerators (``scikit-image`` for GLCM) are used
when present and fall back to a small pure-numpy implementation otherwise.
"""

from __future__ import annotations

from . import indices, phenology, texture

__all__ = ["indices", "phenology", "texture"]
