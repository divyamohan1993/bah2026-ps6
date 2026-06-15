"""AgriStress modelling subpackage.

Crop-type classification (RandomForest / gradient-boosting / satellite
embeddings), stage-aware moisture-stress detection, a training driver with a
synthetic-data generator for offline end-to-end demos, and accuracy-assessment
utilities (overall accuracy, Cohen's kappa, Olofsson area-adjusted accuracy).
"""

from __future__ import annotations

__all__ = [
    "crop_classifier",
    "stress",
    "evaluate",
    "train",
]
