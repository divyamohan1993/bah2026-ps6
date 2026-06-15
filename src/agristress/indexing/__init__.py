"""Indexing: H3 spatial indexing + an O(1) feature store for the advisory layer.

* :mod:`~agristress.indexing.h3_index`     — thin wrappers over the ``h3`` v4 lib
  (graceful fallback when ``h3`` is absent) and a ``cube → H3 table`` flattener.
* :mod:`~agristress.indexing.feature_store`— ``FeatureStore`` interface plus an
  in-memory and a Parquet-backed reference implementation (Redis/Feast noted as the
  production backend).
"""

from __future__ import annotations

from agristress.indexing.feature_store import (
    FeatureStore,
    InMemoryFeatureStore,
    ParquetFeatureStore,
)
from agristress.indexing.h3_index import (
    H3_AVAILABLE,
    cell_to_children,
    cell_to_latlng,
    cell_to_parent,
    cells_for_polygon,
    cube_to_h3_table,
    grid_disk,
    latlng_to_cell,
)

__all__ = [
    "H3_AVAILABLE",
    "latlng_to_cell",
    "cell_to_latlng",
    "cell_to_parent",
    "cell_to_children",
    "grid_disk",
    "cells_for_polygon",
    "cube_to_h3_table",
    "FeatureStore",
    "InMemoryFeatureStore",
    "ParquetFeatureStore",
]
