"""Feature store: O(1) ``(h3_cell, date) → features`` lookup for the advisory layer.

The advisory service needs constant-time access to the per-cell, per-date feature
vector (VIs, soil moisture, phenophase, stress indices …). This module defines the
:class:`FeatureStore` interface and two reference implementations:

* :class:`InMemoryFeatureStore` — a plain ``dict`` keyed by ``(h3_cell, date)``;
  O(1) put/get, ideal for tests, notebooks and a single-process API.
* :class:`ParquetFeatureStore`  — partitioned Parquet on disk; rows are buffered
  in memory for O(1) reads and flushed to columnar Parquet for durable, queryable
  storage (good for the pilot command-area scale).

For production multi-tenant serving, back the same interface with **Redis**
(hash per cell) or **Feast** (online store) — the contract is intentionally tiny so
swapping the backend is a drop-in change.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


def _norm_date(date: Any) -> str:
    """Normalise a date key to an ISO ``YYYY-MM-DD`` string."""
    if hasattr(date, "strftime"):
        return date.strftime("%Y-%m-%d")
    s = str(date)
    return s[:10] if len(s) >= 10 else s


class FeatureStore(ABC):
    """Abstract key-value feature store keyed on ``(h3_cell, date)``.

    Implementations must provide :meth:`put` and :meth:`get`. Convenience helpers
    (:meth:`get_or_default`, ``__contains__``) are derived from those.
    """

    @abstractmethod
    def put(self, h3_cell: str, date: Any, features: dict[str, Any]) -> None:
        """Store the ``features`` mapping for ``(h3_cell, date)`` (upsert)."""

    @abstractmethod
    def get(self, h3_cell: str, date: Any) -> dict[str, Any] | None:
        """Return the stored features for ``(h3_cell, date)`` or ``None`` if absent."""

    # -- derived helpers -----------------------------------------------------
    def get_or_default(
        self, h3_cell: str, date: Any, default: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        """:meth:`get` with a caller-supplied default for misses."""
        out = self.get(h3_cell, date)
        return out if out is not None else default

    def __contains__(self, key: tuple[str, Any]) -> bool:
        h3_cell, date = key
        return self.get(h3_cell, date) is not None


class InMemoryFeatureStore(FeatureStore):
    """In-memory ``dict``-backed store — O(1) put/get, non-durable."""

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], dict[str, Any]] = {}

    def put(self, h3_cell: str, date: Any, features: dict[str, Any]) -> None:
        self._store[(h3_cell, _norm_date(date))] = dict(features)

    def get(self, h3_cell: str, date: Any) -> dict[str, Any] | None:
        out = self._store.get((h3_cell, _norm_date(date)))
        return dict(out) if out is not None else None

    def bulk_put(self, rows) -> int:
        """Insert many ``(h3_cell, date, features)`` tuples; returns the count."""
        n = 0
        for h3_cell, date, features in rows:
            self.put(h3_cell, date, features)
            n += 1
        return n

    def __len__(self) -> int:
        return len(self._store)

    def keys(self):
        """Iterate stored ``(h3_cell, date)`` keys."""
        return self._store.keys()

    @classmethod
    def from_h3_table(cls, table, *, value_col: str = "value") -> "InMemoryFeatureStore":
        """Build a store from a tidy ``(h3_cell, date, variable, value)`` DataFrame.

        Rows sharing ``(h3_cell, date)`` are pivoted into a single feature dict
        ``{variable: value}``.
        """
        store = cls()
        # Pivot long → wide per (cell, date) without requiring pandas at import time.
        grouped: dict[tuple[str, str], dict[str, Any]] = {}
        for row in table.itertuples(index=False):
            key = (row.h3_cell, _norm_date(row.date))
            grouped.setdefault(key, {})[row.variable] = getattr(row, value_col)
        for (cell, date), feats in grouped.items():
            store.put(cell, date, feats)
        return store


class ParquetFeatureStore(FeatureStore):
    """Parquet-backed store: in-memory index for O(1) reads, durable on flush.

    Writes are buffered in a ``dict`` (so :meth:`get` is O(1)); :meth:`flush`
    serialises the buffer to a single Parquet file (one row per ``(cell, date)``,
    features stored as a JSON string column for schema flexibility). :meth:`load`
    repopulates the buffer from an existing file. Requires ``pandas`` + a Parquet
    engine (``pyarrow``).

    Parameters
    ----------
    path
        Parquet file path (created on :meth:`flush`).
    autoload
        Load an existing file at ``path`` on construction (default ``True``).
    """

    def __init__(self, path: str | Path, *, autoload: bool = True) -> None:
        self.path = Path(path)
        self._buffer: dict[tuple[str, str], dict[str, Any]] = {}
        self._dirty = False
        if autoload and self.path.exists():
            self.load()

    def put(self, h3_cell: str, date: Any, features: dict[str, Any]) -> None:
        self._buffer[(h3_cell, _norm_date(date))] = dict(features)
        self._dirty = True

    def get(self, h3_cell: str, date: Any) -> dict[str, Any] | None:
        out = self._buffer.get((h3_cell, _norm_date(date)))
        return dict(out) if out is not None else None

    def flush(self) -> Path:
        """Serialise the in-memory buffer to ``self.path`` (Parquet) and return it."""
        import pandas as pd

        rows = [
            {"h3_cell": cell, "date": date, "features": json.dumps(feats)}
            for (cell, date), feats in self._buffer.items()
        ]
        df = pd.DataFrame(rows, columns=["h3_cell", "date", "features"])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.path, index=False)
        self._dirty = False
        return self.path

    def load(self) -> int:
        """(Re)load the buffer from ``self.path``; returns the row count."""
        import pandas as pd

        df = pd.read_parquet(self.path)
        self._buffer.clear()
        for row in df.itertuples(index=False):
            self._buffer[(row.h3_cell, _norm_date(row.date))] = json.loads(row.features)
        self._dirty = False
        return len(self._buffer)

    def __len__(self) -> int:
        return len(self._buffer)

    def __enter__(self) -> "ParquetFeatureStore":
        return self

    def __exit__(self, *exc) -> None:
        if self._dirty:
            self.flush()
