"""Cache abstraction for the AgriStress serving layer.

The serving API is an *O(1) read-hot-path*: clients (the dashboard / field app)
query pre-materialised crop / stress / advisory values keyed by ``(layer, h3, date)``.
To keep latency flat under load we sit a cache in front of the feature store.

Two backends are supported behind one :class:`Cache` interface:

* **Redis** — used automatically when the ``REDIS_URL`` environment variable is
  set *and* the :mod:`redis` client import succeeds. Values are JSON-encoded.
* **In-memory LRU** — the default, fully-implemented, zero-dependency fallback so
  the API runs offline with no external services. Backed by an ``OrderedDict``
  with a bounded capacity (true LRU eviction) and per-key TTL.

Nothing here requires cloud credentials. The in-memory path is the reference
implementation; Redis is an optional drop-in for horizontal scaling.
"""

from __future__ import annotations

import functools
import json
import os
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Hashable, Iterable
from typing import Any

__all__ = ["Cache", "cache_key", "get_cache", "set_cache"]

# Sentinel distinguishing "key absent" from a stored ``None`` value.
_MISSING = object()


def cache_key(layer: str, h3: str | None = None, date: str | None = None, **extra: Any) -> str:
    """Build a deterministic cache key.

    Keys are namespaced by ``layer`` and ordered so identical logical lookups
    always collapse to the same string regardless of kwarg ordering. This is the
    canonical ``(layer, h3, date)`` materialised-view key used across serving.
    """

    parts: list[str] = [str(layer)]
    if h3 is not None:
        parts.append(f"h3={h3}")
    if date is not None:
        parts.append(f"date={date}")
    for name in sorted(extra):
        parts.append(f"{name}={extra[name]}")
    return ":".join(parts)


class _LRU:
    """A small thread-safe LRU dict with per-key TTL.

    Capacity-bounded: inserting beyond ``maxsize`` evicts the least-recently used
    entry, giving amortised O(1) ``get``/``set`` — the property the serving layer
    relies on for a flat read latency profile.
    """

    def __init__(self, maxsize: int = 4096) -> None:
        self.maxsize = max(1, int(maxsize))
        self._data: "OrderedDict[str, tuple[Any, float | None]]" = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str, default: Any = _MISSING) -> Any:
        now = time.monotonic()
        with self._lock:
            item = self._data.get(key, _MISSING)
            if item is _MISSING:
                self.misses += 1
                return default
            value, expires = item
            if expires is not None and expires < now:
                # Expired — evict lazily.
                self._data.pop(key, None)
                self.misses += 1
                return default
            self._data.move_to_end(key)
            self.hits += 1
            return value

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        expires = (time.monotonic() + ttl) if ttl and ttl > 0 else None
        with self._lock:
            self._data[key] = (value, expires)
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self.hits = 0
            self.misses = 0

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not _MISSING

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


class Cache:
    """Unified cache facade.

    Parameters
    ----------
    redis_url:
        Connection URL. Defaults to ``$REDIS_URL``. When unset/unavailable the
        in-memory LRU backend is used.
    maxsize:
        Capacity of the in-memory LRU backend (ignored for Redis).
    namespace:
        Prefix applied to every key (useful to isolate environments on a shared
        Redis instance).
    default_ttl:
        Optional default TTL (seconds) applied when callers don't pass one.
    """

    def __init__(
        self,
        redis_url: str | None = None,
        maxsize: int = 4096,
        namespace: str = "agristress",
        default_ttl: float | None = None,
    ) -> None:
        self.namespace = namespace
        self.default_ttl = default_ttl
        self._local = _LRU(maxsize=maxsize)
        self._redis: Any = None
        self.backend = "memory"

        url = redis_url if redis_url is not None else os.environ.get("REDIS_URL")
        if url:
            self._try_connect_redis(url)

    # -- backend wiring ----------------------------------------------------
    def _try_connect_redis(self, url: str) -> None:
        try:  # pragma: no cover - exercised only when redis is installed
            import redis  # type: ignore

            client = redis.Redis.from_url(url, decode_responses=True)
            client.ping()
            self._redis = client
            self.backend = "redis"
        except Exception:
            # Any failure (missing lib, refused connection) silently falls back
            # to the in-memory backend so the API never hard-fails on startup.
            self._redis = None
            self.backend = "memory"

    def _nk(self, key: str) -> str:
        return f"{self.namespace}:{key}"

    # -- core operations ---------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        nk = self._nk(key)
        if self._redis is not None:  # pragma: no cover
            raw = self._redis.get(nk)
            if raw is None:
                return default
            try:
                return json.loads(raw)
            except (TypeError, ValueError):
                return raw
        val = self._local.get(nk, _MISSING)
        return default if val is _MISSING else val

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        nk = self._nk(key)
        ttl = self.default_ttl if ttl is None else ttl
        if self._redis is not None:  # pragma: no cover
            payload = value if isinstance(value, (str, bytes)) else json.dumps(value, default=str)
            if ttl and ttl > 0:
                self._redis.set(nk, payload, ex=int(ttl))
            else:
                self._redis.set(nk, payload)
            return
        self._local.set(nk, value, ttl=ttl)

    def delete(self, key: str) -> None:
        nk = self._nk(key)
        if self._redis is not None:  # pragma: no cover
            self._redis.delete(nk)
            return
        self._local.delete(nk)

    def __contains__(self, key: str) -> bool:
        return self.get(key, _MISSING) is not _MISSING

    def clear(self) -> None:
        if self._redis is not None:  # pragma: no cover
            for k in self._redis.scan_iter(f"{self.namespace}:*"):
                self._redis.delete(k)
            return
        self._local.clear()

    # -- materialised-view helpers ----------------------------------------
    def get_or_set(
        self,
        key: str,
        producer: Callable[[], Any],
        ttl: float | None = None,
    ) -> Any:
        """Return cached ``key`` or compute via ``producer`` and store it.

        This is the O(1) materialised-lookup primitive: a cache hit returns
        immediately; a miss computes once and back-fills.
        """

        val = self.get(key, _MISSING)
        if val is not _MISSING:
            return val
        produced = producer()
        self.set(key, produced, ttl=ttl)
        return produced

    def memoize(
        self,
        layer: str,
        ttl: float | None = None,
        key_args: Iterable[str] | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator caching a function's result under a ``(layer, ...)`` key.

        ``key_args`` selects which keyword arguments participate in the key
        (defaults to ``h3`` and ``date``). Intended for the feature-store lookup
        functions backing the serving endpoints.
        """

        selected = tuple(key_args) if key_args is not None else ("h3", "date")

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                key_kwargs = {k: kwargs.get(k) for k in selected if kwargs.get(k) is not None}
                key = cache_key(layer, **key_kwargs)
                return self.get_or_set(key, lambda: func(*args, **kwargs), ttl=ttl)

            wrapper.cache_clear = self.clear  # type: ignore[attr-defined]
            return wrapper

        return decorator

    def stats(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "namespace": self.namespace,
            "size": len(self._local) if self._redis is None else None,
            "hits": self._local.hits,
            "misses": self._local.misses,
        }


# Module-level default cache used by the API when no instance is injected.
_DEFAULT: Cache | None = None
_DEFAULT_LOCK = threading.Lock()


def get_cache() -> Cache:
    """Return (lazily creating) the process-wide default :class:`Cache`."""

    global _DEFAULT
    if _DEFAULT is None:
        with _DEFAULT_LOCK:
            if _DEFAULT is None:
                _DEFAULT = Cache()
    return _DEFAULT


def set_cache(cache: Cache) -> None:
    """Override the process-wide default cache (used in tests / app factory)."""

    global _DEFAULT
    _DEFAULT = cache


def _unused_hashable(_: Hashable) -> None:  # pragma: no cover - typing import guard
    return None
