"""AgriStress — multi-sensor crop-stress & irrigation-advisory toolkit (ISRO BAH 2026 PS6).

NOTE: This root package initialiser is intentionally minimal. It only exposes the
package version so that subpackages (``agristress.catalog``, ``agristress.ingestion``,
...) are importable. Cross-cutting re-exports should be added by the package owner;
keeping this file thin avoids merge conflicts between independently developed modules.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
