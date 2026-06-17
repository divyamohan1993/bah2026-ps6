"""AgriStress irrigation-advisory engine (ISRO BAH 2026 PS6).

Pipeline: FAO-56 reference ET (``et0``) -> crop coefficient (``kc``) -> root-zone
water balance & 8-day deficit (``water_balance``) -> stage-aware advisory and
canal-command aggregation (``advisory``).

All numerics are pure ``numpy`` / ``pandas`` and validated against the FAO-56
worked examples (see ``tests/test_irrigation.py``).
"""

from __future__ import annotations

from .advisory import (
    Advisory,
    CommandAreaPlan,
    FieldAdvisory,
    IrrigationAdvisory,
    IrrigationStatus,
    advisory_map,
    aggregate_to_command,
)
from .et0 import (
    et0_hargreaves,
    et0_penman_monteith,
    extraterrestrial_radiation,
)
from .kc import (
    CropCoefficients,
    adjust_kc_mid_for_climate,
    get_crop,
    kc_curve,
    kc_for_stage,
    kc_from_ndvi,
)
from .water_balance import (
    RootZoneWaterBalance,
    SoilProperties,
    effective_rainfall_scs_cn,
    effective_rainfall_usda_monthly,
    eight_day_deficit,
    stress_coefficient,
    swi_from_surface_sm,
)

__all__ = [
    "Advisory",
    "CommandAreaPlan",
    "CropCoefficients",
    "FieldAdvisory",
    # advisory
    "IrrigationAdvisory",
    "IrrigationStatus",
    "RootZoneWaterBalance",
    # water balance
    "SoilProperties",
    "adjust_kc_mid_for_climate",
    "advisory_map",
    "aggregate_to_command",
    "effective_rainfall_scs_cn",
    "effective_rainfall_usda_monthly",
    "eight_day_deficit",
    "et0_hargreaves",
    # et0
    "et0_penman_monteith",
    "extraterrestrial_radiation",
    "get_crop",
    "kc_curve",
    # kc
    "kc_for_stage",
    "kc_from_ndvi",
    "stress_coefficient",
    "swi_from_surface_sm",
]
