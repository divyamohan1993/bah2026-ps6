"""Stage-aware irrigation advisory and canal-command aggregation.

Maps the FAO-56 root-zone state ``(Dr, RAW, TAW, Ks, growth_stage)`` onto a 5-class
operational advisory and rolls field-level advisories up to a canal command area
with *warabandi* (rotational water supply) turn-times.

Status classes
--------------
``NO_IRRIGATION`` (0)  root zone near field capacity — hold.
``WATCH``         (1)  depletion building but well within RAW.
``IRRIGATE_SOON`` (2)  approaching the readily-available-water limit (RAW).
``IRRIGATE_NOW``  (3)  Dr has reached/exceeded RAW — stress imminent.
``CRITICAL``      (4)  severe depletion (Dr -> TAW, Ks small) — yield loss.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Iterable, Optional, Sequence

import numpy as np

__all__ = [
    "IrrigationStatus",
    "Advisory",
    "IrrigationAdvisory",
    "FieldAdvisory",
    "CommandAreaPlan",
    "aggregate_to_command",
    "advisory_map",
]


class IrrigationStatus(IntEnum):
    """Five-class irrigation advisory status (ordered by urgency)."""

    NO_IRRIGATION = 0
    WATCH = 1
    IRRIGATE_SOON = 2
    IRRIGATE_NOW = 3
    CRITICAL = 4

    @property
    def label(self) -> str:
        return self.name.replace("_", " ").title()


@dataclass
class Advisory:
    """Per-field advisory result."""

    status: IrrigationStatus
    dr: float
    raw: float
    taw: float
    ks: float
    dnet: float                 # net irrigation depth to refill to FC [mm]
    dgross: float               # gross depth = dnet / Ea [mm]
    days_to_trigger: float      # days until Dr reaches RAW at current ETc (>=0)
    depletion_fraction: float   # Dr / TAW [-]
    growth_stage: str
    recommendation: str
    ponded: bool = False


# stage-aware tightening of the management allowed depletion fraction.
# flowering / mid-season are the most sensitive -> smaller effective p.
_STAGE_P_SCALE = {
    "ini": 1.10,
    "dev": 1.00,
    "mid": 0.85,     # flowering / reproductive — irrigate earlier
    "flowering": 0.80,
    "reproductive": 0.80,
    "late": 1.05,
    "maturity": 1.20,
}


@dataclass
class IrrigationAdvisory:
    """Turn a root-zone water-balance state into an operational advisory.

    Parameters
    ----------
    application_efficiency : float
        Field application efficiency ``Ea`` used for the gross depth.
    soon_fraction : float
        Fraction of RAW at which to flag ``IRRIGATE_SOON`` (default 0.75).
    watch_fraction : float
        Fraction of RAW at which to flag ``WATCH`` (default 0.40).
    critical_ks : float
        ``Ks`` below which the status is forced to ``CRITICAL`` (default 0.4).
    """

    application_efficiency: float = 0.6
    soon_fraction: float = 0.75
    watch_fraction: float = 0.40
    critical_ks: float = 0.40

    def _effective_p(self, p: float, growth_stage: str) -> float:
        scale = _STAGE_P_SCALE.get(str(growth_stage).strip().lower(), 1.0)
        return float(np.clip(p * scale, 0.05, 0.95))

    def evaluate(
        self,
        dr: float,
        taw: float,
        p: float = 0.5,
        ks: Optional[float] = None,
        growth_stage: str = "mid",
        etc: float = 0.0,
        ponded: bool = False,
    ) -> Advisory:
        """Classify a single field.

        Parameters
        ----------
        dr : float
            Current root-zone depletion [mm].
        taw : float
            Total available water [mm].
        p : float
            Base depletion fraction (stage-adjusted internally).
        ks : float, optional
            Stress coefficient; recomputed from ``dr, taw, p_eff`` if omitted.
        growth_stage : str
            Growth-stage name (tightens ``p`` at flowering).
        etc : float
            Current crop ET [mm/day] — used for the days-to-trigger estimate.
        ponded : bool
            Paddy rice flag (continuous submergence / AWD branch).
        """
        from .water_balance import stress_coefficient  # local import: avoid cycle

        p_eff = self._effective_p(p, growth_stage)
        raw = p_eff * taw
        if ks is None:
            ks = float(stress_coefficient(dr, taw, p_eff))

        depl_frac = dr / taw if taw > 0 else 0.0
        dnet = max(dr, 0.0)
        dgross = dnet / self.application_efficiency

        # days until depletion reaches RAW at the current crop ET draw-down
        if etc > 1e-9:
            days_to_trigger = max((raw - dr) / etc, 0.0)
        else:
            days_to_trigger = float("inf") if dr < raw else 0.0

        if ponded:
            status, rec = self._classify_ponded(dr, raw, taw, growth_stage)
        else:
            status, rec = self._classify_upland(
                dr, raw, taw, ks, depl_frac, growth_stage, days_to_trigger
            )

        return Advisory(
            status=status,
            dr=float(dr),
            raw=float(raw),
            taw=float(taw),
            ks=float(ks),
            dnet=float(dnet),
            dgross=float(dgross),
            days_to_trigger=float(days_to_trigger),
            depletion_fraction=float(depl_frac),
            growth_stage=str(growth_stage),
            recommendation=rec,
            ponded=bool(ponded),
        )

    # convenience alias
    __call__ = evaluate

    def _classify_upland(
        self, dr, raw, taw, ks, depl_frac, stage, days_to_trigger
    ) -> tuple[IrrigationStatus, str]:
        soon_thr = self.soon_fraction * raw
        watch_thr = self.watch_fraction * raw

        if ks <= self.critical_ks or depl_frac >= 0.90:
            s = IrrigationStatus.CRITICAL
            rec = (
                f"CRITICAL water stress ({stage}): Ks={ks:.2f}, depletion "
                f"{depl_frac*100:.0f}% of TAW. Irrigate immediately (net "
                f"{max(dr,0):.0f} mm) to avoid yield loss."
            )
        elif dr >= raw:
            s = IrrigationStatus.IRRIGATE_NOW
            rec = (
                f"Depletion has reached RAW ({dr:.0f}>={raw:.0f} mm) at {stage}. "
                f"Irrigate now (net {max(dr,0):.0f} mm)."
            )
        elif dr >= soon_thr:
            s = IrrigationStatus.IRRIGATE_SOON
            rec = (
                f"Approaching RAW at {stage} (Dr={dr:.0f}/{raw:.0f} mm); "
                f"schedule irrigation within ~{days_to_trigger:.0f} day(s)."
            )
        elif dr >= watch_thr:
            s = IrrigationStatus.WATCH
            rec = (
                f"Soil drying at {stage} (Dr={dr:.0f}/{raw:.0f} mm). Monitor; "
                f"no irrigation required yet."
            )
        else:
            s = IrrigationStatus.NO_IRRIGATION
            rec = (
                f"Root zone near field capacity (Dr={dr:.0f} mm) at {stage}. "
                f"No irrigation needed."
            )
        return s, rec

    def _classify_ponded(
        self, dr, raw, taw, stage
    ) -> tuple[IrrigationStatus, str]:
        """Paddy-rice branch: maintain ponding, allow shallow AWD draw-down.

        For lowland rice the target is a thin ponded layer; once the depletion
        approaches the AWD threshold (~RAW, i.e. water table a few cm below
        surface) re-flood. Maturity stage permits final dry-down.
        """
        s_lower = str(stage).strip().lower()
        if s_lower in ("late", "maturity", "senescence"):
            # terminal drainage for harvest
            if dr >= raw:
                return (
                    IrrigationStatus.NO_IRRIGATION,
                    f"Rice at {stage}: terminal drainage — stop irrigation for harvest.",
                )
        if dr >= taw * 0.90:
            return (
                IrrigationStatus.CRITICAL,
                f"Paddy dried well below AWD safe limit (Dr={dr:.0f} mm). "
                f"Re-flood immediately to restore ponding.",
            )
        if dr >= raw:
            return (
                IrrigationStatus.IRRIGATE_NOW,
                f"AWD threshold reached (Dr={dr:.0f}>={raw:.0f} mm). Re-flood paddy "
                f"to 5 cm ponding now.",
            )
        if dr >= 0.6 * raw:
            return (
                IrrigationStatus.IRRIGATE_SOON,
                f"Paddy water table dropping (Dr={dr:.0f}/{raw:.0f} mm); prepare to "
                f"re-flood within 1-2 days.",
            )
        if dr <= 0.05 * max(taw, 1e-9):
            return (
                IrrigationStatus.NO_IRRIGATION,
                f"Paddy adequately ponded (Dr={dr:.0f} mm). Hold.",
            )
        return (
            IrrigationStatus.WATCH,
            f"Paddy ponding within AWD safe range (Dr={dr:.0f} mm). Monitor.",
        )


# --- command-area aggregation ----------------------------------------------------

@dataclass
class FieldAdvisory:
    """One command-area parcel: its advisory plus geometry / routing metadata."""

    field_id: str
    advisory: Advisory
    area_ha: float
    is_tail_end: bool = False
    distance_from_head_m: float = 0.0


@dataclass
class CommandAreaPlan:
    """Aggregated irrigation plan for a canal command area / outlet."""

    total_volume_m3: float
    turn_time_hours: float
    outlet_discharge_m3s: float
    n_fields: int
    n_need_irrigation: int
    order: list[str] = field(default_factory=list)         # field_ids, delivery order
    per_field_volume_m3: dict[str, float] = field(default_factory=dict)
    per_field_hours: dict[str, float] = field(default_factory=dict)


def aggregate_to_command(
    field_advisories: Iterable[FieldAdvisory],
    outlet_discharge_q: float,
    *,
    tail_end_priority: bool = True,
    only_due: bool = True,
) -> CommandAreaPlan:
    """Aggregate field advisories into a command-area volume and *warabandi* plan.

    Parameters
    ----------
    field_advisories : iterable of FieldAdvisory
        Per-parcel advisories with area [ha].
    outlet_discharge_q : float
        Available outlet discharge ``Q`` [m3/s].
    tail_end_priority : bool
        If True, order delivery tail-end-first (then by urgency) to counter the
        head-reach bias typical of canal systems.
    only_due : bool
        If True, only parcels at ``IRRIGATE_SOON`` or worse contribute volume.

    Returns
    -------
    CommandAreaPlan
        Total gross volume ``V = sum(dgross * A)``, *warabandi* turn-time
        ``T = V / Q``, and the per-field delivery order / volumes / turn durations.

    Notes
    -----
    Unit handling: ``dgross`` [mm] over ``A`` [ha] gives
    ``V = dgross[mm] * A[ha] * 10`` m3  (1 mm over 1 ha = 10 m3).
    """
    if outlet_discharge_q <= 0:
        raise ValueError("outlet_discharge_q (Q) must be > 0 m3/s.")

    fields = list(field_advisories)
    due_threshold = IrrigationStatus.IRRIGATE_SOON

    per_vol: dict[str, float] = {}
    per_hours: dict[str, float] = {}
    total_volume = 0.0
    n_need = 0

    contributing: list[FieldAdvisory] = []
    for fa in fields:
        is_due = fa.advisory.status >= due_threshold
        if is_due:
            n_need += 1
        if only_due and not is_due:
            per_vol[fa.field_id] = 0.0
            per_hours[fa.field_id] = 0.0
            continue
        vol = fa.advisory.dgross * fa.area_ha * 10.0   # mm*ha -> m3
        per_vol[fa.field_id] = vol
        per_hours[fa.field_id] = vol / outlet_discharge_q / 3600.0
        total_volume += vol
        contributing.append(fa)

    # delivery ordering: tail-end first (further from head), then urgency, then size
    def _sort_key(fa: FieldAdvisory):
        tail_rank = -(fa.distance_from_head_m) if tail_end_priority else 0.0
        flag_rank = -1.0 if (tail_end_priority and fa.is_tail_end) else 0.0
        return (
            flag_rank,
            tail_rank,
            -int(fa.advisory.status),
            -fa.advisory.dgross,
        )

    ordered = sorted(contributing, key=_sort_key)
    order_ids = [fa.field_id for fa in ordered]

    turn_time_hours = total_volume / outlet_discharge_q / 3600.0

    return CommandAreaPlan(
        total_volume_m3=float(total_volume),
        turn_time_hours=float(turn_time_hours),
        outlet_discharge_m3s=float(outlet_discharge_q),
        n_fields=len(fields),
        n_need_irrigation=n_need,
        order=order_ids,
        per_field_volume_m3=per_vol,
        per_field_hours=per_hours,
    )


# --- vectorised raster classification -------------------------------------------

def advisory_map(
    deficit_array: np.ndarray,
    raw_array: np.ndarray,
    ks_array: Optional[np.ndarray] = None,
    stage_array: Optional[np.ndarray] = None,
    taw_array: Optional[np.ndarray] = None,
    *,
    soon_fraction: float = 0.75,
    watch_fraction: float = 0.40,
    critical_ks: float = 0.40,
    nodata: float = np.nan,
) -> np.ndarray:
    """Vectorised per-pixel advisory class raster (returns ``int8``).

    Classifies each pixel into an :class:`IrrigationStatus` integer code using the
    same thresholds as :class:`IrrigationAdvisory` but fully vectorised for rasters.

    Parameters
    ----------
    deficit_array : ndarray
        Root-zone depletion ``Dr`` [mm].
    raw_array : ndarray
        Readily available water ``RAW`` [mm] (already stage-adjusted if desired).
    ks_array : ndarray, optional
        Stress coefficient. If given, ``Ks <= critical_ks`` forces ``CRITICAL``.
    stage_array : ndarray, optional
        (Reserved) per-pixel stage codes; thresholds are applied via ``raw_array``.
    taw_array : ndarray, optional
        Total available water [mm]; enables the ``depletion >= 0.9*TAW`` CRITICAL
        rule. If omitted, only the ``ks``-based critical test applies.
    nodata : float
        Value in ``deficit_array`` treated as nodata (mapped to ``-1``).

    Returns
    -------
    ndarray of int8
        Integer class codes (0-4); ``-1`` where input is nodata.
    """
    dr = np.asarray(deficit_array, dtype=float)
    raw = np.asarray(raw_array, dtype=float)
    raw = np.broadcast_to(raw, dr.shape)

    out = np.full(dr.shape, IrrigationStatus.NO_IRRIGATION, dtype=np.int8)

    soon_thr = soon_fraction * raw
    watch_thr = watch_fraction * raw

    # order matters: assign from least to most severe so severe overwrites.
    out[dr >= watch_thr] = IrrigationStatus.WATCH
    out[dr >= soon_thr] = IrrigationStatus.IRRIGATE_SOON
    out[dr >= raw] = IrrigationStatus.IRRIGATE_NOW

    crit = np.zeros(dr.shape, dtype=bool)
    if ks_array is not None:
        ks = np.broadcast_to(np.asarray(ks_array, dtype=float), dr.shape)
        crit |= ks <= critical_ks
    if taw_array is not None:
        taw = np.broadcast_to(np.asarray(taw_array, dtype=float), dr.shape)
        with np.errstate(divide="ignore", invalid="ignore"):
            crit |= np.where(taw > 0, dr / taw >= 0.90, False)
    out[crit] = IrrigationStatus.CRITICAL

    # nodata handling
    if np.isnan(nodata):
        nd = np.isnan(dr)
    else:
        nd = dr == nodata
    out[nd] = -1

    return out
