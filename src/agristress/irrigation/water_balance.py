"""FAO-56 root-zone soil-water balance.

Implements the daily soil-water depletion bookkeeping of FAO-56 Chapter 8
(Eqs. 82-90) for a single root zone:

    Dr_i = Dr_{i-1} - (P - RO)_i - I_i - CR_i + ETc_adj_i + DP_i

with

    TAW = 1000 * (theta_FC - theta_WP) * Zr           (Eq. 82)
    RAW = p * TAW                                       (Eq. 83)
    Ks  = (TAW - Dr) / ((1 - p) * TAW),  clipped [0,1]  (Eq. 84)
    ETc_adj = Ks * (Kcb + Ke) * ET0      (single-Kc: Ks * Kc * ET0)  (Eq. 81/80)

Effective rainfall can be estimated with the USDA-SCS monthly method (Eq. via
FAO-56 / USDA TR-21) or a daily SCS curve-number runoff model.

The module is pure ``numpy`` so :class:`RootZoneWaterBalance` can be vectorised over
a raster of fields by passing array soil properties.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

__all__ = [
    "SoilProperties",
    "taw_from_soil",
    "raw_from_taw",
    "stress_coefficient",
    "effective_rainfall_usda_monthly",
    "effective_rainfall_scs_cn",
    "swi_from_surface_sm",
    "WaterBalanceState",
    "RootZoneWaterBalance",
    "eight_day_deficit",
]


# --- soil ------------------------------------------------------------------------

@dataclass
class SoilProperties:
    """Root-zone soil hydraulic properties.

    Parameters
    ----------
    theta_fc : float
        Volumetric water content at field capacity [m3 m-3].
    theta_wp : float
        Volumetric water content at permanent wilting point [m3 m-3].
    zr : float
        Effective rooting depth [m].
    p : float
        Soil-water depletion fraction for no stress [-] (default 0.5).
    theta_sat : float, optional
        Saturation water content [m3 m-3]; defaults to ``theta_fc`` (no transient
        ponding storage modelled unless provided).
    """

    theta_fc: float
    theta_wp: float
    zr: float
    p: float = 0.5
    theta_sat: Optional[float] = None

    def taw(self) -> float:
        """Total available water in the root zone [mm] (FAO-56 Eq. 82)."""
        return 1000.0 * (self.theta_fc - self.theta_wp) * self.zr

    def raw(self, p: Optional[float] = None) -> float:
        """Readily available water [mm] (FAO-56 Eq. 83): ``RAW = p * TAW``."""
        pp = self.p if p is None else p
        return pp * self.taw()

    def total_evaporable_water(self) -> float:
        """Crude TEW for surface layer [mm] (informational helper)."""
        ze = 0.10  # 0.10 m evaporable surface layer (FAO-56 Table 19)
        return 1000.0 * (self.theta_fc - 0.5 * self.theta_wp) * ze


def taw_from_soil(theta_fc: float, theta_wp: float, zr: float) -> float:
    """``TAW = 1000 * (theta_fc - theta_wp) * Zr`` [mm]."""
    return 1000.0 * (np.asarray(theta_fc, float) - np.asarray(theta_wp, float)) * np.asarray(zr, float)


def raw_from_taw(taw: float, p: float) -> float:
    """``RAW = p * TAW`` [mm]."""
    return np.asarray(p, float) * np.asarray(taw, float)


def stress_coefficient(dr: float, taw: float, p: float) -> np.ndarray:
    r"""Water-stress coefficient :math:`K_s` [-] (FAO-56 Eq. 84).

    ::

        Ks = (TAW - Dr) / ((1 - p) * TAW)   for Dr > RAW
        Ks = 1                              for Dr <= RAW

    clipped to ``[0, 1]``. ``p`` is clipped to ``[0, 1)`` to avoid division by zero.
    """
    dr = np.asarray(dr, dtype=float)
    taw = np.asarray(taw, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 0.0, 0.999999)
    raw = p * taw

    with np.errstate(divide="ignore", invalid="ignore"):
        ks = np.where(
            taw > 0.0,
            (taw - dr) / ((1.0 - p) * taw),
            1.0,
        )
    # no stress while within readily-available water
    ks = np.where(dr <= raw, 1.0, ks)
    ks = np.clip(ks, 0.0, 1.0)
    return float(ks) if ks.ndim == 0 else ks


# --- effective rainfall ----------------------------------------------------------

def effective_rainfall_usda_monthly(
    p_month_mm: float | np.ndarray,
) -> np.ndarray:
    """USDA-SCS effective monthly rainfall [mm] (TR-21, as in FAO-56 / CROPWAT).

    ::

        Peff = P * (125 - 0.2 * P) / 125     for P <= 250 mm
        Peff = 125 + 0.1 * P                 for P >  250 mm
    """
    p = np.asarray(p_month_mm, dtype=float)
    peff = np.where(
        p <= 250.0,
        p * (125.0 - 0.2 * p) / 125.0,
        125.0 + 0.1 * p,
    )
    peff = np.clip(peff, 0.0, None)
    return float(peff) if peff.ndim == 0 else peff


def effective_rainfall_scs_cn(
    p_day_mm: float | np.ndarray,
    curve_number: float = 75.0,
) -> np.ndarray:
    """Daily effective rainfall via the SCS curve-number runoff model [mm].

    Runoff ``Q`` (SCS-CN, AMC-II)::

        S  = 25400 / CN - 254                 (potential retention, mm)
        Ia = 0.2 * S                          (initial abstraction)
        Q  = (P - Ia)^2 / (P - Ia + S)        for P > Ia, else 0

    Effective (infiltrated) rainfall available to the root zone is ``P - Q``.
    """
    p = np.asarray(p_day_mm, dtype=float)
    cn = np.clip(np.asarray(curve_number, dtype=float), 1.0, 100.0)
    s = 25400.0 / cn - 254.0
    ia = 0.2 * s
    runoff = np.where(p > ia, np.power(p - ia, 2) / (p - ia + s), 0.0)
    peff = np.clip(p - runoff, 0.0, None)
    return float(peff) if peff.ndim == 0 else peff


# --- satellite soil-moisture initialisation -------------------------------------

def swi_from_surface_sm(
    ssm: float | np.ndarray,
    time: float | np.ndarray | None = None,
    t_char: float = 5.0,
    prev_swi: float | np.ndarray | None = None,
    prev_time: float | np.ndarray | None = None,
) -> np.ndarray:
    """Soil Water Index (SWI) — exponential filter of surface soil moisture.

    Wagner et al. (1999) recursive exponential filter that propagates a surface
    soil-moisture observation (e.g. SMAP / Sentinel-1 SSM) to the root zone with a
    characteristic time-length ``T`` [days]::

        SWI_t = SWI_{t-1} + K_t * (SSM_t - SWI_{t-1})
        K_t   = K_{t-1} / (K_{t-1} + exp(-(t - t_{t-1}) / T))

    With no prior state it simply returns ``ssm`` (cold start). Both ``ssm`` and the
    returned SWI are expected in degree-of-saturation units ``[0, 1]`` (or a
    consistent ``[0, 100]``), which :class:`RootZoneWaterBalance` converts to an
    initial depletion via ``Dr0 = TAW * (1 - SWI_norm)``.
    """
    ssm = np.asarray(ssm, dtype=float)
    if prev_swi is None or time is None or prev_time is None:
        return float(ssm) if ssm.ndim == 0 else ssm
    prev_swi = np.asarray(prev_swi, dtype=float)
    dt = np.asarray(time, dtype=float) - np.asarray(prev_time, dtype=float)
    gain = 1.0 / (1.0 + np.exp(-np.clip(dt, 0.0, None) / float(t_char)))
    swi = prev_swi + gain * (ssm - prev_swi)
    return float(swi) if swi.ndim == 0 else swi


# --- daily root-zone water balance ----------------------------------------------

@dataclass
class WaterBalanceState:
    """Single-day output of the root-zone water balance."""

    day: int
    dr: float          # root-zone depletion at end of day [mm]
    ks: float          # stress coefficient [-]
    etc: float         # potential crop ET, Kc*ET0 [mm]
    etc_adj: float     # actual (stress-reduced) crop ET, Ks*Kc*ET0 [mm]
    peff: float        # effective (infiltrated) rainfall [mm]
    irrigation: float  # applied net irrigation [mm]
    deep_perc: float   # deep percolation below root zone [mm]
    runoff: float      # surface runoff [mm]


@dataclass
class RootZoneWaterBalance:
    """Daily FAO-56 single-layer root-zone water balance.

    Parameters
    ----------
    soil : SoilProperties
        Root-zone soil properties (provides TAW, RAW, p).
    dr0 : float
        Initial depletion [mm] at the start of the simulation (0 = field capacity).

    Notes
    -----
    Sign convention follows FAO-56: depletion ``Dr`` increases with crop water use
    (``ETc_adj``) and deep percolation, and decreases with rainfall/irrigation/
    capillary rise. ``Dr`` is bounded to ``[0, TAW]``; water that would push it
    below 0 (over-filling) becomes deep percolation ``DP``.
    """

    soil: SoilProperties
    dr0: float = 0.0
    _dr: float = field(init=False)

    def __post_init__(self) -> None:
        self._dr = float(np.clip(self.dr0, 0.0, self.soil.taw()))

    # -- properties ---------------------------------------------------------------
    @property
    def dr(self) -> float:
        """Current root-zone depletion [mm]."""
        return self._dr

    @property
    def taw(self) -> float:
        return self.soil.taw()

    @property
    def raw(self) -> float:
        return self.soil.raw()

    @property
    def ks(self) -> float:
        return float(stress_coefficient(self._dr, self.taw, self.soil.p))

    @classmethod
    def from_surface_sm(
        cls, soil: SoilProperties, swi_norm: float, **kw
    ) -> "RootZoneWaterBalance":
        """Build with the initial depletion derived from a normalised SWI ``[0,1]``.

        ``Dr0 = TAW * (1 - SWI_norm)`` — saturated root zone (SWI=1) -> Dr0=0.
        """
        swi_norm = float(np.clip(swi_norm, 0.0, 1.0))
        dr0 = soil.taw() * (1.0 - swi_norm)
        return cls(soil=soil, dr0=dr0, **kw)

    # -- stepping -----------------------------------------------------------------
    def step(
        self,
        et0: float,
        kc: float,
        rain: float = 0.0,
        irrigation: float = 0.0,
        capillary_rise: float = 0.0,
        runoff: Optional[float] = None,
        curve_number: Optional[float] = None,
        ke: float = 0.0,
        day: int = 0,
    ) -> WaterBalanceState:
        """Advance the balance by one day and return the day's state.

        Parameters
        ----------
        et0 : float
            Reference ET [mm/day].
        kc : float
            Crop coefficient (single Kc, or Kcb if ``ke`` is supplied separately).
        rain : float
            Gross precipitation [mm].
        irrigation : float
            Net irrigation applied [mm].
        capillary_rise : float
            Capillary rise into the root zone [mm] (usually 0).
        runoff : float, optional
            Surface runoff [mm]. If ``None`` and ``curve_number`` is given, runoff
            is derived with the SCS-CN model; otherwise runoff is 0.
        curve_number : float, optional
            SCS curve number for runoff partitioning of ``rain``.
        ke : float
            Soil-evaporation coefficient (dual-Kc); total Kc used is ``kc + ke``.
        day : int
            Day index (stored in the returned state).
        """
        taw = self.taw
        # effective rainfall (gross rain minus runoff)
        if runoff is None:
            if curve_number is not None:
                peff = float(effective_rainfall_scs_cn(rain, curve_number))
                ro = float(rain) - peff
            else:
                ro = 0.0
                peff = float(rain)
        else:
            ro = float(runoff)
            peff = float(rain) - ro

        # stress coefficient evaluated on the *start-of-day* depletion
        ks = float(stress_coefficient(self._dr, taw, self.soil.p))

        kc_total = float(kc) + float(ke)
        etc = kc_total * float(et0)
        etc_adj = ks * etc

        # FAO-56 Eq. 85: Dr_i = Dr_{i-1} - (P-RO) - I - CR + ETc_adj + DP
        dr_pre = self._dr - peff - float(irrigation) - float(capillary_rise) + etc_adj

        # deep percolation: any over-filling below field capacity (Dr < 0)
        deep_perc = float(max(-dr_pre, 0.0))
        dr_new = dr_pre + deep_perc          # -> >= 0 when over-filled
        # cannot deplete beyond wilting (TAW)
        dr_new = float(min(dr_new, taw))

        self._dr = dr_new
        return WaterBalanceState(
            day=day,
            dr=dr_new,
            ks=ks,
            etc=etc,
            etc_adj=etc_adj,
            peff=peff,
            irrigation=float(irrigation),
            deep_perc=deep_perc,
            runoff=ro,
        )

    def run(
        self,
        et0: Sequence[float] | np.ndarray,
        kc: Sequence[float] | np.ndarray | float,
        rain: Sequence[float] | np.ndarray | float = 0.0,
        irrigation: Sequence[float] | np.ndarray | float = 0.0,
        capillary_rise: Sequence[float] | np.ndarray | float = 0.0,
        curve_number: Optional[float] = None,
    ) -> list[WaterBalanceState]:
        """Run the balance over an ET0 series; returns a list of daily states."""
        et0 = np.atleast_1d(np.asarray(et0, dtype=float))
        n = et0.shape[0]
        kc = np.broadcast_to(np.asarray(kc, dtype=float), (n,))
        rain = np.broadcast_to(np.asarray(rain, dtype=float), (n,))
        irr = np.broadcast_to(np.asarray(irrigation, dtype=float), (n,))
        cr = np.broadcast_to(np.asarray(capillary_rise, dtype=float), (n,))

        out: list[WaterBalanceState] = []
        for i in range(n):
            out.append(
                self.step(
                    et0=float(et0[i]),
                    kc=float(kc[i]),
                    rain=float(rain[i]),
                    irrigation=float(irr[i]),
                    capillary_rise=float(cr[i]),
                    curve_number=curve_number,
                    day=i,
                )
            )
        return out


# --- 8-day deficit ---------------------------------------------------------------

@dataclass
class EightDayDeficit:
    """Result of an 8-day forward deficit accumulation."""

    dr: float        # depletion at the end of the window [mm]
    dnet: float      # net irrigation requirement = Dr [mm]
    dgross: float    # gross requirement = dnet / Ea [mm]
    ks_end: float    # stress coefficient at window end [-]
    states: list[WaterBalanceState]


def eight_day_deficit(
    soil: SoilProperties,
    et0: Sequence[float] | np.ndarray,
    kc: Sequence[float] | np.ndarray | float,
    rain: Sequence[float] | np.ndarray | float = 0.0,
    dr0: float = 0.0,
    application_efficiency: float = 0.6,
    irrigation: Sequence[float] | np.ndarray | float = 0.0,
    curve_number: Optional[float] = None,
    days: int = 8,
) -> EightDayDeficit:
    """Accumulate the root-zone depletion over an 8-day window.

    Runs :class:`RootZoneWaterBalance` for ``days`` (default 8 — the MODIS/ET 8-day
    compositing period) and reports the terminal depletion as the net irrigation
    requirement ``dnet`` and the gross requirement ``dgross = dnet / Ea``.

    Parameters
    ----------
    application_efficiency : float
        Irrigation application (field) efficiency ``Ea`` in ``(0, 1]``.
    """
    if not (0.0 < application_efficiency <= 1.0):
        raise ValueError("application_efficiency (Ea) must be in (0, 1].")

    et0 = np.atleast_1d(np.asarray(et0, dtype=float))
    if et0.shape[0] < days:
        # pad with the last value so short series still produce an 8-day estimate
        pad = np.full(days - et0.shape[0], et0[-1])
        et0 = np.concatenate([et0, pad])
    et0 = et0[:days]

    wb = RootZoneWaterBalance(soil=soil, dr0=dr0)
    states = wb.run(
        et0=et0,
        kc=kc,
        rain=rain,
        irrigation=irrigation,
        curve_number=curve_number,
    )
    dr = states[-1].dr
    dnet = dr
    dgross = dnet / application_efficiency
    return EightDayDeficit(
        dr=dr,
        dnet=dnet,
        dgross=dgross,
        ks_end=states[-1].ks,
        states=states,
    )
