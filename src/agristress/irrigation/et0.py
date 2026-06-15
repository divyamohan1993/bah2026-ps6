"""FAO-56 reference evapotranspiration (ET0).

Implements the FAO Penman-Monteith equation (Allen et al., 1998, FAO Irrigation &
Drainage Paper No. 56) together with all of its intermediate quantities, plus a
Hargreaves fallback for data-sparse situations.

All functions are pure ``numpy`` and accept either Python scalars or array-likes
(broadcasting follows ``numpy`` rules), so they can be applied pixel-wise over a
raster or row-wise over a meteorological time-series.

References
----------
Allen, R.G., Pereira, L.S., Raes, D., Smith, M. (1998). *Crop evapotranspiration -
Guidelines for computing crop water requirements*. FAO Irrigation and Drainage
Paper 56. Rome, FAO.  (Equations are cited inline as "Eq. N".)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "GSC",
    "STEFAN_BOLTZMANN",
    "ALBEDO",
    "saturation_vapour_pressure",
    "slope_saturation_vapour_pressure",
    "mean_saturation_vapour_pressure",
    "actual_vapour_pressure",
    "atmospheric_pressure",
    "psychrometric_constant",
    "inverse_relative_distance_earth_sun",
    "solar_declination",
    "sunset_hour_angle",
    "extraterrestrial_radiation",
    "clear_sky_radiation",
    "net_shortwave_radiation",
    "net_longwave_radiation",
    "net_radiation",
    "et0_penman_monteith",
    "et0_hargreaves",
    "ET0Components",
]

# --- physical constants (FAO-56) -------------------------------------------------

#: Solar constant [MJ m-2 min-1] (Eq. 28).
GSC: float = 0.0820
#: Stefan-Boltzmann constant [MJ K-4 m-2 day-1] (Eq. 39).
STEFAN_BOLTZMANN: float = 4.903e-9
#: Reference (grass) surface albedo [-] (Eq. 38).
ALBEDO: float = 0.23


# --- vapour pressure -------------------------------------------------------------

def saturation_vapour_pressure(t: np.ndarray | float) -> np.ndarray:
    r"""Saturation vapour pressure :math:`e^\circ(T)` at air temperature *t* [kPa].

    FAO-56 Eq. 11::

        e0(T) = 0.6108 * exp(17.27 * T / (T + 237.3))

    Parameters
    ----------
    t : float or array-like
        Air temperature [deg C].
    """
    t = np.asarray(t, dtype=float)
    return 0.6108 * np.exp((17.27 * t) / (t + 237.3))


def slope_saturation_vapour_pressure(t: np.ndarray | float) -> np.ndarray:
    r"""Slope of the saturation vapour pressure curve :math:`\Delta` [kPa/degC].

    FAO-56 Eq. 13. Evaluated at the mean daily air temperature.
    """
    t = np.asarray(t, dtype=float)
    es = 0.6108 * np.exp((17.27 * t) / (t + 237.3))
    return (4098.0 * es) / np.power(t + 237.3, 2)


def mean_saturation_vapour_pressure(tmin: float, tmax: float) -> np.ndarray:
    r"""Mean saturation vapour pressure :math:`e_s` [kPa] (FAO-56 Eq. 12).

    Computed as the mean of ``e0(Tmax)`` and ``e0(Tmin)`` — *not* ``e0(Tmean)`` —
    because of the non-linearity of ``e0(T)``.
    """
    return 0.5 * (saturation_vapour_pressure(tmax) + saturation_vapour_pressure(tmin))


def actual_vapour_pressure(
    tmin: float,
    tmax: float,
    rh_min: Optional[float] = None,
    rh_max: Optional[float] = None,
    rh_mean: Optional[float] = None,
    tdew: Optional[float] = None,
) -> np.ndarray:
    r"""Actual vapour pressure :math:`e_a` [kPa].

    Selection of formula follows FAO-56 §3.7 in order of data availability:

    * **Tdew** available -> ``ea = e0(Tdew)`` (Eq. 14).
    * **RHmin & RHmax** available -> Eq. 17
      ``ea = (e0(Tmin)*RHmax/100 + e0(Tmax)*RHmin/100) / 2``.
    * **RHmax** only -> Eq. 18 ``ea = e0(Tmin) * RHmax / 100``.
    * **RHmean** only -> Eq. 19 ``ea = RHmean/100 * (e0(Tmax)+e0(Tmin))/2``.
    """
    if tdew is not None:
        return saturation_vapour_pressure(tdew)

    e_tmin = saturation_vapour_pressure(tmin)
    e_tmax = saturation_vapour_pressure(tmax)

    if rh_min is not None and rh_max is not None:
        rh_min = np.asarray(rh_min, dtype=float)
        rh_max = np.asarray(rh_max, dtype=float)
        return 0.5 * (e_tmin * rh_max / 100.0 + e_tmax * rh_min / 100.0)
    if rh_max is not None:
        rh_max = np.asarray(rh_max, dtype=float)
        return e_tmin * rh_max / 100.0
    if rh_mean is not None:
        rh_mean = np.asarray(rh_mean, dtype=float)
        return (rh_mean / 100.0) * 0.5 * (e_tmax + e_tmin)

    raise ValueError(
        "actual_vapour_pressure requires one of: tdew, (rh_min & rh_max), "
        "rh_max, or rh_mean."
    )


# --- pressure / psychrometric ----------------------------------------------------

def atmospheric_pressure(elevation: float) -> np.ndarray:
    r"""Atmospheric pressure *P* [kPa] from elevation *z* [m] (FAO-56 Eq. 7)."""
    z = np.asarray(elevation, dtype=float)
    return 101.3 * np.power((293.0 - 0.0065 * z) / 293.0, 5.26)


def psychrometric_constant(pressure: float) -> np.ndarray:
    r"""Psychrometric constant :math:`\gamma` [kPa/degC] (FAO-56 Eq. 8).

    ``gamma = 0.000665 * P`` (coefficient bundles cp, ratio of molecular weights and
    latent heat of vaporisation).
    """
    return 0.000665 * np.asarray(pressure, dtype=float)


# --- radiation -------------------------------------------------------------------

def inverse_relative_distance_earth_sun(doy: int) -> np.ndarray:
    r"""Inverse relative Earth-Sun distance :math:`d_r` [-] (FAO-56 Eq. 23)."""
    doy = np.asarray(doy, dtype=float)
    return 1.0 + 0.033 * np.cos(2.0 * np.pi * doy / 365.0)


def solar_declination(doy: int) -> np.ndarray:
    r"""Solar declination :math:`\delta` [rad] (FAO-56 Eq. 24)."""
    doy = np.asarray(doy, dtype=float)
    return 0.409 * np.sin(2.0 * np.pi * doy / 365.0 - 1.39)


def sunset_hour_angle(lat_rad: np.ndarray, decl: np.ndarray) -> np.ndarray:
    r"""Sunset hour angle :math:`\omega_s` [rad] (FAO-56 Eq. 25).

    ``omega_s = arccos(-tan(phi) * tan(delta))`` with the argument clipped to
    ``[-1, 1]`` so polar day/night do not produce ``NaN``.
    """
    x = -np.tan(lat_rad) * np.tan(decl)
    x = np.clip(x, -1.0, 1.0)
    return np.arccos(x)


def extraterrestrial_radiation(lat: float, doy: int) -> np.ndarray:
    r"""Extraterrestrial radiation :math:`R_a` [MJ m-2 day-1] (FAO-56 Eq. 21).

    Parameters
    ----------
    lat : float
        Latitude [decimal degrees], positive north.
    doy : int
        Day of year (1-365/366).
    """
    phi = np.deg2rad(np.asarray(lat, dtype=float))
    dr = inverse_relative_distance_earth_sun(doy)
    decl = solar_declination(doy)
    ws = sunset_hour_angle(phi, decl)
    ra = (
        (24.0 * 60.0 / np.pi)
        * GSC
        * dr
        * (
            ws * np.sin(phi) * np.sin(decl)
            + np.cos(phi) * np.cos(decl) * np.sin(ws)
        )
    )
    return ra


def clear_sky_radiation(ra: np.ndarray, elevation: float) -> np.ndarray:
    r"""Clear-sky solar radiation :math:`R_{so}` [MJ m-2 day-1] (FAO-56 Eq. 37)."""
    z = np.asarray(elevation, dtype=float)
    return (0.75 + 2e-5 * z) * np.asarray(ra, dtype=float)


def net_shortwave_radiation(rs: np.ndarray, albedo: float = ALBEDO) -> np.ndarray:
    r"""Net shortwave radiation :math:`R_{ns}` [MJ m-2 day-1] (FAO-56 Eq. 38).

    ``Rns = (1 - albedo) * Rs`` -> ``0.77 * Rs`` for the reference grass surface.
    """
    return (1.0 - albedo) * np.asarray(rs, dtype=float)


def net_longwave_radiation(
    tmin: float,
    tmax: float,
    ea: np.ndarray,
    rs: np.ndarray,
    rso: np.ndarray,
) -> np.ndarray:
    r"""Net longwave radiation :math:`R_{nl}` [MJ m-2 day-1] (FAO-56 Eq. 39).

    Stefan-Boltzmann law using the *mean of the 4th powers* of absolute Tmax/Tmin.
    The cloudiness factor uses the ratio ``Rs/Rso`` clipped to ``[0.3, 1.0]``.
    """
    tmaxk = np.asarray(tmax, dtype=float) + 273.16
    tmink = np.asarray(tmin, dtype=float) + 273.16
    ea = np.asarray(ea, dtype=float)
    rs = np.asarray(rs, dtype=float)
    rso = np.asarray(rso, dtype=float)

    # cloudiness factor; guard against Rso == 0 (polar night) and clip per FAO-56.
    with np.errstate(divide="ignore", invalid="ignore"):
        rs_rso = np.where(rso > 0.0, rs / rso, 1.0)
    rs_rso = np.clip(rs_rso, 0.3, 1.0)

    return (
        STEFAN_BOLTZMANN
        * 0.5
        * (np.power(tmaxk, 4) + np.power(tmink, 4))
        * (0.34 - 0.14 * np.sqrt(ea))
        * (1.35 * rs_rso - 0.35)
    )


def net_radiation(rns: np.ndarray, rnl: np.ndarray) -> np.ndarray:
    r"""Net radiation :math:`R_n = R_{ns} - R_{nl}` [MJ m-2 day-1] (FAO-56 Eq. 40)."""
    return np.asarray(rns, dtype=float) - np.asarray(rnl, dtype=float)


# --- ET0 -------------------------------------------------------------------------

@dataclass
class ET0Components:
    """Container holding ET0 and every intermediate (useful for QA / debugging)."""

    et0: float
    delta: float
    gamma: float
    pressure: float
    es: float
    ea: float
    ra: float
    rso: float
    rns: float
    rnl: float
    rn: float
    tmean: float


def et0_penman_monteith(
    tmin: float,
    tmax: float,
    rh_min: Optional[float] = None,
    rh_max: Optional[float] = None,
    u2: float = 2.0,
    rs: Optional[float] = None,
    elevation: float = 0.0,
    lat: float = 0.0,
    doy: int = 1,
    *,
    rh_mean: Optional[float] = None,
    tdew: Optional[float] = None,
    g: float = 0.0,
    tmean: Optional[float] = None,
    return_components: bool = False,
):
    r"""FAO-56 Penman-Monteith reference evapotranspiration ET0 [mm/day].

    FAO-56 Eq. 6::

        ET0 = (0.408 * Delta * (Rn - G) + gamma * 900/(T+273) * u2 * (es - ea))
              / (Delta + gamma * (1 + 0.34 * u2))

    Parameters
    ----------
    tmin, tmax : float
        Daily minimum / maximum air temperature [deg C].
    rh_min, rh_max : float, optional
        Daily minimum / maximum relative humidity [%].
    u2 : float
        Wind speed at 2 m [m/s]. Defaults to 2.0 m/s (FAO-56 default when missing).
    rs : float, optional
        Incoming solar (shortwave) radiation [MJ m-2 day-1]. **Required.**
    elevation : float
        Station elevation above sea level [m].
    lat : float
        Latitude [decimal degrees], positive north.
    doy : int
        Day of year (used for Ra).
    rh_mean, tdew : float, optional
        Alternative humidity inputs (see :func:`actual_vapour_pressure`).
    g : float
        Soil heat flux density [MJ m-2 day-1]; ~0 for daily steps.
    tmean : float, optional
        Mean air temperature [deg C]. Defaults to ``(tmin + tmax) / 2``.
    return_components : bool
        If True, return an :class:`ET0Components` instead of a bare ET0 value.

    Returns
    -------
    float or ET0Components
        ET0 in mm/day (>= 0), or the full component breakdown.
    """
    if rs is None:
        raise ValueError(
            "et0_penman_monteith requires solar radiation `rs` [MJ m-2 day-1]. "
            "Use et0_hargreaves(tmin, tmax, ra) if only temperature is available."
        )

    tmin = np.asarray(tmin, dtype=float)
    tmax = np.asarray(tmax, dtype=float)
    if tmean is None:
        t = (tmin + tmax) / 2.0
    else:
        t = np.asarray(tmean, dtype=float)

    # vapour pressure terms
    es = mean_saturation_vapour_pressure(tmin, tmax)
    ea = actual_vapour_pressure(
        tmin, tmax, rh_min=rh_min, rh_max=rh_max, rh_mean=rh_mean, tdew=tdew
    )

    # slope & psychrometric constant
    delta = slope_saturation_vapour_pressure(t)
    pressure = atmospheric_pressure(elevation)
    gamma = psychrometric_constant(pressure)

    # radiation chain
    ra = extraterrestrial_radiation(lat, doy)
    rso = clear_sky_radiation(ra, elevation)
    rns = net_shortwave_radiation(rs)
    rnl = net_longwave_radiation(tmin, tmax, ea, rs, rso)
    rn = net_radiation(rns, rnl)

    u2 = np.asarray(u2, dtype=float)
    g = np.asarray(g, dtype=float)

    numerator = 0.408 * delta * (rn - g) + gamma * (900.0 / (t + 273.0)) * u2 * (es - ea)
    denominator = delta + gamma * (1.0 + 0.34 * u2)
    et0 = numerator / denominator
    et0 = np.maximum(et0, 0.0)

    if return_components:
        return ET0Components(
            et0=float(et0),
            delta=float(delta),
            gamma=float(gamma),
            pressure=float(pressure),
            es=float(es),
            ea=float(ea),
            ra=float(ra),
            rso=float(rso),
            rns=float(rns),
            rnl=float(rnl),
            rn=float(rn),
            tmean=float(t),
        )
    return et0 if et0.ndim else float(et0)


def et0_hargreaves(tmin: float, tmax: float, ra: float) -> np.ndarray:
    r"""Hargreaves-Samani ET0 [mm/day] — temperature-only fallback (FAO-56 Eq. 52).

    ::

        ET0 = 0.0023 * (Tmean + 17.8) * sqrt(Tmax - Tmin) * Ra * 0.408

    where ``Ra`` is extraterrestrial radiation [MJ m-2 day-1] and the ``0.408``
    factor converts MJ m-2 day-1 to mm/day equivalent evaporation.
    """
    tmin = np.asarray(tmin, dtype=float)
    tmax = np.asarray(tmax, dtype=float)
    ra = np.asarray(ra, dtype=float)
    tmean = (tmin + tmax) / 2.0
    tr = np.clip(tmax - tmin, 0.0, None)
    et0 = 0.0023 * (tmean + 17.8) * np.sqrt(tr) * ra * 0.408
    et0 = np.maximum(et0, 0.0)
    return et0 if et0.ndim else float(et0)
