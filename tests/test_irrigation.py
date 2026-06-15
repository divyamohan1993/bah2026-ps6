"""Tests for the AgriStress irrigation-advisory engine.

Validated against FAO-56 (Allen et al., 1998) worked examples and the
documented water-balance identities (TAW/RAW/Ks, 8-day deficit, advisory).
"""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np
import pytest

from agristress.irrigation.advisory import (
    FieldAdvisory,
    IrrigationAdvisory,
    IrrigationStatus,
    advisory_map,
    aggregate_to_command,
)
from agristress.irrigation.et0 import (
    actual_vapour_pressure,
    atmospheric_pressure,
    et0_hargreaves,
    et0_penman_monteith,
    extraterrestrial_radiation,
    psychrometric_constant,
    saturation_vapour_pressure,
    slope_saturation_vapour_pressure,
)
from agristress.irrigation.kc import (
    adjust_kc_mid_for_climate,
    available_crops,
    get_crop,
    growth_stage_for_das,
    kc_curve,
    kc_for_stage,
    kc_from_ndvi,
)
from agristress.irrigation.water_balance import (
    RootZoneWaterBalance,
    SoilProperties,
    effective_rainfall_scs_cn,
    effective_rainfall_usda_monthly,
    eight_day_deficit,
    raw_from_taw,
    stress_coefficient,
    swi_from_surface_sm,
    taw_from_soil,
)

# =============================================================================
# ET0 — FAO-56 Penman-Monteith
# =============================================================================


class TestET0PenmanMonteith:
    """FAO-56 Example 18 (Brussels, 6 July): ET0 = 3.88 mm/day."""

    # canonical FAO-56 Example 18 inputs
    INPUTS: ClassVar[dict[str, float]] = {
        "tmin": 12.3,
        "tmax": 21.5,
        "rh_min": 63,
        "rh_max": 84,
        "u2": 2.078,
        "rs": 22.07,
        "elevation": 100,
        "lat": 50.80,
        "doy": 187,
    }

    def test_et0_matches_fao56_example18(self):
        et0 = et0_penman_monteith(**self.INPUTS)
        # FAO-56 reports 3.88 mm/day; require within +/- 0.2 mm/d
        assert et0 == pytest.approx(3.88, abs=0.2)

    def test_components_match_fao56_example18(self):
        c = et0_penman_monteith(**self.INPUTS, return_components=True)
        assert c.delta == pytest.approx(0.122, abs=0.005)  # slope
        assert c.gamma == pytest.approx(0.0666, abs=0.001)  # psychrometric
        assert c.ea == pytest.approx(1.409, abs=0.02)  # actual vp
        assert c.es == pytest.approx(1.997, abs=0.02)  # sat vp
        assert c.ra == pytest.approx(41.09, abs=0.3)  # extraterrestrial
        assert c.rso == pytest.approx(30.90, abs=0.3)  # clear-sky
        assert c.rns == pytest.approx(17.0, abs=0.2)  # net shortwave
        assert c.rnl == pytest.approx(3.71, abs=0.2)  # net longwave
        assert c.rn == pytest.approx(13.28, abs=0.3)  # net radiation

    def test_saturation_vapour_pressure_table(self):
        # FAO-56 Eq. 11 reference points: e0(T) = 0.6108*exp(17.27*T/(T+237.3))
        # e0(12.3)=1.431, e0(21.5)=2.564 (used directly in Example 18).
        assert saturation_vapour_pressure(1.5) == pytest.approx(0.681, abs=0.01)
        assert saturation_vapour_pressure(12.3) == pytest.approx(1.431, abs=0.01)
        assert saturation_vapour_pressure(21.5) == pytest.approx(2.564, abs=0.01)
        assert saturation_vapour_pressure(24.0) == pytest.approx(2.984, abs=0.01)

    def test_atmospheric_pressure_and_psychrometric(self):
        # at z=100 m, P ~= 100.1 kPa, gamma ~= 0.0666
        p = atmospheric_pressure(100.0)
        assert p == pytest.approx(100.1, abs=0.3)
        assert psychrometric_constant(p) == pytest.approx(0.0666, abs=0.001)
        # sea level P = 101.3 kPa
        assert atmospheric_pressure(0.0) == pytest.approx(101.3, abs=0.01)

    def test_slope_at_known_temperature(self):
        # FAO-56 Table 2.4: Delta at 16.9 deg C ~= 0.122 kPa/degC
        tmean = 0.5 * (12.3 + 21.5)
        assert slope_saturation_vapour_pressure(tmean) == pytest.approx(0.122, abs=0.005)

    def test_actual_vapour_pressure_from_rh(self):
        ea = actual_vapour_pressure(tmin=12.3, tmax=21.5, rh_min=63, rh_max=84)
        assert ea == pytest.approx(1.409, abs=0.02)

    def test_actual_vapour_pressure_from_tdew(self):
        # ea = e0(Tdew)
        assert actual_vapour_pressure(10, 25, tdew=12.0) == pytest.approx(
            saturation_vapour_pressure(12.0)
        )

    def test_actual_vapour_pressure_requires_humidity(self):
        with pytest.raises(ValueError):
            actual_vapour_pressure(10, 25)

    def test_requires_solar_radiation(self):
        with pytest.raises(ValueError):
            et0_penman_monteith(
                tmin=12, tmax=22, rh_min=60, rh_max=80, u2=2, elevation=100, lat=50, doy=180
            )

    def test_et0_non_negative(self):
        # cold, humid, low radiation -> small but non-negative ET0
        et0 = et0_penman_monteith(
            tmin=-5, tmax=0, rh_min=90, rh_max=100, u2=1, rs=1.0, elevation=0, lat=60, doy=350
        )
        assert et0 >= 0.0

    def test_vectorised_inputs(self):
        et0 = et0_penman_monteith(
            tmin=np.array([12.3, 12.3]),
            tmax=np.array([21.5, 21.5]),
            rh_min=np.array([63, 63]),
            rh_max=np.array([84, 84]),
            u2=2.078,
            rs=np.array([22.07, 22.07]),
            elevation=100,
            lat=50.80,
            doy=187,
        )
        assert np.shape(et0) == (2,)
        assert np.allclose(et0, 3.88, atol=0.2)


class TestExtraterrestrialRadiation:
    def test_ra_example18(self):
        # FAO-56 Example 18: lat 50.80N, doy 187 -> Ra ~= 41.09 MJ/m2/day
        ra = extraterrestrial_radiation(lat=50.80, doy=187)
        assert ra == pytest.approx(41.09, abs=0.3)

    def test_ra_positive_and_seasonal(self):
        ra_summer = extraterrestrial_radiation(lat=30.0, doy=172)  # ~summer solstice
        ra_winter = extraterrestrial_radiation(lat=30.0, doy=355)  # ~winter solstice
        assert ra_summer > ra_winter > 0


class TestHargreaves:
    def test_hargreaves_reasonable(self):
        ra = extraterrestrial_radiation(lat=50.80, doy=187)
        et0 = et0_hargreaves(12.3, 21.5, ra)
        # temperature-only estimate should be in the same ballpark as PM (3.88)
        assert 3.0 < et0 < 5.0

    def test_hargreaves_zero_temp_range(self):
        ra = extraterrestrial_radiation(lat=30.0, doy=180)
        assert et0_hargreaves(20.0, 20.0, ra) == pytest.approx(0.0)


# =============================================================================
# Kc curves
# =============================================================================


class TestKc:
    def test_all_required_crops_present(self):
        crops = set(available_crops())
        assert {"rice", "wheat", "maize", "cotton", "sugarcane", "pulses"} <= crops

    @pytest.mark.parametrize(
        "crop,kc_ini,kc_mid,kc_end",
        [
            ("rice", 1.05, 1.20, 0.90),
            ("wheat", 0.30, 1.15, 0.30),
            ("maize", 0.30, 1.20, 0.50),
            ("cotton", 0.35, 1.18, 0.60),
            ("sugarcane", 0.40, 1.25, 0.75),
            ("pulses", 0.40, 1.15, 0.35),
        ],
    )
    def test_kc_table_values(self, crop, kc_ini, kc_mid, kc_end):
        c = get_crop(crop)
        assert c.kc_ini == pytest.approx(kc_ini)
        assert c.kc_mid == pytest.approx(kc_mid)
        assert c.kc_end == pytest.approx(kc_end)
        assert c.zr_m > 0
        assert 0 < c.depletion_p < 1

    def test_kc_for_stage(self):
        assert kc_for_stage("wheat", "ini") == pytest.approx(0.30)
        assert kc_for_stage("wheat", "mid") == pytest.approx(1.15)
        assert kc_for_stage("wheat", "late") == pytest.approx(0.30)
        # dev is the mean of ini and mid
        assert kc_for_stage("wheat", "dev") == pytest.approx(0.5 * (0.30 + 1.15))

    def test_kc_curve_piecewise(self):
        # wheat lengths [30, 40, 40, 30]; boundaries (30, 70, 110, 140)
        assert kc_curve("wheat", 0) == pytest.approx(0.30)  # ini
        assert kc_curve("wheat", 30) == pytest.approx(0.30)  # end of ini
        # mid-development (50 d -> 20/40 through dev): 0.30 + 0.5*(1.15-0.30)
        assert kc_curve("wheat", 50) == pytest.approx(0.30 + 0.5 * 0.85)
        assert kc_curve("wheat", 90) == pytest.approx(1.15)  # mid season
        assert kc_curve("wheat", 140) == pytest.approx(0.30)  # harvest
        # beyond season clamps to Kc_end
        assert kc_curve("wheat", 200) == pytest.approx(0.30)

    def test_kc_curve_monotonic_in_development(self):
        das = np.arange(30, 71)
        kc = kc_curve("wheat", das)
        assert np.all(np.diff(kc) >= -1e-9)  # non-decreasing through dev
        assert kc[0] == pytest.approx(0.30)
        assert kc[-1] == pytest.approx(1.15)

    def test_kc_curve_vectorised(self):
        kc = kc_curve("maize", np.array([0, 30, 100, 150]))
        assert kc.shape == (4,)

    def test_growth_stage_for_das(self):
        assert growth_stage_for_das("wheat", 10) == "ini"
        assert growth_stage_for_das("wheat", 50) == "dev"
        assert growth_stage_for_das("wheat", 90) == "mid"
        assert growth_stage_for_das("wheat", 130) == "late"

    def test_kc_from_ndvi(self):
        # closed canopy NDVI ~0.86 -> Kc near mid-season (~1.1)
        assert kc_from_ndvi(0.86) == pytest.approx(1.08, abs=0.1)
        # bare soil NDVI ~0.12 -> Kc ~0
        assert kc_from_ndvi(0.12) == pytest.approx(0.0, abs=0.05)
        # clipping
        assert kc_from_ndvi(-0.2) == 0.0
        assert kc_from_ndvi(2.0) <= 1.3

    def test_kc_from_ndvi_vectorised(self):
        kc = kc_from_ndvi(np.array([0.1, 0.5, 0.9]))
        assert kc.shape == (3,)
        assert np.all(np.diff(kc) > 0)

    def test_climate_adjustment_increases_in_arid_windy(self):
        base = 1.15
        adj = adjust_kc_mid_for_climate(base, u2=4.0, rh_min=20.0, plant_height_m=2.0)
        assert adj > base
        # reference conditions (u2=2, RHmin=45) leave Kc unchanged
        assert adjust_kc_mid_for_climate(base, u2=2.0, rh_min=45.0) == pytest.approx(base)

    def test_unknown_crop_raises(self):
        with pytest.raises(KeyError):
            get_crop("banana")


# =============================================================================
# Water balance — TAW / RAW / Ks
# =============================================================================


class TestSoilWaterCapacity:
    def test_taw_formula(self):
        # 1000*(0.22-0.10)*1.0 = 120 mm
        soil = SoilProperties(theta_fc=0.22, theta_wp=0.10, zr=1.0)
        assert soil.taw() == pytest.approx(120.0)
        assert taw_from_soil(0.22, 0.10, 1.0) == pytest.approx(120.0)

    def test_raw_formula(self):
        soil = SoilProperties(theta_fc=0.22, theta_wp=0.10, zr=1.0, p=0.5)
        assert soil.raw() == pytest.approx(60.0)
        assert soil.raw(p=0.4) == pytest.approx(48.0)
        assert raw_from_taw(120.0, 0.5) == pytest.approx(60.0)


class TestStressCoefficient:
    """FAO-56 Eq. 84: Ks = (TAW - Dr) / ((1 - p) * TAW)."""

    def test_ks_half_at_documented_point(self):
        # TAW=120, p=0.5 -> RAW=60. Dr=90 -> Ks=(120-90)/((1-0.5)*120)=30/60=0.5
        assert stress_coefficient(90.0, 120.0, 0.5) == pytest.approx(0.5)

    def test_ks_one_within_raw(self):
        # Dr <= RAW -> no stress
        assert stress_coefficient(60.0, 120.0, 0.5) == pytest.approx(1.0)
        assert stress_coefficient(30.0, 120.0, 0.5) == pytest.approx(1.0)
        assert stress_coefficient(0.0, 120.0, 0.5) == pytest.approx(1.0)

    def test_ks_zero_at_taw(self):
        assert stress_coefficient(120.0, 120.0, 0.5) == pytest.approx(0.0)

    def test_ks_clamped(self):
        # Dr beyond TAW still clamps to 0
        assert stress_coefficient(150.0, 120.0, 0.5) == pytest.approx(0.0)

    def test_ks_vectorised(self):
        dr = np.array([0.0, 60.0, 90.0, 120.0])
        ks = stress_coefficient(dr, 120.0, 0.5)
        assert np.allclose(ks, [1.0, 1.0, 0.5, 0.0])


class TestEffectiveRainfall:
    def test_usda_monthly_low(self):
        # P=100 -> 100*(125-20)/125 = 84 mm
        assert effective_rainfall_usda_monthly(100.0) == pytest.approx(84.0)

    def test_usda_monthly_high(self):
        # P=300 -> 125 + 0.1*300 = 155 mm
        assert effective_rainfall_usda_monthly(300.0) == pytest.approx(155.0)

    def test_scs_cn_small_event_all_infiltrates(self):
        # tiny rain below initial abstraction -> no runoff -> peff == P
        assert effective_rainfall_scs_cn(2.0, curve_number=75) == pytest.approx(2.0)

    def test_scs_cn_runoff_reduces_effective(self):
        peff = effective_rainfall_scs_cn(80.0, curve_number=85)
        assert 0.0 < peff < 80.0


class TestRootZoneWaterBalance:
    def make_soil(self):
        return SoilProperties(theta_fc=0.22, theta_wp=0.10, zr=1.0, p=0.5)

    def test_depletion_increases_with_et(self):
        wb = RootZoneWaterBalance(soil=self.make_soil(), dr0=0.0)
        st = wb.step(et0=6.0, kc=1.15)  # ETc = 6.9 mm
        assert st.dr == pytest.approx(6.9, abs=1e-6)
        assert st.etc == pytest.approx(6.9)
        assert st.etc_adj == pytest.approx(6.9)  # no stress at FC

    def test_rain_reduces_depletion(self):
        wb = RootZoneWaterBalance(soil=self.make_soil(), dr0=50.0)
        st = wb.step(et0=0.0, kc=1.0, rain=20.0)
        assert st.dr == pytest.approx(30.0)

    def test_depletion_bounded_at_field_capacity(self):
        # over-filling beyond FC -> Dr clamps to 0 and produces deep percolation
        wb = RootZoneWaterBalance(soil=self.make_soil(), dr0=10.0)
        st = wb.step(et0=0.0, kc=1.0, rain=50.0)
        assert st.dr == pytest.approx(0.0)
        assert st.deep_perc == pytest.approx(40.0)

    def test_depletion_bounded_at_taw(self):
        wb = RootZoneWaterBalance(soil=self.make_soil(), dr0=110.0)
        st = wb.step(et0=20.0, kc=1.2)
        assert st.dr <= self.make_soil().taw() + 1e-9

    def test_mass_conservation(self):
        # closed balance: Dr_end == Dr_0 - sum(P_eff) - sum(I) + sum(ETc_adj) + sum(DP)
        soil = self.make_soil()
        wb = RootZoneWaterBalance(soil=soil, dr0=10.0)
        rng = np.random.default_rng(0)
        et0 = rng.uniform(2, 8, size=20)
        rain = rng.uniform(0, 15, size=20)
        irr = rng.uniform(0, 5, size=20)
        states = wb.run(et0=et0, kc=1.0, rain=rain, irrigation=irr)
        dr_end = states[-1].dr
        sum_peff = sum(s.peff for s in states)
        sum_irr = sum(s.irrigation for s in states)
        sum_etc = sum(s.etc_adj for s in states)
        sum_dp = sum(s.deep_perc for s in states)
        expected = 10.0 - sum_peff - sum_irr + sum_etc + sum_dp
        assert dr_end == pytest.approx(expected, abs=1e-6)

    def test_ks_reduces_et_when_stressed(self):
        soil = self.make_soil()
        wb = RootZoneWaterBalance(soil=soil, dr0=90.0)  # Dr>RAW -> Ks=0.5
        st = wb.step(et0=6.0, kc=1.0)
        assert st.ks == pytest.approx(0.5, abs=1e-6)
        assert st.etc_adj == pytest.approx(0.5 * 6.0, abs=1e-6)

    def test_from_surface_sm_initialisation(self):
        soil = self.make_soil()
        # SWI=1 (saturated root zone) -> Dr0 = 0
        wb_full = RootZoneWaterBalance.from_surface_sm(soil, swi_norm=1.0)
        assert wb_full.dr == pytest.approx(0.0)
        # SWI=0 (dry) -> Dr0 = TAW
        wb_dry = RootZoneWaterBalance.from_surface_sm(soil, swi_norm=0.0)
        assert wb_dry.dr == pytest.approx(soil.taw())


class TestSwiFilter:
    def test_cold_start_returns_ssm(self):
        assert swi_from_surface_sm(0.4) == pytest.approx(0.4)

    def test_filter_moves_towards_observation(self):
        swi = swi_from_surface_sm(0.6, time=1.0, prev_swi=0.2, prev_time=0.0, t_char=5.0)
        assert 0.2 < swi < 0.6


# =============================================================================
# 8-day deficit — FAO-56 worked example
# =============================================================================


class TestEightDayDeficit:
    """Worked example: wheat mid-season, sandy-loam, TAW=120, p=0.5 -> RAW=60,
    ET0=6, Kc=1.15, Ea=0.6.  Net irrigation depth at the RAW threshold = 60 mm,
    gross = 60/0.6 = 100 mm."""

    def make_soil(self):
        # 1000*(0.22-0.10)*1.0 = 120 mm -> TAW=120, RAW(p=0.5)=60
        return SoilProperties(theta_fc=0.22, theta_wp=0.10, zr=1.0, p=0.5)

    def test_worked_example_dnet_dgross_at_threshold(self):
        soil = self.make_soil()
        # irrigation triggered at RAW: dnet = RAW = 60, dgross = 60/0.6 = 100
        dnet = soil.raw()
        dgross = dnet / 0.6
        assert dnet == pytest.approx(60.0)
        assert dgross == pytest.approx(100.0)

    def test_eight_day_accumulation(self):
        soil = self.make_soil()
        res = eight_day_deficit(
            soil,
            et0=[6.0] * 8,
            kc=1.15,
            application_efficiency=0.6,
        )
        # 8 days * Kc*ET0 (6.9) = 55.2 mm depletion (no stress, Dr<RAW throughout)
        assert res.dr == pytest.approx(55.2, abs=1e-6)
        assert res.dnet == pytest.approx(res.dr)
        assert res.dgross == pytest.approx(res.dnet / 0.6)

    def test_dgross_is_dnet_over_ea(self):
        soil = self.make_soil()
        res = eight_day_deficit(soil, et0=[6.0] * 8, kc=1.15, application_efficiency=0.6)
        assert res.dgross == pytest.approx(res.dnet / 0.6)

    def test_short_series_padded(self):
        soil = self.make_soil()
        res = eight_day_deficit(soil, et0=[6.0] * 3, kc=1.15)
        assert len(res.states) == 8

    def test_invalid_efficiency_raises(self):
        soil = self.make_soil()
        with pytest.raises(ValueError):
            eight_day_deficit(soil, et0=[6.0] * 8, kc=1.15, application_efficiency=0.0)


# =============================================================================
# Advisory
# =============================================================================


class TestIrrigationAdvisory:
    def test_no_irrigation_at_field_capacity(self):
        adv = IrrigationAdvisory()
        a = adv.evaluate(dr=0.0, taw=120.0, p=0.5, growth_stage="mid", etc=6.9)
        assert a.status == IrrigationStatus.NO_IRRIGATION

    def test_critical_as_dr_approaches_taw(self):
        adv = IrrigationAdvisory()
        a = adv.evaluate(dr=119.0, taw=120.0, p=0.5, growth_stage="mid", etc=6.9)
        assert a.status == IrrigationStatus.CRITICAL
        assert a.ks < 0.4

    def test_worked_example_irrigate_now_dnet_dgross(self):
        # at the RAW trigger (Dr=60, TAW=120, p=0.5, dev stage keeps p_eff=0.5):
        # status=IRRIGATE_NOW, dnet=60, dgross=100
        adv = IrrigationAdvisory(application_efficiency=0.6)
        a = adv.evaluate(dr=60.0, taw=120.0, p=0.5, growth_stage="dev", etc=6.9)
        assert a.status == IrrigationStatus.IRRIGATE_NOW
        assert a.raw == pytest.approx(60.0)
        assert a.dnet == pytest.approx(60.0)
        assert a.dgross == pytest.approx(100.0)

    def test_status_progression(self):
        adv = IrrigationAdvisory()
        # increasing depletion -> non-decreasing urgency (dev stage, p_eff=0.5)
        statuses = [
            adv.evaluate(dr=d, taw=120.0, p=0.5, growth_stage="dev", etc=6.9).status
            for d in [0, 30, 50, 65, 119]
        ]
        codes = [int(s) for s in statuses]
        assert codes == sorted(codes)
        assert codes[0] == IrrigationStatus.NO_IRRIGATION
        assert codes[-1] == IrrigationStatus.CRITICAL

    def test_flowering_tightens_threshold(self):
        adv = IrrigationAdvisory()
        # same Dr; flowering (mid) uses smaller effective p -> smaller RAW -> earlier trigger
        a_mid = adv.evaluate(dr=52.0, taw=120.0, p=0.5, growth_stage="flowering", etc=6.9)
        a_dev = adv.evaluate(dr=52.0, taw=120.0, p=0.5, growth_stage="dev", etc=6.9)
        assert int(a_mid.status) >= int(a_dev.status)
        assert a_mid.raw < a_dev.raw

    def test_days_to_trigger(self):
        adv = IrrigationAdvisory()
        a = adv.evaluate(dr=0.0, taw=120.0, p=0.5, growth_stage="dev", etc=6.0)
        # RAW=60, etc=6 -> ~10 days to reach RAW
        assert a.days_to_trigger == pytest.approx(10.0, abs=1e-6)

    def test_rice_ponding_branch(self):
        adv = IrrigationAdvisory()
        soil_taw = 100.0
        # well ponded -> hold
        a0 = adv.evaluate(dr=0.0, taw=soil_taw, p=0.2, growth_stage="mid", ponded=True)
        assert a0.status == IrrigationStatus.NO_IRRIGATION
        # AWD threshold reached -> re-flood now
        raw = 0.2 * soil_taw
        a1 = adv.evaluate(dr=raw, taw=soil_taw, p=0.2, growth_stage="mid", ponded=True)
        assert a1.status == IrrigationStatus.IRRIGATE_NOW
        assert "re-flood" in a1.recommendation.lower() or "reflood" in a1.recommendation.lower()
        # deep dry-down -> critical
        a2 = adv.evaluate(dr=0.95 * soil_taw, taw=soil_taw, p=0.2, growth_stage="mid", ponded=True)
        assert a2.status == IrrigationStatus.CRITICAL

    def test_recommendation_is_string(self):
        adv = IrrigationAdvisory()
        a = adv.evaluate(dr=70.0, taw=120.0, p=0.5, growth_stage="mid", etc=6.9)
        assert isinstance(a.recommendation, str) and len(a.recommendation) > 0


class TestAdvisoryMap:
    def test_class_raster_gradient(self):
        dr = np.array([0.0, 30.0, 50.0, 65.0, 115.0])
        raw = np.full(5, 60.0)
        taw = np.full(5, 120.0)
        ks = np.array([1.0, 1.0, 1.0, 0.9, 0.05])
        out = advisory_map(dr, raw, ks, taw_array=taw)
        assert out.dtype == np.int8
        assert out.tolist() == [0, 1, 2, 3, 4]

    def test_critical_from_ks(self):
        dr = np.array([10.0])
        raw = np.array([60.0])
        ks = np.array([0.2])
        out = advisory_map(dr, raw, ks)
        assert out[0] == IrrigationStatus.CRITICAL

    def test_nodata_handling(self):
        dr = np.array([np.nan, 30.0])
        raw = np.full(2, 60.0)
        out = advisory_map(dr, raw)
        assert out[0] == -1
        assert out[1] == IrrigationStatus.WATCH

    def test_2d_raster(self):
        dr = np.array([[0.0, 65.0], [50.0, 115.0]])
        raw = np.full((2, 2), 60.0)
        taw = np.full((2, 2), 120.0)
        ks = np.array([[1.0, 0.9], [1.0, 0.05]])
        out = advisory_map(dr, raw, ks, taw_array=taw)
        assert out.shape == (2, 2)
        assert out.tolist() == [[0, 3], [2, 4]]


class TestCommandAggregation:
    def _due_field(self, fid, area, dist=0.0, tail=False):
        adv = IrrigationAdvisory(application_efficiency=0.6)
        # Dr=RAW -> IRRIGATE_NOW, dgross = (Dr/Ea) = 60/0.6 = 100 mm
        a = adv.evaluate(dr=60.0, taw=120.0, p=0.5, growth_stage="dev", etc=6.9)
        return FieldAdvisory(
            field_id=fid, advisory=a, area_ha=area, is_tail_end=tail, distance_from_head_m=dist
        )

    def test_volume_and_turn_time(self):
        # two fields: dgross=100 mm over 10 ha and 5 ha
        # V = 100*10*10 + 100*5*10 = 15000 m3 ; Q=0.03 -> T = 15000/0.03/3600 h
        f1 = self._due_field("F1", 10.0, dist=100.0)
        f2 = self._due_field("F2", 5.0, dist=900.0, tail=True)
        plan = aggregate_to_command([f1, f2], outlet_discharge_q=0.03)
        assert plan.total_volume_m3 == pytest.approx(15000.0)
        assert plan.turn_time_hours == pytest.approx(15000.0 / 0.03 / 3600.0)
        assert plan.n_need_irrigation == 2

    def test_unit_conversion_mm_ha_to_m3(self):
        # 1 mm over 1 ha == 10 m3 ; dgross=100mm over 1ha -> 1000 m3
        f = self._due_field("F", 1.0)
        plan = aggregate_to_command([f], outlet_discharge_q=1.0)
        assert plan.total_volume_m3 == pytest.approx(1000.0)

    def test_tail_end_priority_ordering(self):
        f_head = self._due_field("HEAD", 10.0, dist=50.0)
        f_tail = self._due_field("TAIL", 10.0, dist=950.0, tail=True)
        plan = aggregate_to_command(
            [f_head, f_tail], outlet_discharge_q=0.05, tail_end_priority=True
        )
        assert plan.order[0] == "TAIL"  # tail-end served first

    def test_only_due_excludes_non_due(self):
        adv = IrrigationAdvisory()
        wet = FieldAdvisory(
            "WET",
            adv.evaluate(dr=0.0, taw=120.0, p=0.5, growth_stage="dev", etc=6.9),
            area_ha=10.0,
        )
        plan = aggregate_to_command(
            [self._due_field("DUE", 10.0), wet], outlet_discharge_q=0.05, only_due=True
        )
        assert plan.per_field_volume_m3["WET"] == 0.0
        assert plan.n_need_irrigation == 1

    def test_invalid_discharge_raises(self):
        with pytest.raises(ValueError):
            aggregate_to_command([self._due_field("F", 1.0)], outlet_discharge_q=0.0)


# =============================================================================
# End-to-end pipeline smoke test
# =============================================================================


def test_end_to_end_pipeline():
    """ET0 -> Kc -> water balance -> 8-day deficit -> advisory."""
    # 1. ET0 (FAO-56 Example 18)
    et0 = et0_penman_monteith(
        tmin=12.3,
        tmax=21.5,
        rh_min=63,
        rh_max=84,
        u2=2.078,
        rs=22.07,
        elevation=100,
        lat=50.80,
        doy=187,
    )
    assert et0 == pytest.approx(3.88, abs=0.2)

    # 2. Kc for wheat mid-season
    kc = kc_for_stage("wheat", "mid")
    assert kc == pytest.approx(1.15)

    # 3. Soil + root-zone balance for 8 days
    soil = SoilProperties(theta_fc=0.22, theta_wp=0.10, zr=1.0, p=0.5)
    res = eight_day_deficit(soil, et0=[et0] * 8, kc=kc, application_efficiency=0.6)
    assert res.dr > 0
    assert math.isfinite(res.dgross)

    # 4. Advisory on the resulting depletion
    adv = IrrigationAdvisory(application_efficiency=0.6)
    a = adv.evaluate(dr=res.dr, taw=soil.taw(), p=soil.p, growth_stage="mid", etc=kc * et0)
    assert a.status in set(IrrigationStatus)
    assert a.dgross == pytest.approx(a.dnet / 0.6)
