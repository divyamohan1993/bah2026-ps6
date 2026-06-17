"""Tests for the fusion subpackage (gap-fill, SWI, triple-collocation, consensus, cube)."""

from __future__ import annotations

import numpy as np
import pytest

from agristress.fusion import (
    build_datacube,
    gap_fill_temporal,
    multi_sensor_consensus,
    sar_to_ndvi,
    starfm,
    swi_exponential_filter,
    triple_collocation,
    whittaker_smooth,
)
from agristress.fusion.datacube import CUBE_VARS


# ---------------------------------------------------------------------------
# Temporal gap-filling
# ---------------------------------------------------------------------------
def test_whittaker_fills_gaps():
    t = np.linspace(0, 2 * np.pi, 40)
    clean = 0.5 + 0.4 * np.sin(t)
    y = clean.copy()
    y[5:9] = np.nan  # a cloud gap
    y[20] = np.nan
    y[33:35] = np.nan

    filled = whittaker_smooth(y, lambda_=20.0)

    assert filled.shape == y.shape
    assert np.isfinite(filled).all(), "no NaNs should remain after gap-filling"
    # Filled values track the underlying signal reasonably well.
    assert np.max(np.abs(filled - clean)) < 0.15


def test_gap_fill_dispatcher_methods_agree_shape():
    y = np.array([0.2, np.nan, 0.4, 0.5, np.nan, 0.55, 0.5, 0.4, np.nan, 0.3])
    w = gap_fill_temporal(y, method="whittaker", lambda_=5.0)
    s = gap_fill_temporal(y, method="savgol", window=5, polyorder=2)
    assert w.shape == y.shape == s.shape
    assert np.isfinite(w).all() and np.isfinite(s).all()


def test_gap_fill_temporal_on_stack():
    rng = np.random.default_rng(0)
    stack = 0.5 + 0.3 * np.sin(np.linspace(0, 6, 24))[:, None, None] * np.ones((24, 3, 3))
    stack = stack + rng.normal(0, 0.01, stack.shape)
    stack[3:6, 0, 0] = np.nan
    out = gap_fill_temporal(stack, method="whittaker", lambda_=10.0)
    assert out.shape == stack.shape
    assert np.isfinite(out).all()


# ---------------------------------------------------------------------------
# Soil Water Index — monotone smoothing / lag behaviour
# ---------------------------------------------------------------------------
def test_swi_smooths_and_is_bounded():
    n = 60
    time = np.arange(n)
    rng = np.random.default_rng(1)
    base = 0.3 + 0.15 * np.sin(time / 8.0)
    ssm = np.clip(base + rng.normal(0, 0.05, n), 0, 1)

    swi = swi_exponential_filter(ssm, time, T=10.0)

    assert swi.shape == ssm.shape
    assert np.isfinite(swi).all()
    # SWI is a low-pass of SSM: its temporal variance must be smaller.
    assert np.var(np.diff(swi)) < np.var(np.diff(ssm))
    # Stays within the data envelope (no overshoot).
    assert swi.min() >= ssm.min() - 1e-9
    assert swi.max() <= ssm.max() + 1e-9


def test_swi_step_response_is_monotonic():
    # A step in SSM should give a monotonically rising, lagged SWI (no oscillation).
    time = np.arange(30)
    ssm = np.where(time < 10, 0.1, 0.4).astype(float)
    swi = swi_exponential_filter(ssm, time, T=8.0)
    rising = np.diff(swi[10:])
    assert (rising >= -1e-9).all(), "SWI must not oscillate on a positive step"
    assert swi[-1] > swi[9]  # eventually approaches the new level


def test_swi_carries_over_nan_gaps():
    time = np.arange(10)
    ssm = np.array([0.2, 0.25, np.nan, np.nan, 0.3, 0.32, np.nan, 0.35, 0.36, 0.4])
    swi = swi_exponential_filter(ssm, time, T=5.0)
    assert swi.shape == ssm.shape
    assert np.isfinite(swi).all(), "gaps are filled by persistence"


# ---------------------------------------------------------------------------
# Triple collocation
# ---------------------------------------------------------------------------
def test_triple_collocation_returns_three_variances():
    rng = np.random.default_rng(7)
    n = 2000
    truth = rng.normal(0.3, 0.1, n)
    # Three products = scaled truth + independent noise of different magnitude.
    x = truth + rng.normal(0, 0.02, n)
    y = 1.0 * truth + rng.normal(0, 0.05, n)
    z = truth + rng.normal(0, 0.08, n)

    res = triple_collocation(x, y, z)

    assert set(res["var_err"]) == {"x", "y", "z"}
    vex, vey, vez = (res["var_err"][k] for k in ("x", "y", "z"))
    for v in (vex, vey, vez):
        assert np.isfinite(v) and v >= 0
    # Recovered error ordering matches the injected noise (x < y < z).
    assert vex < vey < vez
    # Magnitudes near the true error variances (0.02^2, 0.05^2, 0.08^2).
    assert abs(vex - 0.02**2) < 0.0015
    assert abs(vez - 0.08**2) < 0.004
    assert res["n"] == n


def test_triple_collocation_requires_enough_samples():
    with pytest.raises(ValueError):
        triple_collocation([1.0, np.nan], [1.0, 2.0], [np.nan, 2.0])


# ---------------------------------------------------------------------------
# Multi-sensor consensus
# ---------------------------------------------------------------------------
def test_consensus_rejects_outlier():
    # Four agreeing estimates + one gross outlier per pixel.
    base = np.full((5, 4, 4), 0.6)
    base[0] += 0.01
    base[1] -= 0.01
    base[2] += 0.005
    base[3] -= 0.005
    base[4] = 5.0  # outlier sensor

    consensus, kept = multi_sensor_consensus(base, return_mask=True)

    assert consensus.shape == (4, 4)
    # Consensus close to the inlier mean (~0.6), NOT dragged toward 5.0.
    assert np.all(np.abs(consensus - 0.6) < 0.05)
    # The outlier source is rejected everywhere.
    assert not kept[4].any()
    assert kept[:4].all()


def test_consensus_weighted_and_scalar():
    # Scalar (1-D) inputs: three estimates, one outlier.
    vals = np.array([0.50, 0.52, 0.48, 2.0])
    out = multi_sensor_consensus(vals)
    assert np.ndim(out) == 0 or out.shape == ()
    assert abs(float(out) - 0.5) < 0.05


# ---------------------------------------------------------------------------
# STARFM smoke + SAR->NDVI demo
# ---------------------------------------------------------------------------
def test_starfm_predicts_coarse_change():
    rng = np.random.default_rng(3)
    fine_t0 = rng.uniform(0.2, 0.6, (20, 20))
    coarse_t0 = fine_t0 + rng.normal(0, 0.01, fine_t0.shape)
    # Coarse brightens uniformly by +0.1 at t1; prediction should follow.
    coarse_t1 = coarse_t0 + 0.1
    pred = starfm(fine_t0, coarse_t0, coarse_t1, win=9)
    assert pred.shape == fine_t0.shape
    assert abs(float(np.mean(pred - fine_t0)) - 0.1) < 0.03


def test_sar_to_ndvi_demo_monotonic_and_bounded():
    rvi = np.linspace(0.0, 2.0, 50)
    ndvi = sar_to_ndvi({"rvi": rvi})
    assert ndvi.shape == rvi.shape
    assert (ndvi >= 0).all() and (ndvi <= 1).all()
    assert (np.diff(ndvi) >= -1e-9).all(), "NDVI proxy must increase with RVI"


# ---------------------------------------------------------------------------
# Datacube (DEMO mode)
# ---------------------------------------------------------------------------
def test_build_datacube_demo_dims_and_vars():
    ds = build_datacube(height=16, width=12, n_times=10, seed=5)

    assert set(ds.dims) == {"time", "y", "x"}
    assert ds.sizes["time"] == 10
    assert ds.sizes["y"] == 16
    assert ds.sizes["x"] == 12
    # All canonical cube variables are present.
    for var in CUBE_VARS:
        assert var in ds.data_vars, f"missing cube variable {var!r}"
    assert ds.attrs.get("mode") == "demo"


def test_build_datacube_demo_has_realistic_phenology_and_clouds():
    ds = build_datacube(height=20, width=20, n_times=20, seed=11)
    ndvi = ds["ndvi"].values  # has cloud NaNs

    # Cloud gaps exist (qa==0 ⇒ NDVI is NaN there).
    assert np.isnan(ndvi).any(), "demo cube should contain cloud gaps"
    assert (ds["qa"].values == 0).any()

    # Seasonal phenology: spatial-mean NDVI peaks somewhere in the interior,
    # i.e. it rises then falls rather than being flat.
    season = np.nanmean(ndvi, axis=(1, 2))
    assert np.nanmax(season) - np.nanmin(season) > 0.1
    peak = int(np.nanargmax(season))
    assert 0 < peak < len(season) - 1

    # Soil moisture and sigma are finite & physical.
    assert np.isfinite(ds["soil_moisture"].values).all()
    assert (ds["soil_moisture"].values >= 0).all()
    assert (ds["sigma"].values > 0).all()


def test_build_datacube_real_layers_path():
    nt, h, w = 6, 8, 8
    rng = np.random.default_rng(0)
    layers = {
        "ndvi": rng.uniform(0, 1, (nt, h, w)).astype("float32"),
        "red": rng.uniform(0, 0.3, (nt, h, w)).astype("float32"),
    }
    ds = build_datacube(layers=layers, demo=False, times=None, n_times=nt)
    assert ds.sizes == {"time": nt, "y": h, "x": w}
    # qa / sigma are auto-added when absent.
    assert "qa" in ds.data_vars and "sigma" in ds.data_vars
    assert ds.attrs["mode"] == "real"
