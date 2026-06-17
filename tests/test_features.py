"""Tests for the feature-engineering subpackage (indices, texture, phenology).

Covers spectral-index value ranges & NaN-safety, GLCM texture properties (both
skimage and numpy-fallback paths), double-logistic recovery of a known curve's
SOS, harmonic features, Whittaker smoothing, and GDD-based growth-stage
assignment.
"""

from __future__ import annotations

import numpy as np
import pytest

from agristress.features import indices as ix
from agristress.features import phenology as ph
from agristress.features import texture as tx


# --------------------------------------------------------------------------- #
# Spectral indices
# --------------------------------------------------------------------------- #
@pytest.fixture
def reflectance():
    rng = np.random.default_rng(7)
    n = 2000
    return {
        "nir": rng.uniform(0.20, 0.60, n),
        "red": rng.uniform(0.02, 0.20, n),
        "blue": rng.uniform(0.01, 0.10, n),
        "green": rng.uniform(0.05, 0.20, n),
        "red_edge": rng.uniform(0.10, 0.40, n),
        "swir1": rng.uniform(0.10, 0.40, n),
        "swir2": rng.uniform(0.05, 0.30, n),
        "nir_n": rng.uniform(0.20, 0.60, n),
    }


@pytest.mark.parametrize(
    "fn_name,args",
    [
        ("ndvi", ("nir", "red")),
        ("ndwi_gao", ("nir", "swir1")),
        ("ndwi_mcfeeters", ("green", "nir")),
        ("ndmi", ("nir_n", "swir1")),
        ("ndre", ("nir", "red_edge")),
    ],
)
def test_normalized_difference_indices_in_unit_range(reflectance, fn_name, args):
    """All normalized-difference indices must lie within [-1, 1]."""
    fn = getattr(ix, fn_name)
    out = fn(*[reflectance[a] for a in args])
    finite = out[np.isfinite(out)]
    assert finite.size > 0
    assert np.all(finite >= -1.0 - 1e-9)
    assert np.all(finite <= 1.0 + 1e-9)


def test_savi_msavi_bounded(reflectance):
    """SAVI/MSAVI for positive vegetation stay within a sane (~[-1.5, 1.5]) range."""
    savi = ix.savi(reflectance["nir"], reflectance["red"])
    msavi = ix.msavi(reflectance["nir"], reflectance["red"])
    for arr in (savi, msavi):
        f = arr[np.isfinite(arr)]
        assert np.all(f >= -1.5) and np.all(f <= 1.5)


def test_ndvi_known_values():
    """NDVI sanity at analytic points."""
    assert ix.ndvi(0.8, 0.2) == pytest.approx(0.6)
    assert ix.ndvi(0.5, 0.5) == pytest.approx(0.0)
    assert ix.ndvi(0.0, 1.0) == pytest.approx(-1.0)


def test_indices_are_nan_safe():
    """Zero / invalid denominators yield NaN, never an exception or warning."""
    assert np.isnan(ix.ndvi(0.0, 0.0))
    assert np.isnan(ix.str_index(0.0))
    assert np.isnan(ix.gcvi(0.5, 0.0))
    # Vectorised: a zero-denominator element among valid ones -> NaN there only.
    nir = np.array([0.5, 0.0, 0.4])
    red = np.array([0.1, 0.0, 0.05])
    out = ix.ndvi(nir, red)
    assert np.isnan(out[1])
    assert np.all(np.isfinite(out[[0, 2]]))


def test_str_index_optram_formula():
    """STR = (1 - SWIR)^2 / (2 * SWIR)."""
    swir = 0.25
    expected = (1 - swir) ** 2 / (2 * swir)
    assert ix.str_index(swir) == pytest.approx(expected)


def test_sar_rvi_and_cross_ratio():
    """RVI = 4*VH/(VV+VH); cross-ratio = VH/VV."""
    vv, vh = 0.09, 0.03
    assert ix.sar_rvi(vv, vh) == pytest.approx(4 * vh / (vv + vh))
    assert ix.sar_cross_ratio(vv, vh) == pytest.approx(vh / vv)
    # RVI should be non-negative for non-negative power inputs.
    assert ix.sar_rvi(0.1, 0.05) >= 0


def test_evi_evi2_finite(reflectance):
    """EVI / EVI2 produce finite values for valid reflectance (not range-bound)."""
    evi = ix.evi(reflectance["nir"], reflectance["red"], reflectance["blue"])
    evi2 = ix.evi2(reflectance["nir"], reflectance["red"])
    assert np.isfinite(evi).mean() > 0.99
    assert np.isfinite(evi2).mean() > 0.99


def test_nmdi_psri_finite(reflectance):
    nmdi = ix.nmdi(reflectance["nir"], reflectance["swir1"], reflectance["swir2"])
    psri = ix.psri(reflectance["red"], reflectance["blue"], reflectance["red_edge"])
    assert np.isfinite(nmdi).mean() > 0.95
    assert np.isfinite(psri).mean() > 0.95


def test_stack_indices_skips_missing_bands():
    """stack_indices computes only indices whose bands are present, never raises."""
    bands = {"nir": np.array([0.5]), "red": np.array([0.1])}  # no blue/swir
    out = ix.stack_indices(bands)
    assert "ndvi" in out and "evi2" in out
    assert "evi" not in out  # needs blue
    assert "ndmi" not in out  # needs nir_n + swir1


# --------------------------------------------------------------------------- #
# Texture (GLCM)
# --------------------------------------------------------------------------- #
def test_glcm_features_keys_present():
    """The four headline Haralick properties are returned."""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (32, 32)).astype(float)
    feats = tx.glcm_features(img, levels=16)
    for prop in ("contrast", "homogeneity", "entropy", "correlation"):
        assert f"glcm_{prop}" in feats
        assert np.isfinite(feats[f"glcm_{prop}"])


def test_glcm_constant_image_zero_contrast():
    """A constant image has zero contrast and zero entropy."""
    img = np.full((20, 20), 5.0)
    feats = tx.glcm_features(img, levels=8)
    assert feats["glcm_contrast"] == pytest.approx(0.0, abs=1e-9)
    assert feats["glcm_entropy"] == pytest.approx(0.0, abs=1e-9)


def test_glcm_numpy_fallback_matches_structure():
    """The pure-numpy GLCM path returns the same property set as the fast path."""
    rng = np.random.default_rng(3)
    img = rng.integers(0, 64, (24, 24)).astype(float)
    glcm = tx.graycomatrix_np(img, levels=16, distance=1)
    props = tx.glcm_props(glcm)
    for prop in ("contrast", "homogeneity", "entropy", "correlation", "energy"):
        assert prop in props and np.isfinite(props[prop])
    # A normalised GLCM sums to 1 per angle.
    assert glcm.sum(axis=(0, 1)) == pytest.approx(np.ones(glcm.shape[2]))


def test_glcm_contrast_orders_with_structure():
    """A high-frequency checkerboard has higher contrast than a smooth gradient."""
    n = 32
    checker = np.indices((n, n)).sum(axis=0) % 2 * 255.0
    gradient = np.tile(np.linspace(0, 255, n), (n, 1))
    c_checker = tx.glcm_features(checker, levels=16)["glcm_contrast"]
    c_grad = tx.glcm_features(gradient, levels=16)["glcm_contrast"]
    assert c_checker > c_grad


# --------------------------------------------------------------------------- #
# Phenology
# --------------------------------------------------------------------------- #
def test_whittaker_smooth_reduces_noise():
    """Smoothing a noisy sinusoid lowers RMSE to the clean signal."""
    t = np.linspace(0, 2 * np.pi, 60)
    clean = 0.5 + 0.4 * np.sin(t)
    rng = np.random.default_rng(1)
    noisy = clean + rng.normal(0, 0.08, t.size)
    sm = ph.whittaker_smooth(noisy, lmbda=20.0)
    assert np.sqrt(np.mean((sm - clean) ** 2)) < np.sqrt(np.mean((noisy - clean) ** 2))


def test_whittaker_handles_nan_gaps():
    """NaNs are gap-filled rather than propagated."""
    y = np.array([0.1, 0.2, np.nan, 0.4, 0.5, np.nan, 0.7])
    sm = ph.whittaker_smooth(y, lmbda=5.0)
    assert np.all(np.isfinite(sm))


def test_double_logistic_recovers_sos():
    """double_logistic_fit recovers a known curve's SOS within tolerance."""
    t = np.linspace(1, 240, 24)
    true_sos = 60.0
    vi = ph.double_logistic(t, base=0.15, amp=0.70, sos=true_sos, r_sp=0.15, eos=180.0, r_au=0.12)
    rng = np.random.default_rng(2)
    vi_noisy = vi + rng.normal(0, 0.02, vi.size)
    fit = ph.double_logistic_fit(t, vi_noisy, amp_threshold=0.5)
    assert fit.success
    # The fitted spring-inflection parameter should be close to the truth.
    assert fit.params["sos"] == pytest.approx(true_sos, abs=10.0)
    # Derived phenometrics are internally consistent.
    assert fit.sos < fit.pos < fit.eos
    assert fit.lgp == pytest.approx(fit.eos - fit.sos, abs=1e-6)
    assert fit.amplitude > 0.5


def test_double_logistic_fallback_on_short_series():
    """Too-few points -> graceful fallback metrics, no exception."""
    t = np.array([1.0, 2.0, 3.0])
    vi = np.array([0.2, 0.5, 0.3])
    fit = ph.double_logistic_fit(t, vi)
    assert fit.success is False
    assert np.isfinite(fit.pos)


def test_harmonic_features_capture_amplitude():
    """A pure cosine is recovered with the right first-harmonic amplitude."""
    t = np.linspace(0, 365, 48)
    amp = 0.3
    vi = 0.5 + amp * np.cos(2 * np.pi * t / 365.0)
    feats = ph.harmonic_features(t, vi, n=3, period=365.0)
    assert feats["harm_amp1"] == pytest.approx(amp, abs=0.05)
    assert feats["harm_mean"] == pytest.approx(0.5, abs=0.05)
    assert feats["harm_rmse"] < 1e-3


def test_phenometrics_returns_flat_dict():
    t = np.linspace(1, 240, 20)
    vi = ph.double_logistic(t, 0.15, 0.6, 50, 0.12, 170, 0.1)
    feats = ph.phenometrics(t, vi)
    for key in ("sos", "pos", "eos", "lgp", "amplitude", "integral", "vi_mean", "harm_amp1"):
        assert key in feats


# --------------------------------------------------------------------------- #
# GDD & growth stage
# --------------------------------------------------------------------------- #
def test_gdd_accumulate_monotone():
    """Cumulative GDD is non-decreasing and matches the hand calculation."""
    tmin = np.full(10, 15.0)
    tmax = np.full(10, 25.0)
    g = ph.gdd_accumulate(tmin, tmax, tbase=10.0)
    assert np.all(np.diff(g) >= 0)
    # daily = (25+15)/2 - 10 = 10 -> cumulative day 10 == 100
    assert g[-1] == pytest.approx(100.0)


def test_gdd_cold_day_no_negative():
    """Below-base temperatures contribute zero, never negative GDD."""
    tmin = np.array([-5.0, -5.0])
    tmax = np.array([2.0, 2.0])
    g = ph.gdd_accumulate(tmin, tmax, tbase=10.0)
    assert np.all(g >= 0)
    assert g[-1] == pytest.approx(0.0)


def test_assign_growth_stage_progression():
    """GDD thresholds map to the FAO-56 stage sequence for a crop."""
    stages = [ph.assign_growth_stage(g, "wheat") for g in (50, 400, 1000, 1600, 2500)]
    assert stages == ["initial", "development", "mid-season", "late-season", "mature"]


def test_assign_growth_stage_vectorized_and_default():
    out = ph.assign_growth_stage(np.array([50, 1000, 2500]), "maize")
    assert list(map(str, out)) == ["initial", "mid-season", "mature"]
    # Unknown crop falls back to the default table without error.
    assert isinstance(ph.assign_growth_stage(500, "unknown_crop"), str)


def test_growth_stage_fractions_cover_all_stages():
    frac = ph.growth_stage_fractions("rice")
    assert set(frac) == set(ph.STAGE_LABELS)
    # Intervals are contiguous and ascending.
    lowers = [frac[s][0] for s in ph.STAGE_LABELS]
    assert lowers == sorted(lowers)
