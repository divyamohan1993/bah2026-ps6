"""Tests for the modelling subpackage (classifier, stress, evaluate, train).

Verifies that the synthetic end-to-end pipeline reaches the PS6 accuracy target
(OA > 0.85, kappa > 0.8), the satellite-embedding classifier works on 64-D
vectors, Olofsson area-adjusted accuracy returns confidence intervals, and the
stage-aware stress model returns valid severity classes that depend on growth
stage.
"""

from __future__ import annotations

import numpy as np
import pytest

from agristress.models import evaluate as ev
from agristress.models import stress as st
from agristress.models.crop_classifier import (
    CropClassifier,
    SatelliteEmbeddingClassifier,
    available_backends,
)
from agristress.models.train import (
    make_synthetic_crop_dataset,
    spatial_block_split,
    train_crop_model,
)


# --------------------------------------------------------------------------- #
# Synthetic dataset & end-to-end training
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def synth():
    return make_synthetic_crop_dataset(n_per_class=120, n_timesteps=12, n_classes=4, seed=11)


def test_synthetic_dataset_shape(synth):
    assert synth["X"].shape[0] == 4 * 120
    assert len(synth["feature_names"]) == synth["X"].shape[1]
    assert set(np.unique(synth["y"])) == set(synth["classes"])
    assert np.isfinite(synth["X"]).all()  # imputed, no NaNs


def test_end_to_end_train_reaches_target_oa(synth):
    """RandomForest on the synthetic set must exceed OA > 0.85 and kappa > 0.8."""
    res = train_crop_model(
        synth["X"],
        synth["y"],
        model="rf",
        feature_names=synth["feature_names"],
        coords=synth["coords"],
        split="spatial",
        n_blocks=6,
        random_state=11,
    )
    assert res.metrics["overall_accuracy"] > 0.85
    assert res.metrics["cohen_kappa"] > 0.8
    # Per-class report present and confusion matrix square.
    cm = np.asarray(res.metrics["confusion_matrix"])
    assert cm.shape[0] == cm.shape[1] == len(synth["classes"])


def test_train_lightgbm_falls_back_gracefully(synth):
    """Requesting lightgbm works whether or not it is installed (falls back to RF)."""
    res = train_crop_model(
        synth["X"],
        synth["y"],
        model="lightgbm",
        feature_names=synth["feature_names"],
        split="random",
        random_state=5,
    )
    assert res.metrics["overall_accuracy"] > 0.85
    backends = available_backends()
    if backends["lightgbm"]:
        assert "lightgbm" in res.backend
    else:
        assert "random_forest" in res.backend


def test_classifier_predict_proba_and_importance(synth):
    clf = CropClassifier(model="rf", n_estimators=100, random_state=0)
    clf.fit(synth["X"], synth["y"], feature_names=synth["feature_names"])
    proba = clf.predict_proba(synth["X"][:5])
    assert proba.shape == (5, len(clf.classes_))
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    imp = clf.feature_importance()
    assert isinstance(imp, dict) and len(imp) > 0
    assert all(v >= 0 for v in imp.values())


def test_train_saves_model_and_report(synth, tmp_path):
    res = train_crop_model(
        synth["X"],
        synth["y"],
        model="rf",
        feature_names=synth["feature_names"],
        split="random",
        random_state=1,
        save_dir=tmp_path,
        model_name="demo",
    )
    assert res.report_path is not None
    import json
    from pathlib import Path

    report = json.loads(Path(res.report_path).read_text())
    assert "overall_accuracy" in report and "confusion_matrix" in report
    # joblib model should round-trip if joblib is available.
    if res.model_path is not None:
        import joblib

        loaded = joblib.load(res.model_path)
        assert "model" in loaded
        pred = loaded["model"].predict(synth["X"][:3])
        assert len(pred) == 3


# --------------------------------------------------------------------------- #
# Spatial split (leakage control)
# --------------------------------------------------------------------------- #
def test_spatial_block_split_disjoint(synth):
    tr, te = spatial_block_split(
        synth["coords"], synth["y"], test_size=0.3, n_blocks=6, axis="grid", seed=0
    )
    assert len(set(tr).intersection(te)) == 0
    assert tr.size > 0 and te.size > 0
    assert tr.size + te.size == synth["X"].shape[0]


# --------------------------------------------------------------------------- #
# Satellite-embedding classifier
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def embeddings():
    """Synthetic 64-D class-clustered embeddings (AlphaEarth-like)."""
    rng = np.random.default_rng(3)
    K, d, per = 4, 64, 80
    centers = rng.normal(0, 1.0, (K, d))
    X = np.vstack([centers[k] + rng.normal(0, 0.35, (per, d)) for k in range(K)])
    y = np.repeat(np.arange(K), per)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


@pytest.mark.parametrize("method", ["cosine", "knn", "logistic", "rf"])
def test_embedding_classifier_methods(embeddings, method):
    """All embedding heads classify well-separated 64-D clusters accurately."""
    X, y = embeddings
    n = len(y)
    cut = int(n * 0.7)
    clf = SatelliteEmbeddingClassifier(method=method).fit(X[:cut], y[:cut])
    pred = clf.predict(X[cut:])
    oa = ev.overall_accuracy(y[cut:], pred)
    assert oa > 0.85
    proba = clf.predict_proba(X[cut : 5 + cut])
    assert proba.shape[1] == len(clf.classes_)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_embedding_train_via_driver(embeddings):
    X, y = embeddings
    res = train_crop_model(X, y, model="embedding", method="cosine", split="random", random_state=0)
    assert res.metrics["overall_accuracy"] > 0.85
    assert res.backend.startswith("embedding")


# --------------------------------------------------------------------------- #
# Evaluation metrics
# --------------------------------------------------------------------------- #
def test_overall_accuracy_and_kappa_perfect():
    y = np.array([0, 1, 2, 0, 1, 2])
    assert ev.overall_accuracy(y, y) == pytest.approx(1.0)
    assert ev.cohen_kappa(y, y) == pytest.approx(1.0)


def test_per_class_metrics_structure():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 2])
    rep = ev.classification_report_dict(y_true, y_pred)
    assert set(rep["per_class"]) == {"0", "1", "2"}
    assert 0.0 <= rep["overall_accuracy"] <= 1.0
    assert "macro" in rep and "weighted" in rep
    cm = np.asarray(rep["confusion_matrix"])
    assert cm.sum() == len(y_true)


def test_olofsson_accuracy_returns_ci():
    """Olofsson area-adjusted accuracy returns OA + 95% CI and adjusted areas."""
    cm = np.array(
        [
            [80, 4, 2, 1],
            [5, 70, 3, 2],
            [3, 4, 90, 3],
            [2, 1, 2, 75],
        ],
        dtype=float,
    )
    areas = [0.40, 0.20, 0.25, 0.15]
    res = ev.olofsson_accuracy(cm, areas)
    oa = res["overall_accuracy"]
    lo, hi = res["overall_accuracy_ci"]
    assert 0.0 <= oa <= 1.0
    assert lo <= oa <= hi
    assert hi - lo > 0  # a non-degenerate interval
    # User's / producer's accuracies are per-class and within [0, 1].
    assert res["users_accuracy"].shape == (4,)
    assert np.all((res["users_accuracy"] >= 0) & (res["users_accuracy"] <= 1))
    # Error-adjusted area proportions sum to ~1.
    assert res["area_proportion"].sum() == pytest.approx(1.0, abs=1e-6)
    # CI arrays have a [low, high] column per class.
    assert res["users_accuracy_ci"].shape == (4, 2)


def test_olofsson_weights_renormalised():
    """Un-normalised area weights are handled (renormalised internally)."""
    cm = np.array([[50, 5], [4, 60]], dtype=float)
    res = ev.olofsson_accuracy(cm, [2.0, 3.0])  # sum != 1
    assert res["weights"].sum() == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# OPTRAM & condition indices
# --------------------------------------------------------------------------- #
def test_optram_soil_moisture_bounds():
    ndvi = np.array([0.2, 0.5, 0.8])
    strv = np.array([5.0, 3.0, 1.0])
    edges = {"i_dry": 0.5, "s_dry": 0.5, "i_wet": 6.0, "s_wet": 0.5}
    w = st.optram_soil_moisture(ndvi, strv, edges)
    assert np.all((w >= 0) & (w <= 1))
    # Drier STR (lower) -> lower moisture for fixed edges.
    assert w[0] > w[2]


def test_condition_indices_ranges():
    vci = st.vci(np.array([0.2, 0.5, 0.8]), np.array([0.2]), np.array([0.8]))
    assert vci[0] == pytest.approx(0.0)
    assert vci[2] == pytest.approx(1.0)
    vhi = st.vhi(np.array([0.4]), np.array([0.6]), alpha=0.5)
    assert vhi[0] == pytest.approx(0.5)
    tvdi = st.tvdi(np.array([0.5]), np.array([305.0]), dry_edge=(320.0, -20.0), wet_edge=295.0)
    assert 0.0 <= tvdi[0] <= 1.0


def test_anomaly_vs_baseline_zscore():
    baseline = np.array([0.5, 0.55, 0.6, 0.45, 0.5])
    z = st.anomaly_vs_baseline(np.array([baseline.mean()]), baseline)
    assert z[0] == pytest.approx(0.0, abs=1e-9)
    # A value below the baseline mean gives a negative z (stress signal).
    z2 = st.anomaly_vs_baseline(np.array([0.2]), baseline)
    assert z2[0] < 0


# --------------------------------------------------------------------------- #
# Stage-aware stress
# --------------------------------------------------------------------------- #
def test_stage_aware_returns_valid_classes():
    saw = st.StageAwareStress()
    res = saw.classify(np.array([0.75, 0.50, 0.35, 0.22, 0.10]), "mid-season")
    assert np.all(np.isin(res.severity, [0, 1, 2, 3, 4]))
    assert res.severity[0] == 0  # healthy
    assert res.severity[-1] == 4  # severe
    assert list(map(str, res.labels)) == ["none", "moderate", "high", "severe", "severe"]


def test_stage_aware_threshold_depends_on_stage():
    """Same condition value is stricter (more stressed) during mid-season."""
    saw = st.StageAwareStress()
    cond = 0.40
    mid = saw.severity(cond, "mid-season")
    mature = saw.severity(cond, "mature")
    assert mid > mature  # crop more sensitive at mid-season -> higher severity


def test_stage_aware_stress_index_inversion():
    """higher_is_stress=True treats the input as a stress index and inverts it."""
    saw = st.StageAwareStress(higher_is_stress=True)
    # High TVDI (0.85) -> severe.
    assert saw.severity(0.85, "mid-season") == 4
    # Low TVDI (0.05) -> none.
    assert saw.severity(0.05, "mid-season") == 0


def test_stage_aware_handles_nan():
    saw = st.StageAwareStress()
    res = saw.classify(np.array([np.nan]), "initial")
    assert res.severity[0] == -1
    assert str(res.labels[0]) == "no_data"
