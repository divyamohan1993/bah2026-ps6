"""Training driver, synthetic-data generator, and leakage-aware splitting.

Provides:

* :func:`make_synthetic_crop_dataset` -- generate a labelled, multi-temporal
  feature table for >=4 crop classes with realistic, separable phenology so an
  end-to-end ``train -> predict -> evaluate`` runs fully offline.
* :func:`build_feature_stack` -- assemble a flat feature matrix from a
  multi-temporal datacube / table (indices time series + SAR + texture +
  phenometrics).
* :func:`spatial_block_split` -- block / group train-test split that keeps
  spatially contiguous samples together to avoid optimistic leakage.
* :func:`train_crop_model` -- fit a :class:`~agristress.models.crop_classifier.CropClassifier`
  or :class:`~agristress.models.crop_classifier.SatelliteEmbeddingClassifier`,
  evaluate it, and (optionally) persist the model + a JSON metrics report via
  ``joblib``.

No Earth Engine / credentials required.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import evaluate as _eval
from .crop_classifier import CropClassifier, SatelliteEmbeddingClassifier

__all__ = [
    "make_synthetic_crop_dataset",
    "build_feature_stack",
    "spatial_block_split",
    "train_crop_model",
    "TrainResult",
]


# --------------------------------------------------------------------------- #
# Synthetic multi-temporal crop dataset
# --------------------------------------------------------------------------- #
# Canonical phenology archetypes for four kharif/rabi crops, expressed as
# double-logistic-like NDVI curves over a season.  Each crop differs in
# amplitude, timing (SOS/POS), season length, and SAR backscatter so that a
# multi-temporal classifier can separate them well.
@dataclass
class _CropArchetype:
    name: str
    base: float  # baseline NDVI
    amp: float  # peak NDVI above baseline
    sos: float  # start-of-season (fraction of season 0-1)
    eos: float  # end-of-season (fraction of season 0-1)
    rise: float  # green-up steepness
    fall: float  # senescence steepness
    vh_mean: float  # mean SAR VH backscatter (linear)
    vv_mean: float  # mean SAR VV backscatter (linear)
    texture: float  # mean GLCM-contrast-like structural value


_DEFAULT_ARCHETYPES = [
    _CropArchetype("rice", base=0.18, amp=0.62, sos=0.28, eos=0.78, rise=14, fall=10, vh_mean=0.030, vv_mean=0.090, texture=0.18),
    _CropArchetype("wheat", base=0.15, amp=0.70, sos=0.18, eos=0.70, rise=12, fall=9, vh_mean=0.018, vv_mean=0.070, texture=0.12),
    _CropArchetype("maize", base=0.16, amp=0.66, sos=0.22, eos=0.62, rise=16, fall=13, vh_mean=0.024, vv_mean=0.080, texture=0.22),
    _CropArchetype("cotton", base=0.20, amp=0.50, sos=0.30, eos=0.88, rise=8, fall=6, vh_mean=0.040, vv_mean=0.100, texture=0.30),
    _CropArchetype("sugarcane", base=0.25, amp=0.55, sos=0.10, eos=0.95, rise=7, fall=5, vh_mean=0.045, vv_mean=0.110, texture=0.35),
]


def _logistic_curve(t01: np.ndarray, a: _CropArchetype) -> np.ndarray:
    spring = 1.0 / (1.0 + np.exp(-a.rise * (t01 - a.sos)))
    autumn = 1.0 / (1.0 + np.exp(-a.fall * (t01 - a.eos)))
    return a.base + a.amp * (spring - autumn)


def make_synthetic_crop_dataset(
    n_per_class: int = 120,
    n_timesteps: int = 12,
    n_classes: int = 4,
    *,
    noise: float = 0.05,
    grid_size: int | None = None,
    seed: int = 0,
    return_frame: bool = False,
):
    """Generate a labelled multi-temporal crop dataset for offline demos / tests.

    Each sample is a season-long NDVI trajectory (``n_timesteps`` points) plus
    SAR (VV/VH/ratio) and a structural/texture feature, derived from per-crop
    archetypes with Gaussian noise.  Phenometrics are appended so the feature
    table mirrors what :func:`build_feature_stack` produces for real data.

    Parameters
    ----------
    n_per_class
        Samples per crop class.
    n_timesteps
        NDVI observations per season.
    n_classes
        Number of crop classes (2-5, drawn from the built-in archetypes).
    noise
        Std of additive Gaussian noise on NDVI (and proportionally on SAR).
    grid_size
        If given, sample coordinates are placed on a ``grid_size x grid_size``
        lattice with class-coherent spatial blocks (for testing
        :func:`spatial_block_split`); otherwise random coordinates are used.
    seed
        RNG seed for reproducibility.
    return_frame
        If True (and pandas available), also return a ``pandas.DataFrame``.

    Returns
    -------
    dict
        ``X`` (n, n_features), ``y`` (n,), ``feature_names`` (list),
        ``coords`` (n, 2), ``time`` (n_timesteps,), ``classes`` (list),
        and optionally ``frame``.
    """
    from ..features.phenology import phenometrics

    rng = np.random.default_rng(seed)
    archetypes = _DEFAULT_ARCHETYPES[: max(2, min(n_classes, len(_DEFAULT_ARCHETYPES)))]
    time = np.linspace(0.0, 1.0, n_timesteps)
    doy = 1.0 + time * 240.0  # map season fraction -> day-of-year (~8-month season)

    rows: list[np.ndarray] = []
    labels: list[str] = []
    coords: list[tuple[float, float]] = []
    feature_names: list[str] | None = None

    n_total = len(archetypes) * n_per_class
    side = grid_size if grid_size is not None else int(np.ceil(np.sqrt(n_total))) + 1

    counter = 0
    for ci, arch in enumerate(archetypes):
        clean = _logistic_curve(time, arch)
        for _ in range(n_per_class):
            # --- NDVI time series with noise + small phase jitter ---
            phase = rng.normal(0.0, 0.02)
            amp_jit = rng.normal(1.0, 0.06)
            ndvi_ts = arch.base + (clean - arch.base) * amp_jit
            ndvi_ts = np.interp(time, np.clip(time + phase, 0, 1), ndvi_ts)
            ndvi_ts = np.clip(ndvi_ts + rng.normal(0, noise, n_timesteps), -0.1, 1.0)

            # --- SAR features (modulated by greenness; volume scattering up
            #     with biomass) ---
            green_frac = np.clip((ndvi_ts - arch.base) / max(arch.amp, 1e-3), 0, 1)
            vh = np.clip(arch.vh_mean * (1 + 0.8 * green_frac) + rng.normal(0, noise * 0.4, n_timesteps), 1e-4, None)
            vv = np.clip(arch.vv_mean * (1 + 0.4 * green_frac) + rng.normal(0, noise * 0.4, n_timesteps), 1e-4, None)
            cr = vh / vv  # cross-ratio
            rvi = 4.0 * vh / (vv + vh)

            # --- structural / texture summary ---
            texture = np.clip(arch.texture + rng.normal(0, 0.03), 0, None)

            # --- phenometrics from the NDVI trajectory ---
            ph = phenometrics(doy, ndvi_ts, n_harmonics=2, smooth_lambda=2.0)

            # --- assemble flat feature vector ---
            feat: dict[str, float] = {}
            for k in range(n_timesteps):
                feat[f"ndvi_t{k:02d}"] = float(ndvi_ts[k])
            feat["sar_vh_mean"] = float(np.mean(vh))
            feat["sar_vv_mean"] = float(np.mean(vv))
            feat["sar_cr_mean"] = float(np.mean(cr))
            feat["sar_rvi_mean"] = float(np.mean(rvi))
            feat["sar_vh_max"] = float(np.max(vh))
            feat["texture_contrast"] = float(texture)
            # Append a curated subset of phenometrics (drop noisy raw harmonics
            # phases that add little and can carry NaNs for short series).
            for key in ("sos", "pos", "eos", "lgp", "amplitude", "integral", "vi_mean", "vi_std", "vi_max", "harm_amp1", "harm_amp2"):
                feat[f"ph_{key}"] = float(ph.get(key, np.nan))

            if feature_names is None:
                feature_names = list(feat.keys())
            rows.append(np.array([feat[name] for name in feature_names], dtype=np.float64))
            labels.append(arch.name)

            # --- spatial coordinates: class-coherent blocks if grid requested ---
            if grid_size is not None:
                # Each class occupies a horizontal band of the grid.
                band_h = max(1, side // len(archetypes))
                r = ci * band_h + (counter % band_h)
                c = (counter // band_h) % side
                coords.append((float(r), float(c)))
            else:
                coords.append((float(rng.uniform(0, side)), float(rng.uniform(0, side))))
            counter += 1

    X = np.vstack(rows)
    y = np.asarray(labels, dtype=object)
    coords_arr = np.asarray(coords, dtype=np.float64)
    # NaN-fill any phenometric gaps with column medians (robust, leakage-free
    # here because this is the full synthetic set used for the demo).
    X = _impute_nan(X)

    out: dict[str, Any] = {
        "X": X,
        "y": y,
        "feature_names": feature_names,
        "coords": coords_arr,
        "time": doy,
        "classes": [a.name for a in archetypes],
    }
    if return_frame:
        try:
            import pandas as pd

            df = pd.DataFrame(X, columns=feature_names)
            df["label"] = y
            df["x"] = coords_arr[:, 0]
            df["y"] = coords_arr[:, 1]
            out["frame"] = df
        except Exception:  # pragma: no cover
            pass
    return out


def _impute_nan(X: np.ndarray) -> np.ndarray:
    """Replace NaNs/Infs column-wise with the column median (0 if all-NaN)."""
    X = np.asarray(X, dtype=np.float64).copy()
    X[~np.isfinite(X)] = np.nan
    for j in range(X.shape[1]):
        col = X[:, j]
        m = np.nanmedian(col) if np.isfinite(col).any() else 0.0
        if not np.isfinite(m):
            m = 0.0
        col[~np.isfinite(col)] = m
        X[:, j] = col
    return X


# --------------------------------------------------------------------------- #
# Feature-stack builder (for real datacubes / tables)
# --------------------------------------------------------------------------- #
def build_feature_stack(
    datacube_or_table: Any,
    *,
    band_names: Sequence[str] | None = None,
    time: Sequence[float] | None = None,
    indices: Sequence[str] | None = None,
    include_phenology: bool = True,
    vi_for_phenology: str = "ndvi",
) -> tuple[np.ndarray, list[str]]:
    """Assemble a flat ``(n_samples, n_features)`` matrix from multi-temporal input.

    Accepts either:

    * a **2-D table** (``pandas.DataFrame`` or ndarray) already in
      sample-by-feature form -> returned (numerically coerced) as-is; or
    * a **3-D datacube** ``(n_samples, n_time, n_bands)`` -> per-time-step
      spectral indices are computed via
      :func:`agristress.features.indices.stack_indices`, flattened across time,
      and (optionally) phenometrics of the chosen VI are appended.

    Parameters
    ----------
    datacube_or_table
        Table or 3-D array (see above).
    band_names
        Names for the band axis of a 3-D cube (length = ``n_bands``).  Required
        for the cube path so bands can be mapped to index inputs.
    time
        Time stamps (length = ``n_time``); defaults to ``0..n_time-1``.  Used for
        phenometrics.
    indices
        Subset of index names to compute (see ``stack_indices``); ``None`` uses
        every index whose required bands are present.
    include_phenology
        Append double-logistic + harmonic phenometrics of ``vi_for_phenology``.
    vi_for_phenology
        Which computed index drives the phenometrics (default ``"ndvi"``).

    Returns
    -------
    (X, feature_names)
    """
    from ..features.indices import stack_indices
    from ..features.phenology import phenometrics

    arr = np.asarray(datacube_or_table, dtype=np.float64) if not hasattr(datacube_or_table, "values") else None
    if arr is None:  # pandas DataFrame
        df = datacube_or_table
        names = [str(c) for c in df.columns]
        return _impute_nan(np.asarray(df.to_numpy(), dtype=np.float64)), names

    if arr.ndim == 2:
        names = [f"f{j}" for j in range(arr.shape[1])]
        return _impute_nan(arr), names

    if arr.ndim != 3:
        raise ValueError("datacube must be 2-D (table) or 3-D (samples, time, bands)")

    n_samples, n_time, n_bands = arr.shape
    if band_names is None or len(band_names) != n_bands:
        raise ValueError("band_names (length n_bands) is required for a 3-D datacube")
    t = np.asarray(time, dtype=np.float64) if time is not None else np.arange(n_time, dtype=np.float64)

    # Per-time-step index series.
    index_series: dict[str, np.ndarray] = {}
    for ti in range(n_time):
        bands = {name: arr[:, ti, bi] for bi, name in enumerate(band_names)}
        computed = stack_indices(bands, which=list(indices) if indices is not None else None)
        for iname, vals in computed.items():
            index_series.setdefault(iname, np.full((n_samples, n_time), np.nan))
            index_series[iname][:, ti] = vals

    feature_cols: list[np.ndarray] = []
    feature_names: list[str] = []
    for iname in sorted(index_series):
        series = index_series[iname]
        for ti in range(n_time):
            feature_cols.append(series[:, ti])
            feature_names.append(f"{iname}_t{ti:02d}")
        # Simple temporal stats per index (cheap, informative).
        feature_cols.extend([np.nanmean(series, axis=1), np.nanmax(series, axis=1), np.nanstd(series, axis=1)])
        feature_names.extend([f"{iname}_mean", f"{iname}_max", f"{iname}_std"])

    if include_phenology and vi_for_phenology in index_series:
        vi = index_series[vi_for_phenology]
        ph_rows = []
        for s in range(n_samples):
            ph = phenometrics(t, vi[s], n_harmonics=2)
            ph_rows.append(ph)
        ph_keys = ["sos", "pos", "eos", "lgp", "amplitude", "integral", "vi_mean", "vi_std", "vi_max", "harm_amp1", "harm_amp2"]
        for key in ph_keys:
            feature_cols.append(np.array([r.get(key, np.nan) for r in ph_rows]))
            feature_names.append(f"ph_{key}")

    X = np.column_stack(feature_cols) if feature_cols else np.empty((n_samples, 0))
    return _impute_nan(X), feature_names


# --------------------------------------------------------------------------- #
# Leakage-aware split
# --------------------------------------------------------------------------- #
def spatial_block_split(
    coords: np.ndarray,
    y: Sequence | None = None,
    *,
    test_size: float = 0.3,
    n_blocks: int = 8,
    axis: str = "x",
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Block train/test split that keeps spatially contiguous samples together.

    Coordinates are binned into ``n_blocks`` strips along ``axis`` (``"x"``,
    ``"y"`` or ``"grid"`` for a checkerboard of ``n_blocks`` x ``n_blocks``
    tiles).  Whole blocks are assigned to train or test, so neighbouring pixels
    cannot leak across the split — a realistic test of map generalisation.

    Parameters
    ----------
    coords
        ``(n_samples, 2)`` array of spatial coordinates.
    y
        Optional labels (unused for assignment; accepted for API symmetry).
    test_size
        Approximate fraction of blocks assigned to the test set.
    n_blocks
        Number of strips (or grid divisions per axis).
    axis
        ``"x"``, ``"y"`` or ``"grid"``.
    seed
        RNG seed for block assignment.

    Returns
    -------
    (train_idx, test_idx)
        Integer index arrays into the sample dimension.
    """
    coords = np.asarray(coords, dtype=np.float64)
    n = coords.shape[0]
    rng = np.random.default_rng(seed)

    def _bin(vals: np.ndarray, k: int) -> np.ndarray:
        lo, hi = float(np.min(vals)), float(np.max(vals))
        if hi <= lo:
            return np.zeros_like(vals, dtype=np.int64)
        edges = np.linspace(lo, hi, k + 1)
        b = np.clip(np.digitize(vals, edges[1:-1]), 0, k - 1)
        return b

    if axis == "x":
        block_id = _bin(coords[:, 0], n_blocks)
        n_total_blocks = n_blocks
    elif axis == "y":
        block_id = _bin(coords[:, 1], n_blocks)
        n_total_blocks = n_blocks
    elif axis == "grid":
        bx = _bin(coords[:, 0], n_blocks)
        by = _bin(coords[:, 1], n_blocks)
        block_id = bx * n_blocks + by
        n_total_blocks = n_blocks * n_blocks
    else:
        raise ValueError("axis must be 'x', 'y' or 'grid'")

    unique_blocks = np.unique(block_id)
    rng.shuffle(unique_blocks)
    n_test = max(1, int(round(len(unique_blocks) * test_size)))
    test_blocks = set(unique_blocks[:n_test].tolist())

    is_test = np.array([b in test_blocks for b in block_id])
    test_idx = np.where(is_test)[0]
    train_idx = np.where(~is_test)[0]
    # Guard degenerate splits (e.g. all samples in one block).
    if train_idx.size == 0 or test_idx.size == 0:
        perm = rng.permutation(n)
        cut = int(n * (1 - test_size))
        train_idx, test_idx = perm[:cut], perm[cut:]
    return train_idx, test_idx


# --------------------------------------------------------------------------- #
# Training driver
# --------------------------------------------------------------------------- #
@dataclass
class TrainResult:
    """Outcome of :func:`train_crop_model`."""

    model: Any
    metrics: dict
    feature_names: list[str] | None
    train_idx: np.ndarray
    test_idx: np.ndarray
    backend: str = ""
    model_path: str | None = None
    report_path: str | None = None


def train_crop_model(
    datacube_or_table: Any,
    labels: Sequence,
    *,
    model: str = "rf",
    feature_names: Sequence[str] | None = None,
    coords: np.ndarray | None = None,
    test_size: float = 0.3,
    split: str = "spatial",
    n_blocks: int = 8,
    save_dir: str | Path | None = None,
    model_name: str = "crop_model",
    random_state: int = 42,
    **model_kwargs,
) -> TrainResult:
    """Build features, fit a crop classifier, evaluate, and optionally persist.

    Parameters
    ----------
    datacube_or_table
        A 2-D feature table, a 3-D datacube (see :func:`build_feature_stack`),
        or — for ``model='embedding'`` — an ``(n_samples, d)`` embedding matrix.
    labels
        Per-sample crop labels.
    model
        ``'rf'`` / ``'lightgbm'`` / ``'xgboost'`` / ``'catboost'`` / ``'gbm'``
        for the tabular :class:`CropClassifier`, or ``'embedding'`` for the
        :class:`SatelliteEmbeddingClassifier` (cosine head).
    feature_names
        Optional feature names (for importances / reports).
    coords
        ``(n_samples, 2)`` coordinates for the spatial split.  Required when
        ``split='spatial'``; otherwise a random split is used.
    test_size, split, n_blocks
        Splitting controls.  ``split`` is ``'spatial'`` or ``'random'``.
    save_dir
        If given, persist ``<model_name>.joblib`` and ``<model_name>_metrics.json``.
    random_state
        Seed for the split and the model.
    **model_kwargs
        Forwarded to the classifier constructor.

    Returns
    -------
    TrainResult
        Fitted model, metrics dict (OA / kappa / per-class / confusion matrix),
        the train/test indices and any saved paths.
    """
    y = np.asarray(labels).ravel()

    is_embedding = model.lower() in ("embedding", "alphaearth", "satellite")
    if is_embedding:
        X = np.asarray(datacube_or_table, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("embedding model expects a 2-D (n_samples, d) matrix")
        names = feature_names if feature_names is not None else [f"emb{j}" for j in range(X.shape[1])]
        names = list(names)
    else:
        X, names = build_feature_stack(datacube_or_table)
        if feature_names is not None and len(feature_names) == X.shape[1]:
            names = list(feature_names)

    # --- split (leakage-aware where coords are available) ---
    if split == "spatial" and coords is not None:
        train_idx, test_idx = spatial_block_split(
            coords, y, test_size=test_size, n_blocks=n_blocks, axis="grid", seed=random_state
        )
    else:
        rng = np.random.default_rng(random_state)
        perm = rng.permutation(len(y))
        cut = int(len(y) * (1 - test_size))
        train_idx, test_idx = perm[:cut], perm[cut:]

    # --- model ---
    if is_embedding:
        method = model_kwargs.pop("method", "cosine")
        clf: Any = SatelliteEmbeddingClassifier(method=method, random_state=random_state, **model_kwargs)
        clf.fit(X[train_idx], y[train_idx])
        backend = f"embedding:{method}"
    else:
        clf = CropClassifier(model=model, random_state=random_state, **model_kwargs)
        clf.fit(X[train_idx], y[train_idx], feature_names=names)
        backend = clf.backend

    # --- evaluate on held-out test set ---
    y_pred = clf.predict(X[test_idx])
    metrics = _eval.classification_report_dict(y[test_idx], y_pred)
    metrics["backend"] = backend
    metrics["n_train"] = int(train_idx.size)
    metrics["n_test"] = int(test_idx.size)
    if not is_embedding:
        fi = clf.feature_importance()
        if isinstance(fi, dict):
            # Keep top-15 for a compact report.
            metrics["top_features"] = dict(list(fi.items())[:15])

    model_path = report_path = None
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        try:
            import joblib

            model_path = str(save_dir / f"{model_name}.joblib")
            joblib.dump({"model": clf, "feature_names": names, "classes": list(clf.classes_)}, model_path)
        except Exception:  # pragma: no cover - joblib should be present
            model_path = None
        report_path = str(save_dir / f"{model_name}_metrics.json")
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2, default=_json_default)

    return TrainResult(
        model=clf,
        metrics=metrics,
        feature_names=names,
        train_idx=train_idx,
        test_idx=test_idx,
        backend=backend,
        model_path=model_path,
        report_path=report_path,
    )


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
