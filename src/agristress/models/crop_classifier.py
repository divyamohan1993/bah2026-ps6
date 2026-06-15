"""Crop-type classifiers.

Two complementary classifiers:

* :class:`CropClassifier` -- a tabular classifier over the multi-temporal
  feature stack (spectral indices + SAR + texture + phenometrics).  Defaults to
  scikit-learn's ``RandomForestClassifier`` and transparently upgrades to a
  gradient-boosting backend (XGBoost / LightGBM / CatBoost) when installed and
  requested.  Class imbalance is handled via ``class_weight='balanced'`` (RF /
  sklearn-GBM) or per-sample weights (boosting backends).

* :class:`SatelliteEmbeddingClassifier` -- consumes pre-computed 64-D satellite
  embeddings (e.g. Google AlphaEarth annual embeddings) and classifies either
  with a light learned head (kNN / logistic / RF) or by nearest-class
  **cosine similarity** to per-class mean embeddings (a strong, label-efficient
  baseline that needs no heavy training).

Both expose a common ``fit / predict / predict_proba`` API so the training
driver can swap them freely.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = [
    "CropClassifier",
    "SatelliteEmbeddingClassifier",
    "available_backends",
]


def _try_import(name: str):
    try:
        return __import__(name)
    except Exception:
        return None


def available_backends() -> dict[str, bool]:
    """Report which optional gradient-boosting backends are importable."""
    return {
        "sklearn": _try_import("sklearn") is not None,
        "xgboost": _try_import("xgboost") is not None,
        "lightgbm": _try_import("lightgbm") is not None,
        "catboost": _try_import("catboost") is not None,
    }


class CropClassifier:
    """Tabular crop-type classifier with pluggable backends.

    Parameters
    ----------
    model
        One of ``"rf"`` / ``"random_forest"``, ``"xgboost"``, ``"lightgbm"``,
        ``"catboost"``, or ``"gbm"`` (sklearn HistGradientBoosting).  If the
        requested boosting backend is unavailable, falls back to RandomForest
        and records the substitution in :attr:`backend`.
    n_estimators, max_depth, learning_rate, random_state
        Common hyper-parameters forwarded to the chosen backend where relevant.
    class_weight
        ``"balanced"`` (default) compensates for class imbalance.  For boosting
        backends without a native ``class_weight`` it is translated into
        per-sample weights at ``fit`` time.
    **kwargs
        Extra backend-specific keyword arguments.
    """

    def __init__(
        self,
        model: str = "rf",
        *,
        n_estimators: int = 300,
        max_depth: int | None = None,
        learning_rate: float = 0.1,
        random_state: int = 42,
        class_weight: str | dict | None = "balanced",
        **kwargs,
    ) -> None:
        self.requested_model = model
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.class_weight = class_weight
        self.kwargs = kwargs

        self.backend: str = ""
        self.estimator = None
        self.classes_: np.ndarray | None = None
        self.feature_names_: list[str] | None = None
        self._build()

    # -- construction --------------------------------------------------------
    def _build(self) -> None:
        m = self.requested_model.lower()
        if m in ("rf", "random_forest", "randomforest"):
            self._build_rf()
        elif m in ("xgboost", "xgb"):
            self._build_xgb()
        elif m in ("lightgbm", "lgbm", "lgb"):
            self._build_lgbm()
        elif m in ("catboost", "cat"):
            self._build_catboost()
        elif m in ("gbm", "histgbm", "hist_gradient_boosting"):
            self._build_hist_gbm()
        else:
            raise ValueError(f"unknown model '{self.requested_model}'")

    def _build_rf(self) -> None:
        from sklearn.ensemble import RandomForestClassifier

        self.estimator = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=-1,
            **self.kwargs,
        )
        self.backend = "random_forest"

    def _build_hist_gbm(self) -> None:
        from sklearn.ensemble import HistGradientBoostingClassifier

        self.estimator = HistGradientBoostingClassifier(
            max_iter=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            **self.kwargs,
        )
        self.backend = "hist_gradient_boosting"

    def _build_xgb(self) -> None:
        xgb = _try_import("xgboost")
        if xgb is None:
            self._build_rf()
            self.backend = "random_forest (xgboost unavailable)"
            return
        from xgboost import XGBClassifier

        self.estimator = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth or 6,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            tree_method="hist",
            eval_metric="mlogloss",
            n_jobs=-1,
            **self.kwargs,
        )
        self.backend = "xgboost"

    def _build_lgbm(self) -> None:
        lgb = _try_import("lightgbm")
        if lgb is None:
            self._build_rf()
            self.backend = "random_forest (lightgbm unavailable)"
            return
        from lightgbm import LGBMClassifier

        self.estimator = LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth or -1,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            class_weight=self.class_weight,
            n_jobs=-1,
            verbose=-1,
            **self.kwargs,
        )
        self.backend = "lightgbm"

    def _build_catboost(self) -> None:
        cat = _try_import("catboost")
        if cat is None:
            self._build_rf()
            self.backend = "random_forest (catboost unavailable)"
            return
        from catboost import CatBoostClassifier

        self.estimator = CatBoostClassifier(
            iterations=self.n_estimators,
            depth=self.max_depth or 6,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            auto_class_weights="Balanced" if self.class_weight == "balanced" else None,
            verbose=False,
            **self.kwargs,
        )
        self.backend = "catboost"

    # -- helpers -------------------------------------------------------------
    def _sample_weights(self, y: np.ndarray) -> np.ndarray | None:
        """Balanced per-sample weights for backends lacking ``class_weight``."""
        if self.class_weight != "balanced":
            return None
        try:
            from sklearn.utils.class_weight import compute_sample_weight

            return compute_sample_weight("balanced", y)
        except Exception:  # pragma: no cover
            classes, counts = np.unique(y, return_counts=True)
            freq = {c: n for c, n in zip(classes, counts)}
            n_total = len(y)
            n_cls = len(classes)
            return np.array([n_total / (n_cls * freq[v]) for v in y], dtype=np.float64)

    @staticmethod
    def _as_2d(X) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr

    # -- API -----------------------------------------------------------------
    def fit(self, X, y, feature_names: Sequence[str] | None = None) -> "CropClassifier":
        """Fit the classifier on feature matrix ``X`` and labels ``y``."""
        Xa = self._as_2d(X)
        ya = np.asarray(y).ravel()
        self.feature_names_ = list(feature_names) if feature_names is not None else None

        needs_sw = self.backend.startswith("xgboost")
        if needs_sw:
            sw = self._sample_weights(ya)
            # XGBoost needs 0..K-1 integer labels; use a LabelEncoder.
            from sklearn.preprocessing import LabelEncoder

            self._le = LabelEncoder().fit(ya)
            self.estimator.fit(Xa, self._le.transform(ya), sample_weight=sw)
            self.classes_ = self._le.classes_
        else:
            self._le = None
            self.estimator.fit(Xa, ya)
            self.classes_ = np.asarray(getattr(self.estimator, "classes_", np.unique(ya)))
        return self

    def predict(self, X) -> np.ndarray:
        """Predict crop-type labels for ``X``."""
        Xa = self._as_2d(X)
        pred = self.estimator.predict(Xa)
        if getattr(self, "_le", None) is not None:
            pred = self._le.inverse_transform(np.asarray(pred).astype(int))
        return np.asarray(pred)

    def predict_proba(self, X) -> np.ndarray:
        """Class-probability matrix ``(n_samples, n_classes)`` aligned to ``classes_``."""
        Xa = self._as_2d(X)
        if hasattr(self.estimator, "predict_proba"):
            return np.asarray(self.estimator.predict_proba(Xa))
        # Fallback: one-hot of hard predictions.
        pred = self.predict(Xa)
        classes = list(self.classes_)
        out = np.zeros((len(pred), len(classes)))
        for i, p in enumerate(pred):
            out[i, classes.index(p)] = 1.0
        return out

    def feature_importance(self) -> dict[str, float] | np.ndarray:
        """Feature importances, as a ``{name: value}`` dict if names are known.

        Uses the backend's native ``feature_importances_`` (impurity / gain).
        For models lacking it (e.g. some configurations), returns an empty dict.
        """
        imp = getattr(self.estimator, "feature_importances_", None)
        if imp is None:
            return {}
        imp = np.asarray(imp, dtype=np.float64)
        if self.feature_names_ is not None and len(self.feature_names_) == len(imp):
            order = np.argsort(imp)[::-1]
            return {self.feature_names_[i]: float(imp[i]) for i in order}
        return imp


class SatelliteEmbeddingClassifier:
    """Classifier over fixed-length satellite embedding vectors (e.g. AlphaEarth 64-D).

    Two interchangeable heads:

    * ``method="cosine"`` (default) -- build a per-class **centroid** in
      (optionally L2-normalised) embedding space and assign each sample to the
      class with the highest cosine similarity.  No iterative training; ideal
      when labels are scarce but embeddings are informative.

    * ``method in {"knn", "logistic", "rf"}`` -- fit a light scikit-learn head
      (``KNeighborsClassifier`` with cosine metric / ``LogisticRegression`` /
      ``RandomForestClassifier``) on the embeddings.

    Parameters
    ----------
    method
        Classification head (see above).
    n_neighbors
        Neighbours for the kNN head.
    normalize
        L2-normalise embeddings before similarity / fitting (recommended for
        cosine geometry).
    random_state
        Seed for the learned heads.
    """

    def __init__(
        self,
        method: str = "cosine",
        *,
        n_neighbors: int = 5,
        normalize: bool = True,
        random_state: int = 42,
        **kwargs,
    ) -> None:
        self.method = method.lower()
        self.n_neighbors = n_neighbors
        self.normalize = normalize
        self.random_state = random_state
        self.kwargs = kwargs

        self.classes_: np.ndarray | None = None
        self._centroids: np.ndarray | None = None
        self._head = None

    @staticmethod
    def _l2(X: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(X, axis=1, keepdims=True)
        norm = np.where(norm < 1e-12, 1.0, norm)
        return X / norm

    def _prep(self, X) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return self._l2(arr) if self.normalize else arr

    def fit(self, X, y) -> "SatelliteEmbeddingClassifier":
        """Fit centroids or the light head on embeddings ``X`` with labels ``y``."""
        Xa = self._prep(X)
        ya = np.asarray(y).ravel()
        self.classes_ = np.unique(ya)

        if self.method == "cosine":
            cents = []
            for c in self.classes_:
                m = Xa[ya == c].mean(axis=0)
                cents.append(m)
            cents = np.vstack(cents)
            self._centroids = self._l2(cents) if self.normalize else cents
            return self

        if self.method == "knn":
            from sklearn.neighbors import KNeighborsClassifier

            self._head = KNeighborsClassifier(
                n_neighbors=min(self.n_neighbors, len(ya)),
                metric="cosine",
                **self.kwargs,
            )
        elif self.method == "logistic":
            from sklearn.linear_model import LogisticRegression

            self._head = LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=self.random_state,
                **self.kwargs,
            )
        elif self.method in ("rf", "random_forest"):
            from sklearn.ensemble import RandomForestClassifier

            self._head = RandomForestClassifier(
                n_estimators=self.kwargs.pop("n_estimators", 200),
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=-1,
                **self.kwargs,
            )
        else:
            raise ValueError(f"unknown embedding method '{self.method}'")

        self._head.fit(Xa, ya)
        self.classes_ = np.asarray(self._head.classes_)
        return self

    def _cosine_scores(self, Xa: np.ndarray) -> np.ndarray:
        # With both sides L2-normalised this is the cosine similarity matrix.
        return Xa @ self._centroids.T

    def predict(self, X) -> np.ndarray:
        """Predict labels for embedding rows ``X``."""
        Xa = self._prep(X)
        if self.method == "cosine":
            scores = self._cosine_scores(Xa)
            idx = np.argmax(scores, axis=1)
            return self.classes_[idx]
        return np.asarray(self._head.predict(Xa))

    def predict_proba(self, X) -> np.ndarray:
        """Class scores as probabilities (softmax of cosine sims for the cosine head)."""
        Xa = self._prep(X)
        if self.method == "cosine":
            scores = self._cosine_scores(Xa)
            # Softmax over similarities -> pseudo-probabilities.
            scores = scores - scores.max(axis=1, keepdims=True)
            ex = np.exp(scores)
            return ex / ex.sum(axis=1, keepdims=True)
        if hasattr(self._head, "predict_proba"):
            return np.asarray(self._head.predict_proba(Xa))
        pred = self.predict(Xa)
        classes = list(self.classes_)
        out = np.zeros((len(pred), len(classes)))
        for i, p in enumerate(pred):
            out[i, classes.index(p)] = 1.0
        return out
