"""Accuracy-assessment utilities for crop-type maps.

Standard confusion-matrix metrics (overall accuracy, Cohen's kappa, per-class
precision / recall / F1) plus the **Olofsson et al. (2014)** area-adjusted
accuracy and stratified-area estimates with 95 % confidence intervals — the
recommended good-practice protocol for remote-sensing map validation.

Functions accept either label vectors (``y_true``, ``y_pred``) or a
pre-computed confusion matrix.  scikit-learn is used where convenient but the
Olofsson estimators are implemented directly so they remain exact and
inspectable.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = [
    "confusion_matrix",
    "overall_accuracy",
    "cohen_kappa",
    "per_class_metrics",
    "classification_report_dict",
    "olofsson_accuracy",
]


def _labels_from(y_true: Sequence, y_pred: Sequence, labels: Sequence | None) -> np.ndarray:
    if labels is not None:
        return np.asarray(labels)
    uniq = np.unique(np.concatenate([np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()]))
    return uniq


def confusion_matrix(
    y_true: Sequence,
    y_pred: Sequence,
    labels: Sequence | None = None,
) -> np.ndarray:
    """Confusion matrix with rows = true (reference), cols = predicted (map).

    This row/col convention matches the Olofsson estimators below.  Uses
    scikit-learn when available, else a numpy implementation.
    """
    labels = _labels_from(y_true, y_pred, labels)
    try:
        from sklearn.metrics import confusion_matrix as _skcm

        return np.asarray(_skcm(y_true, y_pred, labels=labels))
    except Exception:  # pragma: no cover
        index = {lab: i for i, lab in enumerate(labels)}
        cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
        for t, p in zip(np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()):
            if t in index and p in index:
                cm[index[t], index[p]] += 1
        return cm


def overall_accuracy(y_true: Sequence, y_pred: Sequence) -> float:
    """Overall accuracy = fraction of samples classified correctly."""
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    if yt.size == 0:
        return float("nan")
    return float(np.mean(yt == yp))


def cohen_kappa(y_true: Sequence, y_pred: Sequence) -> float:
    """Cohen's kappa coefficient of agreement (chance-corrected)."""
    cm = confusion_matrix(y_true, y_pred).astype(np.float64)
    n = cm.sum()
    if n == 0:
        return float("nan")
    po = np.trace(cm) / n
    row = cm.sum(axis=1)
    col = cm.sum(axis=0)
    pe = float(np.sum(row * col)) / (n * n)
    if abs(1.0 - pe) < 1e-12:
        return 1.0 if po == 1.0 else 0.0
    return float((po - pe) / (1.0 - pe))


def per_class_metrics(
    y_true: Sequence,
    y_pred: Sequence,
    labels: Sequence | None = None,
) -> dict:
    """Per-class precision, recall, F1 and support (+ macro / weighted means).

    Computed from the confusion matrix so it is consistent with
    :func:`confusion_matrix` and works without scikit-learn.
    """
    labels = _labels_from(y_true, y_pred, labels)
    cm = confusion_matrix(y_true, y_pred, labels=labels).astype(np.float64)
    tp = np.diag(cm)
    pred_sum = cm.sum(axis=0)  # column totals -> predicted per class
    true_sum = cm.sum(axis=1)  # row totals -> reference per class

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(pred_sum > 0, tp / pred_sum, 0.0)
        recall = np.where(true_sum > 0, tp / true_sum, 0.0)
        denom = precision + recall
        f1 = np.where(denom > 0, 2 * precision * recall / denom, 0.0)

    per_class = {}
    for i, lab in enumerate(labels):
        per_class[str(lab)] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(true_sum[i]),
        }

    total = true_sum.sum()
    weighted = total > 0
    result = {
        "per_class": per_class,
        "macro": {
            "precision": float(np.mean(precision)),
            "recall": float(np.mean(recall)),
            "f1": float(np.mean(f1)),
        },
        "weighted": {
            "precision": float(np.sum(precision * true_sum) / total) if weighted else 0.0,
            "recall": float(np.sum(recall * true_sum) / total) if weighted else 0.0,
            "f1": float(np.sum(f1 * true_sum) / total) if weighted else 0.0,
        },
    }
    return result


def classification_report_dict(
    y_true: Sequence,
    y_pred: Sequence,
    labels: Sequence | None = None,
) -> dict:
    """A single dict bundling OA, kappa, the confusion matrix and per-class metrics.

    Designed to be JSON-serialisable (numpy arrays converted to lists) for the
    metrics report written by :func:`agristress.models.train.train_crop_model`.
    """
    labels = _labels_from(y_true, y_pred, labels)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pcm = per_class_metrics(y_true, y_pred, labels=labels)
    return {
        "overall_accuracy": overall_accuracy(y_true, y_pred),
        "cohen_kappa": cohen_kappa(y_true, y_pred),
        "n_samples": int(np.asarray(y_true).size),
        "labels": [str(x) for x in labels],
        "confusion_matrix": cm.tolist(),
        "per_class": pcm["per_class"],
        "macro": pcm["macro"],
        "weighted": pcm["weighted"],
    }


def olofsson_accuracy(
    cm: np.ndarray,
    mapped_area_fractions: Sequence[float],
    *,
    z: float = 1.96,
) -> dict:
    """Area-adjusted accuracy & area estimates (Olofsson et al. 2014).

    Converts a sample-count confusion matrix into an **area-weighted** error
    matrix using the mapped class-area proportions ``W_i``, then derives:

    * overall accuracy with 95 % CI,
    * per-class user's & producer's accuracy with 95 % CIs,
    * stratified (error-adjusted) area estimates with 95 % CIs.

    Parameters
    ----------
    cm
        Confusion matrix of **sample counts**, rows = map (predicted) class,
        cols = reference (true) class.  (This is the orientation used in the
        Olofsson paper: rows are the strata you sampled from.)
    mapped_area_fractions
        Proportion of the total mapped area assigned to each map class
        ``W_i`` (rows of ``cm``).  Need not sum exactly to 1 (it is
        renormalised); length must equal ``cm.shape[0]``.
    z
        Normal critical value for the confidence interval (1.96 -> 95 %).

    Returns
    -------
    dict
        ``overall_accuracy``, ``overall_accuracy_ci``,
        ``users_accuracy`` / ``producers_accuracy`` (arrays) and their ``_ci``,
        ``area_proportion`` (error-adjusted) and ``area_proportion_ci``.

    Notes
    -----
    The CI half-width for OA follows Eq. 5 of Olofsson et al. (2014):
    ``SE(O) = sqrt( sum_i W_i^2 * U_i * (1 - U_i) / (n_i - 1) )``.
    """
    cm = np.asarray(cm, dtype=np.float64)
    q = cm.shape[0]
    if cm.shape[0] != cm.shape[1]:
        raise ValueError("confusion matrix must be square")
    w = np.asarray(mapped_area_fractions, dtype=np.float64)
    if w.size != q:
        raise ValueError("mapped_area_fractions length must match cm rows")
    w_sum = w.sum()
    if w_sum <= 0:
        raise ValueError("mapped_area_fractions must sum to a positive value")
    w = w / w_sum

    n_i = cm.sum(axis=1)  # samples per map (row) class
    # Guard empty rows: a class with no samples contributes nothing.
    safe_ni = np.where(n_i > 0, n_i, 1.0)

    # Area-weighted error-matrix cell proportions  p_ij = W_i * n_ij / n_i.
    p = (w[:, None] * cm) / safe_ni[:, None]
    p = np.where(n_i[:, None] > 0, p, 0.0)

    p_dot_j = p.sum(axis=0)  # column sums -> reference class proportions
    p_i_dot = p.sum(axis=1)  # row sums (== W_i)
    diag = np.diag(p)

    # --- Overall accuracy (Eq. 1) and its SE (Eq. 5) ---
    oa = float(np.sum(diag))
    users = np.divide(diag, p_i_dot, out=np.full(q, np.nan), where=p_i_dot > 0)
    producers = np.divide(diag, p_dot_j, out=np.full(q, np.nan), where=p_dot_j > 0)

    var_oa = 0.0
    for i in range(q):
        if n_i[i] > 1 and np.isfinite(users[i]):
            var_oa += (w[i] ** 2) * users[i] * (1.0 - users[i]) / (n_i[i] - 1.0)
    se_oa = float(np.sqrt(var_oa))

    # --- User's accuracy CI (Eq. 6): SE(U_i) = sqrt(U_i(1-U_i)/(n_i-1)) ---
    se_users = np.full(q, np.nan)
    for i in range(q):
        if n_i[i] > 1 and np.isfinite(users[i]):
            se_users[i] = np.sqrt(users[i] * (1.0 - users[i]) / (n_i[i] - 1.0))

    # --- Producer's accuracy CI (Eq. 7) ---
    n_j = cm.sum(axis=0)  # reference counts per class (column sums of counts)
    n_dot_j_hat = np.zeros(q)
    for j in range(q):
        # Estimated total reference area in class j (denominator of N_.j-hat).
        n_dot_j_hat[j] = np.sum(np.where(n_i > 0, w * cm[:, j] / safe_ni, 0.0))
    se_producers = np.full(q, np.nan)
    for j in range(q):
        if not np.isfinite(producers[j]) or n_dot_j_hat[j] <= 0:
            continue
        nj_hat_area = n_dot_j_hat[j]
        term1 = 0.0
        if n_j[j] > 1:
            njj = cm[j, j]
            term1 = (w[j] ** 2) * (1.0 - producers[j]) ** 2 * users[j] * (1.0 - users[j]) / (safe_ni[j] - 1.0) if n_i[j] > 1 and np.isfinite(users[j]) else 0.0
        term2 = 0.0
        for i in range(q):
            if i == j or n_i[i] <= 1:
                continue
            nij = cm[i, j]
            uij = nij / safe_ni[i]
            term2 += (w[i] ** 2) * uij * (1.0 - uij) / (safe_ni[i] - 1.0)
        var_pj = (1.0 / (nj_hat_area ** 2)) * (term1 + (producers[j] ** 2) * term2)
        se_producers[j] = np.sqrt(max(var_pj, 0.0))

    # --- Error-adjusted area proportions (Eq. 9-10) ---
    area_prop = p_dot_j.copy()
    se_area = np.zeros(q)
    for j in range(q):
        acc = 0.0
        for i in range(q):
            if n_i[i] <= 1:
                continue
            pij = p[i, j]
            acc += (w[i] * pij - pij ** 2) / (safe_ni[i] - 1.0)
        se_area[j] = np.sqrt(max(acc, 0.0))

    return {
        "overall_accuracy": oa,
        "overall_accuracy_se": se_oa,
        "overall_accuracy_ci": (oa - z * se_oa, oa + z * se_oa),
        "users_accuracy": users,
        "users_accuracy_se": se_users,
        "users_accuracy_ci": np.column_stack([users - z * se_users, users + z * se_users]),
        "producers_accuracy": producers,
        "producers_accuracy_se": se_producers,
        "producers_accuracy_ci": np.column_stack([producers - z * se_producers, producers + z * se_producers]),
        "area_proportion": area_prop,
        "area_proportion_se": se_area,
        "area_proportion_ci": np.column_stack([area_prop - z * se_area, area_prop + z * se_area]),
        "weights": w,
    }
