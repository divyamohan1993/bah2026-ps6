"""GLCM (Grey-Level Co-occurrence Matrix) texture features.

Texture captures the spatial arrangement of grey levels and is a strong
discriminator between structurally different crops (e.g. orchard vs. paddy)
that may share similar mean spectra.

Primary path uses :func:`skimage.feature.graycomatrix` /
:func:`skimage.feature.graycoprops` when scikit-image is installed.  A compact,
dependency-free numpy fallback computes the same Haralick statistics so the
module is importable and usable without scikit-image.

The four headline properties returned are **contrast**, **homogeneity**,
**entropy** and **correlation** (Haralick et al. 1973); energy/ASM and
dissimilarity are also available.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

__all__ = [
    "quantize",
    "graycomatrix_np",
    "glcm_props",
    "glcm_features",
    "HAS_SKIMAGE",
]

try:  # optional accelerator
    from skimage.feature import graycomatrix as _sk_graycomatrix
    from skimage.feature import graycoprops as _sk_graycoprops

    HAS_SKIMAGE = True
except Exception:  # pragma: no cover - exercised only when skimage missing
    _sk_graycomatrix = None
    _sk_graycoprops = None
    HAS_SKIMAGE = False

# Properties skimage can compute directly; entropy is handled separately.
_SK_PROPS = ("contrast", "homogeneity", "energy", "correlation", "dissimilarity", "ASM")


def quantize(image: np.ndarray, levels: int = 16) -> np.ndarray:
    """Linearly rescale an image to integer grey levels ``[0, levels-1]``.

    NaNs are mapped to level 0.  A constant image maps to all zeros.  This is
    required before building a GLCM so the matrix stays a manageable
    ``levels x levels``.
    """
    if levels < 2:
        raise ValueError("levels must be >= 2")
    img = np.asarray(image, dtype=np.float64)
    finite = np.isfinite(img)
    if not finite.any():
        return np.zeros(img.shape, dtype=np.int64)
    lo = float(np.min(img[finite]))
    hi = float(np.max(img[finite]))
    if hi <= lo:
        return np.zeros(img.shape, dtype=np.int64)
    scaled = (img - lo) / (hi - lo) * (levels - 1)
    scaled = np.where(finite, scaled, 0.0)
    q = np.rint(scaled).astype(np.int64)
    return np.clip(q, 0, levels - 1)


# Offsets for the four canonical directions (row, col) at a given distance.
def _direction_offsets(distance: int) -> dict[int, tuple[int, int]]:
    return {
        0: (0, distance),  # horizontal  (0 deg)
        45: (-distance, distance),  # diagonal     (45 deg)
        90: (-distance, 0),  # vertical    (90 deg)
        135: (-distance, -distance),  # anti-diag   (135 deg)
    }


def graycomatrix_np(
    image: np.ndarray,
    *,
    levels: int = 16,
    distance: int = 1,
    angles: Iterable[int] = (0, 45, 90, 135),
    symmetric: bool = True,
    normed: bool = True,
    already_quantized: bool = False,
) -> np.ndarray:
    """Pure-numpy grey-level co-occurrence matrix.

    Returns an array of shape ``(levels, levels, n_angles)``.  Mirrors the
    semantics of :func:`skimage.feature.graycomatrix` for the directions used
    here (results are averaged over angles by :func:`glcm_features`).
    """
    q = image if already_quantized else quantize(image, levels)
    q = np.asarray(q, dtype=np.int64)
    if q.ndim != 2:
        raise ValueError("image must be 2-D")
    offsets = _direction_offsets(distance)
    angle_list = list(angles)
    out = np.zeros((levels, levels, len(angle_list)), dtype=np.float64)

    rows, cols = q.shape
    for k, ang in enumerate(angle_list):
        dr, dc = offsets[ang]
        # Source window and its shifted neighbour, clipped to valid overlap.
        r0_s, r1_s = max(0, -dr), rows - max(0, dr)
        c0_s, c1_s = max(0, -dc), cols - max(0, dc)
        if r1_s <= r0_s or c1_s <= c0_s:
            continue
        src = q[r0_s:r1_s, c0_s:c1_s]
        nbr = q[r0_s + dr : r1_s + dr, c0_s + dc : c1_s + dc]
        idx = src.ravel() * levels + nbr.ravel()
        counts = np.bincount(idx, minlength=levels * levels)
        glcm = counts.reshape(levels, levels).astype(np.float64)
        if symmetric:
            glcm = glcm + glcm.T
        out[:, :, k] = glcm

    if normed:
        sums = out.sum(axis=(0, 1), keepdims=True)
        sums[sums == 0] = 1.0
        out = out / sums
    return out


def _props_from_glcm(glcm: np.ndarray) -> dict[str, float]:
    """Compute Haralick statistics from a single normalised GLCM (LxL)."""
    levels = glcm.shape[0]
    i = np.arange(levels, dtype=np.float64)[:, None]
    j = np.arange(levels, dtype=np.float64)[None, :]
    p = glcm
    total = p.sum()
    if total <= 0:
        return {k: float("nan") for k in ("contrast", "homogeneity", "energy", "correlation", "dissimilarity", "ASM", "entropy")}
    p = p / total  # ensure normalised

    diff = i - j
    contrast = float(np.sum(p * diff * diff))
    dissimilarity = float(np.sum(p * np.abs(diff)))
    homogeneity = float(np.sum(p / (1.0 + diff * diff)))
    asm = float(np.sum(p * p))
    energy = float(np.sqrt(asm))
    nz = p > 0
    entropy = float(-np.sum(p[nz] * np.log(p[nz])))

    mu_i = float(np.sum(i * p))
    mu_j = float(np.sum(j * p))
    var_i = float(np.sum(((i - mu_i) ** 2) * p))
    var_j = float(np.sum(((j - mu_j) ** 2) * p))
    denom = np.sqrt(var_i * var_j)
    if denom < 1e-12:
        # No grey-level variation -> correlation undefined; convention: 1.0.
        correlation = 1.0
    else:
        cov = float(np.sum((i - mu_i) * (j - mu_j) * p))
        correlation = float(cov / denom)

    return {
        "contrast": contrast,
        "dissimilarity": dissimilarity,
        "homogeneity": homogeneity,
        "energy": energy,
        "ASM": asm,
        "correlation": correlation,
        "entropy": entropy,
    }


def glcm_props(glcm: np.ndarray) -> dict[str, float]:
    """Average Haralick properties over the angle axis of a GLCM stack.

    ``glcm`` is ``(levels, levels, n_angles)`` (as returned by
    :func:`graycomatrix_np` or skimage).  Returns the angle-averaged
    rotation-invariant statistics.
    """
    glcm = np.asarray(glcm, dtype=np.float64)
    if glcm.ndim == 2:
        glcm = glcm[:, :, None]
    n_ang = glcm.shape[2]
    acc: dict[str, float] = {}
    for k in range(n_ang):
        props = _props_from_glcm(glcm[:, :, k])
        for key, val in props.items():
            acc[key] = acc.get(key, 0.0) + val
    return {key: val / n_ang for key, val in acc.items()}


def glcm_features(
    image: np.ndarray,
    *,
    levels: int = 16,
    distance: int = 1,
    angles: Iterable[int] = (0, 45, 90, 135),
    prefix: str = "glcm_",
) -> dict[str, float]:
    """Compute angle-averaged GLCM texture features for a 2-D image patch.

    Returns a flat ``{name: value}`` dict containing at least ``contrast``,
    ``homogeneity``, ``entropy`` and ``correlation`` (plus ``energy``, ``ASM``,
    ``dissimilarity``), keyed with ``prefix``.

    Uses scikit-image if available, otherwise the numpy fallback.  Output keys
    and value semantics are identical across both paths.
    """
    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError("glcm_features expects a 2-D image patch")
    angle_list = list(angles)

    if HAS_SKIMAGE:
        q = quantize(img, levels)
        rad = [np.deg2rad(a) for a in angle_list]
        glcm = _sk_graycomatrix(
            q.astype(np.uint8),
            distances=[distance],
            angles=rad,
            levels=levels,
            symmetric=True,
            normed=True,
        )
        feats: dict[str, float] = {}
        for prop in ("contrast", "homogeneity", "energy", "correlation", "dissimilarity", "ASM"):
            # graycoprops -> shape (n_distances, n_angles); average over angles.
            feats[prop] = float(np.mean(_sk_graycoprops(glcm, prop)))
        # Entropy is not provided by graycoprops; compute from the matrices.
        ent = 0.0
        for a in range(glcm.shape[3]):
            p = glcm[:, :, 0, a]
            nz = p > 0
            ent += float(-np.sum(p[nz] * np.log(p[nz])))
        feats["entropy"] = ent / glcm.shape[3]
    else:  # pure-numpy fallback
        glcm = graycomatrix_np(
            img,
            levels=levels,
            distance=distance,
            angles=angle_list,
            symmetric=True,
            normed=True,
        )
        feats = glcm_props(glcm)

    return {f"{prefix}{k}": v for k, v in feats.items()}
