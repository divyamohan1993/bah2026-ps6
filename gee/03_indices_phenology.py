"""AgriStress · GEE step 03 — spectral indices & phenology (SOS/EOS) on Earth Engine.

Consumes the harmonised 8-day optical composites from step 01 and produces the
vegetation/water index time-series plus per-pixel phenology metrics
(DATA_FUSION Stage 5).

Indices (per composite, common bands blue/green/red/nir/swir1/swir2)
--------------------------------------------------------------------
* **NDVI** = (nir − red) / (nir + red)                      — greenness/vigour
* **EVI**  = 2.5·(nir − red) / (nir + 6·red − 7.5·blue + 1) — high-biomass NDVI
* **NDWI** = (green − nir) / (green + nir)                  — open-water / canopy water (McFeeters)
* **NDMI** = (nir − swir1) / (nir + swir1)                  — vegetation moisture
* **STR**  = (1 − swir2)² / (2·swir2)                       — Soil Tillage/moisture (OPTRAM transform)

Phenology
---------
Two interoperable estimators of Start/End of Season on the NDVI series:

* **Harmonic regression** (1–2 harmonics + linear trend) — fast, robust, fills
  short gaps; SOS/EOS taken as up/down crossings of an amplitude threshold of
  the fitted curve. (HANTS-style, DATA_FUSION §3.1.)
* **Double-logistic** fit (TIMESAT-style, DATA_FUSION §5.1) — asymmetric
  green-up + senescence; provided as a server-side curve-evaluation helper plus
  a documented client-side fitting hook (the non-linear fit is typically done
  per-sample, not per-pixel, on GEE).

Import-safe without credentials. Run :func:`main` after ``init_ee``.
"""

from __future__ import annotations

from typing import Any

from gee._auth import EarthEngineUnavailable, init_ee

INDEX_BANDS: list[str] = ["NDVI", "EVI", "NDWI", "NDMI", "STR"]


def add_indices(ee: Any, img: Any) -> Any:
    """Add NDVI/EVI/NDWI/NDMI/STR bands to a harmonised optical image."""
    nir = img.select("nir")
    red = img.select("red")
    green = img.select("green")
    blue = img.select("blue")
    swir1 = img.select("swir1")
    swir2 = img.select("swir2")

    ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
    evi = (
        nir.subtract(red)
        .multiply(2.5)
        .divide(nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1))
        .rename("EVI")
    )
    ndwi = green.subtract(nir).divide(green.add(nir)).rename("NDWI")
    ndmi = nir.subtract(swir1).divide(nir.add(swir1)).rename("NDMI")
    # STR (OPTRAM swir transform); guard swir2→0 to avoid div-by-zero.
    str_band = (
        ee.Image(1).subtract(swir2).pow(2).divide(swir2.multiply(2).max(1e-4)).rename("STR")
    )
    return img.addBands([ndvi, evi, ndwi, ndmi, str_band])


def index_time_series(ee: Any, optical_collection: Any) -> Any:
    """Map :func:`add_indices` over a composite collection → index ImageCollection."""
    return optical_collection.map(lambda img: add_indices(ee, img).select(INDEX_BANDS).copyProperties(img, ["system:time_start"]))


# --------------------------------------------------------------------------
# Harmonic phenology
# --------------------------------------------------------------------------
def _add_harmonic_terms(ee: Any, img: Any, n_harmonics: int = 2) -> Any:
    """Add constant, linear-time and cos/sin harmonic predictor bands to an image."""
    t = ee.Image(img.date().difference(ee.Date("1970-01-01"), "year")).float().rename("t")
    img = img.addBands(ee.Image.constant(1).rename("constant")).addBands(t)
    for k in range(1, n_harmonics + 1):
        omega = 2.0 * 3.141592653589793 * k
        img = img.addBands(t.multiply(omega).cos().rename(f"cos{k}"))
        img = img.addBands(t.multiply(omega).sin().rename(f"sin{k}"))
    return img


def harmonic_phenology(ee: Any, ndvi_collection: Any, n_harmonics: int = 2) -> Any:
    """Fit a harmonic model to the NDVI series and return amplitude/phase/fit.

    Returns an image with bands: ``amplitude`` (seasonal NDVI swing),
    ``phase`` (peak timing, radians), and the regression coefficients. SOS/EOS
    can be derived from threshold crossings of the reconstructed curve.
    """
    predictors = ["constant", "t"] + [
        f"{fn}{k}" for k in range(1, n_harmonics + 1) for fn in ("cos", "sin")
    ]
    harmonic = ndvi_collection.map(lambda img: _add_harmonic_terms(ee, img, n_harmonics))
    fit = harmonic.select(predictors + ["NDVI"]).reduce(
        ee.Reducer.linearRegression(numX=len(predictors), numY=1)
    )
    coeffs = fit.select("coefficients").arrayProject([0]).arrayFlatten([predictors])

    # First-harmonic amplitude & phase (peak-greenness timing).
    cos1 = coeffs.select("cos1")
    sin1 = coeffs.select("sin1")
    amplitude = cos1.hypot(sin1).rename("amplitude")
    phase = sin1.atan2(cos1).rename("phase")
    return coeffs.addBands([amplitude, phase])


def sos_eos_from_threshold(ee: Any, ndvi_collection: Any, amp_fraction: float = 0.2) -> Any:
    """Estimate SOS/EOS day-of-year as crossings of an amplitude threshold.

    Threshold = min + ``amp_fraction``·(max − min) of the per-pixel NDVI series.
    SOS = first 8-day step above threshold; EOS = last step above threshold.
    Returns an image with ``SOS_doy`` / ``EOS_doy`` / ``LGP`` (= EOS − SOS).
    """
    def _with_doy(img: Any) -> Any:
        doy = ee.Image(img.date().getRelative("day", "year")).int().rename("doy")
        return img.addBands(doy)

    coll = ndvi_collection.map(_with_doy)
    ndvi_min = coll.select("NDVI").min()
    ndvi_max = coll.select("NDVI").max()
    threshold = ndvi_min.add(ndvi_max.subtract(ndvi_min).multiply(amp_fraction))

    def _mask_above(img: Any) -> Any:
        above = img.select("NDVI").gte(threshold)
        return img.select("doy").updateMask(above)

    above = coll.map(_mask_above)
    sos = above.select("doy").min().rename("SOS_doy")
    eos = above.select("doy").max().rename("EOS_doy")
    lgp = eos.subtract(sos).rename("LGP")
    return ee.Image.cat([sos, eos, lgp])


# --------------------------------------------------------------------------
# Double-logistic (TIMESAT-style) — server-side curve evaluation
# --------------------------------------------------------------------------
def double_logistic(ee: Any, t: Any, params: dict[str, Any]) -> Any:
    """Evaluate the double-logistic phenology model at time(s) ``t``.

    f(t) = wNDVI + (mNDVI − wNDVI) · [ 1/(1+exp(−rsp·(t−S))) − 1/(1+exp(−rau·(t−A))) ]

    where ``wNDVI``/``mNDVI`` are winter/peak NDVI, ``S``/``A`` the green-up /
    senescence inflection times, and ``rsp``/``rau`` their rates. Parameters are
    typically fit client-side per training sample; this helper reconstructs the
    smooth curve (and hence SOS/EOS) on the server once params are known.
    """
    w = ee.Image.constant(params["wNDVI"])
    m = ee.Image.constant(params["mNDVI"])
    green = ee.Image.constant(1).divide(
        ee.Image.constant(1).add(ee.Image(t).subtract(params["S"]).multiply(-params["rsp"]).exp())
    )
    senesce = ee.Image.constant(1).divide(
        ee.Image.constant(1).add(ee.Image(t).subtract(params["A"]).multiply(-params["rau"]).exp())
    )
    return w.add(m.subtract(w).multiply(green.subtract(senesce)))


def fit_double_logistic_numpy(times: Any, ndvi: Any) -> dict[str, float]:  # pragma: no cover - optional
    """Client-side double-logistic fit on a 1-D NDVI series (numpy + scipy).

    Returns the parameter dict consumed by :func:`double_logistic`. Used for
    per-sample phenology where a non-linear least-squares fit is appropriate;
    kept here so the GEE and offline paths share one model definition. Degrades
    with a clear message if scipy is unavailable.
    """
    import numpy as np

    try:
        from scipy.optimize import curve_fit  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise EarthEngineUnavailable(
            "fit_double_logistic_numpy needs scipy (pip install scipy)."
        ) from exc

    t = np.asarray(times, dtype=float)
    y = np.asarray(ndvi, dtype=float)

    def model(tt, w, m, S, rsp, A, rau):
        return w + (m - w) * (
            1.0 / (1.0 + np.exp(-rsp * (tt - S))) - 1.0 / (1.0 + np.exp(-rau * (tt - A)))
        )

    p0 = [float(np.nanmin(y)), float(np.nanmax(y)), float(t.mean() - t.ptp() / 4),
          0.1, float(t.mean() + t.ptp() / 4), 0.1]
    popt, _ = curve_fit(model, t, y, p0=p0, maxfev=20000)
    keys = ["wNDVI", "mNDVI", "S", "rsp", "A", "rau"]
    return dict(zip(keys, [float(v) for v in popt]))


def main(aoi: Any = None, start: str = "2023-06-01", end: str = "2023-11-30", project: str | None = None) -> Any:
    """Build index time-series + harmonic/threshold phenology. Requires EE auth."""
    ee = init_ee(project)
    if aoi is None:
        aoi = [76.30, 30.60, 76.55, 30.80]

    # Lazy import of step 01 by file path (numeric module name).
    import importlib.util
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location("gee_01", os.path.join(here, "01_optical_harmonize.py"))
    opt = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(opt)  # type: ignore[union-attr]

    merged = opt.build_harmonized_collection(ee, aoi, start, end)
    comps = opt.to_8day_composites(ee, merged, start, end)
    idx = index_time_series(ee, comps)
    pheno = sos_eos_from_threshold(ee, idx)
    try:
        print(f"[gee/03] index bands: {INDEX_BANDS}; phenology bands: SOS_doy/EOS_doy/LGP")
        print(f"[gee/03] index series length: {idx.size().getInfo()} composites")
    except Exception as exc:  # pragma: no cover
        print(f"[gee/03] built index/phenology (getInfo skipped: {exc})")
    return {"indices": idx, "phenology": pheno}


if __name__ == "__main__":  # pragma: no cover
    import os
    import sys

    try:
        main(project=os.environ.get("EE_PROJECT"))
    except EarthEngineUnavailable as exc:
        print(str(exc))
        sys.exit(1)
