"""Offline tests for the ``gee/`` Earth Engine scripts and demo notebooks.

These tests run with **no Earth Engine and no network**. They assert that:

* every ``gee/*.py`` script is syntactically valid (``py_compile``);
* every ``gee/*.py`` script imports cleanly even when ``earthengine-api`` is
  absent (i.e. ``import ee`` is properly guarded and all EE calls are deferred
  into functions);
* the two demo notebooks are valid JSON notebooks (parseable, correct top-level
  keys, well-formed cells);
* ``gee/05_crop_classification.py`` references the AlphaEarth Satellite Embedding
  asset id and computes an ``errorMatrix`` with Cohen's Kappa.

Only the standard library + pytest are required; ``ee`` / ``nbformat`` are NOT
imported, so the suite passes in a minimal environment.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import py_compile
import sys
from pathlib import Path

import pytest

# --- locate gee/ and notebooks/ relative to the repo root -------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
GEE_DIR = REPO_ROOT / "gee"
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"

# Ensure the repo root is importable so `import gee._auth` resolves (PEP-420
# namespace package — there is no gee/__init__.py).
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _gee_scripts() -> list[Path]:
    return sorted(GEE_DIR.glob("*.py"))


def _numbered_scripts() -> list[Path]:
    """The pipeline step scripts whose module names start with a digit."""
    return [p for p in _gee_scripts() if p.name[0].isdigit()]


# ---------------------------------------------------------------------------
# Directory / file presence
# ---------------------------------------------------------------------------
def test_gee_dir_exists() -> None:
    assert GEE_DIR.is_dir(), "gee/ directory is missing"
    assert _gee_scripts(), "gee/ contains no .py scripts"


@pytest.mark.parametrize(
    "fname",
    [
        "00_auth.py",
        "01_optical_harmonize.py",
        "02_sar_s1.py",
        "03_indices_phenology.py",
        "04_soil_moisture.py",
        "05_crop_classification.py",
        "06_et0_advisory.py",
    ],
)
def test_expected_scripts_present(fname: str) -> None:
    assert (GEE_DIR / fname).is_file(), f"expected gee/{fname} to exist"


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("script", _gee_scripts(), ids=lambda p: p.name)
def test_gee_script_compiles(script: Path) -> None:
    """Every gee/*.py must be syntactically valid."""
    try:
        py_compile.compile(str(script), doraise=True)
    except py_compile.PyCompileError as exc:  # pragma: no cover - failure path
        pytest.fail(f"{script.name} failed to compile: {exc}")


# ---------------------------------------------------------------------------
# Import without Earth Engine (guarded imports)
# ---------------------------------------------------------------------------
def test_auth_module_imports_without_ee() -> None:
    """gee._auth imports offline and reports EE as unavailable (no crash)."""
    auth = importlib.import_module("gee._auth")
    # earthengine-api is not installed in the test env → ee_available() is False,
    # but importing must not raise and init_ee must fail loudly, not silently.
    assert hasattr(auth, "init_ee")
    assert hasattr(auth, "ee_available")
    assert auth.ee_available() is False
    with pytest.raises(auth.EarthEngineUnavailable):
        auth.init_ee(project="dummy-project")


@pytest.mark.parametrize("script", _numbered_scripts(), ids=lambda p: p.name)
def test_numbered_script_imports_without_ee(script: Path) -> None:
    """Each numbered step imports cleanly with EE absent (calls are deferred)."""
    spec = importlib.util.spec_from_file_location(f"gee_{script.stem}", script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)  # must not raise even though `ee` is missing
    except Exception as exc:  # pragma: no cover - failure path
        pytest.fail(f"importing {script.name} crashed without EE: {type(exc).__name__}: {exc}")
    # Every step exposes a main() entry point.
    assert hasattr(mod, "main"), f"{script.name} must define main()"


def test_main_without_ee_raises_clear_error() -> None:
    """Calling a step's main() without EE raises EarthEngineUnavailable, not noise."""
    auth = importlib.import_module("gee._auth")
    script = GEE_DIR / "01_optical_harmonize.py"
    spec = importlib.util.spec_from_file_location("gee_01_main", script)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    with pytest.raises(auth.EarthEngineUnavailable):
        mod.main(aoi=[0, 0, 1, 1], start="2023-01-01", end="2023-02-01")


# ---------------------------------------------------------------------------
# Notebooks are valid JSON notebooks
# ---------------------------------------------------------------------------
NOTEBOOKS = ["01_quickstart_demo.ipynb", "02_gee_pipeline.ipynb"]


@pytest.mark.parametrize("nb_name", NOTEBOOKS)
def test_notebook_is_valid_json_notebook(nb_name: str) -> None:
    """Each .ipynb parses as JSON and has the required notebook structure."""
    path = NOTEBOOKS_DIR / nb_name
    assert path.is_file(), f"missing notebook {nb_name}"
    with path.open(encoding="utf-8") as fh:
        nb = json.load(fh)  # raises JSONDecodeError if malformed

    assert isinstance(nb, dict), f"{nb_name}: top level must be an object"
    assert nb.get("nbformat") == 4, f"{nb_name}: expected nbformat 4"
    assert "nbformat_minor" in nb, f"{nb_name}: missing nbformat_minor"
    assert isinstance(nb.get("cells"), list) and nb["cells"], f"{nb_name}: no cells"
    assert isinstance(nb.get("metadata"), dict), f"{nb_name}: missing metadata"

    for i, cell in enumerate(nb["cells"]):
        assert cell.get("cell_type") in {"code", "markdown", "raw"}, f"{nb_name} cell {i}: bad type"
        assert "source" in cell, f"{nb_name} cell {i}: missing source"
        assert isinstance(cell["source"], (list, str)), f"{nb_name} cell {i}: bad source"
        if cell["cell_type"] == "code":
            # nbformat 4 code cells require these keys.
            assert "outputs" in cell, f"{nb_name} cell {i}: code cell missing outputs"
            assert "execution_count" in cell, (
                f"{nb_name} cell {i}: code cell missing execution_count"
            )


def test_quickstart_notebook_is_offline_demo() -> None:
    """The quickstart calls run_demo() and has a credential-free fallback path."""
    text = (NOTEBOOKS_DIR / "01_quickstart_demo.ipynb").read_text(encoding="utf-8")
    assert "run_demo" in text, "quickstart should call pipeline.orchestrator.run_demo()"
    assert "agristress.pipeline.orchestrator" in text
    # A synthetic fallback so it runs even when agristress isn't importable yet.
    assert "synthetic" in text.lower()


# ---------------------------------------------------------------------------
# gee/05 — AlphaEarth embedding + errorMatrix/Kappa
# ---------------------------------------------------------------------------
def _load_step05():
    script = GEE_DIR / "05_crop_classification.py"
    spec = importlib.util.spec_from_file_location("gee_05_classification", script)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod, script.read_text(encoding="utf-8")


def test_step05_references_alphaearth_embedding() -> None:
    """gee/05 must use the AlphaEarth Satellite Embedding asset id (64-D)."""
    mod, source = _load_step05()
    asset = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
    assert asset in source, "gee/05 should reference the AlphaEarth embedding asset id"
    assert getattr(mod, "SATELLITE_EMBEDDING_COLLECTION", None) == asset
    # 64-D embedding band list A00..A63.
    assert hasattr(mod, "EMBEDDING_BANDS")
    assert len(mod.EMBEDDING_BANDS) == 64, "AlphaEarth embedding is 64-D"
    assert mod.EMBEDDING_BANDS[0] == "A00" and mod.EMBEDDING_BANDS[-1] == "A63"
    # A function to materialise the embedding image exists.
    assert hasattr(mod, "satellite_embedding_image")


def test_step05_computes_error_matrix_and_kappa() -> None:
    """gee/05 must validate via errorMatrix and expose Overall Accuracy + Kappa."""
    mod, source = _load_step05()
    assert "errorMatrix" in source, "gee/05 should compute a confusion errorMatrix"
    assert ".kappa()" in source, "gee/05 should compute Cohen's Kappa from the errorMatrix"
    assert ".accuracy()" in source, "gee/05 should compute Overall Accuracy"
    # The accuracy-assessment helper returns both metrics.
    assert hasattr(mod, "accuracy_assessment")
    assert hasattr(mod, "classify_with_features")
    # And it trains a smileRandomForest classifier.
    assert "smileRandomForest" in source, "gee/05 should train ee.Classifier.smileRandomForest"
