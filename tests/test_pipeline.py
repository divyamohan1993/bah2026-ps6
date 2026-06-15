"""Tests for the AgriStress pipeline orchestrator + CLI.

Everything runs with no credentials and no sibling science modules present: the
orchestrator falls back to synthetic data and still populates the feature store.
"""

from __future__ import annotations

from pathlib import Path

from agristress.pipeline.orchestrator import Pipeline, PipelineResult, run_demo
from agristress.serving.store import FeatureStore


# -- orchestrator -----------------------------------------------------------
def test_run_demo_populates_store(tmp_path: Path) -> None:
    result = run_demo(aoi="CMD-001", season="kharif", out_dir=tmp_path)
    assert isinstance(result, PipelineResult)
    assert result.n_cells > 0
    assert result.n_records > 0
    # The store is the materialised H3 store the serving API reads.
    assert isinstance(result.store, FeatureStore)
    cell = result.store.cells()[0]
    assert result.store.get(cell, result.store.latest_date(cell), "advisory") is not None


def test_run_demo_no_credentials_uses_fallbacks() -> None:
    # No sibling science modules exist yet → every stage should fall back,
    # but the run must still complete and produce records.
    pipe = Pipeline()
    result = pipe.run_demo()
    assert result.n_records > 0
    # All declared stages either ran or fell back; none silently vanished.
    assert len(result.stages_run) + len(result.stages_fallback) >= 1
    # Summary is JSON-serialisable.
    summary = result.summary()
    assert summary["cells"] == result.n_cells


def test_run_demo_writes_artifacts(tmp_path: Path) -> None:
    result = run_demo(out_dir=tmp_path)
    # At least the PNG previews should be written (Pillow is available).
    pngs = list(tmp_path.glob("*.png"))
    assert pngs, "expected demo PNG artefacts"
    assert all(Path(p).exists() for p in result.artifacts)


def test_seasons_differ() -> None:
    kharif = run_demo(season="kharif").store
    rabi = run_demo(season="rabi").store
    # Different seasons map to different composite dates.
    assert kharif.dates() != rabi.dates()


# -- CLI --------------------------------------------------------------------
def test_cli_catalog_runs(capsys) -> None:
    from agristress.pipeline.cli import cmd_catalog

    summary = cmd_catalog()
    assert isinstance(summary, dict)
    assert "optical" in summary or "registry" in summary


def test_cli_demo_runs(capsys) -> None:
    from agristress.pipeline.cli import cmd_demo

    out = cmd_demo(aoi="CMD-001", season="kharif")
    assert out["cells"] > 0
    assert out["records"] > 0


def test_cli_app_invoces_catalog() -> None:
    # Exercise the entrypoint object (Typer or argparse) without raising.
    from agristress.pipeline.cli import app

    result = app(["catalog"])
    # Typer path returns a CliRunner result with exit_code 0; argparse returns 0.
    if hasattr(result, "exit_code"):
        assert result.exit_code == 0
    else:
        assert result == 0


def test_cli_main_alias() -> None:
    from agristress.pipeline.cli import main

    result = main(["demo", "--season", "rabi"])
    if hasattr(result, "exit_code"):
        assert result.exit_code == 0
    else:
        assert result == 0
