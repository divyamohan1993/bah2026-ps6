"""AgriStress command-line interface.

Exposes the pipeline stages and the serving app as subcommands::

    agristress catalog        # print the sensor registry summary
    agristress ingest  ...    # ingest satellite collections (best-effort)
    agristress fuse    ...     # build the fused datacube
    agristress features ...    # extract spectral / SAR / phenology features
    agristress train   ...     # train crop + stress models
    agristress advisory ...    # generate 8-day irrigation advisory
    agristress demo           # run the whole chain on synthetic data (no creds)
    agristress serve          # launch the FastAPI serving app via uvicorn

Uses `Typer <https://typer.tiangolo.com>`_ when installed, otherwise falls back
to a functionally-equivalent :mod:`argparse` CLI so the command always works.
The module-level ``app`` object is the console-script entrypoint
(``pyproject``: ``agristress = "agristress.pipeline.cli:app"``); ``main()`` is a
plain callable wrapper.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

# --------------------------------------------------------------------------
# Stage implementations (shared between Typer and argparse front-ends)
# --------------------------------------------------------------------------
_DEFAULT_AOI = "CMD-001"
_DEFAULT_SEASON = "kharif"

# Serving bind defaults. Cloud Run (and most container PaaS) inject the listen
# port via ``$PORT`` and require the server to bind ``0.0.0.0`` so the platform
# health-check / router can reach it. We honour ``$HOST``/``$PORT`` here so the
# same image runs locally, in docker-compose and on Cloud Run with no flags.
# 0.0.0.0 is intentional: inside a container we must bind all interfaces so the
# Cloud Run / docker port-mapping can route to the process.
_DEFAULT_HOST = os.environ.get("HOST", "0.0.0.0")
_DEFAULT_PORT = int(os.environ.get("PORT", "8080"))


def _print(obj: Any) -> None:
    """Pretty-print a result (rich if available, else json/plain)."""

    try:
        from rich import print_json  # type: ignore

        print_json(data=obj)
        return
    except Exception:
        pass
    if isinstance(obj, (dict, list)):
        print(json.dumps(obj, indent=2, default=str))
    else:
        print(obj)


def cmd_catalog() -> dict[str, Any]:
    """Print the sensor registry summary (defensive: synthesises if absent)."""

    summary: dict[str, Any] | None = None
    try:
        import importlib

        catalog = importlib.import_module("agristress.catalog")
        for name in ("registry_summary", "summary", "get_registry", "registry", "SENSOR_REGISTRY"):
            obj = getattr(catalog, name, None)
            data = obj() if callable(obj) else obj
            if data:
                summary = data if isinstance(data, dict) else {"registry": data}
                break
    except Exception:
        summary = None

    if summary is None:
        # Fallback registry mirroring PS6 datasets so the command is informative.
        summary = {
            "source": "fallback (agristress.catalog not available yet)",
            "optical": ["LISS-IV", "LISS-III", "AWiFS", "Sentinel-2", "Landsat-8/9", "MODIS"],
            "sar": ["EOS-04", "Sentinel-1", "NISAR-SAR"],
            "ancillary": ["rainfall", "reference_ET", "canal_command", "soil", "crop_coeff"],
            "n_sensors": 9,
        }
    _print(summary)
    return summary


def cmd_demo(aoi: str = _DEFAULT_AOI, season: str = _DEFAULT_SEASON) -> dict[str, Any]:
    """Run the full demo pipeline on synthetic data (no credentials)."""

    from .orchestrator import run_demo

    result = run_demo(aoi=aoi, season=season)
    out = result.summary()
    _print(out)
    return out


def _stage(name: str, aoi: str, season: str) -> dict[str, Any]:
    """Run a single named pipeline stage best-effort and report status."""

    from .orchestrator import Pipeline

    pipe = Pipeline()
    pipe.context.update({"aoi": aoi, "season": season})
    mapping = {
        s[0]: s
        for s in __import__("agristress.pipeline.orchestrator", fromlist=["_STAGES"])._STAGES
    }
    # Map user-facing verbs to internal stage attrs.
    verb_to_attr = {
        "ingest": "ingestion",
        "fuse": "fusion",
        "features": "features",
        "train": "models",
        "advisory": "irrigation",
    }
    attr = verb_to_attr.get(name, name)
    info: dict[str, Any] = {"stage": name, "aoi": aoi, "season": season}
    if attr in mapping:
        _, module_path, names = mapping[attr]
        pipe._run_stage(attr, module_path, names)
        info["status"] = "ran" if attr in pipe.stages_run else "fallback (module not ready)"
    else:
        info["status"] = "unknown stage"
    _print(info)
    return info


def cmd_ingest(aoi: str = _DEFAULT_AOI, season: str = _DEFAULT_SEASON) -> dict[str, Any]:
    return _stage("ingest", aoi, season)


def cmd_fuse(aoi: str = _DEFAULT_AOI, season: str = _DEFAULT_SEASON) -> dict[str, Any]:
    return _stage("fuse", aoi, season)


def cmd_features(aoi: str = _DEFAULT_AOI, season: str = _DEFAULT_SEASON) -> dict[str, Any]:
    return _stage("features", aoi, season)


def cmd_train(aoi: str = _DEFAULT_AOI, season: str = _DEFAULT_SEASON) -> dict[str, Any]:
    return _stage("train", aoi, season)


def cmd_advisory(aoi: str = _DEFAULT_AOI, season: str = _DEFAULT_SEASON) -> dict[str, Any]:
    return _stage("advisory", aoi, season)


def cmd_serve(host: str | None = None, port: int | None = None, reload: bool = False) -> None:
    """Launch the FastAPI serving app via uvicorn.

    ``host``/``port`` fall back to ``$HOST``/``$PORT`` (then ``0.0.0.0:8080``),
    so the container image is Cloud Run-ready with no arguments: Cloud Run sets
    ``$PORT`` and expects the process to listen on ``0.0.0.0:$PORT``.
    """

    host = host or _DEFAULT_HOST
    port = _DEFAULT_PORT if port is None else port

    try:
        import uvicorn
    except Exception as exc:  # pragma: no cover - serving extra not installed
        print(
            f"uvicorn is required to serve (install with `pip install 'agristress[serving]'`): {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    # Import string lets uvicorn manage reload workers; the module builds `app`.
    print(f"Starting AgriStress serving API on http://{host}:{port} ...")
    uvicorn.run("agristress.serving.api:app", host=host, port=port, reload=reload)


# --------------------------------------------------------------------------
# Typer front-end (preferred) with argparse fallback
# --------------------------------------------------------------------------
def _build_typer() -> Any | None:
    try:
        import typer
    except Exception:
        return None

    cli = typer.Typer(
        add_completion=False,
        help="AgriStress — crop-type, moisture-stress & irrigation-advisory toolkit (BAH 2026 PS6).",
        no_args_is_help=True,
    )

    aoi_opt = typer.Option(_DEFAULT_AOI, help="Area-of-interest / command id.")
    season_opt = typer.Option(_DEFAULT_SEASON, help="Crop season: kharif/rabi/zaid.")

    @cli.command()
    def catalog() -> None:
        """Print the sensor registry summary."""
        cmd_catalog()

    @cli.command()
    def ingest(aoi: str = aoi_opt, season: str = season_opt) -> None:
        """Ingest satellite collections for the AOI/season."""
        cmd_ingest(aoi, season)

    @cli.command()
    def fuse(aoi: str = aoi_opt, season: str = season_opt) -> None:
        """Build the fused optical+SAR datacube."""
        cmd_fuse(aoi, season)

    @cli.command()
    def features(aoi: str = aoi_opt, season: str = season_opt) -> None:
        """Extract spectral / SAR / phenology features."""
        cmd_features(aoi, season)

    @cli.command()
    def train(aoi: str = aoi_opt, season: str = season_opt) -> None:
        """Train crop-type and moisture-stress models."""
        cmd_train(aoi, season)

    @cli.command()
    def advisory(aoi: str = aoi_opt, season: str = season_opt) -> None:
        """Generate the 8-day irrigation advisory."""
        cmd_advisory(aoi, season)

    @cli.command()
    def demo(aoi: str = aoi_opt, season: str = season_opt) -> None:
        """Run the full pipeline on synthetic data (no credentials)."""
        cmd_demo(aoi, season)

    @cli.command()
    def serve(
        host: str = typer.Option(_DEFAULT_HOST, help="Bind host (env: HOST)."),
        port: int = typer.Option(_DEFAULT_PORT, help="Bind port (env: PORT)."),
        reload: bool = typer.Option(False, help="Auto-reload (dev)."),
    ) -> None:
        """Launch the FastAPI serving app via uvicorn (Cloud Run-ready)."""
        cmd_serve(host, port, reload)

    return cli


def _argparse_main(argv: list[str] | None = None) -> int:
    """Fallback CLI when Typer isn't installed."""

    import argparse

    parser = argparse.ArgumentParser(
        prog="agristress", description="AgriStress CLI (BAH 2026 PS6)."
    )
    sub = parser.add_subparsers(dest="command")

    def _add_aoi_season(p: argparse.ArgumentParser) -> None:
        p.add_argument("--aoi", default=_DEFAULT_AOI)
        p.add_argument("--season", default=_DEFAULT_SEASON)

    sub.add_parser("catalog", help="Print the sensor registry summary.")
    for verb in ("ingest", "fuse", "features", "train", "advisory", "demo"):
        _add_aoi_season(sub.add_parser(verb))
    serve_p = sub.add_parser("serve", help="Launch the serving API.")
    serve_p.add_argument("--host", default=_DEFAULT_HOST, help="Bind host (env: HOST).")
    serve_p.add_argument("--port", type=int, default=_DEFAULT_PORT, help="Bind port (env: PORT).")
    serve_p.add_argument("--reload", action="store_true")

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0

    dispatch = {
        "catalog": lambda: cmd_catalog(),
        "ingest": lambda: cmd_ingest(args.aoi, args.season),
        "fuse": lambda: cmd_fuse(args.aoi, args.season),
        "features": lambda: cmd_features(args.aoi, args.season),
        "train": lambda: cmd_train(args.aoi, args.season),
        "advisory": lambda: cmd_advisory(args.aoi, args.season),
        "demo": lambda: cmd_demo(args.aoi, args.season),
        "serve": lambda: cmd_serve(args.host, args.port, args.reload),
    }
    dispatch[args.command]()
    return 0


# Module-level entrypoint object referenced by pyproject [project.scripts].
_typer_app = _build_typer()


def app(argv: list[str] | None = None) -> Any:
    """Console-script entrypoint.

    Invokes the Typer app when available (Typer/Click parse ``sys.argv`` itself),
    otherwise the argparse fallback. Callable with ``argv`` for testing.
    """

    if _typer_app is not None:
        if argv is None:
            return _typer_app()
        # Drive Typer/Click programmatically with explicit args (for tests).
        from typer.testing import CliRunner  # type: ignore

        result = CliRunner().invoke(_typer_app, argv)
        if result.exception and not isinstance(result.exception, SystemExit):
            raise result.exception
        return result
    return _argparse_main(argv)


def main(argv: list[str] | None = None) -> Any:
    """Plain ``main()`` wrapper (alias of :func:`app`)."""

    return app(argv)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_argparse_main() if _typer_app is None else (app() or 0))
