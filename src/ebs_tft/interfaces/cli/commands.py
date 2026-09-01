"""Expose local command-line entry points."""

from __future__ import annotations

from pathlib import Path

import typer

from ebs_tft.application.usecases import pilot

app = typer.Typer(no_args_is_help=True)


@app.callback()
def root() -> None:
    """Run EBS TFT project workflows."""


@app.command("local-pilot")
def local_pilot(
    config: Path = typer.Option(
        Path("notebooks/pilot.yaml"),
        "--config",
        help="Path to the exact-schema local pilot YAML.",
    ),
    replace_output: bool = typer.Option(
        False,
        "--replace-output",
        help="Delete the configured *_outputs directory before training.",
    ),
) -> None:
    """Run the bounded native-resolution TFT/DeepLOB feasibility pilot."""
    specification = pilot.load_specification(path=config)
    pilot.run(specification=specification, replace_output=replace_output)


@app.command("local-model-sanity")
def local_model_sanity(
    output_dir: Path = typer.Option(
        Path("notebooks/model_sanity_outputs"),
        "--output-dir",
        help="Directory for deterministic adapter sanity evidence.",
    ),
    replace_output: bool = typer.Option(
        False,
        "--replace-output",
        help="Delete the configured *_outputs directory before training.",
    ),
) -> None:
    """Run deterministic controlled-signal checks for both model adapters."""
    pilot.run_model_sanity(
        output_dir=output_dir.resolve(), replace_output=replace_output
    )


@app.command("local-pilot-matrix")
def local_pilot_matrix(
    config: Path = typer.Option(
        Path("notebooks/pilot_matrix.yaml"),
        "--config",
        help="Path to the bounded multi-date pilot-matrix YAML.",
    ),
    reuse_existing: bool = typer.Option(
        False,
        "--reuse-existing",
        help="Rebuild matrix tables from complete existing per-date outputs.",
    ),
    replace_output: bool = typer.Option(
        False,
        "--replace-output",
        help="Delete configured per-date *_outputs directories before training.",
    ),
) -> None:
    """Run and combine the configured multi-date, multi-horizon pilots."""
    pilot.run_matrix_from_config(
        path=config,
        reuse_existing=reuse_existing,
        replace_output=replace_output,
    )


def main() -> None:
    """Invoke the project CLI."""
    app()


if __name__ == "__main__":
    main()
