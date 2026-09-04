"""Expose local command-line entry points."""

from __future__ import annotations

from pathlib import Path

import typer

from ebs_tft.application.usecases import pilot, research_protocol

app = typer.Typer(no_args_is_help=True)


@app.callback()
def root() -> None:
    """Run EBS TFT project workflows."""


@app.command("local-pilot")
def local_pilot(
    config: Path = typer.Option(
        Path("notebooks/pilot_smoke.yaml"),
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
        ...,
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


@app.command("local-multi-session")
def local_multi_session(
    config: Path = typer.Option(
        Path("notebooks/multi_session_development.yaml"),
        "--config",
        help="Path to the day-aware multi-session development YAML.",
    ),
    replace_output: bool = typer.Option(
        False,
        "--replace-output",
        help="Delete the configured *_outputs directory before training.",
    ),
) -> None:
    """Run day-aware training and later-session development validation."""
    specification = pilot.load_multi_session_specification(path=config)
    pilot.run_multi_session(specification=specification, replace_output=replace_output)


@app.command("research-session-audit")
def research_session_audit(
    config: Path = typer.Option(
        Path("notebooks/research_protocol.yaml"),
        "--config",
        help="Path to the exact-schema research-protocol YAML.",
    ),
    replace_output: bool = typer.Option(
        False,
        "--replace-output",
        help="Delete the configured research output before auditing.",
    ),
) -> None:
    """Audit all configured sessions and freeze chronological split identities."""
    protocol = research_protocol.load_protocol(path=config)
    research_protocol.run_session_audit(
        protocol=protocol,
        protocol_path=config.resolve(),
        replace_output=replace_output,
    )


@app.command("research-baseline-gate")
def research_baseline_gate(
    config: Path = typer.Option(
        Path("notebooks/research_protocol.yaml"),
        "--config",
        help="Path to the audited research-protocol YAML.",
    ),
    replace_output: bool = typer.Option(
        False,
        "--replace-output",
        help="Delete prior rolling-baseline outputs before evaluation.",
    ),
) -> None:
    """Evaluate rolling defensive baselines before neural GPU work."""
    protocol = research_protocol.load_protocol(path=config)
    research_protocol.run_baseline_gate(
        protocol=protocol,
        protocol_path=config.resolve(),
        replace_output=replace_output,
    )


@app.command("research-model-protocol")
def research_model_protocol(
    config: Path = typer.Option(
        Path("notebooks/research_protocol.yaml"),
        "--config",
        help="Path to the exact-schema research-protocol YAML.",
    ),
    replace_output: bool = typer.Option(
        False,
        "--replace-output",
        help="Delete prior model-protocol verification output.",
    ),
) -> None:
    """Verify and disclose model-adapter capabilities before GPU training."""
    protocol = research_protocol.load_protocol(path=config)
    research_protocol.run_model_protocol_verification(
        protocol=protocol, replace_output=replace_output
    )


@app.command("research-neural-benchmark")
def research_neural_benchmark(
    config: Path = typer.Option(
        Path("notebooks/research_protocol.yaml"),
        "--config",
        help="Path to the audited research-protocol YAML.",
    ),
    policy: Path = typer.Option(
        Path("notebooks/research_neural_benchmark.yaml"),
        "--policy",
        help="Path to the frozen neural optimization policy YAML.",
    ),
    replace_output: bool = typer.Option(
        False,
        "--replace-output",
        help="Delete prior gated-neural outputs before training.",
    ),
    maximum_new_cells: int | None = typer.Option(
        None,
        "--maximum-new-cells",
        min=1,
        help="Pause safely after this many newly completed cells.",
    ),
) -> None:
    """Run the finite rolling neural benchmark admitted by baseline evidence."""
    loaded_protocol = research_protocol.load_protocol(path=config)
    loaded_policy = research_protocol.load_policy(path=policy)
    try:
        research_protocol.run_neural_benchmark(
            protocol=loaded_protocol,
            protocol_path=config.resolve(),
            policy=loaded_policy,
            policy_path=policy.resolve(),
            replace_output=replace_output,
            maximum_new_cells=maximum_new_cells,
        )
    except research_protocol.NeuralBenchmarkPausedError as exc:
        typer.echo(str(exc))


def main() -> None:
    """Invoke the project CLI."""
    app()


if __name__ == "__main__":
    main()
