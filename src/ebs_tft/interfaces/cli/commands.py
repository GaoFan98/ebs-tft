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


@app.command("research-freeze-locked-evaluation")
def research_freeze_locked_evaluation(
    config: Path = typer.Option(Path("notebooks/research_protocol.yaml"), "--config"),
    policy: Path = typer.Option(
        Path("notebooks/research_neural_benchmark.yaml"), "--policy"
    ),
) -> None:
    """Freeze final candidates and durations without reading locked outcomes."""
    loaded_protocol = research_protocol.load_protocol(path=config)
    loaded_policy = research_protocol.load_policy(path=policy)
    research_protocol.freeze_locked_evaluation_plan(
        protocol=loaded_protocol,
        protocol_path=config.resolve(),
        policy=loaded_policy,
        policy_path=policy.resolve(),
    )


@app.command("research-locked-evaluation")
def research_locked_evaluation(
    plan_sha256: str = typer.Option(
        ..., "--plan-sha256", help="Exact SHA-256 printed by the freeze command."
    ),
    config: Path = typer.Option(Path("notebooks/research_protocol.yaml"), "--config"),
    policy: Path = typer.Option(
        Path("notebooks/research_neural_benchmark.yaml"), "--policy"
    ),
    maximum_new_cells: int | None = typer.Option(
        None,
        "--maximum-new-cells",
        min=1,
        help="Pause after this many newly completed final model cells.",
    ),
) -> None:
    """Run the frozen one-time final test; completed results cannot be replaced."""
    loaded_protocol = research_protocol.load_protocol(path=config)
    loaded_policy = research_protocol.load_policy(path=policy)
    try:
        research_protocol.run_locked_evaluation(
            protocol=loaded_protocol,
            protocol_path=config.resolve(),
            policy=loaded_policy,
            policy_path=policy.resolve(),
            plan_sha256=plan_sha256,
            maximum_new_cells=maximum_new_cells,
        )
    except research_protocol.LockedEvaluationPausedError as exc:
        typer.echo(str(exc))


@app.command("research-freeze-cross-instrument")
def research_freeze_cross_instrument(
    config: Path = typer.Option(Path("notebooks/research_protocol.yaml"), "--config"),
    policy: Path = typer.Option(
        Path("notebooks/research_neural_benchmark.yaml"), "--policy"
    ),
) -> None:
    """Freeze transfer targets and checkpoints without reading target outcomes."""
    loaded_protocol = research_protocol.load_protocol(path=config)
    research_protocol.freeze_cross_instrument_plan(
        protocol=loaded_protocol,
        protocol_path=config.resolve(),
        policy_path=policy.resolve(),
    )


@app.command("research-cross-instrument")
def research_cross_instrument(
    plan_sha256: str = typer.Option(
        ..., "--plan-sha256", help="Exact SHA-256 printed by the freeze command."
    ),
    config: Path = typer.Option(Path("notebooks/research_protocol.yaml"), "--config"),
    policy: Path = typer.Option(
        Path("notebooks/research_neural_benchmark.yaml"), "--policy"
    ),
    maximum_new_cells: int | None = typer.Option(
        None,
        "--maximum-new-cells",
        min=1,
        help="Pause after this many newly completed inference cells.",
    ),
) -> None:
    """Evaluate frozen EUR/USD checkpoints on the two target instruments."""
    loaded_protocol = research_protocol.load_protocol(path=config)
    loaded_policy = research_protocol.load_policy(path=policy)
    try:
        research_protocol.run_cross_instrument_evaluation(
            protocol=loaded_protocol,
            protocol_path=config.resolve(),
            policy=loaded_policy,
            policy_path=policy.resolve(),
            plan_sha256=plan_sha256,
            maximum_new_cells=maximum_new_cells,
        )
    except research_protocol.CrossInstrumentPausedError as exc:
        typer.echo(str(exc))


@app.command("research-final-report")
def research_final_report(
    config: Path = typer.Option(Path("notebooks/research_protocol.yaml"), "--config"),
    output_dir: Path = typer.Option(
        Path("notebooks/research_final_analysis_outputs"),
        "--output-dir",
        help="Directory for the verified, reporting-only 2024 analysis.",
    ),
    replace_output: bool = typer.Option(
        False,
        "--replace-output",
        help="Rebuild derived report files without changing experimental evidence.",
    ),
) -> None:
    """Verify immutable 2024 evidence and build the final analysis report."""
    loaded_protocol = research_protocol.load_protocol(path=config)
    research_protocol.run_final_report(
        protocol=loaded_protocol,
        protocol_path=config.resolve(),
        output_dir=output_dir.resolve(),
        replace_output=replace_output,
    )


@app.command("research-temporal-audit")
def research_temporal_audit(
    config: Path = typer.Option(Path("notebooks/research_protocol.yaml"), "--config"),
    temporal_policy: Path = typer.Option(
        Path("notebooks/research_temporal_evaluation.yaml"), "--temporal-policy"
    ),
) -> None:
    """Audit the external-year sample without calculating target outcomes."""
    loaded_protocol = research_protocol.load_protocol(path=config)
    loaded_temporal_policy = research_protocol.load_temporal_policy(
        path=temporal_policy
    )
    research_protocol.run_temporal_audit(
        protocol=loaded_protocol,
        protocol_path=config.resolve(),
        temporal_policy=loaded_temporal_policy,
        temporal_policy_path=temporal_policy.resolve(),
    )


@app.command("research-freeze-temporal-evaluation")
def research_freeze_temporal_evaluation(
    config: Path = typer.Option(Path("notebooks/research_protocol.yaml"), "--config"),
    policy: Path = typer.Option(
        Path("notebooks/research_neural_benchmark.yaml"), "--policy"
    ),
    temporal_policy: Path = typer.Option(
        Path("notebooks/research_temporal_evaluation.yaml"), "--temporal-policy"
    ),
) -> None:
    """Freeze external-year sessions without reading their outcomes."""
    loaded_protocol = research_protocol.load_protocol(path=config)
    loaded_temporal_policy = research_protocol.load_temporal_policy(
        path=temporal_policy
    )
    research_protocol.freeze_temporal_evaluation_plan(
        protocol=loaded_protocol,
        protocol_path=config.resolve(),
        neural_policy_path=policy.resolve(),
        temporal_policy=loaded_temporal_policy,
        temporal_policy_path=temporal_policy.resolve(),
    )


@app.command("research-temporal-evaluation")
def research_temporal_evaluation(
    plan_sha256: str = typer.Option(
        ..., "--plan-sha256", help="Exact SHA-256 printed by the freeze command."
    ),
    config: Path = typer.Option(Path("notebooks/research_protocol.yaml"), "--config"),
    policy: Path = typer.Option(
        Path("notebooks/research_neural_benchmark.yaml"), "--policy"
    ),
    temporal_policy: Path = typer.Option(
        Path("notebooks/research_temporal_evaluation.yaml"), "--temporal-policy"
    ),
    maximum_new_sessions: int | None = typer.Option(
        None,
        "--maximum-new-sessions",
        min=1,
        help="Pause after this many newly completed instrument sessions.",
    ),
) -> None:
    """Run the frozen, inference-only external-year evaluation."""
    loaded_protocol = research_protocol.load_protocol(path=config)
    loaded_policy = research_protocol.load_policy(path=policy)
    loaded_temporal_policy = research_protocol.load_temporal_policy(
        path=temporal_policy
    )
    try:
        research_protocol.run_temporal_evaluation(
            protocol=loaded_protocol,
            protocol_path=config.resolve(),
            neural_policy=loaded_policy,
            neural_policy_path=policy.resolve(),
            temporal_policy=loaded_temporal_policy,
            temporal_policy_path=temporal_policy.resolve(),
            plan_sha256=plan_sha256,
            maximum_new_sessions=maximum_new_sessions,
        )
    except research_protocol.TemporalEvaluationPausedError as exc:
        typer.echo(str(exc))


def main() -> None:
    """Invoke the project CLI."""
    app()


if __name__ == "__main__":
    main()
