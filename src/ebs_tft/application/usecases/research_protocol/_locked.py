"""Freeze and execute the one-time locked EUR/USD evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import time
from contextlib import closing
from pathlib import Path
from typing import cast

import attrs
import numpy as np
import polars as pl
import sklearn
import torch
import yaml

from ebs_tft.application.usecases.research_protocol import _baseline, _neural
from ebs_tft.data.parsers import ebs_csv
from ebs_tft.data.repositories import checkpoint as checkpoint_repository
from ebs_tft.domain import model as model_domain
from ebs_tft.domain.pilot import operations as pilot_operations
from ebs_tft.domain.pilot import training as pilot_training
from ebs_tft.domain.research import models as research_models
from ebs_tft.domain.research import operations as research_operations

LOCKED_EVALUATION_IMPLEMENTATION_VERSION = 1


@attrs.frozen
class LockedEvaluationPlanResult:
    """Reference the frozen plan that does not inspect locked outcomes."""

    plan_path: Path
    plan_sha256: str


@attrs.frozen
class LockedEvaluationResult:
    """Reference completed one-time locked evaluation artifacts."""

    output_dir: Path
    metrics_path: Path
    comparisons_path: Path
    decision_path: Path
    terminal_summary_path: Path


class LockedEvaluationPausedError(Exception):
    """Indicate a safe pause after a bounded number of final model cells."""

    def __init__(self, *, completed_cells: int, total_cells: int, output_dir: Path):
        self.completed_cells = completed_cells
        self.total_cells = total_cells
        self.output_dir = output_dir
        super().__init__(
            f"Locked evaluation paused safely after {completed_cells}/{total_cells} "
            f"model cells; resume with the same plan hash. Outputs: {output_dir}"
        )


def freeze_plan(
    *,
    protocol: research_models.ResearchProtocol,
    protocol_path: Path,
    policy: research_models.NeuralBenchmarkPolicy,
    policy_path: Path,
) -> LockedEvaluationPlanResult:
    """Freeze accepted candidates and final epoch counts from development only."""
    manifest_path = protocol.output_dir / "split_manifest.yaml"
    audit_path = protocol.output_dir / "session_audit.csv"
    neural_dir = protocol.output_dir / "neural_benchmark"
    neural_summary_path = neural_dir / "run_summary.json"
    neural_gate_path = neural_dir / "gate_decision.json"
    neural_identity_path = neural_dir / "run_identity.json"
    required = (neural_summary_path, neural_gate_path, neural_identity_path)
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("complete the 64-cell neural benchmark first")
    folds = _baseline._load_folds(
        manifest_path=manifest_path,
        audit_path=audit_path,
        protocol_path=protocol_path,
        protocol=protocol,
    )
    _baseline._verify_cached_states(
        folds=folds, audit_path=audit_path, protocol=protocol
    )
    summary = _neural._json_mapping(path=neural_summary_path)
    cell_summaries = tuple((neural_dir / "cells").rglob("cell_summary.json"))
    if (
        not isinstance(summary.get("cells"), int)
        or summary.get("cells") != len(cell_summaries)
        or (neural_dir / "progress_summary.json").exists()
    ):
        raise ValueError("neural benchmark is not complete")
    identity = _neural._json_mapping(path=neural_identity_path)
    expected_hashes = {
        "protocol_sha256": _baseline._sha256_file(path=protocol_path),
        "policy_sha256": _baseline._sha256_file(path=policy_path),
        "manifest_sha256": _baseline._sha256_file(path=manifest_path),
        "audit_sha256": _baseline._sha256_file(path=audit_path),
    }
    if any(identity.get(key) != value for key, value in expected_hashes.items()):
        raise ValueError("neural benchmark inputs do not match current inputs")
    gate = _neural._json_mapping(path=neural_gate_path)
    if gate.get("locked_evaluation_used") is not False:
        raise ValueError("candidate gate must contain development evidence only")
    comparison_path = neural_dir / "paired_baseline_comparisons.csv"
    if not comparison_path.is_file():
        raise ValueError("neural benchmark comparison evidence is missing")
    recomputed_gate = _neural._gate_decision(
        comparisons=pl.read_csv(comparison_path), protocol=protocol
    )
    if gate != recomputed_gate:
        raise ValueError("neural candidate gate does not match its comparison evidence")
    raw_candidates = gate.get("accepted_model_depth_horizons")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("neural benchmark admitted no locked-evaluation candidate")
    development_sessions = _development_sessions(folds=folds)
    final_sessions = _final_sessions(
        manifest_path=manifest_path, protocol=protocol
    )
    cells: list[dict[str, object]] = []
    for raw_candidate in raw_candidates:
        if not isinstance(raw_candidate, dict):
            raise ValueError("neural gate candidate is invalid")
        candidate = cast(dict[str, object], raw_candidate)
        model_name = _string(candidate, "model")
        depth = _integer(candidate, "depth")
        horizon_milliseconds = _integer(candidate, "horizon_milliseconds")
        if model_name not in protocol.models or depth not in protocol.depths:
            raise ValueError("neural gate candidate is outside the protocol")
        if horizon_milliseconds not in protocol.forecast_horizons_milliseconds:
            raise ValueError("neural gate horizon is outside the protocol")
        for seed in protocol.random_seeds:
            best_epochs: list[dict[str, object]] = []
            for fold in folds:
                cell_dir = (
                    neural_dir
                    / "cells"
                    / fold.identifier
                    / f"h{horizon_milliseconds}"
                    / f"depth_{depth}"
                    / model_name
                    / f"seed_{seed}"
                )
                history_path = cell_dir / "training_history.json"
                cell_summary_path = cell_dir / "cell_summary.json"
                if not history_path.is_file() or not cell_summary_path.is_file():
                    raise ValueError(
                        f"accepted development cell is incomplete: {cell_dir}"
                    )
                cell_summary = _neural._json_mapping(path=cell_summary_path)
                artifact_hashes = cell_summary.get("artifact_sha256")
                if not isinstance(artifact_hashes, dict) or artifact_hashes.get(
                    "training_history.json"
                ) != _baseline._sha256_file(path=history_path):
                    raise ValueError(
                        f"development history failed integrity check: {cell_dir}"
                    )
                history = _neural._json_mapping(path=history_path)
                best_epochs.append(
                    {
                        "fold": fold.identifier,
                        "best_epoch": _integer(history, "best_epoch"),
                        "history_sha256": _baseline._sha256_file(path=history_path),
                    }
                )
            ordered = sorted(_integer(item, "best_epoch") for item in best_epochs)
            final_epochs = ordered[len(ordered) // 2]
            cells.append(
                {
                    "model": model_name,
                    "depth": depth,
                    "horizon_milliseconds": horizon_milliseconds,
                    "horizon_steps": (
                        horizon_milliseconds
                        // protocol.state_interval_milliseconds
                    ),
                    "seed": seed,
                    "fixed_epochs": final_epochs,
                    "development_best_epochs": best_epochs,
                }
            )
    plan = {
        "schema_version": 1,
        "implementation_version": LOCKED_EVALUATION_IMPLEMENTATION_VERSION,
        **expected_hashes,
        "neural_run_summary_sha256": _baseline._sha256_file(
            path=neural_summary_path
        ),
        "neural_gate_sha256": _baseline._sha256_file(path=neural_gate_path),
        "neural_run_identity_sha256": _baseline._sha256_file(
            path=neural_identity_path
        ),
        "neural_comparisons_sha256": _baseline._sha256_file(path=comparison_path),
        "training_duration_rule": (
            "upper median of best_epoch across the development folds, "
            "separately for each accepted model/depth/horizon/seed"
        ),
        "evaluation_rule": (
            "Across-seed mean neural performance must have a strictly positive "
            "lower paired-session bootstrap confidence bound over a same-data "
            "logistic baseline for every primary metric."
        ),
        "development_sessions": [
            _identity_payload(item=item, protocol=protocol)
            for item in development_sessions
        ],
        "final_test_sessions": [
            _identity_payload(item=item, protocol=protocol) for item in final_sessions
        ],
        "reserved_earlier_locked_dates_are_unused": True,
        "cells": cells,
        "policy": attrs.asdict(policy),
        "primary_metrics": [item.value for item in protocol.primary_metrics],
        "supporting_metrics": [item.value for item in protocol.supporting_metrics],
        "bootstrap_repetitions": protocol.bootstrap_repetitions,
        "confidence_level": protocol.confidence_level,
        "locked_outcomes_inspected": False,
    }
    output_dir = protocol.output_dir / "locked_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "plan.json"
    serialized = json.dumps(plan, indent=2)
    if plan_path.is_file() and plan_path.read_text(encoding="utf-8") != serialized:
        raise ValueError("a different locked-evaluation plan is already frozen")
    _neural._write_text_atomically(text=serialized, path=plan_path)
    plan_hash = _baseline._sha256_file(path=plan_path)
    print("EBS locked evaluation plan frozen")
    print("WARNING: locked outcomes were not inspected.")
    print(f"development_sessions={len(development_sessions)}")
    print(f"final_test_sessions={len(final_sessions)}")
    print(f"model_cells={len(cells)}")
    print(f"plan_sha256={plan_hash}")
    return LockedEvaluationPlanResult(plan_path=plan_path, plan_sha256=plan_hash)


def run(
    *,
    protocol: research_models.ResearchProtocol,
    protocol_path: Path,
    policy: research_models.NeuralBenchmarkPolicy,
    policy_path: Path,
    plan_sha256: str,
    maximum_new_cells: int | None = None,
) -> LockedEvaluationResult:
    """Execute exactly one frozen plan against final locked sessions."""
    if maximum_new_cells is not None and (
        isinstance(maximum_new_cells, bool) or maximum_new_cells <= 0
    ):
        raise ValueError("maximum_new_cells must be positive or null")
    started = time.perf_counter()
    output_dir = protocol.output_dir / "locked_evaluation"
    plan_path = output_dir / "plan.json"
    if not plan_path.is_file() or _baseline._sha256_file(path=plan_path) != plan_sha256:
        raise ValueError("--plan-sha256 must match the frozen plan exactly")
    if (output_dir / "run_summary.json").is_file():
        raise ValueError("locked evaluation is already complete and cannot be rerun")
    plan = _neural._json_mapping(path=plan_path)
    _verify_plan_inputs(
        plan=plan,
        protocol_path=protocol_path,
        policy_path=policy_path,
        protocol=protocol,
    )
    development_sessions = _planned_sessions(
        value=plan.get("development_sessions"), protocol=protocol
    )
    final_sessions = _planned_sessions(
        value=plan.get("final_test_sessions"), protocol=protocol
    )
    _verify_sources(sessions=(*development_sessions, *final_sessions))
    _materialize_locked_cache(protocol=protocol, sessions=final_sessions)
    raw_cells = plan.get("cells")
    if not isinstance(raw_cells, list) or not raw_cells:
        raise ValueError("frozen plan has no model cells")
    device = model_domain.select_device(requested=policy.device)
    all_metrics: list[pl.DataFrame] = []
    completed_cells = 0
    new_cells = 0
    grouped = _group_cells(raw_cells=raw_cells)
    baseline_frames: list[pl.DataFrame] = []
    for (depth, horizon_steps), cells in grouped.items():
        training_corpus, evaluation_corpus = _prepare_final_corpora(
            protocol=protocol,
            development_sessions=development_sessions,
            final_sessions=final_sessions,
            depth=depth,
            horizon_steps=horizon_steps,
        )
        training_dataset = model_domain.SequenceDataset(
            lob_features=training_corpus.lob_features,
            auxiliary_features=training_corpus.auxiliary_features,
            labels=training_corpus.labels,
            target_indices=training_corpus.target_indices,
            context_steps=protocol.context_milliseconds
            // protocol.state_interval_milliseconds,
        ).to(device)
        evaluation_dataset = model_domain.SequenceDataset(
            lob_features=evaluation_corpus.lob_features,
            auxiliary_features=evaluation_corpus.auxiliary_features,
            labels=evaluation_corpus.labels,
            target_indices=evaluation_corpus.target_indices,
            context_steps=protocol.context_milliseconds
            // protocol.state_interval_milliseconds,
        ).to(device)
        baseline_frames.append(
            _baseline_outputs(
                protocol=protocol,
                training_corpus=training_corpus,
                evaluation_corpus=evaluation_corpus,
                final_sessions=final_sessions,
                depth=depth,
                horizon_steps=horizon_steps,
            )
        )
        for cell in cells:
            completed = _completed_locked_cell(
                output_dir=output_dir,
                cell=cell,
                plan_sha256=plan_sha256,
                final_sessions=final_sessions,
            )
            completed_cells += 1
            if completed is not None:
                all_metrics.append(completed)
                continue
            print(
                f"[locked-evaluation] cell={completed_cells}/{len(raw_cells)} "
                f"model={_string(cell, 'model')} seed={_integer(cell, 'seed')} "
                f"fixed_epochs={_integer(cell, 'fixed_epochs')}",
                flush=True,
            )
            all_metrics.append(
                _fit_locked_cell(
                    output_dir=output_dir,
                    cell=cell,
                    plan_sha256=plan_sha256,
                    policy=policy,
                    protocol=protocol,
                    device=device,
                    training_dataset=training_dataset,
                    evaluation_dataset=evaluation_dataset,
                    evaluation_corpus=evaluation_corpus,
                    final_sessions=final_sessions,
                )
            )
            new_cells += 1
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if (
                maximum_new_cells is not None
                and new_cells >= maximum_new_cells
                and completed_cells < len(raw_cells)
            ):
                raise LockedEvaluationPausedError(
                    completed_cells=completed_cells,
                    total_cells=len(raw_cells),
                    output_dir=output_dir,
                )
        del training_dataset, evaluation_dataset
        del training_corpus, evaluation_corpus
    metrics = pl.concat([*all_metrics, *baseline_frames]).sort(
        ["model", "seed", "validation_date"]
    )
    metrics_path = output_dir / "session_metrics.csv"
    _neural._write_csv_atomically(data=metrics, path=metrics_path)
    comparisons = _locked_comparisons(metrics=metrics, protocol=protocol)
    comparisons_path = output_dir / "paired_baseline_comparisons.csv"
    _neural._write_csv_atomically(data=comparisons, path=comparisons_path)
    decision = _locked_decision(comparisons=comparisons, protocol=protocol)
    decision_path = output_dir / "decision.json"
    _neural._write_text_atomically(
        text=json.dumps(decision, indent=2), path=decision_path
    )
    elapsed = time.perf_counter() - started
    passed = cast(list[dict[str, object]], decision["confirmed_candidates"])
    terminal = "\n".join(
        (
            "EBS one-time locked evaluation completed",
            "WARNING: locked outcomes have now been inspected; do not tune and rerun.",
            f"instrument={protocol.development_instrument.value}",
            f"device={device}",
            f"final_test_sessions={len(final_sessions)}",
            f"model_cells={len(raw_cells)}",
            f"confirmed_candidates={len(passed)}",
            f"elapsed_seconds={elapsed:.2f}",
            f"outputs={output_dir}",
        )
    )
    terminal_path = output_dir / "terminal_summary.txt"
    _neural._write_text_atomically(text=terminal, path=terminal_path)
    _neural._write_text_atomically(
        text=json.dumps(
            {
                "schema_version": 1,
                "implementation_version": LOCKED_EVALUATION_IMPLEMENTATION_VERSION,
                "plan_sha256": plan_sha256,
                "locked_evaluation_used": True,
                "environment": {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "torch": torch.__version__,
                    "sklearn": sklearn.__version__,
                    "device": str(device),
                },
                "elapsed_seconds": elapsed,
                "artifacts": {
                    "session_metrics": str(metrics_path),
                    "paired_baseline_comparisons": str(comparisons_path),
                    "decision": str(decision_path),
                },
            },
            indent=2,
        ),
        path=output_dir / "run_summary.json",
    )
    print(terminal)
    return LockedEvaluationResult(
        output_dir=output_dir,
        metrics_path=metrics_path,
        comparisons_path=comparisons_path,
        decision_path=decision_path,
        terminal_summary_path=terminal_path,
    )


def _development_sessions(
    *, folds: tuple[research_models.RollingFold, ...]
) -> tuple[research_models.SessionIdentity, ...]:
    by_date = {
        item.trading_date: item
        for fold in folds
        for item in (*fold.training_sessions, *fold.validation_sessions)
    }
    return tuple(by_date[key] for key in sorted(by_date))


def _final_sessions(
    *, manifest_path: Path, protocol: research_models.ResearchProtocol
) -> tuple[research_models.SessionIdentity, ...]:
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = yaml.safe_load(stream)
    if not isinstance(manifest, dict):
        raise ValueError("split manifest must be a mapping")
    final = manifest.get("final_test_sessions")
    if not isinstance(final, dict):
        raise ValueError("manifest final_test_sessions must be a mapping")
    sessions = _baseline._manifest_sessions(
        value=final.get(protocol.development_instrument.value), protocol=protocol
    )
    if not sessions:
        raise ValueError("manifest has no final locked test sessions")
    return sessions


def _identity_payload(
    *, item: research_models.SessionIdentity, protocol: research_models.ResearchProtocol
) -> dict[str, str]:
    return {
        "trading_date": item.trading_date.isoformat(),
        "raw_path": str(item.path.relative_to(protocol.data_dir)),
        "sha256": item.sha256,
    }


def _planned_sessions(
    *, value: object, protocol: research_models.ResearchProtocol
) -> tuple[research_models.SessionIdentity, ...]:
    return _baseline._manifest_sessions(value=value, protocol=protocol)


def _verify_plan_inputs(
    *,
    plan: dict[str, object],
    protocol_path: Path,
    policy_path: Path,
    protocol: research_models.ResearchProtocol,
) -> None:
    paths = {
        "protocol_sha256": protocol_path,
        "policy_sha256": policy_path,
        "manifest_sha256": protocol.output_dir / "split_manifest.yaml",
        "audit_sha256": protocol.output_dir / "session_audit.csv",
        "neural_run_summary_sha256": (
            protocol.output_dir / "neural_benchmark" / "run_summary.json"
        ),
        "neural_gate_sha256": (
            protocol.output_dir / "neural_benchmark" / "gate_decision.json"
        ),
        "neural_run_identity_sha256": (
            protocol.output_dir / "neural_benchmark" / "run_identity.json"
        ),
        "neural_comparisons_sha256": (
            protocol.output_dir
            / "neural_benchmark"
            / "paired_baseline_comparisons.csv"
        ),
    }
    if any(
        not path.is_file() or plan.get(key) != _baseline._sha256_file(path=path)
        for key, path in paths.items()
    ):
        raise ValueError("frozen plan inputs changed after plan creation")
    if plan.get("locked_outcomes_inspected") is not False:
        raise ValueError("frozen plan is not pre-evaluation evidence")


def _verify_sources(
    *, sessions: tuple[research_models.SessionIdentity, ...]
) -> None:
    for item in sessions:
        if not item.path.is_file() or _baseline._sha256_file(
            path=item.path
        ) != item.sha256:
            raise ValueError(f"raw source does not match frozen plan: {item.path}")


def _materialize_locked_cache(
    *,
    protocol: research_models.ResearchProtocol,
    sessions: tuple[research_models.SessionIdentity, ...],
) -> None:
    cache_dir = protocol.output_dir / "locked_evaluation" / "native_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for position, identity in enumerate(sessions, start=1):
        cache_path = cache_dir / f"{identity.trading_date.isoformat()}.parquet"
        hash_path = cache_path.with_suffix(".sha256")
        if cache_path.is_file() and hash_path.is_file():
            if hash_path.read_text(encoding="utf-8").strip() == _baseline._sha256_file(
                path=cache_path
            ):
                continue
            raise ValueError(
                f"locked native cache failed integrity check: {cache_path}"
            )
        print(
            f"[locked-evaluation] reconstruct={position}/{len(sessions)} "
            f"date={identity.trading_date.isoformat()}",
            flush=True,
        )
        with closing(
            ebs_csv.parse_rows(
                path=identity.path,
                expected_instrument=identity.instrument,
                expected_trading_date=identity.trading_date,
            )
        ) as records:
            states = pilot_operations.build_native_states(
                records=records,
                instrument=identity.instrument,
                trading_date=identity.trading_date,
                grid_steps=None,
                maximum_staleness_steps=(
                    protocol.maximum_staleness_milliseconds
                    // protocol.state_interval_milliseconds
                ),
                maximum_depth=protocol.audit_policy.required_depth,
            )
        temporary = cache_path.with_suffix(".parquet.tmp")
        states.write_parquet(temporary)
        temporary.replace(cache_path)
        _neural._write_text_atomically(
            text=_baseline._sha256_file(path=cache_path), path=hash_path
        )


def _extract_final_session(
    *,
    protocol: research_models.ResearchProtocol,
    identity: research_models.SessionIdentity,
    depth: int,
    horizon_steps: int,
) -> pilot_training.RawSessionData:
    cache_path = (
        protocol.output_dir
        / "locked_evaluation"
        / "native_cache"
        / f"{identity.trading_date.isoformat()}.parquet"
    )
    states = pilot_operations.add_direction_targets(
        data=pl.read_parquet(cache_path), horizon_steps=(horizon_steps,)
    )
    return pilot_training.extract_session(
        data=states,
        trading_date=identity.trading_date,
        depth=depth,
        horizon_steps=horizon_steps,
    )


def _prepare_final_corpora(
    *,
    protocol: research_models.ResearchProtocol,
    development_sessions: tuple[research_models.SessionIdentity, ...],
    final_sessions: tuple[research_models.SessionIdentity, ...],
    depth: int,
    horizon_steps: int,
) -> tuple[pilot_training.PreparedCorpus, pilot_training.PreparedCorpus]:
    scaler = pilot_training.fit_feature_scaler(
        sessions=(
            _baseline._extract_session(
                protocol=protocol,
                identity=item,
                depth=depth,
                horizon_steps=horizon_steps,
            )
            for item in development_sessions
        )
    )
    context_steps = (
        protocol.context_milliseconds // protocol.state_interval_milliseconds
    )
    training = pilot_training.combine_sessions(
        sessions=tuple(
            pilot_training.apply_feature_scaler(
                session=_baseline._extract_session(
                    protocol=protocol,
                    identity=item,
                    depth=depth,
                    horizon_steps=horizon_steps,
                ),
                scaler=scaler,
            )
            for item in development_sessions
        ),
        context_steps=context_steps,
        horizon_steps=horizon_steps,
        maximum_windows=None,
        stride_steps=research_operations.training_stride_steps(
            protocol=protocol,
            horizon_milliseconds=(
                horizon_steps * protocol.state_interval_milliseconds
            ),
        ),
    )
    evaluation = pilot_training.combine_sessions(
        sessions=tuple(
            pilot_training.apply_feature_scaler(
                session=_extract_final_session(
                    protocol=protocol,
                    identity=item,
                    depth=depth,
                    horizon_steps=horizon_steps,
                ),
                scaler=scaler,
            )
            for item in final_sessions
        ),
        context_steps=context_steps,
        horizon_steps=horizon_steps,
        maximum_windows=None,
        stride_steps=(
            protocol.evaluation_stride_milliseconds
            // protocol.state_interval_milliseconds
        ),
    )
    return training, evaluation


def _group_cells(
    *, raw_cells: list[object]
) -> dict[tuple[int, int], list[dict[str, object]]]:
    grouped: dict[tuple[int, int], list[dict[str, object]]] = {}
    for raw in raw_cells:
        if not isinstance(raw, dict):
            raise ValueError("frozen plan cell is invalid")
        cell = cast(dict[str, object], raw)
        key = (_integer(cell, "depth"), _integer(cell, "horizon_steps"))
        grouped.setdefault(key, []).append(cell)
    return grouped


def _cell_dir(*, output_dir: Path, cell: dict[str, object]) -> Path:
    return (
        output_dir
        / "cells"
        / f"h{_integer(cell, 'horizon_milliseconds')}"
        / f"depth_{_integer(cell, 'depth')}"
        / _string(cell, "model")
        / f"seed_{_integer(cell, 'seed')}"
    )


def _cell_fingerprint(*, cell: dict[str, object], plan_sha256: str) -> str:
    return hashlib.sha256(
        json.dumps(
            {"plan_sha256": plan_sha256, "cell": cell}, sort_keys=True
        ).encode()
    ).hexdigest()


def _completed_locked_cell(
    *,
    output_dir: Path,
    cell: dict[str, object],
    plan_sha256: str,
    final_sessions: tuple[research_models.SessionIdentity, ...],
) -> pl.DataFrame | None:
    cell_dir = _cell_dir(output_dir=output_dir, cell=cell)
    summary_path = cell_dir / "cell_summary.json"
    if not summary_path.is_file():
        return None
    summary = _neural._json_mapping(path=summary_path)
    if summary.get("fingerprint") != _cell_fingerprint(
        cell=cell, plan_sha256=plan_sha256
    ):
        raise ValueError(f"locked cell fingerprint mismatch: {cell_dir}")
    required = (cell_dir / "final.pt", cell_dir / "session_metrics.csv")
    hashes = summary.get("artifact_sha256")
    if not isinstance(hashes, dict) or any(
        hashes.get(path.name) != _baseline._sha256_file(path=path) for path in required
    ):
        raise ValueError(f"locked cell artifact integrity failure: {cell_dir}")
    metrics = pl.read_csv(
        cell_dir / "session_metrics.csv",
        schema_overrides={"validation_date": pl.Date},
    )
    if set(metrics["validation_date"].to_list()) != {
        item.trading_date for item in final_sessions
    }:
        raise ValueError(f"locked cell has invalid session coverage: {cell_dir}")
    return metrics


def _fit_locked_cell(
    *,
    output_dir: Path,
    cell: dict[str, object],
    plan_sha256: str,
    policy: research_models.NeuralBenchmarkPolicy,
    protocol: research_models.ResearchProtocol,
    device: torch.device,
    training_dataset: model_domain.SequenceDataset,
    evaluation_dataset: model_domain.SequenceDataset,
    evaluation_corpus: pilot_training.PreparedCorpus,
    final_sessions: tuple[research_models.SessionIdentity, ...],
) -> pl.DataFrame:
    cell_dir = _cell_dir(output_dir=output_dir, cell=cell)
    cell_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = _cell_fingerprint(cell=cell, plan_sha256=plan_sha256)
    seed = _integer(cell, "seed")
    model_name = _string(cell, "model")
    model_domain.set_random_seed(seed=seed)
    classifier = model_domain.build_direction_classifier(
        model_name=model_name,
        auxiliary_size=evaluation_corpus.auxiliary_features.shape[1],
        hidden_size=policy.hidden_size,
    )
    latest_path = cell_dir / "latest.pt"
    resume_state = _read_fixed_checkpoint(path=latest_path, fingerprint=fingerprint)
    result = model_domain.fit_classifier_fixed_epochs(
        classifier=classifier,
        training_data=training_dataset,
        device=device,
        epochs=_integer(cell, "fixed_epochs"),
        batch_size=policy.batch_size,
        learning_rate=policy.learning_rate,
        weight_decay=policy.weight_decay,
        gradient_clip_norm=policy.gradient_clip_norm,
        random_seed=seed,
        resume_state=resume_state,
        epoch_observer=lambda metric: print(
            f"[locked-evaluation:{model_name}:s{seed}] "
            f"epoch={metric.epoch}/{_integer(cell, 'fixed_epochs')} "
            f"step={metric.optimizer_step} loss={metric.training_loss:.6f} "
            f"gradient_norm={metric.gradient_norm:.4f}",
            flush=True,
        ),
        checkpoint_observer=lambda state: _write_fixed_checkpoint(
            path=latest_path, fingerprint=fingerprint, state=state
        ),
    )
    final_path = cell_dir / "final.pt"
    checkpoint_repository.write(
        path=final_path,
        payload={
            "kind": "fixed_final_model",
            "fingerprint": fingerprint,
            "classifier_state": result.latest_state.classifier_state,
            "fixed_epochs": result.epochs_completed,
        },
    )
    prediction = model_domain.predict_classifier(
        classifier=classifier,
        dataset=evaluation_dataset,
        device=device,
        batch_size=policy.evaluation_batch_size,
    )
    metrics = _neural_outputs(
        protocol=protocol,
        corpus=evaluation_corpus,
        sessions=final_sessions,
        probabilities=prediction.probabilities,
        model_name=model_name,
        depth=_integer(cell, "depth"),
        horizon_steps=_integer(cell, "horizon_steps"),
        seed=seed,
        parameter_count=model_domain.parameter_count(classifier=classifier),
    )
    metrics_path = cell_dir / "session_metrics.csv"
    _neural._write_csv_atomically(data=metrics, path=metrics_path)
    _neural._write_text_atomically(
        text=json.dumps(
            {
                "fingerprint": fingerprint,
                "fixed_epochs": result.epochs_completed,
                "fit_elapsed_seconds": result.fit_elapsed_seconds,
                "locked_evaluation_used": True,
                "artifact_sha256": {
                    path.name: _baseline._sha256_file(path=path)
                    for path in (final_path, metrics_path)
                },
            },
            indent=2,
        ),
        path=cell_dir / "cell_summary.json",
    )
    return metrics


def _write_fixed_checkpoint(
    *, path: Path, fingerprint: str, state: model_domain.FixedTrainingState
) -> None:
    checkpoint_repository.write(
        path=path,
        payload={
            "kind": "fixed_training_state",
            "fingerprint": fingerprint,
            "epoch": state.epoch,
            "classifier_state": state.classifier_state,
            "optimizer_state": state.optimizer_state,
            "history": [attrs.asdict(item) for item in state.history],
            "torch_random_state": state.torch_random_state,
        },
    )


def _read_fixed_checkpoint(
    *, path: Path, fingerprint: str
) -> model_domain.FixedTrainingState | None:
    if not path.is_file():
        return None
    payload = checkpoint_repository.read(path=path)
    if payload.get("kind") != "fixed_training_state" or payload.get(
        "fingerprint"
    ) != fingerprint:
        raise ValueError(f"incompatible locked-evaluation checkpoint: {path}")
    raw_history = payload.get("history")
    classifier_state = payload.get("classifier_state")
    optimizer_state = payload.get("optimizer_state")
    random_state = payload.get("torch_random_state")
    if (
        not isinstance(raw_history, list)
        or not isinstance(classifier_state, dict)
        or not isinstance(optimizer_state, dict)
        or not isinstance(random_state, torch.Tensor)
    ):
        raise ValueError(f"invalid locked-evaluation checkpoint: {path}")
    history = tuple(
        model_domain.FixedEpochMetric(
            epoch=_integer(cast(dict[str, object], item), "epoch"),
            training_loss=_number(cast(dict[str, object], item), "training_loss"),
            gradient_norm=_number(cast(dict[str, object], item), "gradient_norm"),
            optimizer_step=_integer(cast(dict[str, object], item), "optimizer_step"),
        )
        for item in raw_history
        if isinstance(item, dict)
    )
    if len(history) != len(raw_history):
        raise ValueError(f"invalid locked-evaluation checkpoint history: {path}")
    return model_domain.FixedTrainingState(
        epoch=_integer(payload, "epoch"),
        classifier_state=cast(dict[str, torch.Tensor], classifier_state),
        optimizer_state=cast(dict[str, object], optimizer_state),
        history=history,
        torch_random_state=random_state,
    )


def _neural_outputs(
    *,
    protocol: research_models.ResearchProtocol,
    corpus: pilot_training.PreparedCorpus,
    sessions: tuple[research_models.SessionIdentity, ...],
    probabilities: np.ndarray,
    model_name: str,
    depth: int,
    horizon_steps: int,
    seed: int,
    parameter_count: int,
) -> pl.DataFrame:
    labels = corpus.labels[corpus.target_indices]
    rows: list[dict[str, object]] = []
    for offset, length, identity in zip(
        corpus.session_offsets, corpus.session_lengths, sessions, strict=True
    ):
        selection = (corpus.target_indices >= offset) & (
            corpus.target_indices < offset + length
        )
        row = model_domain.direction_metric_row(
            model_name=model_name,
            depth=depth,
            horizon_steps=horizon_steps,
            seed=seed,
            labels=labels[selection],
            probabilities=probabilities[selection],
            parameter_count=parameter_count,
        )
        row.update(
            {
                "instrument": protocol.development_instrument.value,
                "fold": "locked_final",
                "validation_date": identity.trading_date,
            }
        )
        rows.append(row)
    return pl.DataFrame(rows)


def _baseline_outputs(
    *,
    protocol: research_models.ResearchProtocol,
    training_corpus: pilot_training.PreparedCorpus,
    evaluation_corpus: pilot_training.PreparedCorpus,
    final_sessions: tuple[research_models.SessionIdentity, ...],
    depth: int,
    horizon_steps: int,
) -> pl.DataFrame:
    probabilities = model_domain.fit_defensive_baselines(
        training=training_corpus, evaluation=evaluation_corpus
    )["logistic"]
    return _neural_outputs(
        protocol=protocol,
        corpus=evaluation_corpus,
        sessions=final_sessions,
        probabilities=probabilities,
        model_name="logistic",
        depth=depth,
        horizon_steps=horizon_steps,
        seed=-1,
        parameter_count=0,
    )


def _locked_comparisons(
    *, metrics: pl.DataFrame, protocol: research_models.ResearchProtocol
) -> pl.DataFrame:
    metric_names = tuple(
        item.value for item in (*protocol.primary_metrics, *protocol.supporting_metrics)
    )
    neural = metrics.filter(pl.col("seed") >= 0).group_by(
        ["validation_date", "model", "depth", "horizon_steps", "horizon_milliseconds"]
    ).agg([pl.col(name).mean().alias(name) for name in metric_names])
    baseline = metrics.filter(pl.col("model") == "logistic")
    rows: list[dict[str, object]] = []
    dimensions = neural.select(
        "model", "depth", "horizon_steps", "horizon_milliseconds"
    ).unique()
    for dimension in dimensions.iter_rows(named=True):
        selected = neural.filter(
            (pl.col("model") == dimension["model"])
            & (pl.col("depth") == dimension["depth"])
            & (pl.col("horizon_steps") == dimension["horizon_steps"])
        )
        reference = baseline.filter(
            (pl.col("depth") == dimension["depth"])
            & (pl.col("horizon_steps") == dimension["horizon_steps"])
        )
        for metric_name in metric_names:
            interval = research_operations.paired_session_interval(
                shallower_by_session=_date_metric(reference, metric_name),
                deeper_by_session=_date_metric(selected, metric_name),
                repetitions=protocol.bootstrap_repetitions,
                confidence_level=protocol.confidence_level,
                random_seed=911 + int(dimension["horizon_steps"]),
            )
            rows.append(
                {
                    "comparison": "seed_mean_neural_minus_logistic_same_depth",
                    **dimension,
                    "metric": metric_name,
                    "favorable_direction": _baseline._favorable_direction(
                        metric_name=metric_name
                    ),
                    **interval,
                }
            )
    return pl.DataFrame(rows).sort(["model", "metric"])


def _date_metric(data: pl.DataFrame, metric_name: str) -> dict[str, float]:
    return {
        str(row["validation_date"]): float(row[metric_name])
        for row in data.iter_rows(named=True)
    }


def _locked_decision(
    *, comparisons: pl.DataFrame, protocol: research_models.ResearchProtocol
) -> dict[str, object]:
    primary = tuple(item.value for item in protocol.primary_metrics)
    confirmed: list[dict[str, object]] = []
    evidence: dict[str, bool] = {}
    dimensions = comparisons.select(
        "model", "depth", "horizon_milliseconds"
    ).unique()
    for dimension in dimensions.iter_rows(named=True):
        rows = comparisons.filter(
            (pl.col("model") == dimension["model"])
            & (pl.col("depth") == dimension["depth"])
            & (pl.col("horizon_milliseconds") == dimension["horizon_milliseconds"])
            & pl.col("metric").is_in(primary)
        )
        passed = rows.height == len(primary) and bool(
            (rows["confidence_lower"] > 0.0).all()
        )
        key = (
            f"{dimension['model']}:d{dimension['depth']}:"
            f"h{dimension['horizon_milliseconds']}"
        )
        evidence[key] = passed
        if passed:
            confirmed.append(dict(dimension))
    return {
        "decision_rule": (
            "The frozen candidate is confirmed only when the across-seed mean has "
            "a strictly positive lower paired-session bootstrap confidence bound "
            "over the same-data logistic baseline for every primary metric."
        ),
        "primary_metrics": list(primary),
        "confirmed_by_candidate": evidence,
        "confirmed_candidates": confirmed,
        "locked_evaluation_used": True,
        "retuning_permitted": False,
    }


def _string(data: dict[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _integer(data: dict[str, object], key: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _number(data: dict[str, object], key: str) -> float:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{key} must be numeric")
    if not math.isfinite(float(value)):
        raise ValueError(f"{key} must be finite")
    return float(value)
