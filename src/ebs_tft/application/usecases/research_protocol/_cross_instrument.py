"""Freeze and run cross-instrument transfer evaluation without retraining."""

from __future__ import annotations

import datetime
import hashlib
import json
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

from ebs_tft.application.usecases.research_protocol import _baseline, _locked, _neural
from ebs_tft.data.parsers import ebs_csv
from ebs_tft.data.repositories import checkpoint as checkpoint_repository
from ebs_tft.domain import model as model_domain
from ebs_tft.domain.orderbook import models as orderbook_models
from ebs_tft.domain.pilot import operations as pilot_operations
from ebs_tft.domain.pilot import training as pilot_training
from ebs_tft.domain.research import models as research_models
from ebs_tft.domain.research import operations as research_operations

CROSS_INSTRUMENT_IMPLEMENTATION_VERSION = 1


@attrs.frozen
class CrossInstrumentPlanResult:
    """Reference an outcome-blind frozen cross-instrument plan."""

    plan_path: Path
    plan_sha256: str


@attrs.frozen
class CrossInstrumentResult:
    """Reference completed cross-instrument transfer artifacts."""

    output_dir: Path
    metrics_path: Path
    comparisons_path: Path
    decision_path: Path
    terminal_summary_path: Path


class CrossInstrumentPausedError(Exception):
    """Indicate a safe pause after one or more inference cells."""

    def __init__(self, *, completed_cells: int, total_cells: int, output_dir: Path):
        self.completed_cells = completed_cells
        self.total_cells = total_cells
        self.output_dir = output_dir
        super().__init__(
            f"Cross-instrument evaluation paused safely after "
            f"{completed_cells}/{total_cells} cells; resume with the same plan hash. "
            f"Outputs: {output_dir}"
        )


def freeze_plan(
    *,
    protocol: research_models.ResearchProtocol,
    protocol_path: Path,
    policy_path: Path,
) -> CrossInstrumentPlanResult:
    """Freeze confirmed models, checkpoints, and target session identities."""
    output_dir = protocol.output_dir / "cross_instrument_evaluation"
    locked_dir = protocol.output_dir / "locked_evaluation"
    manifest_path = protocol.output_dir / "split_manifest.yaml"
    audit_path = protocol.output_dir / "session_audit.csv"
    locked_plan_path = locked_dir / "plan.json"
    locked_summary_path = locked_dir / "run_summary.json"
    locked_decision_path = locked_dir / "decision.json"
    locked_comparisons_path = locked_dir / "paired_baseline_comparisons.csv"
    required = (
        manifest_path,
        audit_path,
        locked_plan_path,
        locked_summary_path,
        locked_decision_path,
        locked_comparisons_path,
    )
    if any(not path.is_file() for path in required):
        raise FileNotFoundError(
            "complete and preserve the locked EUR/USD evaluation first"
        )
    locked_plan = _neural._json_mapping(path=locked_plan_path)
    locked_summary = _neural._json_mapping(path=locked_summary_path)
    locked_decision = _neural._json_mapping(path=locked_decision_path)
    recomputed_decision = _locked._locked_decision(
        comparisons=pl.read_csv(locked_comparisons_path), protocol=protocol
    )
    if _normalized_locked_decision(locked_decision) != _normalized_locked_decision(
        recomputed_decision
    ):
        raise ValueError("locked decision does not match its comparison evidence")
    if (
        locked_summary.get("locked_evaluation_used") is not True
        or locked_decision.get("locked_evaluation_used") is not True
        or locked_decision.get("retuning_permitted") is not False
    ):
        raise ValueError("locked EUR/USD evaluation is not valid final evidence")
    if locked_summary.get("plan_sha256") != _baseline._sha256_file(
        path=locked_plan_path
    ):
        raise ValueError("locked EUR/USD result does not match its frozen plan")
    if locked_plan.get("protocol_sha256") != _baseline._sha256_file(
        path=protocol_path
    ) or locked_plan.get("policy_sha256") != _baseline._sha256_file(path=policy_path):
        raise ValueError("locked EUR/USD result belongs to different inputs")
    raw_confirmed = locked_decision.get("confirmed_candidates")
    if not isinstance(raw_confirmed, list) or not raw_confirmed:
        raise ValueError("locked EUR/USD evaluation confirmed no transfer candidate")
    raw_locked_cells = locked_plan.get("cells")
    if not isinstance(raw_locked_cells, list):
        raise ValueError("locked EUR/USD plan has invalid cells")
    target_instruments = tuple(
        item
        for item in protocol.instruments
        if item is not protocol.development_instrument
    )
    if not target_instruments:
        raise ValueError("protocol declares no cross-instrument target")
    target_sessions = _target_sessions(
        manifest_path=manifest_path,
        protocol=protocol,
        instruments=target_instruments,
    )
    development_sessions = _sessions_from_payload(
        value=locked_plan.get("development_sessions"),
        protocol=protocol,
        instrument=protocol.development_instrument,
    )
    cells: list[dict[str, object]] = []
    for raw_candidate in raw_confirmed:
        if not isinstance(raw_candidate, dict):
            raise ValueError("locked confirmed candidate is invalid")
        candidate = cast(dict[str, object], raw_candidate)
        model_name = _locked._string(candidate, "model")
        depth = _locked._integer(candidate, "depth")
        horizon_milliseconds = _locked._integer(
            candidate, "horizon_milliseconds"
        )
        matching_locked_cells = [
            cast(dict[str, object], item)
            for item in raw_locked_cells
            if isinstance(item, dict)
            and item.get("model") == model_name
            and item.get("depth") == depth
            and item.get("horizon_milliseconds") == horizon_milliseconds
        ]
        if {item.get("seed") for item in matching_locked_cells} != set(
            protocol.random_seeds
        ):
            raise ValueError("confirmed candidate lacks every frozen seed checkpoint")
        for locked_cell in matching_locked_cells:
            source_dir = _locked._cell_dir(
                output_dir=locked_dir, cell=locked_cell
            )
            checkpoint_path = source_dir / "final.pt"
            summary_path = source_dir / "cell_summary.json"
            if not checkpoint_path.is_file() or not summary_path.is_file():
                raise ValueError(f"final checkpoint is missing: {source_dir}")
            cell_summary = _neural._json_mapping(path=summary_path)
            hashes = cell_summary.get("artifact_sha256")
            checkpoint_hash = _baseline._sha256_file(path=checkpoint_path)
            if (
                not isinstance(hashes, dict)
                or hashes.get("final.pt") != checkpoint_hash
            ):
                raise ValueError(
                    f"final checkpoint failed integrity check: {source_dir}"
                )
            for instrument in target_instruments:
                cells.append(
                    {
                        "instrument": instrument.value,
                        "model": model_name,
                        "depth": depth,
                        "horizon_milliseconds": horizon_milliseconds,
                        "horizon_steps": (
                            horizon_milliseconds
                            // protocol.state_interval_milliseconds
                        ),
                        "seed": _locked._integer(locked_cell, "seed"),
                        "source_checkpoint": str(
                            checkpoint_path.relative_to(protocol.output_dir)
                        ),
                        "source_checkpoint_sha256": checkpoint_hash,
                    }
                )
    input_hashes = {
        "protocol_sha256": _baseline._sha256_file(path=protocol_path),
        "policy_sha256": _baseline._sha256_file(path=policy_path),
        "manifest_sha256": _baseline._sha256_file(path=manifest_path),
        "audit_sha256": _baseline._sha256_file(path=audit_path),
        "locked_plan_sha256": _baseline._sha256_file(path=locked_plan_path),
        "locked_run_summary_sha256": _baseline._sha256_file(
            path=locked_summary_path
        ),
        "locked_decision_sha256": _baseline._sha256_file(
            path=locked_decision_path
        ),
        "locked_comparisons_sha256": _baseline._sha256_file(
            path=locked_comparisons_path
        ),
    }
    plan = {
        "schema_version": 1,
        "implementation_version": CROSS_INSTRUMENT_IMPLEMENTATION_VERSION,
        **input_hashes,
        "source_instrument": protocol.development_instrument.value,
        "target_instruments": [item.value for item in target_instruments],
        "development_sessions": [
            _identity_payload(item=item, protocol=protocol)
            for item in development_sessions
        ],
        "target_sessions": {
            instrument.value: [
                _identity_payload(item=item, protocol=protocol)
                for item in target_sessions[instrument]
            ]
            for instrument in target_instruments
        },
        "cells": sorted(
            cells,
            key=lambda item: (
                cast(str, item["instrument"]),
                cast(str, item["model"]),
                cast(int, item["seed"]),
            ),
        ),
        "evaluation_rule": (
            "Reuse the exact frozen EUR/USD final checkpoints without retraining. "
            "For each target instrument, the across-seed neural mean must have a "
            "strictly positive lower paired-session bootstrap confidence bound "
            "over the EUR/USD-trained logistic reference for every primary metric."
        ),
        "primary_metrics": [item.value for item in protocol.primary_metrics],
        "supporting_metrics": [item.value for item in protocol.supporting_metrics],
        "bootstrap_repetitions": protocol.bootstrap_repetitions,
        "confidence_level": protocol.confidence_level,
        "neural_retraining_permitted": False,
        "target_outcomes_inspected": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "plan.json"
    serialized = json.dumps(plan, indent=2)
    if plan_path.is_file() and plan_path.read_text(encoding="utf-8") != serialized:
        raise ValueError("a different cross-instrument plan is already frozen")
    _neural._write_text_atomically(text=serialized, path=plan_path)
    plan_hash = _baseline._sha256_file(path=plan_path)
    print("EBS cross-instrument evaluation plan frozen")
    print("WARNING: target-instrument outcomes were not inspected.")
    print(f"source_instrument={protocol.development_instrument.value}")
    print("target_instruments=" + ",".join(item.value for item in target_instruments))
    print(
        "target_sessions="
        + ",".join(
            f"{item.value}:{len(target_sessions[item])}"
            for item in target_instruments
        )
    )
    print(f"inference_cells={len(cells)}")
    print(f"plan_sha256={plan_hash}")
    return CrossInstrumentPlanResult(plan_path=plan_path, plan_sha256=plan_hash)


def run(
    *,
    protocol: research_models.ResearchProtocol,
    protocol_path: Path,
    policy: research_models.NeuralBenchmarkPolicy,
    policy_path: Path,
    plan_sha256: str,
    maximum_new_cells: int | None = None,
) -> CrossInstrumentResult:
    """Score frozen checkpoints on every frozen target instrument."""
    if maximum_new_cells is not None and (
        isinstance(maximum_new_cells, bool) or maximum_new_cells <= 0
    ):
        raise ValueError("maximum_new_cells must be positive or null")
    started = time.perf_counter()
    output_dir = protocol.output_dir / "cross_instrument_evaluation"
    plan_path = output_dir / "plan.json"
    if not plan_path.is_file() or _baseline._sha256_file(path=plan_path) != plan_sha256:
        raise ValueError("--plan-sha256 must match the frozen cross-instrument plan")
    if (output_dir / "run_summary.json").is_file():
        raise ValueError("cross-instrument evaluation is complete and cannot be rerun")
    plan = _neural._json_mapping(path=plan_path)
    _verify_plan_inputs(
        plan=plan,
        protocol=protocol,
        protocol_path=protocol_path,
        policy_path=policy_path,
    )
    development_sessions = _sessions_from_payload(
        value=plan.get("development_sessions"),
        protocol=protocol,
        instrument=protocol.development_instrument,
    )
    raw_target_sessions = plan.get("target_sessions")
    if not isinstance(raw_target_sessions, dict):
        raise ValueError("cross-instrument plan target sessions are invalid")
    target_sessions = {
        orderbook_models.Instrument(instrument): _sessions_from_payload(
            value=value,
            protocol=protocol,
            instrument=orderbook_models.Instrument(instrument),
        )
        for instrument, value in raw_target_sessions.items()
        if isinstance(instrument, str)
    }
    _verify_sources(
        sessions=(
            *development_sessions,
            *(item for sessions in target_sessions.values() for item in sessions),
        )
    )
    for instrument, sessions in target_sessions.items():
        _materialize_target_cache(
            protocol=protocol, instrument=instrument, sessions=sessions
        )
    raw_cells = plan.get("cells")
    if not isinstance(raw_cells, list) or not raw_cells:
        raise ValueError("cross-instrument plan has no inference cells")
    cells = [
        cast(dict[str, object], item) for item in raw_cells if isinstance(item, dict)
    ]
    if len(cells) != len(raw_cells):
        raise ValueError("cross-instrument plan contains an invalid cell")
    device = model_domain.select_device(requested=policy.device)
    metric_frames: list[pl.DataFrame] = []
    baseline_frames: list[pl.DataFrame] = []
    completed_cells = 0
    new_cells = 0
    dimensions = sorted(
        {
            (_locked._integer(cell, "depth"), _locked._integer(cell, "horizon_steps"))
            for cell in cells
        }
    )
    for depth, horizon_steps in dimensions:
        scaler, training_corpus = _prepare_source_training(
            protocol=protocol,
            development_sessions=development_sessions,
            depth=depth,
            horizon_steps=horizon_steps,
        )
        for instrument, sessions in target_sessions.items():
            evaluation_corpus = _prepare_target_corpus(
                protocol=protocol,
                scaler=scaler,
                instrument=instrument,
                sessions=sessions,
                depth=depth,
                horizon_steps=horizon_steps,
            )
            evaluation_dataset = model_domain.SequenceDataset(
                lob_features=evaluation_corpus.lob_features,
                auxiliary_features=evaluation_corpus.auxiliary_features,
                labels=evaluation_corpus.labels,
                target_indices=evaluation_corpus.target_indices,
                context_steps=(
                    protocol.context_milliseconds
                    // protocol.state_interval_milliseconds
                ),
            ).to(device)
            baseline_frames.append(
                _baseline_outputs(
                    protocol=protocol,
                    training_corpus=training_corpus,
                    evaluation_corpus=evaluation_corpus,
                    sessions=sessions,
                    instrument=instrument,
                    depth=depth,
                    horizon_steps=horizon_steps,
                )
            )
            selected_cells = [
                cell
                for cell in cells
                if cell.get("instrument") == instrument.value
                and cell.get("depth") == depth
                and cell.get("horizon_steps") == horizon_steps
            ]
            for cell in selected_cells:
                completed_cells += 1
                completed = _completed_cell(
                    output_dir=output_dir,
                    cell=cell,
                    plan_sha256=plan_sha256,
                    sessions=sessions,
                )
                if completed is not None:
                    metric_frames.append(completed)
                    continue
                print(
                    f"[cross-instrument] cell={completed_cells}/{len(cells)} "
                    f"instrument={instrument.value} "
                    f"model={_locked._string(cell, 'model')} "
                    f"seed={_locked._integer(cell, 'seed')}",
                    flush=True,
                )
                metric_frames.append(
                    _evaluate_cell(
                        output_dir=output_dir,
                        cell=cell,
                        plan_sha256=plan_sha256,
                        protocol=protocol,
                        policy=policy,
                        device=device,
                        dataset=evaluation_dataset,
                        corpus=evaluation_corpus,
                        sessions=sessions,
                        instrument=instrument,
                    )
                )
                new_cells += 1
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                if (
                    maximum_new_cells is not None
                    and new_cells >= maximum_new_cells
                    and completed_cells < len(cells)
                ):
                    raise CrossInstrumentPausedError(
                        completed_cells=completed_cells,
                        total_cells=len(cells),
                        output_dir=output_dir,
                    )
            del evaluation_dataset, evaluation_corpus
        del training_corpus
    metrics = pl.concat([*metric_frames, *baseline_frames]).sort(
        ["instrument", "model", "seed", "validation_date"]
    )
    metrics_path = output_dir / "session_metrics.csv"
    _neural._write_csv_atomically(data=metrics, path=metrics_path)
    comparisons = _comparisons(metrics=metrics, protocol=protocol)
    comparisons_path = output_dir / "paired_baseline_comparisons.csv"
    _neural._write_csv_atomically(data=comparisons, path=comparisons_path)
    decision = _decision(comparisons=comparisons, protocol=protocol)
    decision_path = output_dir / "decision.json"
    _neural._write_text_atomically(
        text=json.dumps(decision, indent=2), path=decision_path
    )
    elapsed = time.perf_counter() - started
    confirmed = cast(list[dict[str, object]], decision["confirmed_transfers"])
    terminal = "\n".join(
        (
            "EBS frozen cross-instrument evaluation completed",
            "WARNING: target outcomes are now inspected; do not tune and rerun.",
            f"source_instrument={protocol.development_instrument.value}",
            "target_instruments="
            + ",".join(item.value for item in target_sessions),
            f"target_sessions={sum(len(item) for item in target_sessions.values())}",
            f"inference_cells={len(cells)}",
            f"confirmed_transfers={len(confirmed)}",
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
                "implementation_version": CROSS_INSTRUMENT_IMPLEMENTATION_VERSION,
                "plan_sha256": plan_sha256,
                "cross_instrument_outcomes_used": True,
                "neural_retraining_used": False,
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
    return CrossInstrumentResult(
        output_dir=output_dir,
        metrics_path=metrics_path,
        comparisons_path=comparisons_path,
        decision_path=decision_path,
        terminal_summary_path=terminal_path,
    )


def _target_sessions(
    *,
    manifest_path: Path,
    protocol: research_models.ResearchProtocol,
    instruments: tuple[orderbook_models.Instrument, ...],
) -> dict[orderbook_models.Instrument, tuple[research_models.SessionIdentity, ...]]:
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = yaml.safe_load(stream)
    if not isinstance(manifest, dict):
        raise ValueError("split manifest must be a mapping")
    final = manifest.get("final_test_sessions")
    if not isinstance(final, dict):
        raise ValueError("manifest final_test_sessions must be a mapping")
    result = {
        instrument: _sessions_from_payload(
            value=final.get(instrument.value),
            protocol=protocol,
            instrument=instrument,
        )
        for instrument in instruments
    }
    if any(not sessions for sessions in result.values()):
        raise ValueError("each target instrument requires final test sessions")
    expected_dates = {
        item
        for item in protocol.split_policy.locked_evaluation_dates
        if item > protocol.split_policy.development_end_date
    }
    if any(
        {item.trading_date for item in sessions} != expected_dates
        for sessions in result.values()
    ):
        raise ValueError("target instruments do not cover every final locked date")
    return result


def _sessions_from_payload(
    *,
    value: object,
    protocol: research_models.ResearchProtocol,
    instrument: orderbook_models.Instrument,
) -> tuple[research_models.SessionIdentity, ...]:
    if not isinstance(value, list):
        raise ValueError("session payload must be a list")
    sessions: list[research_models.SessionIdentity] = []
    for raw in value:
        if not isinstance(raw, dict) or set(raw) != {
            "trading_date",
            "raw_path",
            "sha256",
        }:
            raise ValueError("session payload has invalid fields")
        item = cast(dict[str, object], raw)
        trading_date = item.get("trading_date")
        raw_path = item.get("raw_path")
        sha256 = item.get("sha256")
        if not all(
            isinstance(field, str) for field in (trading_date, raw_path, sha256)
        ):
            raise ValueError("session payload fields must be strings")
        sessions.append(
            research_models.SessionIdentity(
                instrument=instrument,
                trading_date=datetime.date.fromisoformat(cast(str, trading_date)),
                path=protocol.data_dir / cast(str, raw_path),
                sha256=cast(str, sha256),
            )
        )
    return tuple(sessions)


def _identity_payload(
    *, item: research_models.SessionIdentity, protocol: research_models.ResearchProtocol
) -> dict[str, str]:
    return {
        "trading_date": item.trading_date.isoformat(),
        "raw_path": str(item.path.relative_to(protocol.data_dir)),
        "sha256": item.sha256,
    }


def _verify_plan_inputs(
    *,
    plan: dict[str, object],
    protocol: research_models.ResearchProtocol,
    protocol_path: Path,
    policy_path: Path,
) -> None:
    paths = {
        "protocol_sha256": protocol_path,
        "policy_sha256": policy_path,
        "manifest_sha256": protocol.output_dir / "split_manifest.yaml",
        "audit_sha256": protocol.output_dir / "session_audit.csv",
        "locked_plan_sha256": protocol.output_dir / "locked_evaluation" / "plan.json",
        "locked_run_summary_sha256": (
            protocol.output_dir / "locked_evaluation" / "run_summary.json"
        ),
        "locked_decision_sha256": (
            protocol.output_dir / "locked_evaluation" / "decision.json"
        ),
        "locked_comparisons_sha256": (
            protocol.output_dir
            / "locked_evaluation"
            / "paired_baseline_comparisons.csv"
        ),
    }
    if any(
        not path.is_file() or plan.get(key) != _baseline._sha256_file(path=path)
        for key, path in paths.items()
    ):
        raise ValueError("cross-instrument plan inputs changed after freezing")
    if (
        plan.get("target_outcomes_inspected") is not False
        or plan.get("neural_retraining_permitted") is not False
    ):
        raise ValueError("cross-instrument plan violates the frozen boundary")


def _verify_sources(
    *, sessions: tuple[research_models.SessionIdentity, ...]
) -> None:
    for item in sessions:
        if not item.path.is_file() or _baseline._sha256_file(
            path=item.path
        ) != item.sha256:
            raise ValueError(f"raw source does not match frozen plan: {item.path}")


def _materialize_target_cache(
    *,
    protocol: research_models.ResearchProtocol,
    instrument: orderbook_models.Instrument,
    sessions: tuple[research_models.SessionIdentity, ...],
) -> None:
    cache_dir = (
        protocol.output_dir
        / "cross_instrument_evaluation"
        / "native_cache"
        / instrument.value
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    for position, identity in enumerate(sessions, start=1):
        cache_path = cache_dir / f"{identity.trading_date.isoformat()}.parquet"
        hash_path = cache_path.with_suffix(".sha256")
        if cache_path.is_file() and hash_path.is_file():
            if hash_path.read_text(encoding="utf-8").strip() == _baseline._sha256_file(
                path=cache_path
            ):
                continue
            raise ValueError(f"target cache failed integrity check: {cache_path}")
        print(
            f"[cross-instrument] reconstruct={instrument.value}:"
            f"{position}/{len(sessions)} date={identity.trading_date.isoformat()}",
            flush=True,
        )
        with closing(
            ebs_csv.parse_rows(
                path=identity.path,
                expected_instrument=instrument,
                expected_trading_date=identity.trading_date,
            )
        ) as records:
            states = pilot_operations.build_native_states(
                records=records,
                instrument=instrument,
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


def _prepare_source_training(
    *,
    protocol: research_models.ResearchProtocol,
    development_sessions: tuple[research_models.SessionIdentity, ...],
    depth: int,
    horizon_steps: int,
) -> tuple[pilot_training.FeatureScaler, pilot_training.PreparedCorpus]:
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
        context_steps=(
            protocol.context_milliseconds // protocol.state_interval_milliseconds
        ),
        horizon_steps=horizon_steps,
        maximum_windows=None,
        stride_steps=research_operations.training_stride_steps(
            protocol=protocol,
            horizon_milliseconds=(
                horizon_steps * protocol.state_interval_milliseconds
            ),
        ),
    )
    return scaler, training


def _prepare_target_corpus(
    *,
    protocol: research_models.ResearchProtocol,
    scaler: pilot_training.FeatureScaler,
    instrument: orderbook_models.Instrument,
    sessions: tuple[research_models.SessionIdentity, ...],
    depth: int,
    horizon_steps: int,
) -> pilot_training.PreparedCorpus:
    return pilot_training.combine_sessions(
        sessions=tuple(
            pilot_training.apply_feature_scaler(
                session=_extract_target_session(
                    protocol=protocol,
                    instrument=instrument,
                    identity=item,
                    depth=depth,
                    horizon_steps=horizon_steps,
                ),
                scaler=scaler,
            )
            for item in sessions
        ),
        context_steps=(
            protocol.context_milliseconds // protocol.state_interval_milliseconds
        ),
        horizon_steps=horizon_steps,
        maximum_windows=None,
        stride_steps=(
            protocol.evaluation_stride_milliseconds
            // protocol.state_interval_milliseconds
        ),
    )


def _extract_target_session(
    *,
    protocol: research_models.ResearchProtocol,
    instrument: orderbook_models.Instrument,
    identity: research_models.SessionIdentity,
    depth: int,
    horizon_steps: int,
) -> pilot_training.RawSessionData:
    cache_path = (
        protocol.output_dir
        / "cross_instrument_evaluation"
        / "native_cache"
        / instrument.value
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


def _cell_fingerprint(*, cell: dict[str, object], plan_sha256: str) -> str:
    return hashlib.sha256(
        json.dumps({"plan_sha256": plan_sha256, "cell": cell}, sort_keys=True).encode()
    ).hexdigest()


def _cell_dir(*, output_dir: Path, cell: dict[str, object]) -> Path:
    return (
        output_dir
        / "cells"
        / _locked._string(cell, "instrument")
        / f"h{_locked._integer(cell, 'horizon_milliseconds')}"
        / f"depth_{_locked._integer(cell, 'depth')}"
        / _locked._string(cell, "model")
        / f"seed_{_locked._integer(cell, 'seed')}"
    )


def _completed_cell(
    *,
    output_dir: Path,
    cell: dict[str, object],
    plan_sha256: str,
    sessions: tuple[research_models.SessionIdentity, ...],
) -> pl.DataFrame | None:
    cell_dir = _cell_dir(output_dir=output_dir, cell=cell)
    summary_path = cell_dir / "cell_summary.json"
    if not summary_path.is_file():
        return None
    summary = _neural._json_mapping(path=summary_path)
    if summary.get("fingerprint") != _cell_fingerprint(
        cell=cell, plan_sha256=plan_sha256
    ):
        raise ValueError(f"cross-instrument cell fingerprint mismatch: {cell_dir}")
    metrics_path = cell_dir / "session_metrics.csv"
    hashes = summary.get("artifact_sha256")
    if (
        not metrics_path.is_file()
        or not isinstance(hashes, dict)
        or hashes.get("session_metrics.csv")
        != _baseline._sha256_file(path=metrics_path)
    ):
        raise ValueError(f"cross-instrument cell failed integrity check: {cell_dir}")
    metrics = pl.read_csv(
        metrics_path, schema_overrides={"validation_date": pl.Date}
    )
    if set(metrics["validation_date"].to_list()) != {
        item.trading_date for item in sessions
    }:
        raise ValueError(
            f"cross-instrument cell session coverage is invalid: {cell_dir}"
        )
    return metrics


def _evaluate_cell(
    *,
    output_dir: Path,
    cell: dict[str, object],
    plan_sha256: str,
    protocol: research_models.ResearchProtocol,
    policy: research_models.NeuralBenchmarkPolicy,
    device: torch.device,
    dataset: model_domain.SequenceDataset,
    corpus: pilot_training.PreparedCorpus,
    sessions: tuple[research_models.SessionIdentity, ...],
    instrument: orderbook_models.Instrument,
) -> pl.DataFrame:
    checkpoint_path = protocol.output_dir / _locked._string(
        cell, "source_checkpoint"
    )
    if _baseline._sha256_file(path=checkpoint_path) != _locked._string(
        cell, "source_checkpoint_sha256"
    ):
        raise ValueError(f"source checkpoint changed: {checkpoint_path}")
    payload = checkpoint_repository.read(path=checkpoint_path)
    state = payload.get("classifier_state")
    if payload.get("kind") != "fixed_final_model" or not isinstance(state, dict):
        raise ValueError(f"source checkpoint is invalid: {checkpoint_path}")
    model_name = _locked._string(cell, "model")
    classifier = model_domain.build_direction_classifier(
        model_name=model_name,
        auxiliary_size=corpus.auxiliary_features.shape[1],
        hidden_size=policy.hidden_size,
    )
    classifier.load_state_dict(cast(dict[str, torch.Tensor], state))
    prediction = model_domain.predict_classifier(
        classifier=classifier,
        dataset=dataset,
        device=device,
        batch_size=policy.evaluation_batch_size,
    )
    metrics = _metric_outputs(
        protocol=protocol,
        corpus=corpus,
        sessions=sessions,
        instrument=instrument,
        probabilities=prediction.probabilities,
        model_name=model_name,
        depth=_locked._integer(cell, "depth"),
        horizon_steps=_locked._integer(cell, "horizon_steps"),
        seed=_locked._integer(cell, "seed"),
        parameter_count=model_domain.parameter_count(classifier=classifier),
    )
    cell_dir = _cell_dir(output_dir=output_dir, cell=cell)
    cell_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = cell_dir / "session_metrics.csv"
    _neural._write_csv_atomically(data=metrics, path=metrics_path)
    _neural._write_text_atomically(
        text=json.dumps(
            {
                "fingerprint": _cell_fingerprint(
                    cell=cell, plan_sha256=plan_sha256
                ),
                "source_checkpoint_sha256": _locked._string(
                    cell, "source_checkpoint_sha256"
                ),
                "neural_retraining_used": False,
                "target_outcomes_used": True,
                "artifact_sha256": {
                    "session_metrics.csv": _baseline._sha256_file(path=metrics_path)
                },
            },
            indent=2,
        ),
        path=cell_dir / "cell_summary.json",
    )
    return metrics


def _metric_outputs(
    *,
    protocol: research_models.ResearchProtocol,
    corpus: pilot_training.PreparedCorpus,
    sessions: tuple[research_models.SessionIdentity, ...],
    instrument: orderbook_models.Instrument,
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
                "instrument": instrument.value,
                "source_instrument": protocol.development_instrument.value,
                "fold": "cross_instrument_final",
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
    sessions: tuple[research_models.SessionIdentity, ...],
    instrument: orderbook_models.Instrument,
    depth: int,
    horizon_steps: int,
) -> pl.DataFrame:
    probabilities = model_domain.fit_defensive_baselines(
        training=training_corpus, evaluation=evaluation_corpus
    )["logistic"]
    return _metric_outputs(
        protocol=protocol,
        corpus=evaluation_corpus,
        sessions=sessions,
        instrument=instrument,
        probabilities=probabilities,
        model_name="logistic",
        depth=depth,
        horizon_steps=horizon_steps,
        seed=-1,
        parameter_count=0,
    )


def _comparisons(
    *, metrics: pl.DataFrame, protocol: research_models.ResearchProtocol
) -> pl.DataFrame:
    metric_names = tuple(
        item.value for item in (*protocol.primary_metrics, *protocol.supporting_metrics)
    )
    neural = metrics.filter(pl.col("seed") >= 0).group_by(
        [
            "instrument",
            "validation_date",
            "model",
            "depth",
            "horizon_steps",
            "horizon_milliseconds",
        ]
    ).agg([pl.col(name).mean().alias(name) for name in metric_names])
    baseline = metrics.filter(pl.col("model") == "logistic")
    rows: list[dict[str, object]] = []
    dimensions = neural.select(
        "instrument", "model", "depth", "horizon_steps", "horizon_milliseconds"
    ).unique().sort(
        ["instrument", "model", "depth", "horizon_steps", "horizon_milliseconds"]
    )
    for dimension in dimensions.iter_rows(named=True):
        selected = neural.filter(
            (pl.col("instrument") == dimension["instrument"])
            & (pl.col("model") == dimension["model"])
            & (pl.col("depth") == dimension["depth"])
            & (pl.col("horizon_steps") == dimension["horizon_steps"])
        )
        reference = baseline.filter(
            (pl.col("instrument") == dimension["instrument"])
            & (pl.col("depth") == dimension["depth"])
            & (pl.col("horizon_steps") == dimension["horizon_steps"])
        )
        for metric_name in metric_names:
            interval = research_operations.paired_session_interval(
                shallower_by_session=_locked._date_metric(reference, metric_name),
                deeper_by_session=_locked._date_metric(selected, metric_name),
                repetitions=protocol.bootstrap_repetitions,
                confidence_level=protocol.confidence_level,
                random_seed=(
                    1301
                    + int(dimension["horizon_steps"])
                    + sum(ord(char) for char in cast(str, dimension["instrument"]))
                ),
            )
            rows.append(
                {
                    "comparison": "seed_mean_neural_minus_transfer_logistic",
                    **dimension,
                    "metric": metric_name,
                    "favorable_direction": _baseline._favorable_direction(
                        metric_name=metric_name
                    ),
                    **interval,
                }
            )
    return pl.DataFrame(rows).sort(["instrument", "model", "metric"])


def _decision(
    *, comparisons: pl.DataFrame, protocol: research_models.ResearchProtocol
) -> dict[str, object]:
    primary = tuple(item.value for item in protocol.primary_metrics)
    evidence: dict[str, bool] = {}
    confirmed: list[dict[str, object]] = []
    dimensions = comparisons.select(
        "instrument", "model", "depth", "horizon_milliseconds"
    ).unique().sort(["instrument", "model", "depth", "horizon_milliseconds"])
    for dimension in dimensions.iter_rows(named=True):
        rows = comparisons.filter(
            (pl.col("instrument") == dimension["instrument"])
            & (pl.col("model") == dimension["model"])
            & (pl.col("depth") == dimension["depth"])
            & (pl.col("horizon_milliseconds") == dimension["horizon_milliseconds"])
            & pl.col("metric").is_in(primary)
        )
        passed = rows.height == len(primary) and bool(
            (rows["confidence_lower"] > 0.0).all()
        )
        key = (
            f"{dimension['instrument']}:{dimension['model']}:"
            f"d{dimension['depth']}:h{dimension['horizon_milliseconds']}"
        )
        evidence[key] = passed
        if passed:
            confirmed.append(dict(dimension))
    return {
        "decision_rule": (
            "A frozen EUR/USD checkpoint transfers only when its across-seed mean "
            "has a strictly positive lower paired-session bootstrap confidence "
            "bound over the EUR/USD-trained logistic reference for every primary "
            "metric on the target instrument."
        ),
        "primary_metrics": list(primary),
        "confirmed_by_transfer": evidence,
        "confirmed_transfers": confirmed,
        "cross_instrument_outcomes_used": True,
        "neural_retraining_used": False,
        "retuning_permitted": False,
    }


def _normalized_locked_decision(
    decision: dict[str, object],
) -> dict[str, object]:
    """Normalize the semantically unordered candidate list for verification."""
    normalized = dict(decision)
    candidates = normalized.get("confirmed_candidates")
    if isinstance(candidates, list):
        normalized["confirmed_candidates"] = sorted(
            candidates,
            key=lambda item: json.dumps(item, sort_keys=True),
        )
    return normalized
