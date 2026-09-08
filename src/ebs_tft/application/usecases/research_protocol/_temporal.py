"""Audit, freeze, and run the external-year temporal evaluation."""

from __future__ import annotations

import datetime
import hashlib
import json
import platform
import time
from concurrent import futures
from pathlib import Path
from typing import cast

import attrs
import numpy as np
import polars as pl
import sklearn
import torch

from ebs_tft.application.usecases.research_protocol import (
    _audit,
    _baseline,
    _cross_instrument,
    _locked,
    _neural,
)
from ebs_tft.data.repositories import checkpoint as checkpoint_repository
from ebs_tft.data.repositories import raw_file as raw_file_repository
from ebs_tft.domain import model as model_domain
from ebs_tft.domain.orderbook import models as orderbook_models
from ebs_tft.domain.pilot import operations as pilot_operations
from ebs_tft.domain.pilot import training as pilot_training
from ebs_tft.domain.research import models as research_models
from ebs_tft.domain.research import operations as research_operations

TEMPORAL_EVALUATION_IMPLEMENTATION_VERSION = 1


@attrs.frozen
class TemporalAuditResult:
    """Reference the outcome-blind external-year structural audit."""

    output_dir: Path
    audit_path: Path
    summary_path: Path


@attrs.frozen
class TemporalPlanResult:
    """Reference a hash-addressed external-year evaluation plan."""

    plan_path: Path
    plan_sha256: str


@attrs.frozen
class TemporalEvaluationResult:
    """Reference completed temporal-evaluation artifacts."""

    output_dir: Path
    metrics_path: Path
    comparisons_path: Path
    decision_path: Path
    terminal_summary_path: Path


class TemporalEvaluationPausedError(Exception):
    """Indicate a safe pause after complete external-year sessions."""

    def __init__(
        self, *, completed_sessions: int, total_sessions: int, output_dir: Path
    ):
        self.completed_sessions = completed_sessions
        self.total_sessions = total_sessions
        self.output_dir = output_dir
        super().__init__(
            "Temporal evaluation paused safely after "
            f"{completed_sessions}/{total_sessions} sessions; resume with the same "
            f"plan hash. Outputs: {output_dir}"
        )


def run_audit(
    *,
    protocol: research_models.ResearchProtocol,
    protocol_path: Path,
    temporal_policy: research_models.TemporalEvaluationPolicy,
    temporal_policy_path: Path,
) -> TemporalAuditResult:
    """Audit every external-year source structurally without calculating labels."""
    if temporal_policy.evaluation_year in protocol.years:
        raise ValueError("temporal year must be external to the development protocol")
    if not set(temporal_policy.instruments).issubset(protocol.instruments):
        raise ValueError(
            "temporal instruments must be present in the research protocol"
        )
    output_dir = protocol.output_dir / "temporal_evaluation"
    if (output_dir / "plan.json").exists() or (
        output_dir / "run_summary.json"
    ).exists():
        raise ValueError("temporal audit is frozen and cannot be changed")
    files = tuple(
        raw_file_repository.find_raw_files(
            data_dir=protocol.data_dir,
            instruments=tuple(item.value for item in temporal_policy.instruments),
            years=(temporal_policy.evaluation_year,),
        )
    )
    if not files:
        raise ValueError(
            f"no {temporal_policy.evaluation_year} temporal-evaluation files found"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = output_dir / "native_cache"
    receipt_root = output_dir / "audit_receipts"
    started = time.perf_counter()
    records: list[_audit._AuditRecord] = []
    pending: list[raw_file_repository.RawDataFile] = []
    for item in files:
        completed = _load_audit_receipt(
            raw_data_file=item,
            protocol=protocol,
            cache_root=cache_root,
            receipt_root=receipt_root,
        )
        if completed is None:
            pending.append(item)
        else:
            records.append(completed)
    if pending:
        with futures.ProcessPoolExecutor(
            max_workers=protocol.audit_workers
        ) as executor:
            work = {
                executor.submit(
                    _audit._audit_file,
                    raw_data_file=item,
                    protocol=protocol,
                    force_redact_outcomes=True,
                    cache_root=cache_root,
                ): item
                for item in pending
            }
            completed_count = len(records)
            for future in futures.as_completed(work):
                raw_data_file = work[future]
                record = future.result()
                _write_audit_receipt(
                    record=record,
                    raw_data_file=raw_data_file,
                    protocol=protocol,
                    receipt_root=receipt_root,
                )
                published = _load_audit_receipt(
                    raw_data_file=raw_data_file,
                    protocol=protocol,
                    cache_root=cache_root,
                    receipt_root=receipt_root,
                )
                if published is None:
                    raise RuntimeError("published temporal audit receipt is missing")
                records.append(published)
                completed_count += 1
                print(
                    f"[temporal-audit] {completed_count}/{len(files)} "
                    f"{record.identity.instrument.value} "
                    f"{record.identity.trading_date.isoformat()} "
                    f"eligible={record.eligible}",
                    flush=True,
                )
    records.sort(
        key=lambda item: (item.identity.instrument.value, item.identity.trading_date)
    )
    audit_path = output_dir / "session_audit.csv"
    _neural._write_csv_atomically(
        data=pl.DataFrame([item.row for item in records]), path=audit_path
    )
    common_dates = _common_eligible_dates(
        audit=pl.read_csv(audit_path, schema_overrides={"trading_date": pl.Date}),
        temporal_policy=temporal_policy,
    )
    elapsed = time.perf_counter() - started
    summary = {
        "schema_version": 1,
        "implementation_version": TEMPORAL_EVALUATION_IMPLEMENTATION_VERSION,
        "protocol_sha256": _baseline._sha256_file(path=protocol_path),
        "temporal_policy_sha256": _baseline._sha256_file(path=temporal_policy_path),
        "audit_sha256": _baseline._sha256_file(path=audit_path),
        "evaluation_year": temporal_policy.evaluation_year,
        "discovered_sessions": len(records),
        "eligible_sessions": sum(item.eligible for item in records),
        "common_eligible_dates": len(common_dates),
        "outcomes_redacted": sum(
            bool(item.row["outcomes_redacted"]) for item in records
        ),
        "target_outcomes_inspected": False,
        "elapsed_seconds": elapsed,
    }
    summary_path = output_dir / "audit_summary.json"
    _neural._write_text_atomically(
        text=json.dumps(summary, indent=2), path=summary_path
    )
    terminal = "\n".join(
        (
            "EBS external-year structural audit completed",
            "WARNING: temporal outcomes were not inspected.",
            f"evaluation_year={temporal_policy.evaluation_year}",
            f"discovered_sessions={len(records)}",
            f"eligible_sessions={summary['eligible_sessions']}",
            f"common_eligible_dates={len(common_dates)}",
            f"elapsed_seconds={elapsed:.2f}",
            f"outputs={output_dir}",
        )
    )
    _neural._write_text_atomically(
        text=terminal, path=output_dir / "audit_terminal_summary.txt"
    )
    print(terminal)
    return TemporalAuditResult(
        output_dir=output_dir, audit_path=audit_path, summary_path=summary_path
    )


def freeze_plan(
    *,
    protocol: research_models.ResearchProtocol,
    protocol_path: Path,
    neural_policy_path: Path,
    temporal_policy: research_models.TemporalEvaluationPolicy,
    temporal_policy_path: Path,
) -> TemporalPlanResult:
    """Freeze external-year sessions and final checkpoint identities."""
    output_dir = protocol.output_dir / "temporal_evaluation"
    audit_path = output_dir / "session_audit.csv"
    audit_summary_path = output_dir / "audit_summary.json"
    locked_dir = protocol.output_dir / "locked_evaluation"
    cross_dir = protocol.output_dir / "cross_instrument_evaluation"
    paths = {
        "protocol_sha256": protocol_path,
        "neural_policy_sha256": neural_policy_path,
        "temporal_policy_sha256": temporal_policy_path,
        "temporal_audit_sha256": audit_path,
        "temporal_audit_summary_sha256": audit_summary_path,
        "locked_plan_sha256": locked_dir / "plan.json",
        "locked_run_summary_sha256": locked_dir / "run_summary.json",
        "locked_decision_sha256": locked_dir / "decision.json",
        "locked_comparisons_sha256": locked_dir / "paired_baseline_comparisons.csv",
        "cross_plan_sha256": cross_dir / "plan.json",
        "cross_run_summary_sha256": cross_dir / "run_summary.json",
        "cross_decision_sha256": cross_dir / "decision.json",
        "cross_comparisons_sha256": cross_dir / "paired_baseline_comparisons.csv",
    }
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError(
            "complete the temporal audit, locked evaluation, and cross-instrument "
            "evaluation first"
        )
    audit_summary = _neural._json_mapping(path=audit_summary_path)
    if (
        audit_summary.get("target_outcomes_inspected") is not False
        or audit_summary.get("audit_sha256") != _baseline._sha256_file(path=audit_path)
        or audit_summary.get("protocol_sha256")
        != _baseline._sha256_file(path=protocol_path)
        or audit_summary.get("temporal_policy_sha256")
        != _baseline._sha256_file(path=temporal_policy_path)
    ):
        raise ValueError("temporal structural audit failed its integrity check")
    locked_decision = _validated_locked_decision(
        protocol=protocol,
        locked_dir=locked_dir,
        protocol_path=protocol_path,
        neural_policy_path=neural_policy_path,
    )
    _validate_cross_evidence(protocol=protocol, cross_dir=cross_dir)
    audit = pl.read_csv(audit_path, schema_overrides={"trading_date": pl.Date})
    common_dates = _common_eligible_dates(audit=audit, temporal_policy=temporal_policy)
    temporal_sessions = {
        instrument: _temporal_identities(
            audit=audit,
            dates=common_dates,
            instrument=instrument,
            protocol=protocol,
        )
        for instrument in temporal_policy.instruments
    }
    _cross_instrument._verify_sources(
        sessions=tuple(
            item for sessions in temporal_sessions.values() for item in sessions
        )
    )
    locked_plan = _neural._json_mapping(path=locked_dir / "plan.json")
    development_sessions = _cross_instrument._sessions_from_payload(
        value=locked_plan.get("development_sessions"),
        protocol=protocol,
        instrument=protocol.development_instrument,
    )
    cells = _frozen_cells(
        protocol=protocol,
        locked_dir=locked_dir,
        locked_plan=locked_plan,
        locked_decision=locked_decision,
        instruments=temporal_policy.instruments,
    )
    input_hashes = {
        key: _baseline._sha256_file(path=path) for key, path in paths.items()
    }
    plan = {
        "schema_version": 1,
        "implementation_version": TEMPORAL_EVALUATION_IMPLEMENTATION_VERSION,
        **input_hashes,
        "evaluation_year": temporal_policy.evaluation_year,
        "source_instrument": protocol.development_instrument.value,
        "primary_instrument": temporal_policy.primary_instrument.value,
        "session_selection": temporal_policy.session_selection,
        "common_evaluation_dates": [item.isoformat() for item in common_dates],
        "development_sessions": [
            _cross_instrument._identity_payload(item=item, protocol=protocol)
            for item in development_sessions
        ],
        "temporal_sessions": {
            instrument.value: [
                _temporal_identity_payload(
                    item=item, protocol=protocol, output_dir=output_dir
                )
                for item in temporal_sessions[instrument]
            ]
            for instrument in temporal_policy.instruments
        },
        "cells": cells,
        "primary_metrics": [item.value for item in protocol.primary_metrics],
        "supporting_metrics": [item.value for item in protocol.supporting_metrics],
        "bootstrap_repetitions": protocol.bootstrap_repetitions,
        "confidence_level": protocol.confidence_level,
        "evaluation_rule": (
            "Reuse both locked-confirmed EUR/USD model specifications and exact "
            "checkpoints without retraining. Primary temporal confirmation is "
            "evaluated on EUR/USD; other instruments are combined temporal and "
            "cross-instrument stress tests."
        ),
        "neural_retraining_permitted": False,
        "temporal_outcomes_inspected": False,
    }
    plan_path = output_dir / "plan.json"
    serialized = json.dumps(plan, indent=2)
    if plan_path.is_file() and plan_path.read_text(encoding="utf-8") != serialized:
        raise ValueError("a different temporal-evaluation plan is already frozen")
    _neural._write_text_atomically(text=serialized, path=plan_path)
    plan_hash = _baseline._sha256_file(path=plan_path)
    print("EBS external-year evaluation plan frozen")
    print("WARNING: temporal outcomes were not inspected.")
    print(f"evaluation_year={temporal_policy.evaluation_year}")
    print("instruments=" + ",".join(item.value for item in temporal_policy.instruments))
    print(f"common_sessions_per_instrument={len(common_dates)}")
    print(f"model_instrument_cells={len(cells)}")
    print(f"session_tasks={len(common_dates) * len(temporal_policy.instruments)}")
    print(f"plan_sha256={plan_hash}")
    return TemporalPlanResult(plan_path=plan_path, plan_sha256=plan_hash)


def run(
    *,
    protocol: research_models.ResearchProtocol,
    protocol_path: Path,
    neural_policy: research_models.NeuralBenchmarkPolicy,
    neural_policy_path: Path,
    temporal_policy: research_models.TemporalEvaluationPolicy,
    temporal_policy_path: Path,
    plan_sha256: str,
    maximum_new_sessions: int | None = None,
) -> TemporalEvaluationResult:
    """Evaluate frozen models one external-year session at a time."""
    if maximum_new_sessions is not None and (
        isinstance(maximum_new_sessions, bool) or maximum_new_sessions <= 0
    ):
        raise ValueError("maximum_new_sessions must be positive or null")
    started = time.perf_counter()
    output_dir = protocol.output_dir / "temporal_evaluation"
    plan_path = output_dir / "plan.json"
    if not plan_path.is_file() or _baseline._sha256_file(path=plan_path) != plan_sha256:
        raise ValueError("--plan-sha256 must match the frozen temporal plan")
    if (output_dir / "run_summary.json").is_file():
        raise ValueError("temporal evaluation is complete and cannot be rerun")
    plan = _neural._json_mapping(path=plan_path)
    _verify_plan_inputs(
        plan=plan,
        protocol=protocol,
        protocol_path=protocol_path,
        neural_policy_path=neural_policy_path,
        temporal_policy_path=temporal_policy_path,
    )
    development_sessions = _cross_instrument._sessions_from_payload(
        value=plan.get("development_sessions"),
        protocol=protocol,
        instrument=protocol.development_instrument,
    )
    temporal_sessions = _sessions_from_plan(
        plan=plan, protocol=protocol, temporal_policy=temporal_policy
    )
    cells = _cells_from_plan(plan=plan)
    _verify_frozen_sources(
        protocol=protocol,
        output_dir=output_dir,
        development_sessions=development_sessions,
        temporal_sessions=temporal_sessions,
        cells=cells,
    )
    dimensions = {
        (_locked._integer(cell, "depth"), _locked._integer(cell, "horizon_steps"))
        for cell in cells
    }
    if len(dimensions) != 1:
        raise ValueError("temporal evaluation requires one frozen depth/horizon")
    depth, horizon_steps = next(iter(dimensions))
    scaler, fitted_baseline = _prepare_source_reference(
        protocol=protocol,
        development_sessions=development_sessions,
        depth=depth,
        horizon_steps=horizon_steps,
    )
    device = model_domain.select_device(requested=neural_policy.device)
    classifiers = _load_classifiers(
        protocol=protocol,
        policy=neural_policy,
        cells=cells,
        device=device,
    )
    metrics: list[pl.DataFrame] = []
    all_sessions = [
        (instrument, session)
        for instrument in temporal_policy.instruments
        for session in temporal_sessions[instrument]
    ]
    newly_completed = 0
    for position, (instrument, session) in enumerate(all_sessions, start=1):
        completed = _completed_session(
            output_dir=output_dir,
            instrument=instrument,
            session=session,
            plan_sha256=plan_sha256,
            cells=cells,
        )
        if completed is not None:
            metrics.append(completed)
            continue
        print(
            f"[temporal-evaluation] session={position}/{len(all_sessions)} "
            f"instrument={instrument.value} date={session.trading_date.isoformat()}",
            flush=True,
        )
        metrics.append(
            _evaluate_session(
                protocol=protocol,
                policy=neural_policy,
                output_dir=output_dir,
                plan_sha256=plan_sha256,
                instrument=instrument,
                session=session,
                depth=depth,
                horizon_steps=horizon_steps,
                scaler=scaler,
                fitted_baseline=fitted_baseline,
                classifiers=classifiers,
                cells=cells,
                device=device,
            )
        )
        newly_completed += 1
        if device.type == "cuda":
            torch.cuda.empty_cache()
        _write_progress(
            output_dir=output_dir,
            completed_sessions=position,
            total_sessions=len(all_sessions),
            plan_sha256=plan_sha256,
        )
        if (
            maximum_new_sessions is not None
            and newly_completed >= maximum_new_sessions
            and position < len(all_sessions)
        ):
            raise TemporalEvaluationPausedError(
                completed_sessions=position,
                total_sessions=len(all_sessions),
                output_dir=output_dir,
            )
    combined = pl.concat(metrics).sort(
        ["instrument", "model", "seed", "validation_date"]
    )
    metrics_path = output_dir / "session_metrics.csv"
    _neural._write_csv_atomically(data=combined, path=metrics_path)
    comparisons = _cross_instrument._comparisons(metrics=combined, protocol=protocol)
    comparisons = comparisons.with_columns(
        pl.lit("seed_mean_neural_minus_frozen_source_logistic").alias("comparison")
    )
    comparisons_path = output_dir / "paired_baseline_comparisons.csv"
    _neural._write_csv_atomically(data=comparisons, path=comparisons_path)
    decision = _temporal_decision(
        comparisons=comparisons,
        protocol=protocol,
        primary_instrument=temporal_policy.primary_instrument,
    )
    decision_path = output_dir / "decision.json"
    _neural._write_text_atomically(
        text=json.dumps(decision, indent=2), path=decision_path
    )
    monthly_path = output_dir / "monthly_metric_summary.csv"
    _neural._write_csv_atomically(
        data=_monthly_summary(metrics=combined, protocol=protocol), path=monthly_path
    )
    elapsed = time.perf_counter() - started
    confirmed = cast(list[dict[str, object]], decision["temporally_confirmed_models"])
    terminal = "\n".join(
        (
            "EBS frozen external-year temporal evaluation completed",
            "WARNING: temporal outcomes are now inspected; do not tune and rerun.",
            f"evaluation_year={temporal_policy.evaluation_year}",
            f"primary_instrument={temporal_policy.primary_instrument.value}",
            "instruments="
            + ",".join(item.value for item in temporal_policy.instruments),
            f"sessions={len(all_sessions)}",
            f"model_instrument_cells={len(cells)}",
            f"temporally_confirmed_models={len(confirmed)}",
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
                "implementation_version": TEMPORAL_EVALUATION_IMPLEMENTATION_VERSION,
                "plan_sha256": plan_sha256,
                "temporal_outcomes_used": True,
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
                    "monthly_metric_summary": str(monthly_path),
                    "decision": str(decision_path),
                },
            },
            indent=2,
        ),
        path=output_dir / "run_summary.json",
    )
    (output_dir / "progress_summary.json").unlink(missing_ok=True)
    print(terminal)
    return TemporalEvaluationResult(
        output_dir=output_dir,
        metrics_path=metrics_path,
        comparisons_path=comparisons_path,
        decision_path=decision_path,
        terminal_summary_path=terminal_path,
    )


def _audit_receipt_path(
    *, receipt_root: Path, raw_data_file: raw_file_repository.RawDataFile
) -> Path:
    return (
        receipt_root
        / raw_data_file.instrument
        / f"{raw_data_file.trading_date.isoformat()}.json"
    )


def _load_audit_receipt(
    *,
    raw_data_file: raw_file_repository.RawDataFile,
    protocol: research_models.ResearchProtocol,
    cache_root: Path,
    receipt_root: Path,
) -> _audit._AuditRecord | None:
    path = _audit_receipt_path(receipt_root=receipt_root, raw_data_file=raw_data_file)
    if not path.is_file():
        return None
    payload = _neural._json_mapping(path=path)
    if (
        payload.get("source_size_bytes") != raw_data_file.size_bytes
        or payload.get("source_modified_time_ns") != raw_data_file.modified_time_ns
    ):
        raise ValueError(f"temporal audit source changed: {raw_data_file.path}")
    if payload.get("protocol_structural_rules") != _structural_rules(protocol=protocol):
        raise ValueError("temporal audit structural rules changed during resume")
    row = payload.get("row")
    if not isinstance(row, dict):
        raise ValueError(f"temporal audit receipt is invalid: {path}")
    typed_row = cast(dict[str, object], row)
    eligible = typed_row.get("eligible") is True
    cache_hash = typed_row.get("native_cache_sha256")
    if eligible:
        cache_path = (
            cache_root
            / raw_data_file.instrument
            / f"{raw_data_file.trading_date.isoformat()}.parquet"
        )
        if (
            not cache_path.is_file()
            or not isinstance(cache_hash, str)
            or _baseline._sha256_file(path=cache_path) != cache_hash
        ):
            raise ValueError(f"temporal audit cache is invalid: {cache_path}")
    identity = research_models.SessionIdentity(
        instrument=orderbook_models.Instrument(raw_data_file.instrument),
        trading_date=raw_data_file.trading_date,
        path=raw_data_file.path,
        sha256=_locked._string(typed_row, "sha256"),
    )
    return _audit._AuditRecord(identity=identity, eligible=eligible, row=typed_row)


def _write_audit_receipt(
    *,
    record: _audit._AuditRecord,
    raw_data_file: raw_file_repository.RawDataFile,
    protocol: research_models.ResearchProtocol,
    receipt_root: Path,
) -> None:
    path = _audit_receipt_path(receipt_root=receipt_root, raw_data_file=raw_data_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    _neural._write_text_atomically(
        text=json.dumps(
            {
                "protocol_structural_rules": _structural_rules(protocol=protocol),
                "source_size_bytes": raw_data_file.size_bytes,
                "source_modified_time_ns": raw_data_file.modified_time_ns,
                "row": record.row,
            },
            indent=2,
            default=_json_default,
        ),
        path=path,
    )


def _structural_rules(*, protocol: research_models.ResearchProtocol) -> dict[str, int]:
    return {
        "state_interval_milliseconds": protocol.state_interval_milliseconds,
        "maximum_staleness_milliseconds": protocol.maximum_staleness_milliseconds,
        "minimum_duration_milliseconds": (
            protocol.audit_policy.minimum_duration_milliseconds
        ),
        "minimum_observed_states": protocol.audit_policy.minimum_observed_states,
        "required_depth": protocol.audit_policy.required_depth,
    }


def _json_default(value: object) -> str:
    if isinstance(value, datetime.date | datetime.datetime):
        return value.isoformat()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _common_eligible_dates(
    *, audit: pl.DataFrame, temporal_policy: research_models.TemporalEvaluationPolicy
) -> tuple[datetime.date, ...]:
    required = {
        "instrument",
        "trading_date",
        "eligible",
        "outcomes_redacted",
        "native_cache_sha256",
    }
    if not required.issubset(audit.columns):
        raise ValueError("temporal audit lacks required structural fields")
    eligible = audit.filter(
        pl.col("eligible")
        & pl.col("outcomes_redacted")
        & pl.col("native_cache_sha256").is_not_null()
    )
    date_sets = [
        set(
            eligible.filter(pl.col("instrument") == instrument.value)[
                "trading_date"
            ].to_list()
        )
        for instrument in temporal_policy.instruments
    ]
    common = tuple(sorted(set.intersection(*date_sets))) if date_sets else ()
    if len(common) < temporal_policy.minimum_common_eligible_sessions:
        raise ValueError(
            "temporal audit has only "
            f"{len(common)} common eligible dates; requires "
            f"{temporal_policy.minimum_common_eligible_sessions}"
        )
    if any(item.year != temporal_policy.evaluation_year for item in common):
        raise ValueError("temporal audit contains a date outside the evaluation year")
    return tuple(cast(datetime.date, item) for item in common)


def _temporal_identities(
    *,
    audit: pl.DataFrame,
    dates: tuple[datetime.date, ...],
    instrument: orderbook_models.Instrument,
    protocol: research_models.ResearchProtocol,
) -> tuple[research_models.SessionIdentity, ...]:
    rows = audit.filter(
        (pl.col("instrument") == instrument.value) & pl.col("trading_date").is_in(dates)
    ).sort("trading_date")
    if rows.height != len(dates):
        raise ValueError(
            f"temporal audit coverage is incomplete for {instrument.value}"
        )
    identities: list[research_models.SessionIdentity] = []
    for row in rows.iter_rows(named=True):
        raw_path = Path(cast(str, row["raw_path"]))
        try:
            raw_path.relative_to(protocol.data_dir)
        except ValueError as exc:
            raise ValueError(
                "temporal source is outside the protocol data directory"
            ) from exc
        identities.append(
            research_models.SessionIdentity(
                instrument=instrument,
                trading_date=cast(datetime.date, row["trading_date"]),
                path=raw_path,
                sha256=cast(str, row["sha256"]),
            )
        )
    return tuple(identities)


def _temporal_identity_payload(
    *,
    item: research_models.SessionIdentity,
    protocol: research_models.ResearchProtocol,
    output_dir: Path,
) -> dict[str, str]:
    cache_path = (
        output_dir
        / "native_cache"
        / item.instrument.value
        / f"{item.trading_date.isoformat()}.parquet"
    )
    return {
        **_cross_instrument._identity_payload(item=item, protocol=protocol),
        "native_cache": str(cache_path.relative_to(protocol.output_dir)),
        "native_cache_sha256": _baseline._sha256_file(path=cache_path),
    }


def _validated_locked_decision(
    *,
    protocol: research_models.ResearchProtocol,
    locked_dir: Path,
    protocol_path: Path,
    neural_policy_path: Path,
) -> dict[str, object]:
    locked_plan = _neural._json_mapping(path=locked_dir / "plan.json")
    locked_summary = _neural._json_mapping(path=locked_dir / "run_summary.json")
    locked_decision = _neural._json_mapping(path=locked_dir / "decision.json")
    recomputed = _locked._locked_decision(
        comparisons=pl.read_csv(locked_dir / "paired_baseline_comparisons.csv"),
        protocol=protocol,
    )
    if _cross_instrument._normalized_locked_decision(
        locked_decision
    ) != _cross_instrument._normalized_locked_decision(recomputed):
        raise ValueError("locked decision does not match its comparison evidence")
    if (
        locked_decision.get("locked_evaluation_used") is not True
        or locked_decision.get("retuning_permitted") is not False
        or locked_summary.get("plan_sha256")
        != _baseline._sha256_file(path=locked_dir / "plan.json")
        or locked_plan.get("protocol_sha256")
        != _baseline._sha256_file(path=protocol_path)
        or locked_plan.get("policy_sha256")
        != _baseline._sha256_file(path=neural_policy_path)
    ):
        raise ValueError("locked evaluation is not valid frozen evidence")
    return locked_decision


def _validate_cross_evidence(
    *, protocol: research_models.ResearchProtocol, cross_dir: Path
) -> None:
    summary = _neural._json_mapping(path=cross_dir / "run_summary.json")
    decision = _neural._json_mapping(path=cross_dir / "decision.json")
    recomputed = _cross_instrument._decision(
        comparisons=pl.read_csv(cross_dir / "paired_baseline_comparisons.csv"),
        protocol=protocol,
    )
    if decision != recomputed:
        raise ValueError("cross-instrument decision does not match its evidence")
    if (
        summary.get("cross_instrument_outcomes_used") is not True
        or summary.get("neural_retraining_used") is not False
        or summary.get("plan_sha256")
        != _baseline._sha256_file(path=cross_dir / "plan.json")
        or decision.get("retuning_permitted") is not False
    ):
        raise ValueError("cross-instrument evaluation is not valid frozen evidence")


def _frozen_cells(
    *,
    protocol: research_models.ResearchProtocol,
    locked_dir: Path,
    locked_plan: dict[str, object],
    locked_decision: dict[str, object],
    instruments: tuple[orderbook_models.Instrument, ...],
) -> list[dict[str, object]]:
    candidates = locked_decision.get("confirmed_candidates")
    locked_cells = locked_plan.get("cells")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("locked evaluation confirmed no temporal candidate")
    if not isinstance(locked_cells, list):
        raise ValueError("locked plan cells are invalid")
    result: list[dict[str, object]] = []
    for raw_candidate in candidates:
        if not isinstance(raw_candidate, dict):
            raise ValueError("locked candidate is invalid")
        candidate = cast(dict[str, object], raw_candidate)
        model_name = _locked._string(candidate, "model")
        depth = _locked._integer(candidate, "depth")
        horizon = _locked._integer(candidate, "horizon_milliseconds")
        matched = [
            cast(dict[str, object], item)
            for item in locked_cells
            if isinstance(item, dict)
            and item.get("model") == model_name
            and item.get("depth") == depth
            and item.get("horizon_milliseconds") == horizon
        ]
        if {item.get("seed") for item in matched} != set(protocol.random_seeds):
            raise ValueError("locked candidate lacks every frozen seed checkpoint")
        for locked_cell in matched:
            source_dir = _locked._cell_dir(output_dir=locked_dir, cell=locked_cell)
            checkpoint_path = source_dir / "final.pt"
            summary_path = source_dir / "cell_summary.json"
            if not checkpoint_path.is_file() or not summary_path.is_file():
                raise ValueError(f"final checkpoint is missing: {checkpoint_path}")
            checkpoint_hash = _baseline._sha256_file(path=checkpoint_path)
            summary = _neural._json_mapping(path=summary_path)
            artifact_hashes = summary.get("artifact_sha256")
            if (
                not isinstance(artifact_hashes, dict)
                or artifact_hashes.get("final.pt") != checkpoint_hash
            ):
                raise ValueError(
                    f"final checkpoint failed integrity check: {checkpoint_path}"
                )
            for instrument in instruments:
                result.append(
                    {
                        "instrument": instrument.value,
                        "model": model_name,
                        "depth": depth,
                        "horizon_milliseconds": horizon,
                        "horizon_steps": horizon
                        // protocol.state_interval_milliseconds,
                        "seed": _locked._integer(locked_cell, "seed"),
                        "source_checkpoint": str(
                            checkpoint_path.relative_to(protocol.output_dir)
                        ),
                        "source_checkpoint_sha256": checkpoint_hash,
                    }
                )
    return sorted(
        result,
        key=lambda item: (
            cast(str, item["instrument"]),
            cast(str, item["model"]),
            cast(int, item["seed"]),
        ),
    )


def _verify_plan_inputs(
    *,
    plan: dict[str, object],
    protocol: research_models.ResearchProtocol,
    protocol_path: Path,
    neural_policy_path: Path,
    temporal_policy_path: Path,
) -> None:
    output_dir = protocol.output_dir / "temporal_evaluation"
    locked_dir = protocol.output_dir / "locked_evaluation"
    cross_dir = protocol.output_dir / "cross_instrument_evaluation"
    paths = {
        "protocol_sha256": protocol_path,
        "neural_policy_sha256": neural_policy_path,
        "temporal_policy_sha256": temporal_policy_path,
        "temporal_audit_sha256": output_dir / "session_audit.csv",
        "temporal_audit_summary_sha256": output_dir / "audit_summary.json",
        "locked_plan_sha256": locked_dir / "plan.json",
        "locked_run_summary_sha256": locked_dir / "run_summary.json",
        "locked_decision_sha256": locked_dir / "decision.json",
        "locked_comparisons_sha256": locked_dir / "paired_baseline_comparisons.csv",
        "cross_plan_sha256": cross_dir / "plan.json",
        "cross_run_summary_sha256": cross_dir / "run_summary.json",
        "cross_decision_sha256": cross_dir / "decision.json",
        "cross_comparisons_sha256": cross_dir / "paired_baseline_comparisons.csv",
    }
    if any(
        not path.is_file() or plan.get(key) != _baseline._sha256_file(path=path)
        for key, path in paths.items()
    ):
        raise ValueError("temporal plan inputs changed after freezing")
    if (
        plan.get("temporal_outcomes_inspected") is not False
        or plan.get("neural_retraining_permitted") is not False
    ):
        raise ValueError("temporal plan violates the frozen boundary")


def _sessions_from_plan(
    *,
    plan: dict[str, object],
    protocol: research_models.ResearchProtocol,
    temporal_policy: research_models.TemporalEvaluationPolicy,
) -> dict[orderbook_models.Instrument, tuple[research_models.SessionIdentity, ...]]:
    raw = plan.get("temporal_sessions")
    if not isinstance(raw, dict):
        raise ValueError("temporal session plan is invalid")
    result: dict[
        orderbook_models.Instrument, tuple[research_models.SessionIdentity, ...]
    ] = {}
    for instrument in temporal_policy.instruments:
        payload = raw.get(instrument.value)
        if not isinstance(payload, list):
            raise ValueError(f"temporal sessions missing for {instrument.value}")
        stripped: list[dict[str, str]] = []
        for item in payload:
            if not isinstance(item, dict):
                raise ValueError("temporal session identity is invalid")
            fields = cast(dict[str, object], item)
            stripped.append(
                {
                    "trading_date": _locked._string(fields, "trading_date"),
                    "raw_path": _locked._string(fields, "raw_path"),
                    "sha256": _locked._string(fields, "sha256"),
                }
            )
        result[instrument] = _cross_instrument._sessions_from_payload(
            value=stripped, protocol=protocol, instrument=instrument
        )
    expected_dates = plan.get("common_evaluation_dates")
    if not isinstance(expected_dates, list) or any(
        not isinstance(item, str) for item in expected_dates
    ):
        raise ValueError("common temporal dates are invalid")
    dates = tuple(
        datetime.date.fromisoformat(cast(str, item)) for item in expected_dates
    )
    if any(
        tuple(item.trading_date for item in sessions) != dates
        for sessions in result.values()
    ):
        raise ValueError("temporal session dates do not match the frozen common dates")
    return result


def _cells_from_plan(*, plan: dict[str, object]) -> list[dict[str, object]]:
    raw = plan.get("cells")
    if (
        not isinstance(raw, list)
        or not raw
        or any(not isinstance(item, dict) for item in raw)
    ):
        raise ValueError("temporal plan cells are invalid")
    return [cast(dict[str, object], item) for item in raw]


def _verify_frozen_sources(
    *,
    protocol: research_models.ResearchProtocol,
    output_dir: Path,
    development_sessions: tuple[research_models.SessionIdentity, ...],
    temporal_sessions: dict[
        orderbook_models.Instrument, tuple[research_models.SessionIdentity, ...]
    ],
    cells: list[dict[str, object]],
) -> None:
    _cross_instrument._verify_sources(
        sessions=(
            *development_sessions,
            *(item for sessions in temporal_sessions.values() for item in sessions),
        )
    )
    plan = _neural._json_mapping(path=output_dir / "plan.json")
    raw_sessions = plan.get("temporal_sessions")
    if not isinstance(raw_sessions, dict):
        raise ValueError("temporal sessions are invalid")
    for instrument, sessions in temporal_sessions.items():
        payload = raw_sessions.get(instrument.value)
        if not isinstance(payload, list):
            raise ValueError("temporal cache identities are invalid")
        for session, raw_item in zip(sessions, payload, strict=True):
            if not isinstance(raw_item, dict):
                raise ValueError("temporal cache identity is invalid")
            item = cast(dict[str, object], raw_item)
            cache_path = protocol.output_dir / _locked._string(item, "native_cache")
            if not cache_path.is_file() or _baseline._sha256_file(
                path=cache_path
            ) != _locked._string(item, "native_cache_sha256"):
                raise ValueError(
                    f"temporal cache changed for {instrument.value} "
                    f"{session.trading_date.isoformat()}"
                )
    for cell in cells:
        checkpoint = protocol.output_dir / _locked._string(cell, "source_checkpoint")
        if not checkpoint.is_file() or _baseline._sha256_file(
            path=checkpoint
        ) != _locked._string(cell, "source_checkpoint_sha256"):
            raise ValueError(f"frozen checkpoint changed: {checkpoint}")


def _prepare_source_reference(
    *,
    protocol: research_models.ResearchProtocol,
    development_sessions: tuple[research_models.SessionIdentity, ...],
    depth: int,
    horizon_steps: int,
) -> tuple[pilot_training.FeatureScaler, model_domain.DefensiveBaselineModel]:
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
    stride_steps = research_operations.training_stride_steps(
        protocol=protocol,
        horizon_milliseconds=horizon_steps * protocol.state_interval_milliseconds,
    )
    baseline_sessions = tuple(
        _baseline._prepare_baseline_session(
            protocol=protocol,
            identity=item,
            depth=depth,
            horizon_steps=horizon_steps,
            scaler=scaler,
            stride_steps=stride_steps,
        )
        for item in development_sessions
    )
    return scaler, model_domain.fit_defensive_baseline_model(sessions=baseline_sessions)


def _load_classifiers(
    *,
    protocol: research_models.ResearchProtocol,
    policy: research_models.NeuralBenchmarkPolicy,
    cells: list[dict[str, object]],
    device: torch.device,
) -> dict[tuple[str, int], torch.nn.Module]:
    result: dict[tuple[str, int], torch.nn.Module] = {}
    for cell in cells:
        key = (_locked._string(cell, "model"), _locked._integer(cell, "seed"))
        if key in result:
            continue
        checkpoint_path = protocol.output_dir / _locked._string(
            cell, "source_checkpoint"
        )
        payload = checkpoint_repository.read(path=checkpoint_path)
        state = payload.get("classifier_state")
        if payload.get("kind") != "fixed_final_model" or not isinstance(state, dict):
            raise ValueError(f"source checkpoint is invalid: {checkpoint_path}")
        classifier = model_domain.build_direction_classifier(
            model_name=key[0],
            auxiliary_size=len(pilot_training.AUXILIARY_FEATURE_ORDER),
            hidden_size=policy.hidden_size,
        )
        classifier.load_state_dict(cast(dict[str, torch.Tensor], state))
        result[key] = classifier.to(device)
    return result


def _session_dir(
    *, output_dir: Path, instrument: orderbook_models.Instrument, date: datetime.date
) -> Path:
    return output_dir / "session_results" / instrument.value / date.isoformat()


def _session_fingerprint(
    *,
    plan_sha256: str,
    instrument: orderbook_models.Instrument,
    session: research_models.SessionIdentity,
    cells: list[dict[str, object]],
) -> str:
    relevant = [item for item in cells if item.get("instrument") == instrument.value]
    payload = {
        "plan_sha256": plan_sha256,
        "instrument": instrument.value,
        "trading_date": session.trading_date.isoformat(),
        "source_sha256": session.sha256,
        "cells": relevant,
    }
    return _sha256_text(json.dumps(payload, sort_keys=True))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _completed_session(
    *,
    output_dir: Path,
    instrument: orderbook_models.Instrument,
    session: research_models.SessionIdentity,
    plan_sha256: str,
    cells: list[dict[str, object]],
) -> pl.DataFrame | None:
    session_dir = _session_dir(
        output_dir=output_dir, instrument=instrument, date=session.trading_date
    )
    summary_path = session_dir / "session_summary.json"
    metrics_path = session_dir / "metrics.csv"
    if not summary_path.is_file():
        return None
    summary = _neural._json_mapping(path=summary_path)
    if summary.get("fingerprint") != _session_fingerprint(
        plan_sha256=plan_sha256,
        instrument=instrument,
        session=session,
        cells=cells,
    ):
        raise ValueError(f"temporal session fingerprint mismatch: {session_dir}")
    if not metrics_path.is_file() or summary.get(
        "metrics_sha256"
    ) != _baseline._sha256_file(path=metrics_path):
        raise ValueError(f"temporal session failed integrity check: {session_dir}")
    metrics = pl.read_csv(metrics_path, schema_overrides={"validation_date": pl.Date})
    expected_models = {
        (_locked._string(item, "model"), _locked._integer(item, "seed"))
        for item in cells
        if item.get("instrument") == instrument.value
    }
    observed_models = set(
        metrics.filter(pl.col("seed") >= 0).select("model", "seed").iter_rows()
    )
    if expected_models != observed_models or metrics.height != len(expected_models) + 1:
        raise ValueError(f"temporal session model coverage is invalid: {session_dir}")
    return metrics


def _evaluate_session(
    *,
    protocol: research_models.ResearchProtocol,
    policy: research_models.NeuralBenchmarkPolicy,
    output_dir: Path,
    plan_sha256: str,
    instrument: orderbook_models.Instrument,
    session: research_models.SessionIdentity,
    depth: int,
    horizon_steps: int,
    scaler: pilot_training.FeatureScaler,
    fitted_baseline: model_domain.DefensiveBaselineModel,
    classifiers: dict[tuple[str, int], torch.nn.Module],
    cells: list[dict[str, object]],
    device: torch.device,
) -> pl.DataFrame:
    raw = _extract_temporal_session(
        protocol=protocol,
        output_dir=output_dir,
        instrument=instrument,
        session=session,
        depth=depth,
        horizon_steps=horizon_steps,
    )
    baseline_session = pilot_training.prepare_baseline_session(
        session=raw,
        scaler=scaler,
        context_steps=protocol.context_milliseconds
        // protocol.state_interval_milliseconds,
        horizon_steps=horizon_steps,
        stride_steps=protocol.evaluation_stride_milliseconds
        // protocol.state_interval_milliseconds,
    )
    baseline_probabilities = model_domain.predict_defensive_baselines(
        fitted=fitted_baseline, evaluation=baseline_session
    )["logistic"]
    rows = [
        _metric_row(
            protocol=protocol,
            instrument=instrument,
            date=session.trading_date,
            labels=baseline_session.labels,
            probabilities=baseline_probabilities,
            model_name="logistic",
            seed=-1,
            depth=depth,
            horizon_steps=horizon_steps,
            parameter_count=0,
        )
    ]
    scaled = pilot_training.apply_feature_scaler(session=raw, scaler=scaler)
    corpus = pilot_training.combine_sessions(
        sessions=(scaled,),
        context_steps=protocol.context_milliseconds
        // protocol.state_interval_milliseconds,
        horizon_steps=horizon_steps,
        maximum_windows=None,
        stride_steps=protocol.evaluation_stride_milliseconds
        // protocol.state_interval_milliseconds,
    )
    dataset = model_domain.SequenceDataset(
        lob_features=corpus.lob_features,
        auxiliary_features=corpus.auxiliary_features,
        labels=corpus.labels,
        target_indices=corpus.target_indices,
        context_steps=protocol.context_milliseconds
        // protocol.state_interval_milliseconds,
    ).to(device)
    relevant_cells = [
        item for item in cells if item.get("instrument") == instrument.value
    ]
    labels = corpus.labels[corpus.target_indices]
    if not np.array_equal(labels, baseline_session.labels):
        raise ValueError("temporal neural and baseline targets are not aligned")
    for cell in relevant_cells:
        key = (_locked._string(cell, "model"), _locked._integer(cell, "seed"))
        classifier = classifiers[key]
        prediction = model_domain.predict_classifier(
            classifier=classifier,
            dataset=dataset,
            device=device,
            batch_size=policy.evaluation_batch_size,
        )
        rows.append(
            _metric_row(
                protocol=protocol,
                instrument=instrument,
                date=session.trading_date,
                labels=labels,
                probabilities=prediction.probabilities,
                model_name=key[0],
                seed=key[1],
                depth=depth,
                horizon_steps=horizon_steps,
                parameter_count=model_domain.parameter_count(classifier=classifier),
            )
        )
    metrics = pl.DataFrame(rows)
    session_dir = _session_dir(
        output_dir=output_dir, instrument=instrument, date=session.trading_date
    )
    session_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = session_dir / "metrics.csv"
    _neural._write_csv_atomically(data=metrics, path=metrics_path)
    _neural._write_text_atomically(
        text=json.dumps(
            {
                "fingerprint": _session_fingerprint(
                    plan_sha256=plan_sha256,
                    instrument=instrument,
                    session=session,
                    cells=cells,
                ),
                "neural_retraining_used": False,
                "temporal_outcomes_used": True,
                "metrics_sha256": _baseline._sha256_file(path=metrics_path),
            },
            indent=2,
        ),
        path=session_dir / "session_summary.json",
    )
    return metrics


def _extract_temporal_session(
    *,
    protocol: research_models.ResearchProtocol,
    output_dir: Path,
    instrument: orderbook_models.Instrument,
    session: research_models.SessionIdentity,
    depth: int,
    horizon_steps: int,
) -> pilot_training.RawSessionData:
    cache_path = (
        output_dir
        / "native_cache"
        / instrument.value
        / f"{session.trading_date.isoformat()}.parquet"
    )
    states = pilot_operations.add_direction_targets(
        data=pl.read_parquet(cache_path), horizon_steps=(horizon_steps,)
    )
    return pilot_training.extract_session(
        data=states,
        trading_date=session.trading_date,
        depth=depth,
        horizon_steps=horizon_steps,
    )


def _metric_row(
    *,
    protocol: research_models.ResearchProtocol,
    instrument: orderbook_models.Instrument,
    date: datetime.date,
    labels: np.ndarray,
    probabilities: np.ndarray,
    model_name: str,
    seed: int,
    depth: int,
    horizon_steps: int,
    parameter_count: int,
) -> dict[str, object]:
    row = model_domain.direction_metric_row(
        model_name=model_name,
        depth=depth,
        horizon_steps=horizon_steps,
        seed=seed,
        labels=labels,
        probabilities=probabilities,
        parameter_count=parameter_count,
    )
    row.update(
        {
            "instrument": instrument.value,
            "source_instrument": protocol.development_instrument.value,
            "fold": "external_year_final",
            "validation_date": date,
        }
    )
    return row


def _temporal_decision(
    *,
    comparisons: pl.DataFrame,
    protocol: research_models.ResearchProtocol,
    primary_instrument: orderbook_models.Instrument,
) -> dict[str, object]:
    primary_metrics = tuple(item.value for item in protocol.primary_metrics)
    evidence: dict[str, bool] = {}
    confirmed_stress_tests: list[dict[str, object]] = []
    dimensions = (
        comparisons.select("instrument", "model", "depth", "horizon_milliseconds")
        .unique()
        .sort(["instrument", "model", "depth", "horizon_milliseconds"])
    )
    for dimension in dimensions.iter_rows(named=True):
        rows = comparisons.filter(
            (pl.col("instrument") == dimension["instrument"])
            & (pl.col("model") == dimension["model"])
            & (pl.col("depth") == dimension["depth"])
            & (pl.col("horizon_milliseconds") == dimension["horizon_milliseconds"])
            & pl.col("metric").is_in(primary_metrics)
        )
        passed = rows.height == len(primary_metrics) and bool(
            (rows["confidence_lower"] > 0.0).all()
        )
        key = (
            f"{dimension['instrument']}:{dimension['model']}:"
            f"d{dimension['depth']}:h{dimension['horizon_milliseconds']}"
        )
        evidence[key] = passed
        if passed:
            confirmed_stress_tests.append(dict(dimension))
    temporal_models = [
        {
            "model": item["model"],
            "depth": item["depth"],
            "horizon_milliseconds": item["horizon_milliseconds"],
        }
        for item in confirmed_stress_tests
        if item["instrument"] == primary_instrument.value
    ]
    return {
        "decision_rule": (
            "Primary temporal confirmation requires a strictly positive lower "
            "paired-session bootstrap confidence bound over the frozen-source "
            "logistic baseline for every primary metric on the predeclared primary "
            "instrument. Other instruments are reported as combined temporal and "
            "cross-instrument stress tests."
        ),
        "primary_instrument": primary_instrument.value,
        "primary_metrics": list(primary_metrics),
        "confirmed_by_instrument_model": evidence,
        "temporally_confirmed_models": temporal_models,
        "confirmed_instrument_year_stress_tests": confirmed_stress_tests,
        "temporal_outcomes_used": True,
        "neural_retraining_used": False,
        "retuning_permitted": False,
    }


def _monthly_summary(
    *, metrics: pl.DataFrame, protocol: research_models.ResearchProtocol
) -> pl.DataFrame:
    metric_names = tuple(
        item.value for item in (*protocol.primary_metrics, *protocol.supporting_metrics)
    )
    return (
        metrics.with_columns(
            pl.col("validation_date").cast(pl.Date).dt.strftime("%Y-%m").alias("month")
        )
        .group_by(
            [
                "instrument",
                "month",
                "model",
                "depth",
                "horizon_milliseconds",
                "seed",
            ]
        )
        .agg(
            pl.len().alias("sessions"),
            *[pl.col(name).mean().alias(name) for name in metric_names],
        )
        .sort(["instrument", "month", "model", "seed"])
    )


def _write_progress(
    *, output_dir: Path, completed_sessions: int, total_sessions: int, plan_sha256: str
) -> None:
    _neural._write_text_atomically(
        text=json.dumps(
            {
                "status": "running",
                "completed_sessions": completed_sessions,
                "total_sessions": total_sessions,
                "plan_sha256": plan_sha256,
                "resume_with_same_plan_hash": True,
            },
            indent=2,
        ),
        path=output_dir / "progress_summary.json",
    )
