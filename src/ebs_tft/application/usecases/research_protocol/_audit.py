"""Audit every configured EBS session and freeze chronological split identities."""

from __future__ import annotations

import datetime
import hashlib
import json
import time
from collections import Counter
from collections.abc import Generator, Iterable
from concurrent import futures
from contextlib import closing
from pathlib import Path
from typing import cast

import attrs
import polars as pl
import yaml

from ebs_tft.data.parsers import ebs_csv
from ebs_tft.data.repositories import artifact as artifact_repository
from ebs_tft.data.repositories import raw_file as raw_file_repository
from ebs_tft.domain.orderbook import models as orderbook_models
from ebs_tft.domain.pilot import models as pilot_models
from ebs_tft.domain.pilot import operations as pilot_operations
from ebs_tft.domain.research import models as research_models
from ebs_tft.domain.research import operations as research_operations


@attrs.frozen
class SessionAuditResult:
    """Reference the durable outputs from one all-session audit."""

    output_dir: Path
    audit_path: Path
    manifest_path: Path
    summary_path: Path
    terminal_summary_path: Path


@attrs.define
class _RecordTracker:
    first_timestamp: datetime.datetime | None = None
    latest_timestamp: datetime.datetime | None = None
    maximum_source_depth: int = 0

    def observe(self, *, record: orderbook_models.RawRecord) -> None:
        if self.first_timestamp is None:
            self.first_timestamp = record.timestamp
        self.latest_timestamp = record.timestamp
        if isinstance(record, orderbook_models.RawQuote):
            self.maximum_source_depth = max(self.maximum_source_depth, record.level)


@attrs.frozen
class _AuditRecord:
    identity: research_models.SessionIdentity
    eligible: bool
    row: dict[str, object]


def run(
    *,
    protocol: research_models.ResearchProtocol,
    protocol_path: Path,
    replace_output: bool = False,
) -> SessionAuditResult:
    """Audit configured sources and write an immutable rolling-fold manifest."""
    started = time.perf_counter()
    artifact_repository.prepare_run_directory(
        path=protocol.output_dir, replace=replace_output
    )
    files = tuple(
        raw_file_repository.find_raw_files(
            data_dir=protocol.data_dir,
            instruments=tuple(item.value for item in protocol.instruments),
            years=protocol.years,
        )
    )
    if not files:
        raise ValueError("research protocol discovered no raw EBS files")
    audited = list(_audit_files(files=files, protocol=protocol))

    audit_path = protocol.output_dir / "session_audit.csv"
    pl.DataFrame([item.row for item in audited]).sort(
        ["instrument", "trading_date"]
    ).write_csv(audit_path)
    manifest = _manifest(
        protocol=protocol,
        protocol_path=protocol_path,
        audited=tuple(audited),
        audit_path=audit_path,
    )
    manifest_path = protocol.output_dir / "split_manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    elapsed = time.perf_counter() - started
    instrument_summary = _instrument_summary(audited=tuple(audited), protocol=protocol)
    summary = {
        "warning": "Development split manifest; locked outcomes remain uninspected.",
        "protocol_sha256": _sha256_file(path=protocol_path),
        "discovered_sessions": len(audited),
        "eligible_sessions": sum(item.eligible for item in audited),
        "locked_sessions": sum(bool(item.row["evaluation_locked"]) for item in audited),
        "outcomes_redacted": sum(
            bool(item.row["outcomes_redacted"]) for item in audited
        ),
        "by_instrument": instrument_summary,
        "elapsed_seconds": elapsed,
        "artifacts": {
            "session_audit": str(audit_path),
            "split_manifest": str(manifest_path),
        },
    }
    summary_path = protocol.output_dir / "audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    terminal_summary = "\n".join(
        (
            "EBS all-session structural audit completed",
            "WARNING: locked evaluation outcomes were not inspected.",
            f"discovered_sessions={summary['discovered_sessions']}",
            f"eligible_sessions={summary['eligible_sessions']}",
            f"locked_sessions={summary['locked_sessions']}",
            f"outcomes_redacted={summary['outcomes_redacted']}",
            f"elapsed_seconds={elapsed:.2f}",
            f"outputs={protocol.output_dir}",
        )
    )
    terminal_summary_path = protocol.output_dir / "terminal_summary.txt"
    terminal_summary_path.write_text(terminal_summary, encoding="utf-8")
    print(terminal_summary)
    return SessionAuditResult(
        output_dir=protocol.output_dir,
        audit_path=audit_path,
        manifest_path=manifest_path,
        summary_path=summary_path,
        terminal_summary_path=terminal_summary_path,
    )


def _instrument_summary(
    *,
    audited: tuple[_AuditRecord, ...],
    protocol: research_models.ResearchProtocol,
) -> dict[str, object]:
    """Summarize coverage and technical exclusions without locked outcomes."""
    result: dict[str, object] = {}
    for instrument in protocol.instruments:
        records = tuple(
            item for item in audited if item.identity.instrument is instrument
        )
        observed_locked_dates = {
            item.identity.trading_date
            for item in records
            if bool(item.row["evaluation_locked"])
        }
        exclusion_reasons: Counter[str] = Counter()
        for item in records:
            exclusion_reasons.update(
                reason
                for reason in str(item.row["exclusion_reasons"]).split(";")
                if reason
            )
        result[instrument.value] = {
            "discovered_sessions": len(records),
            "eligible_sessions": sum(item.eligible for item in records),
            "excluded_sessions": sum(not item.eligible for item in records),
            "locked_sessions": len(observed_locked_dates),
            "missing_declared_locked_dates": [
                item.isoformat()
                for item in protocol.split_policy.locked_evaluation_dates
                if item not in observed_locked_dates
            ],
            "exclusion_reasons": dict(sorted(exclusion_reasons.items())),
        }
    return result


def _audit_files(
    *,
    files: tuple[raw_file_repository.RawDataFile, ...],
    protocol: research_models.ResearchProtocol,
) -> Generator[_AuditRecord]:
    """Audit independent sessions with bounded process-level parallelism."""
    if protocol.audit_workers == 1:
        for position, raw_data_file in enumerate(files, start=1):
            result = _audit_file(raw_data_file=raw_data_file, protocol=protocol)
            _print_progress(position=position, total=len(files), result=result)
            yield result
        return
    with futures.ProcessPoolExecutor(max_workers=protocol.audit_workers) as executor:
        pending = {
            executor.submit(
                _audit_file, raw_data_file=raw_data_file, protocol=protocol
            ): raw_data_file
            for raw_data_file in files
        }
        completed = 0
        for future in futures.as_completed(pending):
            result = future.result()
            completed += 1
            _print_progress(position=completed, total=len(files), result=result)
            yield result


def _print_progress(*, position: int, total: int, result: _AuditRecord) -> None:
    print(
        f"[session-audit] {position}/{total} "
        f"{result.identity.instrument.value} "
        f"{result.identity.trading_date.isoformat()} "
        f"eligible={result.eligible}",
        flush=True,
    )


def _audit_file(
    *,
    raw_data_file: raw_file_repository.RawDataFile,
    protocol: research_models.ResearchProtocol,
) -> _AuditRecord:
    instrument = orderbook_models.Instrument(raw_data_file.instrument)
    source_sha256 = raw_file_repository.get_content_fingerprint(
        raw_data_file=raw_data_file
    )
    identity = research_models.SessionIdentity(
        instrument=instrument,
        trading_date=raw_data_file.trading_date,
        path=raw_data_file.path,
        sha256=source_sha256,
    )
    audit = ebs_csv.ParseAudit()
    tracker = _RecordTracker()
    try:
        with closing(
            ebs_csv.parse_rows(
                path=raw_data_file.path,
                expected_instrument=instrument,
                expected_trading_date=raw_data_file.trading_date,
                audit=audit,
            )
        ) as parsed:
            states = pilot_operations.build_native_states(
                records=_track_records(records=parsed, tracker=tracker),
                instrument=instrument,
                trading_date=raw_data_file.trading_date,
                grid_steps=None,
                maximum_staleness_steps=(
                    protocol.maximum_staleness_milliseconds
                    // protocol.state_interval_milliseconds
                ),
                maximum_depth=protocol.audit_policy.required_depth,
            )
        record = _successful_record(
            identity=identity,
            raw_data_file=raw_data_file,
            protocol=protocol,
            audit=audit,
            tracker=tracker,
            states=states,
        )
        if (
            record.eligible
            and identity.instrument is protocol.development_instrument
            and identity.trading_date <= protocol.split_policy.development_end_date
            and identity.trading_date
            not in protocol.split_policy.locked_evaluation_dates
        ):
            cache_dir = protocol.output_dir / "native_cache" / identity.instrument.value
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / f"{identity.trading_date.isoformat()}.parquet"
            states.write_parquet(cache_path)
            record = attrs.evolve(
                record,
                row={
                    **record.row,
                    "native_cache_sha256": _sha256_file(path=cache_path),
                },
            )
        return record
    except (
        OSError,
        ValueError,
        ebs_csv.UnableToParseRowError,
        pilot_operations.InvalidNativeStateError,
        pl.exceptions.PolarsError,
    ) as exc:
        return _failed_record(
            identity=identity,
            raw_data_file=raw_data_file,
            protocol=protocol,
            audit=audit,
            tracker=tracker,
            error=f"{type(exc).__name__}: {exc}",
        )


def _successful_record(
    *,
    identity: research_models.SessionIdentity,
    raw_data_file: raw_file_repository.RawDataFile,
    protocol: research_models.ResearchProtocol,
    audit: ebs_csv.ParseAudit,
    tracker: _RecordTracker,
    states: pl.DataFrame,
) -> _AuditRecord:
    if tracker.first_timestamp is None or tracker.latest_timestamp is None:
        raise ValueError("parsed session contains no timestamped records")
    duration_milliseconds = int(
        (tracker.latest_timestamp - tracker.first_timestamp).total_seconds() * 1_000
    )
    bid = pl.col(orderbook_models.bid_price_col(level=1))
    ask = pl.col(orderbook_models.ask_price_col(level=1))
    present = bid.is_not_null() & ask.is_not_null()
    fresh = (
        pl.col(pilot_models.COL_BID_AGE_MILLISECONDS)
        <= protocol.maximum_staleness_milliseconds
    ) & (
        pl.col(pilot_models.COL_ASK_AGE_MILLISECONDS)
        <= protocol.maximum_staleness_milliseconds
    )
    l1_observed = present & fresh & (bid < ask)
    structural = states.select(
        l1_observed.sum().alias("observed_states"),
        (~present).sum().alias("missing_l1_states"),
        (present & ~fresh).sum().alias("stale_l1_states"),
        (present & fresh & (bid >= ask)).sum().alias("crossed_l1_states"),
        pl.col(orderbook_models.COL_BOOK_OBSERVED)
        .sum()
        .alias("required_depth_observed_states"),
        (pl.col(pilot_models.COL_BID_UPDATED) | pl.col(pilot_models.COL_ASK_UPDATED))
        .sum()
        .alias("quote_update_states"),
    ).row(0, named=True)
    required_depth_observed_states = int(structural["required_depth_observed_states"])
    eligible, reasons = research_operations.evaluate_session_eligibility(
        duration_milliseconds=duration_milliseconds,
        required_depth_observed_states=required_depth_observed_states,
        maximum_source_depth=tracker.maximum_source_depth,
        parse_error=None,
        policy=protocol.audit_policy,
    )
    locked = identity.trading_date in protocol.split_policy.locked_evaluation_dates
    redact = locked and protocol.audit_policy.redact_locked_outcomes
    row = _base_row(
        identity=identity,
        raw_data_file=raw_data_file,
        audit=audit,
        tracker=tracker,
        duration_milliseconds=duration_milliseconds,
        grid_states=states.height,
        structural=structural,
        locked=locked,
        redact=redact,
        eligible=eligible,
        reasons=reasons,
        parse_error=None,
    )
    if redact:
        _redact_target_fields(row=row, protocol=protocol)
    else:
        target_states = states.with_columns(
            l1_observed.alias(orderbook_models.COL_BOOK_OBSERVED),
            pl.when(l1_observed)
            .then((bid + ask) / 2.0)
            .otherwise(None)
            .alias(orderbook_models.COL_MID_PRICE),
        )
        target_states = pilot_operations.add_direction_targets(
            data=target_states, horizon_steps=protocol.horizon_steps
        )
        balances = pilot_operations.target_balance(
            data=target_states, horizon_steps=protocol.horizon_steps
        )
        for balance in balances.iter_rows(named=True):
            suffix = f"h{int(balance['horizon_milliseconds'])}"
            for field in ("down", "flat", "up", "total", "flat_percentage"):
                row[f"{field}_{suffix}"] = balance[field]
            row[f"price_changes_{suffix}"] = int(balance["down"]) + int(balance["up"])
    return _AuditRecord(identity=identity, eligible=eligible, row=row)


def _failed_record(
    *,
    identity: research_models.SessionIdentity,
    raw_data_file: raw_file_repository.RawDataFile,
    protocol: research_models.ResearchProtocol,
    audit: ebs_csv.ParseAudit,
    tracker: _RecordTracker,
    error: str,
) -> _AuditRecord:
    duration_milliseconds = (
        int(
            (tracker.latest_timestamp - tracker.first_timestamp).total_seconds() * 1_000
        )
        if tracker.first_timestamp is not None and tracker.latest_timestamp is not None
        else 0
    )
    _, reasons = research_operations.evaluate_session_eligibility(
        duration_milliseconds=duration_milliseconds,
        required_depth_observed_states=0,
        maximum_source_depth=tracker.maximum_source_depth,
        parse_error=error,
        policy=protocol.audit_policy,
    )
    locked = identity.trading_date in protocol.split_policy.locked_evaluation_dates
    row = _base_row(
        identity=identity,
        raw_data_file=raw_data_file,
        audit=audit,
        tracker=tracker,
        duration_milliseconds=duration_milliseconds,
        grid_states=0,
        structural={
            "observed_states": 0,
            "missing_l1_states": 0,
            "stale_l1_states": 0,
            "crossed_l1_states": 0,
            "required_depth_observed_states": 0,
            "quote_update_states": 0,
        },
        locked=locked,
        redact=locked and protocol.audit_policy.redact_locked_outcomes,
        eligible=False,
        reasons=reasons,
        parse_error=error,
    )
    _redact_target_fields(row=row, protocol=protocol)
    return _AuditRecord(identity=identity, eligible=False, row=row)


def _redact_target_fields(
    *, row: dict[str, object], protocol: research_models.ResearchProtocol
) -> None:
    """Add empty target fields without calculating any locked outcome."""
    for horizon in protocol.forecast_horizons_milliseconds:
        suffix = f"h{horizon}"
        for field in (
            "down",
            "flat",
            "up",
            "total",
            "flat_percentage",
            "price_changes",
        ):
            row[f"{field}_{suffix}"] = None


def _base_row(
    *,
    identity: research_models.SessionIdentity,
    raw_data_file: raw_file_repository.RawDataFile,
    audit: ebs_csv.ParseAudit,
    tracker: _RecordTracker,
    duration_milliseconds: int,
    grid_states: int,
    structural: dict[str, object],
    locked: bool,
    redact: bool,
    eligible: bool,
    reasons: tuple[str, ...],
    parse_error: str | None,
) -> dict[str, object]:
    denominator = max(grid_states, 1)
    return {
        "instrument": identity.instrument.value,
        "trading_date": identity.trading_date,
        "raw_path": str(identity.path),
        "sha256": identity.sha256,
        "compressed_size_bytes": raw_data_file.size_bytes,
        "physical_lines": audit.physical_lines,
        "quote_rows": audit.quote_rows,
        "deal_rows": audit.deal_rows,
        "empty_lines": audit.empty_lines,
        "error_rows": audit.error_rows,
        "first_source_timestamp": tracker.first_timestamp,
        "last_source_timestamp": tracker.latest_timestamp,
        "duration_milliseconds": duration_milliseconds,
        "maximum_source_depth": tracker.maximum_source_depth,
        "grid_states": grid_states,
        **structural,
        "observed_percentage": 100.0
        * cast(int, structural["observed_states"])
        / denominator,
        "required_depth_observed_percentage": 100.0
        * cast(int, structural["required_depth_observed_states"])
        / denominator,
        "evaluation_locked": locked,
        "outcomes_redacted": redact,
        "eligible": eligible,
        "exclusion_reasons": ";".join(reasons),
        "parse_error": parse_error,
        "native_cache_sha256": None,
    }


def _track_records(
    *, records: Iterable[orderbook_models.RawRecord], tracker: _RecordTracker
) -> Generator[orderbook_models.RawRecord]:
    for record in records:
        tracker.observe(record=record)
        yield record


def _manifest(
    *,
    protocol: research_models.ResearchProtocol,
    protocol_path: Path,
    audited: tuple[_AuditRecord, ...],
    audit_path: Path,
) -> dict[str, object]:
    development_folds: dict[str, object] = {}
    locked = frozenset(protocol.split_policy.locked_evaluation_dates)
    for instrument in protocol.instruments:
        sessions = tuple(
            item.identity
            for item in audited
            if item.eligible and item.identity.instrument is instrument
        )
        folds = research_operations.build_rolling_folds(
            sessions=sessions, policy=protocol.split_policy
        )
        development_folds[instrument.value] = [
            {
                "identifier": fold.identifier,
                "training_sessions": [
                    _manifest_session(identity=item, data_dir=protocol.data_dir)
                    for item in fold.training_sessions
                ],
                "validation_sessions": [
                    _manifest_session(identity=item, data_dir=protocol.data_dir)
                    for item in fold.validation_sessions
                ],
            }
            for fold in folds
        ]
    final_test = {
        instrument.value: [
            _manifest_session(identity=item.identity, data_dir=protocol.data_dir)
            for item in audited
            if item.eligible
            and item.identity.instrument is instrument
            and item.identity.trading_date in locked
            and item.identity.trading_date > protocol.split_policy.development_end_date
        ]
        for instrument in protocol.instruments
    }
    reserved_earlier = {
        instrument.value: [
            item.identity.trading_date.isoformat()
            for item in audited
            if item.eligible
            and item.identity.instrument is instrument
            and item.identity.trading_date in locked
            and item.identity.trading_date <= protocol.split_policy.development_end_date
        ]
        for instrument in protocol.instruments
    }
    return {
        "schema_version": 1,
        "protocol_sha256": _sha256_file(path=protocol_path),
        "audit_sha256": _sha256_file(path=audit_path),
        "development_end_date": protocol.split_policy.development_end_date.isoformat(),
        "development_folds": development_folds,
        "final_test_sessions": final_test,
        "reserved_earlier_locked_dates": reserved_earlier,
        "rules": {
            "development_instrument": protocol.development_instrument.value,
            "native_state_interval_milliseconds": protocol.state_interval_milliseconds,
            "forecast_horizons_milliseconds": list(
                protocol.forecast_horizons_milliseconds
            ),
            "training_stride_milliseconds": dict(protocol.training_stride_milliseconds),
            "evaluation_stride_milliseconds": (protocol.evaluation_stride_milliseconds),
            "depths": list(protocol.depths),
            "models": list(protocol.models),
            "random_seeds": list(protocol.random_seeds),
            "validation_checks_per_epoch": protocol.validation_checks_per_epoch,
            "primary_metrics": [item.value for item in protocol.primary_metrics],
            "supporting_metrics": [item.value for item in protocol.supporting_metrics],
            "bootstrap_repetitions": protocol.bootstrap_repetitions,
            "confidence_level": protocol.confidence_level,
        },
    }


def _manifest_session(
    *, identity: research_models.SessionIdentity, data_dir: Path
) -> dict[str, str]:
    return {
        "trading_date": identity.trading_date.isoformat(),
        "raw_path": str(identity.path.relative_to(data_dir)),
        "sha256": identity.sha256,
    }


def _sha256_file(*, path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
