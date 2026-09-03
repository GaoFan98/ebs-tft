"""Evaluate defensive baselines across immutable rolling session folds."""

from __future__ import annotations

import datetime
import hashlib
import json
import time
from pathlib import Path
from typing import cast

import attrs
import numpy as np
import polars as pl
import yaml

from ebs_tft.data.repositories import artifact as artifact_repository
from ebs_tft.domain import model as model_domain
from ebs_tft.domain.orderbook import models as orderbook_models
from ebs_tft.domain.pilot import operations as pilot_operations
from ebs_tft.domain.pilot import training as pilot_training
from ebs_tft.domain.research import models as research_models
from ebs_tft.domain.research import operations as research_operations


@attrs.frozen
class BaselineGateResult:
    """Reference rolling-baseline metrics and the finite gate decision."""

    output_dir: Path
    metrics_path: Path
    comparisons_path: Path
    gate_path: Path
    terminal_summary_path: Path


def run(
    *,
    protocol: research_models.ResearchProtocol,
    protocol_path: Path,
    replace_output: bool = False,
) -> BaselineGateResult:
    """Run rolling defensive baselines without fitting a neural network."""
    started = time.perf_counter()
    manifest_path = protocol.output_dir / "split_manifest.yaml"
    audit_path = protocol.output_dir / "session_audit.csv"
    folds = _load_folds(
        manifest_path=manifest_path,
        audit_path=audit_path,
        protocol_path=protocol_path,
        protocol=protocol,
    )
    _verify_cached_states(folds=folds, audit_path=audit_path, protocol=protocol)
    output_dir = protocol.output_dir / "baseline_gate"
    artifact_repository.prepare_run_directory(path=output_dir, replace=replace_output)
    metric_rows: list[dict[str, object]] = []
    for fold_position, fold in enumerate(folds, start=1):
        print(
            f"[baseline-gate] fold={fold.identifier} "
            f"position={fold_position}/{len(folds)}",
            flush=True,
        )
        training_states = tuple(
            _load_cached_states(protocol=protocol, identity=item)
            for item in fold.training_sessions
        )
        validation_states = tuple(
            _load_cached_states(protocol=protocol, identity=item)
            for item in fold.validation_sessions
        )
        training_states = tuple(
            pilot_operations.add_direction_targets(
                data=frame, horizon_steps=protocol.horizon_steps
            )
            for frame in training_states
        )
        validation_states = tuple(
            pilot_operations.add_direction_targets(
                data=frame, horizon_steps=protocol.horizon_steps
            )
            for frame in validation_states
        )
        for depth in protocol.depths:
            for horizon_steps in protocol.horizon_steps:
                raw_training = tuple(
                    pilot_training.extract_session(
                        data=frame,
                        trading_date=identity.trading_date,
                        depth=depth,
                        horizon_steps=horizon_steps,
                    )
                    for frame, identity in zip(
                        training_states, fold.training_sessions, strict=True
                    )
                )
                raw_validation = tuple(
                    pilot_training.extract_session(
                        data=frame,
                        trading_date=identity.trading_date,
                        depth=depth,
                        horizon_steps=horizon_steps,
                    )
                    for frame, identity in zip(
                        validation_states, fold.validation_sessions, strict=True
                    )
                )
                scaler = pilot_training.fit_feature_scaler(sessions=raw_training)
                scaled_training = tuple(
                    pilot_training.apply_feature_scaler(session=item, scaler=scaler)
                    for item in raw_training
                )
                scaled_validation = tuple(
                    pilot_training.apply_feature_scaler(session=item, scaler=scaler)
                    for item in raw_validation
                )
                horizon_milliseconds = (
                    horizon_steps * protocol.state_interval_milliseconds
                )
                training = pilot_training.combine_sessions(
                    sessions=scaled_training,
                    context_steps=(
                        protocol.context_milliseconds
                        // protocol.state_interval_milliseconds
                    ),
                    horizon_steps=horizon_steps,
                    maximum_windows=None,
                    stride_steps=research_operations.training_stride_steps(
                        protocol=protocol,
                        horizon_milliseconds=horizon_milliseconds,
                    ),
                )
                validation = pilot_training.combine_sessions(
                    sessions=scaled_validation,
                    context_steps=(
                        protocol.context_milliseconds
                        // protocol.state_interval_milliseconds
                    ),
                    horizon_steps=horizon_steps,
                    maximum_windows=None,
                    stride_steps=(
                        protocol.evaluation_stride_milliseconds
                        // protocol.state_interval_milliseconds
                    ),
                )
                baseline_probabilities = model_domain.fit_defensive_baselines(
                    training=training, evaluation=validation
                )
                metric_rows.extend(
                    _session_metric_rows(
                        instrument=protocol.development_instrument,
                        fold=fold,
                        corpus=validation,
                        probabilities=baseline_probabilities,
                        depth=depth,
                        horizon_steps=horizon_steps,
                    )
                )
        del training_states, validation_states

    metrics = pl.DataFrame(metric_rows).sort(
        ["fold", "validation_date", "horizon_steps", "model", "depth"]
    )
    metrics_path = output_dir / "session_metrics.csv"
    metrics.write_csv(metrics_path)
    comparisons = _paired_comparisons(metrics=metrics, protocol=protocol)
    comparisons_path = output_dir / "paired_session_comparisons.csv"
    comparisons.write_csv(comparisons_path)
    gate = _gate_decision(comparisons=comparisons, protocol=protocol)
    gate_path = output_dir / "gate_decision.json"
    gate_path.write_text(json.dumps(gate, indent=2), encoding="utf-8")
    elapsed = time.perf_counter() - started
    terminal_summary = "\n".join(
        (
            "EBS rolling defensive-baseline gate completed",
            "WARNING: development evidence only; locked evaluation was not used.",
            f"instrument={protocol.development_instrument.value}",
            f"folds={len(folds)}",
            f"validation_sessions={metrics['validation_date'].n_unique()}",
            f"eligible_for_neural_benchmark={gate['eligible_for_neural_benchmark']}",
            f"deeper_depth_supported={gate['deeper_depth_supported']}",
            f"elapsed_seconds={elapsed:.2f}",
            f"outputs={output_dir}",
        )
    )
    terminal_summary_path = output_dir / "terminal_summary.txt"
    terminal_summary_path.write_text(terminal_summary, encoding="utf-8")
    print(terminal_summary)
    return BaselineGateResult(
        output_dir=output_dir,
        metrics_path=metrics_path,
        comparisons_path=comparisons_path,
        gate_path=gate_path,
        terminal_summary_path=terminal_summary_path,
    )


def _load_folds(
    *,
    manifest_path: Path,
    audit_path: Path,
    protocol_path: Path,
    protocol: research_models.ResearchProtocol,
) -> tuple[research_models.RollingFold, ...]:
    if not manifest_path.is_file() or not audit_path.is_file():
        raise FileNotFoundError("run research-session-audit before baseline evaluation")
    with manifest_path.open(encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)
    if not isinstance(loaded, dict):
        raise ValueError("split manifest must be a mapping")
    manifest = cast(dict[str, object], loaded)
    if manifest.get("protocol_sha256") != _sha256_file(path=protocol_path):
        raise ValueError("split manifest does not match the research protocol")
    if manifest.get("audit_sha256") != _sha256_file(path=audit_path):
        raise ValueError("split manifest does not match the session audit")
    all_folds = manifest.get("development_folds")
    if not isinstance(all_folds, dict):
        raise ValueError("split manifest development_folds must be a mapping")
    raw_folds = all_folds.get(protocol.development_instrument.value)
    if not isinstance(raw_folds, list):
        raise ValueError("manifest has no development-instrument folds")
    folds: list[research_models.RollingFold] = []
    for raw_fold in raw_folds:
        if not isinstance(raw_fold, dict):
            raise ValueError("manifest fold must be a mapping")
        fold = cast(dict[str, object], raw_fold)
        identifier = fold.get("identifier")
        if not isinstance(identifier, str):
            raise ValueError("manifest fold identifier must be a string")
        folds.append(
            research_models.RollingFold(
                identifier=identifier,
                training_sessions=_manifest_sessions(
                    value=fold.get("training_sessions"), protocol=protocol
                ),
                validation_sessions=_manifest_sessions(
                    value=fold.get("validation_sessions"), protocol=protocol
                ),
            )
        )
    return tuple(folds)


def _manifest_sessions(
    *, value: object, protocol: research_models.ResearchProtocol
) -> tuple[research_models.SessionIdentity, ...]:
    if not isinstance(value, list):
        raise ValueError("manifest sessions must be a list")
    sessions: list[research_models.SessionIdentity] = []
    for raw_session in value:
        if not isinstance(raw_session, dict):
            raise ValueError("manifest session must be a mapping")
        session = cast(dict[str, object], raw_session)
        if set(session) != {"trading_date", "raw_path", "sha256"} or not all(
            isinstance(session[key], str) for key in session
        ):
            raise ValueError("manifest session has invalid fields")
        sessions.append(
            research_models.SessionIdentity(
                instrument=protocol.development_instrument,
                trading_date=datetime.date.fromisoformat(
                    cast(str, session["trading_date"])
                ),
                path=protocol.data_dir / cast(str, session["raw_path"]),
                sha256=cast(str, session["sha256"]),
            )
        )
    return tuple(sessions)


def _load_cached_states(
    *,
    protocol: research_models.ResearchProtocol,
    identity: research_models.SessionIdentity,
) -> pl.DataFrame:
    path = (
        protocol.output_dir
        / "native_cache"
        / identity.instrument.value
        / f"{identity.trading_date.isoformat()}.parquet"
    )
    if not path.is_file():
        raise FileNotFoundError(f"audited native-state cache is missing: {path}")
    return pl.read_parquet(path)


def _verify_cached_states(
    *,
    folds: tuple[research_models.RollingFold, ...],
    audit_path: Path,
    protocol: research_models.ResearchProtocol,
) -> None:
    """Reject missing or modified derived state artifacts before evaluation."""
    audit = pl.read_csv(audit_path, schema_overrides={"trading_date": pl.Date})
    expected_hashes = {
        cast(datetime.date, row["trading_date"]): cast(str, row["native_cache_sha256"])
        for row in audit.filter(
            (pl.col("instrument") == protocol.development_instrument.value)
            & pl.col("native_cache_sha256").is_not_null()
        ).iter_rows(named=True)
    }
    identities = {
        item.trading_date: item
        for fold in folds
        for item in (*fold.training_sessions, *fold.validation_sessions)
    }
    for trading_date, identity in identities.items():
        if not identity.path.is_file() or _sha256_file(path=identity.path) != (
            identity.sha256
        ):
            raise ValueError(
                f"raw source no longer matches the split manifest: {identity.path}"
            )
        try:
            expected_hash = expected_hashes[trading_date]
        except KeyError as exc:
            raise ValueError(
                f"audit has no native-cache hash for {trading_date.isoformat()}"
            ) from exc
        cache_path = (
            protocol.output_dir
            / "native_cache"
            / identity.instrument.value
            / f"{trading_date.isoformat()}.parquet"
        )
        if not cache_path.is_file() or _sha256_file(path=cache_path) != expected_hash:
            raise ValueError(
                f"native-state cache failed integrity verification: {cache_path}"
            )


def _session_metric_rows(
    *,
    instrument: orderbook_models.Instrument,
    fold: research_models.RollingFold,
    corpus: pilot_training.PreparedCorpus,
    probabilities: dict[str, np.ndarray],
    depth: int,
    horizon_steps: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    labels = corpus.labels[corpus.target_indices]
    for offset, length, session in zip(
        corpus.session_offsets,
        corpus.session_lengths,
        fold.validation_sessions,
        strict=True,
    ):
        selection = (corpus.target_indices >= offset) & (
            corpus.target_indices < offset + length
        )
        for model_name, model_probabilities in probabilities.items():
            row = model_domain.direction_metric_row(
                model_name=model_name,
                depth=depth,
                horizon_steps=horizon_steps,
                seed=-1,
                labels=labels[selection],
                probabilities=model_probabilities[selection],
                parameter_count=0,
            )
            row.update(
                {
                    "instrument": instrument.value,
                    "fold": fold.identifier,
                    "validation_date": session.trading_date,
                }
            )
            rows.append(row)
    return rows


def _paired_comparisons(
    *, metrics: pl.DataFrame, protocol: research_models.ResearchProtocol
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    metric_names = tuple(
        item.value for item in (*protocol.primary_metrics, *protocol.supporting_metrics)
    )
    for horizon_steps in protocol.horizon_steps:
        horizon_data = metrics.filter(pl.col("horizon_steps") == horizon_steps)
        for model_name in ("empirical_prior", "last_move", "majority", "logistic"):
            model_data = horizon_data.filter(pl.col("model") == model_name)
            for metric_name in metric_names:
                shallower = _metric_by_session(
                    data=model_data.filter(pl.col("depth") == min(protocol.depths)),
                    metric_name=metric_name,
                )
                deeper = _metric_by_session(
                    data=model_data.filter(pl.col("depth") == max(protocol.depths)),
                    metric_name=metric_name,
                )
                interval = research_operations.paired_session_interval(
                    shallower_by_session=shallower,
                    deeper_by_session=deeper,
                    repetitions=protocol.bootstrap_repetitions,
                    confidence_level=protocol.confidence_level,
                    random_seed=17 + horizon_steps,
                )
                rows.append(
                    {
                        "comparison": "deeper_minus_shallower",
                        "model": model_name,
                        "horizon_steps": horizon_steps,
                        "horizon_milliseconds": (
                            horizon_steps * protocol.state_interval_milliseconds
                        ),
                        "metric": metric_name,
                        "favorable_direction": _favorable_direction(
                            metric_name=metric_name
                        ),
                        **interval,
                    }
                )
        logistic_l1 = horizon_data.filter(
            (pl.col("model") == "logistic") & (pl.col("depth") == min(protocol.depths))
        )
        prior_l1 = horizon_data.filter(
            (pl.col("model") == "empirical_prior")
            & (pl.col("depth") == min(protocol.depths))
        )
        for metric_name in metric_names:
            interval = research_operations.paired_session_interval(
                shallower_by_session=_metric_by_session(
                    data=prior_l1, metric_name=metric_name
                ),
                deeper_by_session=_metric_by_session(
                    data=logistic_l1, metric_name=metric_name
                ),
                repetitions=protocol.bootstrap_repetitions,
                confidence_level=protocol.confidence_level,
                random_seed=71 + horizon_steps,
            )
            rows.append(
                {
                    "comparison": "logistic_l1_minus_empirical_prior_l1",
                    "model": "logistic",
                    "horizon_steps": horizon_steps,
                    "horizon_milliseconds": (
                        horizon_steps * protocol.state_interval_milliseconds
                    ),
                    "metric": metric_name,
                    "favorable_direction": _favorable_direction(
                        metric_name=metric_name
                    ),
                    **interval,
                }
            )
    return pl.DataFrame(rows).sort(["comparison", "horizon_steps", "model", "metric"])


def _metric_by_session(*, data: pl.DataFrame, metric_name: str) -> dict[str, float]:
    return {
        f"{row['fold']}:{row['validation_date']}": float(row[metric_name])
        for row in data.iter_rows(named=True)
    }


def _favorable_direction(*, metric_name: str) -> str:
    """Describe how to interpret a raw second-minus-first metric delta."""
    if metric_name in {
        research_models.EvaluationMetric.LOG_LOSS.value,
        research_models.EvaluationMetric.MULTICLASS_BRIER.value,
    }:
        return "negative"
    return "positive"


def _gate_decision(
    *, comparisons: pl.DataFrame, protocol: research_models.ResearchProtocol
) -> dict[str, object]:
    primary = tuple(item.value for item in protocol.primary_metrics)
    signal_rows = comparisons.filter(
        (pl.col("comparison") == "logistic_l1_minus_empirical_prior_l1")
        & pl.col("metric").is_in(primary)
    )
    depth_rows = comparisons.filter(
        (pl.col("comparison") == "deeper_minus_shallower")
        & (pl.col("model") == "logistic")
        & pl.col("metric").is_in(primary)
    )
    signal_by_horizon = {
        str(horizon): bool(
            cast(
                float,
                signal_rows.filter(pl.col("horizon_milliseconds") == horizon)[
                    "confidence_lower"
                ].min(),
            )
            > 0.0
        )
        for horizon in protocol.forecast_horizons_milliseconds
    }
    depth_by_horizon = {
        str(horizon): bool(
            cast(
                float,
                depth_rows.filter(pl.col("horizon_milliseconds") == horizon)[
                    "confidence_lower"
                ].min(),
            )
            > 0.0
        )
        for horizon in protocol.forecast_horizons_milliseconds
    }
    return {
        "decision_rule": (
            "A horizon passes only when the lower session-bootstrap confidence "
            "bound is above zero for every primary metric."
        ),
        "primary_metrics": list(primary),
        "baseline_signal_by_horizon": signal_by_horizon,
        "depth_support_by_horizon": depth_by_horizon,
        "eligible_for_neural_benchmark": any(signal_by_horizon.values()),
        "deeper_depth_supported": any(depth_by_horizon.values()),
        "locked_evaluation_used": False,
    }


def _sha256_file(*, path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
