"""Freeze and run the retrospective March Level-1 versus Level-10 replication."""

from __future__ import annotations

import json
import platform
import time
from contextlib import closing
from pathlib import Path
from typing import cast

import attrs
import polars as pl
import sklearn
import torch

from ebs_tft.application.usecases.research_protocol import (
    _baseline,
    _depth_extension,
    _locked,
    _neural,
)
from ebs_tft.data.parsers import ebs_csv
from ebs_tft.domain import model as model_domain
from ebs_tft.domain.pilot import operations as pilot_operations
from ebs_tft.domain.pilot import training as pilot_training
from ebs_tft.domain.research import models as research_models
from ebs_tft.domain.research import operations as research_operations

DEPTH_REPLICATION_IMPLEMENTATION_VERSION = 1
_HORIZON_MILLISECONDS = 30_000
_SHALLOW_DEPTH = 1
_DEEP_DEPTH = 10


@attrs.frozen
class DepthReplicationPlanResult:
    """Reference the frozen retrospective depth-replication plan."""

    plan_path: Path
    plan_sha256: str


@attrs.frozen
class DepthReplicationResult:
    """Reference the completed retrospective depth-replication evidence."""

    output_dir: Path
    metrics_path: Path
    comparisons_path: Path
    decision_path: Path
    workbook_path: Path


class DepthReplicationPausedError(Exception):
    """Indicate a safe pause after a bounded number of Level-10 final cells."""

    def __init__(self, *, completed_cells: int, total_cells: int, output_dir: Path):
        self.completed_cells = completed_cells
        self.total_cells = total_cells
        self.output_dir = output_dir
        super().__init__(
            "Depth replication paused safely after "
            f"{completed_cells}/{total_cells} cells; resume with the same plan hash. "
            f"Outputs: {output_dir}"
        )


def freeze_plan(
    *,
    protocol: research_models.ResearchProtocol,
    protocol_path: Path,
    policy: research_models.NeuralBenchmarkPolicy,
    policy_path: Path,
) -> DepthReplicationPlanResult:
    """Freeze supported models and Level-10 durations before their March run."""
    _depth_extension._validate_design(protocol=protocol)
    manifest_path = protocol.output_dir / "split_manifest.yaml"
    audit_path = protocol.output_dir / "session_audit.csv"
    folds = _baseline._load_folds(
        manifest_path=manifest_path,
        audit_path=audit_path,
        protocol_path=protocol_path,
        protocol=protocol,
    )
    _baseline._verify_cached_states(
        folds=folds, audit_path=audit_path, protocol=protocol
    )
    depth_dir = protocol.output_dir / "depth_extension"
    locked_dir = protocol.output_dir / "locked_evaluation"
    depth_paths = {
        "depth_run_identity_sha256": depth_dir / "run_identity.json",
        "depth_run_summary_sha256": depth_dir / "run_summary.json",
        "depth_metrics_sha256": depth_dir / "session_metrics.csv",
        "depth_comparisons_sha256": depth_dir / "paired_depth_comparisons.csv",
        "depth_decision_sha256": depth_dir / "decision.json",
        "depth_cache_manifest_sha256": depth_dir / "level_10_cache_manifest.json",
    }
    locked_paths = {
        "level_1_plan_sha256": locked_dir / "plan.json",
        "level_1_run_summary_sha256": locked_dir / "run_summary.json",
        "level_1_metrics_sha256": locked_dir / "session_metrics.csv",
        "level_1_decision_sha256": locked_dir / "decision.json",
    }
    _require_files(paths=(*depth_paths.values(), *locked_paths.values()))
    comparisons = pl.read_csv(depth_paths["depth_comparisons_sha256"])
    decision = _neural._json_mapping(path=depth_paths["depth_decision_sha256"])
    if decision != _depth_extension._decision(
        comparisons=comparisons, protocol=protocol
    ):
        raise ValueError("depth decision does not match its comparison evidence")
    supported = decision.get("supported_models")
    if (
        not isinstance(supported, list)
        or not supported
        or not all(
            isinstance(model, str) and model in protocol.models for model in supported
        )
    ):
        raise ValueError("development depth extension admitted no valid model")
    depth_summary = _neural._json_mapping(path=depth_paths["depth_run_summary_sha256"])
    if depth_summary.get("status") != "complete":
        raise ValueError("development depth extension is incomplete")
    locked_summary = _neural._json_mapping(
        path=locked_paths["level_1_run_summary_sha256"]
    )
    locked_decision = _neural._json_mapping(
        path=locked_paths["level_1_decision_sha256"]
    )
    if (
        locked_summary.get("locked_evaluation_used") is not True
        or locked_decision.get("locked_evaluation_used") is not True
    ):
        raise ValueError("completed Level-1 March evidence is required")
    confirmed = locked_decision.get("confirmed_candidates")
    if not isinstance(confirmed, list) or not all(
        {
            "model": model,
            "depth": _SHALLOW_DEPTH,
            "horizon_milliseconds": _HORIZON_MILLISECONDS,
        }
        in confirmed
        for model in supported
    ):
        raise ValueError("supported depth models lack confirmed Level-1 March evidence")

    cells = _development_cells(
        depth_dir=depth_dir,
        folds=folds,
        models=tuple(cast(list[str], supported)),
        seeds=protocol.random_seeds,
    )
    development_sessions = _locked._development_sessions(folds=folds)
    final_sessions = _locked._final_sessions(
        manifest_path=manifest_path, protocol=protocol
    )
    source_hashes = {
        "protocol_sha256": _baseline._sha256_file(path=protocol_path),
        "policy_sha256": _baseline._sha256_file(path=policy_path),
        "manifest_sha256": _baseline._sha256_file(path=manifest_path),
        "audit_sha256": _baseline._sha256_file(path=audit_path),
        **{
            key: _baseline._sha256_file(path=path)
            for key, path in {**depth_paths, **locked_paths}.items()
        },
    }
    plan = {
        "schema_version": 1,
        "implementation_version": DEPTH_REPLICATION_IMPLEMENTATION_VERSION,
        **source_hashes,
        "experiment": {
            "instrument": protocol.development_instrument.value,
            "horizon_milliseconds": _HORIZON_MILLISECONDS,
            "comparison": "paired_seed_mean_level_10_minus_level_1",
            "native_resolution": True,
            "retrospective_replication": True,
        },
        "development_sessions": [
            _locked._identity_payload(item=item, protocol=protocol)
            for item in development_sessions
        ],
        "replication_sessions": [
            _locked._identity_payload(item=item, protocol=protocol)
            for item in final_sessions
        ],
        "cells": cells,
        "policy": attrs.asdict(policy),
        "primary_metrics": [item.value for item in protocol.primary_metrics],
        "supporting_metrics": [item.value for item in protocol.supporting_metrics],
        "bootstrap_repetitions": protocol.bootstrap_repetitions,
        "confidence_level": protocol.confidence_level,
        "training_duration_rule": (
            "upper median Level-10 development best_epoch for each admitted "
            "model and seed"
        ),
        "march_outcomes_previously_inspected": True,
        "retuning_permitted": False,
    }
    output_dir = protocol.output_dir / "depth_replication"
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "plan.json"
    serialized = json.dumps(plan, indent=2)
    if plan_path.is_file() and plan_path.read_text(encoding="utf-8") != serialized:
        raise ValueError("a different depth-replication plan is already frozen")
    _neural._write_text_atomically(text=serialized, path=plan_path)
    plan_sha256 = _baseline._sha256_file(path=plan_path)
    print("EBS retrospective depth-replication plan frozen")
    print("WARNING: March outcomes were previously inspected; this is not pristine.")
    print(f"development_sessions={len(development_sessions)}")
    print(f"replication_sessions={len(final_sessions)}")
    print(f"model_cells={len(cells)}")
    print(f"plan_sha256={plan_sha256}")
    return DepthReplicationPlanResult(plan_path=plan_path, plan_sha256=plan_sha256)


def run(
    *,
    protocol: research_models.ResearchProtocol,
    protocol_path: Path,
    policy: research_models.NeuralBenchmarkPolicy,
    policy_path: Path,
    plan_sha256: str,
    maximum_new_cells: int | None = None,
) -> DepthReplicationResult:
    """Execute the frozen Level-10 models on the retrospective March sessions."""
    if maximum_new_cells is not None and (
        isinstance(maximum_new_cells, bool) or maximum_new_cells <= 0
    ):
        raise ValueError("maximum_new_cells must be positive or null")
    started = time.perf_counter()
    output_dir = protocol.output_dir / "depth_replication"
    plan_path = output_dir / "plan.json"
    if not plan_path.is_file() or _baseline._sha256_file(path=plan_path) != plan_sha256:
        raise ValueError("--plan-sha256 must match the frozen plan exactly")
    if (output_dir / "run_summary.json").is_file():
        raise ValueError("depth replication is already complete and cannot be rerun")
    plan = _neural._json_mapping(path=plan_path)
    _verify_plan_inputs(
        plan=plan,
        protocol=protocol,
        protocol_path=protocol_path,
        policy_path=policy_path,
    )
    development_sessions = _locked._planned_sessions(
        value=plan.get("development_sessions"), protocol=protocol
    )
    replication_sessions = _locked._planned_sessions(
        value=plan.get("replication_sessions"), protocol=protocol
    )
    _locked._verify_sources(sessions=(*development_sessions, *replication_sessions))
    raw_cells = plan.get("cells")
    if not isinstance(raw_cells, list) or not raw_cells:
        raise ValueError("frozen depth-replication plan has no cells")
    cells = tuple(
        cast(dict[str, object], item) for item in raw_cells if isinstance(item, dict)
    )
    if len(cells) != len(raw_cells):
        raise ValueError("frozen depth-replication cell is invalid")
    depth_dir = protocol.output_dir / "depth_extension"
    _verify_development_histories(cells=cells, depth_dir=depth_dir)
    _verify_development_cache(
        sessions=development_sessions,
        cache_dir=depth_dir / "native_cache_level_10",
        manifest_path=depth_dir / "level_10_cache_manifest.json",
    )
    cache_dir = output_dir / "native_cache_level_10"
    _materialize_replication_cache(
        protocol=protocol,
        sessions=replication_sessions,
        cache_dir=cache_dir,
    )
    horizon_steps = _HORIZON_MILLISECONDS // protocol.state_interval_milliseconds
    training_corpus, evaluation_corpus = _prepare_corpora(
        protocol=protocol,
        development_sessions=development_sessions,
        replication_sessions=replication_sessions,
        development_cache_dir=depth_dir / "native_cache_level_10",
        replication_cache_dir=cache_dir,
        working_dir=output_dir,
        horizon_steps=horizon_steps,
    )
    context_steps = (
        protocol.context_milliseconds // protocol.state_interval_milliseconds
    )
    training_dataset = model_domain.SequenceDataset(
        lob_features=training_corpus.lob_features,
        auxiliary_features=training_corpus.auxiliary_features,
        labels=training_corpus.labels,
        target_indices=training_corpus.target_indices,
        context_steps=context_steps,
    )
    evaluation_dataset = model_domain.SequenceDataset(
        lob_features=evaluation_corpus.lob_features,
        auxiliary_features=evaluation_corpus.auxiliary_features,
        labels=evaluation_corpus.labels,
        target_indices=evaluation_corpus.target_indices,
        context_steps=context_steps,
    )
    device = model_domain.select_device(requested=policy.device)
    deep_frames: list[pl.DataFrame] = []
    completed_cells = 0
    resumed_cells = 0
    new_cells = 0
    for cell in cells:
        completed_cells += 1
        completed = _locked._completed_locked_cell(
            output_dir=output_dir,
            cell=cell,
            plan_sha256=plan_sha256,
            final_sessions=replication_sessions,
        )
        if completed is not None:
            deep_frames.append(completed)
            resumed_cells += 1
            continue
        print(
            f"[depth-replication] cell={completed_cells}/{len(cells)} "
            f"model={_locked._string(cell, 'model')} "
            f"seed={_locked._integer(cell, 'seed')} "
            f"fixed_epochs={_locked._integer(cell, 'fixed_epochs')}",
            flush=True,
        )
        deep_frames.append(
            _locked._fit_locked_cell(
                output_dir=output_dir,
                cell=cell,
                plan_sha256=plan_sha256,
                policy=policy,
                protocol=protocol,
                device=device,
                training_dataset=training_dataset,
                evaluation_dataset=evaluation_dataset,
                evaluation_corpus=evaluation_corpus,
                final_sessions=replication_sessions,
            )
        )
        new_cells += 1
        print(
            f"[depth-replication] cell={completed_cells}/{len(cells)} checkpoint=saved",
            flush=True,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if (
            maximum_new_cells is not None
            and new_cells >= maximum_new_cells
            and completed_cells < len(cells)
        ):
            raise DepthReplicationPausedError(
                completed_cells=completed_cells,
                total_cells=len(cells),
                output_dir=output_dir,
            )
    deep_metrics = pl.concat(deep_frames)
    shallow_metrics = _verified_shallow_metrics(
        protocol=protocol,
        models=tuple(_locked._string(cell, "model") for cell in cells),
        replication_sessions=replication_sessions,
        locked_dir=protocol.output_dir / "locked_evaluation",
    )
    metrics = pl.concat([shallow_metrics, deep_metrics]).sort(
        ["depth", "model", "seed", "validation_date"]
    )
    metrics_path = output_dir / "session_metrics.csv"
    _neural._write_csv_atomically(data=metrics, path=metrics_path)
    comparisons = _depth_extension._paired_depth_comparisons(
        metrics=metrics, protocol=protocol
    )
    comparisons_path = output_dir / "paired_depth_comparisons.csv"
    _neural._write_csv_atomically(data=comparisons, path=comparisons_path)
    deltas = _depth_extension._session_depth_deltas(metrics=metrics, protocol=protocol)
    _neural._write_csv_atomically(
        data=deltas, path=output_dir / "session_depth_deltas.csv"
    )
    absolute = _depth_extension._absolute_metric_summary(
        metrics=metrics, protocol=protocol
    )
    _neural._write_csv_atomically(
        data=absolute, path=output_dir / "absolute_metric_summary.csv"
    )
    training = _training_details(output_dir=output_dir)
    _neural._write_csv_atomically(
        data=training, path=output_dir / "training_details.csv"
    )
    decision = _replication_decision(comparisons=comparisons, protocol=protocol)
    decision_path = output_dir / "decision.json"
    _neural._write_text_atomically(
        text=json.dumps(decision, indent=2), path=decision_path
    )
    workbook_path = output_dir / "eurusd_level1_vs_level10_march_replication.xlsx"
    _depth_extension._write_workbook(
        path=workbook_path,
        protocol=protocol,
        decision=decision,
        comparisons=comparisons,
        absolute=absolute,
        deltas=deltas,
        training=training,
        metrics=metrics,
        report_title="EUR/USD — March Level 1 vs Level 10 replication",
        design_description=(
            "Same 4 retrospective March sessions and 2 seeds for both depths; "
            "only development-admitted models are evaluated."
        ),
        limitation=(
            "Retrospective replication: March outcomes were previously inspected. "
            "No retuning or rerun based on these outcomes is permitted."
        ),
    )
    elapsed = time.perf_counter() - started
    summary = {
        "schema_version": 1,
        "implementation_version": DEPTH_REPLICATION_IMPLEMENTATION_VERSION,
        "plan_sha256": plan_sha256,
        "status": "complete",
        "retrospective_replication": True,
        "retuning_permitted": False,
        "resumed_cells": resumed_cells,
        "new_cells": new_cells,
        "elapsed_seconds": elapsed,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "sklearn": sklearn.__version__,
            "device": str(device),
        },
        "artifacts": {
            "session_metrics": str(metrics_path),
            "paired_depth_comparisons": str(comparisons_path),
            "decision": str(decision_path),
            "workbook": str(workbook_path),
        },
    }
    _neural._write_text_atomically(
        text=json.dumps(summary, indent=2), path=output_dir / "run_summary.json"
    )
    terminal = "\n".join(
        (
            "EBS retrospective March depth replication completed",
            "WARNING: March outcomes were previously inspected; "
            "do not retune or rerun.",
            f"instrument={protocol.development_instrument.value}",
            f"replication_sessions={len(replication_sessions)}",
            f"model_cells={len(cells)}",
            "models_with_supported_deeper_depth="
            f"{len(cast(list[object], decision['supported_models']))}",
            f"elapsed_seconds={elapsed:.2f}",
            f"report={workbook_path}",
            f"outputs={output_dir}",
        )
    )
    _neural._write_text_atomically(
        text=terminal, path=output_dir / "terminal_summary.txt"
    )
    print(terminal)
    return DepthReplicationResult(
        output_dir=output_dir,
        metrics_path=metrics_path,
        comparisons_path=comparisons_path,
        decision_path=decision_path,
        workbook_path=workbook_path,
    )


def _development_cells(
    *,
    depth_dir: Path,
    folds: tuple[research_models.RollingFold, ...],
    models: tuple[str, ...],
    seeds: tuple[int, ...],
) -> list[dict[str, object]]:
    cells: list[dict[str, object]] = []
    for model in models:
        for seed in seeds:
            histories: list[dict[str, object]] = []
            for fold in folds:
                path = (
                    depth_dir
                    / "cells"
                    / fold.identifier
                    / f"h{_HORIZON_MILLISECONDS}"
                    / f"depth_{_DEEP_DEPTH}"
                    / model
                    / f"seed_{seed}"
                    / "training_history.json"
                )
                if not path.is_file():
                    raise ValueError(f"development Level-10 history is missing: {path}")
                summary_path = path.parent / "cell_summary.json"
                if not summary_path.is_file():
                    raise ValueError(
                        f"development Level-10 summary is missing: {summary_path}"
                    )
                summary = _neural._json_mapping(path=summary_path)
                artifact_hashes = summary.get("artifact_sha256")
                if not isinstance(artifact_hashes, dict) or artifact_hashes.get(
                    path.name
                ) != _baseline._sha256_file(path=path):
                    raise ValueError(
                        f"development Level-10 history failed integrity: {path}"
                    )
                history = _neural._json_mapping(path=path)
                histories.append(
                    {
                        "fold": fold.identifier,
                        "best_epoch": _locked._integer(history, "best_epoch"),
                        "history_sha256": _baseline._sha256_file(path=path),
                    }
                )
            ordered = sorted(_locked._integer(item, "best_epoch") for item in histories)
            cells.append(
                {
                    "model": model,
                    "depth": _DEEP_DEPTH,
                    "horizon_milliseconds": _HORIZON_MILLISECONDS,
                    "horizon_steps": 300,
                    "seed": seed,
                    "fixed_epochs": ordered[len(ordered) // 2],
                    "development_best_epochs": histories,
                }
            )
    return cells


def _prepare_corpora(
    *,
    protocol: research_models.ResearchProtocol,
    development_sessions: tuple[research_models.SessionIdentity, ...],
    replication_sessions: tuple[research_models.SessionIdentity, ...],
    development_cache_dir: Path,
    replication_cache_dir: Path,
    working_dir: Path,
    horizon_steps: int,
) -> tuple[pilot_training.PreparedCorpus, pilot_training.PreparedCorpus]:
    scaler = pilot_training.fit_feature_scaler(
        sessions=(
            session
            for _, session in _depth_extension._deep_sessions_with_progress(
                protocol=protocol,
                identities=development_sessions,
                cache_dir=development_cache_dir,
                horizon_steps=horizon_steps,
                stage="replication-fit-scaler",
            )
        )
    )
    context_steps = (
        protocol.context_milliseconds // protocol.state_interval_milliseconds
    )
    training = _depth_extension._prepare_memmap_corpus(
        protocol=protocol,
        identities=development_sessions,
        output_dir=working_dir,
        cache_dir=development_cache_dir,
        scaler=scaler,
        horizon_steps=horizon_steps,
        context_steps=context_steps,
        stride_steps=research_operations.training_stride_steps(
            protocol=protocol, horizon_milliseconds=_HORIZON_MILLISECONDS
        ),
        pack_selected_windows=True,
        name="replication_training",
    )
    evaluation = _depth_extension._prepare_memmap_corpus(
        protocol=protocol,
        identities=replication_sessions,
        output_dir=working_dir,
        cache_dir=replication_cache_dir,
        scaler=scaler,
        horizon_steps=horizon_steps,
        context_steps=context_steps,
        stride_steps=(
            protocol.evaluation_stride_milliseconds
            // protocol.state_interval_milliseconds
        ),
        pack_selected_windows=False,
        name="replication_evaluation",
    )
    return training, evaluation


def _materialize_replication_cache(
    *,
    protocol: research_models.ResearchProtocol,
    sessions: tuple[research_models.SessionIdentity, ...],
    cache_dir: Path,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for position, identity in enumerate(sessions, start=1):
        cache_path = cache_dir / f"{identity.trading_date.isoformat()}.parquet"
        hash_path = cache_path.with_suffix(".sha256")
        if cache_path.is_file() and hash_path.is_file():
            if hash_path.read_text(encoding="utf-8").strip() == _baseline._sha256_file(
                path=cache_path
            ):
                continue
            raise ValueError(f"replication cache failed integrity check: {cache_path}")
        print(
            f"[depth-replication] reconstruct={position}/{len(sessions)} "
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
                maximum_depth=_DEEP_DEPTH,
            )
        temporary = cache_path.with_suffix(".parquet.tmp")
        states.write_parquet(temporary)
        temporary.replace(cache_path)
        _neural._write_text_atomically(
            text=_baseline._sha256_file(path=cache_path), path=hash_path
        )


def _verify_development_histories(
    *, cells: tuple[dict[str, object], ...], depth_dir: Path
) -> None:
    for cell in cells:
        raw_histories = cell.get("development_best_epochs")
        if not isinstance(raw_histories, list) or not raw_histories:
            raise ValueError("frozen cell has no development histories")
        for raw in raw_histories:
            if not isinstance(raw, dict):
                raise ValueError("frozen development history is invalid")
            history = cast(dict[str, object], raw)
            fold = _locked._string(history, "fold")
            path = (
                depth_dir
                / "cells"
                / fold
                / f"h{_locked._integer(cell, 'horizon_milliseconds')}"
                / f"depth_{_locked._integer(cell, 'depth')}"
                / _locked._string(cell, "model")
                / f"seed_{_locked._integer(cell, 'seed')}"
                / "training_history.json"
            )
            if not path.is_file() or history.get(
                "history_sha256"
            ) != _baseline._sha256_file(path=path):
                raise ValueError(f"development history changed after freeze: {path}")
            loaded = _neural._json_mapping(path=path)
            if history.get("best_epoch") != loaded.get("best_epoch"):
                raise ValueError(f"development best epoch changed after freeze: {path}")


def _verify_development_cache(
    *,
    sessions: tuple[research_models.SessionIdentity, ...],
    cache_dir: Path,
    manifest_path: Path,
) -> None:
    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, list):
        raise ValueError("Level-10 cache manifest must be a list")
    rows = {
        str(row.get("trading_date")): cast(dict[str, object], row)
        for row in loaded
        if isinstance(row, dict)
    }
    expected_dates = {session.trading_date.isoformat() for session in sessions}
    if set(rows) != expected_dates or len(loaded) != len(expected_dates):
        raise ValueError("Level-10 cache manifest has unexpected session coverage")
    for session in sessions:
        key = session.trading_date.isoformat()
        row = rows[key]
        cache_path = cache_dir / f"{key}.parquet"
        if (
            row.get("raw_sha256") != session.sha256
            or not cache_path.is_file()
            or row.get("cache_sha256") != _baseline._sha256_file(path=cache_path)
        ):
            raise ValueError(f"Level-10 development cache failed verification: {key}")


def _verified_shallow_metrics(
    *,
    protocol: research_models.ResearchProtocol,
    models: tuple[str, ...],
    replication_sessions: tuple[research_models.SessionIdentity, ...],
    locked_dir: Path,
) -> pl.DataFrame:
    metrics = pl.read_csv(
        locked_dir / "session_metrics.csv",
        schema_overrides={"validation_date": pl.Date},
    ).filter(
        (pl.col("depth") == _SHALLOW_DEPTH)
        & (pl.col("horizon_milliseconds") == _HORIZON_MILLISECONDS)
        & (pl.col("seed") >= 0)
        & pl.col("model").is_in(models)
    )
    expected = {
        (model, seed, session.trading_date)
        for model in set(models)
        for seed in protocol.random_seeds
        for session in replication_sessions
    }
    actual = {
        (str(row["model"]), int(row["seed"]), row["validation_date"])
        for row in metrics.iter_rows(named=True)
    }
    if actual != expected or metrics.height != len(expected):
        raise ValueError("Level-1 March metrics do not have exact matched coverage")
    return metrics


def _training_details(*, output_dir: Path) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted((output_dir / "cells").rglob("cell_summary.json")):
        summary = _neural._json_mapping(path=path)
        rows.append(
            {
                "model": path.parent.parent.name,
                "seed": int(path.parent.name.removeprefix("seed_")),
                "depth": _DEEP_DEPTH,
                "fixed_epochs": summary["fixed_epochs"],
                "fit_elapsed_seconds": summary["fit_elapsed_seconds"],
            }
        )
    return pl.DataFrame(rows).sort("model", "seed")


def _replication_decision(
    *, comparisons: pl.DataFrame, protocol: research_models.ResearchProtocol
) -> dict[str, object]:
    primary = [item.value for item in protocol.primary_metrics]
    support: dict[str, bool] = {}
    for model in comparisons["model"].unique().sort().to_list():
        rows = comparisons.filter(
            (pl.col("model") == model) & pl.col("metric").is_in(primary)
        )
        support[str(model)] = rows.height == len(primary) and bool(
            rows["metric_passed"].all()
        )
    return {
        "decision_rule": (
            "The development-admitted model replicates only when the March "
            "Level-10 minus Level-1 paired-session confidence interval is "
            "favorable for both macro_f1 and mcc."
        ),
        "instrument": protocol.development_instrument.value,
        "horizon_milliseconds": _HORIZON_MILLISECONDS,
        "comparison": "level_10_minus_level_1",
        "primary_metrics": primary,
        "deeper_depth_supported_by_model": support,
        "supported_models": [model for model, passed in support.items() if passed],
        "native_state_interval_milliseconds": protocol.state_interval_milliseconds,
        "minute_aggregation_used": False,
        "development_only": False,
        "retrospective_replication": True,
        "march_outcomes_previously_inspected": True,
        "retuning_permitted": False,
    }


def _verify_plan_inputs(
    *,
    plan: dict[str, object],
    protocol: research_models.ResearchProtocol,
    protocol_path: Path,
    policy_path: Path,
) -> None:
    depth_dir = protocol.output_dir / "depth_extension"
    locked_dir = protocol.output_dir / "locked_evaluation"
    paths = {
        "protocol_sha256": protocol_path,
        "policy_sha256": policy_path,
        "manifest_sha256": protocol.output_dir / "split_manifest.yaml",
        "audit_sha256": protocol.output_dir / "session_audit.csv",
        "depth_run_identity_sha256": depth_dir / "run_identity.json",
        "depth_run_summary_sha256": depth_dir / "run_summary.json",
        "depth_metrics_sha256": depth_dir / "session_metrics.csv",
        "depth_comparisons_sha256": depth_dir / "paired_depth_comparisons.csv",
        "depth_decision_sha256": depth_dir / "decision.json",
        "depth_cache_manifest_sha256": depth_dir / "level_10_cache_manifest.json",
        "level_1_plan_sha256": locked_dir / "plan.json",
        "level_1_run_summary_sha256": locked_dir / "run_summary.json",
        "level_1_metrics_sha256": locked_dir / "session_metrics.csv",
        "level_1_decision_sha256": locked_dir / "decision.json",
    }
    if any(
        not path.is_file() or plan.get(key) != _baseline._sha256_file(path=path)
        for key, path in paths.items()
    ):
        raise ValueError("frozen depth-replication inputs changed")
    if (
        plan.get("march_outcomes_previously_inspected") is not True
        or plan.get("retuning_permitted") is not False
    ):
        raise ValueError("depth-replication plan misstates retrospective status")


def _require_files(*, paths: tuple[Path, ...]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "complete development and Level-1 March artifacts are required: "
            + ", ".join(missing)
        )
