"""Evaluate whether native Level-2--10 book data improves EUR/USD forecasts."""

from __future__ import annotations

import json
import math
import platform
import time
from pathlib import Path
from typing import cast

import attrs
import polars as pl
import sklearn
import torch
import xlsxwriter

from ebs_tft.application.usecases.research_protocol import _baseline, _neural
from ebs_tft.data.repositories import artifact as artifact_repository
from ebs_tft.domain import model as model_domain
from ebs_tft.domain.research import models as research_models
from ebs_tft.domain.research import operations as research_operations

DEPTH_EXTENSION_IMPLEMENTATION_VERSION = 1
_HORIZON_MILLISECONDS = 30_000
_SHALLOW_DEPTH = 1
_DEEP_DEPTH = 10


@attrs.frozen
class DepthExtensionResult:
    """Reference the completed depth-extension evidence and workbook."""

    output_dir: Path
    comparisons_path: Path
    decision_path: Path
    workbook_path: Path


class DepthExtensionPausedError(Exception):
    """Indicate that a requested bounded batch completed safely."""

    def __init__(self, *, completed_cells: int, total_cells: int, output_dir: Path):
        self.completed_cells = completed_cells
        self.total_cells = total_cells
        self.output_dir = output_dir
        super().__init__(
            f"Depth extension paused safely after {completed_cells}/{total_cells} "
            f"Level-10 cells; resume without --replace-output. Outputs: {output_dir}"
        )


def run(
    *,
    protocol: research_models.ResearchProtocol,
    protocol_path: Path,
    policy: research_models.NeuralBenchmarkPolicy,
    policy_path: Path,
    replace_output: bool = False,
    maximum_new_cells: int | None = None,
) -> DepthExtensionResult:
    """Train only missing depth-10 cells and pair them with frozen depth-1 evidence."""
    if maximum_new_cells is not None and (
        isinstance(maximum_new_cells, bool) or maximum_new_cells <= 0
    ):
        raise ValueError("maximum_new_cells must be positive or null")
    started = time.perf_counter()
    _validate_design(protocol=protocol)
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
    original_dir = protocol.output_dir / "neural_benchmark"
    original_metrics_path = original_dir / "session_metrics.csv"
    original_identity_path = original_dir / "run_identity.json"
    original_summary_path = original_dir / "run_summary.json"
    original_gate_path = original_dir / "gate_decision.json"
    _require_files(
        (
            original_metrics_path,
            original_identity_path,
            original_summary_path,
            original_gate_path,
        )
    )
    original_metrics = pl.read_csv(
        original_metrics_path, schema_overrides={"validation_date": pl.Date}
    )
    shallow_metrics = _verified_shallow_metrics(
        metrics=original_metrics,
        folds=folds,
        protocol=protocol,
        policy=policy,
        original_dir=original_dir,
        original_identity=_neural._json_mapping(path=original_identity_path),
        original_summary=_neural._json_mapping(path=original_summary_path),
        original_gate=_neural._json_mapping(path=original_gate_path),
        protocol_path=protocol_path,
        policy_path=policy_path,
    )
    output_dir = protocol.output_dir / "depth_extension"
    artifact_repository.prepare_run_directory(
        path=output_dir,
        replace=replace_output,
        replacement_parent=protocol.output_dir,
    )
    identity = _run_identity(
        protocol=protocol,
        protocol_path=protocol_path,
        policy=policy,
        policy_path=policy_path,
        manifest_path=manifest_path,
        audit_path=audit_path,
        original_metrics_path=original_metrics_path,
        original_identity_path=original_identity_path,
        original_summary_path=original_summary_path,
        original_gate_path=original_gate_path,
    )
    _verify_or_write_identity(output_dir=output_dir, identity=identity)
    device = model_domain.select_device(requested=policy.device)
    horizon_steps = _HORIZON_MILLISECONDS // protocol.state_interval_milliseconds
    total_cells = len(folds) * len(protocol.models) * len(protocol.random_seeds)
    completed_cells = 0
    resumed_cells = 0
    new_cells = 0
    deep_frames: list[pl.DataFrame] = []
    for fold in folds:
        cells = tuple(
            _neural._cell(
                protocol=protocol,
                policy=policy,
                fold=fold,
                horizon_steps=horizon_steps,
                depth=_DEEP_DEPTH,
                model_name=model_name,
                seed=seed,
                output_dir=output_dir,
                identity=identity,
            )
            for model_name in protocol.models
            for seed in protocol.random_seeds
        )
        completed = {cell: _neural._completed_cell_metrics(cell=cell) for cell in cells}
        for metrics in completed.values():
            if metrics is not None:
                deep_frames.append(metrics)
                completed_cells += 1
                resumed_cells += 1
        pending = tuple(cell for cell, metrics in completed.items() if metrics is None)
        if not pending:
            continue
        preparation_started = time.perf_counter()
        training_corpus, validation_corpus = _neural._prepare_corpora(
            protocol=protocol,
            fold=fold,
            depth=_DEEP_DEPTH,
            horizon_steps=horizon_steps,
        )
        print(
            f"[depth-extension] corpus={fold.identifier}:h{_HORIZON_MILLISECONDS}:"
            f"d{_DEEP_DEPTH} training_windows={len(training_corpus.target_indices)} "
            f"validation_windows={len(validation_corpus.target_indices)} "
            f"preparation_seconds={time.perf_counter() - preparation_started:.2f}",
            flush=True,
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
        validation_dataset = model_domain.SequenceDataset(
            lob_features=validation_corpus.lob_features,
            auxiliary_features=validation_corpus.auxiliary_features,
            labels=validation_corpus.labels,
            target_indices=validation_corpus.target_indices,
            context_steps=context_steps,
        )
        for cell in pending:
            completed_cells += 1
            print(
                f"[depth-extension] cell={completed_cells}/{total_cells} "
                f"fold={fold.identifier} horizon_ms={_HORIZON_MILLISECONDS} "
                f"depth={_DEEP_DEPTH} model={cell.model_name} seed={cell.seed}",
                flush=True,
            )
            deep_frames.append(
                _neural._fit_cell(
                    cell=cell,
                    protocol=protocol,
                    policy=policy,
                    device=device,
                    training_dataset=training_dataset,
                    validation_dataset=validation_dataset,
                    validation_corpus=validation_corpus,
                )
            )
            new_cells += 1
            print(
                f"[depth-extension] cell={completed_cells}/{total_cells} "
                "checkpoint=saved",
                flush=True,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if (
                maximum_new_cells is not None
                and new_cells >= maximum_new_cells
                and completed_cells < total_cells
            ):
                _write_progress(
                    output_dir=output_dir,
                    completed_cells=completed_cells,
                    total_cells=total_cells,
                )
                raise DepthExtensionPausedError(
                    completed_cells=completed_cells,
                    total_cells=total_cells,
                    output_dir=output_dir,
                )
        del training_dataset, validation_dataset, training_corpus, validation_corpus

    deep_metrics = pl.concat(deep_frames).sort(
        ["fold", "validation_date", "model", "seed"]
    )
    _validate_deep_metrics(metrics=deep_metrics, folds=folds, protocol=protocol)
    combined_metrics = pl.concat([shallow_metrics, deep_metrics]).sort(
        ["depth", "fold", "validation_date", "model", "seed"]
    )
    metrics_path = output_dir / "session_metrics.csv"
    _neural._write_csv_atomically(data=combined_metrics, path=metrics_path)
    comparisons = _paired_depth_comparisons(metrics=combined_metrics, protocol=protocol)
    comparisons_path = output_dir / "paired_depth_comparisons.csv"
    _neural._write_csv_atomically(data=comparisons, path=comparisons_path)
    deltas = _session_depth_deltas(metrics=combined_metrics, protocol=protocol)
    deltas_path = output_dir / "session_depth_deltas.csv"
    _neural._write_csv_atomically(data=deltas, path=deltas_path)
    absolute = _absolute_metric_summary(metrics=combined_metrics, protocol=protocol)
    absolute_path = output_dir / "absolute_metric_summary.csv"
    _neural._write_csv_atomically(data=absolute, path=absolute_path)
    training = _training_details(output_dir=output_dir)
    training_path = output_dir / "training_details.csv"
    _neural._write_csv_atomically(data=training, path=training_path)
    decision = _decision(comparisons=comparisons, protocol=protocol)
    decision_path = output_dir / "decision.json"
    _neural._write_text_atomically(
        text=json.dumps(decision, indent=2), path=decision_path
    )
    workbook_path = output_dir / "eurusd_level1_vs_level10.xlsx"
    _write_workbook(
        path=workbook_path,
        protocol=protocol,
        decision=decision,
        comparisons=comparisons,
        absolute=absolute,
        deltas=deltas,
        training=training,
        metrics=combined_metrics,
    )
    elapsed = time.perf_counter() - started
    summary = {
        **identity,
        "status": "complete",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "sklearn": sklearn.__version__,
            "device": str(device),
        },
        "folds": len(folds),
        "validation_sessions": shallow_metrics.select("fold", "validation_date")
        .unique()
        .height,
        "new_level_10_cells": total_cells,
        "resumed_cells": resumed_cells,
        "elapsed_seconds": elapsed,
        "artifacts": {
            "session_metrics": str(metrics_path),
            "paired_depth_comparisons": str(comparisons_path),
            "session_depth_deltas": str(deltas_path),
            "absolute_metric_summary": str(absolute_path),
            "training_details": str(training_path),
            "decision": str(decision_path),
            "workbook": str(workbook_path),
        },
    }
    _neural._write_text_atomically(
        text=json.dumps(summary, indent=2), path=output_dir / "run_summary.json"
    )
    terminal = "\n".join(
        (
            "EBS EUR/USD native-resolution depth extension completed",
            "WARNING: secondary development evidence; locked outcomes were "
            "already inspected.",
            f"instrument={protocol.development_instrument.value}",
            f"state_interval_milliseconds={protocol.state_interval_milliseconds}",
            f"horizon_milliseconds={_HORIZON_MILLISECONDS}",
            f"comparison=level_{_DEEP_DEPTH}_minus_level_{_SHALLOW_DEPTH}",
            f"new_level_10_cells={total_cells}",
            f"resumed_cells={resumed_cells}",
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
    (output_dir / "progress_summary.json").unlink(missing_ok=True)
    print(terminal)
    return DepthExtensionResult(
        output_dir=output_dir,
        comparisons_path=comparisons_path,
        decision_path=decision_path,
        workbook_path=workbook_path,
    )


def _validate_design(*, protocol: research_models.ResearchProtocol) -> None:
    if protocol.development_instrument.value != "EUR_USD":
        raise ValueError(
            "depth extension is frozen to the EUR_USD development instrument"
        )
    if {_SHALLOW_DEPTH, _DEEP_DEPTH} - set(protocol.depths):
        raise ValueError("depth extension requires declared depths 1 and 10")
    if _HORIZON_MILLISECONDS not in protocol.forecast_horizons_milliseconds:
        raise ValueError("depth extension requires the declared 30000-ms horizon")


def _require_files(paths: tuple[Path, ...]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "complete Level-1 neural benchmark artifacts are required: "
            + ", ".join(missing)
        )


def _verified_shallow_metrics(
    *,
    metrics: pl.DataFrame,
    folds: tuple[research_models.RollingFold, ...],
    protocol: research_models.ResearchProtocol,
    policy: research_models.NeuralBenchmarkPolicy,
    original_dir: Path,
    original_identity: dict[str, object],
    original_summary: dict[str, object],
    original_gate: dict[str, object],
    protocol_path: Path,
    policy_path: Path,
) -> pl.DataFrame:
    if original_identity.get("protocol_sha256") != _baseline._sha256_file(
        path=protocol_path
    ) or original_identity.get("policy_sha256") != _baseline._sha256_file(
        path=policy_path
    ):
        raise ValueError("Level-1 evidence belongs to another protocol or policy")
    if original_summary.get("cells") != 64:
        raise ValueError("Level-1 benchmark is not the completed 64-cell study")
    accepted = original_gate.get("accepted_model_depth_horizons")
    if not isinstance(accepted, list) or not all(
        {
            "model": model,
            "depth": _SHALLOW_DEPTH,
            "horizon_milliseconds": _HORIZON_MILLISECONDS,
        }
        in accepted
        for model in protocol.models
    ):
        raise ValueError("both 30-second Level-1 models must be admitted")
    selected = metrics.filter(
        (pl.col("depth") == _SHALLOW_DEPTH)
        & (pl.col("horizon_milliseconds") == _HORIZON_MILLISECONDS)
        & pl.col("model").is_in(protocol.models)
        & pl.col("seed").is_in(protocol.random_seeds)
    )
    _validate_cell_metrics(
        metrics=selected, folds=folds, protocol=protocol, depth=_SHALLOW_DEPTH
    )
    horizon_steps = _HORIZON_MILLISECONDS // protocol.state_interval_milliseconds
    cell_frames: list[pl.DataFrame] = []
    for fold in folds:
        for model in protocol.models:
            for seed in protocol.random_seeds:
                cell = _neural._cell(
                    protocol=protocol,
                    policy=policy,
                    fold=fold,
                    horizon_steps=horizon_steps,
                    depth=_SHALLOW_DEPTH,
                    model_name=model,
                    seed=seed,
                    output_dir=original_dir,
                    identity=original_identity,
                )
                cell_metrics = _neural._completed_cell_metrics(cell=cell)
                if cell_metrics is None:
                    raise ValueError(
                        f"required Level-1 cell is incomplete: {cell.output_dir}"
                    )
                cell_frames.append(cell_metrics)
    sort_by = ["fold", "validation_date", "model", "seed", "depth"]
    verified_cells = pl.concat(cell_frames).sort(sort_by)
    selected = selected.sort(sort_by)
    if not verified_cells.equals(selected, null_equal=True):
        raise ValueError("combined Level-1 metrics do not match hashed cell artifacts")
    return selected


def _validate_deep_metrics(
    *,
    metrics: pl.DataFrame,
    folds: tuple[research_models.RollingFold, ...],
    protocol: research_models.ResearchProtocol,
) -> None:
    _validate_cell_metrics(
        metrics=metrics, folds=folds, protocol=protocol, depth=_DEEP_DEPTH
    )


def _validate_cell_metrics(
    *,
    metrics: pl.DataFrame,
    folds: tuple[research_models.RollingFold, ...],
    protocol: research_models.ResearchProtocol,
    depth: int,
) -> None:
    actual = {
        (str(row["fold"]), row["validation_date"], str(row["model"]), int(row["seed"]))
        for row in metrics.iter_rows(named=True)
    }
    expected = {
        (fold.identifier, session.trading_date, model, seed)
        for fold in folds
        for session in fold.validation_sessions
        for model in protocol.models
        for seed in protocol.random_seeds
    }
    if metrics.height != len(expected) or actual != expected:
        raise ValueError(
            f"Level-{depth} metrics do not cover exact folds/dates/models/seeds"
        )


def _run_identity(
    *,
    protocol: research_models.ResearchProtocol,
    protocol_path: Path,
    policy: research_models.NeuralBenchmarkPolicy,
    policy_path: Path,
    manifest_path: Path,
    audit_path: Path,
    original_metrics_path: Path,
    original_identity_path: Path,
    original_summary_path: Path,
    original_gate_path: Path,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "implementation_version": DEPTH_EXTENSION_IMPLEMENTATION_VERSION,
        "model_protocol_version": model_domain.MODEL_PROTOCOL_VERSION,
        "protocol_sha256": _baseline._sha256_file(path=protocol_path),
        "policy_sha256": _baseline._sha256_file(path=policy_path),
        "manifest_sha256": _baseline._sha256_file(path=manifest_path),
        "audit_sha256": _baseline._sha256_file(path=audit_path),
        "level_1_metrics_sha256": _baseline._sha256_file(path=original_metrics_path),
        "level_1_identity_sha256": _baseline._sha256_file(path=original_identity_path),
        "level_1_summary_sha256": _baseline._sha256_file(path=original_summary_path),
        "level_1_gate_sha256": _baseline._sha256_file(path=original_gate_path),
        "experiment": {
            "instrument": protocol.development_instrument.value,
            "horizon_milliseconds": _HORIZON_MILLISECONDS,
            "shallow_depth": _SHALLOW_DEPTH,
            "deep_depth": _DEEP_DEPTH,
            "models": list(protocol.models),
            "seeds": list(protocol.random_seeds),
            "comparison": "paired_seed_mean_level_10_minus_level_1",
            "native_resolution": True,
            "development_only": True,
        },
        "policy": attrs.asdict(policy),
        "torch_version": torch.__version__,
    }


def _verify_or_write_identity(*, output_dir: Path, identity: dict[str, object]) -> None:
    identity_path = output_dir / "run_identity.json"
    if identity_path.is_file():
        if _neural._json_mapping(path=identity_path) != identity:
            raise ValueError("depth-extension inputs changed; use --replace-output")
        return
    cells_dir = output_dir / "cells"
    if cells_dir.exists() and any(cells_dir.rglob("cell_summary.json")):
        raise ValueError("depth-extension cells have no identity; use --replace-output")
    _neural._write_text_atomically(
        text=json.dumps(identity, indent=2), path=identity_path
    )


def _seed_averaged(
    *, metrics: pl.DataFrame, protocol: research_models.ResearchProtocol
) -> pl.DataFrame:
    names = tuple(
        metric.value
        for metric in (*protocol.primary_metrics, *protocol.supporting_metrics)
    )
    return metrics.group_by(
        "instrument",
        "fold",
        "validation_date",
        "model",
        "depth",
        "horizon_steps",
        "horizon_milliseconds",
    ).agg([pl.col(name).mean().alias(name) for name in names])


def _paired_depth_comparisons(
    *, metrics: pl.DataFrame, protocol: research_models.ResearchProtocol
) -> pl.DataFrame:
    averaged = _seed_averaged(metrics=metrics, protocol=protocol)
    names = tuple(
        metric.value
        for metric in (*protocol.primary_metrics, *protocol.supporting_metrics)
    )
    rows: list[dict[str, object]] = []
    for model_position, model in enumerate(protocol.models):
        shallow = averaged.filter(
            (pl.col("model") == model) & (pl.col("depth") == _SHALLOW_DEPTH)
        )
        deep = averaged.filter(
            (pl.col("model") == model) & (pl.col("depth") == _DEEP_DEPTH)
        )
        for metric_position, name in enumerate(names):
            interval = research_operations.paired_session_interval(
                shallower_by_session=_neural._metric_by_session(
                    data=shallow, metric_name=name
                ),
                deeper_by_session=_neural._metric_by_session(
                    data=deep, metric_name=name
                ),
                repetitions=protocol.bootstrap_repetitions,
                confidence_level=protocol.confidence_level,
                random_seed=30_010 + model_position * 100 + metric_position,
            )
            direction = _baseline._favorable_direction(metric_name=name)
            passed = (
                float(interval["confidence_lower"]) > 0
                if direction == "positive"
                else float(interval["confidence_upper"]) < 0
            )
            rows.append(
                {
                    "comparison": "seed_mean_level_10_minus_level_1",
                    "instrument": protocol.development_instrument.value,
                    "model": model,
                    "shallow_depth": _SHALLOW_DEPTH,
                    "deep_depth": _DEEP_DEPTH,
                    "horizon_milliseconds": _HORIZON_MILLISECONDS,
                    "metric": name,
                    "favorable_direction": direction,
                    **interval,
                    "metric_passed": passed,
                }
            )
    return pl.DataFrame(rows).sort("model", "metric")


def _session_depth_deltas(
    *, metrics: pl.DataFrame, protocol: research_models.ResearchProtocol
) -> pl.DataFrame:
    names = tuple(
        metric.value
        for metric in (*protocol.primary_metrics, *protocol.supporting_metrics)
    )
    averaged = _seed_averaged(metrics=metrics, protocol=protocol)
    keys = ["instrument", "fold", "validation_date", "model", "horizon_milliseconds"]
    shallow = averaged.filter(pl.col("depth") == _SHALLOW_DEPTH).drop(
        "depth", "horizon_steps"
    )
    deep = averaged.filter(pl.col("depth") == _DEEP_DEPTH).drop(
        "depth", "horizon_steps"
    )
    joined = deep.join(shallow, on=keys, suffix="_level_1", validate="1:1")
    return joined.select(
        *keys,
        *[
            (pl.col(name) - pl.col(f"{name}_level_1")).alias(f"{name}_delta")
            for name in names
        ],
    ).sort("model", "fold", "validation_date")


def _absolute_metric_summary(
    *, metrics: pl.DataFrame, protocol: research_models.ResearchProtocol
) -> pl.DataFrame:
    averaged = _seed_averaged(metrics=metrics, protocol=protocol)
    names = tuple(
        metric.value
        for metric in (*protocol.primary_metrics, *protocol.supporting_metrics)
    )
    frames = []
    for name in names:
        frames.append(
            averaged.group_by("instrument", "model", "depth", "horizon_milliseconds")
            .agg(
                pl.len().alias("sessions"),
                pl.col(name).mean().alias("mean"),
                pl.col(name).std().alias("standard_deviation"),
                pl.col(name).min().alias("minimum"),
                pl.col(name).max().alias("maximum"),
            )
            .with_columns(pl.lit(name).alias("metric"))
        )
    return (
        pl.concat(frames)
        .select(
            "instrument",
            "model",
            "depth",
            "horizon_milliseconds",
            "metric",
            "sessions",
            "mean",
            "standard_deviation",
            "minimum",
            "maximum",
        )
        .sort("model", "depth", "metric")
    )


def _training_details(*, output_dir: Path) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for summary_path in sorted((output_dir / "cells").rglob("cell_summary.json")):
        summary = _neural._json_mapping(path=summary_path)
        history = _neural._json_mapping(
            path=summary_path.parent / "training_history.json"
        )
        rows.append(
            {
                "fold": summary["fold"],
                "model": summary["model"],
                "seed": summary["seed"],
                "depth": summary["depth"],
                "horizon_steps": summary["horizon_steps"],
                "parameter_count": summary["parameter_count"],
                "training_windows": summary["training_windows"],
                "validation_windows": summary["validation_windows"],
                "best_epoch": history["best_epoch"],
                "best_validation_log_loss": history["best_validation_log_loss"],
                "epochs_completed": history["epochs_completed"],
                "stop_reason": history["stop_reason"],
                "fit_elapsed_seconds": summary["fit_elapsed_seconds"],
                "validation_elapsed_seconds": summary["validation_elapsed_seconds"],
                "final_evaluation_elapsed_seconds": summary[
                    "final_evaluation_elapsed_seconds"
                ],
                "peak_cuda_memory_gib": summary["peak_cuda_memory_gib"],
            }
        )
    return pl.DataFrame(rows).sort("fold", "model", "seed")


def _decision(
    *, comparisons: pl.DataFrame, protocol: research_models.ResearchProtocol
) -> dict[str, object]:
    primary = [item.value for item in protocol.primary_metrics]
    support: dict[str, bool] = {}
    for model in protocol.models:
        rows = comparisons.filter(
            (pl.col("model") == model) & pl.col("metric").is_in(primary)
        )
        support[model] = rows.height == len(primary) and bool(
            rows["metric_passed"].all()
        )
    return {
        "decision_rule": (
            "Deeper depth is supported only when the Level-10 minus Level-1 "
            "paired-session "
            "bootstrap confidence interval is favorable for both macro_f1 and mcc."
        ),
        "instrument": protocol.development_instrument.value,
        "horizon_milliseconds": _HORIZON_MILLISECONDS,
        "comparison": "level_10_minus_level_1",
        "primary_metrics": primary,
        "deeper_depth_supported_by_model": support,
        "supported_models": [model for model, passed in support.items() if passed],
        "native_state_interval_milliseconds": protocol.state_interval_milliseconds,
        "training_target_stride_milliseconds": dict(
            protocol.training_stride_milliseconds
        )[_HORIZON_MILLISECONDS],
        "evaluation_target_stride_milliseconds": (
            protocol.evaluation_stride_milliseconds
        ),
        "minute_aggregation_used": False,
        "development_only": True,
        "locked_evaluation_used": False,
        "retuning_permitted": False,
    }


def _write_progress(
    *, output_dir: Path, completed_cells: int, total_cells: int
) -> None:
    _neural._write_text_atomically(
        text=json.dumps(
            {
                "status": "paused",
                "completed_level_10_cells": completed_cells,
                "total_level_10_cells": total_cells,
                "resume_with_replace_output": False,
            },
            indent=2,
        ),
        path=output_dir / "progress_summary.json",
    )


def _write_workbook(
    *,
    path: Path,
    protocol: research_models.ResearchProtocol,
    decision: dict[str, object],
    comparisons: pl.DataFrame,
    absolute: pl.DataFrame,
    deltas: pl.DataFrame,
    training: pl.DataFrame,
    metrics: pl.DataFrame,
) -> None:
    supported_models = cast(list[str], decision["supported_models"])
    support_by_model = cast(
        dict[str, bool], decision["deeper_depth_supported_by_model"]
    )
    workbook = xlsxwriter.Workbook(path)
    workbook.set_properties(
        {
            "title": "EUR/USD native-resolution Level-1 versus Level-10 evidence",
            "author": "EBS TFT research workflow",
            "comments": (
                "Secondary development analysis; no post-locked tuning permitted."
            ),
        }
    )
    title = workbook.add_format(
        {"bold": True, "font_size": 18, "font_color": "#17365D"}
    )
    section = workbook.add_format(
        {"bold": True, "font_color": "#FFFFFF", "bg_color": "#1F4E78"}
    )
    header = workbook.add_format(
        {
            "bold": True,
            "font_color": "#FFFFFF",
            "bg_color": "#4472C4",
            "border": 1,
            "text_wrap": True,
        }
    )
    text = workbook.add_format({"text_wrap": True, "valign": "top"})
    passed = workbook.add_format(
        {"bold": True, "bg_color": "#C6EFCE", "font_color": "#006100"}
    )
    failed = workbook.add_format(
        {"bold": True, "bg_color": "#FFC7CE", "font_color": "#9C0006"}
    )
    warning = workbook.add_format(
        {"text_wrap": True, "bg_color": "#FFF2CC", "font_color": "#7F6000"}
    )
    sheet = workbook.add_worksheet("Executive Summary")
    sheet.hide_gridlines(2)
    sheet.set_column("A:A", 29)
    sheet.set_column("B:B", 100)
    sheet.merge_range("A1:B1", "EUR/USD — Level 1 vs Level 10", title)
    sheet.write_row("A3", ["Question", "Answer"], header)
    rows = [
        (
            "Forecasting method",
            "DeepLOB and TFT three-class direction classifiers; logistic is not "
            "the depth comparator here.",
        ),
        (
            "Raw-data treatment",
            "Native 100-ms causal states; no 1-minute or other bar aggregation. "
            "For computational control, training targets are sampled every 30 "
            "seconds; validation forecasts are evaluated every 100 ms.",
        ),
        (
            "Data in both variants",
            "Transactions plus order-book state; Level 1 uses top of book and "
            "Level 10 adds levels 2–10.",
        ),
        (
            "Currency and horizon",
            f"EUR/USD only; {_HORIZON_MILLISECONDS // 1000}-second forecast horizon.",
        ),
        (
            "Design",
            "Same 4 rolling folds, 20 validation sessions, 2 models, and 2 random "
            "seeds for both depths.",
        ),
        (
            "New training",
            "16 Level-10 cells. The existing 64-cell benchmark is reused and is "
            "not rerun.",
        ),
        (
            "Decision",
            ", ".join(supported_models)
            or "Neither model showed robust improvement from deeper levels.",
        ),
        (
            "Important limitation",
            "Secondary development evidence. Locked outcomes were already "
            "inspected, so this is not a new pristine final test.",
        ),
        (
            "Economic interpretation",
            "A PASS means the deeper book improved both Macro F1 and MCC with a "
            "strictly favorable session-bootstrap confidence bound. It does not "
            "prove profitability.",
        ),
    ]
    for row, (label, value) in enumerate(rows, start=3):
        sheet.write(row, 0, label, section if row in (3, 9) else text)
        sheet.write(row, 1, value, warning if label == "Important limitation" else text)
    start = len(rows) + 5
    sheet.write_row(start, 0, ["Model", "Depth improvement decision"], header)
    for offset, model in enumerate(protocol.models, start=1):
        result = "PASS" if support_by_model[model] else "FAIL"
        sheet.write(start + offset, 0, model)
        sheet.write(start + offset, 1, result, passed if result == "PASS" else failed)
    _write_frame(
        workbook=workbook, name="Paired Comparisons", data=comparisons, header=header
    )
    _write_frame(
        workbook=workbook, name="Absolute Metrics", data=absolute, header=header
    )
    _write_frame(workbook=workbook, name="Session Deltas", data=deltas, header=header)
    stability = (
        deltas.group_by("model")
        .agg(
            pl.len().alias("sessions"),
            (pl.col("macro_f1_delta") > 0).sum().alias("macro_f1_positive_sessions"),
            (pl.col("mcc_delta") > 0).sum().alias("mcc_positive_sessions"),
            ((pl.col("macro_f1_delta") > 0) & (pl.col("mcc_delta") > 0))
            .sum()
            .alias("both_positive_sessions"),
        )
        .sort("model")
    )
    _write_frame(
        workbook=workbook, name="Session Stability", data=stability, header=header
    )
    _write_frame(
        workbook=workbook, name="Training Details", data=training, header=header
    )
    _write_frame(
        workbook=workbook, name="Raw Session Metrics", data=metrics, header=header
    )
    notes = workbook.add_worksheet("Statistical Notes")
    notes.set_column("A:A", 120)
    note_rows = [
        "Experimental unit: trading session, not overlapping 100-ms observation "
        "window.",
        f"Confidence intervals: {protocol.bootstrap_repetitions:,}-replicate "
        "paired session bootstrap at "
        f"{protocol.confidence_level:.0%} confidence.",
        "Seeds are averaged within each fold/session/model/depth before paired "
        "inference.",
        "Higher is better for Macro F1, MCC, and balanced accuracy; lower is "
        "better for log loss and multiclass Brier score.",
        "Primary decision requires favorable confidence bounds for both Macro F1 "
        "and MCC.",
        "This test isolates added book depth: dates, folds, horizon, preprocessing, "
        "models, seeds, and evaluation cadence are matched.",
    ]
    for row, value in enumerate(note_rows):
        notes.write(row, 0, value, text)
    workbook.close()


def _write_frame(
    *,
    workbook: xlsxwriter.Workbook,
    name: str,
    data: pl.DataFrame,
    header: xlsxwriter.format.Format,
) -> None:
    sheet = workbook.add_worksheet(name)
    sheet.freeze_panes(1, 0)
    sheet.autofilter(0, 0, data.height, max(0, data.width - 1))
    for column, name_value in enumerate(data.columns):
        sheet.write(0, column, name_value, header)
        values = data[name_value].to_list()
        width = min(42, max(len(name_value) + 2, *(len(str(v)) + 2 for v in values)))
        sheet.set_column(column, column, width)
    for row_index, row in enumerate(data.iter_rows(), start=1):
        for column, value in enumerate(row):
            if value is None:
                sheet.write_blank(row_index, column, None)
            elif isinstance(value, bool):
                sheet.write_boolean(row_index, column, value)
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                if isinstance(value, float) and not math.isfinite(value):
                    sheet.write(row_index, column, str(value))
                else:
                    sheet.write_number(row_index, column, value)
            else:
                sheet.write(row_index, column, str(value))
