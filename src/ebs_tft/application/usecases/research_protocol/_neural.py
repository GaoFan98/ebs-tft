"""Run the finite neural benchmark admitted by the defensive-baseline gate."""

from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path
from typing import cast

import attrs
import numpy as np
import polars as pl
import sklearn
import torch

from ebs_tft.application.usecases.pilot import _checkpoint
from ebs_tft.application.usecases.research_protocol import _baseline
from ebs_tft.data.repositories import artifact as artifact_repository
from ebs_tft.domain import model as model_domain
from ebs_tft.domain.orderbook import models as orderbook_models
from ebs_tft.domain.pilot import training as pilot_training
from ebs_tft.domain.research import models as research_models
from ebs_tft.domain.research import operations as research_operations

NEURAL_BENCHMARK_IMPLEMENTATION_VERSION = 2


@attrs.frozen
class NeuralBenchmarkResult:
    """Reference durable artifacts from the gated rolling neural benchmark."""

    output_dir: Path
    metrics_path: Path
    comparisons_path: Path
    gate_path: Path
    terminal_summary_path: Path


class NeuralBenchmarkPausedError(Exception):
    """Indicate that the requested number of new cells completed safely."""

    def __init__(self, *, completed_cells: int, total_cells: int, output_dir: Path):
        self.completed_cells = completed_cells
        self.total_cells = total_cells
        self.output_dir = output_dir
        super().__init__(
            f"Neural benchmark paused safely after {completed_cells}/{total_cells} "
            f"cells; resume without --replace-output. Outputs: {output_dir}"
        )


@attrs.frozen
class _Cell:
    fold: research_models.RollingFold
    horizon_steps: int
    depth: int
    model_name: str
    seed: int
    fingerprint: str
    output_dir: Path


def run(
    *,
    protocol: research_models.ResearchProtocol,
    protocol_path: Path,
    policy: research_models.NeuralBenchmarkPolicy,
    policy_path: Path,
    replace_output: bool = False,
    maximum_new_cells: int | None = None,
) -> NeuralBenchmarkResult:
    """Run only neural cells mechanically admitted by completed gate evidence."""
    if maximum_new_cells is not None and (
        isinstance(maximum_new_cells, bool) or maximum_new_cells <= 0
    ):
        raise ValueError("maximum_new_cells must be positive or null")
    started = time.perf_counter()
    manifest_path = protocol.output_dir / "split_manifest.yaml"
    audit_path = protocol.output_dir / "session_audit.csv"
    folds = _baseline._load_folds(
        manifest_path=manifest_path,
        audit_path=audit_path,
        protocol_path=protocol_path,
        protocol=protocol,
    )
    admitted_dimensions, gate_hash = _admitted_dimensions(
        protocol=protocol, protocol_path=protocol_path
    )
    device = model_domain.select_device(requested=policy.device)
    _baseline._verify_cached_states(
        folds=folds, audit_path=audit_path, protocol=protocol
    )
    output_dir = protocol.output_dir / "neural_benchmark"
    artifact_repository.prepare_run_directory(
        path=output_dir,
        replace=replace_output,
        replacement_parent=protocol.output_dir,
    )
    identity = _run_identity(
        protocol_path=protocol_path,
        policy=policy,
        policy_path=policy_path,
        manifest_path=manifest_path,
        audit_path=audit_path,
        gate_hash=gate_hash,
        model_report_path=(
            protocol.output_dir / "model_protocol" / "model_protocol_verification.json"
        ),
        baseline_metrics_path=(
            protocol.output_dir / "baseline_gate" / "session_metrics.csv"
        ),
    )
    _verify_or_write_identity(output_dir=output_dir, identity=identity)
    all_metrics: list[pl.DataFrame] = []
    total_cells = (
        len(folds)
        * sum(len(depths) for _, depths in admitted_dimensions)
        * len(protocol.models)
        * len(protocol.random_seeds)
    )
    completed_cells = 0
    resumed_cells = 0
    new_cells = 0
    for fold in folds:
        for horizon_steps, depths in admitted_dimensions:
            for depth in depths:
                cells = tuple(
                    _cell(
                        protocol=protocol,
                        policy=policy,
                        fold=fold,
                        horizon_steps=horizon_steps,
                        depth=depth,
                        model_name=model_name,
                        seed=seed,
                        output_dir=output_dir,
                        identity=identity,
                    )
                    for model_name in protocol.models
                    for seed in protocol.random_seeds
                )
                completed = {cell: _completed_cell_metrics(cell=cell) for cell in cells}
                for metrics in completed.values():
                    if metrics is not None:
                        all_metrics.append(metrics)
                        completed_cells += 1
                        resumed_cells += 1
                pending = tuple(
                    cell for cell, metrics in completed.items() if metrics is None
                )
                if not pending:
                    continue
                preparation_started = time.perf_counter()
                training_corpus, validation_corpus = _prepare_corpora(
                    protocol=protocol,
                    fold=fold,
                    depth=depth,
                    horizon_steps=horizon_steps,
                )
                preparation_elapsed = time.perf_counter() - preparation_started
                print(
                    "[neural-benchmark] corpus="
                    f"{fold.identifier}:h"
                    f"{horizon_steps * protocol.state_interval_milliseconds}:d{depth} "
                    f"training_windows={len(training_corpus.target_indices)} "
                    f"validation_windows={len(validation_corpus.target_indices)} "
                    f"preparation_seconds={preparation_elapsed:.2f}",
                    flush=True,
                )
                training_dataset = model_domain.SequenceDataset(
                    lob_features=training_corpus.lob_features,
                    auxiliary_features=training_corpus.auxiliary_features,
                    labels=training_corpus.labels,
                    target_indices=training_corpus.target_indices,
                    context_steps=(
                        protocol.context_milliseconds
                        // protocol.state_interval_milliseconds
                    ),
                )
                validation_dataset = model_domain.SequenceDataset(
                    lob_features=validation_corpus.lob_features,
                    auxiliary_features=validation_corpus.auxiliary_features,
                    labels=validation_corpus.labels,
                    target_indices=validation_corpus.target_indices,
                    context_steps=(
                        protocol.context_milliseconds
                        // protocol.state_interval_milliseconds
                    ),
                )
                for cell in pending:
                    completed_cells += 1
                    print(
                        f"[neural-benchmark] cell={completed_cells}/{total_cells} "
                        f"fold={cell.fold.identifier} "
                        "horizon_ms="
                        f"{cell.horizon_steps * protocol.state_interval_milliseconds} "
                        f"depth={cell.depth} model={cell.model_name} seed={cell.seed}",
                        flush=True,
                    )
                    metrics = _fit_cell(
                        cell=cell,
                        protocol=protocol,
                        policy=policy,
                        device=device,
                        training_dataset=training_dataset,
                        validation_dataset=validation_dataset,
                        validation_corpus=validation_corpus,
                    )
                    all_metrics.append(metrics)
                    new_cells += 1
                    print(
                        f"[neural-benchmark] cell={completed_cells}/{total_cells} "
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
                        _write_text_atomically(
                            text=json.dumps(
                                {
                                    "status": "paused",
                                    "completed_cells": completed_cells,
                                    "total_cells": total_cells,
                                    "resume_with_replace_output": False,
                                },
                                indent=2,
                            ),
                            path=output_dir / "progress_summary.json",
                        )
                        raise NeuralBenchmarkPausedError(
                            completed_cells=completed_cells,
                            total_cells=total_cells,
                            output_dir=output_dir,
                        )
                del training_dataset, validation_dataset
                del training_corpus, validation_corpus

    metrics = pl.concat(all_metrics).sort(
        ["fold", "validation_date", "horizon_steps", "model", "seed", "depth"]
    )
    metrics_path = output_dir / "session_metrics.csv"
    _write_csv_atomically(data=metrics, path=metrics_path)
    baseline_metrics = pl.read_csv(
        protocol.output_dir / "baseline_gate" / "session_metrics.csv",
        schema_overrides={"validation_date": pl.Date},
    )
    comparisons = _paired_comparisons(
        neural_metrics=metrics,
        baseline_metrics=baseline_metrics,
        protocol=protocol,
    )
    comparisons_path = output_dir / "paired_baseline_comparisons.csv"
    _write_csv_atomically(data=comparisons, path=comparisons_path)
    gate = _gate_decision(comparisons=comparisons, protocol=protocol)
    accepted_model_depth_horizons = cast(
        list[dict[str, object]], gate["accepted_model_depth_horizons"]
    )
    gate_path = output_dir / "gate_decision.json"
    _write_text_atomically(text=json.dumps(gate, indent=2), path=gate_path)
    elapsed = time.perf_counter() - started
    terminal_summary = "\n".join(
        (
            "EBS gated rolling neural benchmark completed",
            "WARNING: development evidence only; locked evaluation was not used.",
            f"instrument={protocol.development_instrument.value}",
            f"device={device}",
            f"folds={len(folds)}",
            "horizons="
            + ",".join(
                str(item[0] * protocol.state_interval_milliseconds)
                for item in admitted_dimensions
            ),
            "depths="
            + ",".join(
                str(depth)
                for depth in sorted(
                    {depth for _, depths in admitted_dimensions for depth in depths}
                )
            ),
            f"cells={total_cells}",
            f"resumed_cells={resumed_cells}",
            f"accepted_model_depth_horizons={len(accepted_model_depth_horizons)}",
            f"elapsed_seconds={elapsed:.2f}",
            f"outputs={output_dir}",
        )
    )
    terminal_summary_path = output_dir / "terminal_summary.txt"
    _write_text_atomically(text=terminal_summary, path=terminal_summary_path)
    (output_dir / "progress_summary.json").unlink(missing_ok=True)
    _write_text_atomically(
        text=json.dumps(
            {
                **identity,
                "environment": {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "torch": torch.__version__,
                    "sklearn": sklearn.__version__,
                    "device": str(device),
                },
                "folds": len(folds),
                "cells": total_cells,
                "elapsed_seconds": elapsed,
                "artifacts": {
                    "session_metrics": str(metrics_path),
                    "paired_baseline_comparisons": str(comparisons_path),
                    "gate_decision": str(gate_path),
                },
            },
            indent=2,
        ),
        path=output_dir / "run_summary.json",
    )
    print(terminal_summary)
    return NeuralBenchmarkResult(
        output_dir=output_dir,
        metrics_path=metrics_path,
        comparisons_path=comparisons_path,
        gate_path=gate_path,
        terminal_summary_path=terminal_summary_path,
    )


def _admitted_dimensions(
    *, protocol: research_models.ResearchProtocol, protocol_path: Path
) -> tuple[tuple[tuple[int, tuple[int, ...]], ...], str]:
    baseline_dir = protocol.output_dir / "baseline_gate"
    summary_path = baseline_dir / "run_summary.json"
    gate_path = baseline_dir / "gate_decision.json"
    model_report_path = (
        protocol.output_dir / "model_protocol" / "model_protocol_verification.json"
    )
    if not summary_path.is_file() or not gate_path.is_file():
        raise FileNotFoundError("complete the defensive-baseline gate first")
    if not model_report_path.is_file():
        raise FileNotFoundError("complete model-protocol verification first")
    summary = _json_mapping(path=summary_path)
    gate = _json_mapping(path=gate_path)
    model_report = _json_mapping(path=model_report_path)
    if summary.get("protocol_sha256") != _baseline._sha256_file(path=protocol_path):
        raise ValueError("baseline evidence does not match the research protocol")
    if summary.get("manifest_sha256") != _baseline._sha256_file(
        path=protocol.output_dir / "split_manifest.yaml"
    ):
        raise ValueError("baseline evidence does not match the split manifest")
    if summary.get("audit_sha256") != _baseline._sha256_file(
        path=protocol.output_dir / "session_audit.csv"
    ):
        raise ValueError("baseline evidence does not match the session audit")
    if gate.get("locked_evaluation_used") is not False:
        raise ValueError("baseline gate must not use locked evaluation outcomes")
    if gate.get("eligible_for_neural_benchmark") is not True:
        raise ValueError("defensive-baseline gate did not admit neural benchmarking")
    if model_report.get("all_capability_checks_passed") is not True:
        raise ValueError("model-protocol capability verification did not pass")
    if (
        model_report.get("model_protocol_version")
        != model_domain.MODEL_PROTOCOL_VERSION
    ):
        raise ValueError("model-protocol verification belongs to another model version")
    raw_model_checks = model_report.get("models")
    if not isinstance(raw_model_checks, list) or {
        item.get("model")
        for item in raw_model_checks
        if isinstance(item, dict) and item.get("checks_passed") is True
    } != set(protocol.models):
        raise ValueError("model-protocol report does not verify every declared model")
    raw_horizons = gate.get("baseline_signal_by_horizon")
    raw_depths = gate.get("depth_support_by_horizon")
    if not isinstance(raw_horizons, dict) or not isinstance(raw_depths, dict):
        raise ValueError("baseline gate dimensions are invalid")
    expected_keys = {str(item) for item in protocol.forecast_horizons_milliseconds}
    if set(raw_horizons) != expected_keys or set(raw_depths) != expected_keys:
        raise ValueError("baseline gate horizons do not match the research protocol")
    accepted = tuple(
        (
            item // protocol.state_interval_milliseconds,
            (
                tuple(sorted({min(protocol.depths), max(protocol.depths)}))
                if raw_depths[str(item)] is True
                else (min(protocol.depths),)
            ),
        )
        for item in protocol.forecast_horizons_milliseconds
        if raw_horizons[str(item)] is True
    )
    if not accepted:
        raise ValueError("baseline gate admitted no forecast horizon")
    return (
        accepted,
        _baseline._sha256_file(path=gate_path),
    )


def _run_identity(
    *,
    protocol_path: Path,
    policy: research_models.NeuralBenchmarkPolicy,
    policy_path: Path,
    manifest_path: Path,
    audit_path: Path,
    gate_hash: str,
    model_report_path: Path,
    baseline_metrics_path: Path,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "implementation_version": NEURAL_BENCHMARK_IMPLEMENTATION_VERSION,
        "model_protocol_version": model_domain.MODEL_PROTOCOL_VERSION,
        "protocol_sha256": _baseline._sha256_file(path=protocol_path),
        "policy_sha256": _baseline._sha256_file(path=policy_path),
        "manifest_sha256": _baseline._sha256_file(path=manifest_path),
        "audit_sha256": _baseline._sha256_file(path=audit_path),
        "baseline_gate_sha256": gate_hash,
        "baseline_metrics_sha256": _baseline._sha256_file(path=baseline_metrics_path),
        "model_protocol_report_sha256": _baseline._sha256_file(path=model_report_path),
        "policy": attrs.asdict(policy),
        "torch_version": torch.__version__,
    }


def _verify_or_write_identity(*, output_dir: Path, identity: dict[str, object]) -> None:
    identity_path = output_dir / "run_identity.json"
    if identity_path.is_file():
        if _json_mapping(path=identity_path) != identity:
            raise ValueError("neural benchmark inputs changed; use --replace-output")
        return
    cells_dir = output_dir / "cells"
    if cells_dir.exists() and any(cells_dir.rglob("cell_summary.json")):
        raise ValueError("neural cell outputs have no identity; use --replace-output")
    _write_text_atomically(text=json.dumps(identity, indent=2), path=identity_path)


def _cell(
    *,
    protocol: research_models.ResearchProtocol,
    policy: research_models.NeuralBenchmarkPolicy,
    fold: research_models.RollingFold,
    horizon_steps: int,
    depth: int,
    model_name: str,
    seed: int,
    output_dir: Path,
    identity: dict[str, object],
) -> _Cell:
    payload = {
        "run_identity": identity,
        "fold": fold.identifier,
        "training_sessions": [
            {"date": item.trading_date.isoformat(), "sha256": item.sha256}
            for item in fold.training_sessions
        ],
        "validation_sessions": [
            {"date": item.trading_date.isoformat(), "sha256": item.sha256}
            for item in fold.validation_sessions
        ],
        "horizon_steps": horizon_steps,
        "depth": depth,
        "model": model_name,
        "seed": seed,
        "policy": attrs.asdict(policy),
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()
    cell_dir = (
        output_dir
        / "cells"
        / fold.identifier
        / f"h{horizon_steps * protocol.state_interval_milliseconds}"
        / f"depth_{depth}"
        / model_name
        / f"seed_{seed}"
    )
    return _Cell(
        fold=fold,
        horizon_steps=horizon_steps,
        depth=depth,
        model_name=model_name,
        seed=seed,
        fingerprint=fingerprint,
        output_dir=cell_dir,
    )


def _completed_cell_metrics(*, cell: _Cell) -> pl.DataFrame | None:
    summary_path = cell.output_dir / "cell_summary.json"
    if not summary_path.is_file():
        return None
    summary = _json_mapping(path=summary_path)
    if summary.get("fingerprint") != cell.fingerprint:
        raise ValueError(f"completed neural cell is incompatible: {cell.output_dir}")
    required_paths = (
        cell.output_dir / "best.pt",
        cell.output_dir / "latest.pt",
        cell.output_dir / "session_metrics.csv",
        cell.output_dir / "predictions.parquet",
        cell.output_dir / "training_history.json",
    )
    if any(not path.is_file() for path in required_paths):
        raise ValueError(f"completed neural cell is incomplete: {cell.output_dir}")
    raw_hashes = summary.get("artifact_sha256")
    if not isinstance(raw_hashes, dict) or raw_hashes != {
        path.name: _baseline._sha256_file(path=path) for path in required_paths
    }:
        raise ValueError(f"completed neural cell is corrupt: {cell.output_dir}")
    metrics = pl.read_csv(
        cell.output_dir / "session_metrics.csv",
        schema_overrides={"validation_date": pl.Date},
    )
    expected_dates = {item.trading_date for item in cell.fold.validation_sessions}
    expected_dimensions = {
        "fold": cell.fold.identifier,
        "model": cell.model_name,
        "depth": cell.depth,
        "horizon_steps": cell.horizon_steps,
        "seed": cell.seed,
    }
    if (
        metrics.height != len(expected_dates)
        or set(metrics["validation_date"].to_list()) != expected_dates
        or any(
            metrics[column].n_unique() != 1 or metrics[column][0] != expected_value
            for column, expected_value in expected_dimensions.items()
        )
    ):
        raise ValueError(
            f"completed neural cell metrics are invalid: {cell.output_dir}"
        )
    return metrics


def _prepare_corpora(
    *,
    protocol: research_models.ResearchProtocol,
    fold: research_models.RollingFold,
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
            for item in fold.training_sessions
        )
    )
    scaled_training = tuple(
        pilot_training.apply_feature_scaler(
            session=_baseline._extract_session(
                protocol=protocol,
                identity=item,
                depth=depth,
                horizon_steps=horizon_steps,
            ),
            scaler=scaler,
        )
        for item in fold.training_sessions
    )
    training = pilot_training.combine_sessions(
        sessions=scaled_training,
        context_steps=(
            protocol.context_milliseconds // protocol.state_interval_milliseconds
        ),
        horizon_steps=horizon_steps,
        maximum_windows=None,
        stride_steps=research_operations.training_stride_steps(
            protocol=protocol,
            horizon_milliseconds=(horizon_steps * protocol.state_interval_milliseconds),
        ),
    )
    del scaled_training
    scaled_validation = tuple(
        pilot_training.apply_feature_scaler(
            session=_baseline._extract_session(
                protocol=protocol,
                identity=item,
                depth=depth,
                horizon_steps=horizon_steps,
            ),
            scaler=scaler,
        )
        for item in fold.validation_sessions
    )
    validation = pilot_training.combine_sessions(
        sessions=scaled_validation,
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
    return training, validation


def _fit_cell(
    *,
    cell: _Cell,
    protocol: research_models.ResearchProtocol,
    policy: research_models.NeuralBenchmarkPolicy,
    device: torch.device,
    training_dataset: model_domain.SequenceDataset,
    validation_dataset: model_domain.SequenceDataset,
    validation_corpus: pilot_training.PreparedCorpus,
) -> pl.DataFrame:
    cell.output_dir.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model_domain.set_random_seed(seed=cell.seed)
    classifier = model_domain.build_direction_classifier(
        model_name=cell.model_name,
        auxiliary_size=validation_corpus.auxiliary_features.shape[1],
        hidden_size=policy.hidden_size,
    )
    latest_path = cell.output_dir / "latest.pt"
    best_path = cell.output_dir / "best.pt"
    resume_state = _checkpoint.read_latest(
        path=latest_path, fingerprint=cell.fingerprint
    )
    result = model_domain.fit_classifier(
        classifier=classifier,
        training_data=training_dataset,
        validation_data=validation_dataset,
        device=device,
        maximum_epochs=policy.maximum_epochs,
        batch_size=policy.batch_size,
        evaluation_batch_size=policy.evaluation_batch_size,
        learning_rate=policy.learning_rate,
        weight_decay=policy.weight_decay,
        early_stopping_patience=policy.early_stopping_patience,
        early_stopping_minimum_delta=policy.early_stopping_minimum_delta,
        gradient_clip_norm=policy.gradient_clip_norm,
        random_seed=cell.seed,
        validation_checks_per_epoch=protocol.validation_checks_per_epoch,
        resume_state=resume_state,
        epoch_observer=lambda metric: _print_validation(
            cell=cell,
            metric=metric,
            maximum_epochs=policy.maximum_epochs,
            state_interval_milliseconds=protocol.state_interval_milliseconds,
        ),
        checkpoint_observer=lambda state: _checkpoint.write_latest(
            path=latest_path, state=state, fingerprint=cell.fingerprint
        ),
    )
    _checkpoint.write_best(
        path=best_path,
        classifier=classifier,
        fingerprint=cell.fingerprint,
        training_result=result,
    )
    evaluation_classifier = model_domain.build_direction_classifier(
        model_name=cell.model_name,
        auxiliary_size=validation_corpus.auxiliary_features.shape[1],
        hidden_size=policy.hidden_size,
    )
    _checkpoint.load_best(
        path=best_path,
        classifier=evaluation_classifier,
        fingerprint=cell.fingerprint,
    )
    evaluation_started = time.perf_counter()
    prediction = model_domain.predict_classifier(
        classifier=evaluation_classifier,
        dataset=validation_dataset,
        device=device,
        batch_size=policy.evaluation_batch_size,
    )
    evaluation_elapsed_seconds = time.perf_counter() - evaluation_started
    parameter_count = model_domain.parameter_count(classifier=evaluation_classifier)
    metrics, predictions = _session_outputs(
        cell=cell,
        protocol=protocol,
        corpus=validation_corpus,
        probabilities=prediction.probabilities,
        predicted_labels=prediction.labels,
        parameter_count=parameter_count,
    )
    metrics_path = cell.output_dir / "session_metrics.csv"
    predictions_path = cell.output_dir / "predictions.parquet"
    history_path = cell.output_dir / "training_history.json"
    _write_csv_atomically(data=metrics, path=metrics_path)
    _write_parquet_atomically(data=predictions, path=predictions_path)
    _write_text_atomically(
        text=json.dumps(
            {
                "best_epoch": result.best_epoch,
                "best_validation_log_loss": result.best_validation_log_loss,
                "epochs_completed": result.epochs_completed,
                "stop_reason": result.stop_reason,
                "fit_elapsed_seconds": result.fit_elapsed_seconds,
                "validation_elapsed_seconds": result.validation_elapsed_seconds,
                "final_evaluation_elapsed_seconds": evaluation_elapsed_seconds,
                "validations": [attrs.asdict(item) for item in result.history],
            },
            indent=2,
        ),
        path=history_path,
    )
    artifact_hashes = {
        path.name: _baseline._sha256_file(path=path)
        for path in (best_path, metrics_path, predictions_path, history_path)
    }
    if latest_path.is_file():
        artifact_hashes[latest_path.name] = _baseline._sha256_file(path=latest_path)
    _write_text_atomically(
        text=json.dumps(
            {
                "fingerprint": cell.fingerprint,
                "fold": cell.fold.identifier,
                "horizon_steps": cell.horizon_steps,
                "depth": cell.depth,
                "model": cell.model_name,
                "seed": cell.seed,
                "parameter_count": parameter_count,
                "training_batch_size": policy.batch_size,
                "evaluation_batch_size": policy.evaluation_batch_size,
                "training_windows": len(training_dataset),
                "validation_windows": len(validation_dataset),
                "fit_elapsed_seconds": result.fit_elapsed_seconds,
                "validation_elapsed_seconds": result.validation_elapsed_seconds,
                "final_evaluation_elapsed_seconds": evaluation_elapsed_seconds,
                "peak_cuda_memory_gib": _peak_cuda_memory_gib(device=device),
                "locked_evaluation_used": False,
                "artifact_sha256": artifact_hashes,
                "artifacts": {
                    "best": str(best_path),
                    "latest": str(latest_path),
                    "metrics": str(metrics_path),
                    "predictions": str(predictions_path),
                    "history": str(history_path),
                },
            },
            indent=2,
        ),
        path=cell.output_dir / "cell_summary.json",
    )
    return metrics


def _session_outputs(
    *,
    cell: _Cell,
    protocol: research_models.ResearchProtocol,
    corpus: pilot_training.PreparedCorpus,
    probabilities: np.ndarray,
    predicted_labels: np.ndarray,
    parameter_count: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pl.DataFrame] = []
    labels = corpus.labels[corpus.target_indices]
    timestamps = corpus.timestamps[corpus.target_indices]
    for offset, length, identity in zip(
        corpus.session_offsets,
        corpus.session_lengths,
        cell.fold.validation_sessions,
        strict=True,
    ):
        selection = (corpus.target_indices >= offset) & (
            corpus.target_indices < offset + length
        )
        session_labels = labels[selection]
        session_probabilities = probabilities[selection]
        row = model_domain.direction_metric_row(
            model_name=cell.model_name,
            depth=cell.depth,
            horizon_steps=cell.horizon_steps,
            seed=cell.seed,
            labels=session_labels,
            probabilities=session_probabilities,
            parameter_count=parameter_count,
        )
        row.update(
            {
                "instrument": protocol.development_instrument.value,
                "fold": cell.fold.identifier,
                "validation_date": identity.trading_date,
            }
        )
        metric_rows.append(row)
        prediction_frames.append(
            pl.DataFrame(
                {
                    orderbook_models.COL_TIMESTAMP: timestamps[selection],
                    "instrument": protocol.development_instrument.value,
                    "fold": cell.fold.identifier,
                    "validation_date": identity.trading_date,
                    "model": cell.model_name,
                    "depth": cell.depth,
                    "horizon_steps": cell.horizon_steps,
                    "horizon_milliseconds": (
                        cell.horizon_steps * protocol.state_interval_milliseconds
                    ),
                    "seed": cell.seed,
                    "true_direction": session_labels - 1,
                    "predicted_direction": predicted_labels[selection] - 1,
                    "probability_down": session_probabilities[:, 0],
                    "probability_flat": session_probabilities[:, 1],
                    "probability_up": session_probabilities[:, 2],
                }
            )
        )
    return pl.DataFrame(metric_rows), pl.concat(prediction_frames)


def _paired_comparisons(
    *,
    neural_metrics: pl.DataFrame,
    baseline_metrics: pl.DataFrame,
    protocol: research_models.ResearchProtocol,
) -> pl.DataFrame:
    metric_names = tuple(
        item.value for item in (*protocol.primary_metrics, *protocol.supporting_metrics)
    )
    averaged = neural_metrics.group_by(
        [
            "fold",
            "validation_date",
            "model",
            "depth",
            "horizon_steps",
            "horizon_milliseconds",
        ]
    ).agg([pl.col(name).mean().alias(name) for name in metric_names])
    rows: list[dict[str, object]] = []
    for model_name in protocol.models:
        for horizon_steps in sorted(neural_metrics["horizon_steps"].unique()):
            horizon_milliseconds = (
                int(horizon_steps) * protocol.state_interval_milliseconds
            )
            horizon_neural = averaged.filter(
                (pl.col("model") == model_name)
                & (pl.col("horizon_steps") == horizon_steps)
            )
            for depth in sorted(horizon_neural["depth"].unique()):
                neural = horizon_neural.filter(pl.col("depth") == depth)
                baseline = baseline_metrics.filter(
                    (pl.col("model") == "logistic")
                    & (pl.col("depth") == depth)
                    & (pl.col("horizon_steps") == horizon_steps)
                )
                for metric_name in metric_names:
                    interval = research_operations.paired_session_interval(
                        shallower_by_session=_metric_by_session(
                            data=baseline, metric_name=metric_name
                        ),
                        deeper_by_session=_metric_by_session(
                            data=neural, metric_name=metric_name
                        ),
                        repetitions=protocol.bootstrap_repetitions,
                        confidence_level=protocol.confidence_level,
                        random_seed=101 + int(horizon_steps) + int(depth),
                    )
                    rows.append(
                        {
                            "comparison": (
                                "seed_mean_neural_minus_logistic_same_depth"
                            ),
                            "model": model_name,
                            "depth": int(depth),
                            "horizon_steps": int(horizon_steps),
                            "horizon_milliseconds": horizon_milliseconds,
                            "metric": metric_name,
                            "favorable_direction": _baseline._favorable_direction(
                                metric_name=metric_name
                            ),
                            **interval,
                        }
                    )
    return pl.DataFrame(rows).sort(["model", "depth", "horizon_steps", "metric"])


def _metric_by_session(*, data: pl.DataFrame, metric_name: str) -> dict[str, float]:
    return {
        f"{row['fold']}:{row['validation_date']}": float(row[metric_name])
        for row in data.iter_rows(named=True)
    }


def _gate_decision(
    *, comparisons: pl.DataFrame, protocol: research_models.ResearchProtocol
) -> dict[str, object]:
    primary = tuple(item.value for item in protocol.primary_metrics)
    accepted: list[dict[str, object]] = []
    evidence: dict[str, bool] = {}
    for model_name in protocol.models:
        for horizon in sorted(comparisons["horizon_milliseconds"].unique()):
            horizon_rows = comparisons.filter(
                (pl.col("model") == model_name)
                & (pl.col("horizon_milliseconds") == horizon)
            )
            for depth in sorted(horizon_rows["depth"].unique()):
                rows = horizon_rows.filter(
                    (pl.col("depth") == depth) & pl.col("metric").is_in(primary)
                )
                passed = rows.height == len(primary) and bool(
                    (rows["confidence_lower"] > 0.0).all()
                )
                key = f"{model_name}:d{depth}:h{horizon}"
                evidence[key] = passed
                if passed:
                    accepted.append(
                        {
                            "model": model_name,
                            "depth": int(depth),
                            "horizon_milliseconds": int(horizon),
                        }
                    )
    return {
        "decision_rule": (
            "A model-horizon passes only when its across-seed mean has a strictly "
            "positive lower paired-session bootstrap confidence bound over the "
            "same-depth logistic baseline for every primary metric."
        ),
        "primary_metrics": list(primary),
        "neural_signal_by_model_horizon": evidence,
        "accepted_model_depth_horizons": accepted,
        "locked_evaluation_used": False,
    }


def _print_validation(
    *,
    cell: _Cell,
    metric: model_domain.EpochMetric,
    maximum_epochs: int,
    state_interval_milliseconds: int,
) -> None:
    marker = " best" if metric.improved else ""
    print(
        f"[neural-benchmark:{cell.fold.identifier}:{cell.model_name}:"
        f"h{cell.horizon_steps * state_interval_milliseconds}:s{cell.seed}] "
        f"epoch={metric.epoch}/{maximum_epochs} "
        f"validation={metric.validation_index} step={metric.optimizer_step} "
        f"train_loss={metric.training_loss:.6f} "
        f"validation_log_loss={metric.validation_log_loss:.6f} "
        f"gradient_norm={metric.gradient_norm:.4f}{marker}",
        flush=True,
    )


def _peak_cuda_memory_gib(*, device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / (1024**3)


def _json_mapping(*, path: Path) -> dict[str, object]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"JSON artifact must be a mapping: {path}")
    return cast(dict[str, object], loaded)


def _write_csv_atomically(*, data: pl.DataFrame, path: Path) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    data.write_csv(temporary_path)
    temporary_path.replace(path)


def _write_parquet_atomically(*, data: pl.DataFrame, path: Path) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    data.write_parquet(temporary_path)
    temporary_path.replace(path)


def _write_text_atomically(*, text: str, path: Path) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(text, encoding="utf-8")
    temporary_path.replace(path)
