"""Compare a paper-inspired LSTM on the frozen native-resolution EUR/USD task."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import date, datetime
from pathlib import Path
from typing import cast

import attrs
import polars as pl
import torch
import xlsxwriter

from ebs_tft.application.usecases.pilot import _checkpoint
from ebs_tft.application.usecases.research_protocol import _baseline, _locked, _neural
from ebs_tft.data.repositories import checkpoint as checkpoint_repository
from ebs_tft.domain import model as model_domain
from ebs_tft.domain.pilot import training as pilot_training
from ebs_tft.domain.research import models as research_models
from ebs_tft.domain.research import operations as research_operations

IMPLEMENTATION_VERSION = 1
MODEL_NAME = "ito_inspired_lstm"
_METRICS = ("accuracy", "macro_f1", "mcc")


@attrs.frozen
class PaperLstmResult:
    """Reference the completed output of one comparison stage."""

    output_dir: Path
    metrics_path: Path
    comparisons_path: Path
    summary_path: Path


class PaperLstmPausedError(Exception):
    """Indicate that a safe number of new model cells has completed."""


def run(
    *,
    protocol: research_models.ResearchProtocol,
    protocol_path: Path,
    policy: research_models.NeuralBenchmarkPolicy,
    policy_path: Path,
    stage: str,
    maximum_new_cells: int | None = None,
) -> PaperLstmResult:
    """Run development first, then a separately labeled retrospective stage."""
    _validate_protocol(protocol=protocol, stage=stage)
    if maximum_new_cells is not None and (
        isinstance(maximum_new_cells, bool) or maximum_new_cells <= 0
    ):
        raise ValueError("maximum_new_cells must be positive or null")
    started = time.perf_counter()
    folds = _baseline._load_folds(
        manifest_path=protocol.output_dir / "split_manifest.yaml",
        audit_path=protocol.output_dir / "session_audit.csv",
        protocol_path=protocol_path,
        protocol=protocol,
    )
    if len(folds) != 1:
        raise ValueError("the paper-inspired comparison requires one fixed-period fold")
    fold = folds[0]
    _baseline._verify_cached_states(
        folds=folds,
        audit_path=protocol.output_dir / "session_audit.csv",
        protocol=protocol,
    )
    root = protocol.output_dir / "paper_lstm_comparison"
    if stage == "replication":
        development_dir = root / "development"
        if _completed_result(output_dir=development_dir) is None:
            raise ValueError(
                "complete paper-LSTM development before retrospective replication"
            )
        development_identity = _neural._json_mapping(
            path=development_dir / "run_identity.json"
        )
        expected_development_identity = _identity(
            protocol=protocol,
            protocol_path=protocol_path,
            policy_path=policy_path,
            reference_path=protocol.output_dir
            / "neural_benchmark"
            / "session_metrics.csv",
            stage="development",
        )
        if development_identity != expected_development_identity:
            raise ValueError("paper-LSTM development used different frozen inputs")
    output_dir = root / stage
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_path = protocol.output_dir / (
        "neural_benchmark/session_metrics.csv"
        if stage == "development"
        else "locked_evaluation/session_metrics.csv"
    )
    identity = _identity(
        protocol=protocol,
        protocol_path=protocol_path,
        policy_path=policy_path,
        reference_path=reference_path,
        stage=stage,
    )
    _verify_or_write_json(path=output_dir / "run_identity.json", value=identity)
    completed_result = _completed_result(output_dir=output_dir)
    if completed_result is not None:
        return completed_result
    reference = _reference_metrics(
        path=reference_path,
        expected_dates=(
            tuple(item.trading_date for item in fold.validation_sessions)
            if stage == "development"
            else tuple(
                item.trading_date
                for item in _locked._final_sessions(
                    manifest_path=protocol.output_dir / "split_manifest.yaml",
                    protocol=protocol,
                )
            )
        ),
        protocol=protocol,
    )
    plan = (
        _replication_plan(
            root=root, output_dir=output_dir, protocol=protocol, fold=fold
        )
        if stage == "replication"
        else None
    )
    expected_dates = (
        tuple(item.trading_date for item in fold.validation_sessions)
        if stage == "development"
        else tuple(
            item.trading_date
            for item in _locked._final_sessions(
                manifest_path=protocol.output_dir / "split_manifest.yaml",
                protocol=protocol,
            )
        )
    )
    cells = tuple(
        (feature_set, seed)
        for feature_set in model_domain.PAPER_LSTM_FEATURE_SETS
        for seed in protocol.random_seeds
    )
    metrics_by_cell: list[pl.DataFrame] = []
    pending: list[tuple[str, int]] = []
    for feature_set, seed in cells:
        cell_dir = output_dir / "cells" / feature_set / f"seed_{seed}"
        fingerprint = _fingerprint(
            identity=identity, feature_set=feature_set, seed=seed, plan=plan
        )
        completed = _completed_metrics(
            cell_dir=cell_dir,
            fingerprint=fingerprint,
            expected_dates=expected_dates,
            feature_set=feature_set,
            seed=seed,
        )
        if completed is None:
            pending.append((feature_set, seed))
        else:
            metrics_by_cell.append(completed)
    if pending:
        new_cells = 0
        horizon_steps = protocol.horizon_steps[0]
        if stage == "development":
            training_corpus, evaluation_corpus = _neural._prepare_corpora(
                protocol=protocol,
                fold=fold,
                depth=1,
                horizon_steps=horizon_steps,
            )
        else:
            final_sessions = _locked._final_sessions(
                manifest_path=protocol.output_dir / "split_manifest.yaml",
                protocol=protocol,
            )
            _locked._verify_sources(
                sessions=(
                    *fold.training_sessions,
                    *fold.validation_sessions,
                    *final_sessions,
                )
            )
            _locked._materialize_locked_cache(
                protocol=protocol,
                sessions=final_sessions,
                cache_dir=output_dir / "native_cache",
                log_prefix="paper-lstm:replication",
            )
            training_corpus, evaluation_corpus = _locked._prepare_final_corpora(
                protocol=protocol,
                development_sessions=_locked._development_sessions(folds=folds),
                final_sessions=final_sessions,
                depth=1,
                horizon_steps=horizon_steps,
                cache_dir=output_dir / "native_cache",
            )
        device = model_domain.select_device(requested=policy.device)
        context_steps = (
            protocol.context_milliseconds // protocol.state_interval_milliseconds
        )
        training_data = _dataset(
            corpus=training_corpus, context_steps=context_steps, device=device
        )
        evaluation_data = _dataset(
            corpus=evaluation_corpus, context_steps=context_steps, device=device
        )
        for feature_set, seed in pending:
            print(
                f"[paper-lstm:{stage}] cell={len(metrics_by_cell) + 1}/{len(cells)} "
                f"feature_set={feature_set} seed={seed}",
                flush=True,
            )
            cell_dir = output_dir / "cells" / feature_set / f"seed_{seed}"
            fingerprint = _fingerprint(
                identity=identity, feature_set=feature_set, seed=seed, plan=plan
            )
            if stage == "development":
                metrics = _development_cell(
                    cell_dir=cell_dir,
                    fingerprint=fingerprint,
                    feature_set=feature_set,
                    seed=seed,
                    policy=policy,
                    protocol=protocol,
                    training_data=training_data,
                    evaluation_data=evaluation_data,
                    evaluation_corpus=evaluation_corpus,
                    expected_dates=expected_dates,
                    device=device,
                )
            else:
                assert plan is not None
                metrics = _replication_cell(
                    cell_dir=cell_dir,
                    fingerprint=fingerprint,
                    feature_set=feature_set,
                    seed=seed,
                    fixed_epochs=_planned_epochs(
                        plan=plan, feature_set=feature_set, seed=seed
                    ),
                    policy=policy,
                    protocol=protocol,
                    training_data=training_data,
                    evaluation_data=evaluation_data,
                    evaluation_corpus=evaluation_corpus,
                    expected_dates=expected_dates,
                    device=device,
                )
            metrics_by_cell.append(metrics)
            new_cells += 1
            print(f"[paper-lstm:{stage}] checkpoint=saved", flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if (
                maximum_new_cells is not None
                and len(metrics_by_cell) < len(cells)
                and new_cells >= maximum_new_cells
            ):
                raise PaperLstmPausedError(
                    f"Paper LSTM paused after {len(metrics_by_cell)}/{len(cells)} "
                    f"cells; rerun without replacing outputs: {output_dir}"
                )
        del training_data, evaluation_data, training_corpus, evaluation_corpus
    metrics = pl.concat(metrics_by_cell).sort(
        ["feature_set", "seed", "validation_date"]
    )
    _verify_sample_alignment(metrics=metrics, reference=reference)
    metrics_path = output_dir / "session_metrics.csv"
    _neural._write_csv_atomically(data=metrics, path=metrics_path)
    comparisons = _comparisons(
        metrics=metrics, reference=reference, protocol=protocol, stage=stage
    )
    comparisons_path = output_dir / "paired_comparisons.csv"
    _neural._write_csv_atomically(data=comparisons, path=comparisons_path)
    absolute_summary = _absolute_summary(metrics=metrics, reference=reference)
    absolute_summary_path = output_dir / "absolute_summary.csv"
    _neural._write_csv_atomically(data=absolute_summary, path=absolute_summary_path)
    training_details = _training_details(output_dir=output_dir)
    training_details_path = output_dir / "training_details.csv"
    _neural._write_csv_atomically(data=training_details, path=training_details_path)
    workbook_path = output_dir / "paper_lstm_comparison.xlsx"
    _write_workbook(
        path=workbook_path,
        metrics=metrics,
        reference=reference,
        absolute_summary=absolute_summary,
        comparisons=comparisons,
        training_details=training_details,
        stage=stage,
    )
    summary_path = output_dir / "run_summary.json"
    _neural._write_text_atomically(
        text=json.dumps(
            {
                "status": "complete",
                "stage": stage,
                "paper_reproduction": False,
                "minute_aggregation_used": False,
                "state_interval_milliseconds": 100,
                "forecast_horizon_milliseconds": 30000,
                "retrospective_march": stage == "replication",
                "retuning_permitted": False,
                "cells": len(cells),
                "sessions": len(expected_dates),
                "elapsed_seconds": time.perf_counter() - started,
                "identity_sha256": _baseline._sha256_file(
                    path=output_dir / "run_identity.json"
                ),
                "metrics_sha256": _baseline._sha256_file(path=metrics_path),
                "comparisons_sha256": _baseline._sha256_file(path=comparisons_path),
                "absolute_summary_sha256": _baseline._sha256_file(
                    path=absolute_summary_path
                ),
                "workbook_sha256": _baseline._sha256_file(path=workbook_path),
                "training_details_sha256": _baseline._sha256_file(
                    path=training_details_path
                ),
                "training_windows": int(training_details["training_windows"][0]),
                "evaluation_windows": int(training_details["evaluation_windows"][0]),
                "cell_summary_sha256": {
                    str(path.relative_to(output_dir)): _baseline._sha256_file(path=path)
                    for path in sorted(
                        (output_dir / "cells").rglob("cell_summary.json")
                    )
                },
            },
            indent=2,
        ),
        path=summary_path,
    )
    return PaperLstmResult(
        output_dir=output_dir,
        metrics_path=metrics_path,
        comparisons_path=comparisons_path,
        summary_path=summary_path,
    )


def _validate_protocol(
    *, protocol: research_models.ResearchProtocol, stage: str
) -> None:
    if stage not in {"development", "replication"}:
        raise ValueError("stage must be development or replication")
    if (
        protocol.development_instrument.value != "EUR_USD"
        or protocol.years != (2023, 2024)
        or protocol.forecast_horizons_milliseconds != (30000,)
        or protocol.depths != (1,)
        or protocol.state_interval_milliseconds != 100
        or set(protocol.models) != {"deeplob_direction", "tft_direction"}
        or protocol.context_milliseconds != 10000
    ):
        raise ValueError("paper LSTM requires the frozen EUR/USD longitudinal protocol")


def _identity(
    *,
    protocol: research_models.ResearchProtocol,
    protocol_path: Path,
    policy_path: Path,
    reference_path: Path,
    stage: str,
) -> dict[str, object]:
    paths = {
        "protocol_sha256": protocol_path,
        "policy_sha256": policy_path,
        "manifest_sha256": protocol.output_dir / "split_manifest.yaml",
        "audit_sha256": protocol.output_dir / "session_audit.csv",
        "reference_metrics_sha256": reference_path,
        "runner_sha256": Path(__file__),
        "model_sha256": (
            Path(__file__).resolve().parents[3] / "domain" / "model" / "_paper_lstm.py"
        ),
        "training_engine_sha256": (
            Path(__file__).resolve().parents[3] / "domain" / "model" / "_pilot.py"
        ),
        "corpus_engine_sha256": Path(__file__).with_name("_neural.py"),
        "replication_engine_sha256": Path(__file__).with_name("_locked.py"),
    }
    return {
        "schema_version": 1,
        "implementation_version": IMPLEMENTATION_VERSION,
        "stage": stage,
        "feature_sets": list(model_domain.PAPER_LSTM_FEATURE_SETS),
        "architecture": "one_layer_lstm_50_hidden_three_class_softmax",
        "optimizer": "adam",
        "weight_decay": 0,
        "paper_reproduction": False,
        "training_data_semantics": "native_100ms_snapshots_not_order_events",
        **{key: _baseline._sha256_file(path=path) for key, path in paths.items()},
    }


def _verify_or_write_json(*, path: Path, value: dict[str, object]) -> None:
    if path.is_file():
        if _neural._json_mapping(path=path) != value:
            raise ValueError(f"existing experiment identity differs: {path}")
    else:
        _neural._write_text_atomically(text=json.dumps(value, indent=2), path=path)


def _completed_result(*, output_dir: Path) -> PaperLstmResult | None:
    summary_path = output_dir / "run_summary.json"
    if not summary_path.is_file():
        return None
    summary = _neural._json_mapping(path=summary_path)
    metrics_path = output_dir / "session_metrics.csv"
    comparisons_path = output_dir / "paired_comparisons.csv"
    absolute_summary_path = output_dir / "absolute_summary.csv"
    workbook_path = output_dir / "paper_lstm_comparison.xlsx"
    training_details_path = output_dir / "training_details.csv"
    raw_cell_hashes = summary.get("cell_summary_sha256")
    if (
        summary.get("status") != "complete"
        or summary.get("identity_sha256")
        != _baseline._sha256_file(path=output_dir / "run_identity.json")
        or any(
            not path.is_file()
            for path in (
                metrics_path,
                comparisons_path,
                absolute_summary_path,
                workbook_path,
                training_details_path,
            )
        )
        or summary.get("metrics_sha256") != _baseline._sha256_file(path=metrics_path)
        or summary.get("comparisons_sha256")
        != _baseline._sha256_file(path=comparisons_path)
        or summary.get("absolute_summary_sha256")
        != _baseline._sha256_file(path=absolute_summary_path)
        or summary.get("workbook_sha256") != _baseline._sha256_file(path=workbook_path)
        or summary.get("training_details_sha256")
        != _baseline._sha256_file(path=training_details_path)
        or not isinstance(raw_cell_hashes, dict)
        or len(raw_cell_hashes) != summary.get("cells")
    ):
        raise ValueError(
            f"completed paper LSTM output failed integrity check: {output_dir}"
        )
    for relative_path, digest in raw_cell_hashes.items():
        if not isinstance(relative_path, str) or not isinstance(digest, str):
            raise ValueError("completed paper LSTM cell hash list is invalid")
        cell_summary_path = output_dir / relative_path
        if (
            not cell_summary_path.is_file()
            or _baseline._sha256_file(path=cell_summary_path) != digest
        ):
            raise ValueError(
                f"completed paper LSTM cell is corrupt: {cell_summary_path}"
            )
        cell_summary = _neural._json_mapping(path=cell_summary_path)
        artifact_hashes = cell_summary.get("artifact_sha256")
        if not isinstance(artifact_hashes, dict) or not artifact_hashes:
            raise ValueError("completed paper LSTM artifact list is invalid")
        for name, artifact_digest in artifact_hashes.items():
            if not isinstance(name, str) or not isinstance(artifact_digest, str):
                raise ValueError("completed paper LSTM artifact hash is invalid")
            artifact = cell_summary_path.parent / name
            if (
                not artifact.is_file()
                or _baseline._sha256_file(path=artifact) != artifact_digest
            ):
                raise ValueError(
                    f"completed paper LSTM artifact is corrupt: {artifact}"
                )
    return PaperLstmResult(
        output_dir=output_dir,
        metrics_path=metrics_path,
        comparisons_path=comparisons_path,
        summary_path=summary_path,
    )


def _reference_metrics(
    *,
    path: Path,
    expected_dates: tuple[object, ...],
    protocol: research_models.ResearchProtocol,
) -> pl.DataFrame:
    reference = pl.read_csv(path, schema_overrides={"validation_date": pl.Date})
    reference = reference.filter(
        pl.col("model").is_in(list(protocol.models))
        & (pl.col("depth") == 1)
        & (pl.col("horizon_milliseconds") == 30000)
    )
    expected = {
        (model, seed, date)
        for model in protocol.models
        for seed in protocol.random_seeds
        for date in expected_dates
    }
    actual = set(
        zip(
            reference["model"],
            reference["seed"],
            reference["validation_date"],
            strict=True,
        )
    )
    if actual != expected or reference.height != len(expected):
        raise ValueError("reference model metrics do not cover the exact matched cells")
    return reference


def _replication_plan(
    *,
    root: Path,
    output_dir: Path,
    protocol: research_models.ResearchProtocol,
    fold: research_models.RollingFold,
) -> dict[str, object]:
    development_dir = root / "development"
    summary_path = development_dir / "run_summary.json"
    if (
        not summary_path.is_file()
        or _neural._json_mapping(path=summary_path).get("status") != "complete"
    ):
        raise ValueError(
            "complete paper-LSTM development before retrospective replication"
        )
    cells: list[dict[str, object]] = []
    for feature_set in model_domain.PAPER_LSTM_FEATURE_SETS:
        for seed in protocol.random_seeds:
            summary = _neural._json_mapping(
                path=development_dir
                / "cells"
                / feature_set
                / f"seed_{seed}"
                / "cell_summary.json"
            )
            epochs = summary.get("best_epoch")
            if not isinstance(epochs, int) or epochs <= 0:
                raise ValueError("development best epoch is invalid")
            cells.append(
                {"feature_set": feature_set, "seed": seed, "fixed_epochs": epochs}
            )
    final_sessions = _locked._final_sessions(
        manifest_path=protocol.output_dir / "split_manifest.yaml", protocol=protocol
    )
    plan = {
        "schema_version": 1,
        "development_summary_sha256": _baseline._sha256_file(path=summary_path),
        "development_sessions": len(fold.training_sessions)
        + len(fold.validation_sessions),
        "replication_dates": [item.trading_date.isoformat() for item in final_sessions],
        "retrospective_not_pristine": True,
        "cells": cells,
    }
    _verify_or_write_json(path=output_dir / "plan.json", value=plan)
    return plan


def _planned_epochs(*, plan: dict[str, object], feature_set: str, seed: int) -> int:
    cells = cast(list[dict[str, object]], plan["cells"])
    matching = [
        item
        for item in cells
        if item["feature_set"] == feature_set and item["seed"] == seed
    ]
    if len(matching) != 1:
        raise ValueError("replication plan does not contain exactly one matching cell")
    return cast(int, matching[0]["fixed_epochs"])


def _fingerprint(
    *,
    identity: dict[str, object],
    feature_set: str,
    seed: int,
    plan: dict[str, object] | None,
) -> str:
    payload = {
        "identity": identity,
        "feature_set": feature_set,
        "seed": seed,
        "plan": plan,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _dataset(
    *, corpus: pilot_training.PreparedCorpus, context_steps: int, device: torch.device
) -> model_domain.SequenceDataset:
    return model_domain.SequenceDataset(
        lob_features=corpus.lob_features,
        auxiliary_features=corpus.auxiliary_features,
        labels=corpus.labels,
        target_indices=corpus.target_indices,
        context_steps=context_steps,
    ).to(device)


def _completed_metrics(
    *,
    cell_dir: Path,
    fingerprint: str,
    expected_dates: tuple[object, ...],
    feature_set: str,
    seed: int,
) -> pl.DataFrame | None:
    summary_path = cell_dir / "cell_summary.json"
    if not summary_path.is_file():
        return None
    summary = _neural._json_mapping(path=summary_path)
    if summary.get("fingerprint") != fingerprint:
        raise ValueError(f"cell identity differs: {cell_dir}")
    hashes = summary.get("artifact_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError(f"cell artifact list is invalid: {cell_dir}")
    for name, digest in hashes.items():
        path = cell_dir / name
        if not path.is_file() or _baseline._sha256_file(path=path) != digest:
            raise ValueError(f"cell artifact failed integrity verification: {path}")
    metrics = pl.read_csv(
        cell_dir / "session_metrics.csv", schema_overrides={"validation_date": pl.Date}
    )
    if (
        metrics.height != len(expected_dates)
        or set(metrics["validation_date"]) != set(expected_dates)
        or set(metrics["feature_set"]) != {feature_set}
        or set(metrics["seed"]) != {seed}
    ):
        raise ValueError(f"completed cell has invalid session coverage: {cell_dir}")
    return metrics


def _development_cell(
    *,
    cell_dir: Path,
    fingerprint: str,
    feature_set: str,
    seed: int,
    policy: research_models.NeuralBenchmarkPolicy,
    protocol: research_models.ResearchProtocol,
    training_data: model_domain.SequenceDataset,
    evaluation_data: model_domain.SequenceDataset,
    evaluation_corpus: pilot_training.PreparedCorpus,
    expected_dates: tuple[object, ...],
    device: torch.device,
) -> pl.DataFrame:
    cell_dir.mkdir(parents=True, exist_ok=True)
    model_domain.set_random_seed(seed=seed)
    classifier = model_domain.PaperLstmDirectionClassifier(feature_set=feature_set)
    latest_path = cell_dir / "latest.pt"
    result = model_domain.fit_classifier(
        classifier=classifier,
        training_data=training_data,
        validation_data=evaluation_data,
        device=device,
        maximum_epochs=policy.maximum_epochs,
        batch_size=policy.batch_size,
        evaluation_batch_size=policy.evaluation_batch_size,
        learning_rate=policy.learning_rate,
        weight_decay=0.0,
        early_stopping_patience=policy.early_stopping_patience,
        early_stopping_minimum_delta=policy.early_stopping_minimum_delta,
        gradient_clip_norm=policy.gradient_clip_norm,
        random_seed=seed,
        validation_checks_per_epoch=protocol.validation_checks_per_epoch,
        optimizer_name="adam",
        resume_state=_checkpoint.read_latest(path=latest_path, fingerprint=fingerprint),
        epoch_observer=lambda item: print(
            f"[paper-lstm:{feature_set}:s{seed}] epoch={item.epoch} "
            f"validation={item.validation_index} "
            f"log_loss={item.validation_log_loss:.6f}",
            flush=True,
        ),
        checkpoint_observer=lambda state: _checkpoint.write_latest(
            path=latest_path, state=state, fingerprint=fingerprint
        ),
    )
    best_path = cell_dir / "best.pt"
    _checkpoint.write_best(
        path=best_path,
        classifier=classifier,
        fingerprint=fingerprint,
        training_result=result,
    )
    metrics = _session_metrics(
        classifier=classifier,
        corpus=evaluation_corpus,
        dataset=evaluation_data,
        dates=expected_dates,
        feature_set=feature_set,
        seed=seed,
        protocol=protocol,
        device=device,
        batch_size=policy.evaluation_batch_size,
    )
    metrics_path = cell_dir / "session_metrics.csv"
    history_path = cell_dir / "training_history.json"
    _neural._write_csv_atomically(data=metrics, path=metrics_path)
    _neural._write_text_atomically(
        text=json.dumps(
            {
                "best_epoch": result.best_epoch,
                "best_validation_log_loss": result.best_validation_log_loss,
                "stop_reason": result.stop_reason,
                "fit_elapsed_seconds": result.fit_elapsed_seconds,
                "validations": [attrs.asdict(item) for item in result.history],
            },
            indent=2,
        ),
        path=history_path,
    )
    _write_cell_summary(
        cell_dir=cell_dir,
        fingerprint=fingerprint,
        feature_set=feature_set,
        seed=seed,
        best_epoch=result.best_epoch,
        training_windows=len(training_data),
        evaluation_windows=len(evaluation_data),
        parameter_count=model_domain.parameter_count(classifier=classifier),
        fit_elapsed_seconds=result.fit_elapsed_seconds,
        artifacts=(best_path, latest_path, metrics_path, history_path),
    )
    return metrics


def _replication_cell(
    *,
    cell_dir: Path,
    fingerprint: str,
    feature_set: str,
    seed: int,
    fixed_epochs: int,
    policy: research_models.NeuralBenchmarkPolicy,
    protocol: research_models.ResearchProtocol,
    training_data: model_domain.SequenceDataset,
    evaluation_data: model_domain.SequenceDataset,
    evaluation_corpus: pilot_training.PreparedCorpus,
    expected_dates: tuple[object, ...],
    device: torch.device,
) -> pl.DataFrame:
    cell_dir.mkdir(parents=True, exist_ok=True)
    model_domain.set_random_seed(seed=seed)
    classifier = model_domain.PaperLstmDirectionClassifier(feature_set=feature_set)
    latest_path = cell_dir / "latest.pt"
    result = model_domain.fit_classifier_fixed_epochs(
        classifier=classifier,
        training_data=training_data,
        device=device,
        epochs=fixed_epochs,
        batch_size=policy.batch_size,
        learning_rate=policy.learning_rate,
        weight_decay=0.0,
        gradient_clip_norm=policy.gradient_clip_norm,
        random_seed=seed,
        optimizer_name="adam",
        resume_state=_locked._read_fixed_checkpoint(
            path=latest_path, fingerprint=fingerprint
        ),
        epoch_observer=lambda item: print(
            f"[paper-lstm:{feature_set}:s{seed}] "
            f"epoch={item.epoch}/{fixed_epochs} loss={item.training_loss:.6f}",
            flush=True,
        ),
        checkpoint_observer=lambda state: _locked._write_fixed_checkpoint(
            path=latest_path, fingerprint=fingerprint, state=state
        ),
    )
    final_path = cell_dir / "final.pt"
    checkpoint_repository.write(
        path=final_path,
        payload={
            "kind": "paper_lstm_fixed_model",
            "fingerprint": fingerprint,
            "classifier_state": result.latest_state.classifier_state,
            "fixed_epochs": fixed_epochs,
        },
    )
    metrics = _session_metrics(
        classifier=classifier,
        corpus=evaluation_corpus,
        dataset=evaluation_data,
        dates=expected_dates,
        feature_set=feature_set,
        seed=seed,
        protocol=protocol,
        device=device,
        batch_size=policy.evaluation_batch_size,
    )
    metrics_path = cell_dir / "session_metrics.csv"
    _neural._write_csv_atomically(data=metrics, path=metrics_path)
    _write_cell_summary(
        cell_dir=cell_dir,
        fingerprint=fingerprint,
        feature_set=feature_set,
        seed=seed,
        best_epoch=fixed_epochs,
        training_windows=len(training_data),
        evaluation_windows=len(evaluation_data),
        parameter_count=model_domain.parameter_count(classifier=classifier),
        fit_elapsed_seconds=result.fit_elapsed_seconds,
        artifacts=(final_path, latest_path, metrics_path),
    )
    return metrics


def _write_cell_summary(
    *,
    cell_dir: Path,
    fingerprint: str,
    feature_set: str,
    seed: int,
    best_epoch: int,
    training_windows: int,
    evaluation_windows: int,
    parameter_count: int,
    fit_elapsed_seconds: float,
    artifacts: tuple[Path, ...],
) -> None:
    _neural._write_text_atomically(
        text=json.dumps(
            {
                "fingerprint": fingerprint,
                "feature_set": feature_set,
                "seed": seed,
                "best_epoch": best_epoch,
                "training_windows": training_windows,
                "evaluation_windows": evaluation_windows,
                "parameter_count": parameter_count,
                "fit_elapsed_seconds": fit_elapsed_seconds,
                "artifact_sha256": {
                    path.name: _baseline._sha256_file(path=path) for path in artifacts
                },
            },
            indent=2,
        ),
        path=cell_dir / "cell_summary.json",
    )


def _session_metrics(
    *,
    classifier: torch.nn.Module,
    corpus: pilot_training.PreparedCorpus,
    dataset: model_domain.SequenceDataset,
    dates: tuple[object, ...],
    feature_set: str,
    seed: int,
    protocol: research_models.ResearchProtocol,
    device: torch.device,
    batch_size: int,
) -> pl.DataFrame:
    prediction = model_domain.predict_classifier(
        classifier=classifier, dataset=dataset, device=device, batch_size=batch_size
    )
    selected_labels = corpus.labels[corpus.target_indices]
    rows: list[dict[str, object]] = []
    for offset, length, trading_date in zip(
        corpus.session_offsets, corpus.session_lengths, dates, strict=True
    ):
        selected = (corpus.target_indices >= offset) & (
            corpus.target_indices < offset + length
        )
        row = model_domain.direction_metric_row(
            model_name=MODEL_NAME,
            depth=1,
            horizon_steps=protocol.horizon_steps[0],
            seed=seed,
            labels=selected_labels[selected],
            probabilities=prediction.probabilities[selected],
            parameter_count=model_domain.parameter_count(classifier=classifier),
        )
        for direction in ("down", "flat", "up"):
            precision = cast(float, row[f"precision_{direction}"])
            recall = cast(float, row[f"recall_{direction}"])
            row[f"f1_{direction}"] = (
                2.0 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
        row.update(
            {
                "instrument": protocol.development_instrument.value,
                "fold": "fold_01",
                "validation_date": trading_date,
                "feature_set": feature_set,
            }
        )
        rows.append(row)
    return pl.DataFrame(rows)


def _verify_sample_alignment(*, metrics: pl.DataFrame, reference: pl.DataFrame) -> None:
    for row in (
        metrics.select("validation_date", "observations").unique().iter_rows(named=True)
    ):
        matching = reference.filter(pl.col("validation_date") == row["validation_date"])
        if matching.is_empty() or set(matching["observations"]) != {
            row["observations"]
        }:
            raise ValueError("LSTM and reference models have different target windows")
    if (
        metrics.select("validation_date").n_unique()
        != reference.select("validation_date").n_unique()
    ):
        raise ValueError("LSTM and reference models cover different sessions")


def _comparisons(
    *,
    metrics: pl.DataFrame,
    reference: pl.DataFrame,
    protocol: research_models.ResearchProtocol,
    stage: str,
) -> pl.DataFrame:
    groups: dict[str, pl.DataFrame] = {
        feature_set: metrics.filter(pl.col("feature_set") == feature_set)
        for feature_set in model_domain.PAPER_LSTM_FEATURE_SETS
    }
    groups.update(
        {
            model_name: reference.filter(pl.col("model") == model_name)
            for model_name in protocol.models
        }
    )
    pairs = (
        ("transactions_only", "level_1_and_transactions"),
        ("level_1_and_transactions", "deeplob_direction"),
        ("level_1_and_transactions", "tft_direction"),
    )
    rows: list[dict[str, object]] = []
    for reference_name, candidate_name in pairs:
        left = (
            groups[reference_name]
            .group_by("validation_date")
            .agg([pl.col(metric).mean().alias(metric) for metric in _METRICS])
        )
        right = (
            groups[candidate_name]
            .group_by("validation_date")
            .agg([pl.col(metric).mean().alias(metric) for metric in _METRICS])
        )
        for metric in _METRICS:
            interval = research_operations.paired_session_interval(
                shallower_by_session={
                    str(row["validation_date"]): row[metric]
                    for row in left.iter_rows(named=True)
                },
                deeper_by_session={
                    str(row["validation_date"]): row[metric]
                    for row in right.iter_rows(named=True)
                },
                repetitions=protocol.bootstrap_repetitions,
                confidence_level=protocol.confidence_level,
                random_seed=401 + _METRICS.index(metric),
            )
            rows.append(
                {
                    "stage": stage,
                    "comparison": f"{candidate_name}_minus_{reference_name}",
                    "candidate": candidate_name,
                    "reference": reference_name,
                    "metric": metric,
                    **interval,
                }
            )
    return pl.DataFrame(rows).sort("comparison", "metric")


def _absolute_summary(
    *, metrics: pl.DataFrame, reference: pl.DataFrame
) -> pl.DataFrame:
    """Report session-weighted absolute scores without treating windows as IID."""
    lstm = metrics.with_columns(
        (pl.lit(MODEL_NAME) + pl.lit(":") + pl.col("feature_set")).alias(
            "model_variant"
        )
    )
    established = reference.with_columns(pl.col("model").alias("model_variant"))
    common = ("model_variant", "validation_date", "seed", "observations", *_METRICS)
    rows = pl.concat((lstm.select(common), established.select(common)), how="vertical")
    per_session = rows.group_by("model_variant", "validation_date").agg(
        pl.col("observations").first(),
        *[pl.col(metric).mean().alias(metric) for metric in _METRICS],
    )
    return (
        per_session.group_by("model_variant")
        .agg(
            pl.len().alias("sessions"),
            pl.col("observations").sum(),
            *[
                pl.col(metric).mean().alias(f"mean_session_{metric}")
                for metric in _METRICS
            ],
        )
        .sort("model_variant")
    )


def _training_details(*, output_dir: Path) -> pl.DataFrame:
    """Collect durable per-cell sample and compute counts."""
    rows: list[dict[str, object]] = []
    for path in sorted((output_dir / "cells").rglob("cell_summary.json")):
        cell = _neural._json_mapping(path=path)
        rows.append(
            {
                "feature_set": cell["feature_set"],
                "seed": cell["seed"],
                "epochs": cell["best_epoch"],
                "training_windows": cell["training_windows"],
                "evaluation_windows": cell["evaluation_windows"],
                "parameter_count": cell["parameter_count"],
                "fit_elapsed_seconds": cell["fit_elapsed_seconds"],
            }
        )
    if not rows:
        raise ValueError("paper LSTM training details are missing")
    return pl.DataFrame(rows).sort("feature_set", "seed")


def _write_workbook(
    *,
    path: Path,
    metrics: pl.DataFrame,
    reference: pl.DataFrame,
    absolute_summary: pl.DataFrame,
    comparisons: pl.DataFrame,
    training_details: pl.DataFrame,
    stage: str,
) -> None:
    """Write one shareable account of matched scores and study limitations."""
    temporary = path.with_suffix(".xlsx.tmp")
    workbook = xlsxwriter.Workbook(str(temporary))
    heading = workbook.add_format({"bold": True, "bg_color": "#DCEAF7"})
    notes = workbook.add_worksheet("Read me")
    notes.set_column(0, 0, 34)
    notes.set_column(1, 1, 100)
    statements = (
        ("Study", "Paper-inspired LSTM comparison on matched EUR/USD inputs"),
        ("Stage", stage),
        ("Input grid", "Native 100 ms book states; no minute aggregation"),
        ("Input context", "10 seconds (100 states)"),
        ("Target", "Three-class midquote direction 30 seconds ahead"),
        ("Model", "Single-layer, 50-unit LSTM; Adam; three-class output"),
        (
            "Optimization caveat",
            "DeepLOB/TFT used AdamW; differences are not solely architectural",
        ),
        ("Feature contrast", "Transactions only versus transactions plus Level 1"),
        ("Paper reproduction", "No; individual order events are unavailable"),
        ("Statistical unit", "Trading session, not overlapping target window"),
        ("March status", "Previously inspected; retrospective, not pristine"),
        ("Trading claim", "None; this workbook reports movement forecasts"),
    )
    for row_index, (key, value) in enumerate(statements):
        notes.write(row_index, 0, key, heading)
        notes.write(row_index, 1, value)
    _write_sheet(workbook=workbook, name="Absolute scores", frame=absolute_summary)
    _write_sheet(workbook=workbook, name="Paired deltas", frame=comparisons)
    _write_sheet(workbook=workbook, name="Training details", frame=training_details)
    _write_sheet(workbook=workbook, name="LSTM sessions", frame=metrics)
    _write_sheet(workbook=workbook, name="Reference sessions", frame=reference)
    workbook.close()
    temporary.replace(path)


def _write_sheet(
    *, workbook: xlsxwriter.Workbook, name: str, frame: pl.DataFrame
) -> None:
    sheet = workbook.add_worksheet(name)
    heading = workbook.add_format({"bold": True, "bg_color": "#DCEAF7"})
    for column_index, column_name in enumerate(frame.columns):
        sheet.write(0, column_index, column_name, heading)
        sheet.set_column(
            column_index, column_index, min(36, max(15, len(column_name) + 3))
        )
    for row_index, values in enumerate(frame.iter_rows(), start=1):
        for column_index, value in enumerate(values):
            if isinstance(value, date | datetime):
                value = value.isoformat()
            sheet.write(row_index, column_index, value)
    sheet.freeze_panes(1, 0)
    sheet.autofilter(0, 0, frame.height, frame.width - 1)
