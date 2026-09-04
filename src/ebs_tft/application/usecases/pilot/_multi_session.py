"""Run day-aware native-resolution training across independent sessions."""

from __future__ import annotations

import datetime
import enum
import hashlib
import json
import platform
import time
from contextlib import closing
from pathlib import Path

import attrs
import numpy as np
import polars as pl
import sklearn
import torch

from ebs_tft.application.usecases.pilot import _checkpoint, _comparison
from ebs_tft.data.parsers import ebs_csv
from ebs_tft.data.repositories import artifact as artifact_repository
from ebs_tft.domain import model as model_domain
from ebs_tft.domain.orderbook import models as orderbook_models
from ebs_tft.domain.pilot import models as pilot_models
from ebs_tft.domain.pilot import operations as pilot_operations
from ebs_tft.domain.pilot import training as pilot_training


@attrs.frozen
class MultiSessionResult:
    """Reference durable artifacts from one multi-session development run."""

    output_dir: Path
    metrics_path: Path
    predictions_path: Path
    summary_path: Path
    terminal_summary_path: Path


@attrs.frozen
class _LoadedSession:
    specification: pilot_models.PilotSession
    states: pl.DataFrame
    source_sha256: str
    parse_audit: dict[str, int]


def run(
    *,
    specification: pilot_models.MultiSessionSpecification,
    replace_output: bool = False,
) -> MultiSessionResult:
    """Execute day-aware training and later-session development validation."""
    started = time.perf_counter()
    artifact_repository.prepare_run_directory(
        path=specification.output_dir, replace=replace_output
    )
    loaded_sessions = tuple(
        _load_session(session=item, specification=specification)
        for item in specification.all_sessions
    )
    training_loaded = loaded_sessions[:-1]
    validation_loaded = loaded_sessions[-1]
    source_hashes = {
        item.specification.trading_date.isoformat(): item.source_sha256
        for item in loaded_sessions
    }
    state_artifacts: dict[str, str] = {}
    balance_frames: list[pl.DataFrame] = []
    for index, loaded in enumerate(loaded_sessions):
        role = "training" if index < len(training_loaded) else "validation"
        date_text = loaded.specification.trading_date.isoformat()
        states_path = specification.output_dir / f"native_states_{date_text}.parquet"
        loaded.states.write_parquet(states_path)
        state_artifacts[date_text] = str(states_path)
        balance_frames.append(
            pilot_operations.target_balance(
                data=loaded.states, horizon_steps=specification.horizon_steps
            ).with_columns(
                pl.lit(role).alias("role"),
                pl.lit(loaded.specification.trading_date).alias("trading_date"),
            )
        )
    target_balance = pl.concat(balance_frames).sort(
        ["role", "trading_date", "horizon_steps"]
    )
    target_balance_path = specification.output_dir / "session_target_balance.csv"
    target_balance.write_csv(target_balance_path)

    device = model_domain.select_device(requested=specification.device)
    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pl.DataFrame] = []
    split_balance_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    dependence_rows: list[dict[str, object]] = []
    histories: dict[str, object] = {}
    model_metadata: dict[str, object] = {}
    for horizon_steps in specification.modeled_horizon_steps:
        for depth in specification.depths:
            raw_training = tuple(
                pilot_training.extract_session(
                    data=item.states,
                    trading_date=item.specification.trading_date,
                    depth=depth,
                    horizon_steps=horizon_steps,
                )
                for item in training_loaded
            )
            raw_validation = pilot_training.extract_session(
                data=validation_loaded.states,
                trading_date=validation_loaded.specification.trading_date,
                depth=depth,
                horizon_steps=horizon_steps,
            )
            scaler = pilot_training.fit_feature_scaler(sessions=raw_training)
            scaled_training = tuple(
                pilot_training.apply_feature_scaler(session=item, scaler=scaler)
                for item in raw_training
            )
            scaled_validation = pilot_training.apply_feature_scaler(
                session=raw_validation, scaler=scaler
            )
            training_corpus = pilot_training.combine_sessions(
                sessions=scaled_training,
                context_steps=specification.context_steps,
                horizon_steps=horizon_steps,
                maximum_windows=specification.maximum_training_windows,
            )
            validation_corpus = pilot_training.combine_sessions(
                sessions=(scaled_validation,),
                context_steps=specification.context_steps,
                horizon_steps=horizon_steps,
                maximum_windows=specification.maximum_validation_windows,
            )
            _write_preprocessing(
                specification=specification,
                horizon_steps=horizon_steps,
                depth=depth,
                scaler=scaler,
                training=training_corpus,
                validation=validation_corpus,
            )
            split_balance_rows.extend(
                _balance_rows(
                    corpus=training_corpus,
                    role="training",
                    horizon_steps=horizon_steps,
                    depth=depth,
                )
            )
            split_balance_rows.extend(
                _balance_rows(
                    corpus=validation_corpus,
                    role="validation",
                    horizon_steps=horizon_steps,
                    depth=depth,
                )
            )
            feature_rows.extend(
                _feature_rows(
                    sessions=scaled_training,
                    role="training",
                    horizon_steps=horizon_steps,
                    depth=depth,
                )
            )
            feature_rows.extend(
                _feature_rows(
                    sessions=(scaled_validation,),
                    role="validation",
                    horizon_steps=horizon_steps,
                    depth=depth,
                )
            )
            dependence_rows.extend(
                _dependence_rows(
                    corpus=training_corpus,
                    role="training",
                    specification=specification,
                    horizon_steps=horizon_steps,
                    depth=depth,
                )
            )
            dependence_rows.extend(
                _dependence_rows(
                    corpus=validation_corpus,
                    role="validation",
                    specification=specification,
                    horizon_steps=horizon_steps,
                    depth=depth,
                )
            )
            training_dataset = model_domain.SequenceDataset(
                lob_features=training_corpus.lob_features,
                auxiliary_features=training_corpus.auxiliary_features,
                labels=training_corpus.labels,
                target_indices=training_corpus.target_indices,
                context_steps=specification.context_steps,
            )
            validation_dataset = model_domain.SequenceDataset(
                lob_features=validation_corpus.lob_features,
                auxiliary_features=validation_corpus.auxiliary_features,
                labels=validation_corpus.labels,
                target_indices=validation_corpus.target_indices,
                context_steps=specification.context_steps,
            )
            validation_labels = validation_corpus.labels[
                validation_corpus.target_indices
            ]
            baselines = model_domain.fit_defensive_baselines(
                training=training_corpus, evaluation=validation_corpus
            )
            for model_name, probabilities in baselines.items():
                labels = probabilities.argmax(axis=1)
                metric_rows.append(
                    model_domain.direction_metric_row(
                        model_name=model_name,
                        depth=depth,
                        horizon_steps=horizon_steps,
                        seed=-1,
                        labels=validation_labels,
                        probabilities=probabilities,
                        parameter_count=0,
                    )
                )
                prediction_frames.append(
                    model_domain.direction_prediction_frame(
                        corpus=validation_corpus,
                        model_name=model_name,
                        depth=depth,
                        horizon_steps=horizon_steps,
                        seed=-1,
                        labels=labels,
                        probabilities=probabilities,
                        evaluation_role="development_validation",
                    )
                )
            for seed in specification.random_seeds:
                for model_name in specification.models:
                    _fit_neural_cell(
                        model_name=model_name,
                        seed=seed,
                        depth=depth,
                        horizon_steps=horizon_steps,
                        training_dataset=training_dataset,
                        validation_dataset=validation_dataset,
                        validation_corpus=validation_corpus,
                        specification=specification,
                        device=device,
                        source_hashes=source_hashes,
                        histories=histories,
                        model_metadata=model_metadata,
                        metric_rows=metric_rows,
                        prediction_frames=prediction_frames,
                    )

    metrics_data = pl.DataFrame(metric_rows).sort(
        ["horizon_steps", "training_mode", "model", "depth", "seed"]
    )
    metrics_path = specification.output_dir / "validation_metrics.csv"
    metrics_data.write_csv(metrics_path)
    predictions_path = specification.output_dir / "validation_predictions.parquet"
    pl.concat(prediction_frames).sort(
        [
            "horizon_steps",
            "model",
            "depth",
            "seed",
            orderbook_models.COL_TIMESTAMP,
        ]
    ).write_parquet(predictions_path)
    depth_comparison_path = specification.output_dir / "depth_comparison.csv"
    _comparison.cumulative_depth_comparison(metrics=metrics_data).write_csv(
        depth_comparison_path
    )
    split_balance_path = specification.output_dir / "split_target_balance.csv"
    pl.DataFrame(split_balance_rows).sort(
        ["horizon_steps", "depth", "role", "trading_date"]
    ).write_csv(split_balance_path)
    feature_summary_path = specification.output_dir / "session_feature_summary.csv"
    pl.DataFrame(feature_rows).sort(
        ["horizon_steps", "depth", "feature", "role", "trading_date"]
    ).write_csv(feature_summary_path)
    dependence_path = specification.output_dir / "dependence_summary.csv"
    pl.DataFrame(dependence_rows).sort(
        ["horizon_steps", "depth", "role", "trading_date"]
    ).write_csv(dependence_path)
    elapsed = time.perf_counter() - started
    summary = {
        "warning": "Multi-session development validation; not publication evidence.",
        "model_protocol_version": model_domain.MODEL_PROTOCOL_VERSION,
        "evaluation_role": "development_validation",
        "protocol": _jsonable(attrs.asdict(specification)),
        "sources": [
            {
                "path": str(item.specification.raw_path),
                "trading_date": item.specification.trading_date.isoformat(),
                "sha256": item.source_sha256,
                **item.parse_audit,
            }
            for item in loaded_sessions
        ],
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "sklearn": sklearn.__version__,
            "device": str(device),
        },
        "artifacts": {
            "native_states": state_artifacts,
            "session_target_balance": str(target_balance_path),
            "split_target_balance": str(split_balance_path),
            "session_feature_summary": str(feature_summary_path),
            "dependence_summary": str(dependence_path),
            "validation_metrics": str(metrics_path),
            "validation_predictions": str(predictions_path),
            "depth_comparison": str(depth_comparison_path),
        },
        "models": model_metadata,
        "histories": histories,
        "elapsed_seconds": elapsed,
    }
    summary_path = specification.output_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    terminal_summary_path = specification.output_dir / "terminal_summary.txt"
    terminal_summary = _terminal_summary(
        specification=specification,
        metrics=metrics_data,
        target_balance=target_balance,
        device=device,
        elapsed=elapsed,
    )
    terminal_summary_path.write_text(terminal_summary, encoding="utf-8")
    print(terminal_summary)
    return MultiSessionResult(
        output_dir=specification.output_dir,
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        summary_path=summary_path,
        terminal_summary_path=terminal_summary_path,
    )


def _load_session(
    *,
    session: pilot_models.PilotSession,
    specification: pilot_models.MultiSessionSpecification,
) -> _LoadedSession:
    audit = ebs_csv.ParseAudit()
    with closing(
        ebs_csv.parse_rows(
            path=session.raw_path,
            expected_instrument=specification.instrument,
            expected_trading_date=session.trading_date,
            audit=audit,
        )
    ) as records:
        states = pilot_operations.build_native_states(
            records=records,
            instrument=specification.instrument,
            trading_date=session.trading_date,
            grid_steps=session.grid_steps,
            maximum_staleness_steps=specification.maximum_staleness_steps,
            maximum_depth=max(specification.depths),
        )
    states = pilot_operations.add_direction_targets(
        data=states, horizon_steps=specification.horizon_steps
    )
    return _LoadedSession(
        specification=session,
        states=states,
        source_sha256=_sha256_file(path=session.raw_path),
        parse_audit={
            "physical_lines_consumed": audit.physical_lines,
            "quote_rows_consumed": audit.quote_rows,
            "deal_rows_consumed": audit.deal_rows,
        },
    )


def _fit_neural_cell(
    *,
    model_name: pilot_models.ModelName,
    seed: int,
    depth: int,
    horizon_steps: int,
    training_dataset: model_domain.SequenceDataset,
    validation_dataset: model_domain.SequenceDataset,
    validation_corpus: pilot_training.PreparedCorpus,
    specification: pilot_models.MultiSessionSpecification,
    device: torch.device,
    source_hashes: dict[str, str],
    histories: dict[str, object],
    model_metadata: dict[str, object],
    metric_rows: list[dict[str, object]],
    prediction_frames: list[pl.DataFrame],
) -> None:
    model_domain.set_random_seed(seed=seed)
    classifier = _classifier(
        model_name=model_name,
        auxiliary_size=validation_corpus.auxiliary_features.shape[1],
        hidden_size=specification.hidden_size,
    )
    key = f"{model_name.value}_unweighted_h{horizon_steps}_l{depth}_s{seed}"
    fingerprint = _fingerprint(
        specification=specification,
        model_name=model_name,
        seed=seed,
        depth=depth,
        horizon_steps=horizon_steps,
        source_hashes=source_hashes,
    )
    latest_path = specification.output_dir / f"{key}.last.pt"
    best_path = specification.output_dir / f"{key}.best.pt"
    resume_state = _checkpoint.read_latest(path=latest_path, fingerprint=fingerprint)
    result = model_domain.fit_classifier(
        classifier=classifier,
        training_data=training_dataset,
        validation_data=validation_dataset,
        device=device,
        maximum_epochs=specification.maximum_epochs,
        batch_size=specification.batch_size,
        learning_rate=specification.learning_rate,
        weight_decay=specification.weight_decay,
        early_stopping_patience=specification.early_stopping_patience,
        early_stopping_minimum_delta=specification.early_stopping_minimum_delta,
        gradient_clip_norm=specification.gradient_clip_norm,
        random_seed=seed,
        resume_state=resume_state,
        epoch_observer=lambda item: _print_epoch(
            key=key, metric=item, maximum_epochs=specification.maximum_epochs
        ),
        checkpoint_observer=lambda state: _checkpoint.write_latest(
            path=latest_path, state=state, fingerprint=fingerprint
        ),
    )
    _checkpoint.write_best(
        path=best_path,
        classifier=classifier,
        fingerprint=fingerprint,
        training_result=result,
    )
    evaluation_classifier = _classifier(
        model_name=model_name,
        auxiliary_size=validation_corpus.auxiliary_features.shape[1],
        hidden_size=specification.hidden_size,
    )
    _checkpoint.load_best(
        path=best_path, classifier=evaluation_classifier, fingerprint=fingerprint
    )
    prediction = model_domain.predict_classifier(
        classifier=evaluation_classifier,
        dataset=validation_dataset,
        device=device,
        batch_size=specification.batch_size,
    )
    parameter_count = model_domain.parameter_count(classifier=evaluation_classifier)
    histories[key] = {
        "best_epoch": result.best_epoch,
        "best_validation_log_loss": result.best_validation_log_loss,
        "epochs_completed": result.epochs_completed,
        "stop_reason": result.stop_reason,
        "epochs": [attrs.asdict(item) for item in result.history],
    }
    model_metadata[key] = {
        "parameter_count": parameter_count,
        "training_mode": "unweighted",
    }
    labels = validation_corpus.labels[validation_corpus.target_indices]
    metric_rows.append(
        model_domain.direction_metric_row(
            model_name=model_name.value,
            depth=depth,
            horizon_steps=horizon_steps,
            seed=seed,
            labels=labels,
            probabilities=prediction.probabilities,
            parameter_count=parameter_count,
        )
    )
    prediction_frames.append(
        model_domain.direction_prediction_frame(
            corpus=validation_corpus,
            model_name=model_name.value,
            depth=depth,
            horizon_steps=horizon_steps,
            seed=seed,
            labels=prediction.labels,
            probabilities=prediction.probabilities,
            evaluation_role="development_validation",
        )
    )


def _classifier(
    *, model_name: pilot_models.ModelName, auxiliary_size: int, hidden_size: int
) -> torch.nn.Module:
    return model_domain.build_direction_classifier(
        model_name=model_name.value,
        auxiliary_size=auxiliary_size,
        hidden_size=hidden_size,
    )


def _balance_rows(
    *,
    corpus: pilot_training.PreparedCorpus,
    role: str,
    horizon_steps: int,
    depth: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for offset, length, window in zip(
        corpus.session_offsets,
        corpus.session_lengths,
        corpus.session_windows,
        strict=True,
    ):
        mask = (corpus.target_indices >= offset) & (
            corpus.target_indices < offset + length
        )
        indices = corpus.target_indices[mask]
        counts = np.bincount(corpus.labels[indices], minlength=3)
        total = int(counts.sum())
        rows.append(
            {
                "role": role,
                "trading_date": window.trading_date,
                "depth": depth,
                "horizon_steps": horizon_steps,
                "horizon_milliseconds": (
                    horizon_steps * pilot_models.GRID_INTERVAL_MILLISECONDS
                ),
                "down": int(counts[0]),
                "flat": int(counts[1]),
                "up": int(counts[2]),
                "total": total,
                "flat_percentage": 100.0 * float(counts[1]) / total,
            }
        )
    return rows


def _feature_rows(
    *,
    sessions: tuple[pilot_training.RawSessionData, ...],
    role: str,
    horizon_steps: int,
    depth: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for session in sessions:
        active_lob = session.lob_features[session.observed]
        for level in range(depth):
            for feature_index, feature_name in enumerate(
                pilot_training.LOB_FEATURE_ORDER
            ):
                values = active_lob[:, level, feature_index]
                rows.append(
                    _feature_row(
                        session=session,
                        role=role,
                        horizon_steps=horizon_steps,
                        depth=depth,
                        level=level + 1,
                        feature=feature_name,
                        values=values,
                    )
                )
        active_auxiliary = session.auxiliary_features[session.observed]
        for feature_index, feature_name in enumerate(
            pilot_training.AUXILIARY_FEATURE_ORDER
        ):
            rows.append(
                _feature_row(
                    session=session,
                    role=role,
                    horizon_steps=horizon_steps,
                    depth=depth,
                    level=0,
                    feature=feature_name,
                    values=active_auxiliary[:, feature_index],
                )
            )
    return rows


def _feature_row(
    *,
    session: pilot_training.RawSessionData,
    role: str,
    horizon_steps: int,
    depth: int,
    level: int,
    feature: str,
    values: np.ndarray,
) -> dict[str, object]:
    return {
        "role": role,
        "trading_date": session.trading_date,
        "horizon_steps": horizon_steps,
        "horizon_milliseconds": (
            horizon_steps * pilot_models.GRID_INTERVAL_MILLISECONDS
        ),
        "depth": depth,
        "level": level,
        "feature": feature,
        "mean": float(values.mean()),
        "standard_deviation": float(values.std()),
    }


def _dependence_rows(
    *,
    corpus: pilot_training.PreparedCorpus,
    role: str,
    specification: pilot_models.MultiSessionSpecification,
    horizon_steps: int,
    depth: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    separation_steps = max(specification.context_steps, horizon_steps)
    for window in corpus.session_windows:
        duration = window.timestamp_to - window.timestamp_from
        span_steps = int(
            duration.total_seconds() * 1_000 / specification.state_interval_milliseconds
        )
        rows.append(
            {
                "role": role,
                "trading_date": window.trading_date,
                "depth": depth,
                "horizon_steps": horizon_steps,
                "horizon_milliseconds": (
                    horizon_steps * specification.state_interval_milliseconds
                ),
                "candidate_windows": window.candidates,
                "selected_windows": window.selected,
                "approximately_spaced_windows": span_steps // separation_steps + 1,
                "context_overlap_percentage": (
                    100.0
                    * (specification.context_steps - 1)
                    / specification.context_steps
                ),
                "target_interval_overlap_percentage": (
                    100.0 * (horizon_steps - 1) / horizon_steps
                ),
            }
        )
    return rows


def _write_preprocessing(
    *,
    specification: pilot_models.MultiSessionSpecification,
    horizon_steps: int,
    depth: int,
    scaler: pilot_training.FeatureScaler,
    training: pilot_training.PreparedCorpus,
    validation: pilot_training.PreparedCorpus,
) -> None:
    payload = {
        "state_interval_milliseconds": specification.state_interval_milliseconds,
        "horizon_milliseconds": (
            horizon_steps * specification.state_interval_milliseconds
        ),
        "depth": depth,
        "lob_feature_order": pilot_training.LOB_FEATURE_ORDER,
        "auxiliary_feature_order": pilot_training.AUXILIARY_FEATURE_ORDER,
        "lob_means": scaler.lob_means.tolist(),
        "lob_standard_deviations": scaler.lob_standard_deviations.tolist(),
        "auxiliary_means": scaler.auxiliary_means.tolist(),
        "auxiliary_standard_deviations": (
            scaler.auxiliary_standard_deviations.tolist()
        ),
        "training_sessions": [attrs.asdict(item) for item in training.session_windows],
        "validation_sessions": [
            attrs.asdict(item) for item in validation.session_windows
        ],
    }
    path = specification.output_dir / f"preprocessing_h{horizon_steps}_l{depth}.json"
    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")


def _fingerprint(
    *,
    specification: pilot_models.MultiSessionSpecification,
    model_name: pilot_models.ModelName,
    seed: int,
    depth: int,
    horizon_steps: int,
    source_hashes: dict[str, str],
) -> str:
    payload = {
        "model_protocol_version": model_domain.MODEL_PROTOCOL_VERSION,
        "protocol": _jsonable(attrs.asdict(specification)),
        "model": model_name.value,
        "seed": seed,
        "depth": depth,
        "horizon_steps": horizon_steps,
        "source_hashes": source_hashes,
        "torch": torch.__version__,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _sha256_file(*, path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _print_epoch(
    *, key: str, metric: model_domain.EpochMetric, maximum_epochs: int
) -> None:
    marker = " best" if metric.improved else ""
    print(
        f"[{key}] epoch={metric.epoch}/{maximum_epochs} "
        f"validation={metric.validation_index} step={metric.optimizer_step} "
        f"train_loss={metric.training_loss:.6f} "
        f"validation_log_loss={metric.validation_log_loss:.6f} "
        f"gradient_norm={metric.gradient_norm:.4f}{marker}",
        flush=True,
    )


def _terminal_summary(
    *,
    specification: pilot_models.MultiSessionSpecification,
    metrics: pl.DataFrame,
    target_balance: pl.DataFrame,
    device: torch.device,
    elapsed: float,
) -> str:
    return "\n".join(
        (
            "EBS day-aware multi-session development completed",
            "WARNING: development validation only; not publication evidence.",
            f"instrument={specification.instrument.value}",
            "training_dates="
            + ",".join(
                item.trading_date.isoformat()
                for item in specification.training_sessions
            ),
            f"validation_date={specification.validation_session.trading_date.isoformat()}",
            f"device={device}",
            f"elapsed_seconds={elapsed:.2f}",
            "",
            "Per-session target balance:",
            str(target_balance),
            "",
            "Development-validation metrics:",
            str(metrics),
            "",
            f"Outputs: {specification.output_dir}",
        )
    )


def _jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    if isinstance(value, datetime.date | datetime.datetime):
        return value.isoformat()
    if isinstance(value, enum.Enum):
        return value.value
    return value
