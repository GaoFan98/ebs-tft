"""Orchestrate the bounded native-resolution local feasibility pilot."""

from __future__ import annotations

import datetime
import enum
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
from sklearn import linear_model, metrics

from ebs_tft.application.usecases.pilot import _checkpoint, _comparison
from ebs_tft.data.parsers import ebs_csv
from ebs_tft.data.repositories import artifact as artifact_repository
from ebs_tft.domain import model as model_domain
from ebs_tft.domain.orderbook import models as orderbook_models
from ebs_tft.domain.pilot import models as pilot_models
from ebs_tft.domain.pilot import operations as pilot_operations


@attrs.frozen
class PilotResult:
    """Reference the durable outputs produced by one local pilot run."""

    output_dir: Path
    native_states_path: Path
    metrics_path: Path
    predictions_path: Path
    summary_path: Path
    terminal_summary_path: Path


@attrs.frozen
class _PreparedData:
    lob_features: np.ndarray
    auxiliary_features: np.ndarray
    labels: np.ndarray
    timestamps: np.ndarray
    mid_prices: np.ndarray
    training_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    training_candidates: int
    validation_candidates: int
    test_candidates: int
    lob_means: np.ndarray
    lob_standard_deviations: np.ndarray
    auxiliary_means: np.ndarray
    auxiliary_standard_deviations: np.ndarray


def run(
    *, specification: pilot_models.PilotSpecification, replace_output: bool = False
) -> PilotResult:
    """Execute the complete bounded pilot and persist inspectable artifacts."""
    started = time.perf_counter()
    source_sha256 = _sha256_file(path=specification.raw_path)
    artifact_repository.prepare_run_directory(
        path=specification.output_dir, replace=replace_output
    )
    parse_audit = ebs_csv.ParseAudit()
    with closing(
        ebs_csv.parse_rows(
            path=specification.raw_path,
            expected_instrument=specification.instrument,
            expected_trading_date=specification.trading_date,
            audit=parse_audit,
        )
    ) as records:
        states = pilot_operations.build_native_states(
            records=records,
            instrument=specification.instrument,
            trading_date=specification.trading_date,
            grid_steps=specification.grid_steps,
            maximum_staleness_steps=specification.maximum_staleness_steps,
            maximum_depth=max(specification.depths),
        )
    states = pilot_operations.add_direction_targets(
        data=states, horizon_steps=specification.horizon_steps
    )
    native_states_path = specification.output_dir / "native_states.parquet"
    states.write_parquet(native_states_path)
    balances = pilot_operations.target_balance(
        data=states, horizon_steps=specification.horizon_steps
    )
    balances.write_csv(specification.output_dir / "target_balance.csv")

    device = model_domain.select_device(requested=specification.device)
    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pl.DataFrame] = []
    histories: dict[str, object] = {}
    model_metadata: dict[str, object] = {}
    for horizon_steps in specification.modeled_horizon_steps:
        for depth in specification.depths:
            prepared = _prepare_data(
                states=states,
                specification=specification,
                depth=depth,
                horizon_steps=horizon_steps,
            )
            _write_preprocessing(
                prepared=prepared,
                specification=specification,
                depth=depth,
                horizon_steps=horizon_steps,
            )
            datasets = _datasets(prepared=prepared, specification=specification)
            baseline_predictions = _fit_baselines(prepared=prepared)
            for model_name, probabilities in baseline_predictions.items():
                labels = probabilities.argmax(axis=1)
                metric_rows.append(
                    _metric_row(
                        model_name=model_name,
                        depth=depth,
                        horizon_steps=horizon_steps,
                        seed=-1,
                        training_mode="baseline",
                        labels=prepared.labels[prepared.test_indices],
                        probabilities=probabilities,
                        parameter_count=0,
                    )
                )
                prediction_frames.append(
                    _prediction_frame(
                        prepared=prepared,
                        model_name=model_name,
                        depth=depth,
                        horizon_steps=horizon_steps,
                        seed=-1,
                        training_mode="baseline",
                        labels=labels,
                        probabilities=probabilities,
                    )
                )

            training_modes = ["unweighted"]
            if horizon_steps in specification.class_weighted_horizon_steps:
                training_modes.append("class_weighted")
            for training_mode in training_modes:
                seeds = (
                    specification.random_seeds
                    if training_mode == "unweighted"
                    else specification.random_seeds[:1]
                )
                for seed in seeds:
                    for model_name in specification.models:
                        _fit_neural_cell(
                            model_name=model_name,
                            training_mode=training_mode,
                            seed=seed,
                            depth=depth,
                            horizon_steps=horizon_steps,
                            prepared=prepared,
                            datasets=datasets,
                            specification=specification,
                            device=device,
                            histories=histories,
                            model_metadata=model_metadata,
                            metric_rows=metric_rows,
                            prediction_frames=prediction_frames,
                            source_sha256=source_sha256,
                        )

    metrics_data = pl.DataFrame(metric_rows).sort(
        ["horizon_steps", "training_mode", "model", "depth", "seed"]
    )
    metrics_path = specification.output_dir / "metrics.csv"
    metrics_data.write_csv(metrics_path)
    depth_comparison_path = specification.output_dir / "depth_comparison.csv"
    _comparison.cumulative_depth_comparison(metrics=metrics_data).write_csv(
        depth_comparison_path
    )
    predictions_path = specification.output_dir / "predictions.parquet"
    pl.concat(prediction_frames).sort(
        [
            "horizon_steps",
            "model",
            "training_mode",
            "depth",
            "seed",
            orderbook_models.COL_TIMESTAMP,
        ]
    ).write_parquet(predictions_path)
    elapsed = time.perf_counter() - started
    summary = {
        "warning": (
            "Engineering pilot only: bounded windows, dates, seeds, and no "
            "publication inference."
        ),
        "source": {
            "path": str(specification.raw_path),
            "instrument": specification.instrument.value,
            "trading_date": specification.trading_date.isoformat(),
            "physical_lines_consumed": parse_audit.physical_lines,
            "quote_rows_consumed": parse_audit.quote_rows,
            "deal_rows_consumed": parse_audit.deal_rows,
            "sha256": source_sha256,
        },
        "protocol": _jsonable(attrs.asdict(specification)),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "sklearn": sklearn.__version__,
            "device": str(device),
        },
        "artifacts": {
            "native_states": str(native_states_path),
            "target_balance": str(specification.output_dir / "target_balance.csv"),
            "metrics": str(metrics_path),
            "depth_comparison": str(depth_comparison_path),
            "predictions": str(predictions_path),
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
        balances=balances,
        metrics_data=metrics_data,
        device=device,
        elapsed=elapsed,
        output_dir=specification.output_dir,
    )
    terminal_summary_path.write_text(terminal_summary, encoding="utf-8")
    print(terminal_summary)
    return PilotResult(
        output_dir=specification.output_dir,
        native_states_path=native_states_path,
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        summary_path=summary_path,
        terminal_summary_path=terminal_summary_path,
    )


def _prepare_data(
    *,
    states: pl.DataFrame,
    specification: pilot_models.PilotSpecification,
    depth: int,
    horizon_steps: int,
) -> _PreparedData:
    row_count = states.height
    mid = states[orderbook_models.COL_MID_PRICE].fill_null(float("nan")).to_numpy()
    # Depth-specific tensors contain no padded future levels. This makes an L1
    # input structurally incapable of carrying L2-L10 values or learned padding.
    lob = np.zeros((row_count, depth, 6), dtype=np.float32)
    for level in range(1, depth + 1):
        level_index = level - 1
        bid_price = _numeric_column(
            data=states, name=orderbook_models.bid_price_col(level=level)
        )
        ask_price = _numeric_column(
            data=states, name=orderbook_models.ask_price_col(level=level)
        )
        valid_mid = np.isfinite(mid) & (mid != 0)
        lob[valid_mid, level_index, 0] = (
            (bid_price[valid_mid] / mid[valid_mid]) - 1.0
        ) * 10_000
        lob[valid_mid, level_index, 1] = (
            (ask_price[valid_mid] / mid[valid_mid]) - 1.0
        ) * 10_000
        lob[:, level_index, 2] = np.log1p(
            _numeric_column(
                data=states, name=orderbook_models.bid_size_col(level=level)
            )
        )
        lob[:, level_index, 3] = np.log1p(
            _numeric_column(
                data=states, name=orderbook_models.ask_size_col(level=level)
            )
        )
        lob[:, level_index, 4] = np.log1p(
            _numeric_column(
                data=states,
                name=orderbook_models.bid_order_count_col(level=level),
            )
        )
        lob[:, level_index, 5] = np.log1p(
            _numeric_column(
                data=states,
                name=orderbook_models.ask_order_count_col(level=level),
            )
        )
    auxiliary = np.column_stack(
        (
            np.log1p(
                _numeric_column(data=states, name=orderbook_models.COL_BUY_VOLUME)
            ),
            np.log1p(
                _numeric_column(data=states, name=orderbook_models.COL_SELL_VOLUME)
            ),
            np.log1p(
                _numeric_column(data=states, name=orderbook_models.COL_TRADE_COUNT)
            ),
            _numeric_column(data=states, name=orderbook_models.COL_DEAL_FLOW_IMBALANCE),
            _numeric_column(data=states, name=orderbook_models.COL_DEALS_OBSERVED),
            _numeric_column(data=states, name=pilot_models.COL_BID_UPDATED),
            _numeric_column(data=states, name=pilot_models.COL_ASK_UPDATED),
            _numeric_column(data=states, name=pilot_models.COL_BID_AGE_MILLISECONDS),
            _numeric_column(data=states, name=pilot_models.COL_ASK_AGE_MILLISECONDS),
            np.nan_to_num(
                _numeric_column(data=states, name=orderbook_models.COL_SPREAD)
                / mid
                * 10_000,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ),
        )
    ).astype(np.float32)
    target_column = pilot_models.direction_column(horizon_steps=horizon_steps)
    raw_labels = states[target_column].fill_null(-99).to_numpy()
    labels = np.where(raw_labels == -99, -1, raw_labels + 1).astype(np.int64)
    observed = states[orderbook_models.COL_BOOK_OBSERVED].to_numpy()
    split_boundaries = (int(row_count * 0.6), int(row_count * 0.8), row_count)
    training_indices, training_candidates = _candidate_indices(
        labels=labels,
        observed=observed,
        start=0,
        end=split_boundaries[0],
        context_steps=specification.context_steps,
        maximum=specification.maximum_training_windows,
        horizon_steps=horizon_steps,
    )
    validation_indices, validation_candidates = _candidate_indices(
        labels=labels,
        observed=observed,
        start=split_boundaries[0],
        end=split_boundaries[1],
        context_steps=specification.context_steps,
        maximum=specification.maximum_validation_windows,
        horizon_steps=horizon_steps,
    )
    test_indices, test_candidates = _candidate_indices(
        labels=labels,
        observed=observed,
        start=split_boundaries[1],
        end=split_boundaries[2],
        context_steps=specification.context_steps,
        maximum=specification.maximum_test_windows,
        horizon_steps=horizon_steps,
    )
    lob_means, lob_standard_deviations = _standardize_training_only(
        values=lob,
        training_mask=observed[: split_boundaries[0]],
        active_depth=depth,
    )
    auxiliary_means, auxiliary_standard_deviations = (
        _standardize_auxiliary_training_only(
            values=auxiliary, training_mask=observed[: split_boundaries[0]]
        )
    )
    return _PreparedData(
        lob_features=lob,
        auxiliary_features=auxiliary,
        labels=labels,
        timestamps=states[orderbook_models.COL_TIMESTAMP].to_numpy(),
        mid_prices=mid,
        training_indices=training_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
        training_candidates=training_candidates,
        validation_candidates=validation_candidates,
        test_candidates=test_candidates,
        lob_means=lob_means,
        lob_standard_deviations=lob_standard_deviations,
        auxiliary_means=auxiliary_means,
        auxiliary_standard_deviations=auxiliary_standard_deviations,
    )


def _candidate_indices(
    *,
    labels: np.ndarray,
    observed: np.ndarray,
    start: int,
    end: int,
    context_steps: int,
    maximum: int | None,
    horizon_steps: int,
) -> tuple[np.ndarray, int]:
    candidates: list[int] = []
    first = start + context_steps - 1
    for target_index in range(first, end - horizon_steps):
        window_start = target_index - context_steps + 1
        window_is_observed = bool(observed[window_start : target_index + 1].all())
        if labels[target_index] >= 0 and window_is_observed:
            candidates.append(target_index)
    if not candidates:
        raise ValueError(
            f"no valid pilot windows in chronological interval {start}:{end}"
        )
    candidate_array = np.asarray(candidates, dtype=np.int64)
    if maximum is None or len(candidates) <= maximum:
        return candidate_array, len(candidates)
    positions = np.linspace(0, len(candidates) - 1, num=maximum, dtype=np.int64)
    return candidate_array[positions], len(candidates)


def _standardize_training_only(
    *, values: np.ndarray, training_mask: np.ndarray, active_depth: int
) -> tuple[np.ndarray, np.ndarray]:
    active = values[: len(training_mask), :active_depth][training_mask]
    means = active.mean(axis=(0, 1), keepdims=True)
    deviations = active.std(axis=(0, 1), keepdims=True)
    deviations[deviations < 1e-6] = 1.0
    values[:, :active_depth] = (values[:, :active_depth] - means) / deviations
    return means, deviations


def _standardize_auxiliary_training_only(
    *, values: np.ndarray, training_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    active = values[: len(training_mask)][training_mask]
    means = active.mean(axis=0, keepdims=True)
    deviations = active.std(axis=0, keepdims=True)
    deviations[deviations < 1e-6] = 1.0
    values[:] = (values - means) / deviations
    return means, deviations


def _datasets(
    *, prepared: _PreparedData, specification: pilot_models.PilotSpecification
) -> dict[str, model_domain.SequenceDataset]:
    return {
        "training": model_domain.SequenceDataset(
            lob_features=prepared.lob_features,
            auxiliary_features=prepared.auxiliary_features,
            labels=prepared.labels,
            context_steps=specification.context_steps,
            target_indices=prepared.training_indices,
        ),
        "validation": model_domain.SequenceDataset(
            lob_features=prepared.lob_features,
            auxiliary_features=prepared.auxiliary_features,
            labels=prepared.labels,
            context_steps=specification.context_steps,
            target_indices=prepared.validation_indices,
        ),
        "test": model_domain.SequenceDataset(
            lob_features=prepared.lob_features,
            auxiliary_features=prepared.auxiliary_features,
            labels=prepared.labels,
            context_steps=specification.context_steps,
            target_indices=prepared.test_indices,
        ),
    }


def _write_preprocessing(
    *,
    prepared: _PreparedData,
    specification: pilot_models.PilotSpecification,
    depth: int,
    horizon_steps: int,
) -> None:
    payload = {
        "state_interval_milliseconds": specification.state_interval_milliseconds,
        "horizon_milliseconds": (
            horizon_steps * specification.state_interval_milliseconds
        ),
        "depth": depth,
        "lob_feature_order": [
            "bid_price_basis_points_from_mid",
            "ask_price_basis_points_from_mid",
            "log_bid_size",
            "log_ask_size",
            "log_bid_order_count",
            "log_ask_order_count",
        ],
        "auxiliary_feature_order": [
            "log_buy_volume",
            "log_sell_volume",
            "log_trade_count",
            "deal_flow_imbalance",
            "deals_observed",
            "bid_updated",
            "ask_updated",
            "bid_age_milliseconds",
            "ask_age_milliseconds",
            "spread_basis_points",
        ],
        "lob_means": prepared.lob_means.tolist(),
        "lob_standard_deviations": (prepared.lob_standard_deviations.tolist()),
        "auxiliary_means": prepared.auxiliary_means.tolist(),
        "auxiliary_standard_deviations": (
            prepared.auxiliary_standard_deviations.tolist()
        ),
        "windows": {
            "training": _window_summary(
                timestamps=prepared.timestamps,
                indices=prepared.training_indices,
                candidates=prepared.training_candidates,
            ),
            "validation": _window_summary(
                timestamps=prepared.timestamps,
                indices=prepared.validation_indices,
                candidates=prepared.validation_candidates,
            ),
            "test": _window_summary(
                timestamps=prepared.timestamps,
                indices=prepared.test_indices,
                candidates=prepared.test_candidates,
            ),
        },
    }
    path = specification.output_dir / f"preprocessing_h{horizon_steps}_l{depth}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _window_summary(
    *, timestamps: np.ndarray, indices: np.ndarray, candidates: int
) -> dict[str, object]:
    return {
        "candidates": candidates,
        "selected": len(indices),
        "selection_rate": len(indices) / candidates,
        "timestamp_from": str(timestamps[indices[0]]),
        "timestamp_to": str(timestamps[indices[-1]]),
    }


def _classifier(
    *,
    model_name: pilot_models.ModelName,
    prepared: _PreparedData,
    hidden_size: int,
) -> torch.nn.Module:
    if model_name is pilot_models.ModelName.DEEP_LOB:
        return model_domain.DeepLobDirectionClassifier(
            auxiliary_size=prepared.auxiliary_features.shape[1],
            hidden_size=hidden_size,
        )
    if model_name is pilot_models.ModelName.TFT:
        return model_domain.TftDirectionClassifier(
            input_size=(
                prepared.lob_features.shape[1] * prepared.lob_features.shape[2]
                + prepared.auxiliary_features.shape[1]
            ),
            hidden_size=hidden_size,
        )
    raise ValueError(f"unsupported pilot model: {model_name}")


def _fit_neural_cell(
    *,
    model_name: pilot_models.ModelName,
    training_mode: str,
    seed: int,
    depth: int,
    horizon_steps: int,
    prepared: _PreparedData,
    datasets: dict[str, model_domain.SequenceDataset],
    specification: pilot_models.PilotSpecification,
    device: torch.device,
    histories: dict[str, object],
    model_metadata: dict[str, object],
    metric_rows: list[dict[str, object]],
    prediction_frames: list[pl.DataFrame],
    source_sha256: str,
) -> None:
    model_domain.set_random_seed(seed=seed)
    classifier = _classifier(
        model_name=model_name,
        prepared=prepared,
        hidden_size=specification.hidden_size,
    )
    class_weights = (
        _balanced_class_weights(labels=prepared.labels[prepared.training_indices])
        if training_mode == "class_weighted"
        else None
    )
    key = f"{model_name.value}_{training_mode}_h{horizon_steps}_l{depth}_s{seed}"
    fingerprint = _experiment_fingerprint(
        specification=specification,
        model_name=model_name,
        training_mode=training_mode,
        horizon_steps=horizon_steps,
        depth=depth,
        seed=seed,
        source_sha256=source_sha256,
    )
    latest_path = specification.output_dir / f"{key}.last.pt"
    best_path = specification.output_dir / f"{key}.best.pt"
    resume_state = _checkpoint.read_latest(path=latest_path, fingerprint=fingerprint)
    training_result = model_domain.fit_classifier(
        classifier=classifier,
        training_data=datasets["training"],
        validation_data=datasets["validation"],
        device=device,
        maximum_epochs=specification.maximum_epochs,
        batch_size=specification.batch_size,
        learning_rate=specification.learning_rate,
        early_stopping_patience=specification.early_stopping_patience,
        early_stopping_minimum_delta=(specification.early_stopping_minimum_delta),
        gradient_clip_norm=specification.gradient_clip_norm,
        random_seed=seed,
        class_weights=class_weights,
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
        training_result=training_result,
    )
    evaluation_classifier = _classifier(
        model_name=model_name,
        prepared=prepared,
        hidden_size=specification.hidden_size,
    )
    _checkpoint.load_best(
        path=best_path,
        classifier=evaluation_classifier,
        fingerprint=fingerprint,
    )
    prediction = model_domain.predict_classifier(
        classifier=evaluation_classifier,
        dataset=datasets["test"],
        device=device,
        batch_size=specification.batch_size,
    )
    parameters = model_domain.parameter_count(classifier=evaluation_classifier)
    histories[key] = {
        "best_epoch": training_result.best_epoch,
        "best_validation_log_loss": training_result.best_validation_log_loss,
        "epochs_completed": training_result.epochs_completed,
        "stop_reason": training_result.stop_reason,
        "epochs": [attrs.asdict(item) for item in training_result.history],
    }
    model_metadata[key] = {
        "parameter_count": parameters,
        "training_mode": training_mode,
        "description": _model_description(model_name=model_name),
    }
    metric_rows.append(
        _metric_row(
            model_name=model_name.value,
            depth=depth,
            horizon_steps=horizon_steps,
            seed=seed,
            training_mode=training_mode,
            labels=prepared.labels[prepared.test_indices],
            probabilities=prediction.probabilities,
            parameter_count=parameters,
        )
    )
    prediction_frames.append(
        _prediction_frame(
            prepared=prepared,
            model_name=model_name.value,
            depth=depth,
            horizon_steps=horizon_steps,
            seed=seed,
            training_mode=training_mode,
            labels=prediction.labels,
            probabilities=prediction.probabilities,
        )
    )


def _balanced_class_weights(*, labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels, minlength=3).astype(np.float32)
    if np.any(counts == 0):
        raise ValueError("class-weighted pilot requires every class in training")
    return cast(np.ndarray, len(labels) / (3.0 * counts))


def _fit_baselines(*, prepared: _PreparedData) -> dict[str, np.ndarray]:
    training_labels = prepared.labels[prepared.training_indices]
    test_labels = prepared.labels[prepared.test_indices]
    majority_class = int(np.bincount(training_labels, minlength=3).argmax())
    majority = np.zeros((len(test_labels), 3), dtype=np.float64)
    majority[:, majority_class] = 1.0
    class_probabilities = np.bincount(training_labels, minlength=3).astype(np.float64)
    class_probabilities /= class_probabilities.sum()
    empirical_prior = np.tile(class_probabilities, (len(test_labels), 1))
    training_features = np.column_stack(
        (
            prepared.lob_features[prepared.training_indices].reshape(
                len(prepared.training_indices), -1
            ),
            prepared.auxiliary_features[prepared.training_indices],
        )
    )
    test_features = np.column_stack(
        (
            prepared.lob_features[prepared.test_indices].reshape(
                len(prepared.test_indices), -1
            ),
            prepared.auxiliary_features[prepared.test_indices],
        )
    )
    logistic = linear_model.LogisticRegression(
        max_iter=1_000,
        random_state=0,
        class_weight="balanced",
    )
    logistic.fit(training_features, training_labels)
    raw_probabilities = logistic.predict_proba(test_features)
    logistic_probabilities = np.zeros((len(test_labels), 3), dtype=np.float64)
    for source_index, class_label in enumerate(logistic.classes_):
        logistic_probabilities[:, int(class_label)] = raw_probabilities[:, source_index]
    selected_mid_prices = prepared.mid_prices[prepared.test_indices]
    previous_indices = np.maximum(prepared.test_indices - 1, 0)
    previous_mid_prices = prepared.mid_prices[previous_indices]
    last_move_classes = (
        np.sign(selected_mid_prices - previous_mid_prices).astype(np.int64) + 1
    )
    persistence = np.zeros((len(test_labels), 3), dtype=np.float64)
    persistence[np.arange(len(test_labels)), last_move_classes] = 1.0
    return {
        "empirical_prior": empirical_prior,
        "last_move": persistence,
        "majority": majority,
        "logistic": logistic_probabilities,
    }


def _metric_row(
    *,
    model_name: str,
    depth: int,
    horizon_steps: int,
    seed: int,
    training_mode: str,
    labels: np.ndarray,
    probabilities: np.ndarray,
    parameter_count: int,
) -> dict[str, object]:
    probabilities = _normalized_probabilities(probabilities=probabilities)
    predictions = probabilities.argmax(axis=1)
    precision = metrics.precision_score(
        labels,
        predictions,
        labels=[0, 1, 2],
        average=None,
        zero_division=0,
    )
    confusion = metrics.confusion_matrix(labels, predictions, labels=[0, 1, 2])
    one_hot = np.eye(3, dtype=np.float64)[labels]
    return {
        "model": model_name,
        "depth": depth,
        "horizon_steps": horizon_steps,
        "horizon_milliseconds": (
            horizon_steps * pilot_models.GRID_INTERVAL_MILLISECONDS
        ),
        "seed": seed,
        "training_mode": training_mode,
        "observations": len(labels),
        "accuracy": metrics.accuracy_score(labels, predictions),
        "balanced_accuracy": metrics.balanced_accuracy_score(labels, predictions),
        "macro_f1": metrics.f1_score(
            labels, predictions, labels=[0, 1, 2], average="macro", zero_division=0
        ),
        "weighted_f1": metrics.f1_score(
            labels,
            predictions,
            labels=[0, 1, 2],
            average="weighted",
            zero_division=0,
        ),
        "mcc": metrics.matthews_corrcoef(labels, predictions),
        "log_loss": metrics.log_loss(labels, probabilities, labels=[0, 1, 2]),
        "multiclass_brier": float(
            np.square(probabilities - one_hot).sum(axis=1).mean()
        ),
        "calibration_error": _expected_calibration_error(
            labels=labels, probabilities=probabilities
        ),
        "prediction_entropy": float(
            (
                -(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0))).sum(
                    axis=1
                )
            ).mean()
        ),
        "precision_down": precision[0],
        "precision_flat": precision[1],
        "precision_up": precision[2],
        "recall_down": metrics.recall_score(
            labels, predictions, labels=[0], average="macro", zero_division=0
        ),
        "recall_flat": metrics.recall_score(
            labels, predictions, labels=[1], average="macro", zero_division=0
        ),
        "recall_up": metrics.recall_score(
            labels, predictions, labels=[2], average="macro", zero_division=0
        ),
        "confusion_down_down": confusion[0, 0],
        "confusion_down_flat": confusion[0, 1],
        "confusion_down_up": confusion[0, 2],
        "confusion_flat_down": confusion[1, 0],
        "confusion_flat_flat": confusion[1, 1],
        "confusion_flat_up": confusion[1, 2],
        "confusion_up_down": confusion[2, 0],
        "confusion_up_flat": confusion[2, 1],
        "confusion_up_up": confusion[2, 2],
        "parameter_count": parameter_count,
    }


def _expected_calibration_error(
    *, labels: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> float:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index in range(bins):
        upper_inclusive = index == bins - 1
        mask = (confidence >= edges[index]) & (
            confidence <= edges[index + 1]
            if upper_inclusive
            else confidence < edges[index + 1]
        )
        if not mask.any():
            continue
        error += float(mask.mean()) * abs(
            float(correct[mask].mean()) - float(confidence[mask].mean())
        )
    return error


def _prediction_frame(
    *,
    prepared: _PreparedData,
    model_name: str,
    depth: int,
    horizon_steps: int,
    seed: int,
    training_mode: str,
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> pl.DataFrame:
    probabilities = _normalized_probabilities(probabilities=probabilities)
    true_labels = prepared.labels[prepared.test_indices] - 1
    return pl.DataFrame(
        {
            orderbook_models.COL_TIMESTAMP: prepared.timestamps[prepared.test_indices],
            "model": [model_name] * len(labels),
            "depth": [depth] * len(labels),
            "horizon_steps": [horizon_steps] * len(labels),
            "horizon_milliseconds": [
                horizon_steps * pilot_models.GRID_INTERVAL_MILLISECONDS
            ]
            * len(labels),
            "seed": [seed] * len(labels),
            "training_mode": [training_mode] * len(labels),
            "true_direction": true_labels,
            "predicted_direction": labels - 1,
            "probability_down": probabilities[:, 0],
            "probability_flat": probabilities[:, 1],
            "probability_up": probabilities[:, 2],
        }
    )


def _numeric_column(*, data: pl.DataFrame, name: str) -> np.ndarray:
    return data[name].cast(pl.Float64).fill_null(0.0).to_numpy()


def _normalized_probabilities(*, probabilities: np.ndarray) -> np.ndarray:
    normalized = probabilities.astype(np.float64, copy=True)
    row_sums = normalized.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("probability rows must have positive mass")
    return cast(np.ndarray, normalized / row_sums)


def _model_description(*, model_name: pilot_models.ModelName) -> str:
    if model_name is pilot_models.ModelName.DEEP_LOB:
        return (
            "EBS DeepLOB direction adapter with shared level encoding, spatial "
            "convolution, temporal Inception branches, an auxiliary transaction "
            "branch, an LSTM, and a three-class head."
        )
    return (
        "EBS TFT direction adapter with per-variable encoders, variable selection, "
        "LSTM recurrence, gated residual networks, causal attention, and a "
        "three-class head."
    )


def _print_epoch(
    *, key: str, metric: model_domain.EpochMetric, maximum_epochs: int
) -> None:
    marker = " best" if metric.improved else ""
    print(
        f"[{key}] epoch={metric.epoch}/{maximum_epochs} "
        f"train_loss={metric.training_loss:.6f} "
        f"validation_log_loss={metric.validation_log_loss:.6f} "
        f"gradient_norm={metric.gradient_norm:.4f}{marker}",
        flush=True,
    )


def _experiment_fingerprint(
    *,
    specification: pilot_models.PilotSpecification,
    model_name: pilot_models.ModelName,
    training_mode: str,
    horizon_steps: int,
    depth: int,
    seed: int,
    source_sha256: str,
) -> str:
    payload = {
        "protocol": _jsonable(attrs.asdict(specification)),
        "model": model_name.value,
        "training_mode": training_mode,
        "horizon_steps": horizon_steps,
        "depth": depth,
        "seed": seed,
        "source_sha256": source_sha256,
        "torch": torch.__version__,
    }
    encoded = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(*, path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _terminal_summary(
    *,
    specification: pilot_models.PilotSpecification,
    balances: pl.DataFrame,
    metrics_data: pl.DataFrame,
    device: torch.device,
    elapsed: float,
    output_dir: Path,
) -> str:
    return "\n".join(
        (
            "EBS native-resolution local pilot completed",
            "WARNING: engineering pilot only; not publication evidence.",
            f"instrument={specification.instrument.value}",
            f"trading_date={specification.trading_date.isoformat()}",
            f"grid_steps={specification.grid_steps}",
            f"device={device}",
            f"elapsed_seconds={elapsed:.2f}",
            "",
            "Target balance:",
            str(balances),
            "",
            "Metrics:",
            str(
                metrics_data.sort(
                    [
                        "horizon_steps",
                        "training_mode",
                        "model",
                        "depth",
                        "seed",
                    ]
                )
            ),
            "",
            f"Outputs: {output_dir}",
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
