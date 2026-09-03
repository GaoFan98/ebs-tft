"""Evaluate direction models against defensive probability baselines."""

from __future__ import annotations

from typing import cast

import numpy as np
import polars as pl
from sklearn import linear_model, metrics

from ebs_tft.domain.orderbook import models as orderbook_models
from ebs_tft.domain.pilot import models as pilot_models
from ebs_tft.domain.pilot import training as pilot_training


def fit_defensive_baselines(
    *,
    training: pilot_training.PreparedCorpus,
    evaluation: pilot_training.PreparedCorpus,
) -> dict[str, np.ndarray]:
    """Return four defensive baseline probability matrices."""
    training_labels = training.labels[training.target_indices]
    evaluation_labels = evaluation.labels[evaluation.target_indices]
    majority_class = int(np.bincount(training_labels, minlength=3).argmax())
    majority = np.zeros((len(evaluation_labels), 3), dtype=np.float64)
    majority[:, majority_class] = 1.0
    class_probabilities = np.bincount(training_labels, minlength=3).astype(np.float64)
    class_probabilities /= class_probabilities.sum()
    empirical_prior = np.tile(class_probabilities, (len(evaluation_labels), 1))
    training_features = _current_features(
        corpus=training, indices=training.target_indices
    )
    evaluation_features = _current_features(
        corpus=evaluation, indices=evaluation.target_indices
    )
    logistic = linear_model.LogisticRegression(
        max_iter=1_000,
        random_state=0,
        class_weight="balanced",
    )
    logistic.fit(training_features, training_labels)
    raw_probabilities = logistic.predict_proba(evaluation_features)
    logistic_probabilities = np.zeros((len(evaluation_labels), 3), dtype=np.float64)
    for source_index, class_label in enumerate(logistic.classes_):
        logistic_probabilities[:, int(class_label)] = raw_probabilities[:, source_index]
    selected_mid_prices = evaluation.mid_prices[evaluation.target_indices]
    previous_mid_prices = evaluation.mid_prices[evaluation.target_indices - 1]
    last_move_classes = (
        np.sign(selected_mid_prices - previous_mid_prices).astype(np.int64) + 1
    )
    persistence = np.zeros((len(evaluation_labels), 3), dtype=np.float64)
    persistence[np.arange(len(evaluation_labels)), last_move_classes] = 1.0
    return {
        "empirical_prior": empirical_prior,
        "last_move": persistence,
        "majority": majority,
        "logistic": logistic_probabilities,
    }


def direction_metric_row(
    *,
    model_name: str,
    depth: int,
    horizon_steps: int,
    seed: int,
    labels: np.ndarray,
    probabilities: np.ndarray,
    parameter_count: int,
) -> dict[str, object]:
    """Return the complete approved metric row for one evaluation cell."""
    probabilities = normalized_probabilities(probabilities=probabilities)
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
        "training_mode": "unweighted" if seed >= 0 else "baseline",
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
        "calibration_error": expected_calibration_error(
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


def direction_prediction_frame(
    *,
    corpus: pilot_training.PreparedCorpus,
    model_name: str,
    depth: int,
    horizon_steps: int,
    seed: int,
    labels: np.ndarray,
    probabilities: np.ndarray,
    evaluation_role: str,
) -> pl.DataFrame:
    """Return timestamp-aligned development-validation predictions."""
    probabilities = normalized_probabilities(probabilities=probabilities)
    true_labels = corpus.labels[corpus.target_indices] - 1
    return pl.DataFrame(
        {
            orderbook_models.COL_TIMESTAMP: corpus.timestamps[corpus.target_indices],
            "evaluation_role": [evaluation_role] * len(labels),
            "model": [model_name] * len(labels),
            "depth": [depth] * len(labels),
            "horizon_steps": [horizon_steps] * len(labels),
            "horizon_milliseconds": [
                horizon_steps * pilot_models.GRID_INTERVAL_MILLISECONDS
            ]
            * len(labels),
            "seed": [seed] * len(labels),
            "training_mode": ["unweighted" if seed >= 0 else "baseline"] * len(labels),
            "true_direction": true_labels,
            "predicted_direction": labels - 1,
            "probability_down": probabilities[:, 0],
            "probability_flat": probabilities[:, 1],
            "probability_up": probabilities[:, 2],
        }
    )


def normalized_probabilities(*, probabilities: np.ndarray) -> np.ndarray:
    """Return finite three-class probabilities whose rows sum to one."""
    normalized = probabilities.astype(np.float64, copy=True)
    if (
        normalized.ndim != 2
        or normalized.shape[1] != 3
        or not np.isfinite(normalized).all()
        or (normalized < 0).any()
    ):
        raise ValueError("probabilities must be finite non-negative three-class rows")
    row_sums = normalized.sum(axis=1, keepdims=True)
    if (row_sums <= 0).any():
        raise ValueError("probability rows must have positive mass")
    normalized /= row_sums
    return cast(np.ndarray, normalized)


def expected_calibration_error(
    *, labels: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> float:
    """Return confidence-binned multiclass calibration error."""
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
        if mask.any():
            error += float(mask.mean()) * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
    return error


def _current_features(
    *, corpus: pilot_training.PreparedCorpus, indices: np.ndarray
) -> np.ndarray:
    return np.column_stack(
        (
            corpus.lob_features[indices].reshape(len(indices), -1),
            corpus.auxiliary_features[indices],
        )
    )
