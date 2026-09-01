"""Run deterministic learnability checks for both direction adapters."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

import attrs
import numpy as np
import torch

from ebs_tft.data.repositories import artifact as artifact_repository
from ebs_tft.data.repositories import checkpoint as checkpoint_repository
from ebs_tft.domain import model as model_domain
from ebs_tft.domain.pilot import models as pilot_models


class ModelSanityError(Exception):
    """Indicate that a direction adapter cannot learn a controlled signal."""


@attrs.frozen
class ModelSanityResult:
    """Reference the persisted deterministic model-sanity evidence."""

    output_dir: Path
    summary_path: Path


def run(*, output_dir: Path, replace_output: bool = False) -> ModelSanityResult:
    """
    Train both adapters on a balanced causal signal and verify checkpoint reload.

    :raises ModelSanityError: if an adapter fails the controlled learnability gate
    """
    artifact_repository.prepare_run_directory(path=output_dir, replace=replace_output)
    lob_features, auxiliary_features, labels = _controlled_data()
    target_indices = np.arange(19, len(labels), dtype=np.int64)
    partitions = {
        "training": target_indices[:360],
        "validation": target_indices[360:450],
        "test": target_indices[450:],
    }
    datasets = {
        name: model_domain.SequenceDataset(
            lob_features=lob_features,
            auxiliary_features=auxiliary_features,
            labels=labels,
            target_indices=indices,
            context_steps=20,
        )
        for name, indices in partitions.items()
    }
    records: list[dict[str, object]] = []
    for model_name in pilot_models.ModelName:
        model_domain.set_random_seed(seed=7)
        classifier = _classifier(model_name=model_name)
        training_result = model_domain.fit_classifier(
            classifier=classifier,
            training_data=datasets["training"],
            validation_data=datasets["validation"],
            device=torch.device("cpu"),
            maximum_epochs=30,
            batch_size=64,
            learning_rate=0.003,
            early_stopping_patience=5,
            early_stopping_minimum_delta=0.0001,
            gradient_clip_norm=1.0,
            random_seed=7,
            epoch_observer=_observer(model_name=model_name.value),
        )
        checkpoint_path = output_dir / f"{model_name.value}.best.pt"
        checkpoint_repository.write(
            path=checkpoint_path,
            payload={
                "classifier_state": classifier.state_dict(),
                "best_epoch": training_result.best_epoch,
            },
        )
        reloaded = _classifier(model_name=model_name)
        payload = checkpoint_repository.read(path=checkpoint_path)
        raw_state = payload.get("classifier_state")
        if not isinstance(raw_state, dict):
            raise ModelSanityError("sanity checkpoint state is invalid")
        reloaded.load_state_dict(cast(dict[str, torch.Tensor], raw_state))
        prediction = model_domain.predict_classifier(
            classifier=reloaded,
            dataset=datasets["test"],
            device=torch.device("cpu"),
            batch_size=64,
        )
        expected = datasets["test"].target_labels()
        accuracy = float((prediction.labels == expected).mean())
        probability_error = float(
            np.abs(prediction.probabilities.sum(axis=1) - 1.0).max()
        )
        record: dict[str, object] = {
            "model": model_name.value,
            "accuracy": accuracy,
            "maximum_probability_sum_error": probability_error,
            "best_epoch": training_result.best_epoch,
            "best_validation_log_loss": (training_result.best_validation_log_loss),
            "epochs_completed": training_result.epochs_completed,
            "stop_reason": training_result.stop_reason,
        }
        records.append(record)
        if accuracy < 0.95 or probability_error > 1e-6:
            _write_summary(output_dir=output_dir, records=records, passed=False)
            raise ModelSanityError(
                f"{model_name.value} failed controlled learnability: {record}"
            )
    summary_path = _write_summary(output_dir=output_dir, records=records, passed=True)
    print(f"Model sanity gate passed. Outputs: {output_dir}", flush=True)
    return ModelSanityResult(output_dir=output_dir, summary_path=summary_path)


def _controlled_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = 600
    labels = (np.arange(rows) % 3).astype(np.int64)
    lob_features = np.zeros((rows, 3, 6), dtype=np.float32)
    auxiliary_features = np.zeros((rows, 10), dtype=np.float32)
    auxiliary_features[np.arange(rows), labels] = 4.0
    lob_features[np.arange(rows), :, labels] = 2.0
    return lob_features, auxiliary_features, labels


def _classifier(*, model_name: pilot_models.ModelName) -> torch.nn.Module:
    if model_name is pilot_models.ModelName.DEEP_LOB:
        return model_domain.DeepLobDirectionClassifier(
            auxiliary_size=10, hidden_size=16
        )
    return model_domain.TftDirectionClassifier(
        input_size=28, hidden_size=16, attention_heads=4
    )


def _print_epoch(*, model_name: str, metric: model_domain.EpochMetric) -> None:
    marker = " best" if metric.improved else ""
    print(
        f"[sanity:{model_name}] epoch={metric.epoch} "
        f"train_loss={metric.training_loss:.6f} "
        f"validation_log_loss={metric.validation_log_loss:.6f}{marker}",
        flush=True,
    )


def _observer(*, model_name: str) -> Callable[[model_domain.EpochMetric], None]:
    def observe(metric: model_domain.EpochMetric) -> None:
        _print_epoch(model_name=model_name, metric=metric)

    return observe


def _write_summary(
    *, output_dir: Path, records: list[dict[str, object]], passed: bool
) -> Path:
    summary_path = output_dir / "model_sanity.json"
    summary_path.write_text(
        json.dumps({"passed": passed, "models": records}, indent=2),
        encoding="utf-8",
    )
    return summary_path
