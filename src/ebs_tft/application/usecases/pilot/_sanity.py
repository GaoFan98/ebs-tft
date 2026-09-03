"""Run deterministic depth-capability checks for both direction adapters."""

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
    """Indicate that a direction adapter fails a controlled capability gate."""


@attrs.frozen
class ModelSanityResult:
    """Reference the persisted deterministic depth-capability evidence."""

    output_dir: Path
    summary_path: Path


@attrs.frozen
class _CapabilityCase:
    name: str
    lob_features: np.ndarray
    auxiliary_features: np.ndarray
    labels: np.ndarray
    depth: int
    minimum_accuracy: float | None = None
    maximum_accuracy: float | None = None


def run(*, output_dir: Path, replace_output: bool = False) -> ModelSanityResult:
    """
    Verify checkpoint reload, L1 preservation, and deeper-signal accessibility.

    :raises ModelSanityError: if an adapter fails a controlled capability gate
    """
    artifact_repository.prepare_run_directory(path=output_dir, replace=replace_output)
    records: list[dict[str, object]] = []
    for model_name in pilot_models.ModelName:
        model_records: list[dict[str, object]] = []
        for capability in _capability_cases():
            record = _run_capability(
                model_name=model_name,
                capability=capability,
                output_dir=output_dir,
            )
            records.append(record)
            model_records.append(record)
            if not _passed(record=record, capability=capability):
                _write_summary(output_dir=output_dir, records=records, passed=False)
                raise ModelSanityError(
                    f"{model_name.value} failed {capability.name}: {record}"
                )
        parameter_counts = {record["parameter_count"] for record in model_records}
        if len(parameter_counts) != 1:
            _write_summary(output_dir=output_dir, records=records, passed=False)
            raise ModelSanityError(
                f"{model_name.value} changes capacity with observed depth"
            )
    summary_path = _write_summary(output_dir=output_dir, records=records, passed=True)
    print(f"Model depth-capability gate passed. Outputs: {output_dir}", flush=True)
    return ModelSanityResult(output_dir=output_dir, summary_path=summary_path)


def _run_capability(
    *,
    model_name: pilot_models.ModelName,
    capability: _CapabilityCase,
    output_dir: Path,
) -> dict[str, object]:
    datasets = _datasets(capability=capability)
    model_domain.set_random_seed(seed=7)
    classifier = _classifier(model_name=model_name)
    training_result = model_domain.fit_classifier(
        classifier=classifier,
        training_data=datasets["training"],
        validation_data=datasets["validation"],
        device=torch.device("cpu"),
        maximum_epochs=25,
        batch_size=64,
        learning_rate=0.003,
        weight_decay=0.0,
        early_stopping_patience=5,
        early_stopping_minimum_delta=0.0001,
        gradient_clip_norm=1.0,
        random_seed=7,
        epoch_observer=_observer(
            model_name=model_name.value, capability=capability.name
        ),
    )
    checkpoint_path = output_dir / f"{model_name.value}_{capability.name}.best.pt"
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
    probability_error = float(np.abs(prediction.probabilities.sum(axis=1) - 1.0).max())
    return {
        "model": model_name.value,
        "capability": capability.name,
        "depth": capability.depth,
        "accuracy": accuracy,
        "maximum_probability_sum_error": probability_error,
        "minimum_accuracy": capability.minimum_accuracy,
        "maximum_accuracy": capability.maximum_accuracy,
        "best_epoch": training_result.best_epoch,
        "best_validation_log_loss": training_result.best_validation_log_loss,
        "epochs_completed": training_result.epochs_completed,
        "stop_reason": training_result.stop_reason,
        "parameter_count": model_domain.parameter_count(classifier=reloaded),
    }


def _capability_cases() -> tuple[_CapabilityCase, ...]:
    l1_lob, l1_auxiliary, l1_labels = _controlled_data(signal_level=1)
    deep_lob, deep_auxiliary, deep_labels = _controlled_data(signal_level=10)
    return (
        _CapabilityCase(
            name="l1_signal_l1",
            lob_features=l1_lob,
            auxiliary_features=l1_auxiliary,
            labels=l1_labels,
            depth=1,
            minimum_accuracy=0.90,
        ),
        _CapabilityCase(
            name="l1_signal_l10",
            lob_features=l1_lob,
            auxiliary_features=l1_auxiliary,
            labels=l1_labels,
            depth=10,
            minimum_accuracy=0.90,
        ),
        _CapabilityCase(
            name="deep_signal_l1",
            lob_features=deep_lob,
            auxiliary_features=deep_auxiliary,
            labels=deep_labels,
            depth=1,
            maximum_accuracy=0.50,
        ),
        _CapabilityCase(
            name="deep_signal_l10",
            lob_features=deep_lob,
            auxiliary_features=deep_auxiliary,
            labels=deep_labels,
            depth=10,
            minimum_accuracy=0.90,
        ),
    )


def _controlled_data(*, signal_level: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = 900
    generator = np.random.default_rng(20260902 + signal_level)
    labels = np.tile(np.arange(3, dtype=np.int64), rows // 3)
    generator.shuffle(labels)
    lob_features = generator.normal(
        loc=0.0,
        scale=0.15,
        size=(rows, 10, 6),
    ).astype(np.float32)
    auxiliary_features = np.zeros((rows, 10), dtype=np.float32)
    lob_features[np.arange(rows), signal_level - 1, labels] += 4.0
    return lob_features, auxiliary_features, labels


def _datasets(
    *, capability: _CapabilityCase
) -> dict[str, model_domain.SequenceDataset]:
    context_steps = 10
    target_indices = np.arange(
        context_steps - 1, len(capability.labels), dtype=np.int64
    )
    partitions = {
        "training": target_indices[:540],
        "validation": target_indices[540:720],
        "test": target_indices[720:],
    }
    return {
        name: model_domain.SequenceDataset(
            lob_features=capability.lob_features[:, : capability.depth],
            auxiliary_features=capability.auxiliary_features,
            labels=capability.labels,
            target_indices=indices,
            context_steps=context_steps,
        )
        for name, indices in partitions.items()
    }


def _classifier(*, model_name: pilot_models.ModelName) -> torch.nn.Module:
    if model_name is pilot_models.ModelName.DEEP_LOB:
        return model_domain.DeepLobDirectionClassifier(
            auxiliary_size=10, hidden_size=16
        )
    return model_domain.TftDirectionClassifier(
        auxiliary_size=10, hidden_size=16, attention_heads=4
    )


def _passed(*, record: dict[str, object], capability: _CapabilityCase) -> bool:
    accuracy = record["accuracy"]
    probability_error = record["maximum_probability_sum_error"]
    if not isinstance(accuracy, float) or not isinstance(probability_error, float):
        raise ModelSanityError("sanity result metrics are invalid")
    if probability_error > 1e-6:
        return False
    if capability.minimum_accuracy is not None:
        return accuracy >= capability.minimum_accuracy
    if capability.maximum_accuracy is not None:
        return accuracy <= capability.maximum_accuracy
    raise ModelSanityError("capability has no acceptance threshold")


def _observer(
    *, model_name: str, capability: str
) -> Callable[[model_domain.EpochMetric], None]:
    def observe(metric: model_domain.EpochMetric) -> None:
        marker = " best" if metric.improved else ""
        print(
            f"[sanity:{model_name}:{capability}] epoch={metric.epoch} "
            f"train_loss={metric.training_loss:.6f} "
            f"validation_log_loss={metric.validation_log_loss:.6f}{marker}",
            flush=True,
        )

    return observe


def _write_summary(
    *, output_dir: Path, records: list[dict[str, object]], passed: bool
) -> Path:
    summary_path = output_dir / "model_sanity.json"
    summary_path.write_text(
        json.dumps(
            {
                "model_protocol_version": model_domain.MODEL_PROTOCOL_VERSION,
                "passed": passed,
                "capabilities": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary_path
