"""Translate pilot training state to atomic repository payloads."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import torch

from ebs_tft.data.repositories import checkpoint as checkpoint_repository
from ebs_tft.domain import model as model_domain


class IncompatibleCheckpointError(Exception):
    """Indicate that an existing checkpoint belongs to another experiment."""


def write_latest(
    *, path: Path, state: model_domain.TrainingState, fingerprint: str
) -> None:
    """Persist a resumable epoch-boundary state with its experiment identity."""
    checkpoint_repository.write(
        path=path,
        payload={
            "kind": "training_state",
            "fingerprint": fingerprint,
            "epoch": state.epoch,
            "classifier_state": state.classifier_state,
            "optimizer_state": state.optimizer_state,
            "best_classifier_state": state.best_classifier_state,
            "best_epoch": state.best_epoch,
            "best_validation_log_loss": state.best_validation_log_loss,
            "stalled_validations": state.stalled_validations,
            "history": [
                {
                    "epoch": item.epoch,
                    "training_loss": item.training_loss,
                    "validation_log_loss": item.validation_log_loss,
                    "gradient_norm": item.gradient_norm,
                    "improved": item.improved,
                    "validation_index": item.validation_index,
                    "optimizer_step": item.optimizer_step,
                }
                for item in state.history
            ],
            "torch_random_state": state.torch_random_state,
        },
    )


def read_latest(*, path: Path, fingerprint: str) -> model_domain.TrainingState | None:
    """Return resumable state when present and compatible with this experiment."""
    if not path.exists():
        return None
    payload = checkpoint_repository.read(path=path)
    if payload.get("kind") != "training_state":
        raise IncompatibleCheckpointError("checkpoint kind is not training_state")
    if payload.get("fingerprint") != fingerprint:
        raise IncompatibleCheckpointError(
            "checkpoint fingerprint differs from the current experiment"
        )
    raw_history = payload.get("history")
    if not isinstance(raw_history, list):
        raise IncompatibleCheckpointError("checkpoint history is invalid")
    history: list[model_domain.EpochMetric] = []
    for raw_item in raw_history:
        if not isinstance(raw_item, dict):
            raise IncompatibleCheckpointError("checkpoint history item is invalid")
        item = cast(dict[str, object], raw_item)
        history.append(
            model_domain.EpochMetric(
                epoch=_integer(data=item, key="epoch"),
                training_loss=_number(data=item, key="training_loss"),
                validation_log_loss=_number(data=item, key="validation_log_loss"),
                gradient_norm=_number(data=item, key="gradient_norm"),
                improved=_boolean(data=item, key="improved"),
                validation_index=_optional_integer(
                    data=item, key="validation_index", default=1
                ),
                optimizer_step=_optional_integer(
                    data=item, key="optimizer_step", default=0
                ),
            )
        )
    return model_domain.TrainingState(
        epoch=_integer(data=payload, key="epoch"),
        classifier_state=_tensor_mapping(data=payload, key="classifier_state"),
        optimizer_state=_mapping(data=payload, key="optimizer_state"),
        best_classifier_state=_tensor_mapping(
            data=payload, key="best_classifier_state"
        ),
        best_epoch=_integer(data=payload, key="best_epoch"),
        best_validation_log_loss=_number(data=payload, key="best_validation_log_loss"),
        stalled_validations=_integer(data=payload, key="stalled_validations"),
        history=tuple(history),
        torch_random_state=_tensor(data=payload, key="torch_random_state"),
    )


def write_best(
    *,
    path: Path,
    classifier: torch.nn.Module,
    fingerprint: str,
    training_result: model_domain.TrainingResult,
) -> None:
    """Persist the restored best model and its selection evidence."""
    checkpoint_repository.write(
        path=path,
        payload={
            "kind": "best_model",
            "fingerprint": fingerprint,
            "classifier_state": {
                name: value.detach().cpu().clone()
                for name, value in classifier.state_dict().items()
            },
            "best_epoch": training_result.best_epoch,
            "best_validation_log_loss": (training_result.best_validation_log_loss),
            "epochs_completed": training_result.epochs_completed,
            "stop_reason": training_result.stop_reason,
        },
    )


def load_best(*, path: Path, classifier: torch.nn.Module, fingerprint: str) -> None:
    """Load the compatible best state into a freshly initialized classifier."""
    payload = checkpoint_repository.read(path=path)
    if payload.get("kind") != "best_model":
        raise IncompatibleCheckpointError("checkpoint kind is not best_model")
    if payload.get("fingerprint") != fingerprint:
        raise IncompatibleCheckpointError(
            "checkpoint fingerprint differs from the current experiment"
        )
    classifier.load_state_dict(_tensor_mapping(data=payload, key="classifier_state"))


def _mapping(*, data: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise IncompatibleCheckpointError(f"checkpoint {key} must be a mapping")
    return cast(dict[str, object], value)


def _tensor_mapping(
    *, data: Mapping[str, object], key: str
) -> Mapping[str, torch.Tensor]:
    value = _mapping(data=data, key=key)
    if any(not isinstance(item, torch.Tensor) for item in value.values()):
        raise IncompatibleCheckpointError(f"checkpoint {key} values must be tensors")
    return cast(Mapping[str, torch.Tensor], value)


def _tensor(*, data: Mapping[str, object], key: str) -> torch.Tensor:
    value = data.get(key)
    if not isinstance(value, torch.Tensor):
        raise IncompatibleCheckpointError(f"checkpoint {key} must be a tensor")
    return value


def _integer(*, data: Mapping[str, object], key: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise IncompatibleCheckpointError(f"checkpoint {key} must be an integer")
    return value


def _optional_integer(*, data: Mapping[str, object], key: str, default: int) -> int:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise IncompatibleCheckpointError(f"checkpoint {key} must be an integer")
    return value


def _number(*, data: Mapping[str, object], key: str) -> float:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise IncompatibleCheckpointError(f"checkpoint {key} must be numeric")
    return float(value)


def _boolean(*, data: Mapping[str, object], key: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise IncompatibleCheckpointError(f"checkpoint {key} must be boolean")
    return value
