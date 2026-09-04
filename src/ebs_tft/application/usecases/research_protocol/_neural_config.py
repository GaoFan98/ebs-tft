"""Load the exact-schema neural benchmark optimization policy."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import yaml

from ebs_tft.domain.research import models as research_models


class UnableToLoadNeuralBenchmarkPolicyError(Exception):
    """Indicate that a neural benchmark policy is unreadable or invalid."""


def load_policy(*, path: Path) -> research_models.NeuralBenchmarkPolicy:
    """Return one validated neural benchmark optimization policy."""
    try:
        with path.open(encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise UnableToLoadNeuralBenchmarkPolicyError(
            f"Unable to load neural benchmark policy: {path}"
        ) from exc
    if not isinstance(loaded, dict) or not all(isinstance(key, str) for key in loaded):
        raise UnableToLoadNeuralBenchmarkPolicyError("policy must be a mapping")
    data = cast(dict[str, object], loaded)
    expected = {
        "schema_version",
        "maximum_epochs",
        "early_stopping_patience",
        "early_stopping_minimum_delta",
        "gradient_clip_norm",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "hidden_size",
        "device",
    }
    if set(data) != expected:
        missing = sorted(expected - set(data))
        extra = sorted(set(data) - expected)
        raise UnableToLoadNeuralBenchmarkPolicyError(
            f"policy keys differ; missing={missing}, extra={extra}"
        )
    if _integer(data=data, key="schema_version") != 1:
        raise UnableToLoadNeuralBenchmarkPolicyError("unsupported schema_version")
    try:
        return research_models.NeuralBenchmarkPolicy(
            maximum_epochs=_integer(data=data, key="maximum_epochs"),
            early_stopping_patience=_integer(data=data, key="early_stopping_patience"),
            early_stopping_minimum_delta=_float(
                data=data, key="early_stopping_minimum_delta"
            ),
            gradient_clip_norm=_float(data=data, key="gradient_clip_norm"),
            batch_size=_integer(data=data, key="batch_size"),
            learning_rate=_float(data=data, key="learning_rate"),
            weight_decay=_float(data=data, key="weight_decay"),
            hidden_size=_integer(data=data, key="hidden_size"),
            device=_string(data=data, key="device"),
        )
    except (TypeError, ValueError) as exc:
        raise UnableToLoadNeuralBenchmarkPolicyError(
            f"invalid neural benchmark policy: {exc}"
        ) from exc


def _integer(*, data: Mapping[str, object], key: str) -> int:
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _float(*, data: Mapping[str, object], key: str) -> float:
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{key} must be a number")
    return float(value)


def _string(*, data: Mapping[str, object], key: str) -> str:
    value = data[key]
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value
