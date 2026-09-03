"""Load the exact-schema forecasting-research protocol."""

from __future__ import annotations

import datetime
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import yaml

from ebs_tft.domain.orderbook import models as orderbook_models
from ebs_tft.domain.research import models as research_models


class UnableToLoadResearchProtocolError(Exception):
    """Indicate that a research-protocol document is unreadable or invalid."""


def load_protocol(*, path: Path) -> research_models.ResearchProtocol:
    """Return one completely validated forecasting-research protocol."""
    try:
        with path.open(encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise UnableToLoadResearchProtocolError(
            f"Unable to load research protocol: {path}"
        ) from exc
    if not isinstance(loaded, dict) or not all(isinstance(key, str) for key in loaded):
        raise UnableToLoadResearchProtocolError("configuration must be a mapping")
    data = cast(dict[str, object], loaded)
    expected = {
        "schema_version",
        "data_dir",
        "output_dir",
        "instruments",
        "years",
        "state_interval_milliseconds",
        "forecast_horizons_milliseconds",
        "context_milliseconds",
        "maximum_staleness_milliseconds",
        "audit_workers",
        "training_stride_milliseconds",
        "evaluation_stride_milliseconds",
        "audit_policy",
        "split_policy",
        "development_instrument",
        "depths",
        "models",
        "random_seeds",
        "validation_checks_per_epoch",
        "primary_metrics",
        "supporting_metrics",
        "bootstrap_repetitions",
        "confidence_level",
    }
    if set(data) != expected:
        missing = sorted(expected - set(data))
        extra = sorted(set(data) - expected)
        raise UnableToLoadResearchProtocolError(
            f"configuration keys differ; missing={missing}, extra={extra}"
        )
    if _integer(data=data, key="schema_version") != 1:
        raise UnableToLoadResearchProtocolError("unsupported schema_version")
    base_dir = path.resolve().parent
    try:
        instruments = tuple(
            orderbook_models.Instrument(item)
            for item in _string_tuple(data=data, key="instruments")
        )
        return research_models.ResearchProtocol(
            data_dir=(base_dir / _string(data=data, key="data_dir")).resolve(),
            output_dir=(base_dir / _string(data=data, key="output_dir")).resolve(),
            instruments=instruments,
            years=_integer_tuple(data=data, key="years"),
            state_interval_milliseconds=_integer(
                data=data, key="state_interval_milliseconds"
            ),
            forecast_horizons_milliseconds=_integer_tuple(
                data=data, key="forecast_horizons_milliseconds"
            ),
            context_milliseconds=_integer(data=data, key="context_milliseconds"),
            maximum_staleness_milliseconds=_integer(
                data=data, key="maximum_staleness_milliseconds"
            ),
            audit_workers=_integer(data=data, key="audit_workers"),
            training_stride_milliseconds=_stride_mapping(
                value=data["training_stride_milliseconds"]
            ),
            evaluation_stride_milliseconds=_integer(
                data=data, key="evaluation_stride_milliseconds"
            ),
            audit_policy=_audit_policy(value=data["audit_policy"]),
            split_policy=_split_policy(value=data["split_policy"]),
            development_instrument=orderbook_models.Instrument(
                _string(data=data, key="development_instrument")
            ),
            depths=_integer_tuple(data=data, key="depths"),
            models=_string_tuple(data=data, key="models"),
            random_seeds=_integer_tuple(data=data, key="random_seeds"),
            validation_checks_per_epoch=_integer(
                data=data, key="validation_checks_per_epoch"
            ),
            primary_metrics=_metric_tuple(data=data, key="primary_metrics"),
            supporting_metrics=_metric_tuple(data=data, key="supporting_metrics"),
            bootstrap_repetitions=_integer(data=data, key="bootstrap_repetitions"),
            confidence_level=_float(data=data, key="confidence_level"),
        )
    except (TypeError, ValueError) as exc:
        raise UnableToLoadResearchProtocolError(
            f"invalid research protocol: {exc}"
        ) from exc


def _audit_policy(*, value: object) -> research_models.AuditPolicy:
    expected = {
        "minimum_duration_milliseconds",
        "minimum_observed_states",
        "required_depth",
        "redact_locked_outcomes",
    }
    data = _nested_mapping(value=value, expected=expected, name="audit_policy")
    return research_models.AuditPolicy(
        minimum_duration_milliseconds=_integer(
            data=data, key="minimum_duration_milliseconds"
        ),
        minimum_observed_states=_integer(data=data, key="minimum_observed_states"),
        required_depth=_integer(data=data, key="required_depth"),
        redact_locked_outcomes=_boolean(data=data, key="redact_locked_outcomes"),
    )


def _split_policy(*, value: object) -> research_models.SplitPolicy:
    expected = {
        "development_end_date",
        "minimum_training_sessions",
        "validation_sessions_per_fold",
        "fold_step_sessions",
        "locked_evaluation_dates",
    }
    data = _nested_mapping(value=value, expected=expected, name="split_policy")
    return research_models.SplitPolicy(
        development_end_date=datetime.date.fromisoformat(
            _string(data=data, key="development_end_date")
        ),
        minimum_training_sessions=_integer(data=data, key="minimum_training_sessions"),
        validation_sessions_per_fold=_integer(
            data=data, key="validation_sessions_per_fold"
        ),
        fold_step_sessions=_integer(data=data, key="fold_step_sessions"),
        locked_evaluation_dates=tuple(
            datetime.date.fromisoformat(item)
            for item in _string_tuple(data=data, key="locked_evaluation_dates")
        ),
    )


def _nested_mapping(
    *, value: object, expected: set[str], name: str
) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a mapping")
    data = cast(dict[str, object], value)
    if set(data) != expected:
        raise ValueError(f"{name} keys must be exactly {sorted(expected)}")
    return data


def _stride_mapping(*, value: object) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, dict):
        raise ValueError("training_stride_milliseconds must be a mapping")
    pairs: list[tuple[int, int]] = []
    for raw_horizon, raw_stride in value.items():
        if isinstance(raw_horizon, bool) or not isinstance(raw_horizon, int):
            raise ValueError("training stride keys must be integer horizons")
        if isinstance(raw_stride, bool) or not isinstance(raw_stride, int):
            raise ValueError("training stride values must be integers")
        pairs.append((raw_horizon, raw_stride))
    return tuple(sorted(pairs))


def _metric_tuple(
    *, data: Mapping[str, object], key: str
) -> tuple[research_models.EvaluationMetric, ...]:
    return tuple(
        research_models.EvaluationMetric(item)
        for item in _string_tuple(data=data, key=key)
    )


def _string(*, data: Mapping[str, object], key: str) -> str:
    value = data[key]
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


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


def _boolean(*, data: Mapping[str, object], key: str) -> bool:
    value = data[key]
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _integer_tuple(*, data: Mapping[str, object], key: str) -> tuple[int, ...]:
    value = data[key]
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        raise ValueError(f"{key} must be a list of integers")
    return tuple(value)


def _string_tuple(*, data: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = data[key]
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"{key} must be a list of non-empty strings")
    return tuple(value)
