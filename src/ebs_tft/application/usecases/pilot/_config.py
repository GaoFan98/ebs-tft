"""Load the isolated local-pilot configuration boundary."""

from __future__ import annotations

import datetime
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import yaml

from ebs_tft.domain.orderbook import models as orderbook_models
from ebs_tft.domain.pilot import models as pilot_models


class UnableToLoadPilotConfigError(Exception):
    """Indicate that the pilot YAML cannot be read or validated."""


def load_specification(*, path: Path) -> pilot_models.PilotSpecification:
    """Return a typed pilot specification from one exact-schema YAML document."""
    try:
        with path.open(encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise UnableToLoadPilotConfigError(
            f"Unable to load pilot configuration: {path}"
        ) from exc
    if not isinstance(loaded, dict) or not all(isinstance(key, str) for key in loaded):
        raise UnableToLoadPilotConfigError("pilot configuration must be a mapping")
    data = cast(dict[str, object], loaded)
    expected = {
        "schema_version",
        "raw_path",
        "instrument",
        "trading_date",
        "output_dir",
        "grid_steps",
        "state_interval_milliseconds",
        "forecast_horizons_milliseconds",
        "modeled_horizons_milliseconds",
        "class_weighted_horizons_milliseconds",
        "context_milliseconds",
        "maximum_staleness_milliseconds",
        "depths",
        "models",
        "maximum_training_windows",
        "maximum_validation_windows",
        "maximum_test_windows",
        "maximum_epochs",
        "early_stopping_patience",
        "early_stopping_minimum_delta",
        "gradient_clip_norm",
        "batch_size",
        "learning_rate",
        "hidden_size",
        "random_seeds",
        "device",
    }
    if set(data) != expected:
        missing = sorted(expected - set(data))
        extra = sorted(set(data) - expected)
        raise UnableToLoadPilotConfigError(
            f"pilot configuration keys differ; missing={missing}, extra={extra}"
        )
    if _integer(data=data, key="schema_version") != 2:
        raise UnableToLoadPilotConfigError("unsupported pilot schema_version")
    base_dir = path.resolve().parent
    try:
        instrument = orderbook_models.Instrument(_string(data=data, key="instrument"))
        trading_date = datetime.date.fromisoformat(
            _string(data=data, key="trading_date")
        )
        return pilot_models.PilotSpecification(
            raw_path=(base_dir / _string(data=data, key="raw_path")).resolve(),
            instrument=instrument,
            trading_date=trading_date,
            output_dir=(base_dir / _string(data=data, key="output_dir")).resolve(),
            grid_steps=_integer(data=data, key="grid_steps"),
            state_interval_milliseconds=_integer(
                data=data, key="state_interval_milliseconds"
            ),
            forecast_horizons_milliseconds=_integer_tuple(
                data=data, key="forecast_horizons_milliseconds"
            ),
            modeled_horizons_milliseconds=_integer_tuple(
                data=data, key="modeled_horizons_milliseconds"
            ),
            class_weighted_horizons_milliseconds=_integer_tuple(
                data=data, key="class_weighted_horizons_milliseconds"
            ),
            context_milliseconds=_integer(data=data, key="context_milliseconds"),
            maximum_staleness_milliseconds=_integer(
                data=data, key="maximum_staleness_milliseconds"
            ),
            depths=_integer_tuple(data=data, key="depths"),
            models=_models(data=data),
            maximum_training_windows=_optional_integer(
                data=data, key="maximum_training_windows"
            ),
            maximum_validation_windows=_optional_integer(
                data=data, key="maximum_validation_windows"
            ),
            maximum_test_windows=_optional_integer(
                data=data, key="maximum_test_windows"
            ),
            maximum_epochs=_integer(data=data, key="maximum_epochs"),
            early_stopping_patience=_integer(data=data, key="early_stopping_patience"),
            early_stopping_minimum_delta=_float(
                data=data, key="early_stopping_minimum_delta"
            ),
            gradient_clip_norm=_float(data=data, key="gradient_clip_norm"),
            batch_size=_integer(data=data, key="batch_size"),
            learning_rate=_float(data=data, key="learning_rate"),
            hidden_size=_integer(data=data, key="hidden_size"),
            random_seeds=_integer_tuple(data=data, key="random_seeds"),
            device=_string(data=data, key="device"),
        )
    except (ValueError, TypeError) as exc:
        raise UnableToLoadPilotConfigError(
            f"invalid pilot configuration: {exc}"
        ) from exc


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


def _optional_integer(*, data: Mapping[str, object], key: str) -> int | None:
    value = data[key]
    if value is None:
        return None
    return _integer(data=data, key=key)


def _float(*, data: Mapping[str, object], key: str) -> float:
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{key} must be a number")
    return float(value)


def _integer_tuple(*, data: Mapping[str, object], key: str) -> tuple[int, ...]:
    value = data[key]
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        raise ValueError(f"{key} must be a list of integers")
    return tuple(value)


def _models(*, data: Mapping[str, object]) -> tuple[pilot_models.ModelName, ...]:
    value = data["models"]
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError("models must be a list of model names")
    return tuple(pilot_models.ModelName(item) for item in value)
