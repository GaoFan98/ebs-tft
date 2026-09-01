"""
Load and validate the project's local YAML configuration.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import attrs
import yaml

from ebs_tft.domain.orderbook import models as orderbook_models

SCHEMA_VERSION: int = 1
NATIVE_STATE_INTERVAL_MILLISECONDS: int = 100
SUPPORTED_SOURCE_TIMEZONES: frozenset[str] = frozenset({"UTC"})
SUPPORTED_SESSION_CALENDARS: frozenset[str] = frozenset({"EBS_FX_17_NEW_YORK"})
SUPPORTED_FLAT_TARGET_POLICIES: frozenset[str] = frozenset(
    {"three_class", "exclude_exact_flat", "exclude_neutral_band"}
)


class UnableToLoadConfigError(Exception):
    """
    Indicate that a configuration file could not be read or decoded.
    """


class InvalidConfigError(Exception):
    """
    Indicate that configuration content violates the project schema.
    """


@attrs.frozen
class InstrumentsConfig:
    """
    Define the active instruments and maximum cumulative order-book depth.
    """

    schema_version: int
    instruments: tuple[orderbook_models.Instrument, ...]
    maximum_depth: int

    @property
    def depths(self) -> tuple[int, ...]:
        """
        Return every cumulative depth from level one through maximum_depth.
        """
        return tuple(range(1, self.maximum_depth + 1))


@attrs.frozen
class TrainingConfig:
    """
    Define known training inputs and explicitly unresolved research decisions.
    """

    schema_version: int
    raw_data_dir: Path
    processed_data_dir: Path
    state_interval_milliseconds: int
    forecast_horizons_milliseconds: tuple[int, ...]
    source_timezone: str | None
    session_calendar: str | None
    maximum_quote_staleness_milliseconds: int | None
    flat_target_policy: str | None
    random_seeds: tuple[int, ...]

    @property
    def is_research_ready(self) -> bool:
        """
        Test whether all evidence-dependent training choices have been resolved.
        """
        return bool(
            self.forecast_horizons_milliseconds
            and self.source_timezone
            and self.session_calendar
            and self.maximum_quote_staleness_milliseconds is not None
            and self.flat_target_policy
            and self.random_seeds
        )


@attrs.frozen
class ModelDefaultsConfig:
    """
    Define the selected model engine once the baseline phase reaches that decision.
    """

    schema_version: int
    engine: str | None


@attrs.frozen
class ProjectConfig:
    """
    Provide one typed boundary over all project configuration files.
    """

    instruments: InstrumentsConfig
    training: TrainingConfig
    model_defaults: ModelDefaultsConfig


def load_project_config(*, config_dir: Path) -> ProjectConfig:
    """
    Load and validate all project configuration from config_dir.

    Relative data paths are resolved against the project directory containing the
    config directory.

    :raises UnableToLoadConfigError: if a required YAML file cannot be read
    :raises InvalidConfigError: if a YAML document violates the configuration schema
    """
    project_dir = config_dir.resolve().parent
    instruments_data = _load_mapping(path=config_dir / "instruments.yaml")
    training_data = _load_mapping(path=config_dir / "training.yaml")
    model_data = _load_mapping(path=config_dir / "model_defaults.yaml")

    return ProjectConfig(
        instruments=_parse_instruments(data=instruments_data),
        training=_parse_training(data=training_data, project_dir=project_dir),
        model_defaults=_parse_model_defaults(data=model_data),
    )


def _load_mapping(*, path: Path) -> dict[str, object]:
    try:
        with path.open(encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise UnableToLoadConfigError(f"Unable to load configuration: {path}") from exc

    if not isinstance(loaded, dict) or not all(isinstance(key, str) for key in loaded):
        raise InvalidConfigError(
            f"Configuration must be a mapping with string keys: {path}"
        )
    return cast(dict[str, object], loaded)


def _parse_instruments(*, data: Mapping[str, object]) -> InstrumentsConfig:
    _validate_keys(
        data=data,
        expected=frozenset({"schema_version", "instruments", "maximum_depth"}),
        section="instruments",
    )
    schema_version = _schema_version(data=data, section="instruments")
    raw_instruments = _required_list(
        data=data, key="instruments", section="instruments"
    )
    instruments: list[orderbook_models.Instrument] = []
    for index, value in enumerate(raw_instruments):
        if not isinstance(value, str):
            raise InvalidConfigError(
                f"instruments.instruments[{index}] must be a string"
            )
        try:
            instrument = orderbook_models.Instrument.from_filename_part(
                instrument=value
            )
        except ValueError as exc:
            raise InvalidConfigError(
                f"Unsupported instrument in instruments.instruments: {value!r}"
            ) from exc
        instruments.append(instrument)

    if not instruments:
        raise InvalidConfigError("instruments.instruments must not be empty")
    if len(set(instruments)) != len(instruments):
        raise InvalidConfigError("instruments.instruments must not contain duplicates")

    maximum_depth = _required_int(data=data, key="maximum_depth", section="instruments")
    if not 1 <= maximum_depth <= orderbook_models.MAX_LEVELS:
        raise InvalidConfigError(
            "instruments.maximum_depth must be between "
            f"1 and {orderbook_models.MAX_LEVELS} inclusive"
        )

    return InstrumentsConfig(
        schema_version=schema_version,
        instruments=tuple(instruments),
        maximum_depth=maximum_depth,
    )


def _parse_training(*, data: Mapping[str, object], project_dir: Path) -> TrainingConfig:
    expected = frozenset(
        {
            "schema_version",
            "raw_data_dir",
            "processed_data_dir",
            "state_interval_milliseconds",
            "forecast_horizons_milliseconds",
            "source_timezone",
            "session_calendar",
            "maximum_quote_staleness_milliseconds",
            "flat_target_policy",
            "random_seeds",
        }
    )
    _validate_keys(data=data, expected=expected, section="training")
    schema_version = _schema_version(data=data, section="training")
    state_interval = _required_int(
        data=data, key="state_interval_milliseconds", section="training"
    )
    if state_interval != NATIVE_STATE_INTERVAL_MILLISECONDS:
        raise InvalidConfigError(
            "training.state_interval_milliseconds must preserve the verified "
            f"native {NATIVE_STATE_INTERVAL_MILLISECONDS} ms interval"
        )

    horizons = _integer_tuple(
        data=data,
        key="forecast_horizons_milliseconds",
        section="training",
        allow_empty=True,
    )
    if any(horizon <= 0 for horizon in horizons):
        raise InvalidConfigError(
            "training.forecast_horizons_milliseconds values must be positive"
        )
    if any(horizon % state_interval for horizon in horizons):
        raise InvalidConfigError(
            "training.forecast_horizons_milliseconds values must align to the "
            "native state interval"
        )
    _validate_unique(values=horizons, name="training.forecast_horizons_milliseconds")

    random_seeds = _integer_tuple(
        data=data,
        key="random_seeds",
        section="training",
        allow_empty=True,
    )
    if any(seed < 0 for seed in random_seeds):
        raise InvalidConfigError("training.random_seeds values must be non-negative")
    _validate_unique(values=random_seeds, name="training.random_seeds")

    maximum_staleness = _optional_int(
        data=data,
        key="maximum_quote_staleness_milliseconds",
        section="training",
    )
    if maximum_staleness is not None and maximum_staleness <= 0:
        raise InvalidConfigError(
            "training.maximum_quote_staleness_milliseconds must be positive or null"
        )

    source_timezone = _optional_str(
        data=data, key="source_timezone", section="training"
    )
    if (
        source_timezone is not None
        and source_timezone not in SUPPORTED_SOURCE_TIMEZONES
    ):
        raise InvalidConfigError(
            f"Unsupported training.source_timezone: {source_timezone!r}"
        )
    session_calendar = _optional_str(
        data=data, key="session_calendar", section="training"
    )
    if (
        session_calendar is not None
        and session_calendar not in SUPPORTED_SESSION_CALENDARS
    ):
        raise InvalidConfigError(
            f"Unsupported training.session_calendar: {session_calendar!r}"
        )
    flat_target_policy = _optional_str(
        data=data, key="flat_target_policy", section="training"
    )
    if (
        flat_target_policy is not None
        and flat_target_policy not in SUPPORTED_FLAT_TARGET_POLICIES
    ):
        raise InvalidConfigError(
            f"Unsupported training.flat_target_policy: {flat_target_policy!r}"
        )

    return TrainingConfig(
        schema_version=schema_version,
        raw_data_dir=_resolve_project_path(
            value=_required_str(data=data, key="raw_data_dir", section="training"),
            project_dir=project_dir,
        ),
        processed_data_dir=_resolve_project_path(
            value=_required_str(
                data=data, key="processed_data_dir", section="training"
            ),
            project_dir=project_dir,
        ),
        state_interval_milliseconds=state_interval,
        forecast_horizons_milliseconds=horizons,
        source_timezone=source_timezone,
        session_calendar=session_calendar,
        maximum_quote_staleness_milliseconds=maximum_staleness,
        flat_target_policy=flat_target_policy,
        random_seeds=random_seeds,
    )


def _parse_model_defaults(*, data: Mapping[str, object]) -> ModelDefaultsConfig:
    _validate_keys(
        data=data,
        expected=frozenset({"schema_version", "engine"}),
        section="model_defaults",
    )
    return ModelDefaultsConfig(
        schema_version=_schema_version(data=data, section="model_defaults"),
        engine=_optional_str(data=data, key="engine", section="model_defaults"),
    )


def _validate_keys(
    *, data: Mapping[str, object], expected: frozenset[str], section: str
) -> None:
    actual = frozenset(data)
    unknown = sorted(actual - expected)
    missing = sorted(expected - actual)
    if unknown:
        raise InvalidConfigError(f"Unknown {section} keys: {unknown}")
    if missing:
        raise InvalidConfigError(f"Missing {section} keys: {missing}")


def _schema_version(*, data: Mapping[str, object], section: str) -> int:
    version = _required_int(data=data, key="schema_version", section=section)
    if version != SCHEMA_VERSION:
        raise InvalidConfigError(
            f"Unsupported {section}.schema_version: {version}; "
            f"expected {SCHEMA_VERSION}"
        )
    return version


def _required_list(
    *, data: Mapping[str, object], key: str, section: str
) -> list[object]:
    value = data[key]
    if not isinstance(value, list):
        raise InvalidConfigError(f"{section}.{key} must be a list")
    return cast(list[object], value)


def _required_int(*, data: Mapping[str, object], key: str, section: str) -> int:
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidConfigError(f"{section}.{key} must be an integer")
    return value


def _optional_int(*, data: Mapping[str, object], key: str, section: str) -> int | None:
    value = data[key]
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidConfigError(f"{section}.{key} must be an integer or null")
    return value


def _required_str(*, data: Mapping[str, object], key: str, section: str) -> str:
    value = data[key]
    if not isinstance(value, str) or not value.strip():
        raise InvalidConfigError(f"{section}.{key} must be a non-empty string")
    return value


def _optional_str(*, data: Mapping[str, object], key: str, section: str) -> str | None:
    value = data[key]
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise InvalidConfigError(f"{section}.{key} must be a non-empty string or null")
    return value


def _integer_tuple(
    *,
    data: Mapping[str, object],
    key: str,
    section: str,
    allow_empty: bool,
) -> tuple[int, ...]:
    values = _required_list(data=data, key=key, section=section)
    if not allow_empty and not values:
        raise InvalidConfigError(f"{section}.{key} must not be empty")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise InvalidConfigError(f"{section}.{key} must contain only integers")
    return tuple(cast(list[int], values))


def _validate_unique(*, values: tuple[int, ...], name: str) -> None:
    if len(set(values)) != len(values):
        raise InvalidConfigError(f"{name} must not contain duplicates")


def _resolve_project_path(*, value: str, project_dir: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_dir / path).resolve()
