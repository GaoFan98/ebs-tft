"""Load the exact-schema external-year evaluation policy."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import yaml

from ebs_tft.domain.orderbook import models as orderbook_models
from ebs_tft.domain.research import models as research_models


class UnableToLoadTemporalEvaluationPolicyError(Exception):
    """Indicate that the temporal-evaluation policy is unreadable or invalid."""


def load_temporal_policy(*, path: Path) -> research_models.TemporalEvaluationPolicy:
    """Return one completely validated temporal-evaluation policy."""
    try:
        with path.open(encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise UnableToLoadTemporalEvaluationPolicyError(
            f"Unable to load temporal evaluation policy: {path}"
        ) from exc
    expected = {
        "schema_version",
        "evaluation_year",
        "instruments",
        "primary_instrument",
        "minimum_common_eligible_sessions",
        "session_selection",
    }
    if not isinstance(loaded, dict) or not all(isinstance(key, str) for key in loaded):
        raise UnableToLoadTemporalEvaluationPolicyError(
            "temporal policy must be a mapping"
        )
    data = cast(dict[str, object], loaded)
    if set(data) != expected:
        raise UnableToLoadTemporalEvaluationPolicyError(
            f"temporal policy keys must be exactly {sorted(expected)}"
        )
    try:
        schema = _integer(data=data, key="schema_version")
        if schema != 1:
            raise ValueError("unsupported schema_version")
        instruments = tuple(
            orderbook_models.Instrument(item)
            for item in _strings(data=data, key="instruments")
        )
        return research_models.TemporalEvaluationPolicy(
            evaluation_year=_integer(data=data, key="evaluation_year"),
            instruments=instruments,
            primary_instrument=orderbook_models.Instrument(
                _string(data=data, key="primary_instrument")
            ),
            minimum_common_eligible_sessions=_integer(
                data=data, key="minimum_common_eligible_sessions"
            ),
            session_selection=_string(data=data, key="session_selection"),
        )
    except (TypeError, ValueError) as exc:
        raise UnableToLoadTemporalEvaluationPolicyError(
            f"invalid temporal evaluation policy: {exc}"
        ) from exc


def _integer(*, data: dict[str, object], key: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _string(*, data: dict[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _strings(*, data: dict[str, object], key: str) -> tuple[str, ...]:
    value = data.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{key} must be a list of strings")
    return tuple(value)
