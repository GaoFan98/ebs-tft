"""Test the external-year temporal-evaluation policy boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

from ebs_tft.application.usecases import research_protocol


def test_loads_frozen_temporal_policy(tmp_path: Path) -> None:
    path = tmp_path / "temporal.yaml"
    path.write_text(_yaml(), encoding="utf-8")

    actual = research_protocol.load_temporal_policy(path=path)

    assert actual.evaluation_year == 2023
    assert actual.primary_instrument.value == "EUR_USD"
    assert len(actual.instruments) == 3


def test_rejects_adaptive_session_selection(tmp_path: Path) -> None:
    path = tmp_path / "temporal.yaml"
    path.write_text(
        _yaml().replace(
            "all_common_technically_eligible_dates", "manually_selected_dates"
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        research_protocol.UnableToLoadTemporalEvaluationPolicyError,
        match="session_selection",
    ):
        research_protocol.load_temporal_policy(path=path)


def _yaml() -> str:
    return """\
schema_version: 1
evaluation_year: 2023
instruments: [EUR_USD, USD_JPY, EUR_JPY]
primary_instrument: EUR_USD
minimum_common_eligible_sessions: 20
session_selection: all_common_technically_eligible_dates
"""
