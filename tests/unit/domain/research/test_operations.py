"""Test research-protocol split, sampling, and inference operations."""

from __future__ import annotations

import datetime
from pathlib import Path

import pytest

from ebs_tft.domain.orderbook import models as orderbook_models
from ebs_tft.domain.research import models, operations


class TestEvaluateSessionEligibility:
    def test_reports_every_technical_failure_without_using_target_balance(self) -> None:
        policy = models.AuditPolicy(
            minimum_duration_milliseconds=70_000,
            minimum_observed_states=700,
            required_depth=10,
            redact_locked_outcomes=True,
        )

        eligible, reasons = operations.evaluate_session_eligibility(
            duration_milliseconds=10,
            required_depth_observed_states=0,
            maximum_source_depth=1,
            parse_error=None,
            policy=policy,
        )

        assert eligible is False
        assert reasons == (
            "insufficient_duration",
            "insufficient_required_depth_observed_states",
            "insufficient_source_depth",
        )

    def test_rejects_a_policy_that_exposes_locked_outcomes(self) -> None:
        with pytest.raises(ValueError, match="must remain redacted"):
            models.AuditPolicy(
                minimum_duration_milliseconds=1,
                minimum_observed_states=1,
                required_depth=10,
                redact_locked_outcomes=False,
            )


class TestBuildRollingFolds:
    def test_rejects_overlapping_validation_blocks(self) -> None:
        with pytest.raises(ValueError, match="validation session blocks disjoint"):
            models.SplitPolicy(
                development_end_date=datetime.date(2024, 1, 8),
                minimum_training_sessions=3,
                validation_sessions_per_fold=2,
                fold_step_sessions=1,
                locked_evaluation_dates=(datetime.date(2024, 1, 4),),
            )

    def test_excludes_locked_dates_and_preserves_expanding_chronology(self) -> None:
        sessions = tuple(_identity(day=day) for day in range(1, 9))
        policy = models.SplitPolicy(
            development_end_date=datetime.date(2024, 1, 8),
            minimum_training_sessions=3,
            validation_sessions_per_fold=2,
            fold_step_sessions=2,
            locked_evaluation_dates=(datetime.date(2024, 1, 4),),
        )

        actual = operations.build_rolling_folds(sessions=sessions, policy=policy)

        assert len(actual) == 2
        assert tuple(item.trading_date.day for item in actual[0].training_sessions) == (
            1,
            2,
            3,
        )
        assert tuple(
            item.trading_date.day for item in actual[0].validation_sessions
        ) == (5, 6)
        assert tuple(item.trading_date.day for item in actual[1].training_sessions) == (
            1,
            2,
            3,
            5,
            6,
        )
        assert tuple(
            item.trading_date.day for item in actual[1].validation_sessions
        ) == (7, 8)


class TestPairedSessionInterval:
    def test_returns_a_deterministic_session_block_interval(self) -> None:
        shallower = {"early": 0.2, "middle": 0.3, "late": 0.4}
        deeper = {"early": 0.3, "middle": 0.5, "late": 0.7}

        actual = operations.paired_session_interval(
            shallower_by_session=shallower,
            deeper_by_session=deeper,
            repetitions=1_000,
            confidence_level=0.95,
            random_seed=7,
        )

        assert actual["sessions"] == 3
        assert actual["mean_delta"] == pytest.approx(0.2)
        assert float(actual["confidence_lower"]) > 0.0


def _identity(*, day: int) -> models.SessionIdentity:
    trading_date = datetime.date(2024, 1, day)
    return models.SessionIdentity(
        instrument=orderbook_models.Instrument.EUR_USD,
        trading_date=trading_date,
        path=Path(f"{trading_date.isoformat()}.csv.gz"),
        sha256=str(day) * 64,
    )
