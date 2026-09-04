"""Test reusable direction-evaluation operations."""

from __future__ import annotations

import datetime
import warnings

import numpy as np
import pytest

from ebs_tft.domain import model
from ebs_tft.domain.pilot import training as pilot_training


class TestNormalizedProbabilities:
    def test_normalizes_each_valid_three_class_row(self) -> None:
        probabilities = np.array([[2.0, 1.0, 1.0], [1.0, 3.0, 0.0]])

        actual = model.normalized_probabilities(probabilities=probabilities)

        assert np.allclose(actual.sum(axis=1), 1.0)
        assert np.allclose(actual[0], [0.5, 0.25, 0.25])

    @pytest.mark.parametrize(
        "probabilities",
        (
            np.array([1.0, 0.0, 0.0]),
            np.array([[1.0, -1.0, 1.0]]),
            np.array([[0.0, 0.0, 0.0]]),
        ),
    )
    def test_rejects_an_invalid_probability_matrix(
        self, probabilities: np.ndarray
    ) -> None:
        with pytest.raises(ValueError, match="probabilit"):
            model.normalized_probabilities(probabilities=probabilities)


class TestDirectionMetricRow:
    def test_uses_all_three_declared_classes_for_balanced_accuracy(self) -> None:
        labels = np.array([0, 0], dtype=np.int64)
        probabilities = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            actual = model.direction_metric_row(
                model_name="majority",
                depth=1,
                horizon_steps=1,
                seed=-1,
                labels=labels,
                probabilities=probabilities,
                parameter_count=0,
            )

        assert actual["balanced_accuracy"] == 0.0


class TestDefensiveBaselineModel:
    def test_predicts_each_baseline_for_one_compact_session(self) -> None:
        training_sessions = (_baseline_session(labels=np.array([0, 1, 2, 0, 1, 2])),)
        evaluation = _baseline_session(labels=np.array([0, 2]))

        fitted = model.fit_defensive_baseline_model(sessions=training_sessions)
        actual = model.predict_defensive_baselines(fitted=fitted, evaluation=evaluation)

        assert set(actual) == {"empirical_prior", "last_move", "majority", "logistic"}
        assert all(item.shape == (2, 3) for item in actual.values())
        assert all(np.allclose(item.sum(axis=1), 1.0) for item in actual.values())

    def test_matches_the_full_corpus_baseline_implementation(self) -> None:
        raw_training = _raw_session(
            trading_date=datetime.date(2024, 1, 2),
            labels=np.array([0, 1, 2, 0, 1, 2, 0, 1]),
        )
        raw_evaluation = _raw_session(
            trading_date=datetime.date(2024, 1, 3),
            labels=np.array([2, 1, 0, 2, 1, 0, 2, 1]),
        )
        scaler = pilot_training.fit_feature_scaler(sessions=(raw_training,))
        scaled_training = pilot_training.apply_feature_scaler(
            session=raw_training, scaler=scaler
        )
        scaled_evaluation = pilot_training.apply_feature_scaler(
            session=raw_evaluation, scaler=scaler
        )
        training_corpus = pilot_training.combine_sessions(
            sessions=(scaled_training,),
            context_steps=2,
            horizon_steps=1,
            maximum_windows=None,
        )
        evaluation_corpus = pilot_training.combine_sessions(
            sessions=(scaled_evaluation,),
            context_steps=2,
            horizon_steps=1,
            maximum_windows=None,
        )
        compact_training = pilot_training.prepare_baseline_session(
            session=raw_training,
            scaler=scaler,
            context_steps=2,
            horizon_steps=1,
        )
        compact_evaluation = pilot_training.prepare_baseline_session(
            session=raw_evaluation,
            scaler=scaler,
            context_steps=2,
            horizon_steps=1,
        )

        expected = model.fit_defensive_baselines(
            training=training_corpus, evaluation=evaluation_corpus
        )
        fitted = model.fit_defensive_baseline_model(sessions=(compact_training,))
        actual = model.predict_defensive_baselines(
            fitted=fitted, evaluation=compact_evaluation
        )

        assert all(np.allclose(actual[name], expected[name]) for name in expected)


def _baseline_session(*, labels: np.ndarray) -> pilot_training.PreparedBaselineSession:
    observations = len(labels)
    return pilot_training.PreparedBaselineSession(
        trading_date=datetime.date(2024, 1, 2),
        features=np.column_stack(
            (np.arange(observations, dtype=np.float32), labels.astype(np.float32))
        ),
        labels=labels,
        mid_prices=np.arange(observations, dtype=np.float64) + 100.0,
        previous_mid_prices=np.arange(observations, dtype=np.float64) + 99.0,
    )


def _raw_session(
    *, trading_date: datetime.date, labels: np.ndarray
) -> pilot_training.RawSessionData:
    observations = len(labels)
    positions = np.arange(observations, dtype=np.float32)
    return pilot_training.RawSessionData(
        trading_date=trading_date,
        lob_features=np.repeat(positions[:, None, None], 6, axis=2),
        auxiliary_features=np.repeat(positions[:, None], 10, axis=1),
        labels=labels,
        timestamps=(
            np.datetime64("2024-01-01T00:00:00.000")
            + np.arange(observations) * np.timedelta64(100, "ms")
        ),
        mid_prices=np.arange(observations, dtype=np.float64) + 100.0,
        observed=np.ones(observations, dtype=np.bool_),
    )
