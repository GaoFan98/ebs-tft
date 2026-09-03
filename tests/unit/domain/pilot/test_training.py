"""Test day-aware pilot tensor preparation."""

from __future__ import annotations

import datetime

import numpy as np

from ebs_tft.domain import model
from ebs_tft.domain.pilot import training


class TestCombineSessions:
    def test_keeps_every_context_window_inside_its_session(self) -> None:
        earlier = _session(trading_date=datetime.date(2024, 1, 3), value=1.0)
        later = _session(trading_date=datetime.date(2024, 2, 1), value=2.0)

        actual = training.combine_sessions(
            sessions=(earlier, later),
            context_steps=3,
            horizon_steps=2,
            maximum_windows=None,
        )
        dataset = model.SequenceDataset(
            lob_features=actual.lob_features,
            auxiliary_features=actual.auxiliary_features,
            labels=actual.labels,
            target_indices=actual.target_indices,
            context_steps=3,
        )

        assert actual.target_indices.tolist() == [2, 3, 4, 5, 10, 11, 12, 13]
        first_later_window = dataset[4][0].numpy()
        assert np.all(first_later_window == 2.0)
        assert tuple(item.selected for item in actual.session_windows) == (4, 4)

    def test_applies_a_global_cap_without_creating_boundary_indices(self) -> None:
        sessions = (
            _session(trading_date=datetime.date(2024, 1, 3), value=1.0),
            _session(trading_date=datetime.date(2024, 2, 1), value=2.0),
        )

        actual = training.combine_sessions(
            sessions=sessions,
            context_steps=3,
            horizon_steps=2,
            maximum_windows=4,
        )

        assert len(actual.target_indices) == 4
        assert set(actual.target_indices).issubset({2, 3, 4, 5, 10, 11, 12, 13})
        assert sum(item.selected for item in actual.session_windows) == 4

    def test_applies_stride_independently_inside_each_session(self) -> None:
        earlier = _session(trading_date=datetime.date(2024, 1, 3), value=1.0)
        later = _session(trading_date=datetime.date(2024, 2, 1), value=2.0)

        actual = training.combine_sessions(
            sessions=(earlier, later),
            context_steps=3,
            horizon_steps=2,
            maximum_windows=None,
            stride_steps=3,
        )

        assert actual.target_indices.tolist() == [2, 5, 10, 13]
        assert tuple(item.candidates for item in actual.session_windows) == (4, 4)
        assert tuple(item.selected for item in actual.session_windows) == (2, 2)
        assert tuple(item.stride_steps for item in actual.session_windows) == (3, 3)


class TestFitFeatureScaler:
    def test_uses_only_the_sessions_explicitly_supplied_for_training(self) -> None:
        training_session = _session(trading_date=datetime.date(2024, 1, 3), value=1.0)
        validation_session = _session(
            trading_date=datetime.date(2024, 3, 1), value=100.0
        )

        scaler = training.fit_feature_scaler(sessions=(training_session,))
        transformed_validation = training.apply_feature_scaler(
            session=validation_session, scaler=scaler
        )

        assert np.all(scaler.lob_means == 1.0)
        assert np.all(scaler.auxiliary_means == 1.0)
        assert np.all(transformed_validation.lob_features == 99.0)
        assert np.all(transformed_validation.auxiliary_features == 99.0)


def _session(*, trading_date: datetime.date, value: float) -> training.RawSessionData:
    rows = 8
    return training.RawSessionData(
        trading_date=trading_date,
        lob_features=np.full((rows, 1, 6), value, dtype=np.float32),
        auxiliary_features=np.full((rows, 10), value, dtype=np.float32),
        labels=np.zeros(rows, dtype=np.int64),
        timestamps=(
            np.datetime64("2024-01-01T00:00:00.000")
            + np.arange(rows) * np.timedelta64(100, "ms")
        ),
        mid_prices=np.full(rows, 100.0, dtype=np.float64),
        observed=np.ones(rows, dtype=np.bool_),
    )
