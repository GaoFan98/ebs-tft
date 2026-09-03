"""Test reusable direction-evaluation operations."""

from __future__ import annotations

import numpy as np
import pytest

from ebs_tft.domain import model


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
