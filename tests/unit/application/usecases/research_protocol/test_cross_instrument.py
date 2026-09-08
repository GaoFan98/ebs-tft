"""Test cross-instrument evidence verification helpers."""

from ebs_tft.application.usecases.research_protocol import _cross_instrument


def test_locked_decision_candidate_order_is_semantically_irrelevant() -> None:
    """Accept an equivalent stored candidate list regardless of row ordering."""
    first = {"model": "deeplob_direction", "depth": 1, "horizon_milliseconds": 30000}
    second = {"model": "tft_direction", "depth": 1, "horizon_milliseconds": 30000}

    left = _cross_instrument._normalized_locked_decision(
        {"confirmed_candidates": [first, second], "locked_evaluation_used": True}
    )
    right = _cross_instrument._normalized_locked_decision(
        {"confirmed_candidates": [second, first], "locked_evaluation_used": True}
    )

    assert left == right
