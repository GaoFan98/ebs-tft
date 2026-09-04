"""Test loading the frozen neural benchmark optimization policy."""

from __future__ import annotations

from pathlib import Path

import pytest

from ebs_tft.application.usecases import research_protocol


class TestLoadPolicy:
    def test_returns_a_validated_policy(self, tmp_path: Path) -> None:
        path = tmp_path / "policy.yaml"
        path.write_text(_policy_yaml(device="cpu"), encoding="utf-8")

        actual = research_protocol.load_policy(path=path)

        assert actual.maximum_epochs == 2
        assert actual.hidden_size == 8
        assert actual.device == "cpu"

    def test_rejects_an_unknown_key(self, tmp_path: Path) -> None:
        path = tmp_path / "policy.yaml"
        path.write_text(_policy_yaml(device="cpu") + "extra: true\n", encoding="utf-8")

        with pytest.raises(
            research_protocol.UnableToLoadNeuralBenchmarkPolicyError,
            match="keys differ",
        ):
            research_protocol.load_policy(path=path)


def _policy_yaml(*, device: str) -> str:
    return f"""\
schema_version: 1
maximum_epochs: 2
early_stopping_patience: 1
early_stopping_minimum_delta: 0.0001
gradient_clip_norm: 1.0
batch_size: 8
learning_rate: 0.0003
weight_decay: 0.0001
hidden_size: 8
device: {device}
"""
