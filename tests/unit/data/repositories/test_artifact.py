"""Test bounded experiment artifact directory preparation."""

from __future__ import annotations

from pathlib import Path

import pytest

from ebs_tft.data.repositories import artifact


class TestPrepareRunDirectory:
    def test_replaces_a_direct_child_of_a_named_output_directory(
        self, tmp_path: Path
    ) -> None:
        parent = tmp_path / "research_outputs"
        output = parent / "baseline_gate"
        output.mkdir(parents=True)
        (output / "partial.csv").write_text("partial", encoding="utf-8")

        artifact.prepare_run_directory(
            path=output, replace=True, replacement_parent=parent
        )

        assert output.is_dir()
        assert not (output / "partial.csv").exists()

    def test_rejects_replacement_outside_the_explicit_parent(
        self, tmp_path: Path
    ) -> None:
        parent = tmp_path / "research_outputs"
        output = tmp_path / "other" / "baseline_gate"
        output.mkdir(parents=True)

        with pytest.raises(ValueError, match="bounded"):
            artifact.prepare_run_directory(
                path=output, replace=True, replacement_parent=parent
            )
