"""Exercise the resumable comparison without reading licensed raw files."""

from __future__ import annotations

import datetime
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import polars as pl
import pytest

from ebs_tft.application.usecases.research_protocol import (
    _baseline,
    _locked,
    _neural,
    _paper_lstm,
)
from ebs_tft.domain.pilot import training as pilot_training
from ebs_tft.domain.research import models as research_models


def _corpus(*, dates: tuple[datetime.date, ...]) -> pilot_training.PreparedCorpus:
    rows_per_session = 105
    row_count = rows_per_session * len(dates)
    labels = np.arange(row_count, dtype=np.int64) % 3
    indices = np.asarray(
        [
            offset + target
            for offset in range(0, row_count, rows_per_session)
            for target in (99, 100)
        ],
        dtype=np.int64,
    )
    return pilot_training.PreparedCorpus(
        lob_features=np.zeros((row_count, 1, 6), dtype=np.float32),
        auxiliary_features=np.zeros((row_count, 10), dtype=np.float32),
        labels=labels,
        timestamps=np.arange(row_count),
        mid_prices=np.ones(row_count),
        target_indices=indices,
        session_offsets=tuple(range(0, row_count, rows_per_session)),
        session_lengths=(rows_per_session,) * len(dates),
        session_windows=(),
    )


def _reference(*, dates: tuple[datetime.date, ...]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "model": model_name,
                "seed": seed,
                "validation_date": date,
                "depth": 1,
                "horizon_milliseconds": 30000,
                "observations": 2,
                "accuracy": 0.5,
                "macro_f1": 0.4,
                "mcc": 0.1,
            }
            for model_name in ("deeplob_direction", "tft_direction")
            for seed in (7, 19)
            for date in dates
        ]
    )


class TestRun:
    def test_development_resume_and_retrospective_replication(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        development_dates = (
            datetime.date(2024, 1, 2),
            datetime.date(2024, 1, 3),
        )
        march_dates = (
            datetime.date(2024, 3, 6),
            datetime.date(2024, 3, 13),
        )
        training_sessions_dates = (datetime.date(2023, 6, 1),)
        output_dir = tmp_path / "study"
        output_dir.mkdir()
        protocol_path = tmp_path / "protocol.yaml"
        policy_path = tmp_path / "policy.yaml"
        for path in (
            protocol_path,
            policy_path,
            output_dir / "split_manifest.yaml",
            output_dir / "session_audit.csv",
        ):
            path.write_text("frozen input\n")
        for stage_dir, dates in (
            ("neural_benchmark", development_dates),
            ("locked_evaluation", march_dates),
        ):
            reference_dir = output_dir / stage_dir
            reference_dir.mkdir()
            _reference(dates=dates).write_csv(reference_dir / "session_metrics.csv")
        training_sessions = tuple(
            SimpleNamespace(trading_date=date) for date in training_sessions_dates
        )
        validation_sessions = tuple(
            SimpleNamespace(trading_date=date) for date in development_dates
        )
        march_sessions = tuple(
            SimpleNamespace(trading_date=date) for date in march_dates
        )
        fold = SimpleNamespace(
            training_sessions=training_sessions,
            validation_sessions=validation_sessions,
        )
        protocol = cast(
            research_models.ResearchProtocol,
            SimpleNamespace(
                output_dir=output_dir,
                development_instrument=SimpleNamespace(value="EUR_USD"),
                years=(2023, 2024),
                forecast_horizons_milliseconds=(30000,),
                horizon_steps=(300,),
                depths=(1,),
                state_interval_milliseconds=100,
                context_milliseconds=10000,
                models=("deeplob_direction", "tft_direction"),
                random_seeds=(7, 19),
                validation_checks_per_epoch=2,
                bootstrap_repetitions=1000,
                confidence_level=0.95,
            ),
        )
        policy = cast(
            research_models.NeuralBenchmarkPolicy,
            SimpleNamespace(
                device="cpu",
                maximum_epochs=1,
                batch_size=2,
                evaluation_batch_size=4,
                learning_rate=0.001,
                early_stopping_patience=1,
                early_stopping_minimum_delta=0.0,
                gradient_clip_norm=1.0,
            ),
        )
        monkeypatch.setattr(_baseline, "_load_folds", lambda **kwargs: (fold,))
        monkeypatch.setattr(_baseline, "_verify_cached_states", lambda **kwargs: None)
        monkeypatch.setattr(
            _neural,
            "_prepare_corpora",
            lambda **kwargs: (
                _corpus(dates=training_sessions_dates),
                _corpus(dates=development_dates),
            ),
        )
        monkeypatch.setattr(_locked, "_final_sessions", lambda **kwargs: march_sessions)
        monkeypatch.setattr(_locked, "_verify_sources", lambda **kwargs: None)
        monkeypatch.setattr(_locked, "_materialize_locked_cache", lambda **kwargs: None)
        monkeypatch.setattr(
            _locked,
            "_prepare_final_corpora",
            lambda **kwargs: (
                _corpus(dates=training_sessions_dates),
                _corpus(dates=march_dates),
            ),
        )

        with pytest.raises(ValueError, match="complete paper-LSTM development"):
            _paper_lstm.run(
                protocol=protocol,
                protocol_path=protocol_path,
                policy=policy,
                policy_path=policy_path,
                stage="replication",
            )
        with pytest.raises(_paper_lstm.PaperLstmPausedError):
            _paper_lstm.run(
                protocol=protocol,
                protocol_path=protocol_path,
                policy=policy,
                policy_path=policy_path,
                stage="development",
                maximum_new_cells=1,
            )
        development = _paper_lstm.run(
            protocol=protocol,
            protocol_path=protocol_path,
            policy=policy,
            policy_path=policy_path,
            stage="development",
        )
        with pytest.raises(_paper_lstm.PaperLstmPausedError):
            _paper_lstm.run(
                protocol=protocol,
                protocol_path=protocol_path,
                policy=policy,
                policy_path=policy_path,
                stage="replication",
                maximum_new_cells=1,
            )
        replication = _paper_lstm.run(
            protocol=protocol,
            protocol_path=protocol_path,
            policy=policy,
            policy_path=policy_path,
            stage="replication",
        )

        assert pl.read_csv(development.metrics_path).height == 8
        assert pl.read_csv(replication.metrics_path).height == 8
        assert pl.read_csv(replication.comparisons_path).height == 9
        assert pl.read_csv(replication.output_dir / "absolute_summary.csv").height == 4
        details = pl.read_csv(replication.output_dir / "training_details.csv")
        assert details.height == 4
        assert set(details["training_windows"]) == {2}
        assert set(details["evaluation_windows"]) == {4}
        with zipfile.ZipFile(
            replication.output_dir / "paper_lstm_comparison.xlsx"
        ) as archive:
            assert "xl/workbook.xml" in archive.namelist()
        assert (
            _paper_lstm.run(
                protocol=protocol,
                protocol_path=protocol_path,
                policy=policy,
                policy_path=policy_path,
                stage="replication",
            )
            == replication
        )
        checkpoint = (
            replication.output_dir
            / "cells"
            / "level_1_and_transactions"
            / "seed_7"
            / "final.pt"
        )
        checkpoint.write_bytes(b"corrupt")
        with pytest.raises(ValueError, match="artifact is corrupt"):
            _paper_lstm.run(
                protocol=protocol,
                protocol_path=protocol_path,
                policy=policy,
                policy_path=policy_path,
                stage="replication",
            )
