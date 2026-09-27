"""Test the frozen retrospective Level-1 versus Level-10 replication."""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import polars as pl

from ebs_tft.application.usecases import research_protocol
from ebs_tft.application.usecases.research_protocol import _baseline, _depth_replication
from ebs_tft.domain.orderbook import models as orderbook_models
from ebs_tft.domain.research import models as research_models


def _protocol() -> research_models.ResearchProtocol:
    return research_protocol.load_protocol(
        path=Path("notebooks/research_2023_2024_protocol.yaml")
    )


def test_replication_decision_requires_both_primary_metrics() -> None:
    comparisons = pl.DataFrame(
        {
            "model": ["deeplob_direction", "deeplob_direction"],
            "metric": ["macro_f1", "mcc"],
            "metric_passed": [True, False],
        }
    )

    actual = _depth_replication._replication_decision(
        comparisons=comparisons, protocol=_protocol()
    )

    assert actual["deeper_depth_supported_by_model"] == {"deeplob_direction": False}
    assert actual["supported_models"] == []
    assert actual["retrospective_replication"] is True
    assert actual["march_outcomes_previously_inspected"] is True
    assert actual["retuning_permitted"] is False


def test_development_cells_freeze_best_epoch_per_seed(tmp_path: Path) -> None:
    training = research_models.SessionIdentity(
        instrument=orderbook_models.Instrument.EUR_USD,
        trading_date=datetime.date(2023, 12, 28),
        path=tmp_path / "training.csv.gz",
        sha256="training",
    )
    validation = research_models.SessionIdentity(
        instrument=orderbook_models.Instrument.EUR_USD,
        trading_date=datetime.date(2024, 1, 2),
        path=tmp_path / "validation.csv.gz",
        sha256="validation",
    )
    fold = research_models.RollingFold(
        identifier="fold_01",
        training_sessions=(training,),
        validation_sessions=(validation,),
    )
    history_dir = (
        tmp_path
        / "cells"
        / "fold_01"
        / "h30000"
        / "depth_10"
        / "deeplob_direction"
        / "seed_7"
    )
    history_dir.mkdir(parents=True)
    history_path = history_dir / "training_history.json"
    history_path.write_text(json.dumps({"best_epoch": 4}), encoding="utf-8")
    (history_dir / "cell_summary.json").write_text(
        json.dumps(
            {
                "artifact_sha256": {
                    history_path.name: _baseline._sha256_file(path=history_path)
                }
            }
        ),
        encoding="utf-8",
    )

    actual = _depth_replication._development_cells(
        depth_dir=tmp_path,
        folds=(fold,),
        models=("deeplob_direction",),
        seeds=(7,),
    )

    assert len(actual) == 1
    assert actual[0]["model"] == "deeplob_direction"
    assert actual[0]["depth"] == 10
    assert actual[0]["fixed_epochs"] == 4
    assert actual[0]["development_best_epochs"] == [
        {
            "fold": "fold_01",
            "best_epoch": 4,
            "history_sha256": _baseline._sha256_file(path=history_path),
        }
    ]
