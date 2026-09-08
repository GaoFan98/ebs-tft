"""Exercise the pre-GPU research protocol across real file boundaries."""

from __future__ import annotations

import datetime
import gzip
import json
from pathlib import Path

import polars as pl
import pytest
import yaml

from ebs_tft.application.usecases import research_protocol
from ebs_tft.domain.orderbook import models as orderbook_models
from ebs_tft.domain.research import models as research_models


def test_audit_manifest_and_baseline_gate_remain_chronological(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "raw"
    output_dir = tmp_path / "outputs"
    dates = tuple(datetime.date(2024, 1, day) for day in range(1, 7))
    for trading_date in dates:
        for instrument in orderbook_models.Instrument:
            _write_session(
                data_dir=data_dir,
                trading_date=trading_date,
                instrument=instrument,
            )
    temporal_dates = (
        datetime.date(2023, 12, 28),
        datetime.date(2023, 12, 29),
    )
    for trading_date in temporal_dates:
        for instrument in orderbook_models.Instrument:
            _write_session(
                data_dir=data_dir,
                trading_date=trading_date,
                instrument=instrument,
            )
    protocol = _protocol(
        data_dir=data_dir, output_dir=output_dir, locked_dates=dates[-2:]
    )
    protocol_path = tmp_path / "protocol.yaml"
    protocol_path.write_text("schema_version: 1\n", encoding="utf-8")

    audit = research_protocol.run_session_audit(
        protocol=protocol, protocol_path=protocol_path, replace_output=False
    )

    audit_data = pl.read_csv(audit.audit_path)
    assert audit_data.height == 18
    locked = audit_data.filter(pl.col("evaluation_locked"))
    assert locked.height == 6
    assert locked["total_h100"].null_count() == 6
    with audit.manifest_path.open(encoding="utf-8") as stream:
        manifest = yaml.safe_load(stream)
    folds = manifest["development_folds"]["EUR_USD"]
    assert len(folds) == 3
    assert folds[0]["training_sessions"][0]["trading_date"] == "2024-01-01"
    assert folds[-1]["validation_sessions"][0]["trading_date"] == "2024-01-04"
    assert [
        item["trading_date"] for item in manifest["final_test_sessions"]["EUR_USD"]
    ] == ["2024-01-05", "2024-01-06"]

    baseline = research_protocol.run_baseline_gate(
        protocol=protocol, protocol_path=protocol_path, replace_output=False
    )

    metrics = pl.read_csv(baseline.metrics_path)
    comparisons = pl.read_csv(baseline.comparisons_path)
    assert metrics["validation_date"].n_unique() == 3
    assert set(metrics["depth"].unique()) == {1, 10}
    assert set(metrics["model"].unique()) == {
        "empirical_prior",
        "last_move",
        "logistic",
        "majority",
    }
    assert comparisons["sessions"].min() == 3
    assert baseline.gate_path.is_file()

    (baseline.output_dir / "run_summary.json").unlink()
    resumed = research_protocol.run_baseline_gate(
        protocol=protocol, protocol_path=protocol_path, replace_output=False
    )
    assert "resumed_folds=3" in resumed.terminal_summary_path.read_text(
        encoding="utf-8"
    )

    replaced = research_protocol.run_baseline_gate(
        protocol=protocol, protocol_path=protocol_path, replace_output=True
    )
    assert "resumed_folds=0" in replaced.terminal_summary_path.read_text(
        encoding="utf-8"
    )

    research_protocol.run_model_protocol_verification(
        protocol=protocol, replace_output=False
    )
    research_protocol.run_model_protocol_verification(
        protocol=protocol, replace_output=True
    )
    gate = json.loads(replaced.gate_path.read_text(encoding="utf-8"))
    gate["baseline_signal_by_horizon"] = {"100": True}
    gate["depth_support_by_horizon"] = {"100": True}
    gate["eligible_for_neural_benchmark"] = True
    replaced.gate_path.write_text(json.dumps(gate, indent=2), encoding="utf-8")
    policy_path = tmp_path / "neural_policy.yaml"
    policy_path.write_text("schema_version: 2\n", encoding="utf-8")
    policy = research_models.NeuralBenchmarkPolicy(
        maximum_epochs=1,
        early_stopping_patience=1,
        early_stopping_minimum_delta=0.0001,
        gradient_clip_norm=1.0,
        batch_size=8,
        evaluation_batch_size=32,
        learning_rate=0.0003,
        weight_decay=0.0001,
        hidden_size=8,
        device="cpu",
    )

    with pytest.raises(research_protocol.NeuralBenchmarkPausedError) as pause:
        research_protocol.run_neural_benchmark(
            protocol=protocol,
            protocol_path=protocol_path,
            policy=policy,
            policy_path=policy_path,
            replace_output=False,
            maximum_new_cells=1,
        )
    assert pause.value.completed_cells == 1
    assert (pause.value.output_dir / "progress_summary.json").is_file()

    neural = research_protocol.run_neural_benchmark(
        protocol=protocol,
        protocol_path=protocol_path,
        policy=policy,
        policy_path=policy_path,
        replace_output=False,
    )

    neural_metrics = pl.read_csv(neural.metrics_path)
    assert neural_metrics.height == 24
    assert neural_metrics["validation_date"].n_unique() == 3
    assert set(neural_metrics["depth"].unique()) == {1, 10}
    comparisons = pl.read_csv(neural.comparisons_path)
    assert set(comparisons["depth"].unique()) == {1, 10}
    assert neural.gate_path.is_file()
    cell_summary = json.loads(
        next((neural.output_dir / "cells").rglob("cell_summary.json")).read_text(
            encoding="utf-8"
        )
    )
    assert cell_summary["training_batch_size"] == 8
    assert cell_summary["evaluation_batch_size"] == 32
    assert cell_summary["training_windows"] > 0
    assert cell_summary["validation_windows"] > 0
    assert cell_summary["fit_elapsed_seconds"] >= 0
    assert cell_summary["validation_elapsed_seconds"] >= 0
    assert "resumed_cells=1" in neural.terminal_summary_path.read_text(encoding="utf-8")
    assert not (neural.output_dir / "progress_summary.json").exists()
    prediction_dates = set(
        pl.scan_parquet(neural.output_dir / "cells" / "**" / "predictions.parquet")
        .select("validation_date")
        .collect()["validation_date"]
        .unique()
    )
    assert not set(dates[-2:]) & prediction_dates

    (neural.output_dir / "run_summary.json").unlink()
    resumed_neural = research_protocol.run_neural_benchmark(
        protocol=protocol,
        protocol_path=protocol_path,
        policy=policy,
        policy_path=policy_path,
        replace_output=False,
    )
    assert "resumed_cells=24" in resumed_neural.terminal_summary_path.read_text(
        encoding="utf-8"
    )

    neural_gate = json.loads(neural.gate_path.read_text(encoding="utf-8"))
    neural_comparisons = pl.read_csv(neural.comparisons_path).with_columns(
        pl.when(
            (pl.col("model") == "deeplob_direction")
            & (pl.col("depth") == 1)
            & pl.col("metric").is_in(["macro_f1", "mcc"])
        )
        .then(0.01)
        .otherwise(pl.col("confidence_lower"))
        .alias("confidence_lower")
    )
    neural_comparisons.write_csv(neural.comparisons_path)
    neural_gate["neural_signal_by_model_horizon"]["deeplob_direction:d1:h100"] = True
    neural_gate["accepted_model_depth_horizons"] = [
        {
            "model": "deeplob_direction",
            "depth": 1,
            "horizon_milliseconds": 100,
        }
    ]
    neural.gate_path.write_text(json.dumps(neural_gate, indent=2), encoding="utf-8")
    plan = research_protocol.freeze_locked_evaluation_plan(
        protocol=protocol,
        protocol_path=protocol_path,
        policy=policy,
        policy_path=policy_path,
    )
    frozen = json.loads(plan.plan_path.read_text(encoding="utf-8"))
    assert frozen["locked_outcomes_inspected"] is False
    assert len(frozen["development_sessions"]) == 4
    assert len(frozen["final_test_sessions"]) == 2
    assert len(frozen["cells"]) == 2

    with pytest.raises(ValueError, match="plan-sha256"):
        research_protocol.run_locked_evaluation(
            protocol=protocol,
            protocol_path=protocol_path,
            policy=policy,
            policy_path=policy_path,
            plan_sha256="0" * 64,
        )

    with pytest.raises(research_protocol.LockedEvaluationPausedError):
        research_protocol.run_locked_evaluation(
            protocol=protocol,
            protocol_path=protocol_path,
            policy=policy,
            policy_path=policy_path,
            plan_sha256=plan.plan_sha256,
            maximum_new_cells=1,
        )
    locked = research_protocol.run_locked_evaluation(
        protocol=protocol,
        protocol_path=protocol_path,
        policy=policy,
        policy_path=policy_path,
        plan_sha256=plan.plan_sha256,
    )
    locked_metrics = pl.read_csv(locked.metrics_path)
    assert locked_metrics.height == 6
    assert set(locked_metrics["model"]) == {"deeplob_direction", "logistic"}
    assert set(locked_metrics["validation_date"]) == {
        str(dates[-2]),
        str(dates[-1]),
    }
    decision = json.loads(locked.decision_path.read_text(encoding="utf-8"))
    assert decision["locked_evaluation_used"] is True
    assert decision["retuning_permitted"] is False
    with pytest.raises(ValueError, match="already complete"):
        research_protocol.run_locked_evaluation(
            protocol=protocol,
            protocol_path=protocol_path,
            policy=policy,
            policy_path=policy_path,
            plan_sha256=plan.plan_sha256,
        )

    locked_comparisons = pl.read_csv(locked.comparisons_path).with_columns(
        pl.when(
            (pl.col("model") == "deeplob_direction")
            & pl.col("metric").is_in(["macro_f1", "mcc"])
        )
        .then(0.01)
        .otherwise(pl.col("confidence_lower"))
        .alias("confidence_lower")
    )
    locked_comparisons.write_csv(locked.comparisons_path)
    decision["confirmed_by_candidate"]["deeplob_direction:d1:h100"] = True
    decision["confirmed_candidates"] = [
        {
            "model": "deeplob_direction",
            "depth": 1,
            "horizon_milliseconds": 100,
        }
    ]
    locked.decision_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    cross_plan = research_protocol.freeze_cross_instrument_plan(
        protocol=protocol,
        protocol_path=protocol_path,
        policy_path=policy_path,
    )
    frozen_cross = json.loads(cross_plan.plan_path.read_text(encoding="utf-8"))
    assert frozen_cross["target_outcomes_inspected"] is False
    assert frozen_cross["neural_retraining_permitted"] is False
    assert len(frozen_cross["cells"]) == 4

    with pytest.raises(research_protocol.CrossInstrumentPausedError):
        research_protocol.run_cross_instrument_evaluation(
            protocol=protocol,
            protocol_path=protocol_path,
            policy=policy,
            policy_path=policy_path,
            plan_sha256=cross_plan.plan_sha256,
            maximum_new_cells=1,
        )
    cross = research_protocol.run_cross_instrument_evaluation(
        protocol=protocol,
        protocol_path=protocol_path,
        policy=policy,
        policy_path=policy_path,
        plan_sha256=cross_plan.plan_sha256,
    )
    cross_metrics = pl.read_csv(cross.metrics_path)
    assert cross_metrics.height == 12
    assert set(cross_metrics["instrument"]) == {"EUR_JPY", "USD_JPY"}
    assert set(cross_metrics["model"]) == {"deeplob_direction", "logistic"}
    cross_decision = json.loads(cross.decision_path.read_text(encoding="utf-8"))
    assert cross_decision["cross_instrument_outcomes_used"] is True
    assert cross_decision["neural_retraining_used"] is False
    with pytest.raises(ValueError, match="complete"):
        research_protocol.run_cross_instrument_evaluation(
            protocol=protocol,
            protocol_path=protocol_path,
            policy=policy,
            policy_path=policy_path,
            plan_sha256=cross_plan.plan_sha256,
        )

    temporal_policy_path = tmp_path / "temporal_policy.yaml"
    temporal_policy_path.write_text(
        "\n".join(
            (
                "schema_version: 1",
                "evaluation_year: 2023",
                "instruments: [EUR_USD, USD_JPY, EUR_JPY]",
                "primary_instrument: EUR_USD",
                "minimum_common_eligible_sessions: 2",
                "session_selection: all_common_technically_eligible_dates",
                "",
            )
        ),
        encoding="utf-8",
    )
    temporal_policy = research_protocol.load_temporal_policy(path=temporal_policy_path)
    temporal_audit = research_protocol.run_temporal_audit(
        protocol=protocol,
        protocol_path=protocol_path,
        temporal_policy=temporal_policy,
        temporal_policy_path=temporal_policy_path,
    )
    temporal_audit_data = pl.read_csv(temporal_audit.audit_path)
    assert temporal_audit_data.height == 6
    assert temporal_audit_data["outcomes_redacted"].all()
    resumed_temporal_audit = research_protocol.run_temporal_audit(
        protocol=protocol,
        protocol_path=protocol_path,
        temporal_policy=temporal_policy,
        temporal_policy_path=temporal_policy_path,
    )
    assert pl.read_csv(resumed_temporal_audit.audit_path).height == 6
    temporal_plan = research_protocol.freeze_temporal_evaluation_plan(
        protocol=protocol,
        protocol_path=protocol_path,
        neural_policy_path=policy_path,
        temporal_policy=temporal_policy,
        temporal_policy_path=temporal_policy_path,
    )
    frozen_temporal = json.loads(temporal_plan.plan_path.read_text(encoding="utf-8"))
    assert frozen_temporal["temporal_outcomes_inspected"] is False
    assert frozen_temporal["neural_retraining_permitted"] is False
    assert len(frozen_temporal["cells"]) == 6
    with pytest.raises(research_protocol.TemporalEvaluationPausedError):
        research_protocol.run_temporal_evaluation(
            protocol=protocol,
            protocol_path=protocol_path,
            neural_policy=policy,
            neural_policy_path=policy_path,
            temporal_policy=temporal_policy,
            temporal_policy_path=temporal_policy_path,
            plan_sha256=temporal_plan.plan_sha256,
            maximum_new_sessions=1,
        )
    temporal = research_protocol.run_temporal_evaluation(
        protocol=protocol,
        protocol_path=protocol_path,
        neural_policy=policy,
        neural_policy_path=policy_path,
        temporal_policy=temporal_policy,
        temporal_policy_path=temporal_policy_path,
        plan_sha256=temporal_plan.plan_sha256,
    )
    temporal_metrics = pl.read_csv(temporal.metrics_path)
    assert temporal_metrics.height == 18
    assert set(temporal_metrics["instrument"]) == {
        "EUR_USD",
        "EUR_JPY",
        "USD_JPY",
    }
    temporal_decision = json.loads(temporal.decision_path.read_text(encoding="utf-8"))
    assert temporal_decision["temporal_outcomes_used"] is True
    assert temporal_decision["neural_retraining_used"] is False
    assert temporal_decision["retuning_permitted"] is False
    assert not (temporal.output_dir / "progress_summary.json").exists()
    with pytest.raises(ValueError, match="complete"):
        research_protocol.run_temporal_evaluation(
            protocol=protocol,
            protocol_path=protocol_path,
            neural_policy=policy,
            neural_policy_path=policy_path,
            temporal_policy=temporal_policy,
            temporal_policy_path=temporal_policy_path,
            plan_sha256=temporal_plan.plan_sha256,
        )
    assert decision["locked_evaluation_used"] is True
    assert decision["retuning_permitted"] is False
    with pytest.raises(ValueError, match="already complete"):
        research_protocol.run_locked_evaluation(
            protocol=protocol,
            protocol_path=protocol_path,
            policy=policy,
            policy_path=policy_path,
            plan_sha256=plan.plan_sha256,
        )


def _protocol(
    *,
    data_dir: Path,
    output_dir: Path,
    locked_dates: tuple[datetime.date, ...],
) -> research_models.ResearchProtocol:
    return research_models.ResearchProtocol(
        data_dir=data_dir,
        output_dir=output_dir,
        instruments=tuple(orderbook_models.Instrument),
        years=(2024,),
        state_interval_milliseconds=100,
        forecast_horizons_milliseconds=(100,),
        context_milliseconds=200,
        maximum_staleness_milliseconds=1_000,
        audit_workers=1,
        training_stride_milliseconds=((100, 100),),
        evaluation_stride_milliseconds=100,
        audit_policy=research_models.AuditPolicy(
            minimum_duration_milliseconds=1_000,
            minimum_observed_states=10,
            required_depth=10,
            redact_locked_outcomes=True,
        ),
        split_policy=research_models.SplitPolicy(
            development_end_date=datetime.date(2024, 1, 4),
            minimum_training_sessions=1,
            validation_sessions_per_fold=1,
            fold_step_sessions=1,
            locked_evaluation_dates=locked_dates,
        ),
        development_instrument=orderbook_models.Instrument.EUR_USD,
        depths=(1, 10),
        models=("deeplob_direction", "tft_direction"),
        random_seeds=(7, 19),
        validation_checks_per_epoch=2,
        primary_metrics=(
            research_models.EvaluationMetric.MACRO_F1,
            research_models.EvaluationMetric.MCC,
        ),
        supporting_metrics=(
            research_models.EvaluationMetric.BALANCED_ACCURACY,
            research_models.EvaluationMetric.LOG_LOSS,
            research_models.EvaluationMetric.MULTICLASS_BRIER,
        ),
        bootstrap_repetitions=1_000,
        confidence_level=0.95,
    )


def _write_session(
    *,
    data_dir: Path,
    trading_date: datetime.date,
    instrument: orderbook_models.Instrument,
) -> None:
    year_dir = data_dir / str(trading_date.year)
    year_dir.mkdir(parents=True, exist_ok=True)
    path = year_dir / (f"{trading_date:%Y%m%d}-EBS_LVL2_{instrument.value}_0.csv.gz")
    mid_offsets = (0, 1, 1, 0, -1, -1)
    base_mid = 1.1 if instrument is orderbook_models.Instrument.EUR_USD else 150.0
    tick = 0.00001 if instrument is orderbook_models.Instrument.EUR_USD else 0.001
    decimals = 5 if instrument is orderbook_models.Instrument.EUR_USD else 3
    rows: list[str] = []
    for step in range(30):
        timestamp = datetime.datetime.combine(
            trading_date, datetime.time(), tzinfo=datetime.UTC
        ) + datetime.timedelta(milliseconds=step * 100)
        mid = base_mid + mid_offsets[step % len(mid_offsets)] * tick
        for side in (0, 1):
            for level in range(1, 11):
                direction = -1 if side == 0 else 1
                price = mid + direction * (tick * level)
                rows.append(
                    f"{timestamp:%Y/%m/%d},{timestamp:%H:%M:%S.%f}"[:-3]
                    + f",{instrument.to_symbol()},Q,{side},{level},"
                    + f"{price:.{decimals}f},1000000,1\n"
                )
    with gzip.open(path, mode="wt", encoding="utf-8", newline="") as stream:
        stream.writelines(rows)
