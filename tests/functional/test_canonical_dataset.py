"""Exercise raw gzip through canonical depth datasets twice."""

from __future__ import annotations

import datetime
import gzip
from pathlib import Path

from ebs_tft.application import config
from ebs_tft.application.usecases import data_ingestion
from ebs_tft.data.repositories import raw_file
from ebs_tft.domain.dataset import models as dataset_models
from ebs_tft.domain.dataset import operations as dataset_operations
from ebs_tft.domain.orderbook import models as orderbook_models


def test_raw_gzip_to_depth_dataset_is_idempotent(tmp_path: Path) -> None:
    project_config = _config(parent=tmp_path)
    dates = [datetime.date(2024, 1, day) for day in (2, 3, 4)]
    first_checksums: list[str] = []
    repeated_checksums: list[str] = []

    for trading_date in dates:
        raw_data_file = _raw_file(parent=tmp_path, trading_date=trading_date)
        first = data_ingestion.export_file(
            raw_data_file=raw_data_file, project_config=project_config
        )
        repeated = data_ingestion.export_file(
            raw_data_file=raw_data_file, project_config=project_config
        )
        first_checksums.append(first.metadata.content_sha256)
        repeated_checksums.append(repeated.metadata.content_sha256)

    dataset = dataset_operations.build_dataset(
        spec=dataset_models.DatasetSpec(
            depth=dataset_models.DepthSpec(maximum_level=1),
            instruments=(orderbook_models.Instrument.EUR_USD,),
            split_spec=dataset_models.SplitSpec(
                train=dataset_models.DateRange(dates[0], dates[0]),
                validation=dataset_models.DateRange(dates[1], dates[1]),
                test=dataset_models.DateRange(dates[2], dates[2]),
            ),
            forecast_horizon=datetime.timedelta(minutes=1),
            context_length=datetime.timedelta(minutes=1),
            state_interval=datetime.timedelta(milliseconds=100),
            flat_target_policy=dataset_models.FlatTargetPolicy.THREE_CLASS,
            neutral_threshold=0,
            processed_dir=project_config.training.processed_data_dir,
        )
    )

    assert first_checksums == repeated_checksums
    assert [len(dataset[split]) for split in dataset_models.Split] == [1, 1, 1]
    assert all(
        dataset[split][orderbook_models.COL_DIRECTION_TARGET][0] == 1
        for split in dataset_models.Split
    )


def _config(*, parent: Path) -> config.ProjectConfig:
    return config.ProjectConfig(
        instruments=config.InstrumentsConfig(
            schema_version=1,
            instruments=(orderbook_models.Instrument.EUR_USD,),
            maximum_depth=1,
        ),
        training=config.TrainingConfig(
            schema_version=1,
            raw_data_dir=parent / "raw",
            processed_data_dir=parent / "processed",
            state_interval_milliseconds=100,
            forecast_horizons_milliseconds=(),
            source_timezone="UTC",
            session_calendar="EBS_FX_17_NEW_YORK",
            maximum_quote_staleness_milliseconds=60_000,
            flat_target_policy=None,
            random_seeds=(),
        ),
        model_defaults=config.ModelDefaultsConfig(schema_version=1, engine=None),
    )


def _raw_file(*, parent: Path, trading_date: datetime.date) -> raw_file.RawDataFile:
    date_value = trading_date.strftime("%Y/%m/%d")
    label = trading_date.strftime("%Y%m%d")
    path = (
        parent / "raw" / str(trading_date.year) / (f"{label}-EBS_LVL2_EUR_USD_0.csv.gz")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{date_value},12:00:10.000,EUR/USD,Q,0,1,1.10,1000000,1\n",
        f"{date_value},12:00:20.000,EUR/USD,Q,1,1,1.20,1000000,1\n",
        f"{date_value},12:01:10.000,EUR/USD,Q,0,1,1.11,1000000,1\n",
        f"{date_value},12:01:20.000,EUR/USD,Q,1,1,1.21,1000000,1\n",
    ]
    with gzip.open(path, mode="wt", encoding="utf-8", newline="") as stream:
        stream.writelines(lines)
    stat = path.stat()
    return raw_file.RawDataFile(
        path=path,
        instrument="EUR_USD",
        trading_date=trading_date,
        size_bytes=stat.st_size,
        modified_time_ns=stat.st_mtime_ns,
    )
