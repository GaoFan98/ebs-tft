"""Test consolidated EBS normalization across filesystem boundaries."""

from __future__ import annotations

import datetime
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from ebs_tft.application.usecases import data_ingestion
from ebs_tft.data.parsers import ebs_csv
from ebs_tft.domain.orderbook import models


class TestNormalizeConsolidatedYear:
    def test_filters_one_instrument_without_aggregation_and_resumes(
        self, tmp_path: Path
    ) -> None:
        source_dir = tmp_path / "source"
        output_dir = tmp_path / "output"
        source_dir.mkdir()
        source = source_dir / "20230601-EBS_Level2_0_0_0.csv.gz"
        source_rows = (
            "2023/05/31,21:00:00.000,USD/JPY,Q,0,1,140.1,1,1\n"
            '2023/05/31,21:00:00.000,"EUR/USD",Q,0,1,1.1,2,1\n'
            "2023/05/31,21:00:00.100,EUR/USD,D,1,,1.1,2,1,3\n"
        )
        with gzip.open(source, mode="wt", encoding="utf-8", newline="") as stream:
            stream.write(source_rows)
        source_sha256 = _sha256(path=source)

        first = data_ingestion.normalize_consolidated_year(
            source_dir=source_dir,
            output_dir=output_dir,
            year=2023,
            instrument=models.Instrument.EUR_USD,
        )

        output = output_dir / "20230601-EBS_LVL2_EUR_USD_0.csv.gz"
        parsed = list(
            ebs_csv.parse_rows(
                path=output,
                expected_instrument=models.Instrument.EUR_USD,
                expected_trading_date=datetime.date(2023, 6, 1),
            )
        )
        manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
        assert len(parsed) == 2
        assert [record.timestamp.microsecond for record in parsed] == [0, 100_000]
        assert _sha256(path=source) == source_sha256
        assert manifest["aggregation_used"] is False
        assert manifest["transformation"] == "symbol_filter_only"
        assert manifest["sessions"][0]["source_rows"] == 3
        assert manifest["sessions"][0]["selected_rows"] == 2
        assert first.normalized_files == 1
        assert first.reused_files == 0

        resumed = data_ingestion.normalize_consolidated_year(
            source_dir=source_dir,
            output_dir=output_dir,
            year=2023,
            instrument=models.Instrument.EUR_USD,
        )

        assert resumed.normalized_files == 0
        assert resumed.reused_files == 1

        verified = data_ingestion.verify_normalized_year(
            output_dir=output_dir,
            year=2023,
            instrument=models.Instrument.EUR_USD,
        )

        assert verified.verified_files == 1
        assert verified.selected_rows == 2

    def test_rejects_an_unverified_existing_output(self, tmp_path: Path) -> None:
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        source = source_dir / "20230601-EBS_Level2_0_0_0.csv.gz"
        with gzip.open(source, mode="wt", encoding="utf-8") as stream:
            stream.write("2023/05/31,21:00:00.000,EUR/USD,Q,0,1,1.1,1,1\n")
        output = source_dir / "20230601-EBS_LVL2_EUR_USD_0.csv.gz"
        output.write_bytes(b"unverified")

        with pytest.raises(
            data_ingestion.UnableToNormalizeConsolidatedDataError,
            match="Unverified",
        ):
            data_ingestion.normalize_consolidated_year(
                source_dir=source_dir,
                output_dir=source_dir,
                year=2023,
                instrument=models.Instrument.EUR_USD,
            )

    def test_portable_verification_rejects_tampered_output(
        self, tmp_path: Path
    ) -> None:
        source_dir = tmp_path / "source"
        output_dir = tmp_path / "output"
        source_dir.mkdir()
        source = source_dir / "20230601-EBS_Level2_0_0_0.csv.gz"
        with gzip.open(source, mode="wt", encoding="utf-8") as stream:
            stream.write("2023/05/31,21:00:00.000,EUR/USD,Q,0,1,1.1,1,1\n")
        data_ingestion.normalize_consolidated_year(
            source_dir=source_dir,
            output_dir=output_dir,
            year=2023,
            instrument=models.Instrument.EUR_USD,
        )
        output = output_dir / "20230601-EBS_LVL2_EUR_USD_0.csv.gz"
        output.write_bytes(b"tampered")

        with pytest.raises(
            data_ingestion.UnableToNormalizeConsolidatedDataError,
            match="size mismatch",
        ):
            data_ingestion.verify_normalized_year(
                output_dir=output_dir,
                year=2023,
                instrument=models.Instrument.EUR_USD,
            )


def _sha256(*, path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()
