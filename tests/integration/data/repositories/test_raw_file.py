"""
Test raw EBS file discovery against a temporary filesystem.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import pytest

from ebs_tft.data.repositories import raw_file


class TestFindRawFiles:
    def test_find_raw_files_filters_and_orders_valid_files(
        self, tmp_path: Path
    ) -> None:
        year_dir = tmp_path / "2024"
        year_dir.mkdir()
        eur_usd = year_dir / "20240102-EBS_LVL2_EUR_USD_0.csv.gz"
        eur_jpy = year_dir / "20240103-EBS_LVL2_EUR_JPY_0.csv.gz"
        usd_jpy = year_dir / "20240101-EBS_LVL2_USD_JPY_0.csv.gz"
        unselected = year_dir / "20240101-EBS_LVL2_GBP_USD_0.csv.gz"
        for path in (eur_usd, eur_jpy, usd_jpy, unselected):
            path.write_bytes(b"data")

        actual = list(
            raw_file.find_raw_files(
                data_dir=tmp_path,
                instruments=("USD_JPY", "EUR_USD", "EUR_JPY"),
                years=(2024,),
            )
        )

        assert [item.instrument for item in actual] == [
            "EUR_JPY",
            "EUR_USD",
            "USD_JPY",
        ]
        assert [item.path for item in actual] == [eur_jpy, eur_usd, usd_jpy]
        assert actual[0].trading_date == datetime.date(2024, 1, 3)
        assert all(item.size_bytes == 4 for item in actual)
        assert all(item.fingerprint for item in actual)

    def test_find_raw_files_returns_empty_for_a_missing_year(
        self, tmp_path: Path
    ) -> None:
        actual = list(
            raw_file.find_raw_files(
                data_dir=tmp_path,
                instruments=("EUR_USD",),
                years=(2024,),
            )
        )

        assert actual == []

    def test_find_raw_files_rejects_a_malformed_candidate(self, tmp_path: Path) -> None:
        year_dir = tmp_path / "2024"
        year_dir.mkdir()
        (year_dir / "not-an-ebs-file.csv.gz").write_bytes(b"data")

        with pytest.raises(raw_file.InvalidRawDataFileError, match="Malformed"):
            list(
                raw_file.find_raw_files(
                    data_dir=tmp_path,
                    instruments=("EUR_USD",),
                    years=(2024,),
                )
            )

    def test_find_raw_files_rejects_an_invalid_calendar_date(
        self, tmp_path: Path
    ) -> None:
        year_dir = tmp_path / "2024"
        year_dir.mkdir()
        (year_dir / "20240231-EBS_LVL2_EUR_USD_0.csv.gz").write_bytes(b"data")

        with pytest.raises(raw_file.InvalidRawDataFileError, match="Invalid date"):
            list(
                raw_file.find_raw_files(
                    data_dir=tmp_path,
                    instruments=("EUR_USD",),
                    years=(2024,),
                )
            )

    def test_find_raw_files_rejects_a_directory_year_mismatch(
        self, tmp_path: Path
    ) -> None:
        year_dir = tmp_path / "2024"
        year_dir.mkdir()
        (year_dir / "20230101-EBS_LVL2_EUR_USD_0.csv.gz").write_bytes(b"data")

        with pytest.raises(raw_file.InvalidRawDataFileError, match="does not match"):
            list(
                raw_file.find_raw_files(
                    data_dir=tmp_path,
                    instruments=("EUR_USD",),
                    years=(2024,),
                )
            )

    def test_find_raw_files_rejects_duplicate_filters(self, tmp_path: Path) -> None:
        with pytest.raises(raw_file.InvalidRawFileQueryError, match="duplicates"):
            list(
                raw_file.find_raw_files(
                    data_dir=tmp_path,
                    instruments=("EUR_USD", "EUR_USD"),
                    years=(2024,),
                )
            )


class TestGetContentFingerprint:
    def test_get_content_fingerprint_changes_with_source_content(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "source.csv.gz"
        path.write_bytes(b"before")
        stat = path.stat()
        source = raw_file.RawDataFile(
            path=path,
            instrument="EUR_USD",
            trading_date=datetime.date(2024, 1, 2),
            size_bytes=stat.st_size,
            modified_time_ns=stat.st_mtime_ns,
        )
        before = raw_file.get_content_fingerprint(raw_data_file=source)

        path.write_bytes(b"after!")
        after = raw_file.get_content_fingerprint(raw_data_file=source)

        assert before != after
