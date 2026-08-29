"""Test EBS CSV parsing against real gzip and CSV boundaries."""

from __future__ import annotations

import datetime
import gzip
from pathlib import Path

import pytest

from ebs_tft.data.parsers import ebs_csv
from ebs_tft.domain.orderbook import models


class TestParseRows:
    def test_parse_rows_routes_quotes_deals_and_resets_once(
        self, tmp_path: Path
    ) -> None:
        path = _write_gzip(
            parent=tmp_path,
            text=(
                '2024/01/01,22:00:00.000,"EUR/USD",Q,0,1,,0,0\n'
                "2024/01/01,22:00:00.100,EUR/USD,Q,0,1,1.1,1000000,2\n"
                "2024/01/01,22:00:00.200,EUR/USD,D,1,,1.1,1000000,1,2000000\n"
            ),
        )
        audit = ebs_csv.ParseAudit()

        actual = list(
            ebs_csv.parse_rows(
                path=path,
                expected_instrument=models.Instrument.EUR_USD,
                expected_trading_date=datetime.date(2024, 1, 2),
                audit=audit,
            )
        )

        assert len(actual) == 3
        assert isinstance(actual[0], models.RawQuote)
        assert actual[0].price is None
        assert isinstance(actual[2], models.RawDeal)
        assert actual[2].total_volume == 2_000_000
        assert actual[2].extremal_price_volume == 1_000_000
        assert audit.physical_lines == audit.accounted_lines == 3
        assert (audit.quote_rows, audit.deal_rows, audit.error_rows) == (2, 1, 0)

    def test_parse_rows_collects_sanitized_errors_in_audit_mode(
        self, tmp_path: Path
    ) -> None:
        path = _write_gzip(
            parent=tmp_path,
            text=(
                "2024/01/01,22:00:00.000,EUR/USD,Q,0,11,1.1,1,1\n"
                "2024/01/01,22:00:00.100,EUR/USD,D,0,,1.1,2,1,1\n"
                "2024/01/01,22:00:00.200,EUR/USD,Q,0,1,1.1,1,1\n"
            ),
        )
        audit = ebs_csv.ParseAudit()

        actual = list(
            ebs_csv.parse_rows(
                path=path,
                expected_instrument=models.Instrument.EUR_USD,
                expected_trading_date=datetime.date(2024, 1, 2),
                strict=False,
                audit=audit,
            )
        )

        assert len(actual) == 1
        assert audit.error_rows == 2
        assert audit.physical_lines == audit.accounted_lines == 3
        assert [issue.line_number for issue in audit.issues] == [1, 2]
        assert all("EUR/USD" not in issue.reason for issue in audit.issues)

    def test_parse_rows_fails_contextually_in_strict_mode(self, tmp_path: Path) -> None:
        path = _write_gzip(
            parent=tmp_path,
            text="2024/01/01,22:00:00.000,USD/JPY,Q,0,1,1.1,1,1\n",
        )

        with pytest.raises(ebs_csv.UnableToParseRowError, match="line 1"):
            list(
                ebs_csv.parse_rows(
                    path=path,
                    expected_instrument=models.Instrument.EUR_USD,
                    expected_trading_date=datetime.date(2024, 1, 2),
                )
            )

    def test_parse_rows_rejects_out_of_order_timestamps(self, tmp_path: Path) -> None:
        path = _write_gzip(
            parent=tmp_path,
            text=(
                "2024/01/01,22:00:01.000,EUR/USD,Q,0,1,1.1,1,1\n"
                "2024/01/01,22:00:00.000,EUR/USD,Q,1,1,1.2,1,1\n"
            ),
        )

        with pytest.raises(ebs_csv.UnableToParseRowError, match="out of source order"):
            list(
                ebs_csv.parse_rows(
                    path=path,
                    expected_instrument=models.Instrument.EUR_USD,
                    expected_trading_date=datetime.date(2024, 1, 2),
                )
            )

    def test_parse_rows_rejects_invalid_utf8(self, tmp_path: Path) -> None:
        path = tmp_path / "fixture.csv.gz"
        with gzip.open(path, mode="wb") as stream:
            stream.write(b"\xff")

        with pytest.raises(ebs_csv.UnableToParseRowError, match="decode"):
            list(
                ebs_csv.parse_rows(
                    path=path,
                    expected_instrument=models.Instrument.EUR_USD,
                    expected_trading_date=datetime.date(2024, 1, 2),
                )
            )


def _write_gzip(*, parent: Path, text: str) -> Path:
    path = parent / "fixture.csv.gz"
    with gzip.open(path, mode="wt", encoding="utf-8", newline="") as stream:
        stream.write(text)
    return path
