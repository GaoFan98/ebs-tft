"""Test robust construction of the all-session audit table."""

from __future__ import annotations

import polars as pl

from ebs_tft.application.usecases.research_protocol import _audit


def test_audit_frame_infers_late_non_null_fields() -> None:
    """Allow an error string that appears after Polars' default inference window."""
    rows: list[dict[str, object]] = [
        {"session": index, "parse_error": None} for index in range(100)
    ]
    rows.append({"session": 100, "parse_error": "late parse failure"})

    result = _audit._audit_frame(rows=rows)

    assert result.height == 101
    assert result.schema["parse_error"] == pl.String
    assert result["parse_error"].tail(1).item() == "late parse failure"
