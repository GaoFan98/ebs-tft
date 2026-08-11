# Load processed 1 minute bar data from parquet files

from __future__ import annotations

import datetime
import logging
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from ebs_tft.data.repositories import processed as processed_repo
from ebs_tft.domain.orderbook import _models as m

logger = logging.getLogger(__name__)

# Exceptions


class NoBarsFoundError(Exception):
    """
    Raised when a query returns no bars at all.
    """


def load_bars(
    *,
    processed_dir: Path,
    level_group: str,
    instrument: m.Instrument,
    date_from: datetime.date | None = None,
    date_to: datetime.date | None = None,
) -> pl.DataFrame:
    """
    Load all processed bars for a single instrument and level group,
    optionally filtered to a date range.

    :param processed_dir: root of the processed data tree (e.g. data/processed/)
    :param level_group: "l1", "l1_l5", or "l1_l10"
    :param instrument: Instrument enum value to load
    :param date_from: inclusive start date; None means load from the earliest file
    :param date_to: inclusive end date; None means load through the latest file
    :returns: polars DataFrame sorted by timestamp, empty if no files found
    """
    date_from_str = _to_date_str(date_from) if date_from is not None else None
    date_to_str = _to_date_str(date_to) if date_to is not None else None

    file_refs = list(
        processed_repo.find_processed_files(
            processed_dir=processed_dir,
            level_group=level_group,
            instrument=instrument.value,
        )
    )
    # Apply date range filter at the file level
    matching = _filter_by_date(
        file_refs=file_refs,
        date_from_str=date_from_str,
        date_to_str=date_to_str,
    )
    if not matching:
        logger.info(
            "No processed files found for query",
            extra={
                "level_group": level_group,
                "instrument": instrument.value,
                "date_from": date_from_str,
                "date_to": date_to_str,
            },
        )
        return pl.DataFrame()

    frames = _load_files(file_refs=matching)
    if not frames:
        return pl.DataFrame()

    combined = pl.concat(frames, how="diagonal")
    return combined.sort(m.COL_TIMESTAMP)


def load_multi_instrument_bars(
    *,
    processed_dir: Path,
    level_group: str,
    instruments: Sequence[m.Instrument],
    date_from: datetime.date | None = None,
    date_to: datetime.date | None = None,
) -> pl.DataFrame:
    """
    Load and concatenate bars for multiple instruments into a single DataFrame.

    :param processed_dir: root of the processed data tree
    :param level_group: "l1", "l1_l5", or "l1_l10"
    :param instruments: list of Instrument enum values to include
    :param date_from: inclusive start date; None means all available
    :param date_to: inclusive end date; None means all available
    :returns: combined polars DataFrame sorted by (instrument, timestamp)
    :raises NoBarsFoundError: if no data at all was found across all instruments
    """
    all_frames: list[pl.DataFrame] = []
    for instrument in instruments:
        df = load_bars(
            processed_dir=processed_dir,
            level_group=level_group,
            instrument=instrument,
            date_from=date_from,
            date_to=date_to,
        )
        if df.is_empty():
            logger.warning(
                "No bars found for instrument in query",
                extra={
                    "instrument": instrument.value,
                    "level_group": level_group,
                    "date_from": str(date_from),
                    "date_to": str(date_to),
                },
            )
            continue

        all_frames.append(df)
        logger.debug(
            "Loaded bars for instrument",
            extra={
                "instrument": instrument.value,
                "level_group": level_group,
                "rows": len(df),
            },
        )

    if not all_frames:
        raise NoBarsFoundError(
            f"No processed bars found for level_group={level_group!r}, "
            f"instruments={[i.value for i in instruments]}, "
            f"date_from={date_from}, date_to={date_to}. "
            f"Run the exporter first: application/usecases/data_ingestion/_exporter.py"
        )

    combined = pl.concat(all_frames, how="diagonal")
    # Sort by (instrument, timestamp) — neuralforecast expects all rows for
    # one series to be contiguous and chronologically ordered.
    return combined.sort([m.COL_INSTRUMENT, m.COL_TIMESTAMP])


def available_date_range(
    *,
    processed_dir: Path,
    level_group: str,
    instrument: m.Instrument,
) -> tuple[str, str] | None:
    """
    Return the (earliest_date, latest_date) of available processed files
    for a given instrument and level group, or None if no files exist.
    """
    file_refs = list(
        processed_repo.find_processed_files(
            processed_dir=processed_dir,
            level_group=level_group,
            instrument=instrument.value,
        )
    )
    if not file_refs:
        return None
    # find_processed_files() yields in chronological order (sorted by filename)
    earliest = file_refs[0].trading_date
    latest = file_refs[-1].trading_date

    return earliest, latest


# Helper functions
def _to_date_str(date: datetime.date) -> str:
    """
    Convert a datetime.date to the "YYYYMMDD" string format used in filenames.
    """
    return date.strftime("%Y%m%d")


def _filter_by_date(
    *,
    file_refs: list[processed_repo.ProcessedDataFile],
    date_from_str: str | None,
    date_to_str: str | None,
) -> list[processed_repo.ProcessedDataFile]:
    """
    Filter a list of ProcessedDataFile references to those whose trading_date
    falls within [date_from_str, date_to_str] (both inclusive).
    """
    result = []

    for ref in file_refs:
        if date_from_str is not None and ref.trading_date < date_from_str:
            continue
        if date_to_str is not None and ref.trading_date > date_to_str:
            continue
        result.append(ref)

    return result


def _load_files(
    *,
    file_refs: list[processed_repo.ProcessedDataFile],
) -> list[pl.DataFrame]:
    """
    Load each ProcessedDataFile reference into a polars DataFrame.
    """
    frames: list[pl.DataFrame] = []

    for ref in file_refs:
        try:
            df = processed_repo.read_bars(path=ref.path)
        except processed_repo.UnableToReadBarsError as exc:
            # File exists on disk but cannot be read (corrupted,
            # partially written). Log at WARNING and skip rather than crash.
            logger.warning(
                "Skipping unreadable processed file",
                extra={"path": str(ref.path), "error": str(exc)},
            )
            continue
        frames.append(df)

    return frames
