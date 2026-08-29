"""Publish and read canonical full-depth partitions atomically."""

from __future__ import annotations

import datetime
import fcntl
import hashlib
import json
import logging
import os
import uuid
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import cast

import attrs
import polars as pl

from ebs_tft.domain.orderbook import models

SCHEMA_VERSION = 1
_MANIFEST_NAME = "manifest.json"
logger = logging.getLogger(__name__)


class UnableToWriteBarsError(Exception):
    """Indicate that a canonical partition could not be published."""


class UnableToReadBarsError(Exception):
    """Indicate that a canonical partition is absent, corrupt, or incompatible."""


@attrs.frozen
class PartitionMetadata:
    """Describe the content and provenance of one published partition."""

    schema_version: int
    instrument: models.Instrument
    trading_date: datetime.date
    source_fingerprint: str
    config_fingerprint: str
    content_sha256: str
    data_file: str
    row_count: int
    timestamp_from: datetime.datetime
    timestamp_to: datetime.datetime
    maximum_depth: int
    schema: tuple[tuple[str, str], ...]
    quality: tuple[tuple[str, int], ...]


@attrs.frozen
class ProcessedPartition:
    """Reference one manifest-backed canonical partition."""

    manifest_path: Path
    metadata: PartitionMetadata

    @property
    def data_path(self) -> Path:
        """Return the immutable generation selected by the manifest."""
        return self.manifest_path.parent / self.metadata.data_file


def get_partition_dir(
    *, processed_dir: Path, instrument: models.Instrument, trading_date: datetime.date
) -> Path:
    """Return the canonical directory for one instrument/trading-date partition."""
    return processed_dir / instrument.value / trading_date.strftime("%Y%m%d")


def is_current(
    *,
    processed_dir: Path,
    instrument: models.Instrument,
    trading_date: datetime.date,
    source_fingerprint: str,
    config_fingerprint: str,
) -> bool:
    """Return whether a valid published partition has the requested provenance."""
    try:
        partition = load_partition(
            manifest_path=get_partition_dir(
                processed_dir=processed_dir,
                instrument=instrument,
                trading_date=trading_date,
            )
            / _MANIFEST_NAME
        )
        read_bars(partition=partition)
    except UnableToReadBarsError:
        return False
    return (
        partition.metadata.source_fingerprint == source_fingerprint
        and partition.metadata.config_fingerprint == config_fingerprint
    )


def write_bars(
    *,
    processed_dir: Path,
    instrument: models.Instrument,
    trading_date: datetime.date,
    data: pl.DataFrame,
    maximum_depth: int,
    source_fingerprint: str,
    config_fingerprint: str,
    quality: Mapping[str, int],
) -> ProcessedPartition:
    """Validate and atomically publish one immutable Parquet generation."""
    _validate_frame(
        data=data,
        instrument=instrument,
        trading_date=trading_date,
        maximum_depth=maximum_depth,
    )
    if data.is_empty():
        raise UnableToWriteBarsError("canonical partition must not be empty")
    partition_dir = get_partition_dir(
        processed_dir=processed_dir,
        instrument=instrument,
        trading_date=trading_date,
    )
    generation_dir = partition_dir / "generations"
    manifest_path = partition_dir / _MANIFEST_NAME
    lock_path = partition_dir / ".publish.lock"
    generation_name = f"generations/{uuid.uuid4().hex}.parquet"
    final_data_path = partition_dir / generation_name
    temporary_data_path = generation_dir / f".{uuid.uuid4().hex}.tmp"
    temporary_manifest_path = partition_dir / f".{uuid.uuid4().hex}.tmp"

    try:
        generation_dir.mkdir(parents=True, exist_ok=True)
        with lock_path.open(mode="a+b") as lock_stream:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
            try:
                current = load_partition(manifest_path=manifest_path)
                if (
                    current.metadata.source_fingerprint == source_fingerprint
                    and current.metadata.config_fingerprint == config_fingerprint
                ):
                    read_bars(partition=current)
                    return current
            except UnableToReadBarsError:
                logger.info(
                    "No valid current generation; publishing canonical partition",
                    extra={"manifest_path": str(manifest_path)},
                )
            data.write_parquet(temporary_data_path)
            _fsync_file(path=temporary_data_path)
            persisted = pl.read_parquet(temporary_data_path)
            _validate_frame(
                data=persisted,
                instrument=instrument,
                trading_date=trading_date,
                maximum_depth=maximum_depth,
            )
            content_sha256 = _sha256(path=temporary_data_path)
            timestamp_from = cast(datetime.datetime, data[models.COL_TIMESTAMP].min())
            timestamp_to = cast(datetime.datetime, data[models.COL_TIMESTAMP].max())
            metadata = PartitionMetadata(
                schema_version=SCHEMA_VERSION,
                instrument=instrument,
                trading_date=trading_date,
                source_fingerprint=source_fingerprint,
                config_fingerprint=config_fingerprint,
                content_sha256=content_sha256,
                data_file=generation_name,
                row_count=len(data),
                timestamp_from=timestamp_from,
                timestamp_to=timestamp_to,
                maximum_depth=maximum_depth,
                schema=tuple(
                    (column, str(dtype)) for column, dtype in data.schema.items()
                ),
                quality=tuple(sorted(quality.items())),
            )
            os.replace(temporary_data_path, final_data_path)
            _fsync_directory(path=generation_dir)
            with temporary_manifest_path.open(mode="w", encoding="utf-8") as stream:
                json.dump(_serialize(metadata=metadata), stream, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_manifest_path, manifest_path)
            _fsync_directory(path=partition_dir)
            return ProcessedPartition(
                manifest_path=manifest_path,
                metadata=metadata,
            )
    except (OSError, ValueError, pl.exceptions.PolarsError) as exc:
        temporary_data_path.unlink(missing_ok=True)
        temporary_manifest_path.unlink(missing_ok=True)
        raise UnableToWriteBarsError(
            f"Unable to publish processed partition: {partition_dir}"
        ) from exc


def load_partition(*, manifest_path: Path) -> ProcessedPartition:
    """Load and validate one success manifest."""
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("manifest root is not an object")
        metadata = _deserialize(raw=cast(dict[str, object], raw))
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise UnableToReadBarsError(
            f"Invalid partition manifest: {manifest_path}"
        ) from exc
    partition = ProcessedPartition(manifest_path=manifest_path, metadata=metadata)
    if not partition.data_path.is_file():
        raise UnableToReadBarsError(
            f"Manifest data file is missing: {partition.data_path}"
        )
    return partition


def read_bars(*, partition: ProcessedPartition) -> pl.DataFrame:
    """Read one partition and verify its content against the manifest."""
    try:
        verify_partition(partition=partition)
        data = pl.read_parquet(partition.data_path)
        _validate_frame(
            data=data,
            instrument=partition.metadata.instrument,
            trading_date=partition.metadata.trading_date,
            maximum_depth=partition.metadata.maximum_depth,
        )
        if len(data) != partition.metadata.row_count:
            raise ValueError("row count mismatch")
        actual_schema = tuple(
            (column, str(dtype)) for column, dtype in data.schema.items()
        )
        if actual_schema != partition.metadata.schema:
            raise ValueError("schema mismatch")
    except (OSError, ValueError, pl.exceptions.PolarsError) as exc:
        raise UnableToReadBarsError(
            f"Invalid processed data: {partition.data_path}"
        ) from exc
    return data


def verify_partition(*, partition: ProcessedPartition) -> None:
    """Verify immutable bytes selected by a manifest without materializing data."""
    try:
        if _sha256(path=partition.data_path) != partition.metadata.content_sha256:
            raise ValueError("content checksum mismatch")
    except (OSError, ValueError) as exc:
        raise UnableToReadBarsError(
            f"Invalid processed data: {partition.data_path}"
        ) from exc


def find_partitions(
    *,
    processed_dir: Path,
    instruments: Sequence[models.Instrument],
    date_from: datetime.date | None = None,
    date_to: datetime.date | None = None,
) -> Iterator[ProcessedPartition]:
    """Discover requested manifests once and yield them deterministically."""
    if not instruments or len(set(instruments)) != len(instruments):
        raise ValueError("instruments must be non-empty and unique")
    if date_from is not None and date_to is not None and date_from > date_to:
        raise ValueError("date_from must not be after date_to")
    partitions: list[ProcessedPartition] = []
    for instrument in instruments:
        instrument_dir = processed_dir / instrument.value
        if not instrument_dir.exists():
            continue
        try:
            manifests = instrument_dir.glob(f"*/{_MANIFEST_NAME}")
            for manifest_path in manifests:
                partition = load_partition(manifest_path=manifest_path)
                if partition.metadata.instrument is not instrument:
                    raise UnableToReadBarsError(
                        "Manifest is stored under the wrong instrument: "
                        f"{manifest_path}"
                    )
                date = partition.metadata.trading_date
                if date_from is not None and date < date_from:
                    continue
                if date_to is not None and date > date_to:
                    continue
                partitions.append(partition)
        except OSError as exc:
            raise UnableToReadBarsError(
                f"Unable to discover processed partitions: {instrument_dir}"
            ) from exc
    yield from sorted(
        partitions,
        key=lambda item: (
            item.metadata.instrument.value,
            item.metadata.trading_date,
        ),
    )


def _validate_frame(
    *,
    data: pl.DataFrame,
    instrument: models.Instrument,
    trading_date: datetime.date,
    maximum_depth: int,
) -> None:
    expected_columns = models.canonical_bar_columns(max_level=maximum_depth)
    if data.columns != expected_columns:
        raise ValueError("canonical columns do not match the versioned schema")
    if (
        data.null_count()
        .select([models.COL_TIMESTAMP, models.COL_INSTRUMENT, models.COL_TRADING_DATE])
        .sum_horizontal()[0]
    ):
        raise ValueError("canonical identity columns must not contain nulls")
    if data[models.COL_INSTRUMENT].unique().to_list() not in (
        [],
        [instrument.value],
    ):
        raise ValueError("partition contains another instrument")
    keys = [models.COL_INSTRUMENT, models.COL_TIMESTAMP]
    if data.select(keys).n_unique() != len(data):
        raise ValueError("partition keys are not unique")
    if not data[models.COL_TIMESTAMP].is_sorted():
        raise ValueError("partition timestamps are not sorted")
    if data[models.COL_TRADING_DATE].unique().to_list() not in ([], [trading_date]):
        raise ValueError("partition contains another trading date")
    allowed_calendar_dates = {
        trading_date,
        trading_date - datetime.timedelta(days=1),
    }
    if not set(data[models.COL_TIMESTAMP].dt.date().unique().to_list()).issubset(
        allowed_calendar_dates
    ):
        raise ValueError("partition timestamps are outside the trading-date session")
    timestamp_differences = (
        data[models.COL_TIMESTAMP].diff().drop_nulls().unique().to_list()
    )
    if any(
        difference != datetime.timedelta(minutes=1)
        for difference in timestamp_differences
    ):
        raise ValueError("canonical timestamps must use a regular one-minute grid")
    observed = data.filter(pl.col(models.COL_BOOK_OBSERVED))
    if not observed.is_empty():
        bid_l1 = models.bid_price_col(level=1)
        ask_l1 = models.ask_price_col(level=1)
        if observed.filter(pl.col(bid_l1) >= pl.col(ask_l1)).height:
            raise ValueError("canonical data contains a crossed observed book")
        for level in range(1, maximum_depth):
            if observed.filter(
                pl.col(models.bid_price_col(level=level))
                <= pl.col(models.bid_price_col(level=level + 1))
            ).height:
                raise ValueError("canonical bid depth is not strictly decreasing")
            if observed.filter(
                pl.col(models.ask_price_col(level=level))
                >= pl.col(models.ask_price_col(level=level + 1))
            ).height:
                raise ValueError("canonical offer depth is not strictly increasing")
    non_negative_columns = [
        *models.all_bid_size_cols(max_level=maximum_depth),
        *models.all_ask_size_cols(max_level=maximum_depth),
        *models.all_bid_order_count_cols(max_level=maximum_depth),
        *models.all_ask_order_count_cols(max_level=maximum_depth),
    ]
    if any(data.filter(pl.col(column) < 0).height for column in non_negative_columns):
        raise ValueError("canonical size/count columns must be non-negative")


def _serialize(*, metadata: PartitionMetadata) -> dict[str, object]:
    return {
        "schema_version": metadata.schema_version,
        "instrument": metadata.instrument.value,
        "trading_date": metadata.trading_date.isoformat(),
        "source_fingerprint": metadata.source_fingerprint,
        "config_fingerprint": metadata.config_fingerprint,
        "content_sha256": metadata.content_sha256,
        "data_file": metadata.data_file,
        "row_count": metadata.row_count,
        "timestamp_from": metadata.timestamp_from.isoformat(),
        "timestamp_to": metadata.timestamp_to.isoformat(),
        "maximum_depth": metadata.maximum_depth,
        "schema": [list(item) for item in metadata.schema],
        "quality": dict(metadata.quality),
    }


def _deserialize(*, raw: Mapping[str, object]) -> PartitionMetadata:
    if raw["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported processed schema version")
    quality = raw["quality"]
    if not isinstance(quality, dict) or not all(
        isinstance(key, str) and isinstance(value, int)
        for key, value in quality.items()
    ):
        raise ValueError("invalid quality summary")
    schema = raw["schema"]
    if not isinstance(schema, list) or not all(
        isinstance(item, list)
        and len(item) == 2
        and all(isinstance(value, str) for value in item)
        for item in schema
    ):
        raise ValueError("invalid stored schema")
    return PartitionMetadata(
        schema_version=SCHEMA_VERSION,
        instrument=models.Instrument(str(raw["instrument"])),
        trading_date=datetime.date.fromisoformat(str(raw["trading_date"])),
        source_fingerprint=str(raw["source_fingerprint"]),
        config_fingerprint=str(raw["config_fingerprint"]),
        content_sha256=str(raw["content_sha256"]),
        data_file=str(raw["data_file"]),
        row_count=int(str(raw["row_count"])),
        timestamp_from=datetime.datetime.fromisoformat(str(raw["timestamp_from"])),
        timestamp_to=datetime.datetime.fromisoformat(str(raw["timestamp_to"])),
        maximum_depth=int(str(raw["maximum_depth"])),
        schema=tuple((str(item[0]), str(item[1])) for item in schema),
        quality=tuple(sorted(cast(dict[str, int], quality).items())),
    )


def _sha256(*, path: Path) -> str:
    digest = hashlib.sha256()
    with path.open(mode="rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_file(*, path: Path) -> None:
    with path.open(mode="rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(*, path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
