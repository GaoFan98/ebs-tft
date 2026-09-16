"""Normalize consolidated legacy EBS files without changing observations."""

from __future__ import annotations

import csv
import datetime
import gzip
import hashlib
import io
import json
import time
from pathlib import Path
from typing import cast

import attrs

from ebs_tft.data.repositories import consolidated_raw_file
from ebs_tft.domain.orderbook import models

_SCHEMA_VERSION = 1


class UnableToNormalizeConsolidatedDataError(Exception):
    """Indicate an anticipated invalid or inconsistent normalization input."""


@attrs.frozen
class NormalizationResult:
    """Reference durable provenance for one normalized instrument-year."""

    output_dir: Path
    manifest_path: Path
    terminal_summary_path: Path
    source_files: int
    normalized_files: int
    reused_files: int


def normalize_consolidated_year(
    *,
    source_dir: Path,
    output_dir: Path,
    year: int,
    instrument: models.Instrument,
    replace_output: bool = False,
) -> NormalizationResult:
    """
    Extract one instrument from consolidated daily gzip files losslessly.

    CSV values and their source ordering are preserved. No observation is
    aggregated, resampled, rounded, or otherwise transformed.

    :raises UnableToNormalizeConsolidatedDataError: if inputs or prior output fail
        validation
    """
    started = time.perf_counter()
    sources = tuple(
        consolidated_raw_file.find_consolidated_files(
            source_dir=source_dir, expected_year=year
        )
    )
    if not sources:
        raise UnableToNormalizeConsolidatedDataError(
            f"No consolidated EBS sources found in {source_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"normalization_{year}_{instrument.value}.json"
    prior = _load_manifest(
        path=manifest_path,
        year=year,
        instrument=instrument,
        replace_output=replace_output,
    )
    if replace_output:
        _remove_prior_outputs(
            output_dir=output_dir,
            sources=sources,
            instrument=instrument,
            manifest_path=manifest_path,
        )
        prior = {}
    completed = dict(prior)
    normalized_files = 0
    reused_files = 0
    for position, source in enumerate(sources, start=1):
        output_path = _output_path(
            output_dir=output_dir,
            trading_date=source.trading_date,
            instrument=instrument,
        )
        source_sha256 = consolidated_raw_file.get_content_fingerprint(source=source)
        existing = completed.get(source.path.name)
        if existing is not None:
            _verify_completed_entry(
                entry=existing,
                source=source,
                source_sha256=source_sha256,
                output_path=output_path,
            )
            reused_files += 1
            print(
                f"[normalize-ebs] {position}/{len(sources)} "
                f"date={source.trading_date.isoformat()} checkpoint=reused",
                flush=True,
            )
            continue
        if output_path.exists():
            raise UnableToNormalizeConsolidatedDataError(
                f"Unverified normalized output already exists: {output_path}"
            )
        print(
            f"[normalize-ebs] {position}/{len(sources)} "
            f"date={source.trading_date.isoformat()}",
            flush=True,
        )
        entry = _normalize_file(
            source=source,
            source_sha256=source_sha256,
            output_path=output_path,
            instrument=instrument,
        )
        completed[source.path.name] = entry
        _write_manifest(
            path=manifest_path,
            year=year,
            instrument=instrument,
            entries=completed,
        )
        normalized_files += 1
        print(
            f"[normalize-ebs] {position}/{len(sources)} checkpoint=saved "
            f"selected_rows={entry['selected_rows']}",
            flush=True,
        )
    elapsed = time.perf_counter() - started
    terminal_summary = "\n".join(
        (
            "EBS consolidated source normalization completed",
            "WARNING: observations were filtered by symbol only; no aggregation used.",
            f"instrument={instrument.value}",
            f"year={year}",
            f"source_files={len(sources)}",
            f"normalized_files={normalized_files}",
            f"reused_files={reused_files}",
            f"elapsed_seconds={elapsed:.2f}",
            f"manifest={manifest_path}",
        )
    )
    terminal_summary_path = output_dir / (
        f"normalization_{year}_{instrument.value}_terminal.txt"
    )
    _write_text_atomically(text=terminal_summary, path=terminal_summary_path)
    print(terminal_summary)
    return NormalizationResult(
        output_dir=output_dir,
        manifest_path=manifest_path,
        terminal_summary_path=terminal_summary_path,
        source_files=len(sources),
        normalized_files=normalized_files,
        reused_files=reused_files,
    )


def _load_manifest(
    *,
    path: Path,
    year: int,
    instrument: models.Instrument,
    replace_output: bool,
) -> dict[str, dict[str, object]]:
    if not path.exists() or replace_output:
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UnableToNormalizeConsolidatedDataError(
            f"Unable to read normalization manifest: {path}"
        ) from exc
    if not isinstance(loaded, dict):
        raise UnableToNormalizeConsolidatedDataError(
            "Normalization manifest must be a mapping"
        )
    if (
        loaded.get("schema_version") != _SCHEMA_VERSION
        or loaded.get("year") != year
        or loaded.get("instrument") != instrument.value
        or loaded.get("filter_symbol") != instrument.to_symbol()
        or loaded.get("aggregation_used") is not False
    ):
        raise UnableToNormalizeConsolidatedDataError(
            "Normalization manifest identity does not match this run"
        )
    raw_entries = loaded.get("sessions")
    if not isinstance(raw_entries, list):
        raise UnableToNormalizeConsolidatedDataError(
            "Normalization manifest sessions must be a list"
        )
    entries: dict[str, dict[str, object]] = {}
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict):
            raise UnableToNormalizeConsolidatedDataError(
                "Normalization manifest session must be a mapping"
            )
        entry = cast(dict[str, object], raw_entry)
        source_filename = entry.get("source_filename")
        if not isinstance(source_filename, str) or source_filename in entries:
            raise UnableToNormalizeConsolidatedDataError(
                "Normalization manifest has invalid source identities"
            )
        entries[source_filename] = entry
    return entries


def _remove_prior_outputs(
    *,
    output_dir: Path,
    sources: tuple[consolidated_raw_file.ConsolidatedRawDataFile, ...],
    instrument: models.Instrument,
    manifest_path: Path,
) -> None:
    for source in sources:
        output_path = _output_path(
            output_dir=output_dir,
            trading_date=source.trading_date,
            instrument=instrument,
        )
        if output_path.is_file():
            output_path.unlink()
    if manifest_path.is_file():
        manifest_path.unlink()


def _verify_completed_entry(
    *,
    entry: dict[str, object],
    source: consolidated_raw_file.ConsolidatedRawDataFile,
    source_sha256: str,
    output_path: Path,
) -> None:
    if (
        entry.get("trading_date") != source.trading_date.isoformat()
        or entry.get("source_sha256") != source_sha256
        or entry.get("source_size_bytes") != source.size_bytes
        or entry.get("output_filename") != output_path.name
    ):
        raise UnableToNormalizeConsolidatedDataError(
            f"Source changed after normalization: {source.path}"
        )
    output_sha256 = entry.get("output_sha256")
    if not isinstance(output_sha256, str) or not output_path.is_file():
        raise UnableToNormalizeConsolidatedDataError(
            f"Normalized checkpoint is incomplete: {output_path}"
        )
    if _sha256_file(path=output_path) != output_sha256:
        raise UnableToNormalizeConsolidatedDataError(
            f"Normalized output checksum mismatch: {output_path}"
        )


def _normalize_file(
    *,
    source: consolidated_raw_file.ConsolidatedRawDataFile,
    source_sha256: str,
    output_path: Path,
    instrument: models.Instrument,
) -> dict[str, object]:
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    source_rows = 0
    selected_rows = 0
    quote_rows = 0
    deal_rows = 0
    try:
        with (
            gzip.open(
                source.path,
                mode="rt",
                encoding="utf-8",
                errors="strict",
                newline="",
            ) as input_stream,
            temporary.open("wb") as binary_output,
        ):
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=binary_output,
                mtime=0,
            ) as compressed_output:
                with io.TextIOWrapper(
                    compressed_output,
                    encoding="utf-8",
                    errors="strict",
                    newline="",
                ) as output_stream:
                    reader = csv.reader(input_stream, strict=True)
                    writer = csv.writer(output_stream, lineterminator="\n")
                    for source_rows, columns in enumerate(reader, start=1):
                        if len(columns) < 4:
                            raise UnableToNormalizeConsolidatedDataError(
                                f"Malformed row in {source.path} at line {source_rows}"
                            )
                        if columns[2] != instrument.to_symbol():
                            continue
                        marker = columns[3]
                        required_columns = (
                            9 if marker == "Q" else 10 if marker == "D" else 0
                        )
                        if required_columns == 0 or len(columns) != required_columns:
                            raise UnableToNormalizeConsolidatedDataError(
                                f"Malformed {instrument.value} row in {source.path} "
                                f"at line {source_rows}"
                            )
                        writer.writerow(columns)
                        selected_rows += 1
                        quote_rows += marker == "Q"
                        deal_rows += marker == "D"
        temporary.replace(output_path)
    except UnableToNormalizeConsolidatedDataError:
        temporary.unlink(missing_ok=True)
        raise
    except (OSError, EOFError, UnicodeError, csv.Error) as exc:
        temporary.unlink(missing_ok=True)
        raise UnableToNormalizeConsolidatedDataError(
            f"Unable to normalize consolidated EBS file: {source.path}"
        ) from exc
    return {
        "trading_date": source.trading_date.isoformat(),
        "source_filename": source.path.name,
        "source_size_bytes": source.size_bytes,
        "source_sha256": source_sha256,
        "output_filename": output_path.name,
        "output_size_bytes": output_path.stat().st_size,
        "output_sha256": _sha256_file(path=output_path),
        "source_rows": source_rows,
        "selected_rows": selected_rows,
        "quote_rows": quote_rows,
        "deal_rows": deal_rows,
    }


def _output_path(
    *,
    output_dir: Path,
    trading_date: datetime.date,
    instrument: models.Instrument,
) -> Path:
    date_label = trading_date.strftime("%Y%m%d")
    return output_dir / f"{date_label}-EBS_LVL2_{instrument.value}_0.csv.gz"


def _write_manifest(
    *,
    path: Path,
    year: int,
    instrument: models.Instrument,
    entries: dict[str, dict[str, object]],
) -> None:
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "year": year,
        "instrument": instrument.value,
        "filter_symbol": instrument.to_symbol(),
        "aggregation_used": False,
        "transformation": "symbol_filter_only",
        "sessions": [entries[key] for key in sorted(entries)],
    }
    _write_text_atomically(text=json.dumps(payload, indent=2), path=path)


def _write_text_atomically(*, text: str, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _sha256_file(*, path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
