"""Orchestrate one idempotent raw-to-canonical partition export."""

from __future__ import annotations

import hashlib
import json
import logging

from ebs_tft.application import config
from ebs_tft.data.parsers import ebs_csv
from ebs_tft.data.repositories import processed, raw_file
from ebs_tft.domain.orderbook import models, operations

logger = logging.getLogger(__name__)


class UnableToExportPartitionError(Exception):
    """Indicate an anticipated invalid input or unresolved export configuration."""


def export_file(
    *, raw_data_file: raw_file.RawDataFile, project_config: config.ProjectConfig
) -> processed.ProcessedPartition:
    """Parse once, reconstruct, and atomically publish one full-depth partition."""
    try:
        instrument = models.Instrument.from_filename_part(
            instrument=raw_data_file.instrument
        )
        staleness = project_config.training.maximum_quote_staleness_seconds
        if staleness is None:
            raise UnableToExportPartitionError(
                "maximum_quote_staleness_seconds must be configured"
            )
        config_fingerprint = _config_fingerprint(project_config=project_config)
        source_fingerprint = raw_file.get_content_fingerprint(
            raw_data_file=raw_data_file
        )
        if processed.is_current(
            processed_dir=project_config.training.processed_data_dir,
            instrument=instrument,
            trading_date=raw_data_file.trading_date,
            source_fingerprint=source_fingerprint,
            config_fingerprint=config_fingerprint,
        ):
            return processed.load_partition(
                manifest_path=processed.get_partition_dir(
                    processed_dir=project_config.training.processed_data_dir,
                    instrument=instrument,
                    trading_date=raw_data_file.trading_date,
                )
                / "manifest.json"
            )

        parse_audit = ebs_csv.ParseAudit()
        build_audit = operations.BuildAudit()
        records = ebs_csv.parse_rows(
            path=raw_data_file.path,
            expected_instrument=instrument,
            expected_trading_date=raw_data_file.trading_date,
            strict=True,
            audit=parse_audit,
        )
        bars = operations.build_bars(
            records=records,
            instrument=instrument,
            trading_date=raw_data_file.trading_date,
            maximum_depth=project_config.instruments.maximum_depth,
            maximum_staleness_seconds=staleness,
            audit=build_audit,
        )
        if parse_audit.physical_lines != parse_audit.accounted_lines:
            raise UnableToExportPartitionError(
                "parser line accounting did not reconcile"
            )
        quality = {
            "physical_lines": parse_audit.physical_lines,
            "quote_rows": parse_audit.quote_rows,
            "deal_rows": parse_audit.deal_rows,
            "error_rows": parse_audit.error_rows,
            "quote_snapshots": build_audit.quote_snapshots,
            "quote_resets": build_audit.quote_resets,
            "invalid_book_bars": build_audit.invalid_book_bars,
            "stale_book_bars": build_audit.stale_book_bars,
        }
        return processed.write_bars(
            processed_dir=project_config.training.processed_data_dir,
            instrument=instrument,
            trading_date=raw_data_file.trading_date,
            data=bars,
            maximum_depth=project_config.instruments.maximum_depth,
            source_fingerprint=source_fingerprint,
            config_fingerprint=config_fingerprint,
            quality=quality,
        )
    except UnableToExportPartitionError:
        raise
    except (
        ValueError,
        ebs_csv.UnableToParseRowError,
        operations.InvalidBookStateError,
        processed.UnableToWriteBarsError,
    ) as exc:
        raise UnableToExportPartitionError(
            f"Unable to export {raw_data_file.path}"
        ) from exc
    except Exception:
        logger.exception(
            "Unexpected failure exporting EBS partition",
            extra={"path": str(raw_data_file.path)},
        )
        raise


def _config_fingerprint(*, project_config: config.ProjectConfig) -> str:
    payload = {
        "bar_frequency": project_config.training.bar_frequency,
        "maximum_depth": project_config.instruments.maximum_depth,
        "maximum_quote_staleness_seconds": (
            project_config.training.maximum_quote_staleness_seconds
        ),
        "session_calendar": project_config.training.session_calendar,
        "source_timezone": project_config.training.source_timezone,
        "storage_schema_version": processed.SCHEMA_VERSION,
    }
    encoded = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()
