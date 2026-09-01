"""Orchestrate one idempotent raw-to-canonical partition export."""

from __future__ import annotations

import hashlib
import json
import logging

from ebs_tft.application import config
from ebs_tft.data.parsers import ebs_csv
from ebs_tft.data.repositories import processed, raw_file
from ebs_tft.domain.orderbook import models
from ebs_tft.domain.pilot import operations as pilot_operations

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
        staleness_milliseconds = (
            project_config.training.maximum_quote_staleness_milliseconds
        )
        if staleness_milliseconds is None:
            raise UnableToExportPartitionError(
                "maximum_quote_staleness_milliseconds must be configured"
            )
        state_interval = project_config.training.state_interval_milliseconds
        staleness_steps = staleness_milliseconds // state_interval
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
        records = ebs_csv.parse_rows(
            path=raw_data_file.path,
            expected_instrument=instrument,
            expected_trading_date=raw_data_file.trading_date,
            strict=True,
            audit=parse_audit,
        )
        native_states = pilot_operations.build_native_states(
            records=records,
            instrument=instrument,
            trading_date=raw_data_file.trading_date,
            grid_steps=None,
            maximum_depth=project_config.instruments.maximum_depth,
            maximum_staleness_steps=staleness_steps,
        )
        bars = native_states.select(
            models.canonical_bar_columns(
                max_level=project_config.instruments.maximum_depth
            )
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
            "native_states": len(bars),
            "observed_native_states": bars.filter(
                bars[models.COL_BOOK_OBSERVED]
            ).height,
            "unobserved_native_states": bars.filter(
                ~bars[models.COL_BOOK_OBSERVED]
            ).height,
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
        pilot_operations.InvalidNativeStateError,
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
        "state_interval_milliseconds": (
            project_config.training.state_interval_milliseconds
        ),
        "maximum_depth": project_config.instruments.maximum_depth,
        "maximum_quote_staleness_milliseconds": (
            project_config.training.maximum_quote_staleness_milliseconds
        ),
        "session_calendar": project_config.training.session_calendar,
        "source_timezone": project_config.training.source_timezone,
        "storage_schema_version": processed.SCHEMA_VERSION,
    }
    encoded = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()
