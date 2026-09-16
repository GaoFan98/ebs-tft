"""Expose the public data-ingestion use cases."""

from ebs_tft.application.usecases.data_ingestion._exporter import (
    UnableToExportPartitionError,
    export_file,
)
from ebs_tft.application.usecases.data_ingestion._normalizer import (
    NormalizationResult,
    UnableToNormalizeConsolidatedDataError,
    normalize_consolidated_year,
)

__all__ = [
    "NormalizationResult",
    "UnableToExportPartitionError",
    "UnableToNormalizeConsolidatedDataError",
    "export_file",
    "normalize_consolidated_year",
]
