"""
Data.gov Connector Package
=========================

Data.gov integration for the Nautilus trading platform.
Provides access to 346,000+ federal datasets through CKAN API.
"""

from .api_client import DatagovAPIClient
from .config import create_default_config, create_instrument_config, DatagovInstrumentConfig
from .data_types import DatasetCategory, DatasetFrequency, DatagovDataset, DatasetResource
from .instrument_provider import DatagovInstrumentProvider
from .utils import validate_dataset_id, validate_organization, format_dataset_title

__all__ = [
    "DatagovAPIClient",
    "create_default_config",
    "create_instrument_config",
    "DatagovInstrumentConfig",
    "DatasetCategory",
    "DatasetFrequency",
    "DatagovDataset", 
    "DatasetResource",
    "DatagovInstrumentProvider",
    "validate_dataset_id",
    "validate_organization",
    "format_dataset_title"
]