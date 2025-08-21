"""
EDGAR API Connector for NautilusTrader
=====================================

A custom data connector that bridges SEC EDGAR API with NautilusTrader's unified architecture.
Provides structured access to SEC filings for trading algorithms and financial analysis.
"""

from edgar_connector.api_client import EDGARAPIClient
from edgar_connector.config import EDGARConfig
from edgar_connector.data_client import EDGARDataClient
from edgar_connector.data_types import (
    FilingData,
    CompanyFacts,
    SECFiling,
    FilingType,
)
from edgar_connector.instrument_provider import EDGARInstrumentProvider

__all__ = [
    "EDGARAPIClient",
    "EDGARConfig", 
    "EDGARDataClient",
    "EDGARInstrumentProvider",
    "FilingData",
    "CompanyFacts",
    "SECFiling",
    "FilingType",
]

__version__ = "0.1.0"