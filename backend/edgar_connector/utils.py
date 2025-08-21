"""
EDGAR Utility Functions
======================

Utility functions for XBRL parsing, data caching, and SEC data processing.
"""

import json
import logging
import re
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import xml.etree.ElementTree as ET
from urllib.parse import urljoin
from functools import lru_cache

logger = logging.getLogger(__name__)


# Input validation functions
def validate_cik(cik: str) -> str:
    """Validate and normalize CIK format.
    
    Args:
        cik: Central Index Key as string
        
    Returns:
        Normalized 10-digit CIK string
        
    Raises:
        ValueError: If CIK format is invalid
    """
    if not cik or not isinstance(cik, str):
        raise ValueError("CIK must be a non-empty string")
    
    # Remove any non-digit characters
    clean_cik = re.sub(r'\D', '', cik)
    
    if not clean_cik:
        raise ValueError("CIK must contain digits")
    
    if len(clean_cik) > 10:
        raise ValueError("CIK cannot exceed 10 digits")
    
    # Zero-pad to 10 digits
    return clean_cik.zfill(10)


def validate_ticker(ticker: str) -> str:
    """Validate ticker symbol format.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Normalized uppercase ticker
        
    Raises:
        ValueError: If ticker format is invalid
    """
    if not ticker or not isinstance(ticker, str):
        raise ValueError("Ticker must be a non-empty string")
    
    ticker = ticker.strip().upper()
    
    # Basic ticker validation (letters and dots only, 1-5 characters)
    if not re.match(r'^[A-Z]{1,5}(?:\.[A-Z]{1,2})?$', ticker):
        raise ValueError("Invalid ticker format. Must be 1-5 letters, optionally followed by .XX")
    
    return ticker


def validate_form_types(form_types: List[str]) -> List[str]:
    """Validate SEC form types.
    
    Args:
        form_types: List of form type strings
        
    Returns:
        Validated list of form types
        
    Raises:
        ValueError: If any form type is invalid
    """
    if not form_types:
        return []
    
    valid_forms = {
        '10-K', '10-Q', '8-K', 'DEF 14A', 'S-1', '13F-HR',
        '4', '3', '5', 'SC 13D', 'SC 13G', '11-K', '10-K/A',
        '10-Q/A', '8-K/A', 'PREC14A', 'PRER14A'
    }
    
    validated = []
    for form_type in form_types:
        if not isinstance(form_type, str):
            raise ValueError(f"Form type must be string, got {type(form_type)}")
        
        form_type = form_type.strip().upper()
        if form_type not in valid_forms:
            logger.warning(f"Unknown form type: {form_type}")
        
        validated.append(form_type)
    
    return validated


@lru_cache(maxsize=10000)
def normalize_cik(cik: str) -> str:
    """Cached CIK normalization for performance.
    
    Args:
        cik: Central Index Key
        
    Returns:
        Normalized 10-digit CIK
    """
    return validate_cik(cik)


def safe_parse_date(date_str: str, format_str: str = "%Y-%m-%d") -> Optional[datetime]:
    """Safely parse date string with error handling.
    
    Args:
        date_str: Date string to parse
        format_str: Expected date format
        
    Returns:
        Parsed datetime or None if parsing fails
    """
    try:
        return datetime.strptime(date_str, format_str)
    except (ValueError, TypeError) as e:
        logger.warning(f"Failed to parse date '{date_str}': {e}")
        return None


class XBRLParser:
    """Parser for XBRL financial data."""
    
    @staticmethod
    def parse_company_facts(facts_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse company facts JSON from SEC API into structured format.
        
        Args:
            facts_data: Raw company facts data from SEC API
            
        Returns:
            Structured financial data
        """
        try:
            parsed_facts = {
                "entity_name": facts_data.get("entityName", ""),
                "cik": facts_data.get("cik", ""),
                "financial_data": {},
                "taxonomies": []
            }
            
            facts = facts_data.get("facts", {})
            
            for taxonomy, concepts in facts.items():
                parsed_facts["taxonomies"].append(taxonomy)
                
                for concept, concept_data in concepts.items():
                    if "units" not in concept_data:
                        continue
                    
                    # Process each unit type (USD, shares, etc.)
                    for unit, unit_data in concept_data["units"].items():
                        key = f"{taxonomy}:{concept}:{unit}"
                        
                        # Get the most recent value
                        if unit_data and isinstance(unit_data, list):
                            latest_entry = max(unit_data, key=lambda x: x.get("end", ""))
                            parsed_facts["financial_data"][key] = {
                                "value": latest_entry.get("val"),
                                "end_date": latest_entry.get("end"),
                                "start_date": latest_entry.get("start"),
                                "form": latest_entry.get("form"),
                                "frame": latest_entry.get("frame"),
                                "unit": unit,
                                "concept": concept,
                                "taxonomy": taxonomy
                            }
            
            return parsed_facts
            
        except Exception as e:
            logger.error(f"Error parsing company facts: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def extract_key_metrics(facts_data: Dict[str, Any]) -> Dict[str, Optional[Decimal]]:
        """
        Extract key financial metrics from company facts.
        
        Args:
            facts_data: Company facts data
            
        Returns:
            Dict of key financial metrics
        """
        metrics = {
            "revenue": None,
            "net_income": None,
            "total_assets": None,
            "total_liabilities": None,
            "stockholders_equity": None,
            "operating_cash_flow": None,
            "shares_outstanding": None
        }
        
        try:
            parsed = XBRLParser.parse_company_facts(facts_data)
            financial_data = parsed.get("financial_data", {})
            
            # Map GAAP concepts to our standard metrics
            concept_mappings = {
                "revenue": [
                    "us-gaap:Revenues:USD",
                    "us-gaap:SalesRevenueNet:USD",
                    "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax:USD"
                ],
                "net_income": [
                    "us-gaap:NetIncomeLoss:USD",
                    "us-gaap:ProfitLoss:USD"
                ],
                "total_assets": [
                    "us-gaap:Assets:USD"
                ],
                "total_liabilities": [
                    "us-gaap:Liabilities:USD",
                    "us-gaap:LiabilitiesAndStockholdersEquity:USD"
                ],
                "stockholders_equity": [
                    "us-gaap:StockholdersEquity:USD",
                    "us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest:USD"
                ],
                "operating_cash_flow": [
                    "us-gaap:NetCashProvidedByUsedInOperatingActivities:USD"
                ],
                "shares_outstanding": [
                    "us-gaap:CommonStockSharesOutstanding:shares",
                    "us-gaap:WeightedAverageNumberOfSharesOutstandingBasic:shares"
                ]
            }
            
            for metric, concepts in concept_mappings.items():
                for concept in concepts:
                    if concept in financial_data:
                        value = financial_data[concept].get("value")
                        if value is not None:
                            try:
                                metrics[metric] = Decimal(str(value))
                                break  # Use first found value
                            except (InvalidOperation, ValueError):
                                continue
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error extracting key metrics: {e}")
            return metrics


class DataCache:
    """Simple file-based cache for EDGAR data."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path("edgar_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
    def _cache_file_path(self, key: str) -> Path:
        """Get cache file path for key."""
        # Replace invalid filename characters
        safe_key = key.replace("/", "_").replace(":", "_").replace("?", "_")
        return self.cache_dir / f"{safe_key}.json"
    
    def _is_expired(self, file_path: Path, ttl_seconds: int) -> bool:
        """Check if cache file is expired."""
        if not file_path.exists():
            return True
        
        file_age = datetime.now() - datetime.fromtimestamp(file_path.stat().st_mtime)
        return file_age.total_seconds() > ttl_seconds
    
    def get(self, key: str, ttl_seconds: int = 3600) -> Optional[Any]:
        """Get data from cache if not expired."""
        file_path = self._cache_file_path(key)
        
        if self._is_expired(file_path, ttl_seconds):
            return None
        
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error reading cache file {file_path}: {e}")
            return None
    
    def set(self, key: str, data: Any) -> None:
        """Store data in cache."""
        file_path = self._cache_file_path(key)
        
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, default=str, indent=2)
        except Exception as e:
            logger.warning(f"Error writing cache file {file_path}: {e}")
    
    def clear(self, key: Optional[str] = None) -> None:
        """Clear cache (all or specific key)."""
        if key:
            file_path = self._cache_file_path(key)
            if file_path.exists():
                file_path.unlink()
        else:
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()


def normalize_cik(cik: Union[str, int]) -> str:
    """
    Normalize CIK to 10-digit zero-padded string.
    
    Args:
        cik: CIK as string or integer
        
    Returns:
        Normalized CIK string
    """
    return str(cik).zfill(10)


def normalize_ticker(ticker: str) -> str:
    """
    Normalize ticker symbol.
    
    Args:
        ticker: Raw ticker symbol
        
    Returns:
        Normalized ticker (uppercase, trimmed)
    """
    return ticker.strip().upper()


def parse_accession_number(accession: str) -> Dict[str, str]:
    """
    Parse SEC accession number into components.
    
    Format: NNNNNNNNNN-YY-NNNNNN
    
    Args:
        accession: SEC accession number
        
    Returns:
        Dict with parsed components
    """
    parts = accession.split("-")
    if len(parts) != 3:
        return {"raw": accession}
    
    return {
        "raw": accession,
        "cik": parts[0],
        "year": f"20{parts[1]}" if len(parts[1]) == 2 else parts[1],
        "sequence": parts[2]
    }


def build_document_url(cik: str, accession: str, filename: str) -> str:
    """
    Build URL for SEC document.
    
    Args:
        cik: Central Index Key
        accession: Accession number
        filename: Document filename
        
    Returns:
        Full URL to document
    """
    # Remove dashes from accession number for URL
    accession_clean = accession.replace("-", "")
    cik_padded = normalize_cik(cik)
    
    base_url = "https://www.sec.gov/Archives/edgar/data"
    return f"{base_url}/{int(cik)}/{accession_clean}/{filename}"


def format_financial_value(
    value: Union[str, int, float, Decimal],
    unit: str = "USD",
    scale: Optional[int] = None
) -> str:
    """
    Format financial value for display.
    
    Args:
        value: Financial value
        unit: Unit (USD, shares, etc.)
        scale: Scale factor for the value
        
    Returns:
        Formatted string
    """
    try:
        num_value = Decimal(str(value))
        
        # Apply scale if provided
        if scale:
            num_value = num_value * (10 ** scale)
        
        if unit.upper() == "USD":
            if abs(num_value) >= 1_000_000_000:
                return f"${num_value / 1_000_000_000:.2f}B"
            elif abs(num_value) >= 1_000_000:
                return f"${num_value / 1_000_000:.2f}M"
            elif abs(num_value) >= 1_000:
                return f"${num_value / 1_000:.2f}K"
            else:
                return f"${num_value:.2f}"
        else:
            # For non-currency units, use standard number formatting
            if abs(num_value) >= 1_000_000_000:
                return f"{num_value / 1_000_000_000:.2f}B {unit}"
            elif abs(num_value) >= 1_000_000:
                return f"{num_value / 1_000_000:.2f}M {unit}"
            elif abs(num_value) >= 1_000:
                return f"{num_value / 1_000:.2f}K {unit}"
            else:
                return f"{num_value:.2f} {unit}"
                
    except (InvalidOperation, ValueError):
        return f"{value} {unit}"


def extract_period_dates(filing_data: Dict[str, Any]) -> Dict[str, Optional[datetime]]:
    """
    Extract period start and end dates from filing data.
    
    Args:
        filing_data: SEC filing data
        
    Returns:
        Dict with period_start and period_end datetime objects
    """
    dates = {
        "period_start": None,
        "period_end": None
    }
    
    try:
        # Look for common date fields in the filing
        date_fields = ["start", "end", "periodStartDate", "periodEndDate"]
        
        for field in date_fields:
            if field in filing_data:
                date_str = filing_data[field]
                if date_str:
                    try:
                        # Handle various date formats
                        if "T" in date_str:
                            date_obj = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                        else:
                            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                        
                        if "start" in field.lower():
                            dates["period_start"] = date_obj
                        elif "end" in field.lower():
                            dates["period_end"] = date_obj
                    except ValueError:
                        continue
        
        return dates
        
    except Exception as e:
        logger.warning(f"Error extracting period dates: {e}")
        return dates


# Constants for common SEC concepts
COMMON_GAAP_CONCEPTS = {
    "Assets": "us-gaap:Assets",
    "Liabilities": "us-gaap:Liabilities",
    "StockholdersEquity": "us-gaap:StockholdersEquity",
    "Revenues": "us-gaap:Revenues",
    "NetIncomeLoss": "us-gaap:NetIncomeLoss",
    "OperatingIncomeLoss": "us-gaap:OperatingIncomeLoss",
    "GrossProfit": "us-gaap:GrossProfit",
    "CashAndCashEquivalentsAtCarryingValue": "us-gaap:CashAndCashEquivalentsAtCarryingValue",
    "CommonStockSharesOutstanding": "us-gaap:CommonStockSharesOutstanding"
}

MAJOR_FILING_TYPES = [
    "10-K",    # Annual report
    "10-Q",    # Quarterly report
    "8-K",     # Current report
    "DEF 14A", # Proxy statement
    "S-1",     # Registration statement
    "13F-HR"   # Holdings report
]