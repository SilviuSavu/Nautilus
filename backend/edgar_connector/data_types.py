"""
EDGAR Data Types
===============

Custom data types for SEC filings and financial data compatible with NautilusTrader.
"""

from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum
from decimal import Decimal
from pydantic import BaseModel, Field

from nautilus_trader.model.data import CustomData
from nautilus_trader.core.uuid import UUID4


class FilingType(str, Enum):
    """SEC filing types."""
    
    FORM_10K = "10-K"           # Annual report
    FORM_10Q = "10-Q"           # Quarterly report  
    FORM_8K = "8-K"             # Current report
    FORM_DEF14A = "DEF 14A"     # Proxy statement
    FORM_S1 = "S-1"             # Registration statement
    FORM_13F = "13F-HR"         # Holdings report
    FORM_4 = "4"                # Statement of changes in ownership
    FORM_3 = "3"                # Initial statement of ownership
    FORM_5 = "5"                # Annual statement of ownership
    FORM_SC13D = "SC 13D"       # Beneficial ownership report
    FORM_SC13G = "SC 13G"       # Beneficial ownership report (passive)
    
    @classmethod
    def get_major_forms(cls) -> List[str]:
        """Get list of major filing forms."""
        return [cls.FORM_10K, cls.FORM_10Q, cls.FORM_8K, cls.FORM_DEF14A]


class SECEntity(BaseModel):
    """SEC entity information."""
    
    cik: str = Field(description="Central Index Key (CIK)")
    name: str = Field(description="Entity name")
    ticker: Optional[str] = Field(None, description="Stock ticker symbol")
    exchange: Optional[str] = Field(None, description="Stock exchange")
    sic: Optional[str] = Field(None, description="SIC industry code") 
    state_of_incorporation: Optional[str] = Field(None, description="State of incorporation")
    fiscal_year_end: Optional[str] = Field(None, description="Fiscal year end (MMDD format)")


class FilingData(CustomData):
    """SEC filing data compatible with NautilusTrader."""
    
    def __init__(
        self,
        filing_type: FilingType,
        cik: str,
        company_name: str,
        accession_number: str,
        filing_date: datetime,
        period_end_date: Optional[datetime] = None,
        report_url: Optional[str] = None,
        financial_data: Optional[Dict[str, Any]] = None,
        raw_data: Optional[Dict[str, Any]] = None,
        ts_event: Optional[int] = None,
        ts_init: Optional[int] = None,
    ):
        # Convert datetime to nanosecond timestamp for NautilusTrader
        if ts_event is None:
            ts_event = int(filing_date.timestamp() * 1_000_000_000)
        
        if ts_init is None:
            ts_init = int(datetime.utcnow().timestamp() * 1_000_000_000)
            
        super().__init__(ts_event, ts_init)
        
        self.filing_type = filing_type
        self.cik = cik
        self.company_name = company_name
        self.accession_number = accession_number
        self.filing_date = filing_date
        self.period_end_date = period_end_date
        self.report_url = report_url
        self.financial_data = financial_data or {}
        self.raw_data = raw_data or {}


class CompanyFacts(CustomData):
    """Company financial facts from XBRL data."""
    
    def __init__(
        self,
        cik: str,
        company_name: str,
        facts: Dict[str, Any],
        fiscal_year: Optional[int] = None,
        fiscal_quarter: Optional[int] = None,
        ts_event: Optional[int] = None,
        ts_init: Optional[int] = None,
    ):
        if ts_event is None:
            ts_event = int(datetime.utcnow().timestamp() * 1_000_000_000)
        
        if ts_init is None:
            ts_init = ts_event
            
        super().__init__(ts_event, ts_init)
        
        self.cik = cik
        self.company_name = company_name
        self.facts = facts
        self.fiscal_year = fiscal_year
        self.fiscal_quarter = fiscal_quarter


class SECFiling(BaseModel):
    """Structured SEC filing information."""
    
    cik: str
    company_name: str
    form_type: FilingType
    accession_number: str
    filing_date: datetime
    report_date: Optional[datetime] = None
    period_of_report: Optional[datetime] = None
    document_url: Optional[str] = None
    interactive_data_url: Optional[str] = None
    
    # Financial highlights (extracted from filing)
    revenue: Optional[Decimal] = None
    net_income: Optional[Decimal] = None
    total_assets: Optional[Decimal] = None
    total_liabilities: Optional[Decimal] = None
    stockholders_equity: Optional[Decimal] = None
    
    # Additional metadata
    file_size: Optional[int] = None
    amendment: bool = False
    exhibit_count: Optional[int] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Decimal: str
        }


class FinancialMetric(BaseModel):
    """Individual financial metric with metadata."""
    
    name: str
    value: Decimal
    unit: str
    currency: Optional[str] = "USD"
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None
    fiscal_year: Optional[int] = None
    fiscal_quarter: Optional[int] = None
    taxonomy: Optional[str] = None  # GAAP, IFRS, etc.


class FilingSubmission(BaseModel):
    """SEC submission containing multiple filings."""
    
    cik: str
    company_name: str
    filings: List[SECFiling]
    submission_date: datetime
    total_count: int
    
    @property
    def latest_10k(self) -> Optional[SECFiling]:
        """Get the most recent 10-K filing."""
        ten_k_filings = [f for f in self.filings if f.form_type == FilingType.FORM_10K]
        return max(ten_k_filings, key=lambda f: f.filing_date) if ten_k_filings else None
    
    @property
    def latest_10q(self) -> Optional[SECFiling]:
        """Get the most recent 10-Q filing."""
        ten_q_filings = [f for f in self.filings if f.form_type == FilingType.FORM_10Q]
        return max(ten_q_filings, key=lambda f: f.filing_date) if ten_q_filings else None


class EDGARSubscription(BaseModel):
    """Subscription configuration for EDGAR data."""
    
    cik: Optional[str] = None
    ticker: Optional[str] = None
    filing_types: List[FilingType] = Field(default_factory=lambda: [FilingType.FORM_10K, FilingType.FORM_10Q])
    include_amendments: bool = False
    max_age_days: int = 365
    
    def __post_init__(self):
        if not self.cik and not self.ticker:
            raise ValueError("Either cik or ticker must be specified")


# Factory functions for creating data types
def create_filing_data(
    filing_type: str,
    cik: str,
    company_name: str,
    accession_number: str,
    filing_date: datetime,
    **kwargs
) -> FilingData:
    """Factory function to create FilingData instances."""
    return FilingData(
        filing_type=FilingType(filing_type),
        cik=cik,
        company_name=company_name,
        accession_number=accession_number,
        filing_date=filing_date,
        **kwargs
    )


def create_company_facts(
    cik: str,
    company_name: str,
    facts_data: Dict[str, Any],
    **kwargs
) -> CompanyFacts:
    """Factory function to create CompanyFacts instances."""
    return CompanyFacts(
        cik=cik,
        company_name=company_name,
        facts=facts_data,
        **kwargs
    )