"""
EDGAR API Routes
===============

FastAPI routes for SEC EDGAR data integration with the Nautilus trading platform.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, HTTPException, Query, Depends, Path
from pydantic import BaseModel, Field

from edgar_connector.api_client import EDGARAPIClient
from edgar_connector.config import create_default_config, EDGARInstrumentConfig
from edgar_connector.instrument_provider import EDGARInstrumentProvider
from edgar_connector.data_types import FilingType, SECEntity
from edgar_connector.utils import XBRLParser, format_financial_value, validate_cik, validate_ticker, validate_form_types

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/edgar", tags=["EDGAR"])

from functools import lru_cache

# Pydantic models for API responses
class CompanyEntity(BaseModel):
    """Company entity information."""
    cik: str = Field(description="Central Index Key")
    name: str = Field(description="Company name")
    ticker: Optional[str] = Field(None, description="Stock ticker")
    exchange: Optional[str] = Field(None, description="Exchange")


class FilingInfo(BaseModel):
    """Filing information."""
    form_type: str = Field(description="Filing form type")
    filing_date: str = Field(description="Filing date (YYYY-MM-DD)")
    accession_number: str = Field(description="SEC accession number")
    company_name: str = Field(description="Company name")
    cik: str = Field(description="Central Index Key")
    days_ago: int = Field(description="Days since filing")


class FinancialMetric(BaseModel):
    """Financial metric information."""
    metric: str = Field(description="Metric name")
    value: Optional[str] = Field(None, description="Formatted value")
    raw_value: Optional[float] = Field(None, description="Raw numeric value")
    period_end: Optional[str] = Field(None, description="Period end date")


class CompanyFactsResponse(BaseModel):
    """Company facts response."""
    cik: str
    company_name: str
    key_metrics: List[FinancialMetric]
    total_facts: int = Field(description="Total number of facts available")


class TickerResolutionResponse(BaseModel):
    """Ticker resolution response."""
    ticker: str
    cik: Optional[str] = None
    company_name: Optional[str] = None
    found: bool


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    api_healthy: bool
    entities_loaded: int
    cache_enabled: bool


# Dependency functions with proper singleton pattern
@lru_cache(maxsize=1)
def _create_api_client() -> EDGARAPIClient:
    """Create singleton API client with proper configuration."""
    config = create_default_config(
        user_agent="NautilusTrader-Backend trading@nautilus-trader.com",
        rate_limit_requests_per_second=5.0,
        cache_ttl_seconds=1800  # 30 minutes
    )
    return EDGARAPIClient(config)

async def get_api_client() -> EDGARAPIClient:
    """Get thread-safe singleton API client."""
    return _create_api_client()


@lru_cache(maxsize=1)
def _create_instrument_provider() -> EDGARInstrumentProvider:
    """Create singleton instrument provider."""
    api_client = _create_api_client()
    config = EDGARInstrumentConfig(
        update_entities_on_startup=True,
        ticker_cache_ttl=3600  # 1 hour
    )
    return EDGARInstrumentProvider(api_client, config)

async def get_instrument_provider() -> EDGARInstrumentProvider:
    """Get thread-safe singleton instrument provider."""
    provider = _create_instrument_provider()
    # Ensure entities are loaded
    if not provider._is_loaded:
        await provider.load_all_async()
    return provider


# API Routes

@router.get("/health", response_model=HealthResponse)
async def health_check(
    api_client: EDGARAPIClient = Depends(get_api_client),
    instrument_provider: EDGARInstrumentProvider = Depends(get_instrument_provider)
):
    """Check EDGAR API health and service status."""
    try:
        api_healthy = await api_client.health_check()
        entities_count = instrument_provider.get_entity_count()
        
        return HealthResponse(
            status="healthy" if api_healthy else "degraded",
            timestamp=datetime.now().isoformat(),
            api_healthy=api_healthy,
            entities_loaded=entities_count,
            cache_enabled=api_client.config.enable_cache
        )
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unavailable: {e}")


@router.get("/companies/search", response_model=List[CompanyEntity])
async def search_companies(
    q: str = Query(..., description="Search query (company name or ticker)"),
    limit: int = Query(10, ge=1, le=100, description="Maximum results"),
    instrument_provider: EDGARInstrumentProvider = Depends(get_instrument_provider)
):
    """Search for companies by name or ticker."""
    try:
        entities = instrument_provider.search_entities(q, limit)
        
        return [
            CompanyEntity(
                cik=entity.cik,
                name=entity.name,
                ticker=entity.ticker,
                exchange=entity.exchange
            )
            for entity in entities
        ]
    
    except Exception as e:
        logger.error(f"Company search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")


@router.get("/companies/{cik}/facts", response_model=CompanyFactsResponse)
async def get_company_facts(
    cik: str = Path(..., description="Central Index Key"),
    api_client: EDGARAPIClient = Depends(get_api_client),
    instrument_provider: EDGARInstrumentProvider = Depends(get_instrument_provider)
):
    """Get financial facts for a specific company."""
    try:
        # Validate and normalize CIK
        try:
            normalized_cik = validate_cik(cik)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid CIK format: {e}")
        
        # Get company entity for name
        entity = instrument_provider.get_entity_by_cik(normalized_cik)
        company_name = entity.name if entity else "Unknown Company"
        
        # Fetch facts from API
        facts_data = await api_client.get_company_facts(normalized_cik)
        
        if not facts_data:
            raise HTTPException(status_code=404, detail=f"No facts found for CIK {cik}")
        
        # Parse key metrics
        parser = XBRLParser()
        key_metrics_dict = parser.extract_key_metrics(facts_data)
        
        # Convert to response format
        key_metrics = []
        for metric_name, value in key_metrics_dict.items():
            if value is not None:
                key_metrics.append(FinancialMetric(
                    metric=metric_name.replace('_', ' ').title(),
                    value=format_financial_value(value, "USD"),
                    raw_value=float(value)
                ))
        
        # Count total facts
        total_facts = 0
        if "facts" in facts_data:
            for taxonomy in facts_data["facts"].values():
                total_facts += len(taxonomy)
        
        return CompanyFactsResponse(
            cik=normalized_cik,
            company_name=company_name,
            key_metrics=key_metrics,
            total_facts=total_facts
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get company facts for {cik}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve facts: {e}")


@router.get("/companies/{cik}/filings", response_model=List[FilingInfo])
async def get_company_filings(
    cik: str = Path(..., description="Central Index Key"),
    form_types: Optional[List[str]] = Query(None, description="Filter by form types"),
    days_back: int = Query(365, ge=1, le=3650, description="Days back to search"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results"),
    api_client: EDGARAPIClient = Depends(get_api_client)
):
    """Get recent filings for a specific company."""
    try:
        # Validate and normalize CIK
        try:
            normalized_cik = validate_cik(cik)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid CIK format: {e}")
        
        # Validate form types if provided
        if form_types:
            try:
                form_types = validate_form_types(form_types)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid form types: {e}")
        
        # Get submissions
        submissions = await api_client.get_submissions(normalized_cik)
        
        if not submissions or "filings" not in submissions:
            raise HTTPException(status_code=404, detail=f"No filings found for CIK {cik}")
        
        recent_filings = submissions["filings"].get("recent", {})
        
        forms = recent_filings.get("form", [])
        dates = recent_filings.get("filingDate", [])
        accessions = recent_filings.get("accessionNumber", [])
        
        if not forms or not dates or not accessions:
            return []
        
        # Filter and format results
        cutoff_date = datetime.now() - timedelta(days=days_back)
        company_name = submissions.get("name", "Unknown Company")
        
        filings = []
        for form, date_str, accession in zip(forms, dates, accessions):
            try:
                filing_date = datetime.strptime(date_str, "%Y-%m-%d")
                
                # Apply filters
                if filing_date < cutoff_date:
                    continue
                
                if form_types and form not in form_types:
                    continue
                
                days_ago = (datetime.now() - filing_date).days
                
                filings.append(FilingInfo(
                    form_type=form,
                    filing_date=date_str,
                    accession_number=accession,
                    company_name=company_name,
                    cik=normalized_cik,
                    days_ago=days_ago
                ))
                
                if len(filings) >= limit:
                    break
            
            except ValueError:
                continue
        
        return filings
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get filings for {cik}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve filings: {e}")


@router.get("/ticker/{ticker}/resolve", response_model=TickerResolutionResponse)
async def resolve_ticker(
    ticker: str = Path(..., description="Stock ticker symbol"),
    instrument_provider: EDGARInstrumentProvider = Depends(get_instrument_provider)
):
    """Resolve a ticker symbol to CIK and company information."""
    try:
        # Validate ticker format
        try:
            ticker_validated = validate_ticker(ticker)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid ticker format: {e}")
        
        # Try to resolve ticker
        cik = instrument_provider.resolve_ticker_to_cik(ticker_validated)
        
        if cik:
            entity = instrument_provider.get_entity_by_cik(cik)
            return TickerResolutionResponse(
                ticker=ticker_validated,
                cik=cik,
                company_name=entity.name if entity else None,
                found=True
            )
        else:
            return TickerResolutionResponse(
                ticker=ticker_validated,
                found=False
            )
    
    except Exception as e:
        logger.error(f"Failed to resolve ticker {ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Resolution failed: {e}")


@router.get("/ticker/{ticker}/facts", response_model=CompanyFactsResponse)
async def get_facts_by_ticker(
    ticker: str = Path(..., description="Stock ticker symbol"),
    api_client: EDGARAPIClient = Depends(get_api_client),
    instrument_provider: EDGARInstrumentProvider = Depends(get_instrument_provider)
):
    """Get financial facts by ticker symbol."""
    try:
        # Validate ticker format
        try:
            ticker_validated = validate_ticker(ticker)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid ticker format: {e}")
        
        # Resolve ticker to CIK
        cik = instrument_provider.resolve_ticker_to_cik(ticker_validated)
        
        if not cik:
            raise HTTPException(status_code=404, detail=f"Ticker {ticker} not found")
        
        # Forward to CIK-based endpoint
        return await get_company_facts(cik, api_client, instrument_provider)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get facts for ticker {ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve facts: {e}")


@router.get("/ticker/{ticker}/filings", response_model=List[FilingInfo])
async def get_filings_by_ticker(
    ticker: str = Path(..., description="Stock ticker symbol"),
    form_types: Optional[List[str]] = Query(None, description="Filter by form types"),
    days_back: int = Query(365, ge=1, le=3650, description="Days back to search"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results"),
    api_client: EDGARAPIClient = Depends(get_api_client),
    instrument_provider: EDGARInstrumentProvider = Depends(get_instrument_provider)
):
    """Get recent filings by ticker symbol."""
    try:
        # Validate ticker format and form types
        try:
            ticker_validated = validate_ticker(ticker)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid ticker format: {e}")
        
        if form_types:
            try:
                form_types = validate_form_types(form_types)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid form types: {e}")
        
        # Resolve ticker to CIK
        cik = instrument_provider.resolve_ticker_to_cik(ticker_validated)
        
        if not cik:
            raise HTTPException(status_code=404, detail=f"Ticker {ticker} not found")
        
        # Forward to CIK-based endpoint
        return await get_company_filings(cik, form_types, days_back, limit, api_client)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get filings for ticker {ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve filings: {e}")


@router.get("/filing-types", response_model=List[str])
async def get_filing_types():
    """Get list of supported filing types."""
    return [filing_type.value for filing_type in FilingType]


@router.get("/statistics")
async def get_statistics(
    instrument_provider: EDGARInstrumentProvider = Depends(get_instrument_provider)
):
    """Get EDGAR service statistics."""
    try:
        stats = instrument_provider.get_statistics()
        
        return {
            "service": "edgar",
            "status": "active",
            "timestamp": datetime.now().isoformat(),
            **stats
        }
    
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {e}")


# Startup and shutdown handlers
@router.on_event("startup")
async def startup_event():
    """Initialize EDGAR services on startup."""
    logger.info("Initializing EDGAR API services...")
    try:
        # Pre-load the instrument provider
        await get_instrument_provider()
        logger.info("EDGAR services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize EDGAR services: {e}")


@router.on_event("shutdown")
async def shutdown_event():
    """Clean up EDGAR services on shutdown."""
    logger.info("Shutting down EDGAR services...")
    try:
        # Clear LRU cache to allow proper cleanup
        _create_api_client.cache_clear()
        _create_instrument_provider.cache_clear()
        logger.info("EDGAR services shut down successfully")
    except Exception as e:
        logger.error(f"Error during EDGAR services shutdown: {e}")