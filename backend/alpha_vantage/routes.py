"""
Alpha Vantage API Routes
=======================

FastAPI routes for Alpha Vantage market data access.
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Dict, Optional, Any
from datetime import datetime
import logging

from .service import alpha_vantage_service
from .models import (
    AlphaVantageQuote, AlphaVantageTimeSeries, AlphaVantageCompany,
    AlphaVantageSearchResult, AlphaVantageEarnings, AlphaVantageHealthStatus
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/alpha-vantage", tags=["Alpha Vantage"])


@router.get("/health", response_model=AlphaVantageHealthStatus)
async def health_check():
    """Check Alpha Vantage API health and connectivity."""
    try:
        return await alpha_vantage_service.health_check()
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/quote/{symbol}", response_model=AlphaVantageQuote)
async def get_quote(symbol: str):
    """Get real-time quote for a stock symbol."""
    try:
        quote = await alpha_vantage_service.get_quote(symbol)
        return quote
    except Exception as e:
        logger.error(f"Failed to get quote for {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get quote for {symbol}: {str(e)}"
        )


@router.post("/quotes", response_model=Dict[str, AlphaVantageQuote])
async def get_multiple_quotes(symbols: List[str]):
    """Get quotes for multiple symbols."""
    try:
        if len(symbols) > 10:
            raise HTTPException(
                status_code=400,
                detail="Maximum 10 symbols allowed per request"
            )
        
        quotes = await alpha_vantage_service.get_multiple_quotes(symbols)
        return quotes
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get quotes for {symbols}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get quotes: {str(e)}"
        )


@router.get("/intraday/{symbol}", response_model=AlphaVantageTimeSeries)
async def get_intraday_data(
    symbol: str,
    interval: str = Query(default="5min", regex="^(1min|5min|15min|30min|60min)$"),
    extended_hours: bool = Query(default=True)
):
    """Get intraday time series data."""
    try:
        data = await alpha_vantage_service.get_intraday_data(symbol, interval, extended_hours)
        return data
    except Exception as e:
        logger.error(f"Failed to get intraday data for {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get intraday data for {symbol}: {str(e)}"
        )


@router.get("/daily/{symbol}", response_model=AlphaVantageTimeSeries)
async def get_daily_data(
    symbol: str,
    outputsize: str = Query(default="compact", regex="^(compact|full)$")
):
    """Get daily time series data."""
    try:
        data = await alpha_vantage_service.get_daily_data(symbol, outputsize)
        return data
    except Exception as e:
        logger.error(f"Failed to get daily data for {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get daily data for {symbol}: {str(e)}"
        )


@router.get("/search", response_model=List[AlphaVantageSearchResult])
async def search_symbols(
    keywords: str = Query(..., description="Search keywords for symbols"),
    max_results: int = Query(default=10, ge=1, le=20)
):
    """Search for symbols using keywords."""
    try:
        results = await alpha_vantage_service.search_symbols(keywords, max_results)
        return results
    except Exception as e:
        logger.error(f"Failed to search symbols for '{keywords}': {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to search symbols: {str(e)}"
        )


@router.get("/company/{symbol}", response_model=Optional[AlphaVantageCompany])
async def get_company_fundamentals(symbol: str):
    """Get company fundamental data and overview."""
    try:
        company = await alpha_vantage_service.get_company_fundamentals(symbol)
        if company is None:
            raise HTTPException(
                status_code=404,
                detail=f"Company data not found for symbol {symbol}"
            )
        return company
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get company data for {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get company data for {symbol}: {str(e)}"
        )


@router.get("/earnings/{symbol}", response_model=List[AlphaVantageEarnings])
async def get_earnings_data(symbol: str):
    """Get quarterly earnings data."""
    try:
        earnings = await alpha_vantage_service.get_earnings_data(symbol)
        return earnings
    except Exception as e:
        logger.error(f"Failed to get earnings data for {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get earnings data for {symbol}: {str(e)}"
        )


@router.get("/supported-functions", response_model=List[str])
async def get_supported_functions():
    """Get list of supported Alpha Vantage API functions."""
    try:
        functions = await alpha_vantage_service.get_supported_functions()
        return functions
    except Exception as e:
        logger.error(f"Failed to get supported functions: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get supported functions: {str(e)}"
        )


@router.get("/status")
async def get_status():
    """Get Alpha Vantage integration status."""
    try:
        health = await alpha_vantage_service.health_check()
        return {
            "service": "Alpha Vantage",
            "status": health.status,
            "api_key_configured": health.api_key_configured,
            "last_successful_request": health.last_successful_request.isoformat() if health.last_successful_request else None,
            "error_message": health.error_message,
            "capabilities": [
                "real_time_quotes",
                "intraday_data",
                "daily_historical_data",
                "symbol_search",
                "company_fundamentals",
                "earnings_data"
            ],
            "rate_limits": {
                "free_tier": "5 requests per minute, 500 requests per day",
                "premium": "Varies by plan"
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get status: {str(e)}"
        )