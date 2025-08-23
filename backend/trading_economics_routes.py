"""
Trading Economics API Routes
============================

FastAPI routes for accessing Trading Economics global economic data.
Provides 300,000+ economic indicators across 196 countries.
"""

from fastapi import APIRouter, HTTPException, Query, Path
from typing import Dict, List, Optional, Any, Union
from datetime import date, datetime, timedelta
import logging

from trading_economics_integration import trading_economics_integration

logger = logging.getLogger(__name__)

# Create the Trading Economics router
router = APIRouter(prefix="/api/v1/trading-economics", tags=["Trading Economics"])


@router.get("/health")
async def trading_economics_health_check():
    """Check Trading Economics service health and API connectivity."""
    try:
        health_status = await trading_economics_integration.health_check()
        return health_status
    except Exception as e:
        logger.error(f"Trading Economics health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Trading Economics health check failed: {str(e)}")


@router.get("/countries")
async def get_available_countries():
    """Get all available countries with economic data."""
    try:
        countries_info = await trading_economics_integration.get_countries()
        return {
            "success": True,
            "count": len(countries_info),
            "countries": countries_info
        }
    except Exception as e:
        logger.error(f"Failed to get Trading Economics countries: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get countries info: {str(e)}")


@router.get("/indicators")
async def get_available_indicators():
    """Get all available economic indicators with metadata."""
    try:
        indicators_info = await trading_economics_integration.get_indicators()
        return {
            "success": True,
            "count": len(indicators_info),
            "indicators": indicators_info
        }
    except Exception as e:
        logger.error(f"Failed to get Trading Economics indicators: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get indicators info: {str(e)}")


@router.get("/indicators/{country}")
async def get_country_indicators(
    country: str,
    category: Optional[str] = Query(None, description="Filter by category (e.g., GDP, inflation, employment)")
):
    """Get economic indicators for a specific country."""
    try:
        indicators = await trading_economics_integration.get_indicator_data(
            country=country,
            category=category
        )
        return {
            "success": True,
            "country": country,
            "category": category,
            "count": len(indicators),
            "data": indicators
        }
    except Exception as e:
        logger.error(f"Failed to get indicators for {country}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get indicators for {country}: {str(e)}")


@router.get("/indicator/{country}/{indicator}")
async def get_specific_indicator(
    country: str,
    indicator: str,
    start_date: Optional[date] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[date] = Query(None, description="End date (YYYY-MM-DD)")
):
    """Get specific economic indicator data for a country."""
    try:
        data = await trading_economics_integration.get_indicator_data(
            country=country,
            indicator=indicator,
            start_date=start_date,
            end_date=end_date
        )
        return {
            "success": True,
            "country": country,
            "indicator": indicator,
            "data": data
        }
    except Exception as e:
        logger.error(f"Failed to get {indicator} for {country}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get {indicator} for {country}: {str(e)}")


@router.get("/calendar")
async def get_economic_calendar(
    country: Optional[str] = Query(None, description="Filter by country"),
    category: Optional[str] = Query(None, description="Filter by category"),
    importance: Optional[str] = Query(None, description="Filter by importance (high, medium, low)"),
    start_date: Optional[date] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[date] = Query(None, description="End date (YYYY-MM-DD)")
):
    """Get economic calendar events."""
    try:
        calendar_data = await trading_economics_integration.get_calendar(
            country=country,
            category=category,
            importance=importance,
            start_date=start_date,
            end_date=end_date
        )
        return {
            "success": True,
            "filters": {
                "country": country,
                "category": category,
                "importance": importance,
                "start_date": start_date,
                "end_date": end_date
            },
            "count": len(calendar_data),
            "events": calendar_data
        }
    except Exception as e:
        logger.error(f"Failed to get economic calendar: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get economic calendar: {str(e)}")


@router.get("/markets/{market_type}")
async def get_market_data(
    market_type: str = Path(..., description="Market type (currencies, commodities, stocks, bonds)"),
    country: Optional[str] = Query(None, description="Filter by country")
):
    """Get market data by type."""
    try:
        market_data = await trading_economics_integration.get_markets(
            market_type=market_type,
            country=country
        )
        return {
            "success": True,
            "market_type": market_type,
            "country": country,
            "count": len(market_data),
            "data": market_data
        }
    except Exception as e:
        logger.error(f"Failed to get market data for {market_type}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get market data: {str(e)}")


@router.get("/forecast/{country}/{indicator}")
async def get_forecast(
    country: str,
    indicator: str
):
    """Get forecast data for specific indicator."""
    try:
        forecast_data = await trading_economics_integration.get_forecast(
            country=country,
            indicator=indicator
        )
        return {
            "success": True,
            "country": country,
            "indicator": indicator,
            "forecast": forecast_data
        }
    except Exception as e:
        logger.error(f"Failed to get forecast for {indicator} in {country}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get forecast: {str(e)}")


@router.get("/search")
async def search_indicators(
    term: str = Query(..., description="Search term"),
    category: Optional[str] = Query(None, description="Filter by category")
):
    """Search economic indicators by term."""
    try:
        search_results = await trading_economics_integration.search(
            term=term,
            category=category
        )
        return {
            "success": True,
            "search_term": term,
            "category": category,
            "count": len(search_results),
            "results": search_results
        }
    except Exception as e:
        logger.error(f"Failed to search for '{term}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to search: {str(e)}")


@router.get("/statistics")
async def get_statistics():
    """Get Trading Economics service statistics and usage info."""
    try:
        stats = await trading_economics_integration.get_statistics()
        return {
            "success": True,
            "statistics": stats
        }
    except Exception as e:
        logger.error(f"Failed to get Trading Economics statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")


@router.post("/cache/refresh")
async def refresh_cache():
    """Refresh Trading Economics data cache."""
    try:
        cache_status = await trading_economics_integration.refresh_cache()
        return {
            "success": True,
            "cache_refresh": cache_status
        }
    except Exception as e:
        logger.error(f"Failed to refresh Trading Economics cache: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to refresh cache: {str(e)}")


@router.get("/supported-functions")
async def get_supported_functions():
    """Get list of all supported Trading Economics API functions."""
    try:
        functions = await trading_economics_integration.get_supported_functions()
        return {
            "success": True,
            "functions": functions
        }
    except Exception as e:
        logger.error(f"Failed to get supported functions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get supported functions: {str(e)}")