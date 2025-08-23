"""
FRED (Federal Reserve Economic Data) API Routes
==============================================

FastAPI routes for accessing Federal Reserve Bank of St. Louis economic data.
Provides 32+ economic indicators across 5 institutional categories.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, List, Optional, Any
from datetime import date, datetime, timedelta
import logging

from fred_integration import fred_integration

logger = logging.getLogger(__name__)

# Create the FRED router
router = APIRouter(prefix="/api/v1/fred", tags=["FRED Economic Data"])


@router.get("/health")
async def fred_health_check():
    """Check FRED service health and API connectivity."""
    try:
        health_status = await fred_integration.health_check()
        return health_status
    except Exception as e:
        logger.error(f"FRED health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"FRED health check failed: {str(e)}")


@router.get("/series")
async def get_available_series():
    """Get all available FRED economic series with metadata."""
    try:
        series_info = await fred_integration.get_series_info()
        return {
            "success": True,
            "count": len(series_info),
            "series": series_info
        }
    except Exception as e:
        logger.error(f"Failed to get FRED series info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get series info: {str(e)}")


@router.get("/series/{series_id}")
async def get_series_data(
    series_id: str,
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    limit: int = Query(1000, description="Maximum number of observations")
):
    """Get time series data for a specific FRED economic series."""
    try:
        # Parse dates if provided
        start_dt = None
        end_dt = None
        
        if start_date:
            try:
                start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid start_date format. Use YYYY-MM-DD")
                
        if end_date:
            try:
                end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid end_date format. Use YYYY-MM-DD")
        
        # Get the data
        df = await fred_integration.get_series_data(series_id, start_dt, end_dt, limit)
        
        if df.empty:
            return {
                "success": False,
                "series_id": series_id,
                "message": "No data available for the specified series and date range",
                "data": []
            }
        
        # Convert DataFrame to records
        data = df.to_dict('records')
        
        # Convert timestamps to strings for JSON serialization
        for record in data:
            if 'date' in record:
                record['date'] = record['date'].strftime('%Y-%m-%d')
        
        return {
            "success": True,
            "series_id": series_id,
            "count": len(data),
            "data": data
        }
        
    except Exception as e:
        logger.error(f"Failed to get data for series {series_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get series data: {str(e)}")


@router.get("/series/{series_id}/latest")
async def get_latest_series_value(series_id: str):
    """Get the most recent value for a FRED economic series."""
    try:
        latest_value = await fred_integration.get_latest_value(series_id)
        
        if latest_value is None:
            return {
                "success": False,
                "series_id": series_id,
                "message": "No recent data available for this series",
                "value": None
            }
        
        return {
            "success": True,
            "series_id": series_id,
            "value": latest_value,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get latest value for series {series_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get latest value: {str(e)}")


@router.get("/macro-factors")
async def get_macro_factors(
    as_of_date: Optional[str] = Query(None, description="Calculation date (YYYY-MM-DD), defaults to today")
):
    """
    Calculate institutional-grade macro-economic factors from FRED data.
    
    Returns comprehensive macro factors used in professional factor models:
    - Interest rate levels and changes
    - Yield curve factors (level, slope, curvature)
    - Economic activity indicators
    - Inflation and employment trends
    - Financial market stress indicators
    """
    try:
        # Parse the as_of_date if provided
        calc_date = None
        if as_of_date:
            try:
                calc_date = datetime.strptime(as_of_date, "%Y-%m-%d").date()
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid as_of_date format. Use YYYY-MM-DD")
        
        # Calculate the macro factors
        factors = await fred_integration.calculate_macro_factors(calc_date)
        
        return {
            "success": True,
            "calculation_date": (calc_date or date.today()).strftime("%Y-%m-%d"),
            "factor_count": len(factors),
            "factors": factors,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to calculate macro factors: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate macro factors: {str(e)}")


@router.get("/economic-calendar")
async def get_economic_calendar(
    days_ahead: int = Query(30, description="Number of days ahead to show calendar events")
):
    """Get economic data release calendar for upcoming events."""
    try:
        if days_ahead < 1 or days_ahead > 90:
            raise HTTPException(status_code=400, detail="days_ahead must be between 1 and 90")
        
        calendar_events = await fred_integration.get_economic_calendar(days_ahead)
        
        return {
            "success": True,
            "days_ahead": days_ahead,
            "event_count": len(calendar_events),
            "events": calendar_events
        }
        
    except Exception as e:
        logger.error(f"Failed to get economic calendar: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get economic calendar: {str(e)}")


@router.get("/cache/refresh")
async def refresh_cache():
    """Clear and refresh the FRED data cache."""
    try:
        # Clear the cache
        fred_integration.cache.clear()
        
        # Verify API connectivity by getting a test value
        test_value = await fred_integration.get_latest_value("FEDFUNDS")
        
        return {
            "success": True,
            "message": "Cache refreshed successfully",
            "cache_size": len(fred_integration.cache),
            "test_connectivity": test_value is not None,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to refresh FRED cache: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to refresh cache: {str(e)}")


@router.get("/status")
async def get_fred_status():
    """Get comprehensive FRED service status and statistics."""
    try:
        health = await fred_integration.health_check()
        series_info = await fred_integration.get_series_info()
        
        # Try to get some key economic indicators for status
        key_indicators = {}
        try:
            key_indicators["fed_funds_rate"] = await fred_integration.get_latest_value("FEDFUNDS")
            key_indicators["unemployment_rate"] = await fred_integration.get_latest_value("UNRATE")
            key_indicators["10y_treasury"] = await fred_integration.get_latest_value("DGS10")
        except:
            pass  # Don't fail the status call if indicators are unavailable
        
        return {
            "success": True,
            "service_health": health,
            "available_series": len(series_info),
            "cache_size": len(fred_integration.cache),
            "key_indicators": key_indicators,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get FRED status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")