"""
YFinance Routes with Full Implementation
=======================================

Complete YFinance service implementation with:
- Real-time quotes and historical data
- Rate limiting and error handling
- Bulk operations support
- Symbol search functionality
- Robust caching and retry logic

Updated for 2025 with latest yfinance best practices.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

from yfinance_service import yfinance_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/yfinance", tags=["yfinance"])


@router.get("/health")
async def get_yfinance_health() -> Dict[str, Any]:
    """Get YFinance service health check."""
    try:
        return await yfinance_service.health_check()
    except Exception as e:
        logger.error(f"YFinance health check failed: {e}")
        raise HTTPException(status_code=503, detail="YFinance service temporarily unavailable")


@router.get("/status")
async def get_yfinance_status() -> Dict[str, Any]:
    """Get YFinance service status."""
    try:
        health = await yfinance_service.health_check()
        return {
            "status": health.get("status", "unknown"),
            "service": "yfinance",
            "version": health.get("version"),
            "message": "YFinance service with rate limiting and reliability improvements",
            "data_source": "free_tier",
            "features": [
                "Real-time quotes",
                "Historical data", 
                "Bulk operations",
                "Rate limiting protection",
                "Exponential backoff retry",
                "Intelligent caching"
            ],
            "request_count": health.get("request_count", 0),
            "cache_size": health.get("cache_size", 0),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"YFinance status check failed: {e}")
        raise HTTPException(status_code=503, detail="YFinance service temporarily unavailable")


@router.get("/quote/{symbol}")
async def get_quote(symbol: str) -> Dict[str, Any]:
    """Get real-time quote for a symbol."""
    try:
        quote = await yfinance_service.get_quote(symbol)
        return quote.model_dump()
    except Exception as e:
        logger.error(f"Failed to get quote for {symbol}: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to retrieve quote for {symbol}: {str(e)}"
        )


@router.get("/historical/{symbol}")
async def get_historical_data(
    symbol: str,
    period: str = Query(default="1mo", description="Period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max"),
    interval: str = Query(default="1d", description="Interval: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo")
) -> Dict[str, Any]:
    """Get historical data for a symbol."""
    try:
        historical = await yfinance_service.get_historical_data(symbol, period, interval)
        return historical.model_dump()
    except Exception as e:
        logger.error(f"Failed to get historical data for {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve historical data for {symbol}: {str(e)}"
        )


@router.post("/quotes")
async def get_bulk_quotes(symbols: List[str]) -> Dict[str, Any]:
    """Get quotes for multiple symbols efficiently."""
    if not symbols:
        raise HTTPException(status_code=400, detail="No symbols provided")
    
    if len(symbols) > 50:  # Reasonable limit
        raise HTTPException(status_code=400, detail="Too many symbols (max 50)")
    
    try:
        results = await yfinance_service.bulk_quotes(symbols)
        
        # Convert to serializable format
        serialized_results = {}
        for symbol, quote in results.items():
            serialized_results[symbol] = quote.model_dump()
        
        return {
            "quotes": serialized_results,
            "count": len(serialized_results),
            "requested": len(symbols),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get bulk quotes: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve bulk quotes: {str(e)}"
        )


@router.get("/search")
async def search_symbols(
    q: str = Query(..., description="Search query (company name or ticker symbol)")
) -> Dict[str, Any]:
    """Search for symbols by company name or ticker."""
    if not q or len(q.strip()) < 1:
        raise HTTPException(status_code=400, detail="Search query is required")
    
    try:
        results = await yfinance_service.search_symbols(q.strip())
        return {
            "results": results,
            "count": len(results),
            "query": q.strip(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to search symbols for '{q}': {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to search symbols: {str(e)}"
        )


@router.get("/info/{symbol}")
async def get_symbol_info(symbol: str) -> Dict[str, Any]:
    """Get detailed information for a symbol."""
    try:
        quote = await yfinance_service.get_quote(symbol)
        return {
            "symbol": quote.symbol,
            "price": quote.price,
            "currency": quote.currency,
            "market_cap": quote.market_cap,
            "pe_ratio": quote.pe_ratio,
            "dividend_yield": quote.dividend_yield,
            "volume": quote.volume,
            "timestamp": quote.timestamp.isoformat(),
            "source": quote.source
        }
    except Exception as e:
        logger.error(f"Failed to get info for {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve info for {symbol}: {str(e)}"
        )