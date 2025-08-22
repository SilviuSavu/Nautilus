"""
Simple YFinance Routes for Status Endpoint
==========================================

Basic YFinance service to handle status requests and prevent 404 errors.
Note: This is a minimal implementation since the main system uses IBKR data.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/yfinance", tags=["yfinance"])


@router.get("/status")
async def get_yfinance_status() -> Dict[str, Any]:
    """Get YFinance service status."""
    return {
        "status": "available",
        "service": "yfinance",
        "message": "YFinance service is available but not actively used (IBKR is primary data source)",
        "data_source": "secondary",
        "primary_source": "Interactive Brokers"
    }


@router.get("/health")
async def get_yfinance_health() -> Dict[str, Any]:
    """Get YFinance service health check."""
    try:
        # Simple health check
        return {
            "status": "healthy",
            "service": "yfinance",
            "timestamp": "2025-08-22T07:35:00Z",
            "note": "Minimal YFinance service - IBKR is primary data source"
        }
    except Exception as e:
        logger.error(f"YFinance health check failed: {e}")
        raise HTTPException(status_code=503, detail="YFinance service temporarily unavailable")