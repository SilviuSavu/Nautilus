"""
Performance Routes - DISABLED DUE TO CORE RULE #4 VIOLATION

This entire file was heavily using mock data throughout multiple endpoints, 
which violates CORE RULE #4: NO MOCK DATA POLICY

All performance data must come from real NautilusTrader Docker containers.
"""

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/v1/performance", tags=["performance"])

@router.get("/aggregate")
async def get_aggregate_performance():
    raise HTTPException(
        status_code=501,
        detail="Performance routes disabled - contained extensive mock data violating CORE RULE #4. Real implementation required."
    )

@router.get("/history")
async def get_performance_history():
    raise HTTPException(
        status_code=501,
        detail="Performance routes disabled - contained extensive mock data violating CORE RULE #4. Real implementation required."
    )

@router.post("/compare")
async def compare_strategies():
    raise HTTPException(
        status_code=501,
        detail="Performance routes disabled - contained extensive mock data violating CORE RULE #4. Real implementation required."
    )

@router.get("/execution/metrics")
async def get_execution_metrics():
    raise HTTPException(
        status_code=501,
        detail="Performance routes disabled - contained extensive mock data violating CORE RULE #4. Real implementation required."
    )

@router.get("/execution/trades")
async def get_execution_trades():
    raise HTTPException(
        status_code=501,
        detail="Performance routes disabled - contained extensive mock data violating CORE RULE #4. Real implementation required."
    )