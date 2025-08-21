"""
Factor Engine API Routes - Toraniko Integration
FastAPI routes for multi-factor equity risk modeling
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Optional, Any
from datetime import date, datetime
from pydantic import BaseModel, Field
import polars as pl
import logging

from factor_engine_service import factor_engine_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/factor-engine", tags=["Factor Engine"])

# Pydantic models for request/response validation
class AssetReturnsData(BaseModel):
    """Asset returns data for factor calculations"""
    symbol: str
    date: date
    asset_returns: float

class FundamentalData(BaseModel):
    """Fundamental data for value factor calculation"""
    symbol: str
    date: date
    book_price: Optional[float] = None
    sales_price: Optional[float] = None
    cf_price: Optional[float] = None
    market_cap: Optional[float] = None

class FactorRequest(BaseModel):
    """Request model for factor calculations"""
    data: List[Dict[str, Any]]
    trailing_days: Optional[int] = Field(default=252, description="Trailing days for momentum calculation")
    winsor_factor: Optional[float] = Field(default=0.01, description="Winsorization factor")

class PortfolioWeights(BaseModel):
    """Portfolio weights for factor exposure calculation"""
    weights: Dict[str, float] = Field(description="Symbol to weight mapping")
    date: date = Field(description="Date for exposure calculation")

class RiskModelRequest(BaseModel):
    """Request model for risk model creation"""
    universe_symbols: List[str]
    start_date: date
    end_date: date
    universe_size: Optional[int] = Field(default=3000, description="Maximum universe size")

class FactorScores(BaseModel):
    """Response model for factor scores"""
    symbol: str
    date: date
    score: float
    factor_type: str

class FactorExposures(BaseModel):
    """Response model for factor exposures"""
    market_exposure: float
    sector_exposures: Dict[str, float]
    style_exposures: Dict[str, float]
    specific_risk: float
    total_risk: float

@router.get("/health")
async def health_check():
    """Health check endpoint for factor engine service"""
    return {
        "status": "healthy",
        "service": "Toraniko Factor Engine",
        "timestamp": datetime.now().isoformat()
    }

@router.post("/factors/momentum", response_model=List[FactorScores])
async def calculate_momentum_factor(request: FactorRequest):
    """
    Calculate momentum factor scores for given asset returns
    
    The momentum factor measures the tendency of assets that have performed well
    (poorly) in the recent past to continue to perform well (poorly) in the future.
    """
    try:
        # Convert request data to Polars DataFrame
        df = pl.DataFrame(request.data)
        
        # Ensure required columns exist
        required_cols = ['symbol', 'date', 'asset_returns']
        if not all(col in df.columns for col in required_cols):
            raise HTTPException(
                status_code=400,
                detail=f"Data must contain columns: {required_cols}"
            )
        
        # Calculate momentum factor
        momentum_scores = await factor_engine_service.calculate_momentum_factor(
            df,
            trailing_days=request.trailing_days,
            winsor_factor=request.winsor_factor
        )
        
        # Convert to response format
        result = []
        for row in momentum_scores.to_dicts():
            result.append(FactorScores(
                symbol=row['symbol'],
                date=row['date'],
                score=row.get('mom_score', 0.0),
                factor_type='momentum'
            ))
        
        logger.info(f"Calculated momentum factors for {len(result)} observations")
        return result
        
    except Exception as e:
        logger.error(f"Error in momentum factor calculation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/factors/value", response_model=List[FactorScores])
async def calculate_value_factor(request: FactorRequest):
    """
    Calculate value factor scores from fundamental data
    
    The value factor captures the tendency of "cheap" stocks (high book-to-market,
    earnings-to-price, cash flow-to-price) to outperform "expensive" stocks.
    """
    try:
        # Convert request data to Polars DataFrame
        df = pl.DataFrame(request.data)
        
        # Ensure required columns exist
        required_cols = ['date', 'symbol', 'book_price', 'sales_price', 'cf_price']
        if not all(col in df.columns for col in required_cols):
            raise HTTPException(
                status_code=400,
                detail=f"Data must contain columns: {required_cols}"
            )
        
        # Calculate value factor
        value_scores = await factor_engine_service.calculate_value_factor(df)
        
        # Convert to response format
        result = []
        for row in value_scores.to_dicts():
            result.append(FactorScores(
                symbol=row['symbol'],
                date=row['date'],
                score=row.get('val_score', 0.0),
                factor_type='value'
            ))
        
        logger.info(f"Calculated value factors for {len(result)} observations")
        return result
        
    except Exception as e:
        logger.error(f"Error in value factor calculation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/factors/size", response_model=List[FactorScores])
async def calculate_size_factor(request: FactorRequest):
    """
    Calculate size factor scores from market capitalization data
    
    The size factor captures the tendency of small-cap stocks to outperform
    large-cap stocks over certain periods.
    """
    try:
        # Convert request data to Polars DataFrame
        df = pl.DataFrame(request.data)
        
        # Calculate size factor
        size_scores = await factor_engine_service.calculate_size_factor(df)
        
        # Convert to response format
        result = []
        for row in size_scores.to_dicts():
            result.append(FactorScores(
                symbol=row['symbol'],
                date=row['date'],
                score=row.get('sze_score', 0.0),
                factor_type='size'
            ))
        
        logger.info(f"Calculated size factors for {len(result)} observations")
        return result
        
    except Exception as e:
        logger.error(f"Error in size factor calculation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/risk-model/create")
async def create_risk_model(request: RiskModelRequest):
    """
    Create a complete multi-factor risk model for the specified universe
    
    This endpoint creates a comprehensive risk model including factor returns,
    covariance matrices, and specific risk estimates.
    """
    try:
        risk_model = await factor_engine_service.create_risk_model(
            universe_symbols=request.universe_symbols,
            start_date=request.start_date,
            end_date=request.end_date,
            universe_size=request.universe_size
        )
        
        logger.info(f"Created risk model for {len(request.universe_symbols)} symbols")
        return risk_model
        
    except Exception as e:
        logger.error(f"Error creating risk model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/portfolio/exposures", response_model=FactorExposures)
async def calculate_portfolio_exposures(request: PortfolioWeights):
    """
    Calculate factor exposures for a given portfolio
    
    Returns the portfolio's exposure to market, sector, and style factors,
    along with specific and total risk measures.
    """
    try:
        exposures = await factor_engine_service.get_factor_exposures(
            portfolio_weights=request.weights,
            date_requested=request.date
        )
        
        logger.info(f"Calculated exposures for portfolio with {len(request.weights)} positions")
        return FactorExposures(**exposures)
        
    except Exception as e:
        logger.error(f"Error calculating portfolio exposures: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/factors/list")
async def list_available_factors():
    """
    List all available factors in the risk model
    
    Returns information about market, sector, and style factors available
    for analysis and portfolio optimization.
    """
    return {
        "market_factors": ["market"],
        "sector_factors": [
            "Basic Materials", "Communication Services", "Consumer Cyclical",
            "Consumer Defensive", "Energy", "Financial Services", "Healthcare",
            "Industrials", "Real Estate", "Technology", "Utilities"
        ],
        "style_factors": ["momentum", "value", "size"],
        "description": {
            "momentum": "Tendency of well-performing assets to continue performing well",
            "value": "Tendency of 'cheap' stocks to outperform 'expensive' stocks",
            "size": "Tendency of small-cap stocks to outperform large-cap stocks"
        }
    }

@router.get("/info")
async def get_engine_info():
    """Get information about the Toraniko factor engine"""
    return {
        "name": "Toraniko Factor Engine",
        "version": "1.1.0",
        "description": "Multi-factor equity risk model for quantitative trading",
        "capabilities": [
            "Factor score calculation (momentum, value, size)",
            "Factor return estimation",
            "Risk model construction",
            "Portfolio factor exposure analysis",
            "Covariance matrix estimation"
        ],
        "compatible_with": "NautilusTrader, Interactive Brokers",
        "data_requirements": [
            "Asset returns (daily)",
            "Market capitalization",
            "Fundamental data (book value, sales, cash flow)",
            "Sector classifications"
        ],
        "performance": "10+ years of daily data processed in under 1 minute"
    }