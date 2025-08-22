"""
IBKR Technical Factor API Routes
===============================

FastAPI routes for IBKR technical factor calculation and management.
Provides real-time technical factor computation from Interactive Brokers market data.
"""

import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, HTTPException, Query, Depends, Path
from pydantic import BaseModel, Field

from ibkr_technical_factors import ibkr_technical_engine, TechnicalFactorConfig, FactorCategory
from ib_market_data import MarketDataService  # Assuming this exists

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/ibkr/technical", tags=["IBKR Technical Factors"])


# Pydantic models for API requests/responses
class TechnicalFactorRequest(BaseModel):
    """Request model for technical factor calculation."""
    symbol: str = Field(..., description="Stock symbol (e.g., AAPL)")
    start_date: Optional[str] = Field(None, description="Start date for historical data (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date for calculation (YYYY-MM-DD)")
    include_microstructure: bool = Field(True, description="Include microstructure factors")
    include_trend: bool = Field(True, description="Include trend analysis factors")


class UniverseFactorRequest(BaseModel):
    """Request model for universe-wide factor calculation."""
    symbols: List[str] = Field(..., description="List of stock symbols")
    as_of_date: Optional[str] = Field(None, description="Calculation date (YYYY-MM-DD)")
    batch_size: int = Field(50, ge=1, le=500, description="Batch size for processing")


class TechnicalFactorResponse(BaseModel):
    """Response model for technical factors."""
    symbol: str
    calculation_date: str
    factor_count: int
    factors: Dict[str, float]
    data_points: int
    calculation_time_ms: float
    factor_categories: Dict[str, List[str]]


class UniverseFactorResponse(BaseModel):
    """Response model for universe factors."""
    universe_size: int
    successful_calculations: int
    failed_symbols: List[str]
    calculation_date: str
    total_calculation_time_ms: float
    factors_per_symbol: int
    universe_factors: List[Dict[str, Any]]


class HealthResponse(BaseModel):
    """Technical factor service health response."""
    status: str
    timestamp: str
    engine_healthy: bool
    supported_factors: int
    data_connection: bool


class FactorSummaryResponse(BaseModel):
    """Factor summary response."""
    total_factors: str
    categories: Dict[str, Dict[str, Any]]
    data_requirements: Dict[str, Any]


# Dependency functions
async def get_market_data_service() -> MarketDataService:
    """Get market data service dependency."""
    # This would be implemented to connect to your IBKR market data service
    # For now, return a mock service
    class MockMarketDataService:
        async def get_historical_data(self, symbol: str, start_date: date, end_date: date):
            # Mock implementation - replace with actual IBKR data retrieval
            import polars as pl
            import random
            from datetime import timedelta
            
            dates = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]
            base_price = 100.0
            
            data = []
            for d in dates:
                change = random.gauss(0, 0.02)
                base_price *= (1 + change)
                
                data.append({
                    'date': d,
                    'open': base_price,
                    'high': base_price * (1 + abs(random.gauss(0, 0.01))),
                    'low': base_price * (1 - abs(random.gauss(0, 0.01))),
                    'close': base_price,
                    'volume': random.randint(1000000, 5000000),
                    'bid': base_price * 0.995,
                    'ask': base_price * 1.005
                })
            
            return pl.DataFrame(data)
    
    return MockMarketDataService()


# API Routes

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check IBKR technical factor service health."""
    try:
        # Test engine functionality
        engine_healthy = True
        try:
            summary = await ibkr_technical_engine.get_factor_summary()
            factor_count = sum(cat['count'] for cat in summary['categories'].values())
        except Exception as e:
            logger.error(f"Engine health check failed: {e}")
            engine_healthy = False
            factor_count = 0
        
        return HealthResponse(
            status="healthy" if engine_healthy else "degraded",
            timestamp=datetime.now().isoformat(),
            engine_healthy=engine_healthy,
            supported_factors=factor_count,
            data_connection=True  # Would check actual IBKR connection
        )
    
    except Exception as e:
        logger.error(f"Technical factor health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unavailable: {e}")


@router.get("/factors/summary", response_model=FactorSummaryResponse)
async def get_factor_summary():
    """Get summary of available technical factors and their categories."""
    try:
        summary = await ibkr_technical_engine.get_factor_summary()
        return FactorSummaryResponse(**summary)
    
    except Exception as e:
        logger.error(f"Error getting factor summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get factor summary: {e}")


@router.post("/factors/calculate", response_model=TechnicalFactorResponse)
async def calculate_technical_factors(
    request: TechnicalFactorRequest,
    market_data_service: MarketDataService = Depends(get_market_data_service)
):
    """Calculate technical factors for a single symbol."""
    try:
        start_time = datetime.now()
        
        # Parse dates
        end_date = date.today()
        if request.end_date:
            try:
                end_date = datetime.strptime(request.end_date, "%Y-%m-%d").date()
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid end_date format. Use YYYY-MM-DD")
        
        start_date = end_date - timedelta(days=365)  # Default 1 year
        if request.start_date:
            try:
                start_date = datetime.strptime(request.start_date, "%Y-%m-%d").date()
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid start_date format. Use YYYY-MM-DD")
        
        # Validate symbol
        symbol = request.symbol.upper().strip()
        if not symbol:
            raise HTTPException(status_code=400, detail="Symbol cannot be empty")
        
        # Get market data
        try:
            market_data = await market_data_service.get_historical_data(symbol, start_date, end_date)
        except Exception as e:
            logger.error(f"Failed to get market data for {symbol}: {e}")
            raise HTTPException(status_code=404, detail=f"Market data not available for {symbol}")
        
        if len(market_data) == 0:
            raise HTTPException(status_code=404, detail=f"No market data found for {symbol}")
        
        # Configure factor engine
        config = TechnicalFactorConfig(
            lookback_periods={'short': 20, 'medium': 60, 'long': 252, 'ultra_short': 5},
            volatility_windows=[10, 20, 60],
            momentum_windows=[5, 20, 60, 120],
            microstructure_enabled=request.include_microstructure,
            trend_analysis_enabled=request.include_trend
        )
        
        # Update engine config
        ibkr_technical_engine.config = config
        
        # Calculate factors
        factors_df = await ibkr_technical_engine.calculate_technical_factors(
            symbol=symbol,
            market_data=market_data,
            as_of_date=end_date
        )
        
        calculation_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Extract factors from DataFrame
        factor_row = factors_df.to_dicts()[0]
        factors = {}
        factor_categories = {}
        
        for key, value in factor_row.items():
            if key not in ['symbol', 'date', 'calculation_timestamp', 'data_points']:
                if value is not None:
                    factors[key] = float(value)
                    
                    # Categorize factors
                    category = key.split('_')[0]
                    if category not in factor_categories:
                        factor_categories[category] = []
                    factor_categories[category].append(key)
        
        return TechnicalFactorResponse(
            symbol=symbol,
            calculation_date=end_date.isoformat(),
            factor_count=len(factors),
            factors=factors,
            data_points=factor_row.get('data_points', len(market_data)),
            calculation_time_ms=round(calculation_time, 2),
            factor_categories=factor_categories
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating technical factors: {e}")
        raise HTTPException(status_code=500, detail=f"Factor calculation failed: {e}")


@router.post("/factors/universe", response_model=UniverseFactorResponse)
async def calculate_universe_factors(
    request: UniverseFactorRequest,
    market_data_service: MarketDataService = Depends(get_market_data_service)
):
    """Calculate technical factors for multiple symbols in parallel."""
    try:
        start_time = datetime.now()
        
        # Parse calculation date
        calc_date = date.today()
        if request.as_of_date:
            try:
                calc_date = datetime.strptime(request.as_of_date, "%Y-%m-%d").date()
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid as_of_date format. Use YYYY-MM-DD")
        
        # Validate symbols
        symbols = [s.upper().strip() for s in request.symbols if s.strip()]
        if not symbols:
            raise HTTPException(status_code=400, detail="At least one symbol is required")
        
        if len(symbols) > 500:
            raise HTTPException(status_code=400, detail="Maximum 500 symbols allowed")
        
        # Get market data for all symbols
        universe_data = {}
        failed_symbols = []
        
        logger.info(f"Fetching market data for {len(symbols)} symbols")
        
        # Process in batches to avoid overwhelming the system
        batch_size = min(request.batch_size, 50)
        for i in range(0, len(symbols), batch_size):
            batch_symbols = symbols[i:i + batch_size]
            
            # Get data for batch
            batch_tasks = []
            for symbol in batch_symbols:
                start_date = calc_date - timedelta(days=365)  # 1 year lookback
                batch_tasks.append(
                    market_data_service.get_historical_data(symbol, start_date, calc_date)
                )
            
            try:
                import asyncio
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                for j, result in enumerate(batch_results):
                    symbol = batch_symbols[j]
                    if isinstance(result, Exception):
                        logger.warning(f"Failed to get data for {symbol}: {result}")
                        failed_symbols.append(symbol)
                        continue
                    
                    if len(result) == 0:
                        logger.warning(f"No data available for {symbol}")
                        failed_symbols.append(symbol)
                        continue
                    
                    universe_data[symbol] = result
                
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                failed_symbols.extend(batch_symbols)
        
        if not universe_data:
            raise HTTPException(status_code=404, detail="No valid market data found for any symbols")
        
        # Calculate factors for universe
        logger.info(f"Calculating factors for {len(universe_data)} symbols")
        
        universe_factors_df = await ibkr_technical_engine.calculate_universe_factors(
            universe_data=universe_data,
            as_of_date=calc_date
        )
        
        calculation_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Convert to response format
        universe_factors = []
        factors_per_symbol = 0
        
        for row in universe_factors_df.to_dicts():
            symbol_factors = {}
            factor_count = 0
            
            for key, value in row.items():
                if key not in ['symbol', 'date', 'calculation_timestamp', 'data_points']:
                    if value is not None:
                        symbol_factors[key] = float(value)
                        factor_count += 1
            
            universe_factors.append({
                'symbol': row['symbol'],
                'factor_count': factor_count,
                'factors': symbol_factors,
                'data_points': row.get('data_points', 0)
            })
            
            if factors_per_symbol == 0:
                factors_per_symbol = factor_count
        
        return UniverseFactorResponse(
            universe_size=len(symbols),
            successful_calculations=len(universe_factors),
            failed_symbols=failed_symbols,
            calculation_date=calc_date.isoformat(),
            total_calculation_time_ms=round(calculation_time, 2),
            factors_per_symbol=factors_per_symbol,
            universe_factors=universe_factors
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating universe factors: {e}")
        raise HTTPException(status_code=500, detail=f"Universe factor calculation failed: {e}")


@router.get("/factors/categories")
async def get_factor_categories():
    """Get list of available factor categories and their descriptions."""
    try:
        categories = {
            "momentum": {
                "description": "Price momentum and trend following factors",
                "typical_count": 5,
                "examples": ["momentum_short_term", "momentum_vwap_based", "momentum_rsi_signal"]
            },
            "volatility": {
                "description": "Volatility measurement and forecasting factors", 
                "typical_count": 4,
                "examples": ["volatility_realized", "volatility_garch_forecast", "volatility_regime_signal"]
            },
            "microstructure": {
                "description": "Market microstructure and trading pattern factors",
                "typical_count": 4,
                "examples": ["microstructure_effective_spread", "microstructure_trading_intensity"]
            },
            "market_quality": {
                "description": "Liquidity and market efficiency factors",
                "typical_count": 3,
                "examples": ["market_quality_liquidity_score", "market_quality_price_efficiency"]
            },
            "trend": {
                "description": "Trend analysis and persistence factors",
                "typical_count": 4,
                "examples": ["trend_strength", "trend_direction", "trend_persistence"]
            }
        }
        
        return {
            "categories": categories,
            "total_factor_types": sum(cat["typical_count"] for cat in categories.values()),
            "last_updated": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error getting factor categories: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get categories: {e}")


@router.delete("/cache/clear")
async def clear_factor_cache():
    """Clear the technical factor calculation cache."""
    try:
        # Clear engine cache
        cache_size = len(ibkr_technical_engine._factor_cache)
        ibkr_technical_engine._factor_cache.clear()
        
        return {
            "success": True,
            "message": "Factor cache cleared successfully",
            "cleared_entries": cache_size,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error clearing factor cache: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {e}")


# Startup event handler
@router.on_event("startup")
async def startup_event():
    """Initialize IBKR technical factor service on startup."""
    logger.info("Initializing IBKR Technical Factor service...")
    try:
        # Any initialization logic here
        logger.info("IBKR Technical Factor service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize IBKR Technical Factor service: {e}")


# Shutdown event handler  
@router.on_event("shutdown")
async def shutdown_event():
    """Clean up IBKR technical factor service on shutdown."""
    logger.info("Shutting down IBKR Technical Factor service...")
    try:
        # Clean up resources
        ibkr_technical_engine._factor_cache.clear()
        logger.info("IBKR Technical Factor service shut down successfully")
    except Exception as e:
        logger.error(f"Error during IBKR Technical Factor service shutdown: {e}")