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
from edgar_factor_integration import edgar_factor_integration
from fred_integration import fred_integration

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

# EDGAR Integration Endpoints - Story 1.1 Implementation

class FundamentalFactorsResponse(BaseModel):
    """Response model for EDGAR-based fundamental factors."""
    status: str
    calculation_date: str
    total_symbols: int
    successful_calculations: int
    factors_calculated: List[str]
    data: List[Dict[str, Any]]

class FactorEngineStatus(BaseModel):
    """Enhanced factor engine status including EDGAR integration."""
    status: str
    edgar_integration: str
    fred_integration: str
    ibkr_integration: str
    total_factors_available: int
    last_calculation_time: Optional[str]

@router.get("/status/enhanced", response_model=FactorEngineStatus)
async def get_enhanced_factor_engine_status():
    """
    Get enhanced status including all multi-source integrations.
    Shows current Epic 3.0 implementation progress.
    """
    try:
        # Check integration statuses
        edgar_status = "operational" if edgar_factor_integration.edgar_client else "not_initialized"
        fred_status = "operational" if fred_integration.session else "not_initialized"
        ibkr_status = "operational"  # Existing IBKR integration
        
        return FactorEngineStatus(
            status="operational",
            edgar_integration=edgar_status,
            fred_integration=fred_status,
            ibkr_integration=ibkr_status,
            total_factors_available=25,  # 3 Toraniko + ~22 EDGAR fundamental
            last_calculation_time=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error getting enhanced factor engine status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")

@router.post("/factors/edgar/fundamental", response_model=FundamentalFactorsResponse)
async def calculate_edgar_fundamental_factors(
    symbols: List[str],
    as_of_date: Optional[str] = None
):
    """
    Calculate fundamental factors using SEC EDGAR data.
    
    Story 1.1: Complete EDGAR-Factor Integration Implementation
    
    Calculates 20+ fundamental factors including:
    - Quality: ROE, ROA, debt-to-equity, interest coverage, gross margin
    - Growth: Revenue growth, EPS growth, margin expansion (historical analysis)
    - Value: P/B, P/S, P/CF ratios using real SEC filing data
    - Health: Altman Z-score, current ratio, quick ratio, financial stability
    
    This endpoint bridges the existing EDGAR connector with the Toraniko factor engine.
    """
    try:
        # Parse calculation date
        calculation_date = date.today()
        if as_of_date:
            try:
                calculation_date = datetime.strptime(as_of_date, "%Y-%m-%d").date()
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        
        # Initialize EDGAR integration if needed
        if not edgar_factor_integration.edgar_client:
            await edgar_factor_integration.initialize()
        
        # Calculate fundamental factors using SEC data
        factor_df = await edgar_factor_integration.calculate_fundamental_factors(
            symbols=symbols,
            as_of_date=calculation_date
        )
        
        # Convert to API response format
        data = factor_df.to_dicts()
        successful_calculations = len([d for d in data if d.get('cik') is not None])
        
        # Extract factor column names (exclude metadata columns)
        factor_columns = [col for col in factor_df.columns 
                         if col not in ['symbol', 'date', 'cik'] and not col.endswith('_rank')]
        
        return FundamentalFactorsResponse(
            status="success",
            calculation_date=calculation_date.isoformat(),
            total_symbols=len(symbols),
            successful_calculations=successful_calculations,
            factors_calculated=factor_columns,
            data=data
        )
        
    except Exception as e:
        logger.error(f"Error calculating EDGAR fundamental factors: {str(e)}")
        raise HTTPException(status_code=500, detail=f"EDGAR factor calculation failed: {str(e)}")

@router.post("/test/edgar-integration")
async def test_edgar_factor_integration():
    """
    Test endpoint for EDGAR-Factor integration validation.
    
    Story 1.1: Integration testing endpoint for development validation.
    """
    try:
        # Initialize EDGAR integration
        if not edgar_factor_integration.edgar_client:
            await edgar_factor_integration.initialize()
        
        # Test with major stocks
        test_symbols = ['AAPL', 'MSFT', 'GOOGL']
        
        factor_df = await edgar_factor_integration.calculate_fundamental_factors(test_symbols)
        
        return {
            "status": "success",
            "test_symbols": test_symbols,
            "factors_calculated": len([col for col in factor_df.columns if col not in ['symbol', 'date', 'cik']]),
            "data_points": len(factor_df),
            "sample_data": factor_df.head(2).to_dicts() if len(factor_df) > 0 else [],
            "integration_health": "operational",
            "edgar_api_status": "connected"
        }
        
    except Exception as e:
        logger.error(f"EDGAR integration test failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Integration test failed: {str(e)}")

@router.get("/factors/enhanced/list")
async def list_enhanced_available_factors():
    """
    List all available factors including EDGAR integration.
    
    Shows current factor universe with implementation status.
    """
    try:
        factors = {
            "toraniko_factors": {
                "momentum": ["momentum_252d", "momentum_126d", "momentum_63d"],
                "value": ["val_score"],
                "size": ["sze_score"],
                "status": "operational"
            },
            "edgar_fundamental_factors": {
                "quality": ["roe", "roa", "debt_to_equity", "interest_coverage", "gross_margin"],
                "growth": ["revenue_growth_1y", "revenue_growth_3y", "eps_growth_1y", "eps_growth_3y", "margin_expansion"],
                "value": ["book_price", "sales_price", "cf_price"],
                "health": ["current_ratio", "quick_ratio", "altman_z_score"],
                "status": "operational"
            },
            "fred_macro_factors": {
                "economic": ["unemployment_rate", "inflation_rate_yoy", "unemployment_trend_6m"],
                "monetary": ["fed_funds_level", "treasury_10y_level", "treasury_2y_level", "yield_curve_slope", "yield_curve_level", "yield_curve_curvature"],
                "market_regime": ["vix_level", "volatility_regime"],
                "status": "operational"
            },
            "ibkr_technical_factors": {
                "momentum": ["rsi", "macd", "price_momentum"],
                "volatility": ["realized_volatility", "garch_volatility"],
                "microstructure": ["bid_ask_spread", "order_flow"],
                "status": "pending_phase_1"
            },
            "cross_source_factors": {
                "alternative": ["macro_adjusted_pe", "earnings_surprise_momentum", "regime_aware_value"],
                "status": "pending_phase_2"
            }
        }
        
        # Calculate totals
        total_operational = len(factors["toraniko_factors"]["momentum"]) + len(factors["toraniko_factors"]["value"]) + len(factors["toraniko_factors"]["size"])
        total_operational += sum(len(factors["edgar_fundamental_factors"][category]) for category in factors["edgar_fundamental_factors"] if category != "status")
        
        total_planned = sum(
            sum(len(factors[category][subcategory]) for subcategory in factors[category] if subcategory != "status")
            for category in factors if category not in ["toraniko_factors", "edgar_fundamental_factors"]
        )
        
        return {
            "status": "success",
            "factors_by_source": factors,
            "totals": {
                "operational_factors": total_operational,
                "planned_factors": total_planned,
                "total_target": total_operational + total_planned
            },
            "epic_progress": {
                "phase_1_progress": "edgar_integration_complete",
                "current_focus": "fred_integration_development",
                "next_milestone": "cross_source_factors"
            }
        }
        
    except Exception as e:
        logger.error(f"Error listing enhanced factors: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list factors: {str(e)}")

# FRED Integration Endpoints - Story 1.2 Implementation

class MacroFactorsResponse(BaseModel):
    """Response model for FRED-based macro factors."""
    status: str
    calculation_date: str
    factors_calculated: List[str]
    data: Dict[str, Any]
    fred_api_status: str

@router.post("/factors/fred/macro", response_model=MacroFactorsResponse)
async def calculate_fred_macro_factors(
    as_of_date: Optional[str] = None,
    lookback_days: int = 365
):
    """
    Calculate macro-economic factors using FRED data.
    
    Story 1.2: Build FRED API Integration Foundation Implementation
    
    Calculates 15+ macro factors including:
    - Economic: Unemployment rate, inflation trends, economic growth indicators
    - Monetary: Fed funds level, yield curve factors (level, slope, curvature)
    - Market Regime: VIX level, volatility regime classification
    - Rate Changes: 30-day interest rate changes across the curve
    
    This endpoint provides the missing FRED integration for macro factor analysis.
    """
    try:
        # Parse calculation date
        calculation_date = date.today()
        if as_of_date:
            try:
                calculation_date = datetime.strptime(as_of_date, "%Y-%m-%d").date()
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        
        # Initialize FRED integration if needed
        if not fred_integration.session:
            await fred_integration.initialize()
        
        # Calculate macro factors using FRED data
        factor_df = await fred_integration.calculate_macro_factors(
            as_of_date=calculation_date,
            lookback_days=lookback_days
        )
        
        # Convert to API response format
        data = factor_df.to_dicts()[0] if len(factor_df) > 0 else {}
        
        # Extract factor names (exclude metadata columns)
        factor_columns = [col for col in factor_df.columns 
                         if col not in ['date', 'calculation_timestamp']]
        
        return MacroFactorsResponse(
            status="success",
            calculation_date=calculation_date.isoformat(),
            factors_calculated=factor_columns,
            data=data,
            fred_api_status="connected"
        )
        
    except Exception as e:
        logger.error(f"Error calculating FRED macro factors: {str(e)}")
        raise HTTPException(status_code=500, detail=f"FRED factor calculation failed: {str(e)}")

@router.get("/fred/series/{series_id}")
async def get_fred_series_data(
    series_id: str,
    limit: int = 100
):
    """
    Get specific FRED economic time series data.
    
    Useful for exploring individual economic indicators and their historical values.
    """
    try:
        # Initialize FRED integration if needed
        if not fred_integration.session:
            await fred_integration.initialize()
        
        # Get series data
        data_df = await fred_integration.get_series_data(
            series_id=series_id,
            limit=limit
        )
        
        # Convert to response format
        data = data_df.to_dicts()
        
        return {
            "status": "success",
            "series_id": series_id,
            "observations": len(data),
            "data": data
        }
        
    except Exception as e:
        logger.error(f"Error retrieving FRED series {series_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"FRED series retrieval failed: {str(e)}")

@router.get("/fred/key-series")
async def list_fred_key_series():
    """
    List key FRED economic series used for factor calculations.
    
    Shows the core economic indicators integrated into the factor model.
    """
    try:
        key_series = []
        for series_id, series_info in fred_integration.KEY_SERIES.items():
            key_series.append({
                "series_id": series_id,
                "title": series_info.title,
                "frequency": series_info.frequency.value,
                "units": series_info.units,
                "category": series_info.category,
                "seasonal_adjustment": series_info.seasonal_adjustment
            })
        
        return {
            "status": "success",
            "total_series": len(key_series),
            "key_series": key_series,
            "categories": list(set(series["category"] for series in key_series))
        }
        
    except Exception as e:
        logger.error(f"Error listing FRED key series: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list FRED series: {str(e)}")

@router.post("/test/fred-integration")
async def test_fred_integration_endpoint():
    """
    Test endpoint for FRED integration validation.
    
    Story 1.2: Integration testing endpoint for development validation.
    """
    try:
        # Initialize FRED integration
        if not fred_integration.session:
            await fred_integration.initialize()
        
        # Test series data retrieval
        test_series = "DFF"  # Federal Funds Rate - reliable test series
        test_data = await fred_integration.get_series_data(test_series, limit=10)
        
        # Test macro factor calculation
        macro_factors = await fred_integration.calculate_macro_factors()
        
        return {
            "status": "success", 
            "test_series": test_series,
            "test_data_points": len(test_data),
            "macro_factors_calculated": len([col for col in macro_factors.columns if col not in ['date', 'calculation_timestamp']]),
            "sample_factors": macro_factors.to_dicts()[0] if len(macro_factors) > 0 else {},
            "integration_health": "operational",
            "fred_api_status": "connected"
        }
        
    except Exception as e:
        logger.error(f"FRED integration test failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Integration test failed: {str(e)}")

# Initialize EDGAR integration on startup
@router.on_event("startup")
async def initialize_edgar_integration():
    """Initialize EDGAR integration services on startup."""
    try:
        logger.info("Initializing enhanced Factor Engine with EDGAR integration...")
        
        # Initialize existing factor engine service
        await factor_engine_service.initialize()
        
        # Initialize EDGAR integration (Story 1.1)
        await edgar_factor_integration.initialize()
        
        logger.info("Enhanced Factor Engine with EDGAR integration initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize enhanced Factor Engine: {str(e)}")
        # Don't raise exception to allow other services to continue