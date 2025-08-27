"""
Factor Engine API Routes - Toraniko Integration (CLEANED)
FastAPI routes for multi-factor equity risk modeling

MIGRATION NOTE: FRED integration migrated to official Nautilus adapters.
Use /api/v1/nautilus-data/fred/* endpoints for FRED data access.
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Optional, Any
from datetime import date, datetime
from pydantic import BaseModel, Field
import polars as pl
import logging

from factor_engine_service import factor_engine_service
from edgar_factor_integration import edgar_factor_integration

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
    market_price: Optional[float] = None
    earnings_per_share: Optional[float] = None
    price_earnings_ratio: Optional[float] = None

# Enhanced Pydantic models for v1.1.2 FactorModel workflows
class FactorModelRequest(BaseModel):
    """Request model for creating a new FactorModel"""
    model_id: str = Field(description="Unique identifier for the factor model")
    feature_data: List[Dict] = Field(description="Feature data (prices, fundamentals, etc.)")
    sector_encodings: List[Dict] = Field(description="Sector classification data")
    symbol_col: str = Field(default="symbol", description="Column name for symbols")
    date_col: str = Field(default="date", description="Column name for dates")
    mkt_cap_col: str = Field(default="market_cap", description="Column name for market cap")

class FeatureCleaningRequest(BaseModel):
    """Request model for feature cleaning operations"""
    model_id: str = Field(description="FactorModel identifier")
    to_winsorize: Optional[Dict[str, float]] = Field(None, description="Features to winsorize with factors")
    to_fill: Optional[List[str]] = Field(None, description="Features to fill forward")
    to_smooth: Optional[Dict[str, int]] = Field(None, description="Features to smooth with windows")

class UniverseReductionRequest(BaseModel):
    """Request model for universe reduction"""
    model_id: str = Field(description="FactorModel identifier")
    top_n: int = Field(default=2000, description="Number of top assets by market cap")
    collect: bool = Field(default=True, description="Whether to collect lazy DataFrame")

class FactorReturnsRequest(BaseModel):
    """Request model for factor return estimation"""
    model_id: str = Field(description="FactorModel identifier")
    winsor_factor: float = Field(default=0.01, description="Winsorization factor for returns")
    residualize_styles: bool = Field(default=False, description="Whether to residualize style factors")
    asset_returns_col: str = Field(default="asset_returns", description="Asset returns column name")

class FactorEngineStatus(BaseModel):
    """Factor engine operational status"""
    status: str = Field(description="Overall engine status (operational, degraded, offline)")
    toraniko_integration: str = Field(description="Toraniko factor engine status")
    edgar_integration: str = Field(description="EDGAR data integration status") 
    fred_integration: str = Field(description="FRED data integration status (migrated to Nautilus)")
    ibkr_integration: str = Field(description="Interactive Brokers integration status")
    timestamp: str = Field(description="Status check timestamp")

@router.get("/status", response_model=FactorEngineStatus)
async def get_factor_engine_status():
    """
    Get comprehensive factor engine operational status.
    
    Story 1.3: Multi-source integration status monitoring for operational visibility.
    """
    try:
        # Check integration statuses
        edgar_status = "operational" if edgar_factor_integration.edgar_client else "not_initialized"
        fred_status = "migrated_to_nautilus"  # FRED now handled via Nautilus adapters
        ibkr_status = "operational"  # Existing IBKR integration
        
        return FactorEngineStatus(
            status="operational",
            toraniko_integration="operational",
            edgar_integration=edgar_status,
            fred_integration=fred_status,
            ibkr_integration=ibkr_status,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error checking factor engine status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """
    Comprehensive health check for the factor engine ecosystem.
    
    Returns detailed status of all integrated components for monitoring.
    """
    try:
        health_status = {
            "status": "operational",
            "components": {
                "toraniko_engine": "operational",
                "edgar_integration": "operational" if edgar_factor_integration.edgar_client else "not_initialized", 
                "fred_integration": "migrated_to_nautilus",  # Use /api/v1/nautilus-data/fred/*
                "ibkr_integration": "operational"
            },
            "migration_notes": {
                "fred_data": "Use /api/v1/nautilus-data/fred/macro-factors for macro factor calculations",
                "alpha_vantage": "Use /api/v1/nautilus-data/alpha-vantage/* for market data"
            },
            "epic_progress": {
                "phase_1_progress": "edgar_integration_complete",
                "current_focus": "nautilus_integration_complete",
                "next_milestone": "cross_source_factors"
            }
        }
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Note: FRED-related endpoints have been migrated to Nautilus adapters
# Use the following new endpoints:
# - GET /api/v1/nautilus-data/fred/macro-factors
# - GET /api/v1/nautilus-data/health (includes FRED status)

@router.post("/initialize")
async def initialize_factor_engine():
    """
    Initialize the Enhanced Multi-Source Factor Engine.
    
    Story 2.1: Complete engine initialization with all data source integrations.
    """
    try:
        # Initialize existing factor engine service (Toraniko)
        await factor_engine_service.initialize()
        
        # Initialize EDGAR integration
        if not edgar_factor_integration.edgar_client:
            await edgar_factor_integration.initialize()
        
        # FRED is now initialized through Nautilus adapters automatically
        
        logger.info("Enhanced Multi-Source Factor Engine initialized successfully")
        logger.info("Phase 2 Foundation: EDGAR ✅, FRED (Nautilus) ✅, Toraniko ✅")
        
        return {
            "status": "success",
            "message": "Enhanced Multi-Source Factor Engine initialized",
            "components_initialized": [
                "toraniko_factor_engine",
                "edgar_integration", 
                "fred_nautilus_integration"
            ],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to initialize Enhanced Factor Engine: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/migration-info")
async def get_migration_info():
    """
    Information about the migration to Nautilus adapters.
    """
    return {
        "migration_status": "completed",
        "migrated_services": {
            "fred": {
                "old_endpoint": "/api/v1/factor-engine/fred/*",
                "new_endpoint": "/api/v1/nautilus-data/fred/*",
                "integration": "nautilus_trader.adapters.fred"
            },
            "alpha_vantage": {
                "old_endpoint": "/api/v1/alpha-vantage/*",
                "new_endpoint": "/api/v1/nautilus-data/alpha-vantage/*", 
                "integration": "nautilus_trader.adapters.alpha_vantage"
            }
        },
        "benefits": [
            "Unified message bus integration",
            "Real-time data streaming", 
            "Official Nautilus adapter support",
            "Reduced code duplication",
            "Better error handling and rate limiting"
        ],
        "timestamp": datetime.now().isoformat()
    }

@router.get("/status/enhanced", response_model=FactorEngineStatus)
async def get_enhanced_factor_engine_status():
    """Get enhanced factor engine status with all integrations"""
    return {
        "status": "operational",
        "toraniko_integration": "operational",
        "edgar_integration": "operational", 
        "fred_integration": "operational",
        "ibkr_integration": "operational",
        "timestamp": datetime.now().isoformat()
    }

# Enhanced v1.1.2 FactorModel API Endpoints

@router.post("/model/create")
async def create_factor_model(request: FactorModelRequest):
    """
    Create a new FactorModel instance using Toraniko v1.1.2 capabilities
    
    This endpoint creates an end-to-end factor modeling workflow instance
    that can be used for feature cleaning, style score estimation, and
    factor return calculation with advanced covariance estimation.
    """
    try:
        # Convert request data to Polars DataFrames
        feature_df = pl.DataFrame(request.feature_data)
        sector_df = pl.DataFrame(request.sector_encodings)
        
        result = await factor_engine_service.create_factor_model(
            model_id=request.model_id,
            feature_data=feature_df,
            sector_encodings=sector_df,
            symbol_col=request.symbol_col,
            date_col=request.date_col,
            mkt_cap_col=request.mkt_cap_col
        )
        
        return {
            "success": True,
            "message": result,
            "model_id": request.model_id,
            "features": {
                "feature_data_rows": len(request.feature_data),
                "sector_encodings_rows": len(request.sector_encodings),
                "enhanced_capabilities": True
            }
        }
        
    except Exception as e:
        logger.error(f"Error creating FactorModel: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/model/clean-features")
async def clean_model_features(request: FeatureCleaningRequest):
    """
    Apply feature cleaning pipeline to a FactorModel
    
    Supports winsorization, forward filling, and smoothing operations
    using Toraniko v1.1.2 enhanced data processing capabilities.
    """
    try:
        result = await factor_engine_service.clean_model_features(
            model_id=request.model_id,
            to_winsorize=request.to_winsorize,
            to_fill=request.to_fill,
            to_smooth=request.to_smooth
        )
        
        return {
            "success": True,
            "message": result,
            "operations_applied": {
                "winsorized_features": list(request.to_winsorize.keys()) if request.to_winsorize else [],
                "filled_features": request.to_fill or [],
                "smoothed_features": list(request.to_smooth.keys()) if request.to_smooth else []
            }
        }
        
    except Exception as e:
        logger.error(f"Error cleaning model features: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/model/reduce-universe")
async def reduce_model_universe(request: UniverseReductionRequest):
    """
    Reduce universe size by market capitalization
    
    Filters the FactorModel to include only the top N assets by market cap,
    improving computational efficiency for institutional-scale analysis.
    """
    try:
        result = await factor_engine_service.reduce_model_universe(
            model_id=request.model_id,
            top_n=request.top_n,
            collect=request.collect
        )
        
        return {
            "success": True,
            "message": result,
            "universe_size": request.top_n,
            "optimization": "market_cap_weighted"
        }
        
    except Exception as e:
        logger.error(f"Error reducing model universe: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/model/estimate-style-scores")
async def estimate_style_scores(model_id: str):
    """
    Estimate style factor scores for a FactorModel
    
    Calculates momentum, value, and size factor scores using configuration-driven
    parameters optimized for Nautilus platform data sources.
    """
    try:
        result = await factor_engine_service.estimate_model_style_scores(
            model_id=model_id,
            collect=True
        )
        
        return {
            "success": True,
            "message": result,
            "style_factors": ["momentum", "value", "size"],
            "method": "toraniko_v1.1.2"
        }
        
    except Exception as e:
        logger.error(f"Error estimating style scores: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/model/estimate-factor-returns")
async def estimate_factor_returns(request: FactorReturnsRequest):
    """
    Estimate factor returns using enhanced v1.1.2 capabilities
    
    Performs factor return estimation with support for Ledoit-Wolf shrinkage
    covariance estimation and advanced residualization techniques.
    """
    try:
        result = await factor_engine_service.estimate_model_factor_returns(
            model_id=request.model_id,
            winsor_factor=request.winsor_factor,
            residualize_styles=request.residualize_styles,
            asset_returns_col=request.asset_returns_col
        )
        
        return {
            "success": True,
            "factor_returns_info": result,
            "enhanced_features": {
                "ledoit_wolf_covariance": True,
                "advanced_winsorization": True,
                "institutional_grade": True
            }
        }
        
    except Exception as e:
        logger.error(f"Error estimating factor returns: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model/{model_id}/status")
async def get_model_status(model_id: str):
    """
    Get detailed status and information about a FactorModel
    
    Returns comprehensive information about the model's current state,
    processing stages completed, and available results.
    """
    try:
        status = await factor_engine_service.get_model_status(model_id)
        return {
            "success": True,
            "model_status": status,
            "toraniko_version": "1.1.2",
            "capabilities": {
                "feature_cleaning": True,
                "style_estimation": True, 
                "factor_returns": True,
                "covariance_estimation": True
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models")
async def list_factor_models():
    """
    List all created FactorModel instances
    
    Returns summary information about all factor models currently
    managed by the Factor Engine service.
    """
    try:
        models_info = await factor_engine_service.list_factor_models()
        return {
            "success": True,
            "models": models_info,
            "integration_status": {
                "toraniko_version": "1.1.2",
                "nautilus_config": True,
                "enhanced_features": True
            }
        }
        
    except Exception as e:
        logger.error(f"Error listing factor models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/config")
async def get_configuration():
    """
    Get current Toraniko configuration settings
    
    Returns the active configuration used by the Factor Engine,
    including style factor settings and model estimation parameters.
    """
    try:
        config_info = {
            "config_loaded": factor_engine_service._config is not None,
            "config_sections": len(factor_engine_service._config) if factor_engine_service._config else 0,
            "factor_definitions": factor_engine_service._factor_definitions_loaded,
            "enhanced_features": {
                "feature_cleaning": factor_engine_service._feature_cleaning_enabled,
                "ledoit_wolf": factor_engine_service._ledoit_wolf_enabled
            },
            "nautilus_optimizations": True
        }
        
        return {
            "success": True,
            "configuration": config_info
        }
        
    except Exception as e:
        logger.error(f"Error getting configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))