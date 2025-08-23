"""
Multi-DataSource Coordination Routes
===================================

FastAPI routes for coordinating multiple data sources simultaneously.
Provides intelligent routing, fallback mechanisms, and unified data access.
"""

import os
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from enum import Enum

from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks
from pydantic import BaseModel, Field

# Import data services
from fred_integration import fred_integration
from alpha_vantage.service import alpha_vantage_service

# Try to import EDGAR functions, with fallbacks if not available
try:
    from edgar_routes import search_companies, get_facts_by_ticker
    EDGAR_AVAILABLE = True
except ImportError:
    EDGAR_AVAILABLE = False
    # Define fallback functions
    async def search_companies(keywords: str):
        raise HTTPException(status_code=503, detail="EDGAR service not available")
    
    async def get_facts_by_ticker(ticker: str):
        raise HTTPException(status_code=503, detail="EDGAR service not available")

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/multi-datasource", tags=["Multi-DataSource"])

class DataSourceType(str, Enum):
    IBKR = "ibkr"
    ALPHA_VANTAGE = "alpha_vantage"
    FRED = "fred"
    EDGAR = "edgar"
    YFINANCE = "yfinance"
    BACKFILL = "backfill"

class RequestPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class DataSourceConfig(BaseModel):
    """Configuration for a data source."""
    source_id: DataSourceType
    enabled: bool = True
    priority: int = Field(ge=1, le=10, description="Priority level (1=highest)")
    rate_limit_per_minute: Optional[int] = None
    fallback_enabled: bool = True
    timeout_seconds: int = 30

class MultiDataSourceRequest(BaseModel):
    """Request for multi-source data."""
    symbol: Optional[str] = None
    data_type: str = Field(description="Type of data: quote, historical, search, fundamentals, economic")
    timeframe: Optional[str] = None
    keywords: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    priority: RequestPriority = RequestPriority.MEDIUM
    enabled_sources: Optional[List[DataSourceType]] = None
    max_fallback_attempts: int = 3

class DataSourceStatus(BaseModel):
    """Status of a data source."""
    source_id: DataSourceType
    enabled: bool
    status: str  # "operational", "error", "rate_limited", "timeout"
    priority: int
    last_successful_call: Optional[datetime] = None
    rate_limit_remaining: Optional[int] = None
    error_message: Optional[str] = None
    response_time_ms: Optional[float] = None

class MultiDataSourceResponse(BaseModel):
    """Response from multi-source data request."""
    success: bool
    primary_source: DataSourceType
    fallback_sources: List[DataSourceType] = []
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    execution_time_ms: float
    cache_hit: bool = False

class MultiDataSourceManager:
    """Manages multiple data sources with intelligent routing."""
    
    def __init__(self):
        self.configurations: Dict[DataSourceType, DataSourceConfig] = {}
        self.status_cache: Dict[DataSourceType, DataSourceStatus] = {}
        self.response_cache: Dict[str, Any] = {}
        self.last_health_check = datetime.now() - timedelta(minutes=10)
        
        # Initialize default configurations
        self._initialize_default_configs()
    
    def _initialize_default_configs(self):
        """Initialize default configurations for all data sources."""
        default_configs = [
            DataSourceConfig(source_id=DataSourceType.IBKR, priority=1, rate_limit_per_minute=50),
            DataSourceConfig(source_id=DataSourceType.ALPHA_VANTAGE, priority=2, rate_limit_per_minute=5),
            DataSourceConfig(source_id=DataSourceType.FRED, priority=3, rate_limit_per_minute=120),
            DataSourceConfig(source_id=DataSourceType.EDGAR, priority=4, rate_limit_per_minute=10),
            DataSourceConfig(source_id=DataSourceType.YFINANCE, priority=5, enabled=True, rate_limit_per_minute=20),
            DataSourceConfig(source_id=DataSourceType.BACKFILL, priority=6, enabled=True, rate_limit_per_minute=60)
        ]
        
        for config in default_configs:
            self.configurations[config.source_id] = config
    
    async def check_data_source_health(self, source: DataSourceType) -> DataSourceStatus:
        """Check health of a specific data source."""
        start_time = datetime.now()
        
        try:
            status = "error"
            error_message = None
            
            if source == DataSourceType.ALPHA_VANTAGE:
                health = await alpha_vantage_service.health_check()
                status = "operational" if health.status == "operational" else "error"
                error_message = health.error_message
                
            elif source == DataSourceType.FRED:
                # Test FRED connection
                try:
                    test_series = await fred_integration.get_series_info()
                    status = "operational" if len(test_series) > 0 else "error"
                except Exception as e:
                    error_message = str(e)
                    
            elif source == DataSourceType.EDGAR:
                # Test EDGAR connection
                try:
                    if EDGAR_AVAILABLE:
                        # Simple test - make a direct health check to EDGAR API
                        import httpx
                        async with httpx.AsyncClient() as client:
                            response = await client.get("http://localhost:8000/api/v1/edgar/health")
                            if response.status_code == 200:
                                status = "operational"
                            else:
                                error_message = f"EDGAR health check failed: {response.status_code}"
                    else:
                        status = "error"
                        error_message = "EDGAR service not available"
                except Exception as e:
                    error_message = str(e)
                    
            elif source == DataSourceType.IBKR:
                # Would check IB Gateway connection
                status = "operational"  # Simplified for demo
                
            elif source == DataSourceType.YFINANCE:
                # Test YFinance service
                try:
                    from yfinance_service import yfinance_service
                    health = await yfinance_service.health_check()
                    status = health.get("status", "error")
                    if status != "operational":
                        error_message = health.get("error", "YFinance service not operational")
                except Exception as e:
                    error_message = str(e)
                    
            elif source == DataSourceType.BACKFILL:
                # Test Backfill service - check if controller is available
                try:
                    # Import the backfill controller directly 
                    import sys
                    import os
                    sys.path.append(os.path.dirname(__file__))
                    from main import backfill_controller
                    
                    # Simple check - if we can get the status, service is operational
                    controller_status = backfill_controller.get_status()
                    if controller_status:
                        status = "operational"
                    else:
                        error_message = "Backfill controller unavailable"
                except Exception as e:
                    error_message = str(e)
            
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            config = self.configurations.get(source, DataSourceConfig(source_id=source, priority=10))
            
            return DataSourceStatus(
                source_id=source,
                enabled=config.enabled,
                status=status,
                priority=config.priority,
                last_successful_call=datetime.now() if status == "operational" else None,
                error_message=error_message,
                response_time_ms=response_time
            )
            
        except Exception as e:
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            config = self.configurations.get(source, DataSourceConfig(source_id=source, priority=10))
            
            return DataSourceStatus(
                source_id=source,
                enabled=config.enabled,
                status="error",
                priority=config.priority,
                error_message=str(e),
                response_time_ms=response_time
            )
    
    async def get_all_health_status(self) -> List[DataSourceStatus]:
        """Get health status for all configured data sources."""
        # Only refresh if cache is old
        if datetime.now() - self.last_health_check > timedelta(minutes=1):
            health_checks = []
            for source in DataSourceType:
                health_checks.append(self.check_data_source_health(source))
            
            statuses = await asyncio.gather(*health_checks, return_exceptions=True)
            
            for i, status in enumerate(statuses):
                source = list(DataSourceType)[i]
                if isinstance(status, DataSourceStatus):
                    self.status_cache[source] = status
                else:
                    # Handle exception case
                    config = self.configurations.get(source, DataSourceConfig(source_id=source, priority=10))
                    self.status_cache[source] = DataSourceStatus(
                        source_id=source,
                        enabled=config.enabled,
                        status="error",
                        priority=config.priority,
                        error_message=str(status)
                    )
            
            self.last_health_check = datetime.now()
        
        return list(self.status_cache.values())
    
    def get_available_sources(self, data_type: str, enabled_only: bool = True) -> List[DataSourceType]:
        """Get available sources for a specific data type, sorted by priority."""
        # Define which sources can handle which data types
        source_capabilities = {
            DataSourceType.IBKR: ["quote", "historical", "fundamentals"],
            DataSourceType.ALPHA_VANTAGE: ["quote", "historical", "search", "fundamentals"],
            DataSourceType.FRED: ["economic", "historical"],
            DataSourceType.EDGAR: ["fundamentals", "search", "filings"],
            DataSourceType.YFINANCE: ["quote", "historical", "fundamentals"],
            DataSourceType.BACKFILL: ["historical", "batch_processing", "gap_detection"]
        }
        
        available = []
        for source, capabilities in source_capabilities.items():
            if data_type in capabilities:
                config = self.configurations.get(source)
                if not enabled_only or (config and config.enabled):
                    status = self.status_cache.get(source)
                    if not enabled_only or (status and status.status == "operational"):
                        available.append(source)
        
        # Sort by priority
        available.sort(key=lambda s: self.configurations.get(s, DataSourceConfig(source_id=s)).priority)
        return available
    
    async def execute_request_on_source(self, source: DataSourceType, request: MultiDataSourceRequest) -> Dict[str, Any]:
        """Execute a request on a specific data source."""
        try:
            if source == DataSourceType.ALPHA_VANTAGE:
                if request.data_type == "quote" and request.symbol:
                    quote = await alpha_vantage_service.get_quote(request.symbol)
                    return {
                        "symbol": quote.symbol,
                        "price": quote.price,
                        "change": quote.change,
                        "change_percent": quote.change_percent,
                        "volume": quote.volume,
                        "timestamp": quote.timestamp.isoformat()
                    }
                elif request.data_type == "search" and request.keywords:
                    results = await alpha_vantage_service.search_symbols(request.keywords)
                    return {
                        "keywords": request.keywords,
                        "results": [
                            {
                                "symbol": r.symbol,
                                "name": r.name,
                                "type": r.type,
                                "region": r.region,
                                "currency": r.currency,
                                "match_score": r.match_score
                            } for r in results
                        ]
                    }
                elif request.data_type == "fundamentals" and request.symbol:
                    company = await alpha_vantage_service.get_company_fundamentals(request.symbol)
                    if company:
                        return {
                            "symbol": company.symbol,
                            "name": company.name,
                            "sector": company.sector,
                            "industry": company.industry,
                            "market_cap": company.market_capitalization,
                            "pe_ratio": company.pe_ratio,
                            "description": company.description
                        }
                    
            elif source == DataSourceType.FRED:
                if request.data_type == "economic":
                    factors = fred_integration.calculate_macro_factors()
                    return {
                        "calculation_date": datetime.now().isoformat(),
                        "factors": factors
                    }
                    
            elif source == DataSourceType.EDGAR:
                if request.data_type == "search" and request.keywords:
                    results = await search_companies(request.keywords)
                    return results
                elif request.data_type == "fundamentals" and request.symbol:
                    facts = await get_company_facts_by_ticker(request.symbol)
                    return facts
            
            # Add more source implementations here
            raise HTTPException(status_code=501, detail=f"Data type '{request.data_type}' not implemented for {source}")
            
        except Exception as e:
            logger.error(f"Request failed on {source}: {e}")
            raise HTTPException(status_code=500, detail=f"Request failed on {source}: {str(e)}")
    
    async def execute_multi_source_request(self, request: MultiDataSourceRequest) -> MultiDataSourceResponse:
        """Execute a request using multiple sources with fallback."""
        start_time = datetime.now()
        
        # Get available sources for this data type
        if request.enabled_sources:
            available_sources = [s for s in request.enabled_sources if s in self.get_available_sources(request.data_type)]
        else:
            available_sources = self.get_available_sources(request.data_type)
        
        if not available_sources:
            raise HTTPException(
                status_code=503, 
                detail=f"No available sources for data type '{request.data_type}'"
            )
        
        # Try each source in priority order
        primary_source = None
        fallback_sources = []
        last_error = None
        
        for source in available_sources[:request.max_fallback_attempts]:
            try:
                data = await self.execute_request_on_source(source, request)
                
                if primary_source is None:
                    primary_source = source
                else:
                    fallback_sources.append(source)
                
                execution_time = (datetime.now() - start_time).total_seconds() * 1000
                
                return MultiDataSourceResponse(
                    success=True,
                    primary_source=primary_source,
                    fallback_sources=fallback_sources,
                    data=data,
                    metadata={
                        "request_type": request.data_type,
                        "symbol": request.symbol,
                        "keywords": request.keywords,
                        "sources_attempted": len(fallback_sources) + 1,
                        "sources_available": len(available_sources)
                    },
                    execution_time_ms=execution_time
                )
                
            except Exception as e:
                logger.warning(f"Source {source} failed: {e}")
                last_error = e
                fallback_sources.append(source)
                continue
        
        # All sources failed
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        raise HTTPException(
            status_code=503,
            detail=f"All available sources failed. Last error: {str(last_error)}"
        )

# Global manager instance
multi_source_manager = MultiDataSourceManager()

# API Endpoints

@router.get("/health", response_model=List[DataSourceStatus])
async def get_multi_source_health():
    """Get health status for all configured data sources."""
    try:
        return await multi_source_manager.get_all_health_status()
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@router.post("/configure")
async def configure_data_source(config: DataSourceConfig):
    """Configure a data source."""
    try:
        multi_source_manager.configurations[config.source_id] = config
        return {"message": f"Data source {config.source_id} configured successfully"}
    except Exception as e:
        logger.error(f"Configuration failed: {e}")
        raise HTTPException(status_code=500, detail=f"Configuration failed: {str(e)}")

@router.get("/configurations", response_model=List[DataSourceConfig])
async def get_configurations():
    """Get current configurations for all data sources."""
    return list(multi_source_manager.configurations.values())

@router.post("/request", response_model=MultiDataSourceResponse)
async def execute_multi_source_request(request: MultiDataSourceRequest):
    """Execute a multi-source data request with intelligent routing and fallback."""
    try:
        return await multi_source_manager.execute_multi_source_request(request)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Multi-source request failed: {e}")
        raise HTTPException(status_code=500, detail=f"Request failed: {str(e)}")

@router.get("/sources/{data_type}")
async def get_available_sources_for_type(data_type: str):
    """Get available sources for a specific data type."""
    try:
        sources = multi_source_manager.get_available_sources(data_type)
        return {
            "data_type": data_type,
            "available_sources": sources,
            "count": len(sources)
        }
    except Exception as e:
        logger.error(f"Failed to get sources for {data_type}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get sources: {str(e)}")

@router.post("/enable/{source_id}")
async def enable_data_source(source_id: DataSourceType):
    """Enable a data source."""
    try:
        config = multi_source_manager.configurations.get(source_id)
        if config:
            config.enabled = True
        else:
            multi_source_manager.configurations[source_id] = DataSourceConfig(
                source_id=source_id, enabled=True
            )
        return {"message": f"Data source {source_id} enabled"}
    except Exception as e:
        logger.error(f"Failed to enable {source_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to enable: {str(e)}")

@router.post("/disable/{source_id}")
async def disable_data_source(source_id: DataSourceType):
    """Disable a data source."""
    try:
        config = multi_source_manager.configurations.get(source_id)
        if config:
            config.enabled = False
        return {"message": f"Data source {source_id} disabled"}
    except Exception as e:
        logger.error(f"Failed to disable {source_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to disable: {str(e)}")

@router.get("/stats")
async def get_multi_source_stats():
    """Get statistics about multi-source usage."""
    try:
        statuses = await multi_source_manager.get_all_health_status()
        
        enabled_count = sum(1 for s in statuses if s.enabled)
        operational_count = sum(1 for s in statuses if s.status == "operational" and s.enabled)
        
        return {
            "total_sources": len(statuses),
            "enabled_sources": enabled_count,
            "operational_sources": operational_count,
            "coverage_percentage": round((operational_count / len(statuses)) * 100, 1),
            "last_health_check": multi_source_manager.last_health_check.isoformat(),
            "cache_size": len(multi_source_manager.response_cache),
            "sources": [
                {
                    "source_id": s.source_id,
                    "enabled": s.enabled,
                    "status": s.status,
                    "priority": s.priority,
                    "response_time_ms": s.response_time_ms
                } for s in statuses
            ]
        }
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")