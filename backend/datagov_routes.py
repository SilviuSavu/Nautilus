"""
Data.gov API Routes
===================

FastAPI routes for Data.gov dataset integration with the Nautilus trading platform.
Following exact patterns from FRED and EDGAR route implementations.
"""

from fastapi import APIRouter, HTTPException, Query, Depends, Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from datagov_integration import datagov_integration
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Create the Data.gov router
router = APIRouter(prefix="/api/v1/datagov", tags=["Data.gov Datasets"])


# Pydantic models for API responses
class DatasetSummary(BaseModel):
    """Dataset summary response."""
    id: str = Field(description="Dataset ID")
    name: str = Field(description="Dataset machine name")
    title: str = Field(description="Dataset display title")
    description: Optional[str] = Field(None, description="Dataset description")
    category: str = Field(description="Dataset category")
    organization: Dict[str, Optional[str]] = Field(description="Organization info")
    tags: List[str] = Field(description="Dataset tags")
    resource_count: int = Field(description="Total resources")
    data_resources: int = Field(description="Data resources count")
    api_resources: int = Field(description="API resources count")
    estimated_frequency: str = Field(description="Update frequency")
    trading_relevant: bool = Field(description="Trading relevance")
    created: Optional[str] = Field(None, description="Creation date")
    modified: Optional[str] = Field(None, description="Last modified")
    url: Optional[str] = Field(None, description="Dataset URL")


class DatasetResource(BaseModel):
    """Dataset resource information."""
    id: str = Field(description="Resource ID")
    name: str = Field(description="Resource name")
    description: Optional[str] = Field(None, description="Resource description")
    url: str = Field(description="Resource URL")
    format: str = Field(description="Resource format")
    size: Optional[str] = Field(None, description="Resource size")
    created: Optional[str] = Field(None, description="Creation date")
    modified: Optional[str] = Field(None, description="Last modified")
    is_data_resource: bool = Field(description="Contains data")


class DatasetDetails(DatasetSummary):
    """Detailed dataset response."""
    resources: List[DatasetResource] = Field(description="Dataset resources")
    extras: Dict[str, Any] = Field(description="Extra metadata")


class SearchResponse(BaseModel):
    """Dataset search response."""
    success: bool = Field(description="Request success")
    query: str = Field(description="Search query")
    total_count: int = Field(description="Total matching datasets")
    returned_count: int = Field(description="Returned datasets")
    datasets: List[DatasetSummary] = Field(description="Dataset results")
    facets: Dict[str, Any] = Field(description="Search facets")
    timestamp: str = Field(description="Response timestamp")


class OrganizationInfo(BaseModel):
    """Organization information."""
    id: str = Field(description="Organization ID")
    name: str = Field(description="Organization name")
    title: str = Field(description="Organization display title")
    description: Optional[str] = Field(None, description="Organization description")


class HealthResponse(BaseModel):
    """Health check response."""
    service: str = Field(description="Service name")
    status: str = Field(description="Service status")
    api_key_configured: bool = Field(description="API key configured")
    api_accessible: bool = Field(description="API accessible")
    datasets_loaded: int = Field(description="Datasets loaded")
    timestamp: str = Field(description="Check timestamp")


# API Routes

@router.get("/health", response_model=HealthResponse)
async def datagov_health_check():
    """Check Data.gov service health and API connectivity."""
    try:
        health_status = await datagov_integration.health_check()
        
        return HealthResponse(
            service=health_status.get("service", "Data.gov Integration"),
            status=health_status.get("status", "unknown"),
            api_key_configured=health_status.get("api_key_configured", False),
            api_accessible=health_status.get("api_accessible", False),
            datasets_loaded=health_status.get("datasets_loaded", 0),
            timestamp=health_status.get("timestamp", datetime.now().isoformat())
        )
    except Exception as e:
        logger.error(f"Data.gov health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/datasets/search", response_model=SearchResponse)
async def search_datasets(
    q: str = Query("*:*", description="Search query"),
    category: Optional[str] = Query(None, description="Filter by category"),
    organization: Optional[str] = Query(None, description="Filter by organization"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results"),
    trading_relevant: bool = Query(True, description="Only trading-relevant datasets")
):
    """Search for datasets by query, category, or organization."""
    try:
        results = await datagov_integration.search_datasets(
            query=q,
            category=category,
            organization=organization,
            limit=limit,
            trading_relevant_only=trading_relevant
        )
        
        if not results.get("success", False):
            raise HTTPException(status_code=500, detail="Search failed")
        
        # Convert to response model
        datasets = []
        for dataset_dict in results.get("datasets", []):
            datasets.append(DatasetSummary(**dataset_dict))
        
        return SearchResponse(
            success=results["success"],
            query=results["query"],
            total_count=results["total_count"],
            returned_count=results["returned_count"],
            datasets=datasets,
            facets=results["facets"],
            timestamp=results["timestamp"]
        )
        
    except Exception as e:
        logger.error(f"Dataset search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/datasets/{dataset_id}", response_model=DatasetDetails)
async def get_dataset_details(
    dataset_id: str = Path(..., description="Dataset ID")
):
    """Get detailed information for a specific dataset."""
    try:
        result = await datagov_integration.get_dataset_details(dataset_id)
        
        if not result.get("success", False):
            error_msg = result.get("error", "Dataset not found")
            raise HTTPException(status_code=404, detail=error_msg)
        
        dataset_dict = result["dataset"]
        
        # Convert resources
        resources = []
        for resource_dict in dataset_dict.get("resources", []):
            resources.append(DatasetResource(**resource_dict))
        
        # Create response
        dataset_dict["resources"] = resources
        
        return DatasetDetails(**dataset_dict)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get dataset {dataset_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve dataset: {str(e)}")


@router.get("/datasets/trading-relevant", response_model=List[DatasetSummary])
async def get_trading_relevant_datasets(
    limit: int = Query(50, ge=1, le=200, description="Maximum results")
):
    """Get datasets most relevant for trading strategies."""
    try:
        result = await datagov_integration.get_trading_relevant_datasets(limit)
        
        if not result.get("success", False):
            raise HTTPException(status_code=500, detail="Failed to get trading-relevant datasets")
        
        datasets = []
        for dataset_dict in result.get("datasets", []):
            datasets.append(DatasetSummary(**dataset_dict))
        
        return datasets
        
    except Exception as e:
        logger.error(f"Failed to get trading-relevant datasets: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get datasets: {str(e)}")


@router.get("/datasets/category/{category}", response_model=List[DatasetSummary])
async def get_datasets_by_category(
    category: str = Path(..., description="Dataset category"),
    limit: int = Query(50, ge=1, le=200, description="Maximum results")
):
    """Get datasets by category."""
    try:
        result = await datagov_integration.get_datasets_by_category(category)
        
        if not result.get("success", False):
            error_msg = result.get("error", "Category not found")
            if "valid_categories" in result:
                error_msg += f". Valid categories: {', '.join(result['valid_categories'])}"
            raise HTTPException(status_code=400, detail=error_msg)
        
        datasets = []
        for dataset_dict in result.get("datasets", [])[:limit]:
            datasets.append(DatasetSummary(**dataset_dict))
        
        return datasets
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get datasets for category {category}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get category datasets: {str(e)}")


@router.get("/datasets/real-time", response_model=List[DatasetSummary])
async def get_real_time_datasets(
    limit: int = Query(50, ge=1, le=200, description="Maximum results")
):
    """Get datasets with real-time or frequent updates."""
    try:
        result = await datagov_integration.get_real_time_datasets()
        
        if not result.get("success", False):
            raise HTTPException(status_code=500, detail="Failed to get real-time datasets")
        
        datasets = []
        for dataset_dict in result.get("datasets", [])[:limit]:
            datasets.append(DatasetSummary(**dataset_dict))
        
        return datasets
        
    except Exception as e:
        logger.error(f"Failed to get real-time datasets: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get real-time datasets: {str(e)}")


@router.get("/organizations", response_model=List[OrganizationInfo])
async def list_organizations():
    """Get list of government organizations publishing datasets."""
    try:
        result = await datagov_integration.get_organizations()
        
        if not result.get("success", False):
            raise HTTPException(status_code=500, detail="Failed to get organizations")
        
        organizations = []
        for org_dict in result.get("organizations", []):
            organizations.append(OrganizationInfo(**org_dict))
        
        return organizations
        
    except Exception as e:
        logger.error(f"Failed to get organizations: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get organizations: {str(e)}")


@router.get("/categories")
async def get_available_categories():
    """Get list of available dataset categories."""
    try:
        categories = [
            {
                "name": "economic",
                "title": "Economic Indicators",
                "description": "GDP, employment, inflation, and financial market data"
            },
            {
                "name": "agricultural", 
                "title": "Agricultural Data",
                "description": "Crop reports, livestock, commodity prices, and food data"
            },
            {
                "name": "energy",
                "title": "Energy Data", 
                "description": "Oil prices, natural gas, electricity, and renewable energy"
            },
            {
                "name": "financial",
                "title": "Financial Data",
                "description": "Banking, securities, derivatives, and market data"
            },
            {
                "name": "transportation",
                "title": "Transportation", 
                "description": "Traffic, transit, highway, aviation, and shipping data"
            },
            {
                "name": "environmental",
                "title": "Environmental Data",
                "description": "Climate, weather, pollution, and air quality data"
            },
            {
                "name": "demographic",
                "title": "Demographic Data",
                "description": "Population, census, and demographic statistics"
            },
            {
                "name": "health",
                "title": "Health Data",
                "description": "Public health, disease, and healthcare statistics"
            },
            {
                "name": "education",
                "title": "Education Data", 
                "description": "School performance, enrollment, and educational statistics"
            },
            {
                "name": "safety",
                "title": "Safety Data",
                "description": "Crime, emergency, and public safety data"
            },
            {
                "name": "other",
                "title": "Other Categories",
                "description": "Miscellaneous datasets not in other categories"
            }
        ]
        
        return {
            "success": True,
            "count": len(categories),
            "categories": categories,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get categories: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get categories: {str(e)}")


@router.post("/datasets/load")
async def load_datasets(force_reload: bool = Query(False, description="Force reload datasets")):
    """Load datasets from Data.gov (admin operation)."""
    try:
        result = await datagov_integration.load_datasets(force_reload)
        
        return {
            "success": result["success"],
            "message": f"Loaded {result['datasets_loaded']} datasets",
            "datasets_loaded": result["datasets_loaded"],
            "trading_relevant": result["trading_relevant"],
            "categories": result["categories"],
            "load_time_seconds": result["load_time_seconds"],
            "timestamp": result["timestamp"]
        }
        
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load datasets: {str(e)}")


@router.get("/status")
async def get_datagov_status():
    """Get comprehensive Data.gov service status and statistics."""
    try:
        stats = await datagov_integration.get_service_statistics()
        
        return {
            "success": True,
            "service": stats.get("service", "Data.gov Integration"),
            "status": stats.get("status", "unknown"),
            "statistics": stats.get("statistics", {}),
            "timestamp": stats.get("timestamp", datetime.now().isoformat())
        }
        
    except Exception as e:
        logger.error(f"Failed to get Data.gov status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@router.get("/cache/refresh")
async def refresh_cache():
    """Clear and refresh the Data.gov cache."""
    try:
        # Clear caches and reload datasets
        if datagov_integration.api_client:
            datagov_integration.api_client.clear_cache()
        
        # Force reload datasets
        result = await datagov_integration.load_datasets(force_reload=True)
        
        return {
            "success": True,
            "message": "Cache refreshed and datasets reloaded",
            "datasets_loaded": result["datasets_loaded"],
            "load_time_seconds": result["load_time_seconds"],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to refresh Data.gov cache: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to refresh cache: {str(e)}")


# Startup event handler to ensure datasets are loaded
@router.on_event("startup")
async def startup_event():
    """Initialize Data.gov services on startup."""
    logger.info("Initializing Data.gov API services...")
    try:
        # Check if service is configured
        health = await datagov_integration.health_check()
        
        if health.get("api_key_configured", False):
            # Load initial dataset catalog in background
            try:
                await datagov_integration.load_datasets()
                logger.info("Data.gov dataset catalog loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load initial dataset catalog: {e}")
        else:
            logger.warning("Data.gov API key not configured - limited functionality")
            
        logger.info("Data.gov services initialized")
        
    except Exception as e:
        logger.error(f"Failed to initialize Data.gov services: {e}")


@router.on_event("shutdown")
async def shutdown_event():
    """Clean up Data.gov services on shutdown."""
    logger.info("Shutting down Data.gov services...")
    try:
        await datagov_integration.close()
        logger.info("Data.gov services shut down successfully")
    except Exception as e:
        logger.error(f"Error during Data.gov services shutdown: {e}")