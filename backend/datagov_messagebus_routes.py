"""
Data.gov API Routes - MessageBus Integration
Provides FastAPI endpoints that communicate with Data.gov via MessageBus events.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, Path, BackgroundTasks
from pydantic import BaseModel, Field

from datagov_messagebus_service import (
    datagov_messagebus_service,
    request_datagov_health,
    request_datagov_search,
    request_datagov_dataset,
    DatagovEventTypes
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/datagov-mb", tags=["Data.gov MessageBus"])


# Response Models
class HealthResponse(BaseModel):
    status: str
    api_key_configured: bool
    api_accessible: bool
    datasets_loaded: int
    timestamp: str
    message: Optional[str] = None


class DatasetSummary(BaseModel):
    """Dataset summary response via MessageBus."""
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


class SearchResponse(BaseModel):
    """Dataset search response via MessageBus."""
    success: bool = Field(description="Request success")
    query: str = Field(description="Search query")
    total_count: int = Field(description="Total matching datasets")
    returned_count: int = Field(description="Returned datasets")
    datasets: List[DatasetSummary] = Field(description="Dataset results")
    facets: Dict[str, Any] = Field(description="Search facets")
    timestamp: str = Field(description="Response timestamp")


class CategoryInfo(BaseModel):
    """Dataset category information."""
    name: str = Field(description="Category name")
    title: str = Field(description="Category display title")
    description: str = Field(description="Category description")


# MessageBus Event Handlers
class DatagovMessageBusResponseHandler:
    """Handles responses from Data.gov MessageBus service"""
    
    def __init__(self):
        self._pending_requests: Dict[str, asyncio.Future] = {}
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup MessageBus response handlers"""
        # Get MessageBus client and register handlers
        try:
            from messagebus_client import get_messagebus_client
            client = get_messagebus_client()
            
            # Register response handlers
            client.add_message_handler(self._handle_health_response)
            client.add_message_handler(self._handle_search_response)
            client.add_message_handler(self._handle_dataset_response)
            client.add_message_handler(self._handle_error_response)
            
        except Exception as e:
            logger.error(f"Failed to setup MessageBus handlers: {e}")
    
    async def wait_for_response(self, correlation_id: str, timeout: float = 30.0) -> Dict[str, Any]:
        """Wait for a specific response by correlation ID"""
        future = asyncio.Future()
        self._pending_requests[correlation_id] = future
        
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=408, detail="Request timeout")
        finally:
            self._pending_requests.pop(correlation_id, None)
    
    async def _handle_health_response(self, message):
        """Handle health check responses"""
        if message.topic == DatagovEventTypes.HEALTH_RESPONSE:
            correlation_id = message.payload.get("correlation_id")
            if correlation_id in self._pending_requests:
                future = self._pending_requests[correlation_id]
                if not future.done():
                    future.set_result(message.payload)
    
    async def _handle_search_response(self, message):
        """Handle search responses"""
        if message.topic == DatagovEventTypes.SEARCH_RESPONSE:
            correlation_id = message.payload.get("correlation_id")
            if correlation_id in self._pending_requests:
                future = self._pending_requests[correlation_id]
                if not future.done():
                    future.set_result(message.payload)
    
    async def _handle_dataset_response(self, message):
        """Handle dataset detail responses"""
        if message.topic == DatagovEventTypes.DATASET_RESPONSE:
            correlation_id = message.payload.get("correlation_id")
            if correlation_id in self._pending_requests:
                future = self._pending_requests[correlation_id]
                if not future.done():
                    future.set_result(message.payload)
    
    async def _handle_error_response(self, message):
        """Handle error responses"""
        if message.topic == DatagovEventTypes.ERROR_RESPONSE:
            correlation_id = message.payload.get("correlation_id")
            if correlation_id in self._pending_requests:
                future = self._pending_requests[correlation_id]
                if not future.done():
                    error_msg = message.payload.get("error", "Unknown error")
                    future.set_exception(HTTPException(status_code=500, detail=error_msg))


# Global response handler
response_handler = DatagovMessageBusResponseHandler()


# API Routes

@router.get("/health", response_model=HealthResponse)
async def datagov_health_check():
    """Check Data.gov service health via MessageBus."""
    correlation_id = f"health_{int(datetime.utcnow().timestamp() * 1000)}"
    
    try:
        # Send health check request via MessageBus
        await request_datagov_health(correlation_id=correlation_id)
        
        # Wait for response
        response_data = await response_handler.wait_for_response(correlation_id)
        
        return HealthResponse(
            status=response_data.get("status", "unknown"),
            api_key_configured=response_data.get("api_key_configured", False),
            api_accessible=response_data.get("api_accessible", False),
            datasets_loaded=response_data.get("datasets_loaded", 0),
            timestamp=response_data.get("timestamp", datetime.now().isoformat())
        )
    except Exception as e:
        logger.error(f"Data.gov MessageBus health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/datasets/search", response_model=SearchResponse)
async def search_datasets(
    q: str = Query("*:*", description="Search query"),
    category: Optional[str] = Query(None, description="Filter by category"),
    organization: Optional[str] = Query(None, description="Filter by organization"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results"),
    trading_relevant: bool = Query(True, description="Only trading-relevant datasets")
):
    """Search for datasets by query, category, or organization via MessageBus."""
    correlation_id = f"search_{int(datetime.utcnow().timestamp() * 1000)}"
    
    try:
        # Send search request via MessageBus
        await request_datagov_search(
            query=q,
            category=category,
            organization=organization,
            limit=limit,
            trading_relevant_only=trading_relevant,
            correlation_id=correlation_id
        )
        
        # Wait for response
        response_data = await response_handler.wait_for_response(correlation_id)
        
        if not response_data.get("success", False):
            raise HTTPException(status_code=500, detail="Search failed")
        
        # Convert to response model
        datasets = []
        for dataset_dict in response_data.get("datasets", []):
            datasets.append(DatasetSummary(**dataset_dict))
        
        return SearchResponse(
            success=response_data["success"],
            query=response_data["query"],
            total_count=response_data["total_count"],
            returned_count=response_data["returned_count"],
            datasets=datasets,
            facets=response_data["facets"],
            timestamp=response_data["timestamp"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dataset search via MessageBus failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/datasets/{dataset_id}")
async def get_dataset_details(
    dataset_id: str = Path(..., description="Dataset ID")
):
    """Get detailed information for a specific dataset via MessageBus."""
    correlation_id = f"dataset_{int(datetime.utcnow().timestamp() * 1000)}"
    
    try:
        # Send dataset request via MessageBus
        await request_datagov_dataset(dataset_id=dataset_id, correlation_id=correlation_id)
        
        # Wait for response
        response_data = await response_handler.wait_for_response(correlation_id)
        
        if not response_data.get("success", False):
            error_msg = response_data.get("error", "Dataset not found")
            raise HTTPException(status_code=404, detail=error_msg)
        
        return response_data["dataset"]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get dataset {dataset_id} via MessageBus: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve dataset: {str(e)}")


@router.get("/categories")
async def get_available_categories():
    """Get list of available dataset categories via MessageBus."""
    # Categories are static, so we can return them directly
    categories = [
        {"name": "economic", "title": "Economic Indicators", "description": "GDP, employment, inflation, and financial market data"},
        {"name": "agricultural", "title": "Agricultural Data", "description": "Crop reports, livestock, commodity prices, and food data"},
        {"name": "energy", "title": "Energy Data", "description": "Oil prices, natural gas, electricity, and renewable energy"},
        {"name": "financial", "title": "Financial Data", "description": "Banking, securities, derivatives, and market data"},
        {"name": "transportation", "title": "Transportation", "description": "Traffic, transit, highway, aviation, and shipping data"},
        {"name": "environmental", "title": "Environmental Data", "description": "Climate, weather, pollution, and air quality data"},
        {"name": "demographic", "title": "Demographic Data", "description": "Population, census, and demographic statistics"},
        {"name": "health", "title": "Health Data", "description": "Public health, disease, and healthcare statistics"},
        {"name": "education", "title": "Education Data", "description": "School performance, enrollment, and educational statistics"},
        {"name": "safety", "title": "Safety Data", "description": "Crime, emergency, and public safety data"},
        {"name": "other", "title": "Other Categories", "description": "Miscellaneous datasets not in other categories"}
    ]
    
    return {
        "success": True,
        "count": len(categories),
        "categories": categories,
        "timestamp": datetime.now().isoformat()
    }


@router.get("/status")
async def get_datagov_status():
    """Get comprehensive Data.gov service status via MessageBus."""
    correlation_id = f"status_{int(datetime.utcnow().timestamp() * 1000)}"
    
    try:
        # Use health check as status check
        await request_datagov_health(correlation_id=correlation_id)
        response_data = await response_handler.wait_for_response(correlation_id)
        
        return {
            "success": True,
            "service": "Data.gov Integration (MessageBus)",
            "status": response_data.get("status", "unknown"),
            "statistics": {
                "api_key_configured": response_data.get("api_key_configured", False),
                "api_accessible": response_data.get("api_accessible", False),
                "datasets_loaded": response_data.get("datasets_loaded", 0)
            },
            "timestamp": response_data.get("timestamp", datetime.now().isoformat())
        }
        
    except Exception as e:
        logger.error(f"Failed to get Data.gov status via MessageBus: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


# Startup event handler to ensure MessageBus service is started
@router.on_event("startup")
async def startup_event():
    """Initialize Data.gov MessageBus services on startup."""
    logger.info("Initializing Data.gov MessageBus API services...")
    try:
        # Start the MessageBus service
        await datagov_messagebus_service.start()
        logger.info("Data.gov MessageBus services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Data.gov MessageBus services: {e}")


@router.on_event("shutdown")
async def shutdown_event():
    """Clean up Data.gov MessageBus services on shutdown."""
    logger.info("Shutting down Data.gov MessageBus services...")
    try:
        await datagov_messagebus_service.stop()
        logger.info("Data.gov MessageBus services shut down successfully")
    except Exception as e:
        logger.error(f"Error during Data.gov MessageBus services shutdown: {e}")