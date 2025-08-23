"""
Data.gov MessageBus Service
Provides event-driven integration with Data.gov datasets via MessageBus.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from messagebus_client import MessageBusMessage, get_messagebus_client
from datagov_integration import datagov_integration

logger = logging.getLogger(__name__)


@dataclass
class DatagovEventTypes:
    """Event types for Data.gov MessageBus integration"""
    HEALTH_CHECK = "datagov.health_check"
    SEARCH_REQUEST = "datagov.search.request"
    DATASET_REQUEST = "datagov.dataset.request"
    LOAD_DATASETS = "datagov.datasets.load"
    ORGANIZATIONS_REQUEST = "datagov.organizations.request"
    CATEGORIES_REQUEST = "datagov.categories.request"
    TRADING_RELEVANT_REQUEST = "datagov.trading_relevant.request"
    
    # Response events
    HEALTH_RESPONSE = "datagov.health.response"
    SEARCH_RESPONSE = "datagov.search.response"
    DATASET_RESPONSE = "datagov.dataset.response"
    LOAD_RESPONSE = "datagov.load.response"
    ORGANIZATIONS_RESPONSE = "datagov.organizations.response"
    CATEGORIES_RESPONSE = "datagov.categories.response"
    TRADING_RELEVANT_RESPONSE = "datagov.trading_relevant.response"
    ERROR_RESPONSE = "datagov.error"


class DatagovMessageBusService:
    """
    Data.gov MessageBus service that handles federal dataset requests via event-driven architecture.
    """
    
    def __init__(self):
        self.messagebus_client = get_messagebus_client()
        self._running = False
        self._handlers: Dict[str, callable] = {}
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup event handlers for Data.gov events"""
        self._handlers = {
            DatagovEventTypes.HEALTH_CHECK: self._handle_health_check,
            DatagovEventTypes.SEARCH_REQUEST: self._handle_search_request,
            DatagovEventTypes.DATASET_REQUEST: self._handle_dataset_request,
            DatagovEventTypes.LOAD_DATASETS: self._handle_load_datasets,
            DatagovEventTypes.ORGANIZATIONS_REQUEST: self._handle_organizations_request,
            DatagovEventTypes.CATEGORIES_REQUEST: self._handle_categories_request,
            DatagovEventTypes.TRADING_RELEVANT_REQUEST: self._handle_trading_relevant_request,
        }
    
    async def start(self):
        """Start the Data.gov MessageBus service"""
        logger.info("Starting Data.gov MessageBus service")
        self._running = True
        
        # Register message handlers with MessageBus client
        for event_type, handler in self._handlers.items():
            self.messagebus_client.add_message_handler(self._create_message_filter(event_type, handler))
        
        logger.info("Data.gov MessageBus service started")
    
    async def stop(self):
        """Stop the Data.gov MessageBus service"""
        logger.info("Stopping Data.gov MessageBus service")
        self._running = False
        logger.info("Data.gov MessageBus service stopped")
    
    def _create_message_filter(self, event_type: str, handler: callable):
        """Create a message filter for specific event types"""
        async def filtered_handler(message: MessageBusMessage):
            if message.topic == event_type and self._running:
                try:
                    await handler(message)
                except Exception as e:
                    logger.error(f"Error handling {event_type}: {e}")
                    await self._publish_error(message, str(e))
        
        return filtered_handler
    
    async def _publish_response(self, original_message: MessageBusMessage, event_type: str, data: Dict[str, Any]):
        """Publish a response event to the MessageBus"""
        try:
            # Include correlation_id from original message for request tracking
            response_payload = {
                **data,
                "correlation_id": original_message.payload.get("correlation_id"),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            response_message = MessageBusMessage(
                topic=event_type,
                payload=response_payload,
                timestamp=int(datetime.utcnow().timestamp() * 1000),
                message_type="response"
            )
            
            await self.messagebus_client.send_message(
                response_message.topic, 
                response_message.payload, 
                response_message.message_type
            )
            logger.debug(f"Published response: {event_type}")
            
        except Exception as e:
            logger.error(f"Failed to publish response {event_type}: {e}")
    
    async def _publish_error(self, original_message: MessageBusMessage, error_message: str):
        """Publish an error response"""
        error_data = {
            "success": False,
            "error": error_message,
            "error_type": "data_gov_error"
        }
        await self._publish_response(original_message, DatagovEventTypes.ERROR_RESPONSE, error_data)
    
    async def _handle_health_check(self, message: MessageBusMessage):
        """Handle health check requests"""
        logger.info("Handling Data.gov health check request")
        
        try:
            health_status = await datagov_integration.health_check()
            await self._publish_response(message, DatagovEventTypes.HEALTH_RESPONSE, health_status)
        except Exception as e:
            await self._publish_error(message, f"Health check failed: {e}")
    
    async def _handle_search_request(self, message: MessageBusMessage):
        """Handle dataset search requests"""
        payload = message.payload
        logger.info(f"Handling Data.gov search request: {payload.get('query', '*:*')}")
        
        try:
            # Extract search parameters
            query = payload.get("query", "*:*")
            category = payload.get("category")
            organization = payload.get("organization")
            limit = payload.get("limit", 20)
            trading_relevant_only = payload.get("trading_relevant_only", True)
            
            # Perform search
            results = await datagov_integration.search_datasets(
                query=query,
                category=category,
                organization=organization,
                limit=limit,
                trading_relevant_only=trading_relevant_only
            )
            
            await self._publish_response(message, DatagovEventTypes.SEARCH_RESPONSE, results)
            
        except Exception as e:
            await self._publish_error(message, f"Search failed: {e}")
    
    async def _handle_dataset_request(self, message: MessageBusMessage):
        """Handle individual dataset detail requests"""
        payload = message.payload
        dataset_id = payload.get("dataset_id")
        
        if not dataset_id:
            await self._publish_error(message, "dataset_id is required")
            return
        
        logger.info(f"Handling Data.gov dataset request: {dataset_id}")
        
        try:
            result = await datagov_integration.get_dataset_details(dataset_id)
            await self._publish_response(message, DatagovEventTypes.DATASET_RESPONSE, result)
        except Exception as e:
            await self._publish_error(message, f"Dataset request failed: {e}")
    
    async def _handle_load_datasets(self, message: MessageBusMessage):
        """Handle dataset loading requests"""
        payload = message.payload
        force_reload = payload.get("force_reload", False)
        
        logger.info(f"Handling Data.gov dataset load request (force_reload={force_reload})")
        
        try:
            result = await datagov_integration.load_datasets(force_reload=force_reload)
            await self._publish_response(message, DatagovEventTypes.LOAD_RESPONSE, result)
        except Exception as e:
            await self._publish_error(message, f"Dataset loading failed: {e}")
    
    async def _handle_organizations_request(self, message: MessageBusMessage):
        """Handle organizations list requests"""
        logger.info("Handling Data.gov organizations request")
        
        try:
            result = await datagov_integration.get_organizations()
            await self._publish_response(message, DatagovEventTypes.ORGANIZATIONS_RESPONSE, result)
        except Exception as e:
            await self._publish_error(message, f"Organizations request failed: {e}")
    
    async def _handle_categories_request(self, message: MessageBusMessage):
        """Handle categories list requests"""
        logger.info("Handling Data.gov categories request")
        
        try:
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
            
            result = {
                "success": True,
                "count": len(categories),
                "categories": categories,
                "timestamp": datetime.now().isoformat()
            }
            
            await self._publish_response(message, DatagovEventTypes.CATEGORIES_RESPONSE, result)
        except Exception as e:
            await self._publish_error(message, f"Categories request failed: {e}")
    
    async def _handle_trading_relevant_request(self, message: MessageBusMessage):
        """Handle trading-relevant datasets requests"""
        payload = message.payload
        limit = payload.get("limit", 50)
        
        logger.info(f"Handling Data.gov trading-relevant request (limit={limit})")
        
        try:
            result = await datagov_integration.get_trading_relevant_datasets(limit=limit)
            await self._publish_response(message, DatagovEventTypes.TRADING_RELEVANT_RESPONSE, result)
        except Exception as e:
            await self._publish_error(message, f"Trading-relevant request failed: {e}")


# Global service instance
datagov_messagebus_service = DatagovMessageBusService()


# Helper functions for easy event publishing
async def request_datagov_health(correlation_id: str = None) -> None:
    """Request Data.gov health check via MessageBus"""
    message = MessageBusMessage(
        topic=DatagovEventTypes.HEALTH_CHECK,
        payload={"correlation_id": correlation_id or f"health_{int(datetime.utcnow().timestamp())}"},
        timestamp=int(datetime.utcnow().timestamp() * 1000),
        message_type="request"
    )
    
    client = get_messagebus_client()
    await client.send_message(message.topic, message.payload, message.message_type)


async def request_datagov_search(
    query: str = "*:*",
    category: Optional[str] = None,
    organization: Optional[str] = None,
    limit: int = 20,
    trading_relevant_only: bool = True,
    correlation_id: str = None
) -> None:
    """Request Data.gov dataset search via MessageBus"""
    message = MessageBusMessage(
        topic=DatagovEventTypes.SEARCH_REQUEST,
        payload={
            "query": query,
            "category": category,
            "organization": organization,
            "limit": limit,
            "trading_relevant_only": trading_relevant_only,
            "correlation_id": correlation_id or f"search_{int(datetime.utcnow().timestamp())}"
        },
        timestamp=int(datetime.utcnow().timestamp() * 1000),
        message_type="request"
    )
    
    client = get_messagebus_client()
    await client.send_message(message.topic, message.payload, message.message_type)


async def request_datagov_dataset(dataset_id: str, correlation_id: str = None) -> None:
    """Request specific Data.gov dataset via MessageBus"""
    message = MessageBusMessage(
        topic=DatagovEventTypes.DATASET_REQUEST,
        payload={
            "dataset_id": dataset_id,
            "correlation_id": correlation_id or f"dataset_{int(datetime.utcnow().timestamp())}"
        },
        timestamp=int(datetime.utcnow().timestamp() * 1000),
        message_type="request"
    )
    
    client = get_messagebus_client()
    await client.send_message(message.topic, message.payload, message.message_type)