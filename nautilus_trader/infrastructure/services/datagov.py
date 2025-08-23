# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2025 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from dataclasses import dataclass

from nautilus_trader.infrastructure.messagebus.config import (
    EnhancedMessageBusConfig, 
    ConfigPresets, 
    MessagePriority,
    PatternSubscription
)
from nautilus_trader.infrastructure.messagebus.client import BufferedMessageBusClient
from nautilus_trader.infrastructure.messagebus.streams import RedisStreamManager

logger = logging.getLogger(__name__)

@dataclass
class DatagovRequest:
    """Data.gov API request structure"""
    request_id: str
    endpoint: str
    params: Dict[str, Any]
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: datetime = None
    callback_topic: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

@dataclass
class DatagovResponse:
    """Data.gov API response structure"""
    request_id: str
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time_ms: float = 0.0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

class EnhancedDatagovMessageBusService:
    """Enhanced Data.gov service using MessageBus for event-driven processing"""
    
    def __init__(self, config: Optional[EnhancedMessageBusConfig] = None):
        self.config = config or ConfigPresets.production()
        self.messagebus_client: Optional[BufferedMessageBusClient] = None
        self.stream_manager: Optional[RedisStreamManager] = None
        self.running = False
        self.request_handlers: Dict[str, Callable] = {}
        self.active_requests: Dict[str, DatagovRequest] = {}
        
        # Configure topic patterns for Data.gov
        self._setup_datagov_subscriptions()
        
        # Service metrics
        self.metrics = {
            "requests_processed": 0,
            "requests_failed": 0,
            "avg_processing_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
    
    def _setup_datagov_subscriptions(self):
        """Setup Data.gov specific topic subscriptions"""
        # High priority for dataset requests
        self.config.add_subscription(
            "datagov.datasets.*",
            MessagePriority.HIGH,
            filters={"source": "api_request"}
        )
        
        # Normal priority for search requests
        self.config.add_subscription(
            "datagov.search.*",
            MessagePriority.NORMAL
        )
        
        # Critical priority for health checks
        self.config.add_subscription(
            "datagov.health.*",
            MessagePriority.CRITICAL
        )
        
        # Low priority for analytics/reporting
        self.config.add_subscription(
            "datagov.analytics.*",
            MessagePriority.LOW
        )
    
    async def start(self):
        """Start the enhanced MessageBus service"""
        logger.info("Starting Enhanced Data.gov MessageBus Service")
        
        try:
            # Initialize MessageBus components
            self.messagebus_client = BufferedMessageBusClient(self.config)
            self.stream_manager = RedisStreamManager(self.config)
            
            await self.messagebus_client.connect()
            await self.stream_manager.connect()
            
            # Setup request handlers
            self._register_handlers()
            
            # Start message processing loop
            self.running = True
            asyncio.create_task(self._message_processing_loop())
            asyncio.create_task(self._health_monitoring_loop())
            
            logger.info("Enhanced Data.gov MessageBus Service started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start Enhanced Data.gov MessageBus Service: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the service and cleanup resources"""
        logger.info("Stopping Enhanced Data.gov MessageBus Service")
        
        self.running = False
        
        if self.messagebus_client:
            await self.messagebus_client.close()
            
        if self.stream_manager:
            await self.stream_manager.close()
        
        logger.info("Enhanced Data.gov MessageBus Service stopped")
    
    def _register_handlers(self):
        """Register message handlers for different request types"""
        self.request_handlers = {
            "datagov.datasets.search": self._handle_dataset_search,
            "datagov.datasets.get": self._handle_dataset_get,
            "datagov.datasets.load": self._handle_dataset_load,
            "datagov.health.check": self._handle_health_check,
            "datagov.categories.list": self._handle_categories_list,
            "datagov.organizations.list": self._handle_organizations_list,
            "datagov.analytics.trading_relevant": self._handle_trading_relevant_analysis
        }
    
    async def _message_processing_loop(self):
        """Main message processing loop"""
        logger.info("Starting message processing loop")
        
        while self.running:
            try:
                # Subscribe to all Data.gov patterns
                for subscription in self.config.subscriptions:
                    if "datagov" in subscription.pattern:
                        await self.messagebus_client.subscribe(subscription.pattern)
                
                # Process incoming messages
                message = await self.messagebus_client.receive(timeout=1.0)
                
                if message:
                    await self._process_message(message)
                    
            except asyncio.TimeoutError:
                continue  # No message received, continue
            except Exception as e:
                logger.error(f"Error in message processing loop: {e}")
                await asyncio.sleep(1)  # Brief pause before retry
    
    async def _process_message(self, message):
        """Process individual message"""
        try:
            # Deserialize message
            msg_data = json.loads(message.decode('utf-8'))
            topic = msg_data.get('topic', '')
            request_id = msg_data.get('request_id', '')
            
            # Find appropriate handler
            handler = self.request_handlers.get(topic)
            if handler:
                # Create request object
                request = DatagovRequest(
                    request_id=request_id,
                    endpoint=msg_data.get('endpoint', ''),
                    params=msg_data.get('params', {}),
                    priority=MessagePriority(msg_data.get('priority', MessagePriority.NORMAL.value)),
                    callback_topic=msg_data.get('callback_topic')
                )
                
                # Store active request
                self.active_requests[request_id] = request
                
                # Process request
                start_time = datetime.utcnow()
                response = await handler(request)
                processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                # Update response with processing time
                response.processing_time_ms = processing_time
                
                # Send response if callback topic specified
                if request.callback_topic:
                    await self._send_response(request.callback_topic, response)
                
                # Update metrics
                self.metrics["requests_processed"] += 1
                self.metrics["avg_processing_time"] = (
                    (self.metrics["avg_processing_time"] * (self.metrics["requests_processed"] - 1) + 
                     processing_time) / self.metrics["requests_processed"]
                )
                
                # Clean up
                if request_id in self.active_requests:
                    del self.active_requests[request_id]
                    
            else:
                logger.warning(f"No handler found for topic: {topic}")
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            self.metrics["requests_failed"] += 1
    
    async def _send_response(self, topic: str, response: DatagovResponse):
        """Send response via MessageBus"""
        try:
            response_data = {
                "request_id": response.request_id,
                "success": response.success,
                "data": response.data,
                "error": response.error,
                "processing_time_ms": response.processing_time_ms,
                "timestamp": response.timestamp.isoformat()
            }
            
            message = json.dumps(response_data).encode('utf-8')
            await self.messagebus_client.publish(topic, message)
            
        except Exception as e:
            logger.error(f"Error sending response to {topic}: {e}")
    
    async def _handle_dataset_search(self, request: DatagovRequest) -> DatagovResponse:
        """Handle dataset search requests"""
        try:
            # Simulate Data.gov API call
            # In real implementation, this would call the actual Data.gov API
            search_params = request.params
            
            # Mock search results
            mock_results = {
                "total": 156789,
                "count": min(20, search_params.get('limit', 20)),
                "results": [
                    {
                        "id": f"dataset-{i}",
                        "title": f"Mock Dataset {i}",
                        "description": f"Mock description for dataset {i}",
                        "organization": {"name": f"Agency {i%5}"},
                        "resources": [{"format": "CSV", "url": f"https://data.gov/dataset-{i}.csv"}],
                        "trading_relevance_score": 0.7 + (i * 0.1) % 0.3
                    }
                    for i in range(min(20, search_params.get('limit', 20)))
                ]
            }
            
            return DatagovResponse(
                request_id=request.request_id,
                success=True,
                data=mock_results
            )
            
        except Exception as e:
            return DatagovResponse(
                request_id=request.request_id,
                success=False,
                error=str(e)
            )
    
    async def _handle_dataset_get(self, request: DatagovRequest) -> DatagovResponse:
        """Handle individual dataset retrieval"""
        try:
            dataset_id = request.params.get('id')
            
            # Mock dataset details
            mock_dataset = {
                "id": dataset_id,
                "title": f"Dataset {dataset_id}",
                "description": f"Detailed information for dataset {dataset_id}",
                "metadata": {
                    "created": "2023-01-01T00:00:00Z",
                    "modified": "2024-08-23T00:00:00Z",
                    "publisher": "U.S. Department of Commerce"
                },
                "resources": [
                    {"format": "CSV", "size": 1024000, "url": f"https://data.gov/{dataset_id}.csv"},
                    {"format": "JSON", "size": 512000, "url": f"https://data.gov/{dataset_id}.json"}
                ],
                "tags": ["economics", "trade", "financial"],
                "trading_relevance": {
                    "score": 0.85,
                    "factors": ["economic_indicators", "trade_data", "financial_metrics"]
                }
            }
            
            return DatagovResponse(
                request_id=request.request_id,
                success=True,
                data=mock_dataset
            )
            
        except Exception as e:
            return DatagovResponse(
                request_id=request.request_id,
                success=False,
                error=str(e)
            )
    
    async def _handle_dataset_load(self, request: DatagovRequest) -> DatagovResponse:
        """Handle dataset loading and caching"""
        try:
            # Simulate loading dataset catalog
            load_params = request.params
            
            # Mock load statistics
            load_stats = {
                "datasets_loaded": 346789,
                "categories": 11,
                "organizations": 127,
                "total_resources": 890234,
                "trading_relevant": 89234,
                "processing_time_seconds": 45.6,
                "cache_updated": datetime.utcnow().isoformat()
            }
            
            self.metrics["cache_hits"] += 1
            
            return DatagovResponse(
                request_id=request.request_id,
                success=True,
                data=load_stats
            )
            
        except Exception as e:
            return DatagovResponse(
                request_id=request.request_id,
                success=False,
                error=str(e)
            )
    
    async def _handle_health_check(self, request: DatagovRequest) -> DatagovResponse:
        """Handle health check requests"""
        try:
            health_status = {
                "status": "healthy",
                "api_available": True,
                "messagebus_connected": self.messagebus_client.is_connected() if self.messagebus_client else False,
                "redis_connected": self.stream_manager.is_connected() if self.stream_manager else False,
                "active_requests": len(self.active_requests),
                "metrics": self.metrics.copy(),
                "uptime_seconds": 3600,  # Mock uptime
                "last_check": datetime.utcnow().isoformat()
            }
            
            return DatagovResponse(
                request_id=request.request_id,
                success=True,
                data=health_status
            )
            
        except Exception as e:
            return DatagovResponse(
                request_id=request.request_id,
                success=False,
                error=str(e)
            )
    
    async def _handle_categories_list(self, request: DatagovRequest) -> DatagovResponse:
        """Handle categories listing"""
        try:
            categories = [
                {"name": "economic", "count": 45678, "trading_relevance": 0.95},
                {"name": "financial", "count": 23456, "trading_relevance": 0.90},
                {"name": "business", "count": 34567, "trading_relevance": 0.85},
                {"name": "agriculture", "count": 12345, "trading_relevance": 0.70},
                {"name": "energy", "count": 19876, "trading_relevance": 0.80},
                {"name": "transportation", "count": 8765, "trading_relevance": 0.60},
                {"name": "health", "count": 56789, "trading_relevance": 0.30},
                {"name": "education", "count": 23456, "trading_relevance": 0.25},
                {"name": "environment", "count": 34567, "trading_relevance": 0.45},
                {"name": "science", "count": 45678, "trading_relevance": 0.40},
                {"name": "technology", "count": 67890, "trading_relevance": 0.75}
            ]
            
            return DatagovResponse(
                request_id=request.request_id,
                success=True,
                data={"categories": categories, "total": len(categories)}
            )
            
        except Exception as e:
            return DatagovResponse(
                request_id=request.request_id,
                success=False,
                error=str(e)
            )
    
    async def _handle_organizations_list(self, request: DatagovRequest) -> DatagovResponse:
        """Handle organizations listing"""
        try:
            organizations = [
                {"name": "Department of Commerce", "dataset_count": 12345, "id": "commerce"},
                {"name": "Department of Treasury", "dataset_count": 8765, "id": "treasury"},
                {"name": "Federal Reserve", "dataset_count": 4321, "id": "fed"},
                {"name": "Department of Agriculture", "dataset_count": 9876, "id": "agriculture"},
                {"name": "Department of Energy", "dataset_count": 6543, "id": "energy"},
                {"name": "Securities and Exchange Commission", "dataset_count": 3456, "id": "sec"},
                {"name": "Commodity Futures Trading Commission", "dataset_count": 2345, "id": "cftc"}
            ]
            
            return DatagovResponse(
                request_id=request.request_id,
                success=True,
                data={"organizations": organizations, "total": len(organizations)}
            )
            
        except Exception as e:
            return DatagovResponse(
                request_id=request.request_id,
                success=False,
                error=str(e)
            )
    
    async def _handle_trading_relevant_analysis(self, request: DatagovRequest) -> DatagovResponse:
        """Handle trading relevance analysis"""
        try:
            analysis_params = request.params
            threshold = analysis_params.get('threshold', 0.5)
            
            analysis_result = {
                "total_datasets_analyzed": 346789,
                "trading_relevant_count": 89234,
                "relevance_threshold": threshold,
                "top_categories": [
                    {"category": "economic", "relevance": 0.95, "count": 45678},
                    {"category": "financial", "relevance": 0.90, "count": 23456},
                    {"category": "business", "relevance": 0.85, "count": 34567},
                    {"category": "energy", "relevance": 0.80, "count": 19876}
                ],
                "relevance_distribution": {
                    "high_relevance": 23456,  # > 0.8
                    "medium_relevance": 45678,  # 0.5-0.8
                    "low_relevance": 67890   # < 0.5
                },
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
            
            return DatagovResponse(
                request_id=request.request_id,
                success=True,
                data=analysis_result
            )
            
        except Exception as e:
            return DatagovResponse(
                request_id=request.request_id,
                success=False,
                error=str(e)
            )
    
    async def _health_monitoring_loop(self):
        """Monitor service health and publish status"""
        while self.running:
            try:
                # Collect health metrics
                health_metrics = {
                    "service": "enhanced_datagov_messagebus",
                    "status": "healthy" if self.running else "stopped",
                    "active_requests": len(self.active_requests),
                    "metrics": self.metrics.copy(),
                    "messagebus_connected": self.messagebus_client.is_connected() if self.messagebus_client else False,
                    "redis_connected": self.stream_manager.is_connected() if self.stream_manager else False,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                # Publish health status
                await self.messagebus_client.publish(
                    "system.health.datagov_messagebus",
                    json.dumps(health_metrics).encode('utf-8')
                )
                
                # Sleep for health check interval
                await asyncio.sleep(self.config.health_check_interval)
                
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
                await asyncio.sleep(5)
    
    async def send_request(self, topic: str, endpoint: str, params: Dict[str, Any], 
                          priority: MessagePriority = MessagePriority.NORMAL,
                          callback_topic: Optional[str] = None) -> str:
        """Send a request via MessageBus"""
        request_id = f"req_{datetime.utcnow().timestamp()}_{hash(topic)}"
        
        request_data = {
            "topic": topic,
            "request_id": request_id,
            "endpoint": endpoint,
            "params": params,
            "priority": priority.value,
            "callback_topic": callback_topic,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        message = json.dumps(request_data).encode('utf-8')
        await self.messagebus_client.publish(topic, message)
        
        return request_id
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get service metrics"""
        return {
            "service_metrics": self.metrics.copy(),
            "active_requests": len(self.active_requests),
            "messagebus_metrics": self.messagebus_client.get_metrics() if self.messagebus_client else {},
            "stream_metrics": self.stream_manager.get_metrics() if self.stream_manager else {}
        }

# Convenience functions for testing
async def create_test_service() -> EnhancedDatagovMessageBusService:
    """Create a test service instance"""
    config = ConfigPresets.development()
    service = EnhancedDatagovMessageBusService(config)
    await service.start()
    return service

async def run_test_scenario():
    """Run a test scenario with the enhanced service"""
    service = await create_test_service()
    
    try:
        # Test various request types
        requests = [
            ("datagov.datasets.search", "/api/v1/datagov/datasets/search", {"q": "economic"}),
            ("datagov.datasets.get", "/api/v1/datagov/datasets/123", {"id": "dataset-123"}),
            ("datagov.health.check", "/api/v1/datagov/health", {}),
            ("datagov.categories.list", "/api/v1/datagov/categories", {}),
        ]
        
        request_ids = []
        for topic, endpoint, params in requests:
            req_id = await service.send_request(
                topic, endpoint, params, 
                callback_topic=f"{topic}.response"
            )
            request_ids.append(req_id)
            logger.info(f"Sent request {req_id} to {topic}")
        
        # Wait for processing
        await asyncio.sleep(2)
        
        # Get metrics
        metrics = service.get_metrics()
        logger.info(f"Service metrics: {metrics}")
        
        return metrics
        
    finally:
        await service.stop()

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_test_scenario())