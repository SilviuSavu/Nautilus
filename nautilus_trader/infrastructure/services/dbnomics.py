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
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd

from nautilus_trader.infrastructure.messagebus.config import (
    EnhancedMessageBusConfig, 
    ConfigPresets, 
    MessagePriority
)
from nautilus_trader.infrastructure.messagebus.client import BufferedMessageBusClient
from nautilus_trader.infrastructure.messagebus.streams import RedisStreamManager

logger = logging.getLogger(__name__)

@dataclass
class DbnomicsRequest:
    """DBnomics API request structure"""
    request_id: str
    endpoint: str
    params: Dict[str, Any]
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: datetime = None
    callback_topic: Optional[str] = None
    cache_ttl: int = 3600  # Cache TTL in seconds
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

@dataclass
class DbnomicsResponse:
    """DBnomics API response structure"""
    request_id: str
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time_ms: float = 0.0
    cached: bool = False
    provider_info: Optional[Dict[str, str]] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

class EnhancedDbnomicsMessageBusService:
    """Enhanced DBnomics service using MessageBus for event-driven processing"""
    
    def __init__(self, config: Optional[EnhancedMessageBusConfig] = None):
        self.config = config or ConfigPresets.production()
        self.messagebus_client: Optional[BufferedMessageBusClient] = None
        self.stream_manager: Optional[RedisStreamManager] = None
        self.running = False
        self.request_handlers: Dict[str, Any] = {}
        self.active_requests: Dict[str, DbnomicsRequest] = {}
        self.data_cache: Dict[str, Any] = {}
        
        # Configure DBnomics-specific subscriptions
        self._setup_dbnomics_subscriptions()
        
        # Service metrics
        self.metrics = {
            "requests_processed": 0,
            "requests_failed": 0,
            "avg_processing_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "series_fetched": 0,
            "providers_accessed": 0
        }
        
        # Provider information (80+ official providers)
        self.providers_info = {
            "OECD": {"name": "Organisation for Economic Co-operation and Development", "type": "international"},
            "WB": {"name": "World Bank", "type": "international"},
            "IMF": {"name": "International Monetary Fund", "type": "international"},
            "BIS": {"name": "Bank for International Settlements", "type": "central_bank"},
            "ECB": {"name": "European Central Bank", "type": "central_bank"},
            "FED": {"name": "Federal Reserve", "type": "central_bank"},
            "BOE": {"name": "Bank of England", "type": "central_bank"},
            "BOJ": {"name": "Bank of Japan", "type": "central_bank"},
            "SNB": {"name": "Swiss National Bank", "type": "central_bank"},
            "RBA": {"name": "Reserve Bank of Australia", "type": "central_bank"},
            "EUROSTAT": {"name": "European Statistical Office", "type": "statistical"},
            "INSEE": {"name": "Institut National de la Statistique (France)", "type": "statistical"},
            "DESTATIS": {"name": "German Federal Statistical Office", "type": "statistical"},
            "ONS": {"name": "Office for National Statistics (UK)", "type": "statistical"}
        }
    
    def _setup_dbnomics_subscriptions(self):
        """Setup DBnomics specific topic subscriptions"""
        # High priority for real-time data requests
        self.config.add_subscription(
            "dbnomics.series.*",
            MessagePriority.HIGH,
            filters={"real_time": "true"}
        )
        
        # Normal priority for batch data requests
        self.config.add_subscription(
            "dbnomics.datasets.*",
            MessagePriority.NORMAL
        )
        
        # Critical priority for health and system status
        self.config.add_subscription(
            "dbnomics.health.*",
            MessagePriority.CRITICAL
        )
        
        # Low priority for metadata and discovery
        self.config.add_subscription(
            "dbnomics.providers.*",
            MessagePriority.LOW
        )
        
        # Normal priority for search operations
        self.config.add_subscription(
            "dbnomics.search.*",
            MessagePriority.NORMAL
        )
    
    async def start(self):
        """Start the enhanced DBnomics MessageBus service"""
        logger.info("Starting Enhanced DBnomics MessageBus Service")
        
        try:
            # Initialize MessageBus components
            self.messagebus_client = BufferedMessageBusClient(self.config)
            self.stream_manager = RedisStreamManager(self.config)
            
            await self.messagebus_client.connect()
            await self.stream_manager.connect()
            
            # Setup request handlers
            self._register_handlers()
            
            # Start processing loops
            self.running = True
            asyncio.create_task(self._message_processing_loop())
            asyncio.create_task(self._health_monitoring_loop())
            asyncio.create_task(self._cache_cleanup_loop())
            
            logger.info("Enhanced DBnomics MessageBus Service started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start Enhanced DBnomics MessageBus Service: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the service and cleanup resources"""
        logger.info("Stopping Enhanced DBnomics MessageBus Service")
        
        self.running = False
        
        if self.messagebus_client:
            await self.messagebus_client.close()
            
        if self.stream_manager:
            await self.stream_manager.close()
        
        logger.info("Enhanced DBnomics MessageBus Service stopped")
    
    def _register_handlers(self):
        """Register message handlers for different request types"""
        self.request_handlers = {
            "dbnomics.series.fetch": self._handle_series_fetch,
            "dbnomics.series.search": self._handle_series_search,
            "dbnomics.datasets.list": self._handle_datasets_list,
            "dbnomics.providers.list": self._handle_providers_list,
            "dbnomics.providers.info": self._handle_provider_info,
            "dbnomics.health.check": self._handle_health_check,
            "dbnomics.search.global": self._handle_global_search,
            "dbnomics.meta.categories": self._handle_categories,
            "dbnomics.analytics.trading_indicators": self._handle_trading_indicators
        }
    
    async def _message_processing_loop(self):
        """Main message processing loop"""
        logger.info("Starting DBnomics message processing loop")
        
        while self.running:
            try:
                # Subscribe to all DBnomics patterns
                for subscription in self.config.subscriptions:
                    if "dbnomics" in subscription.pattern:
                        await self.messagebus_client.subscribe(subscription.pattern)
                
                # Process incoming messages
                message = await self.messagebus_client.receive(timeout=1.0)
                
                if message:
                    await self._process_message(message)
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in DBnomics message processing loop: {e}")
                await asyncio.sleep(1)
    
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
                request = DbnomicsRequest(
                    request_id=request_id,
                    endpoint=msg_data.get('endpoint', ''),
                    params=msg_data.get('params', {}),
                    priority=MessagePriority(msg_data.get('priority', MessagePriority.NORMAL.value)),
                    callback_topic=msg_data.get('callback_topic'),
                    cache_ttl=msg_data.get('cache_ttl', 3600)
                )
                
                # Store active request
                self.active_requests[request_id] = request
                
                # Process request
                start_time = datetime.utcnow()
                response = await handler(request)
                processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                # Update response
                response.processing_time_ms = processing_time
                
                # Send response if callback specified
                if request.callback_topic:
                    await self._send_response(request.callback_topic, response)
                
                # Update metrics
                self.metrics["requests_processed"] += 1
                self.metrics["avg_processing_time"] = (
                    (self.metrics["avg_processing_time"] * (self.metrics["requests_processed"] - 1) + 
                     processing_time) / self.metrics["requests_processed"]
                )
                
                # Cleanup
                if request_id in self.active_requests:
                    del self.active_requests[request_id]
                    
            else:
                logger.warning(f"No handler found for DBnomics topic: {topic}")
                
        except Exception as e:
            logger.error(f"Error processing DBnomics message: {e}")
            self.metrics["requests_failed"] += 1
    
    async def _send_response(self, topic: str, response: DbnomicsResponse):
        """Send response via MessageBus"""
        try:
            response_data = {
                "request_id": response.request_id,
                "success": response.success,
                "data": response.data,
                "error": response.error,
                "processing_time_ms": response.processing_time_ms,
                "cached": response.cached,
                "provider_info": response.provider_info,
                "timestamp": response.timestamp.isoformat()
            }
            
            message = json.dumps(response_data, default=str).encode('utf-8')
            await self.messagebus_client.publish(topic, message)
            
        except Exception as e:
            logger.error(f"Error sending DBnomics response to {topic}: {e}")
    
    async def _handle_series_fetch(self, request: DbnomicsRequest) -> DbnomicsResponse:
        """Handle series data fetching"""
        try:
            series_code = request.params.get('series_code')
            provider_code = request.params.get('provider_code')
            dataset_code = request.params.get('dataset_code')
            
            # Check cache first
            cache_key = f"{provider_code}:{dataset_code}:{series_code}"
            if cache_key in self.data_cache:
                cache_data = self.data_cache[cache_key]
                if datetime.utcnow() - cache_data['timestamp'] < timedelta(seconds=request.cache_ttl):
                    self.metrics["cache_hits"] += 1
                    return DbnomicsResponse(
                        request_id=request.request_id,
                        success=True,
                        data=cache_data['data'],
                        cached=True,
                        provider_info=self.providers_info.get(provider_code, {})
                    )
            
            # Simulate DBnomics API call
            # In real implementation, this would use the dbnomics Python package
            mock_series_data = {
                "series_code": series_code,
                "provider_code": provider_code,
                "dataset_code": dataset_code,
                "name": f"Economic Series {series_code}",
                "description": f"Time series data for {series_code}",
                "frequency": "M",  # Monthly
                "unit": "Index",
                "data": [
                    {"period": "2023-01", "value": 100.0 + i * 0.5}
                    for i in range(24)  # 2 years of data
                ],
                "metadata": {
                    "source": provider_code,
                    "last_updated": datetime.utcnow().isoformat(),
                    "observations": 24,
                    "start_period": "2023-01",
                    "end_period": "2024-12"
                }
            }
            
            # Cache the result
            self.data_cache[cache_key] = {
                'data': mock_series_data,
                'timestamp': datetime.utcnow()
            }
            self.metrics["cache_misses"] += 1
            self.metrics["series_fetched"] += 1
            
            return DbnomicsResponse(
                request_id=request.request_id,
                success=True,
                data=mock_series_data,
                cached=False,
                provider_info=self.providers_info.get(provider_code, {})
            )
            
        except Exception as e:
            return DbnomicsResponse(
                request_id=request.request_id,
                success=False,
                error=str(e)
            )
    
    async def _handle_series_search(self, request: DbnomicsRequest) -> DbnomicsResponse:
        """Handle series search requests"""
        try:
            query = request.params.get('q', '')
            provider_code = request.params.get('provider_code')
            limit = request.params.get('limit', 20)
            
            # Mock search results
            mock_results = {
                "total": 850000,  # 850K+ series available
                "query": query,
                "provider_code": provider_code,
                "results": [
                    {
                        "series_code": f"SERIES_{i}_{query.upper()}",
                        "provider_code": provider_code or f"PROVIDER_{i%5}",
                        "dataset_code": f"DATASET_{i}",
                        "name": f"Economic Indicator {i}: {query}",
                        "description": f"Time series for {query} indicator {i}",
                        "frequency": ["M", "Q", "A"][i % 3],
                        "unit": ["Index", "Percent", "Currency"][i % 3],
                        "last_updated": (datetime.utcnow() - timedelta(days=i)).isoformat(),
                        "observations": 100 + i * 10,
                        "trading_relevance": 0.6 + (i * 0.1) % 0.4
                    }
                    for i in range(min(limit, 20))
                ]
            }
            
            return DbnomicsResponse(
                request_id=request.request_id,
                success=True,
                data=mock_results
            )
            
        except Exception as e:
            return DbnomicsResponse(
                request_id=request.request_id,
                success=False,
                error=str(e)
            )
    
    async def _handle_providers_list(self, request: DbnomicsRequest) -> DbnomicsResponse:
        """Handle providers list request"""
        try:
            # Return comprehensive provider information
            providers_data = {
                "total": len(self.providers_info),
                "providers": [
                    {
                        "code": code,
                        "name": info["name"],
                        "type": info["type"],
                        "dataset_count": 1000 + hash(code) % 5000,  # Mock dataset counts
                        "series_count": 10000 + hash(code) % 50000,  # Mock series counts
                        "last_updated": datetime.utcnow().isoformat()
                    }
                    for code, info in self.providers_info.items()
                ],
                "by_type": {
                    "central_bank": len([p for p in self.providers_info.values() if p["type"] == "central_bank"]),
                    "statistical": len([p for p in self.providers_info.values() if p["type"] == "statistical"]),
                    "international": len([p for p in self.providers_info.values() if p["type"] == "international"])
                }
            }
            
            self.metrics["providers_accessed"] += 1
            
            return DbnomicsResponse(
                request_id=request.request_id,
                success=True,
                data=providers_data
            )
            
        except Exception as e:
            return DbnomicsResponse(
                request_id=request.request_id,
                success=False,
                error=str(e)
            )
    
    async def _handle_health_check(self, request: DbnomicsRequest) -> DbnomicsResponse:
        """Handle health check requests"""
        try:
            health_status = {
                "status": "healthy",
                "api_available": True,
                "messagebus_connected": self.messagebus_client.is_connected() if self.messagebus_client else False,
                "redis_connected": self.stream_manager.is_connected() if self.stream_manager else False,
                "active_requests": len(self.active_requests),
                "cache_size": len(self.data_cache),
                "providers_available": len(self.providers_info),
                "metrics": self.metrics.copy(),
                "last_check": datetime.utcnow().isoformat()
            }
            
            return DbnomicsResponse(
                request_id=request.request_id,
                success=True,
                data=health_status
            )
            
        except Exception as e:
            return DbnomicsResponse(
                request_id=request.request_id,
                success=False,
                error=str(e)
            )
    
    async def _handle_trading_indicators(self, request: DbnomicsRequest) -> DbnomicsResponse:
        """Handle trading indicators analysis"""
        try:
            analysis_type = request.params.get('type', 'macro_economic')
            
            trading_indicators = {
                "macro_economic": {
                    "gdp_growth": {"provider": "OECD", "series": "GDP_GROWTH_Q", "relevance": 0.95},
                    "inflation_rate": {"provider": "OECD", "series": "CPI_TOTAL", "relevance": 0.90},
                    "unemployment": {"provider": "OECD", "series": "UNEMPLOYMENT_RATE", "relevance": 0.85},
                    "interest_rates": {"provider": "ECB", "series": "INTEREST_RATE", "relevance": 0.95},
                    "money_supply": {"provider": "FED", "series": "M2_MONEY_SUPPLY", "relevance": 0.80}
                },
                "central_bank": {
                    "policy_rate": {"provider": "ECB", "series": "POLICY_RATE", "relevance": 0.98},
                    "balance_sheet": {"provider": "FED", "series": "BALANCE_SHEET", "relevance": 0.85},
                    "reserves": {"provider": "BOE", "series": "BANK_RESERVES", "relevance": 0.75},
                    "lending_rate": {"provider": "BOJ", "series": "LENDING_RATE", "relevance": 0.80}
                },
                "financial": {
                    "yield_curve": {"provider": "BIS", "series": "YIELD_CURVE_10Y", "relevance": 0.90},
                    "credit_spread": {"provider": "BIS", "series": "CREDIT_SPREAD", "relevance": 0.85},
                    "vix_equivalent": {"provider": "BIS", "series": "VOLATILITY_INDEX", "relevance": 0.95}
                }
            }
            
            result = {
                "analysis_type": analysis_type,
                "indicators": trading_indicators.get(analysis_type, trading_indicators["macro_economic"]),
                "total_indicators": len(trading_indicators.get(analysis_type, {})),
                "average_relevance": 0.87,
                "last_updated": datetime.utcnow().isoformat(),
                "data_availability": "real_time"
            }
            
            return DbnomicsResponse(
                request_id=request.request_id,
                success=True,
                data=result
            )
            
        except Exception as e:
            return DbnomicsResponse(
                request_id=request.request_id,
                success=False,
                error=str(e)
            )
    
    async def _cache_cleanup_loop(self):
        """Cleanup expired cache entries"""
        while self.running:
            try:
                current_time = datetime.utcnow()
                expired_keys = []
                
                for key, cache_data in self.data_cache.items():
                    if current_time - cache_data['timestamp'] > timedelta(hours=1):  # 1 hour expiry
                        expired_keys.append(key)
                
                for key in expired_keys:
                    del self.data_cache[key]
                
                if expired_keys:
                    logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in cache cleanup: {e}")
                await asyncio.sleep(60)
    
    async def _health_monitoring_loop(self):
        """Monitor service health and publish status"""
        while self.running:
            try:
                health_metrics = {
                    "service": "enhanced_dbnomics_messagebus",
                    "status": "healthy" if self.running else "stopped",
                    "active_requests": len(self.active_requests),
                    "cache_size": len(self.data_cache),
                    "metrics": self.metrics.copy(),
                    "messagebus_connected": self.messagebus_client.is_connected() if self.messagebus_client else False,
                    "redis_connected": self.stream_manager.is_connected() if self.stream_manager else False,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                await self.messagebus_client.publish(
                    "system.health.dbnomics_messagebus",
                    json.dumps(health_metrics).encode('utf-8')
                )
                
                await asyncio.sleep(self.config.health_check_interval)
                
            except Exception as e:
                logger.error(f"Error in DBnomics health monitoring: {e}")
                await asyncio.sleep(5)
    
    async def send_request(self, topic: str, endpoint: str, params: Dict[str, Any], 
                          priority: MessagePriority = MessagePriority.NORMAL,
                          callback_topic: Optional[str] = None,
                          cache_ttl: int = 3600) -> str:
        """Send a request via MessageBus"""
        request_id = f"dbnomics_req_{datetime.utcnow().timestamp()}_{hash(topic)}"
        
        request_data = {
            "topic": topic,
            "request_id": request_id,
            "endpoint": endpoint,
            "params": params,
            "priority": priority.value,
            "callback_topic": callback_topic,
            "cache_ttl": cache_ttl,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        message = json.dumps(request_data).encode('utf-8')
        await self.messagebus_client.publish(topic, message)
        
        return request_id
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive service metrics"""
        return {
            "service_metrics": self.metrics.copy(),
            "active_requests": len(self.active_requests),
            "cache_metrics": {
                "size": len(self.data_cache),
                "hit_rate": self.metrics["cache_hits"] / max(1, self.metrics["cache_hits"] + self.metrics["cache_misses"])
            },
            "providers_metrics": {
                "available_providers": len(self.providers_info),
                "providers_accessed": self.metrics["providers_accessed"]
            },
            "messagebus_metrics": self.messagebus_client.get_metrics() if self.messagebus_client else {},
            "stream_metrics": self.stream_manager.get_metrics() if self.stream_manager else {}
        }

# Convenience functions
async def create_test_dbnomics_service() -> EnhancedDbnomicsMessageBusService:
    """Create a test DBnomics service instance"""
    config = ConfigPresets.development()
    service = EnhancedDbnomicsMessageBusService(config)
    await service.start()
    return service

async def run_dbnomics_test_scenario():
    """Run DBnomics test scenario"""
    service = await create_test_dbnomics_service()
    
    try:
        # Test various DBnomics requests
        requests = [
            ("dbnomics.series.fetch", "/api/v1/dbnomics/series", 
             {"provider_code": "OECD", "dataset_code": "EO", "series_code": "GDP_GROWTH"}),
            ("dbnomics.providers.list", "/api/v1/dbnomics/providers", {}),
            ("dbnomics.series.search", "/api/v1/dbnomics/series/search", {"q": "inflation"}),
            ("dbnomics.health.check", "/api/v1/dbnomics/health", {}),
            ("dbnomics.analytics.trading_indicators", "/api/v1/dbnomics/trading-indicators", 
             {"type": "macro_economic"})
        ]
        
        request_ids = []
        for topic, endpoint, params in requests:
            req_id = await service.send_request(
                topic, endpoint, params,
                callback_topic=f"{topic}.response"
            )
            request_ids.append(req_id)
            logger.info(f"Sent DBnomics request {req_id} to {topic}")
        
        # Wait for processing
        await asyncio.sleep(3)
        
        # Get metrics
        metrics = service.get_metrics()
        logger.info(f"DBnomics service metrics: {metrics}")
        
        return metrics
        
    finally:
        await service.stop()

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_dbnomics_test_scenario())