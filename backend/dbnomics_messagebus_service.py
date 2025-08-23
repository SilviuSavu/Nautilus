"""
DBnomics MessageBus Service
Provides event-driven integration with DBnomics economic data via MessageBus.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

import dbnomics
import pandas as pd
from messagebus_client import MessageBusMessage, get_messagebus_client

logger = logging.getLogger(__name__)


@dataclass
class DBnomicsEventTypes:
    """Event types for DBnomics MessageBus integration"""
    HEALTH_CHECK = "dbnomics.health_check"
    PROVIDERS_REQUEST = "dbnomics.providers.request"
    SERIES_SEARCH = "dbnomics.series.search"
    SERIES_REQUEST = "dbnomics.series.request"
    STATISTICS_REQUEST = "dbnomics.statistics.request"
    
    # Response events
    HEALTH_RESPONSE = "dbnomics.health.response"
    PROVIDERS_RESPONSE = "dbnomics.providers.response"
    SERIES_DATA = "dbnomics.series.data"
    STATISTICS_RESPONSE = "dbnomics.statistics.response"
    ERROR_RESPONSE = "dbnomics.error"


class DBnomicsMessageBusService:
    """
    DBnomics MessageBus service that handles economic data requests via event-driven architecture.
    """
    
    def __init__(self):
        self.messagebus_client = get_messagebus_client()
        self._running = False
        self._handlers: Dict[str, callable] = {}
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup event handlers for DBnomics events"""
        self._handlers = {
            DBnomicsEventTypes.HEALTH_CHECK: self._handle_health_check,
            DBnomicsEventTypes.PROVIDERS_REQUEST: self._handle_providers_request,
            DBnomicsEventTypes.SERIES_SEARCH: self._handle_series_search,
            DBnomicsEventTypes.SERIES_REQUEST: self._handle_series_request,
            DBnomicsEventTypes.STATISTICS_REQUEST: self._handle_statistics_request,
        }
    
    async def start(self):
        """Start the DBnomics MessageBus service"""
        logger.info("Starting DBnomics MessageBus service")
        self._running = True
        
        # Register message handlers with MessageBus client
        for event_type, handler in self._handlers.items():
            self.messagebus_client.add_message_handler(self._create_message_filter(event_type, handler))
        
        logger.info("DBnomics MessageBus service started")
    
    async def stop(self):
        """Stop the DBnomics MessageBus service"""
        logger.info("Stopping DBnomics MessageBus service")
        self._running = False
        logger.info("DBnomics MessageBus service stopped")
    
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
            
            # Publish through MessageBus (this would send to Redis streams)
            await self._publish_to_messagebus(response_message)
            
        except Exception as e:
            logger.error(f"Failed to publish response: {e}")
    
    async def _publish_error(self, original_message: MessageBusMessage, error: str):
        """Publish an error response"""
        await self._publish_response(
            original_message,
            DBnomicsEventTypes.ERROR_RESPONSE,
            {
                "error": error,
                "original_topic": original_message.topic
            }
        )
    
    async def _publish_to_messagebus(self, message: MessageBusMessage):
        """Publish message to MessageBus - integrate with your existing Redis stream"""
        # This would integrate with your existing MessageBus publishing mechanism
        # For now, we'll log it and broadcast to WebSocket clients
        logger.info(f"Publishing DBnomics event: {message.topic}")
        
        # Broadcast to WebSocket clients (following your existing pattern)
        try:
            from main import broadcast_market_data  # Import your existing broadcast function
            await broadcast_market_data({
                "type": "dbnomics_event",
                "topic": message.topic,
                "data": message.payload,
                "timestamp": message.timestamp
            })
        except ImportError:
            logger.warning("Could not import broadcast_market_data - WebSocket integration disabled")
    
    # Event Handlers
    
    async def _handle_health_check(self, message: MessageBusMessage):
        """Handle DBnomics health check requests"""
        logger.info("Processing DBnomics health check")
        
        try:
            # Test DBnomics connection
            test_data = dbnomics.fetch_series(
                series_ids=["IMF/IFS/A.AD.BOP_BP6_GG."],
                max_nb_series=1
            )
            
            health_status = {
                "status": "healthy" if test_data is not None and not test_data.empty else "degraded",
                "api_available": True,
                "providers": 80,
                "message": "DBnomics API operational"
            }
            
        except Exception as e:
            logger.error(f"DBnomics health check failed: {e}")
            health_status = {
                "status": "error",
                "api_available": False,
                "message": f"DBnomics API unavailable: {str(e)}"
            }
        
        await self._publish_response(
            message,
            DBnomicsEventTypes.HEALTH_RESPONSE,
            health_status
        )
    
    async def _handle_providers_request(self, message: MessageBusMessage):
        """Handle providers list requests"""
        logger.info("Processing DBnomics providers request")
        
        try:
            # Static provider list (in full implementation, query DBnomics metadata API)
            providers = [
                {
                    "code": "IMF",
                    "name": "International Monetary Fund",
                    "website": "https://www.imf.org",
                    "data_count": 50000,
                    "description": "Global economic and financial statistics"
                },
                {
                    "code": "OECD", 
                    "name": "Organisation for Economic Co-operation and Development",
                    "website": "https://www.oecd.org",
                    "data_count": 100000,
                    "description": "Economic statistics and indicators for OECD countries"
                },
                {
                    "code": "ECB",
                    "name": "European Central Bank", 
                    "website": "https://www.ecb.europa.eu",
                    "data_count": 25000,
                    "description": "Eurozone monetary and financial statistics"
                },
                # Add more providers...
            ]
            
            await self._publish_response(
                message,
                DBnomicsEventTypes.PROVIDERS_RESPONSE,
                {"providers": providers}
            )
            
        except Exception as e:
            logger.error(f"Failed to fetch providers: {e}")
            await self._publish_error(message, f"Failed to fetch providers: {str(e)}")
    
    async def _handle_series_search(self, message: MessageBusMessage):
        """Handle series search requests"""
        logger.info("Processing DBnomics series search")
        
        try:
            payload = message.payload
            
            # Extract search parameters
            provider_code = payload.get("provider_code")
            dataset_code = payload.get("dataset_code")
            series_code = payload.get("series_code")
            dimensions = payload.get("dimensions")
            max_nb_series = payload.get("max_nb_series", 50)
            
            # Build DBnomics fetch parameters
            fetch_params = {"max_nb_series": max_nb_series}
            
            if provider_code:
                fetch_params["provider_code"] = provider_code
            if dataset_code:
                fetch_params["dataset_code"] = dataset_code  
            if series_code:
                fetch_params["series_code"] = series_code
            if dimensions:
                fetch_params["dimensions"] = dimensions
            
            # Fetch data from DBnomics
            df = dbnomics.fetch_series(**fetch_params)
            
            if df is None or df.empty:
                series_list = []
            else:
                series_list = await self._process_dataframe_to_series(df, limit=max_nb_series)
            
            await self._publish_response(
                message,
                DBnomicsEventTypes.SERIES_DATA,
                {"series": series_list}
            )
            
        except Exception as e:
            logger.error(f"Failed to search series: {e}")
            await self._publish_error(message, f"Failed to search series: {str(e)}")
    
    async def _handle_series_request(self, message: MessageBusMessage):
        """Handle specific series data requests"""
        logger.info("Processing DBnomics series request")
        
        try:
            payload = message.payload
            
            # Extract series identifier
            provider_code = payload.get("provider_code")
            dataset_code = payload.get("dataset_code")
            series_code = payload.get("series_code")
            
            if not all([provider_code, dataset_code, series_code]):
                await self._publish_error(message, "Missing required parameters: provider_code, dataset_code, series_code")
                return
            
            # Build series ID
            series_id = f"{provider_code}/{dataset_code}/{series_code}"
            
            # Fetch data from DBnomics
            df = dbnomics.fetch_series(series_ids=[series_id], max_nb_series=1)
            
            if df is None or df.empty:
                await self._publish_error(message, f"Series not found: {series_id}")
                return
            
            series_data = await self._process_dataframe_to_series(df, limit=1)
            
            await self._publish_response(
                message,
                DBnomicsEventTypes.SERIES_DATA,
                {"series": series_data[0] if series_data else None}
            )
            
        except Exception as e:
            logger.error(f"Failed to get series: {e}")
            await self._publish_error(message, f"Failed to get series: {str(e)}")
    
    async def _handle_statistics_request(self, message: MessageBusMessage):
        """Handle statistics requests"""
        logger.info("Processing DBnomics statistics request")
        
        try:
            statistics = {
                "total_providers": 80,
                "total_datasets": 1200,
                "total_series": 800000000,
                "last_update": datetime.utcnow().isoformat(),
                "top_providers": [
                    {"name": "OECD", "series_count": 100000},
                    {"name": "IMF", "series_count": 50000},
                    {"name": "World Bank", "series_count": 60000},
                    {"name": "ECB", "series_count": 25000},
                    {"name": "EUROSTAT", "series_count": 75000}
                ]
            }
            
            await self._publish_response(
                message,
                DBnomicsEventTypes.STATISTICS_RESPONSE,
                statistics
            )
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            await self._publish_error(message, f"Failed to get statistics: {str(e)}")
    
    async def _process_dataframe_to_series(self, df: pd.DataFrame, limit: int = None) -> List[Dict[str, Any]]:
        """Convert DBnomics DataFrame to series data structure"""
        series_list = []
        
        # Group by unique series
        if 'series_code' in df.columns:
            series_codes = df['series_code'].unique()
            if limit:
                series_codes = series_codes[:limit]
            
            for series_code in series_codes:
                series_df = df[df['series_code'] == series_code]
                first_row = series_df.iloc[0]
                
                # Extract data points
                data_points = []
                if 'period' in series_df.columns and 'value' in series_df.columns:
                    for _, row in series_df.iterrows():
                        if not pd.isna(row.get('value')):
                            data_points.append({
                                "period": str(row['period']),
                                "value": float(row['value']) if pd.notna(row['value']) else None
                            })
                
                # Create series info
                series_info = {
                    "provider_code": first_row.get('provider_code', ''),
                    "dataset_code": first_row.get('dataset_code', ''),
                    "series_code": str(series_code),
                    "series_name": first_row.get('series_name', ''),
                    "frequency": first_row.get('freq', ''),
                    "unit": first_row.get('unit', ''),
                    "last_update": str(first_row.get('last_update', '')),
                    "first_date": str(series_df['period'].min()) if 'period' in series_df.columns else None,
                    "last_date": str(series_df['period'].max()) if 'period' in series_df.columns else None,
                    "observations_count": len(data_points),
                    "data": data_points[:100]  # Limit data points to prevent large messages
                }
                
                series_list.append(series_info)
        
        return series_list


# Global service instance
dbnomics_messagebus_service = DBnomicsMessageBusService()


# Request helper functions for API routes
async def request_dbnomics_health(correlation_id: str = None):
    """Request DBnomics health check via MessageBus"""
    message = MessageBusMessage(
        topic=DBnomicsEventTypes.HEALTH_CHECK,
        payload={"correlation_id": correlation_id or f"health_{int(datetime.utcnow().timestamp())}"},
        timestamp=int(datetime.utcnow().timestamp() * 1000),
        message_type="request"
    )
    
    await dbnomics_messagebus_service._publish_to_messagebus(message)
    return message.payload["correlation_id"]


async def request_dbnomics_providers(correlation_id: str = None):
    """Request DBnomics providers via MessageBus"""
    message = MessageBusMessage(
        topic=DBnomicsEventTypes.PROVIDERS_REQUEST,
        payload={"correlation_id": correlation_id or f"providers_{int(datetime.utcnow().timestamp())}"},
        timestamp=int(datetime.utcnow().timestamp() * 1000),
        message_type="request"
    )
    
    await dbnomics_messagebus_service._publish_to_messagebus(message)
    return message.payload["correlation_id"]


async def request_dbnomics_series_search(
    provider_code: str = None,
    dataset_code: str = None, 
    series_code: str = None,
    dimensions: Dict = None,
    max_nb_series: int = 50,
    correlation_id: str = None
):
    """Request DBnomics series search via MessageBus"""
    message = MessageBusMessage(
        topic=DBnomicsEventTypes.SERIES_SEARCH,
        payload={
            "correlation_id": correlation_id or f"search_{int(datetime.utcnow().timestamp())}",
            "provider_code": provider_code,
            "dataset_code": dataset_code,
            "series_code": series_code,
            "dimensions": dimensions,
            "max_nb_series": max_nb_series
        },
        timestamp=int(datetime.utcnow().timestamp() * 1000),
        message_type="request"
    )
    
    await dbnomics_messagebus_service._publish_to_messagebus(message)
    return message.payload["correlation_id"]