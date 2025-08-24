#!/usr/bin/env python3
"""
Risk Engine - Simplified orchestrator for risk management services
"""

import asyncio
import logging
import os
import time
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI
import uvicorn

from enhanced_messagebus_client import BufferedMessageBusClient, MessagePriority, EnhancedMessageBusConfig
from clock import Clock, create_clock
from models import RiskLimit, RiskBreach, RiskLimitType, BreachSeverity
from services import RiskCalculationService, RiskMonitoringService, RiskAnalyticsService
from routes import setup_routes


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskEngine:
    """
    Simplified Risk Engine orchestrator
    Coordinates services and manages lifecycle
    """
    
    def __init__(self, clock: Optional[Clock] = None):
        # Clock setup
        self._clock = clock if clock is not None else create_clock("live")
        
        self.app = FastAPI(title="Nautilus Risk Engine", version="1.0.0")
        self.start_time = self._clock.timestamp()
        
        # Services
        self.calculation_service = RiskCalculationService()
        self.analytics_service = RiskAnalyticsService()
        self.monitoring_service = None  # Initialized after messagebus
        
        # MessageBus
        self.messagebus = None
        
        # State tracking
        self.ml_model_loaded = False
        self.event_processing_metrics = {
            "portfolio_events_processed": 0,
            "analytics_requests_processed": 0,
            "optimization_requests_processed": 0,
            "average_processing_time_ms": 0.0,
            "events_per_minute": 0.0,
            "high_priority_events": 0,
            "critical_events": 0,
            "processing_errors": 0
        }
        self.priority_queues = {}
        
        # Setup lifespan
        self.app = FastAPI(
            title="Nautilus Risk Engine",
            version="1.0.0",
            lifespan=self.lifespan
        )
        
        # Routes will be setup after engine start
    
    @property
    def clock(self) -> Clock:
        """Get the clock instance"""
        return self._clock
        
    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        """FastAPI lifespan management"""
        # Startup
        await self.start_engine()
        yield
        # Shutdown
        await self.stop_engine()
    
    async def start_engine(self):
        """Start the risk engine with all services"""
        try:
            logger.info("Starting Risk Engine...")
            
            # Initialize MessageBus
            messagebus_config = EnhancedMessageBusConfig(
                redis_host="redis",
                redis_port=6379,
                consumer_name="risk-engine",
                stream_key="nautilus-risk-streams",
                consumer_group="risk-group",
                buffer_interval_ms=50,
                max_buffer_size=20000,
                heartbeat_interval_secs=30,
                clock=self._clock  # Pass the clock instance
            )
            
            self.messagebus = BufferedMessageBusClient(messagebus_config)
            await self.messagebus.start()
            
            # Initialize monitoring service (needs messagebus)
            self.monitoring_service = RiskMonitoringService(
                self.calculation_service, 
                self.messagebus
            )
            
            # Initialize analytics services
            await self.analytics_service.initialize()
            
            # Setup message handlers
            await self._setup_message_handlers()
            
            # Start monitoring
            await self.monitoring_service.start_monitoring()
            
            # Now setup routes with initialized services
            setup_routes(
                self.app,
                self.calculation_service,
                self.monitoring_service,
                self.analytics_service,
                self.messagebus,
                self.start_time,
                self.event_processing_metrics,
                self.priority_queues,
                self.ml_model_loaded
            )
            
            logger.info("Risk Engine started successfully")
            
        except Exception as e:
            logger.error(f"Risk Engine startup failed: {e}")
            raise
    
    async def stop_engine(self):
        """Stop the risk engine"""
        logger.info("Stopping Risk Engine...")
        
        if self.monitoring_service:
            await self.monitoring_service.stop_monitoring()
        
        if self.messagebus:
            await self.messagebus.stop()
            
        logger.info("Risk Engine stopped")
    
    async def _setup_message_handlers(self):
        """Setup MessageBus event handlers"""
        if not self.messagebus:
            return
            
        # Handler for portfolio events
        async def handle_portfolio_event(message):
            """Handle portfolio events"""
            try:
                # Check if message topic matches pattern
                if hasattr(message, 'topic') and message.topic.startswith("portfolio"):
                    message_data = message.payload
                    portfolio_id = message_data.get("portfolio_id")
                    position_data = message_data.get("position_data", {})
                    
                    # Perform risk check
                    breaches = self.calculation_service.check_position_risk(portfolio_id, position_data)
                    
                    # Update metrics
                    self.event_processing_metrics["portfolio_events_processed"] += 1
                    
                    if breaches:
                        # Publish breach alerts
                        for breach in breaches:
                            await self.messagebus.publish(
                                "risk.breach.detected",
                                {
                                    "portfolio_id": portfolio_id,
                                    "breach_id": breach.breach_id,
                                    "severity": breach.severity.value,
                                    "breach_percentage": breach.breach_percentage
                                },
                                priority=MessagePriority.URGENT if breach.severity == BreachSeverity.CRITICAL else MessagePriority.HIGH
                            )
                            
                            if breach.severity == BreachSeverity.CRITICAL:
                                self.event_processing_metrics["critical_events"] += 1
                            else:
                                self.event_processing_metrics["high_priority_events"] += 1
                                
            except Exception as e:
                logger.error(f"Portfolio event handling error: {e}")
                self.event_processing_metrics["processing_errors"] += 1
        
        # Handler for analytics requests
        async def handle_analytics_request(message):
            """Handle analytics requests"""
            try:
                # Check if message topic matches pattern
                if hasattr(message, 'topic') and message.topic.startswith("analytics.request"):
                    message_data = message.payload
                    request_type = message_data.get("type")
                    portfolio_id = message_data.get("portfolio_id")
                    
                    if request_type == "hybrid_analytics":
                        result = await self.analytics_service.compute_hybrid_analytics(
                            portfolio_id, message_data
                        )
                        
                        # Publish results
                        await self.messagebus.publish(
                            "risk.analytics.computed",
                            {
                                "portfolio_id": portfolio_id,
                                "type": "hybrid_analytics",
                                "result": result
                            },
                            priority=MessagePriority.NORMAL
                        )
                    
                    self.event_processing_metrics["analytics_requests_processed"] += 1
                
            except Exception as e:
                logger.error(f"Analytics request handling error: {e}")
                self.event_processing_metrics["processing_errors"] += 1
        
        # Register handlers with MessageBus
        self.messagebus.add_message_handler(handle_portfolio_event)
        self.messagebus.add_message_handler(handle_analytics_request)


# Create FastAPI app with lifespan
risk_engine = RiskEngine()

if __name__ == "__main__":
    # Configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8200"))
    
    logger.info(f"Starting Risk Engine on {host}:{port}")
    
    uvicorn.run(
        risk_engine.app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )