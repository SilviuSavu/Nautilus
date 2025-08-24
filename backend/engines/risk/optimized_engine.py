#!/usr/bin/env python3
"""
Optimized Risk Engine - High-performance version with async processing and caching
"""

import asyncio
import logging
import os
import time
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
import uvicorn

from enhanced_messagebus_client import BufferedMessageBusClient, MessagePriority, EnhancedMessageBusConfig
from clock import Clock, create_clock
from models import RiskLimit, RiskBreach, RiskLimitType, BreachSeverity
from optimized_services import OptimizedRiskCalculationService, OptimizedRiskMonitoringService
from routes import setup_routes


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizedRiskEngine:
    """
    High-performance Risk Engine with optimized processing
    Features: async calculations, intelligent caching, batch processing
    """
    
    def __init__(self, clock: Optional[Clock] = None):
        # Clock setup
        self._clock = clock if clock is not None else create_clock("live")
        
        # Core FastAPI app
        self.app = FastAPI(
            title="Nautilus Risk Engine (Optimized)",
            version="2.0.0",
            description="High-performance risk management with async processing and intelligent caching",
            lifespan=self.lifespan
        )
        self.start_time = self._clock.timestamp()
        
        # Optimized services
        self.calculation_service = OptimizedRiskCalculationService()
        self.monitoring_service = None  # Initialized after messagebus
        self.analytics_service = None   # Placeholder for future enhancement
        
        # MessageBus
        self.messagebus = None
        
        # Performance tracking
        self.performance_metrics = {
            "total_requests": 0,
            "avg_response_time_ms": 0.0,
            "requests_per_second": 0.0,
            "cache_hit_ratio": 0.0,
            "concurrent_calculations": 0,
            "optimization_level": "high"
        }
        
        # State tracking (optimized)
        self.event_processing_metrics = {
            "portfolio_events_processed": 0,
            "analytics_requests_processed": 0,
            "optimization_requests_processed": 0,
            "average_processing_time_ms": 0.0,
            "events_per_minute": 0.0,
            "high_priority_events": 0,
            "critical_events": 0,
            "processing_errors": 0,
            "batch_processed_events": 0,
            "cache_hits": 0
        }
        
        # Priority queues for message handling
        self.priority_queues = {
            "urgent": asyncio.Queue(maxsize=1000),
            "high": asyncio.Queue(maxsize=2000),
            "normal": asyncio.Queue(maxsize=5000),
            "batch": asyncio.Queue(maxsize=10000)
        }
        
        # Message processing workers
        self.message_workers = []
        self.ml_model_loaded = False
        
        # Performance monitoring
        self.request_times = []
        self.last_metrics_update = time.time()
        
    @property
    def clock(self) -> Clock:
        """Get the clock instance"""
        return self._clock
    
    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        """FastAPI lifespan management with optimized startup/shutdown"""
        # Startup
        await self.start_engine()
        yield
        # Shutdown
        await self.stop_engine()
    
    async def start_engine(self):
        """Start the optimized risk engine"""
        try:
            logger.info("Starting Optimized Risk Engine...")
            
            # Initialize optimized calculation service
            await self.calculation_service.initialize()
            
            # Initialize MessageBus with optimizations
            messagebus_config = EnhancedMessageBusConfig(
                redis_host="redis",
                redis_port=6379,
                consumer_name="optimized-risk-engine",
                stream_key="nautilus-risk-streams",
                consumer_group="optimized-risk-group",
                buffer_interval_ms=25,  # Reduced for better responsiveness
                max_buffer_size=50000,  # Increased buffer size
                heartbeat_interval_secs=15,  # More frequent heartbeats
                clock=self._clock
            )
            
            self.messagebus = BufferedMessageBusClient(messagebus_config)
            await self.messagebus.start()
            
            # Initialize optimized monitoring service
            self.monitoring_service = OptimizedRiskMonitoringService(
                self.calculation_service, 
                self.messagebus
            )
            
            # Start message processing workers
            await self._start_message_workers()
            
            # Start monitoring
            await self.monitoring_service.start_monitoring()
            
            # Setup routes with optimized services
            setup_routes(
                self.app,
                self.calculation_service,
                self.monitoring_service,
                None,  # analytics_service placeholder
                self.messagebus,
                self.start_time,
                self.event_processing_metrics,
                self.priority_queues,
                self.ml_model_loaded
            )
            
            # Add performance monitoring endpoints
            self._add_performance_endpoints()
            
            # Start performance metrics collection
            asyncio.create_task(self._performance_monitoring_loop())
            
            logger.info("Optimized Risk Engine started successfully")
            
        except Exception as e:
            logger.error(f"Optimized Risk Engine startup failed: {e}")
            raise
    
    async def stop_engine(self):
        """Stop the optimized risk engine"""
        logger.info("Stopping Optimized Risk Engine...")
        
        # Stop message workers
        for worker in self.message_workers:
            worker.cancel()
            
        # Stop monitoring service
        if self.monitoring_service:
            await self.monitoring_service.stop_monitoring()
        
        # Stop MessageBus
        if self.messagebus:
            await self.messagebus.stop()
            
        logger.info("Optimized Risk Engine stopped")
    
    async def _start_message_workers(self):
        """Start optimized message processing workers"""
        
        # High-priority workers (critical risk events)
        for i in range(3):
            worker = asyncio.create_task(
                self._priority_message_worker("urgent", max_processing_time=50)
            )
            self.message_workers.append(worker)
        
        # Medium-priority workers (normal risk checks)
        for i in range(5):
            worker = asyncio.create_task(
                self._priority_message_worker("high", max_processing_time=100)
            )
            self.message_workers.append(worker)
        
        # Normal priority workers
        for i in range(3):
            worker = asyncio.create_task(
                self._priority_message_worker("normal", max_processing_time=200)
            )
            self.message_workers.append(worker)
        
        # Batch processing worker
        batch_worker = asyncio.create_task(self._batch_processing_worker())
        self.message_workers.append(batch_worker)
        
        # Setup message handlers with queue routing
        await self._setup_optimized_message_handlers()
    
    async def _priority_message_worker(self, priority_level: str, max_processing_time: int):
        """Worker for processing messages from priority queues"""
        
        queue = self.priority_queues[priority_level]
        
        while True:
            try:
                message = await queue.get()
                start_time = time.time()
                
                # Process message
                await self._process_priority_message(message)
                
                # Track processing time
                processing_time = (time.time() - start_time) * 1000
                
                # Alert if processing is too slow
                if processing_time > max_processing_time:
                    logger.warning(
                        f"{priority_level.upper()} priority message took {processing_time:.2f}ms "
                        f"(limit: {max_processing_time}ms)"
                    )
                
                # Update metrics
                self._update_processing_metrics(processing_time)
                
                queue.task_done()
                
            except Exception as e:
                logger.error(f"{priority_level.upper()} worker error: {e}")
                self.event_processing_metrics["processing_errors"] += 1
    
    async def _batch_processing_worker(self):
        """Worker for batch processing of portfolio updates"""
        
        batch = []
        batch_size = 50
        batch_timeout = 0.1  # 100ms
        last_process_time = time.time()
        
        while True:
            try:
                # Collect messages for batch
                try:
                    message = await asyncio.wait_for(
                        self.priority_queues["batch"].get(),
                        timeout=batch_timeout
                    )
                    batch.append(message)
                except asyncio.TimeoutError:
                    pass
                
                current_time = time.time()
                
                # Process batch if full or timeout reached
                if (len(batch) >= batch_size or 
                    (batch and current_time - last_process_time >= batch_timeout)):
                    
                    start_time = current_time
                    await self._process_message_batch(batch)
                    
                    # Update metrics
                    processing_time = (time.time() - start_time) * 1000
                    self.event_processing_metrics["batch_processed_events"] += len(batch)
                    self._update_processing_metrics(processing_time / len(batch) if batch else 0)
                    
                    # Mark tasks as done
                    for _ in batch:
                        self.priority_queues["batch"].task_done()
                    
                    batch.clear()
                    last_process_time = current_time
                
            except Exception as e:
                logger.error(f"Batch worker error: {e}")
                self.event_processing_metrics["processing_errors"] += 1
    
    async def _setup_optimized_message_handlers(self):
        """Setup message handlers with intelligent routing"""
        
        async def route_message(message):
            """Route incoming messages to appropriate priority queue"""
            try:
                message_data = message.payload
                message_type = message_data.get('type', '')
                priority = message_data.get('priority', 'normal')
                
                # Route to appropriate queue based on message characteristics
                if (priority == 'urgent' or 
                    message_type in ['position_limit_breach', 'critical_risk_event', 'margin_call']):
                    await self.priority_queues["urgent"].put(message)
                    
                elif (priority == 'high' or
                      message_type in ['risk_limit_check', 'portfolio_risk_assessment']):
                    await self.priority_queues["high"].put(message)
                    
                elif message_type in ['portfolio_update', 'position_change', 'market_data_update']:
                    await self.priority_queues["batch"].put(message)
                    
                else:
                    await self.priority_queues["normal"].put(message)
                    
            except Exception as e:
                logger.error(f"Message routing error: {e}")
        
        # Register the router as the main message handler
        self.messagebus.add_message_handler(route_message)
    
    async def _process_priority_message(self, message):
        """Process high/medium priority messages immediately"""
        
        message_data = message.payload
        message_type = message_data.get('type', '')
        
        if message.topic.startswith("portfolio") or message_type == 'risk_limit_check':
            # Process portfolio risk check
            portfolio_id = message_data.get("portfolio_id")
            position_data = message_data.get("position_data", {})
            
            if portfolio_id:
                # Use optimized async risk calculation
                breaches = await self.calculation_service.check_position_risk_async(
                    portfolio_id, position_data
                )
                
                self.event_processing_metrics["portfolio_events_processed"] += 1
                
                # Publish breach alerts if any
                if breaches:
                    for breach in breaches:
                        priority = (MessagePriority.URGENT if breach.severity == BreachSeverity.CRITICAL 
                                  else MessagePriority.HIGH)
                        
                        await self.messagebus.publish(
                            "risk.breach.detected",
                            {
                                "portfolio_id": portfolio_id,
                                "breach_id": breach.breach_id,
                                "severity": breach.severity.value,
                                "breach_percentage": breach.breach_percentage,
                                "current_value": breach.current_value,
                                "limit_value": breach.limit_value
                            },
                            priority=priority
                        )
                        
                        if breach.severity == BreachSeverity.CRITICAL:
                            self.event_processing_metrics["critical_events"] += 1
                        else:
                            self.event_processing_metrics["high_priority_events"] += 1
        
        elif message.topic.startswith("analytics") or message_type == 'analytics_request':
            # Handle analytics requests (placeholder for future enhancement)
            self.event_processing_metrics["analytics_requests_processed"] += 1
    
    async def _process_message_batch(self, messages):
        """Process batch of messages efficiently"""
        
        # Group messages by type for efficient processing
        portfolio_updates = []
        position_changes = []
        
        for message in messages:
            message_data = message.payload
            message_type = message_data.get('type', '')
            
            if message_type == 'portfolio_update':
                portfolio_updates.append(message_data)
            elif message_type == 'position_change':
                position_changes.append(message_data)
        
        # Process portfolio updates in batch
        if portfolio_updates:
            await self._process_portfolio_updates_batch(portfolio_updates)
        
        # Process position changes in batch
        if position_changes:
            await self._process_position_changes_batch(position_changes)
    
    async def _process_portfolio_updates_batch(self, updates):
        """Process multiple portfolio updates efficiently"""
        
        # Group by portfolio_id for efficient processing
        portfolio_groups = {}
        for update in updates:
            portfolio_id = update.get('portfolio_id')
            if portfolio_id:
                if portfolio_id not in portfolio_groups:
                    portfolio_groups[portfolio_id] = []
                portfolio_groups[portfolio_id].append(update)
        
        # Process each portfolio group
        tasks = []
        for portfolio_id, group_updates in portfolio_groups.items():
            # Combine position data for the portfolio
            combined_position_data = self._combine_position_data(group_updates)
            
            # Create async task for risk check
            task = self.calculation_service.check_position_risk_async(
                portfolio_id, combined_position_data
            )
            tasks.append((portfolio_id, task))
        
        # Execute all tasks in parallel
        for portfolio_id, task in tasks:
            try:
                breaches = await task
                if breaches:
                    # Handle breaches (similar to priority message processing)
                    await self._handle_risk_breaches(portfolio_id, breaches)
            except Exception as e:
                logger.error(f"Batch portfolio update error for {portfolio_id}: {e}")
    
    async def _process_position_changes_batch(self, changes):
        """Process multiple position changes efficiently"""
        
        # Similar batching logic for position changes
        for change in changes:
            portfolio_id = change.get('portfolio_id')
            if portfolio_id:
                # Quick risk check for position changes
                position_data = change.get('position_data', {})
                breaches = await self.calculation_service.check_position_risk_async(
                    portfolio_id, position_data
                )
                
                if breaches:
                    await self._handle_risk_breaches(portfolio_id, breaches)
    
    def _combine_position_data(self, updates) -> Dict[str, Any]:
        """Combine multiple position updates into single position data"""
        
        combined = {
            "positions": [],
            "prices": [],
            "market_value": 0.0,
            "unrealized_pnl": 0.0,
            "leverage": 1.0
        }
        
        for update in updates:
            position_data = update.get('position_data', {})
            
            # Aggregate position data
            combined["market_value"] += position_data.get("market_value", 0.0)
            combined["unrealized_pnl"] += position_data.get("unrealized_pnl", 0.0)
            
            if "positions" in position_data:
                combined["positions"].extend(position_data["positions"])
            if "prices" in position_data:
                combined["prices"].extend(position_data["prices"])
        
        return combined
    
    async def _handle_risk_breaches(self, portfolio_id: str, breaches: List[RiskBreach]):
        """Handle detected risk breaches"""
        
        for breach in breaches:
            priority = (MessagePriority.URGENT if breach.severity == BreachSeverity.CRITICAL 
                       else MessagePriority.HIGH)
            
            await self.messagebus.publish(
                "risk.breach.detected",
                {
                    "portfolio_id": portfolio_id,
                    "breach_id": breach.breach_id,
                    "severity": breach.severity.value,
                    "breach_percentage": breach.breach_percentage
                },
                priority=priority
            )
    
    def _update_processing_metrics(self, processing_time_ms: float):
        """Update processing performance metrics"""
        
        # Update average processing time (exponential moving average)
        current_avg = self.event_processing_metrics["average_processing_time_ms"]
        self.event_processing_metrics["average_processing_time_ms"] = (
            0.9 * current_avg + 0.1 * processing_time_ms
        )
        
        # Track individual processing times for performance analysis
        self.request_times.append(processing_time_ms)
        
        # Keep only recent times (last 1000 requests)
        if len(self.request_times) > 1000:
            self.request_times = self.request_times[-1000:]
    
    async def _performance_monitoring_loop(self):
        """Background loop for performance monitoring"""
        
        while True:
            try:
                current_time = time.time()
                time_since_last = current_time - self.last_metrics_update
                
                if time_since_last >= 60:  # Update every minute
                    await self._update_performance_metrics()
                    self.last_metrics_update = current_time
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _update_performance_metrics(self):
        """Update comprehensive performance metrics"""
        
        # Get calculation service performance stats
        calc_stats = self.calculation_service.get_performance_stats()
        
        # Update overall performance metrics
        self.performance_metrics.update({
            "cache_hit_ratio": calc_stats["cache_performance"]["hit_ratio_percent"],
            "avg_response_time_ms": calc_stats["calculation_performance"]["avg_calculation_time_ms"],
            "total_requests": calc_stats["service_metrics"]["risk_checks_processed"],
            "concurrent_calculations": calc_stats["calculation_performance"]["parallel_calculations"]
        })
        
        # Calculate requests per second
        events_processed = self.event_processing_metrics["portfolio_events_processed"]
        uptime_seconds = time.time() - self.start_time
        self.performance_metrics["requests_per_second"] = events_processed / max(uptime_seconds, 1)
        
        # Update events per minute
        self.event_processing_metrics["events_per_minute"] = self.performance_metrics["requests_per_second"] * 60
    
    def _add_performance_endpoints(self):
        """Add performance monitoring endpoints"""
        
        @self.app.get("/performance/metrics")
        async def get_performance_metrics():
            """Get comprehensive performance metrics"""
            
            await self._update_performance_metrics()
            
            return {
                "engine_performance": self.performance_metrics,
                "processing_metrics": self.event_processing_metrics,
                "calculation_performance": self.calculation_service.get_performance_stats(),
                "queue_status": {
                    queue_name: queue.qsize() 
                    for queue_name, queue in self.priority_queues.items()
                },
                "worker_status": {
                    "active_workers": len([w for w in self.message_workers if not w.done()]),
                    "total_workers": len(self.message_workers)
                }
            }
        
        @self.app.get("/performance/cache")
        async def get_cache_performance():
            """Get detailed cache performance metrics"""
            return self.calculation_service.cache.get_cache_stats()
        
        @self.app.post("/performance/clear-cache")
        async def clear_performance_cache():
            """Clear performance cache for testing"""
            self.calculation_service.cache.memory_cache.clear()
            if self.calculation_service.cache.redis_cache:
                await self.calculation_service.cache.redis_cache.flushdb()
            return {"status": "cache_cleared"}


# Create optimized engine instance
optimized_risk_engine = OptimizedRiskEngine()

if __name__ == "__main__":
    # Configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8200"))
    
    logger.info(f"Starting Optimized Risk Engine on {host}:{port}")
    
    uvicorn.run(
        optimized_risk_engine.app,
        host=host,
        port=port,
        log_level="info",
        access_log=True,
        workers=1  # Single worker for async processing
    )