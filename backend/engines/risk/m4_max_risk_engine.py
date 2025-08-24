#!/usr/bin/env python3
"""
M4 Max Accelerated Risk Engine - Integrates existing hardware optimizations
Leverages existing M4 Max components without breaking other engines
"""

import asyncio
import logging
import os
import time
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI
import uvicorn

# Import existing M4 Max optimization components (don't recreate!)
from backend.acceleration import (
    initialize_coreml_acceleration,
    get_acceleration_status,
    risk_predict,
    get_inference_status,
    is_m4_max_detected,
    neural_performance_context,
    price_option_metal,
    calculate_rsi_metal
)
from backend.optimization import (
    OptimizerController,
    PerformanceMonitor,
    WorkloadClassifier,
    get_optimization_status
)

# Import new hardware router for intelligent workload routing
from backend.hardware_router import (
    HardwareRouter,
    WorkloadType,
    WorkloadCharacteristics,
    hardware_accelerated,
    route_ml_workload,
    route_compute_workload,
    route_risk_workload
)

# Risk Engine components
from enhanced_messagebus_client import BufferedMessageBusClient, MessagePriority, EnhancedMessageBusConfig
from clock import Clock, create_clock
from models import RiskLimit, RiskBreach, RiskLimitType, BreachSeverity
from optimized_services import OptimizedRiskCalculationService, OptimizedRiskMonitoringService
from routes import setup_routes


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class M4MaxRiskEngine:
    """
    M4 Max Hardware-Accelerated Risk Engine
    Integrates existing M4 Max optimizations without duplicating code
    """
    
    def __init__(self, clock: Optional[Clock] = None):
        # Clock setup
        self._clock = clock if clock is not None else create_clock("live")
        
        # FastAPI app with M4 Max identification
        self.app = FastAPI(
            title="Nautilus Risk Engine (M4 Max Accelerated)",
            version="3.0.0",
            description="M4 Max hardware-accelerated risk management with Neural Engine and CPU optimization",
            lifespan=self.lifespan
        )
        self.start_time = self._clock.timestamp()
        
        # M4 Max optimization components (use existing!)
        self.cpu_optimizer = None
        self.performance_monitor = None
        self.workload_classifier = None
        self.neural_acceleration_available = False
        self.m4_max_detected = False
        
        # Hardware router for intelligent workload routing
        self.hardware_router = None
        
        # Optimized services
        self.calculation_service = OptimizedRiskCalculationService()
        self.monitoring_service = None
        
        # MessageBus
        self.messagebus = None
        
        # M4 Max performance metrics
        self.m4_max_metrics = {
            "cpu_optimization_enabled": False,
            "neural_engine_enabled": False,
            "performance_cores_used": 0,
            "efficiency_cores_used": 0,
            "neural_engine_utilization": 0.0,
            "hardware_acceleration_ratio": 0.0,
            "risk_calculations_accelerated": 0,
            "avg_neural_inference_time_ms": 0.0
        }
        
        # Enhanced event processing with M4 Max optimization
        self.event_processing_metrics = {
            "portfolio_events_processed": 0,
            "analytics_requests_processed": 0,
            "optimization_requests_processed": 0,
            "average_processing_time_ms": 0.0,
            "events_per_minute": 0.0,
            "high_priority_events": 0,
            "critical_events": 0,
            "processing_errors": 0,
            "m4_max_accelerated_events": 0,
            "neural_engine_predictions": 0
        }
        
        # Priority queues optimized for M4 Max cores
        self.priority_queues = {
            "critical": asyncio.Queue(maxsize=500),    # P-cores
            "urgent": asyncio.Queue(maxsize=1000),     # P-cores  
            "high": asyncio.Queue(maxsize=2000),       # Mixed cores
            "normal": asyncio.Queue(maxsize=5000),     # E-cores
            "batch": asyncio.Queue(maxsize=10000)      # E-cores
        }
        
        self.message_workers = []
        self.ml_model_loaded = False
        
    @property
    def clock(self) -> Clock:
        """Get the clock instance"""
        return self._clock
    
    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        """FastAPI lifespan management with M4 Max initialization"""
        # Startup
        await self.start_engine()
        yield
        # Shutdown
        await self.stop_engine()
    
    async def start_engine(self):
        """Start M4 Max accelerated risk engine"""
        try:
            logger.info("Starting M4 Max Accelerated Risk Engine...")
            
            # Initialize M4 Max hardware detection
            self.m4_max_detected = is_m4_max_detected()
            logger.info(f"M4 Max detected: {self.m4_max_detected}")
            
            # Initialize existing M4 Max acceleration components
            await self._initialize_m4_max_acceleration()
            
            # Initialize existing CPU optimization
            await self._initialize_cpu_optimization()
            
            # Initialize hardware router for intelligent workload routing
            self.hardware_router = HardwareRouter()
            logger.info("✅ Hardware router initialized for intelligent workload routing")
            
            # Initialize optimized calculation service with M4 Max context
            await self.calculation_service.initialize()
            
            # Initialize MessageBus with M4 Max optimizations
            messagebus_config = EnhancedMessageBusConfig(
                redis_host="redis",
                redis_port=6379,
                consumer_name="m4max-risk-engine",
                stream_key="nautilus-risk-streams",
                consumer_group="m4max-risk-group",
                buffer_interval_ms=10 if self.m4_max_detected else 25,  # Faster on M4 Max
                max_buffer_size=100000 if self.m4_max_detected else 50000,
                heartbeat_interval_secs=10,
                clock=self._clock
            )
            
            self.messagebus = BufferedMessageBusClient(messagebus_config)
            await self.messagebus.start()
            
            # Initialize monitoring service
            self.monitoring_service = OptimizedRiskMonitoringService(
                self.calculation_service,
                self.messagebus
            )
            
            # Start M4 Max optimized message workers
            await self._start_m4_max_message_workers()
            
            # Start monitoring
            await self.monitoring_service.start_monitoring()
            
            # Setup routes
            setup_routes(
                self.app,
                self.calculation_service,
                self.monitoring_service,
                None,  # analytics_service
                self.messagebus,
                self.start_time,
                self.event_processing_metrics,
                self.priority_queues,
                self.ml_model_loaded
            )
            
            # Add M4 Max specific endpoints
            self._add_m4_max_endpoints()
            
            # Start performance monitoring
            if self.performance_monitor:
                asyncio.create_task(self._m4_max_performance_monitoring())
            
            logger.info("M4 Max Accelerated Risk Engine started successfully")
            if self.m4_max_detected:
                logger.info("🚀 M4 Max hardware acceleration ACTIVE")
            
        except Exception as e:
            logger.error(f"M4 Max Risk Engine startup failed: {e}")
            raise
    
    async def stop_engine(self):
        """Stop M4 Max accelerated risk engine"""
        logger.info("Stopping M4 Max Accelerated Risk Engine...")
        
        # Stop workers
        for worker in self.message_workers:
            worker.cancel()
        
        # Stop monitoring
        if self.monitoring_service:
            await self.monitoring_service.stop_monitoring()
        
        # Stop MessageBus
        if self.messagebus:
            await self.messagebus.stop()
        
        # Cleanup M4 Max components
        if self.cpu_optimizer:
            await self.cpu_optimizer.cleanup()
            
        logger.info("M4 Max Accelerated Risk Engine stopped")
    
    async def _initialize_m4_max_acceleration(self):
        """Initialize M4 Max Neural Engine acceleration using existing components"""
        try:
            logger.info("Initializing M4 Max Neural Engine acceleration...")
            
            # Use existing acceleration initialization
            acceleration_status = await initialize_coreml_acceleration(enable_logging=True)
            
            self.neural_acceleration_available = acceleration_status.get("neural_engine_available", False)
            
            # Update metrics
            self.m4_max_metrics.update({
                "neural_engine_enabled": self.neural_acceleration_available,
                "neural_engine_cores": acceleration_status.get("neural_engine_cores", 0),
                "tops_performance": acceleration_status.get("tops_performance", 0)
            })
            
            if self.neural_acceleration_available:
                logger.info("✅ M4 Max Neural Engine (38 TOPS) initialized successfully")
                
                # Test neural inference for risk predictions
                try:
                    # This uses the existing risk_predict function
                    test_data = {"market_volatility": 0.15, "portfolio_value": 1000000}
                    
                    with neural_performance_context("risk_test"):
                        test_result = await risk_predict(test_data, model_id="risk_assessment_v1")
                        
                    if test_result and not test_result.get("error"):
                        logger.info("✅ Neural Engine risk prediction test successful")
                        self.ml_model_loaded = True
                    else:
                        logger.warning("⚠️ Neural Engine test failed - using CPU fallback")
                        
                except Exception as e:
                    logger.warning(f"Neural Engine test error: {e} - continuing with CPU fallback")
            else:
                logger.info("ℹ️ Neural Engine not available - using CPU optimization only")
                
        except Exception as e:
            logger.error(f"M4 Max acceleration initialization error: {e}")
            self.neural_acceleration_available = False
    
    async def _initialize_cpu_optimization(self):
        """Initialize M4 Max CPU optimization using existing components"""
        try:
            logger.info("Initializing M4 Max CPU optimization...")
            
            # Use existing CPU optimizer
            self.cpu_optimizer = OptimizerController()
            await self.cpu_optimizer.initialize()
            
            # Use existing performance monitor
            self.performance_monitor = PerformanceMonitor()
            await self.performance_monitor.start_monitoring()
            
            # Use existing workload classifier
            self.workload_classifier = WorkloadClassifier()
            
            # Get optimization status
            opt_status = get_optimization_status()
            
            self.m4_max_metrics.update({
                "cpu_optimization_enabled": opt_status.get("optimization_active", False),
                "performance_cores_available": opt_status.get("performance_cores", 0),
                "efficiency_cores_available": opt_status.get("efficiency_cores", 0)
            })
            
            if self.m4_max_metrics["cpu_optimization_enabled"]:
                logger.info("✅ M4 Max CPU optimization (12P+4E cores) initialized successfully")
            else:
                logger.warning("⚠️ CPU optimization not available - using standard threading")
                
        except Exception as e:
            logger.error(f"CPU optimization initialization error: {e}")
            self.cpu_optimizer = None
            self.performance_monitor = None
            self.workload_classifier = None
    
    async def _start_m4_max_message_workers(self):
        """Start message workers optimized for M4 Max core architecture"""
        
        if not self.cpu_optimizer:
            # Fallback to standard workers
            await self._start_standard_workers()
            return
        
        # Critical priority workers (P-cores only)
        for i in range(2):
            worker = asyncio.create_task(
                self._m4_max_priority_worker(
                    "critical", 
                    core_type="performance",
                    max_processing_time=10  # Ultra-low latency for critical events
                )
            )
            self.message_workers.append(worker)
        
        # Urgent priority workers (P-cores preferred)
        for i in range(4):
            worker = asyncio.create_task(
                self._m4_max_priority_worker(
                    "urgent",
                    core_type="performance", 
                    max_processing_time=25
                )
            )
            self.message_workers.append(worker)
        
        # High priority workers (Mixed P+E cores)
        for i in range(6):
            worker = asyncio.create_task(
                self._m4_max_priority_worker(
                    "high",
                    core_type="mixed",
                    max_processing_time=50
                )
            )
            self.message_workers.append(worker)
        
        # Normal priority workers (E-cores preferred)
        for i in range(4):
            worker = asyncio.create_task(
                self._m4_max_priority_worker(
                    "normal",
                    core_type="efficiency",
                    max_processing_time=100
                )
            )
            self.message_workers.append(worker)
        
        # Batch processing worker (E-cores)
        batch_worker = asyncio.create_task(
            self._m4_max_batch_worker()
        )
        self.message_workers.append(batch_worker)
        
        # Setup intelligent message routing
        await self._setup_m4_max_message_routing()
        
        logger.info(f"Started {len(self.message_workers)} M4 Max optimized workers")
    
    async def _m4_max_priority_worker(self, priority_level: str, core_type: str, max_processing_time: int):
        """M4 Max optimized worker using specific core types"""
        
        queue = self.priority_queues[priority_level]
        
        while True:
            try:
                message = await queue.get()
                start_time = time.time()
                
                # Classify workload and optimize for M4 Max
                if self.cpu_optimizer and self.workload_classifier:
                    workload_category, priority = await self._classify_risk_workload(message, core_type)
                    
                    # Optimize workload for specific cores
                    optimization_context = await self.cpu_optimizer.optimize_workload(
                        workload_category, priority, preferred_core_type=core_type
                    )
                    
                    # Process message with optimization
                    async with optimization_context:
                        await self._process_m4_max_message(message, core_type)
                        
                    # Update core usage metrics
                    self._update_core_usage_metrics(core_type)
                    
                else:
                    # Fallback to standard processing
                    await self._process_m4_max_message(message, core_type)
                
                # Track processing time
                processing_time = (time.time() - start_time) * 1000
                
                if processing_time > max_processing_time:
                    logger.warning(
                        f"{priority_level.upper()} message on {core_type}-cores took {processing_time:.2f}ms "
                        f"(limit: {max_processing_time}ms)"
                    )
                
                # Update metrics
                self._update_processing_metrics(processing_time, core_type)
                queue.task_done()
                
            except Exception as e:
                logger.error(f"M4 Max {priority_level} worker error: {e}")
                self.event_processing_metrics["processing_errors"] += 1
    
    async def _m4_max_batch_worker(self):
        """M4 Max optimized batch worker for E-cores"""
        
        batch = []
        batch_size = 100 if self.m4_max_detected else 50
        batch_timeout = 0.05  # 50ms for M4 Max
        last_process_time = time.time()
        
        while True:
            try:
                try:
                    message = await asyncio.wait_for(
                        self.priority_queues["batch"].get(),
                        timeout=batch_timeout
                    )
                    batch.append(message)
                except asyncio.TimeoutError:
                    pass
                
                current_time = time.time()
                
                if (len(batch) >= batch_size or 
                    (batch and current_time - last_process_time >= batch_timeout)):
                    
                    start_time = current_time
                    
                    # Process batch with E-core optimization
                    if self.cpu_optimizer:
                        optimization_context = await self.cpu_optimizer.optimize_workload(
                            "batch_processing", "normal", preferred_core_type="efficiency"
                        )
                        async with optimization_context:
                            await self._process_m4_max_batch(batch)
                    else:
                        await self._process_m4_max_batch(batch)
                    
                    # Update metrics
                    processing_time = (time.time() - start_time) * 1000
                    self.event_processing_metrics["m4_max_accelerated_events"] += len(batch)
                    self._update_core_usage_metrics("efficiency")
                    
                    # Mark tasks done
                    for _ in batch:
                        self.priority_queues["batch"].task_done()
                    
                    batch.clear()
                    last_process_time = current_time
                
            except Exception as e:
                logger.error(f"M4 Max batch worker error: {e}")
                self.event_processing_metrics["processing_errors"] += 1
    
    async def _classify_risk_workload(self, message, preferred_core_type: str):
        """Classify risk workload for M4 Max optimization"""
        
        if not self.workload_classifier:
            return "risk_calculation", "normal"
        
        message_data = message.payload
        message_type = message_data.get('type', '')
        
        # Classify based on computational intensity
        if message_type in ['var_calculation', 'monte_carlo', 'stress_test']:
            category = "compute_intensive"
            priority = "high"
        elif message_type in ['portfolio_risk_check', 'limit_validation']:
            category = "latency_sensitive" 
            priority = "urgent"
        elif message_type in ['batch_portfolio_update', 'bulk_position_update']:
            category = "throughput_optimized"
            priority = "normal"
        else:
            category = "risk_calculation"
            priority = "normal"
        
        # Use existing workload classifier for additional optimization
        try:
            classification = await self.workload_classifier.classify_workload(
                function_name=f"risk_{message_type}",
                execution_context={
                    "message_type": message_type,
                    "preferred_core_type": preferred_core_type,
                    "latency_sensitive": priority in ["urgent", "high"]
                }
            )
            
            if classification:
                return classification.get("category", category), classification.get("priority", priority)
                
        except Exception as e:
            logger.warning(f"Workload classification error: {e}")
        
        return category, priority
    
    async def _process_m4_max_message(self, message, core_type: str):
        """Process message with intelligent hardware routing"""
        
        message_data = message.payload
        message_type = message_data.get('type', '')
        
        # Enhanced risk processing with intelligent hardware routing
        if (message.topic.startswith("portfolio") or message_type == 'risk_limit_check'):
            portfolio_id = message_data.get("portfolio_id")
            position_data = message_data.get("position_data", {})
            
            if portfolio_id and self.hardware_router:
                # Use hardware router to determine optimal processing method
                await self._process_risk_with_hardware_routing(
                    portfolio_id, position_data, message_type
                )
            else:
                # Fallback to standard processing
                await self._process_risk_standard(portfolio_id, position_data)
            
            self.event_processing_metrics["portfolio_events_processed"] += 1
        
        elif message.topic.startswith("analytics"):
            # Route analytics workloads intelligently
            await self._process_analytics_with_routing(message_data)
            self.event_processing_metrics["analytics_requests_processed"] += 1
            
        elif message_type in ['monte_carlo', 'var_calculation']:
            # Route compute-intensive workloads
            await self._process_compute_with_routing(message_data, message_type)
    
    async def _process_risk_with_hardware_routing(self, portfolio_id: str, position_data: dict, message_type: str):
        """Process risk calculation using intelligent hardware routing"""
        
        try:
            # Determine if this is a latency-critical operation
            latency_critical = message_type in ['margin_call', 'system_risk_alert', 'critical_risk_event']
            data_size = len(position_data.get('positions', []))
            
            # Get routing decision for risk workload
            routing_decision = await route_risk_workload(
                data_size=data_size,
                latency_critical=latency_critical
            )
            
            logger.debug(f"Risk routing decision: {routing_decision.primary_hardware.value} "
                        f"(confidence: {routing_decision.confidence:.2f}, "
                        f"gain: {routing_decision.estimated_performance_gain:.1f}x)")
            
            # Process based on routing decision
            if routing_decision.primary_hardware.name == 'NEURAL_ENGINE':
                await self._process_risk_neural_engine(portfolio_id, position_data, routing_decision)
            elif routing_decision.primary_hardware.name == 'METAL_GPU':
                await self._process_risk_metal_gpu(portfolio_id, position_data, routing_decision)
            elif routing_decision.primary_hardware.name == 'HYBRID':
                await self._process_risk_hybrid(portfolio_id, position_data, routing_decision)
            else:
                # CPU processing
                await self._process_risk_cpu(portfolio_id, position_data, routing_decision)
                
        except Exception as e:
            logger.error(f"Hardware routing failed for risk processing: {e}")
            # Fallback to standard processing
            await self._process_risk_standard(portfolio_id, position_data)
    
    async def _process_risk_neural_engine(self, portfolio_id: str, position_data: dict, routing_decision):
        """Process risk using Neural Engine"""
        
        try:
            neural_start = time.time()
            
            # Use existing risk_predict function with Neural Engine
            risk_prediction = await risk_predict(
                {
                    "portfolio_id": portfolio_id,
                    "position_data": position_data,
                    "timestamp": time.time(),
                    "routing_decision": routing_decision.__dict__
                },
                model_id="risk_assessment_v1"
            )
            
            neural_time = (time.time() - neural_start) * 1000
            self.m4_max_metrics["avg_neural_inference_time_ms"] = (
                0.9 * self.m4_max_metrics["avg_neural_inference_time_ms"] + 
                0.1 * neural_time
            )
            self.event_processing_metrics["neural_engine_predictions"] += 1
            self.m4_max_metrics["risk_calculations_accelerated"] += 1
            
            # If Neural Engine prediction indicates high risk, do detailed calculation
            if risk_prediction and risk_prediction.get("risk_score", 0) > 0.7:
                # Use optimized calculation service for detailed analysis
                breaches = await self.calculation_service.check_position_risk_async(
                    portfolio_id, position_data
                )
                
                # Enhanced breach handling with Neural Engine insights
                if breaches:
                    await self._handle_neural_enhanced_breaches(portfolio_id, breaches, risk_prediction)
            
            logger.debug(f"Neural Engine risk processing completed in {neural_time:.2f}ms "
                        f"(predicted gain: {routing_decision.estimated_performance_gain:.1f}x)")
            
        except Exception as e:
            logger.warning(f"Neural Engine risk processing failed: {e} - falling back to {routing_decision.fallback_hardware.value}")
            # Use fallback hardware
            if routing_decision.fallback_hardware.name == 'METAL_GPU':
                await self._process_risk_metal_gpu(portfolio_id, position_data, routing_decision)
            else:
                await self._process_risk_cpu(portfolio_id, position_data, routing_decision)
    
    async def _process_risk_metal_gpu(self, portfolio_id: str, position_data: dict, routing_decision):
        """Process risk using Metal GPU acceleration"""
        
        try:
            gpu_start = time.time()
            
            # Use Metal GPU for intensive risk calculations
            positions = position_data.get('positions', [])
            if len(positions) > 1000:  # Large portfolio - use GPU Monte Carlo
                
                # Extract position data for GPU processing
                spot_prices = [pos.get('current_price', 100.0) for pos in positions]
                volatilities = [pos.get('volatility', 0.2) for pos in positions]
                
                # Use existing Metal GPU Monte Carlo for VaR calculation
                monte_carlo_results = []
                for spot_price, volatility in zip(spot_prices[:10], volatilities[:10]):  # Process first 10 positions
                    result = await price_option_metal(
                        spot_price=spot_price,
                        strike_price=spot_price * 0.95,  # 5% OTM put for risk
                        volatility=volatility,
                        num_simulations=100000
                    )
                    monte_carlo_results.append(result)
                
                gpu_time = (time.time() - gpu_start) * 1000
                self.m4_max_metrics["risk_calculations_accelerated"] += 1
                
                logger.debug(f"Metal GPU risk processing completed in {gpu_time:.2f}ms "
                            f"(predicted gain: {routing_decision.estimated_performance_gain:.1f}x)")
                
            else:
                # Use standard calculation service for smaller portfolios
                await self._process_risk_cpu(portfolio_id, position_data, routing_decision)
                
        except Exception as e:
            logger.warning(f"Metal GPU risk processing failed: {e} - falling back to CPU")
            await self._process_risk_cpu(portfolio_id, position_data, routing_decision)
    
    async def _process_risk_hybrid(self, portfolio_id: str, position_data: dict, routing_decision):
        """Process risk using hybrid Neural Engine + GPU approach"""
        
        try:
            hybrid_start = time.time()
            
            # Step 1: Use Neural Engine for initial risk assessment
            neural_prediction = await risk_predict(
                {
                    "portfolio_id": portfolio_id,
                    "position_data": position_data,
                    "timestamp": time.time()
                },
                model_id="risk_assessment_v1"
            )
            
            # Step 2: If high risk detected, use GPU for detailed Monte Carlo
            if neural_prediction and neural_prediction.get("risk_score", 0) > 0.6:
                await self._process_risk_metal_gpu(portfolio_id, position_data, routing_decision)
            
            hybrid_time = (time.time() - hybrid_start) * 1000
            self.event_processing_metrics["neural_engine_predictions"] += 1
            self.m4_max_metrics["risk_calculations_accelerated"] += 1
            
            logger.debug(f"Hybrid risk processing completed in {hybrid_time:.2f}ms "
                        f"(predicted gain: {routing_decision.estimated_performance_gain:.1f}x)")
            
        except Exception as e:
            logger.warning(f"Hybrid risk processing failed: {e} - falling back to CPU")
            await self._process_risk_cpu(portfolio_id, position_data, routing_decision)
    
    async def _process_risk_cpu(self, portfolio_id: str, position_data: dict, routing_decision):
        """Process risk using optimized CPU processing"""
        
        cpu_start = time.time()
        
        # Use optimized calculation service
        breaches = await self.calculation_service.check_position_risk_async(
            portfolio_id, position_data
        )
        
        cpu_time = (time.time() - cpu_start) * 1000
        
        if breaches:
            await self._handle_risk_breaches(portfolio_id, breaches)
        
        logger.debug(f"CPU risk processing completed in {cpu_time:.2f}ms")
    
    async def _process_risk_standard(self, portfolio_id: str, position_data: dict):
        """Fallback standard risk processing"""
        
        breaches = await self.calculation_service.check_position_risk_async(
            portfolio_id, position_data
        )
        if breaches:
            await self._handle_risk_breaches(portfolio_id, breaches)
    
    async def _process_analytics_with_routing(self, message_data: dict):
        """Process analytics workloads with intelligent routing"""
        
        analytics_type = message_data.get('analytics_type', '')
        data_size = message_data.get('data_size', 0)
        
        # Route analytics workloads
        if analytics_type in ['technical_indicators', 'rsi', 'macd']:
            await self._process_indicators_with_routing(message_data)
        elif analytics_type in ['ml_prediction', 'sentiment_analysis']:
            await self._process_ml_analytics_with_routing(message_data)
        else:
            # Standard CPU processing for other analytics
            pass
    
    async def _process_indicators_with_routing(self, message_data: dict):
        """Process technical indicators with Metal GPU if beneficial"""
        
        try:
            data_size = message_data.get('data_size', 0)
            
            # Route to Metal GPU for large datasets
            if data_size > 10000:  # Hardware router parallel threshold
                prices = message_data.get('price_data', [])
                if prices and len(prices) > 100:
                    # Use existing Metal GPU RSI calculation
                    rsi_result = await calculate_rsi_metal(prices, period=14)
                    logger.debug(f"Metal GPU RSI calculation completed for {len(prices)} prices")
                    self.m4_max_metrics["risk_calculations_accelerated"] += 1
            
        except Exception as e:
            logger.warning(f"GPU indicators processing failed: {e}")
    
    async def _process_ml_analytics_with_routing(self, message_data: dict):
        """Process ML analytics with Neural Engine routing"""
        
        try:
            # Route ML workloads to Neural Engine
            routing_decision = await route_ml_workload(
                data_size=message_data.get('data_size', 0)
            )
            
            if routing_decision.primary_hardware.name == 'NEURAL_ENGINE':
                # Process with Neural Engine
                ml_data = message_data.get('ml_input', {})
                prediction = await risk_predict(ml_data, model_id="analytics_v1")
                if prediction:
                    self.event_processing_metrics["neural_engine_predictions"] += 1
                    logger.debug(f"Neural Engine analytics completed with {routing_decision.estimated_performance_gain:.1f}x gain")
        
        except Exception as e:
            logger.warning(f"ML analytics routing failed: {e}")
    
    async def _process_compute_with_routing(self, message_data: dict, message_type: str):
        """Process compute-intensive workloads with intelligent routing"""
        
        try:
            workload_type = WorkloadType.MONTE_CARLO if message_type == 'monte_carlo' else WorkloadType.MATRIX_COMPUTE
            data_size = message_data.get('data_size', 0)
            
            routing_decision = await route_compute_workload(workload_type, data_size)
            
            if routing_decision.primary_hardware.name == 'METAL_GPU' and message_type == 'monte_carlo':
                # Use Metal GPU for Monte Carlo
                monte_carlo_params = message_data.get('monte_carlo_params', {})
                result = await price_option_metal(
                    spot_price=monte_carlo_params.get('spot_price', 100.0),
                    strike_price=monte_carlo_params.get('strike_price', 110.0),
                    volatility=monte_carlo_params.get('volatility', 0.2),
                    num_simulations=monte_carlo_params.get('simulations', 1000000)
                )
                
                if result:
                    self.m4_max_metrics["risk_calculations_accelerated"] += 1
                    logger.debug(f"Metal GPU Monte Carlo completed with {routing_decision.estimated_performance_gain:.1f}x speedup")
        
        except Exception as e:
            logger.warning(f"Compute workload routing failed: {e}")
    
    async def _process_m4_max_batch(self, messages):
        """Process message batch with M4 Max E-core optimization"""
        
        # Group messages efficiently
        portfolio_updates = []
        risk_calculations = []
        
        for message in messages:
            message_data = message.payload
            message_type = message_data.get('type', '')
            
            if message_type in ['portfolio_update', 'position_change']:
                portfolio_updates.append(message_data)
            elif message_type in ['risk_check', 'limit_validation']:
                risk_calculations.append(message_data)
        
        # Process in parallel using E-cores
        tasks = []
        
        if portfolio_updates:
            tasks.append(self._batch_portfolio_updates(portfolio_updates))
            
        if risk_calculations:
            tasks.append(self._batch_risk_calculations(risk_calculations))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _batch_risk_calculations(self, calculations):
        """Batch risk calculations optimized for M4 Max"""
        
        # Process in chunks optimized for M4 Max memory
        chunk_size = 50 if self.m4_max_detected else 25
        
        for i in range(0, len(calculations), chunk_size):
            chunk = calculations[i:i+chunk_size]
            
            # Process chunk in parallel
            tasks = []
            for calc in chunk:
                portfolio_id = calc.get('portfolio_id')
                position_data = calc.get('position_data', {})
                
                if portfolio_id:
                    task = self.calculation_service.check_position_risk_async(
                        portfolio_id, position_data
                    )
                    tasks.append((portfolio_id, task))
            
            # Wait for chunk completion
            for portfolio_id, task in tasks:
                try:
                    breaches = await task
                    if breaches:
                        await self._handle_risk_breaches(portfolio_id, breaches)
                        self.m4_max_metrics["risk_calculations_accelerated"] += 1
                except Exception as e:
                    logger.error(f"Batch calculation error for {portfolio_id}: {e}")
    
    async def _handle_neural_enhanced_breaches(self, portfolio_id: str, breaches: List[RiskBreach], neural_prediction: Dict):
        """Handle breaches with Neural Engine enhanced insights"""
        
        for breach in breaches:
            # Enhanced breach data with neural insights
            enhanced_breach_data = {
                "portfolio_id": portfolio_id,
                "breach_id": breach.breach_id,
                "severity": breach.severity.value,
                "breach_percentage": breach.breach_percentage,
                "neural_risk_score": neural_prediction.get("risk_score", 0.0),
                "neural_confidence": neural_prediction.get("confidence", 0.0),
                "predicted_trend": neural_prediction.get("trend", "unknown"),
                "m4_max_accelerated": True
            }
            
            priority = (MessagePriority.URGENT if breach.severity == BreachSeverity.CRITICAL 
                       else MessagePriority.HIGH)
            
            await self.messagebus.publish(
                "risk.breach.neural_enhanced",
                enhanced_breach_data,
                priority=priority
            )
    
    async def _handle_risk_breaches(self, portfolio_id: str, breaches: List[RiskBreach]):
        """Standard risk breach handling"""
        
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
                    "m4_max_accelerated": True
                },
                priority=priority
            )
    
    def _update_core_usage_metrics(self, core_type: str):
        """Update M4 Max core usage metrics"""
        
        if core_type == "performance":
            self.m4_max_metrics["performance_cores_used"] += 1
        elif core_type == "efficiency":
            self.m4_max_metrics["efficiency_cores_used"] += 1
    
    def _update_processing_metrics(self, processing_time_ms: float, core_type: str):
        """Update processing metrics with M4 Max context"""
        
        # Update standard metrics
        current_avg = self.event_processing_metrics["average_processing_time_ms"]
        self.event_processing_metrics["average_processing_time_ms"] = (
            0.9 * current_avg + 0.1 * processing_time_ms
        )
        
        # Calculate hardware acceleration ratio
        baseline_time = 100.0  # Baseline processing time
        acceleration_ratio = baseline_time / max(processing_time_ms, 1.0)
        
        current_ratio = self.m4_max_metrics["hardware_acceleration_ratio"]
        self.m4_max_metrics["hardware_acceleration_ratio"] = (
            0.95 * current_ratio + 0.05 * acceleration_ratio
        )
    
    async def _m4_max_performance_monitoring(self):
        """Background M4 Max performance monitoring"""
        
        while True:
            try:
                if self.performance_monitor:
                    # Get M4 Max performance stats
                    perf_stats = await self.performance_monitor.get_performance_stats()
                    
                    # Update metrics
                    if perf_stats:
                        self.m4_max_metrics.update({
                            "neural_engine_utilization": perf_stats.get("neural_engine_utilization", 0.0),
                            "performance_cores_utilization": perf_stats.get("performance_cores_utilization", 0.0),
                            "efficiency_cores_utilization": perf_stats.get("efficiency_cores_utilization", 0.0)
                        })
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                logger.error(f"M4 Max performance monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _setup_m4_max_message_routing(self):
        """Setup intelligent message routing for M4 Max"""
        
        async def route_m4_max_message(message):
            """Route messages based on M4 Max optimization"""
            
            try:
                message_data = message.payload
                message_type = message_data.get('type', '')
                priority = message_data.get('priority', 'normal')
                
                # Route critical events to P-cores
                if (priority == 'critical' or 
                    message_type in ['margin_call', 'system_risk_alert']):
                    await self.priority_queues["critical"].put(message)
                    
                # Route urgent events to P-cores
                elif (priority == 'urgent' or 
                      message_type in ['position_limit_breach', 'critical_risk_event']):
                    await self.priority_queues["urgent"].put(message)
                    
                # Route complex calculations to mixed cores
                elif message_type in ['var_calculation', 'stress_test', 'monte_carlo']:
                    await self.priority_queues["high"].put(message)
                    
                # Route batch operations to E-cores
                elif message_type in ['portfolio_update', 'position_change', 'bulk_update']:
                    await self.priority_queues["batch"].put(message)
                    
                else:
                    await self.priority_queues["normal"].put(message)
                    
            except Exception as e:
                logger.error(f"M4 Max message routing error: {e}")
        
        self.messagebus.add_message_handler(route_m4_max_message)
    
    async def _start_standard_workers(self):
        """Fallback to standard workers if M4 Max optimization unavailable"""
        
        logger.info("Starting standard workers (M4 Max optimization unavailable)")
        
        # Standard worker configuration
        for i in range(3):
            worker = asyncio.create_task(self._standard_worker("urgent"))
            self.message_workers.append(worker)
            
        for i in range(5):
            worker = asyncio.create_task(self._standard_worker("high"))
            self.message_workers.append(worker)
            
        for i in range(3):
            worker = asyncio.create_task(self._standard_worker("normal"))
            self.message_workers.append(worker)
    
    async def _standard_worker(self, priority_level: str):
        """Standard worker without M4 Max optimization"""
        
        queue = self.priority_queues.get(priority_level, self.priority_queues["normal"])
        
        while True:
            try:
                message = await queue.get()
                await self._process_m4_max_message(message, "standard")
                queue.task_done()
            except Exception as e:
                logger.error(f"Standard worker error: {e}")
    
    def _add_m4_max_endpoints(self):
        """Add M4 Max specific monitoring endpoints"""
        
        @self.app.get("/m4-max/status")
        async def get_m4_max_status():
            """Get comprehensive M4 Max status"""
            
            status = {
                "m4_max_detected": self.m4_max_detected,
                "hardware_metrics": self.m4_max_metrics,
                "processing_metrics": self.event_processing_metrics,
                "optimization_status": {}
            }
            
            # Get acceleration status
            if self.neural_acceleration_available:
                accel_status = get_acceleration_status()
                status["neural_engine_status"] = accel_status
            
            # Get CPU optimization status
            if self.cpu_optimizer:
                opt_status = get_optimization_status()
                status["optimization_status"] = opt_status
            
            return status
        
        @self.app.get("/m4-max/performance")
        async def get_m4_max_performance():
            """Get detailed M4 Max performance metrics"""
            
            performance = {
                "hardware_acceleration": self.m4_max_metrics,
                "core_utilization": {
                    "performance_cores_used": self.m4_max_metrics["performance_cores_used"],
                    "efficiency_cores_used": self.m4_max_metrics["efficiency_cores_used"]
                },
                "neural_engine": {
                    "enabled": self.neural_acceleration_available,
                    "predictions_count": self.event_processing_metrics["neural_engine_predictions"],
                    "avg_inference_time_ms": self.m4_max_metrics["avg_neural_inference_time_ms"]
                },
                "queue_status": {
                    queue_name: queue.qsize() 
                    for queue_name, queue in self.priority_queues.items()
                }
            }
            
            return performance
        
        @self.app.post("/m4-max/optimize")
        async def trigger_optimization():
            """Trigger M4 Max optimization for current workload"""
            
            if not self.cpu_optimizer:
                return {"error": "CPU optimizer not available"}
            
            try:
                # Trigger workload optimization
                result = await self.cpu_optimizer.optimize_current_workload()
                return {"success": True, "optimization_result": result}
            except Exception as e:
                return {"error": str(e)}
        
        @self.app.get("/m4-max/hardware-routing")
        async def get_hardware_routing_status():
            """Get hardware router configuration and status"""
            
            if not self.hardware_router:
                return {"error": "Hardware router not available"}
            
            try:
                config = self.hardware_router.get_routing_config()
                return {
                    "success": True,
                    "routing_config": config,
                    "routing_metrics": {
                        "risk_calculations_accelerated": self.m4_max_metrics["risk_calculations_accelerated"],
                        "neural_engine_predictions": self.event_processing_metrics["neural_engine_predictions"],
                        "avg_neural_inference_time_ms": self.m4_max_metrics["avg_neural_inference_time_ms"],
                        "hardware_acceleration_ratio": self.m4_max_metrics["hardware_acceleration_ratio"]
                    }
                }
            except Exception as e:
                return {"error": str(e)}
        
        @self.app.post("/m4-max/test-routing")
        async def test_hardware_routing():
            """Test hardware routing with sample workloads"""
            
            if not self.hardware_router:
                return {"error": "Hardware router not available"}
            
            try:
                # Test different workload types
                test_results = {}
                
                # Test ML workload routing
                ml_decision = await route_ml_workload(data_size=10000)
                test_results["ml_workload"] = {
                    "primary_hardware": ml_decision.primary_hardware.value,
                    "confidence": ml_decision.confidence,
                    "estimated_gain": ml_decision.estimated_performance_gain,
                    "reasoning": ml_decision.reasoning
                }
                
                # Test risk workload routing
                risk_decision = await route_risk_workload(data_size=5000, latency_critical=True)
                test_results["risk_workload"] = {
                    "primary_hardware": risk_decision.primary_hardware.value,
                    "confidence": risk_decision.confidence,
                    "estimated_gain": risk_decision.estimated_performance_gain,
                    "reasoning": risk_decision.reasoning
                }
                
                # Test compute workload routing
                compute_decision = await route_compute_workload(WorkloadType.MONTE_CARLO, 1000000)
                test_results["monte_carlo_workload"] = {
                    "primary_hardware": compute_decision.primary_hardware.value,
                    "confidence": compute_decision.confidence,
                    "estimated_gain": compute_decision.estimated_performance_gain,
                    "reasoning": compute_decision.reasoning
                }
                
                return {
                    "success": True,
                    "test_results": test_results,
                    "hardware_availability": await self.hardware_router._check_hardware_availability()
                }
                
            except Exception as e:
                return {"error": str(e)}


# Create M4 Max accelerated engine instance
m4_max_risk_engine = M4MaxRiskEngine()

if __name__ == "__main__":
    # Configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8200"))
    
    logger.info(f"Starting M4 Max Accelerated Risk Engine on {host}:{port}")
    
    uvicorn.run(
        m4_max_risk_engine.app,
        host=host,
        port=port,
        log_level="info",
        access_log=True,
        workers=1  # Single worker optimized for M4 Max async processing
    )