#!/usr/bin/env python3
"""
Analytics Hardware Router - Intelligent Analytics Workload Routing to Neural Engine
Specialized hardware routing for analytics calculations with M4 Max optimization.

Routes analytics workloads to optimal hardware based on:
1. Analytics calculation type (performance, risk, correlation, volatility)
2. Data size and complexity
3. Neural Engine availability for ML-enhanced analytics
4. Real-time processing requirements

Key Features:
- Neural Engine priority for ML-enhanced risk analytics
- Metal GPU routing for correlation matrix calculations
- CPU optimization for I/O-intensive portfolio analytics
- Hybrid processing for complex performance attribution
"""

import os
import logging
import time
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
from dataclasses import dataclass
from functools import wraps

# Import base hardware router
try:
    from backend.hardware_router import (
        HardwareRouter,
        WorkloadType,
        WorkloadCharacteristics,
        RoutingDecision,
        HardwareType,
        hardware_accelerated
    )
    HARDWARE_ROUTER_AVAILABLE = True
except ImportError:
    # Fallback definitions
    HARDWARE_ROUTER_AVAILABLE = False
    
    class WorkloadType(Enum):
        ML_INFERENCE = "ml_inference"
        MATRIX_COMPUTE = "matrix_compute"
        DATA_PROCESSING = "data_processing"
        RISK_CALCULATION = "risk_calculation"
    
    class HardwareType(Enum):
        NEURAL_ENGINE = "neural_engine"
        METAL_GPU = "metal_gpu"
        CPU_P_CORES = "cpu_p_cores"
    
    @dataclass
    class RoutingDecision:
        primary_hardware: HardwareType
        confidence: float = 1.0
        reasoning: str = ""
        estimated_performance_gain: float = 1.0

logger = logging.getLogger(__name__)


class AnalyticsWorkloadType(Enum):
    """Analytics-specific workload types for specialized routing"""
    PORTFOLIO_PERFORMANCE = "portfolio_performance"
    RISK_ANALYTICS = "risk_analytics"
    PERFORMANCE_ATTRIBUTION = "performance_attribution"
    CORRELATION_ANALYSIS = "correlation_analysis"
    VOLATILITY_ANALYTICS = "volatility_analytics"
    EXECUTION_QUALITY = "execution_quality"
    MARKET_IMPACT = "market_impact"
    BACKTESTING_ANALYTICS = "backtesting_analytics"
    REAL_TIME_MONITORING = "real_time_monitoring"


@dataclass
class AnalyticsWorkloadCharacteristics:
    """Analytics-specific workload characteristics for optimal routing"""
    workload_type: AnalyticsWorkloadType
    portfolio_size: int = 0                # Number of positions in portfolio
    data_points: int = 0                   # Number of data points to process
    symbols_count: int = 0                 # Number of symbols for correlation analysis
    ml_enhanced: bool = False              # Uses ML models for enhanced analytics
    real_time_required: bool = False       # Real-time processing requirement (<5ms)
    matrix_operations: bool = False        # Requires matrix calculations
    time_series_length: int = 0           # Length of time series data
    risk_complexity: str = "medium"        # "low", "medium", "high" risk calculation complexity
    requires_neural_engine: bool = False  # Explicitly requires Neural Engine
    memory_intensive: bool = False         # High memory usage analytics


@dataclass
class AnalyticsRoutingDecision(RoutingDecision):
    """Analytics-specific routing decision with additional metadata"""
    analytics_optimizations: List[str] = None
    expected_memory_usage_mb: float = 0.0
    estimated_accuracy_gain: float = 1.0
    cache_recommendations: List[str] = None
    
    def __post_init__(self):
        if self.analytics_optimizations is None:
            self.analytics_optimizations = []
        if self.cache_recommendations is None:
            self.cache_recommendations = []


class AnalyticsHardwareRouter:
    """
    Specialized hardware router for analytics calculations
    
    Optimizes routing based on analytics-specific characteristics:
    - Portfolio size and complexity
    - Risk calculation requirements
    - Correlation matrix dimensions
    - Real-time processing needs
    - ML enhancement opportunities
    """
    
    def __init__(self):
        # Initialize base hardware router if available
        if HARDWARE_ROUTER_AVAILABLE:
            try:
                from backend.hardware_router import get_hardware_router
                self.base_router = get_hardware_router()
            except Exception as e:
                logger.warning(f"Failed to initialize base hardware router: {e}")
                self.base_router = None
        else:
            self.base_router = None
        
        # Analytics-specific configuration
        self.neural_engine_enabled = self._get_bool_env('NEURAL_ENGINE_ENABLED', False)
        self.metal_gpu_enabled = self._get_bool_env('METAL_GPU_ENABLED', False)
        self.analytics_optimization = self._get_bool_env('ANALYTICS_OPTIMIZATION', True)
        
        # Analytics thresholds
        self.large_portfolio_threshold = int(os.getenv('LARGE_PORTFOLIO_THRESHOLD', '100'))     # 100+ positions
        self.complex_correlation_threshold = int(os.getenv('COMPLEX_CORRELATION_THRESHOLD', '10'))  # 10+ symbols
        self.real_time_threshold_ms = float(os.getenv('REAL_TIME_THRESHOLD_MS', '5.0'))        # 5ms for real-time
        self.ml_enhancement_threshold = int(os.getenv('ML_ENHANCEMENT_THRESHOLD', '50'))       # 50+ positions for ML
        
        # Performance tracking
        self.routing_decisions = {}
        self.performance_history = {}
        
        logger.info("🧠 Analytics Hardware Router initialized")
        logger.info(f"   Neural Engine: {'✅ Enabled' if self.neural_engine_enabled else '❌ Disabled'}")
        logger.info(f"   Metal GPU: {'✅ Enabled' if self.metal_gpu_enabled else '❌ Disabled'}")
        logger.info(f"   Analytics Optimization: {'✅ Enabled' if self.analytics_optimization else '❌ Disabled'}")
    
    def _get_bool_env(self, key: str, default: bool = False) -> bool:
        """Get boolean environment variable"""
        value = os.getenv(key, str(default)).lower()
        return value in ('true', '1', 'yes', 'on', 'enabled')
    
    async def route_analytics_workload(self, characteristics: AnalyticsWorkloadCharacteristics) -> AnalyticsRoutingDecision:
        """
        Route analytics workload to optimal hardware
        
        Args:
            characteristics: Analytics workload characteristics
            
        Returns:
            AnalyticsRoutingDecision with optimal hardware and optimizations
        """
        start_time = time.time()
        
        try:
            # Route based on analytics workload type
            if characteristics.workload_type == AnalyticsWorkloadType.PORTFOLIO_PERFORMANCE:
                decision = await self._route_portfolio_performance(characteristics)
            elif characteristics.workload_type == AnalyticsWorkloadType.RISK_ANALYTICS:
                decision = await self._route_risk_analytics(characteristics)
            elif characteristics.workload_type == AnalyticsWorkloadType.CORRELATION_ANALYSIS:
                decision = await self._route_correlation_analysis(characteristics)
            elif characteristics.workload_type == AnalyticsWorkloadType.VOLATILITY_ANALYTICS:
                decision = await self._route_volatility_analytics(characteristics)
            elif characteristics.workload_type == AnalyticsWorkloadType.PERFORMANCE_ATTRIBUTION:
                decision = await self._route_performance_attribution(characteristics)
            elif characteristics.workload_type == AnalyticsWorkloadType.EXECUTION_QUALITY:
                decision = await self._route_execution_quality(characteristics)
            elif characteristics.workload_type == AnalyticsWorkloadType.REAL_TIME_MONITORING:
                decision = await self._route_real_time_monitoring(characteristics)
            else:
                decision = await self._route_general_analytics(characteristics)
            
            # Add routing time to decision
            routing_time_ms = (time.time() - start_time) * 1000
            decision.analytics_optimizations.append(f"Routing decision time: {routing_time_ms:.2f}ms")
            
            # Store decision for performance tracking
            self.routing_decisions[f"{characteristics.workload_type.value}_{int(time.time())}"] = decision
            
            logger.debug(f"Analytics routing: {characteristics.workload_type.value} -> {decision.primary_hardware.value} "
                        f"(confidence: {decision.confidence:.2f}, gain: {decision.estimated_performance_gain:.1f}x)")
            
            return decision
            
        except Exception as e:
            logger.error(f"Analytics routing failed: {e}")
            return AnalyticsRoutingDecision(
                primary_hardware=HardwareType.CPU_P_CORES,
                confidence=0.5,
                reasoning=f"Routing failed, using CPU fallback: {str(e)}",
                estimated_performance_gain=1.0
            )
    
    async def _route_portfolio_performance(self, chars: AnalyticsWorkloadCharacteristics) -> AnalyticsRoutingDecision:
        """Route portfolio performance analytics"""
        
        # Large portfolios benefit from parallel processing
        if chars.portfolio_size > self.large_portfolio_threshold and self.metal_gpu_enabled:
            return AnalyticsRoutingDecision(
                primary_hardware=HardwareType.METAL_GPU,
                confidence=0.9,
                reasoning=f"Large portfolio ({chars.portfolio_size} positions) - GPU parallel processing optimal",
                estimated_performance_gain=12.0,
                expected_memory_usage_mb=chars.portfolio_size * 0.5,
                analytics_optimizations=[
                    "GPU parallel position processing",
                    "Matrix-optimized performance calculations"
                ],
                cache_recommendations=[
                    "Cache position weights",
                    "Pre-calculate benchmark returns"
                ]
            )
        
        # ML-enhanced portfolio analytics benefit from Neural Engine
        elif chars.ml_enhanced and self.neural_engine_enabled and chars.portfolio_size > self.ml_enhancement_threshold:
            return AnalyticsRoutingDecision(
                primary_hardware=HardwareType.NEURAL_ENGINE,
                confidence=0.85,
                reasoning=f"ML-enhanced portfolio analytics - Neural Engine optimal ({chars.portfolio_size} positions)",
                estimated_performance_gain=6.5,
                estimated_accuracy_gain=1.15,
                expected_memory_usage_mb=chars.portfolio_size * 0.3,
                analytics_optimizations=[
                    "Neural Engine ML performance prediction",
                    "Enhanced attribution analysis"
                ]
            )
        
        # Standard portfolio analytics - CPU optimized
        else:
            return AnalyticsRoutingDecision(
                primary_hardware=HardwareType.CPU_P_CORES,
                confidence=0.8,
                reasoning=f"Standard portfolio analytics - CPU optimized ({chars.portfolio_size} positions)",
                estimated_performance_gain=2.5,
                expected_memory_usage_mb=chars.portfolio_size * 0.1,
                analytics_optimizations=[
                    "CPU-optimized sequential processing",
                    "Memory-efficient calculations"
                ]
            )
    
    async def _route_risk_analytics(self, chars: AnalyticsWorkloadCharacteristics) -> AnalyticsRoutingDecision:
        """Route risk analytics with Neural Engine priority for ML enhancement"""
        
        # High complexity risk analytics with ML enhancement
        if (chars.risk_complexity == "high" and chars.ml_enhanced and 
            self.neural_engine_enabled and chars.portfolio_size > 20):
            return AnalyticsRoutingDecision(
                primary_hardware=HardwareType.NEURAL_ENGINE,
                secondary_hardware=HardwareType.METAL_GPU if self.metal_gpu_enabled else None,
                confidence=0.95,
                reasoning="High complexity ML-enhanced risk analytics - Neural Engine + GPU hybrid optimal",
                estimated_performance_gain=8.3,  # Based on validated benchmarks
                estimated_accuracy_gain=1.25,
                expected_memory_usage_mb=chars.portfolio_size * 1.2,
                analytics_optimizations=[
                    "Neural Engine ML risk prediction",
                    "GPU Monte Carlo VaR calculations",
                    "Hybrid stress testing"
                ],
                cache_recommendations=[
                    "Cache correlation matrices",
                    "Pre-calculate scenario weights"
                ]
            )
        
        # Medium complexity risk with GPU acceleration
        elif chars.risk_complexity in ["medium", "high"] and self.metal_gpu_enabled and chars.portfolio_size > 50:
            return AnalyticsRoutingDecision(
                primary_hardware=HardwareType.METAL_GPU,
                confidence=0.88,
                reasoning=f"Medium/high complexity risk analytics - GPU Monte Carlo optimal ({chars.portfolio_size} positions)",
                estimated_performance_gain=5.2,
                expected_memory_usage_mb=chars.portfolio_size * 0.8,
                analytics_optimizations=[
                    "GPU-accelerated Monte Carlo simulation",
                    "Parallel VaR/CVaR calculations"
                ]
            )
        
        # Standard risk analytics
        else:
            return AnalyticsRoutingDecision(
                primary_hardware=HardwareType.CPU_P_CORES,
                confidence=0.75,
                reasoning=f"Standard risk analytics - CPU optimized ({chars.risk_complexity} complexity)",
                estimated_performance_gain=1.8,
                expected_memory_usage_mb=chars.portfolio_size * 0.2,
                analytics_optimizations=[
                    "CPU-optimized risk calculations",
                    "Sequential VaR processing"
                ]
            )
    
    async def _route_correlation_analysis(self, chars: AnalyticsWorkloadCharacteristics) -> AnalyticsRoutingDecision:
        """Route correlation analysis - Matrix operations ideal for GPU"""
        
        # Large correlation matrices benefit significantly from GPU
        if chars.symbols_count > self.complex_correlation_threshold and self.metal_gpu_enabled:
            matrix_size = chars.symbols_count * chars.symbols_count
            return AnalyticsRoutingDecision(
                primary_hardware=HardwareType.METAL_GPU,
                confidence=0.95,
                reasoning=f"Large correlation matrix ({chars.symbols_count}x{chars.symbols_count}) - GPU matrix operations optimal",
                estimated_performance_gain=25.0,  # Matrix operations see huge GPU gains
                expected_memory_usage_mb=matrix_size * 0.008,  # 8 bytes per float
                analytics_optimizations=[
                    "GPU-accelerated matrix multiplication",
                    "Parallel correlation calculations",
                    "Optimized eigenvector analysis"
                ],
                cache_recommendations=[
                    "Cache price time series",
                    "Pre-calculate return matrices"
                ]
            )
        
        # Medium-sized correlation analysis
        elif chars.symbols_count > 5 and self.metal_gpu_enabled:
            return AnalyticsRoutingDecision(
                primary_hardware=HardwareType.METAL_GPU,
                confidence=0.8,
                reasoning=f"Medium correlation matrix ({chars.symbols_count} symbols) - GPU beneficial",
                estimated_performance_gain=8.5,
                expected_memory_usage_mb=chars.symbols_count * chars.symbols_count * 0.008,
                analytics_optimizations=[
                    "GPU matrix operations",
                    "Parallel correlation computation"
                ]
            )
        
        # Small correlation analysis - CPU sufficient
        else:
            return AnalyticsRoutingDecision(
                primary_hardware=HardwareType.CPU_P_CORES,
                confidence=0.85,
                reasoning=f"Small correlation analysis ({chars.symbols_count} symbols) - CPU sufficient",
                estimated_performance_gain=1.5,
                expected_memory_usage_mb=chars.symbols_count * chars.symbols_count * 0.008,
                analytics_optimizations=[
                    "CPU-optimized correlation calculation",
                    "Sequential matrix processing"
                ]
            )
    
    async def _route_volatility_analytics(self, chars: AnalyticsWorkloadCharacteristics) -> AnalyticsRoutingDecision:
        """Route volatility analytics with ML enhancement opportunities"""
        
        # ML-enhanced volatility modeling with Neural Engine
        if chars.ml_enhanced and self.neural_engine_enabled and chars.time_series_length > 1000:
            return AnalyticsRoutingDecision(
                primary_hardware=HardwareType.NEURAL_ENGINE,
                confidence=0.9,
                reasoning=f"ML-enhanced volatility modeling - Neural Engine optimal ({chars.time_series_length} data points)",
                estimated_performance_gain=7.5,
                estimated_accuracy_gain=1.2,
                expected_memory_usage_mb=chars.time_series_length * 0.01,
                analytics_optimizations=[
                    "Neural Engine GARCH modeling",
                    "ML volatility prediction",
                    "Regime detection analysis"
                ],
                cache_recommendations=[
                    "Cache historical volatility",
                    "Pre-calculate rolling statistics"
                ]
            )
        
        # Large time series - GPU beneficial for parallel calculations
        elif chars.time_series_length > 5000 and self.metal_gpu_enabled:
            return AnalyticsRoutingDecision(
                primary_hardware=HardwareType.METAL_GPU,
                confidence=0.82,
                reasoning=f"Large time series volatility - GPU parallel processing ({chars.time_series_length} points)",
                estimated_performance_gain=4.8,
                expected_memory_usage_mb=chars.time_series_length * 0.008,
                analytics_optimizations=[
                    "GPU parallel volatility calculation",
                    "Parallel rolling window processing"
                ]
            )
        
        # Standard volatility analytics
        else:
            return AnalyticsRoutingDecision(
                primary_hardware=HardwareType.CPU_P_CORES,
                confidence=0.75,
                reasoning=f"Standard volatility analytics - CPU optimized ({chars.time_series_length} points)",
                estimated_performance_gain=1.6,
                expected_memory_usage_mb=chars.time_series_length * 0.004,
                analytics_optimizations=[
                    "CPU-optimized volatility calculation",
                    "Sequential time series processing"
                ]
            )
    
    async def _route_performance_attribution(self, chars: AnalyticsWorkloadCharacteristics) -> AnalyticsRoutingDecision:
        """Route performance attribution - Complex multi-factor analysis"""
        
        # Large portfolio attribution with ML enhancement
        if (chars.portfolio_size > self.large_portfolio_threshold and chars.ml_enhanced and 
            self.neural_engine_enabled):
            return AnalyticsRoutingDecision(
                primary_hardware=HardwareType.NEURAL_ENGINE,
                secondary_hardware=HardwareType.METAL_GPU if self.metal_gpu_enabled else None,
                confidence=0.92,
                reasoning=f"Large portfolio ML attribution - Hybrid Neural Engine + GPU ({chars.portfolio_size} positions)",
                estimated_performance_gain=9.2,
                estimated_accuracy_gain=1.18,
                expected_memory_usage_mb=chars.portfolio_size * 1.5,
                analytics_optimizations=[
                    "Neural Engine factor analysis",
                    "GPU parallel attribution calculations",
                    "ML-enhanced factor selection"
                ],
                cache_recommendations=[
                    "Cache factor exposures",
                    "Pre-calculate benchmark attribution"
                ]
            )
        
        # Standard attribution analysis
        elif chars.portfolio_size > 20 and self.metal_gpu_enabled:
            return AnalyticsRoutingDecision(
                primary_hardware=HardwareType.METAL_GPU,
                confidence=0.8,
                reasoning=f"Standard attribution analysis - GPU parallel processing ({chars.portfolio_size} positions)",
                estimated_performance_gain=4.5,
                expected_memory_usage_mb=chars.portfolio_size * 0.6,
                analytics_optimizations=[
                    "GPU parallel factor calculations",
                    "Matrix-optimized attribution"
                ]
            )
        
        # Small portfolio attribution
        else:
            return AnalyticsRoutingDecision(
                primary_hardware=HardwareType.CPU_P_CORES,
                confidence=0.7,
                reasoning=f"Small portfolio attribution - CPU sufficient ({chars.portfolio_size} positions)",
                estimated_performance_gain=1.8,
                expected_memory_usage_mb=chars.portfolio_size * 0.3,
                analytics_optimizations=[
                    "CPU-optimized sequential attribution",
                    "Memory-efficient factor analysis"
                ]
            )
    
    async def _route_execution_quality(self, chars: AnalyticsWorkloadCharacteristics) -> AnalyticsRoutingDecision:
        """Route execution quality analytics"""
        
        # Real-time execution quality requires fast processing
        if chars.real_time_required and self.neural_engine_enabled:
            return AnalyticsRoutingDecision(
                primary_hardware=HardwareType.NEURAL_ENGINE,
                confidence=0.88,
                reasoning="Real-time execution quality - Neural Engine low-latency processing",
                estimated_performance_gain=6.0,
                expected_memory_usage_mb=chars.data_points * 0.005,
                analytics_optimizations=[
                    "Neural Engine real-time slippage analysis",
                    "Fast market impact calculation"
                ],
                cache_recommendations=[
                    "Cache recent market data",
                    "Pre-calculate benchmark spreads"
                ]
            )
        
        # Batch execution quality analysis
        else:
            return AnalyticsRoutingDecision(
                primary_hardware=HardwareType.CPU_P_CORES,
                confidence=0.8,
                reasoning="Batch execution quality analysis - CPU optimized",
                estimated_performance_gain=2.0,
                expected_memory_usage_mb=chars.data_points * 0.003,
                analytics_optimizations=[
                    "CPU-optimized execution analysis",
                    "Batch processing optimization"
                ]
            )
    
    async def _route_real_time_monitoring(self, chars: AnalyticsWorkloadCharacteristics) -> AnalyticsRoutingDecision:
        """Route real-time monitoring workloads"""
        
        # Real-time monitoring requires ultra-low latency
        if chars.real_time_required and self.neural_engine_enabled:
            return AnalyticsRoutingDecision(
                primary_hardware=HardwareType.NEURAL_ENGINE,
                confidence=0.95,
                reasoning="Real-time monitoring - Neural Engine ultra-low latency",
                estimated_performance_gain=8.0,
                expected_memory_usage_mb=50,  # Keep memory usage low for real-time
                analytics_optimizations=[
                    "Neural Engine real-time processing",
                    "Ultra-low latency calculations",
                    "Streaming analytics optimization"
                ],
                cache_recommendations=[
                    "Maintain hot cache for recent data",
                    "Pre-compute critical metrics"
                ]
            )
        
        # Standard monitoring
        else:
            return AnalyticsRoutingDecision(
                primary_hardware=HardwareType.CPU_P_CORES,
                confidence=0.8,
                reasoning="Standard monitoring - CPU optimized for continuous processing",
                estimated_performance_gain=2.5,
                expected_memory_usage_mb=100,
                analytics_optimizations=[
                    "CPU-optimized monitoring",
                    "Efficient data streaming"
                ]
            )
    
    async def _route_general_analytics(self, chars: AnalyticsWorkloadCharacteristics) -> AnalyticsRoutingDecision:
        """Route general analytics workloads"""
        
        # Default routing based on characteristics
        if chars.ml_enhanced and self.neural_engine_enabled:
            return AnalyticsRoutingDecision(
                primary_hardware=HardwareType.NEURAL_ENGINE,
                confidence=0.75,
                reasoning="General ML-enhanced analytics - Neural Engine",
                estimated_performance_gain=5.0,
                analytics_optimizations=["Neural Engine ML processing"]
            )
        elif chars.matrix_operations and self.metal_gpu_enabled:
            return AnalyticsRoutingDecision(
                primary_hardware=HardwareType.METAL_GPU,
                confidence=0.8,
                reasoning="General matrix analytics - GPU acceleration",
                estimated_performance_gain=6.0,
                analytics_optimizations=["GPU matrix operations"]
            )
        else:
            return AnalyticsRoutingDecision(
                primary_hardware=HardwareType.CPU_P_CORES,
                confidence=0.7,
                reasoning="General analytics - CPU processing",
                estimated_performance_gain=1.5,
                analytics_optimizations=["CPU-optimized processing"]
            )
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get analytics routing statistics"""
        
        if not self.routing_decisions:
            return {"status": "No routing decisions recorded"}
        
        # Analyze routing patterns
        hardware_usage = {}
        workload_types = {}
        performance_gains = []
        
        for decision_id, decision in self.routing_decisions.items():
            # Count hardware usage
            hardware = decision.primary_hardware.value
            hardware_usage[hardware] = hardware_usage.get(hardware, 0) + 1
            
            # Count workload types
            workload_type = decision_id.split('_')[0]  # Extract workload type from ID
            workload_types[workload_type] = workload_types.get(workload_type, 0) + 1
            
            # Collect performance gains
            performance_gains.append(decision.estimated_performance_gain)
        
        return {
            "total_routing_decisions": len(self.routing_decisions),
            "hardware_usage_distribution": hardware_usage,
            "workload_type_distribution": workload_types,
            "performance_statistics": {
                "average_performance_gain": sum(performance_gains) / len(performance_gains),
                "max_performance_gain": max(performance_gains),
                "min_performance_gain": min(performance_gains)
            },
            "hardware_preferences": {
                "neural_engine_usage_rate": hardware_usage.get("neural_engine", 0) / len(self.routing_decisions),
                "metal_gpu_usage_rate": hardware_usage.get("metal_gpu", 0) / len(self.routing_decisions),
                "cpu_usage_rate": hardware_usage.get("cpu_p_cores", 0) / len(self.routing_decisions)
            }
        }


# Global analytics router instance
_analytics_router: Optional[AnalyticsHardwareRouter] = None

def get_analytics_hardware_router() -> AnalyticsHardwareRouter:
    """Get global analytics hardware router instance"""
    global _analytics_router
    if _analytics_router is None:
        _analytics_router = AnalyticsHardwareRouter()
    return _analytics_router

def analytics_accelerated(workload_type: AnalyticsWorkloadType, **characteristics):
    """
    Decorator for automatic analytics hardware acceleration routing
    
    Usage:
        @analytics_accelerated(AnalyticsWorkloadType.PORTFOLIO_PERFORMANCE, portfolio_size=100)
        async def calculate_portfolio_metrics(data):
            # Function automatically routed to optimal hardware
            pass
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            router = get_analytics_hardware_router()
            
            # Create analytics workload characteristics
            analytics_chars = AnalyticsWorkloadCharacteristics(
                workload_type=workload_type,
                **characteristics
            )
            
            # Get routing decision
            decision = await router.route_analytics_workload(analytics_chars)
            
            logger.info(f"Analytics routing {func.__name__} to {decision.primary_hardware.value} "
                       f"(confidence: {decision.confidence:.2f}, gain: {decision.estimated_performance_gain:.1f}x)")
            logger.debug(f"Routing reason: {decision.reasoning}")
            logger.debug(f"Optimizations: {decision.analytics_optimizations}")
            
            # Add routing info to kwargs
            kwargs['_analytics_routing'] = decision
            
            # Execute function
            return await func(*args, **kwargs)
            
        return wrapper
    return decorator

# Convenience functions for common analytics workloads
async def route_portfolio_analytics(portfolio_size: int, ml_enhanced: bool = False) -> AnalyticsRoutingDecision:
    """Route portfolio analytics workload"""
    router = get_analytics_hardware_router()
    chars = AnalyticsWorkloadCharacteristics(
        workload_type=AnalyticsWorkloadType.PORTFOLIO_PERFORMANCE,
        portfolio_size=portfolio_size,
        ml_enhanced=ml_enhanced
    )
    return await router.route_analytics_workload(chars)

async def route_risk_analytics(portfolio_size: int, risk_complexity: str = "medium", 
                              ml_enhanced: bool = False) -> AnalyticsRoutingDecision:
    """Route risk analytics workload"""
    router = get_analytics_hardware_router()
    chars = AnalyticsWorkloadCharacteristics(
        workload_type=AnalyticsWorkloadType.RISK_ANALYTICS,
        portfolio_size=portfolio_size,
        risk_complexity=risk_complexity,
        ml_enhanced=ml_enhanced
    )
    return await router.route_analytics_workload(chars)

async def route_correlation_analysis(symbols_count: int, data_points: int = 252) -> AnalyticsRoutingDecision:
    """Route correlation analysis workload"""
    router = get_analytics_hardware_router()
    chars = AnalyticsWorkloadCharacteristics(
        workload_type=AnalyticsWorkloadType.CORRELATION_ANALYSIS,
        symbols_count=symbols_count,
        data_points=data_points,
        matrix_operations=True
    )
    return await router.route_analytics_workload(chars)

async def route_volatility_analytics(time_series_length: int, ml_enhanced: bool = False) -> AnalyticsRoutingDecision:
    """Route volatility analytics workload"""
    router = get_analytics_hardware_router()
    chars = AnalyticsWorkloadCharacteristics(
        workload_type=AnalyticsWorkloadType.VOLATILITY_ANALYTICS,
        time_series_length=time_series_length,
        ml_enhanced=ml_enhanced
    )
    return await router.route_analytics_workload(chars)

if __name__ == "__main__":
    # Test analytics hardware router
    async def test_analytics_router():
        router = AnalyticsHardwareRouter()
        
        # Test portfolio performance routing
        portfolio_chars = AnalyticsWorkloadCharacteristics(
            workload_type=AnalyticsWorkloadType.PORTFOLIO_PERFORMANCE,
            portfolio_size=150,
            ml_enhanced=True
        )
        decision = await router.route_analytics_workload(portfolio_chars)
        print(f"Portfolio Analytics: {decision.primary_hardware.value} (gain: {decision.estimated_performance_gain:.1f}x)")
        print(f"  Reasoning: {decision.reasoning}")
        print(f"  Optimizations: {decision.analytics_optimizations}")
        
        # Test correlation analysis routing
        correlation_chars = AnalyticsWorkloadCharacteristics(
            workload_type=AnalyticsWorkloadType.CORRELATION_ANALYSIS,
            symbols_count=15,
            matrix_operations=True
        )
        decision = await router.route_analytics_workload(correlation_chars)
        print(f"Correlation Analysis: {decision.primary_hardware.value} (gain: {decision.estimated_performance_gain:.1f}x)")
        print(f"  Reasoning: {decision.reasoning}")
        
        # Test risk analytics routing
        risk_chars = AnalyticsWorkloadCharacteristics(
            workload_type=AnalyticsWorkloadType.RISK_ANALYTICS,
            portfolio_size=80,
            risk_complexity="high",
            ml_enhanced=True
        )
        decision = await router.route_analytics_workload(risk_chars)
        print(f"Risk Analytics: {decision.primary_hardware.value} (gain: {decision.estimated_performance_gain:.1f}x)")
        print(f"  Reasoning: {decision.reasoning}")
        
        # Show routing statistics
        stats = router.get_routing_statistics()
        print(f"Routing Statistics: {stats}")
    
    asyncio.run(test_analytics_router())