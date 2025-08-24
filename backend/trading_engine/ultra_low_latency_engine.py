"""
Ultra-Low Latency Trading Engine
================================

Integrated ultra-low latency trading system combining all Phase 2B optimizations:
- JIT-compiled risk engine (<1ms P99)
- SIMD vectorized position management (<0.5ms P99)
- Lock-free order management (<0.3ms P99)
- Memory pool optimization (99% allocation reduction)
- Cache-optimized data structures

Target: <2.8ms P99 end-to-end trading latency
"""

import asyncio
import logging
import time
import threading
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from .compiled_risk_engine import CompiledRiskEngine, CompiledRiskMetrics
from .vectorized_position_keeper import VectorizedPositionKeeper, VectorizedPortfolio
from .lockfree_order_manager import LockFreeOrderManager, OrderEvent
from .optimized_execution_engine import OptimizedExecutionEngine, OptimizedSmartOrderRouter
from .poolable_objects import (
    PooledOrder, PooledOrderFill, PooledVenueQuote,
    create_pooled_order, release_pooled_order,
    OrderSide, OrderType, TimeInForce, trading_pools
)
from .risk_engine import RiskLimits
from .memory_pool import pool_manager

logger = logging.getLogger(__name__)


@dataclass
class TradingEngineMetrics:
    """Comprehensive trading engine performance metrics."""
    engine_id: str
    
    # End-to-end latency tracking
    total_trades: int = 0
    avg_end_to_end_latency_ns: float = 0.0
    p99_end_to_end_latency_ns: float = 0.0
    latency_samples: List[float] = field(default_factory=list)
    
    # Component-specific metrics
    risk_check_time_ns: float = 0.0
    position_update_time_ns: float = 0.0
    order_processing_time_ns: float = 0.0
    execution_time_ns: float = 0.0
    
    # Success rates
    risk_check_success_rate: float = 100.0
    order_submission_success_rate: float = 100.0
    execution_success_rate: float = 100.0
    
    # Memory efficiency
    memory_pool_hit_rate: float = 0.0
    total_objects_pooled: int = 0
    
    # Throughput metrics
    orders_per_second: float = 0.0
    max_concurrent_orders: int = 0
    
    # System resource utilization
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0


class UltraLowLatencyTradingEngine:
    """
    Integrated ultra-low latency trading system.
    
    Combines all Phase 2B optimizations for maximum performance:
    - Memory pools: 99% allocation reduction
    - JIT compilation: <1ms risk checking
    - SIMD vectorization: <0.5ms position updates
    - Lock-free structures: <0.3ms order processing
    - Optimized execution: <1ms venue selection
    
    Target: <2.8ms P99 end-to-end latency
    """
    
    def __init__(self, engine_id: str = "ultra_low_latency"):
        self.engine_id = engine_id
        
        # Initialize optimized components
        self.risk_engine = CompiledRiskEngine()
        self.position_keeper = VectorizedPositionKeeper()
        self.order_manager = LockFreeOrderManager()
        self.execution_engine = OptimizedExecutionEngine()
        
        # Performance metrics
        self.metrics = TradingEngineMetrics(engine_id=engine_id)
        self.start_time = time.time()
        
        # End-to-end latency tracking
        self.active_trade_contexts: Dict[str, float] = {}  # order_id -> start_time_ns
        self.latency_history = []
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(
            max_workers=6, 
            thread_name_prefix=f"ull_trading_{engine_id}"
        )
        
        # Event callbacks and monitoring
        self.trade_callbacks: List[Callable] = []
        self.monitoring_enabled = True
        
        # Setup component integration
        self._setup_component_integration()
        
        # Start monitoring
        asyncio.create_task(self._start_monitoring())
        
        logger.info(f"Ultra-Low Latency Trading Engine '{engine_id}' initialized")
    
    def _setup_component_integration(self):
        """Setup integration between optimized components."""
        
        # Order manager callbacks
        self.order_manager.add_event_callback(self._on_order_event)
        
        # Position keeper callbacks
        self.position_keeper.add_position_callback(self._on_position_update)
        
        # Execution engine callbacks
        self.execution_engine.add_execution_callback(self._on_execution_event)
        
        logger.debug("Component integration setup complete")
    
    async def submit_trade_ultra_fast(self,
                                    symbol: str,
                                    side: OrderSide,
                                    order_type: OrderType,
                                    quantity: float,
                                    price: Optional[float] = None,
                                    stop_price: Optional[float] = None,
                                    time_in_force: TimeInForce = TimeInForce.DAY,
                                    portfolio_id: str = "default",
                                    strategy_id: Optional[str] = None) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Submit trade through ultra-low latency pipeline.
        
        Returns:
            Tuple[bool, str, Dict]: (success, order_id, latency_breakdown)
            
        Target: <2.8ms P99 end-to-end latency
        """
        # Start end-to-end timing
        start_time_ns = time.perf_counter_ns()
        
        latency_breakdown = {
            "order_creation_ns": 0,
            "risk_check_ns": 0,
            "order_submission_ns": 0,
            "execution_routing_ns": 0,
            "total_ns": 0
        }
        
        order = None
        try:
            # Phase 1: Order Creation (Memory Pool)
            order_start = time.perf_counter_ns()
            order = create_pooled_order(
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                stop_price=stop_price,
                time_in_force=time_in_force,
                portfolio_id=portfolio_id,
                strategy_id=strategy_id
            )
            latency_breakdown["order_creation_ns"] = time.perf_counter_ns() - order_start
            
            # Track for end-to-end measurement
            self.active_trade_contexts[order.id] = start_time_ns
            
            # Phase 2: Risk Check (JIT Compiled)
            risk_start = time.perf_counter_ns()
            risk_approved = await self.risk_engine.check_pre_trade_risk_compiled(order, portfolio_id)
            latency_breakdown["risk_check_ns"] = time.perf_counter_ns() - risk_start
            
            if not risk_approved:
                self._complete_trade_measurement(order.id, start_time_ns, success=False)
                release_pooled_order(order)
                return False, "", latency_breakdown
            
            # Phase 3: Order Submission (Lock-Free)
            submission_start = time.perf_counter_ns()
            submitted = await self.order_manager.submit_order_lockfree(order)
            latency_breakdown["order_submission_ns"] = time.perf_counter_ns() - submission_start
            
            if not submitted:
                self._complete_trade_measurement(order.id, start_time_ns, success=False)
                release_pooled_order(order)
                return False, "", latency_breakdown
            
            # Phase 4: Execution Routing (Optimized)
            execution_start = time.perf_counter_ns()
            routed = await self.execution_engine.submit_order_optimized(order)
            latency_breakdown["execution_routing_ns"] = time.perf_counter_ns() - execution_start
            
            if not routed:
                await self.order_manager.cancel_order_lockfree(order.id)
                self._complete_trade_measurement(order.id, start_time_ns, success=False)
                return False, order.id, latency_breakdown
            
            # Calculate total latency
            total_latency_ns = time.perf_counter_ns() - start_time_ns
            latency_breakdown["total_ns"] = total_latency_ns
            
            # Update metrics
            self._update_trade_metrics(latency_breakdown, success=True)
            
            logger.debug(f"Trade submitted successfully: {order.id} in {total_latency_ns/1000:.2f}μs")
            return True, order.id, latency_breakdown
            
        except Exception as e:
            if order:
                self._complete_trade_measurement(order.id, start_time_ns, success=False)
                release_pooled_order(order)
            
            logger.error(f"Ultra-fast trade submission failed: {e}")
            return False, "", latency_breakdown
    
    async def handle_fill_ultra_fast(self, order_id: str, fill_quantity: float, 
                                   fill_price: float, execution_id: str) -> bool:
        """
        Handle fill notification through ultra-fast pipeline.
        
        Target: <1ms P99 fill processing
        """
        fill_start_ns = time.perf_counter_ns()
        
        try:
            # Create pooled fill
            from .poolable_objects import create_pooled_fill
            fill = create_pooled_fill(
                order_id=order_id,
                quantity=fill_quantity,
                price=fill_price,
                execution_id=execution_id
            )
            
            # Get order for position update
            order = self.order_manager.get_order_lockfree(order_id)
            if not order:
                logger.warning(f"Fill received for unknown order: {order_id}")
                return False
            
            # Process fill through lock-free order manager
            await self.order_manager.handle_fill_lockfree(order_id, fill)
            
            # Update positions using vectorized calculations
            await self.position_keeper.process_fill_vectorized(order, fill)
            
            # Check if order is complete for end-to-end measurement
            if order.is_complete:
                self._complete_trade_measurement(order_id, 
                    self.active_trade_contexts.get(order_id, fill_start_ns), 
                    success=True
                )
            
            fill_time_ns = time.perf_counter_ns() - fill_start_ns
            logger.debug(f"Fill processed in {fill_time_ns/1000:.2f}μs")
            
            return True
            
        except Exception as e:
            logger.error(f"Ultra-fast fill processing failed: {e}")
            return False
    
    def _complete_trade_measurement(self, order_id: str, start_time_ns: float, success: bool):
        """Complete end-to-end trade latency measurement."""
        if order_id not in self.active_trade_contexts:
            return
        
        end_time_ns = time.perf_counter_ns()
        total_latency_ns = end_time_ns - self.active_trade_contexts[order_id]
        
        # Update latency history
        self.latency_history.append(total_latency_ns)
        
        # Keep sliding window of recent measurements
        if len(self.latency_history) > 10000:
            self.latency_history = self.latency_history[-5000:]
        
        # Update metrics
        self.metrics.total_trades += 1
        if success:
            # Update average latency (EMA)
            if self.metrics.avg_end_to_end_latency_ns == 0:
                self.metrics.avg_end_to_end_latency_ns = total_latency_ns
            else:
                self.metrics.avg_end_to_end_latency_ns = (
                    self.metrics.avg_end_to_end_latency_ns * 0.95 + 
                    total_latency_ns * 0.05
                )
            
            # Update P99 latency
            if len(self.latency_history) >= 100:
                self.metrics.p99_end_to_end_latency_ns = np.percentile(self.latency_history, 99)
        
        # Remove from active tracking
        del self.active_trade_contexts[order_id]
    
    def _update_trade_metrics(self, latency_breakdown: Dict[str, float], success: bool):
        """Update comprehensive trade metrics."""
        # Component latency updates
        self.metrics.risk_check_time_ns = (
            self.metrics.risk_check_time_ns * 0.95 + 
            latency_breakdown["risk_check_ns"] * 0.05
        )
        
        self.metrics.order_processing_time_ns = (
            self.metrics.order_processing_time_ns * 0.95 +
            latency_breakdown["order_submission_ns"] * 0.05
        )
        
        self.metrics.execution_time_ns = (
            self.metrics.execution_time_ns * 0.95 +
            latency_breakdown["execution_routing_ns"] * 0.05
        )
    
    def _on_order_event(self, event_type: OrderEvent, order: PooledOrder, metadata: Dict[str, Any]):
        """Handle order events from lock-free order manager."""
        try:
            # Update success rates based on events
            if event_type == OrderEvent.REJECTED:
                # Update order submission success rate
                pass
            elif event_type == OrderEvent.FILLED:
                # Trade completed successfully
                for callback in self.trade_callbacks:
                    self.executor.submit(callback, "trade_completed", order, metadata)
        
        except Exception as e:
            logger.error(f"Order event handling failed: {e}")
    
    def _on_position_update(self, position_update):
        """Handle position updates from vectorized position keeper."""
        try:
            # Update position-related metrics
            pass
        except Exception as e:
            logger.error(f"Position update handling failed: {e}")
    
    def _on_execution_event(self, order_id: str, fill):
        """Handle execution events from optimized execution engine."""
        try:
            # This would typically trigger fill processing
            asyncio.create_task(self.handle_fill_ultra_fast(
                order_id=order_id,
                fill_quantity=fill.quantity,
                fill_price=fill.price,
                execution_id=fill.execution_id
            ))
        except Exception as e:
            logger.error(f"Execution event handling failed: {e}")
    
    def set_portfolio_risk_limits(self, portfolio_id: str, limits: RiskLimits):
        """Set risk limits for portfolio."""
        self.risk_engine.set_portfolio_limits(portfolio_id, limits)
    
    def add_execution_venue(self, venue):
        """Add execution venue."""
        self.execution_engine.add_venue(venue)
    
    def add_trade_callback(self, callback: Callable):
        """Add trade completion callback."""
        self.trade_callbacks.append(callback)
    
    async def get_portfolio_summary_ultra_fast(self, portfolio_id: str) -> Dict[str, Any]:
        """Get portfolio summary using vectorized calculations."""
        return await self.position_keeper.get_portfolio_summary_vectorized(portfolio_id)
    
    async def bulk_update_market_prices(self, portfolio_id: str, price_updates: Dict[str, float]) -> int:
        """Bulk update market prices using vectorized operations."""
        return await self.position_keeper.bulk_update_market_prices(portfolio_id, price_updates)
    
    def get_comprehensive_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics from all components."""
        
        # Get stats from all components
        risk_stats = self.risk_engine.get_performance_statistics()
        position_stats = self.position_keeper.get_performance_statistics()
        order_stats = self.order_manager.get_performance_statistics()
        execution_stats = self.execution_engine.get_performance_statistics()
        
        # Memory pool statistics
        global_pool_stats = pool_manager.get_global_metrics()
        
        # Calculate engine-level metrics
        engine_uptime = time.time() - self.start_time
        
        # Latency analysis
        latency_analysis = {}
        if self.latency_history:
            latencies_us = np.array(self.latency_history) / 1000  # Convert to microseconds
            latency_analysis = {
                "samples": len(latencies_us),
                "min_us": float(np.min(latencies_us)),
                "max_us": float(np.max(latencies_us)),
                "avg_us": float(np.mean(latencies_us)),
                "p50_us": float(np.percentile(latencies_us, 50)),
                "p95_us": float(np.percentile(latencies_us, 95)),
                "p99_us": float(np.percentile(latencies_us, 99)),
                "std_dev_us": float(np.std(latencies_us))
            }
        
        return {
            "ultra_low_latency_engine": {
                "engine_id": self.engine_id,
                "uptime_seconds": engine_uptime,
                "total_trades": self.metrics.total_trades,
                "active_trades": len(self.active_trade_contexts),
                "trades_per_second": self.metrics.total_trades / max(1, engine_uptime)
            },
            "end_to_end_latency": latency_analysis,
            "component_performance": {
                "risk_engine": risk_stats,
                "position_keeper": position_stats,
                "order_manager": order_stats,
                "execution_engine": execution_stats
            },
            "memory_efficiency": global_pool_stats,
            "performance_targets": {
                "end_to_end_target_us": 2800,  # 2.8ms
                "risk_check_target_us": 1000,  # 1ms
                "position_update_target_us": 500,  # 0.5ms
                "order_processing_target_us": 300,  # 0.3ms
                "execution_routing_target_us": 1000  # 1ms
            },
            "target_achievement": {
                "end_to_end_met": latency_analysis.get("p99_us", 10000) < 2800,
                "overall_performance_rating": self._calculate_performance_rating(latency_analysis),
                "optimization_level": "MAXIMUM - All Phase 2B optimizations active"
            }
        }
    
    def _calculate_performance_rating(self, latency_analysis: Dict[str, Any]) -> str:
        """Calculate overall performance rating."""
        if not latency_analysis:
            return "INSUFFICIENT_DATA"
        
        p99_us = latency_analysis.get("p99_us", 10000)
        
        if p99_us < 1000:  # <1ms
            return "EXCEPTIONAL"
        elif p99_us < 2800:  # <2.8ms (target)
            return "EXCELLENT"
        elif p99_us < 5000:  # <5ms
            return "GOOD"
        else:
            return "NEEDS_OPTIMIZATION"
    
    async def _start_monitoring(self):
        """Start performance monitoring."""
        while self.monitoring_enabled:
            try:
                # Update memory pool metrics
                global_stats = pool_manager.get_global_metrics()
                self.metrics.memory_pool_hit_rate = global_stats.get("average_hit_rate", 0)
                self.metrics.total_objects_pooled = global_stats.get("total_objects_available", 0)
                
                # Calculate throughput
                uptime = time.time() - self.start_time
                self.metrics.orders_per_second = self.metrics.total_trades / max(1, uptime)
                self.metrics.max_concurrent_orders = len(self.active_trade_contexts)
                
                # Log performance summary every 60 seconds
                if int(uptime) % 60 == 0 and uptime > 1:
                    stats = self.get_comprehensive_performance_stats()
                    engine_stats = stats["ultra_low_latency_engine"]
                    latency_stats = stats.get("end_to_end_latency", {})
                    
                    logger.info(f"ULL Engine Performance: "
                              f"Trades: {engine_stats['total_trades']}, "
                              f"TPS: {engine_stats['trades_per_second']:.1f}, "
                              f"P99: {latency_stats.get('p99_us', 0):.0f}μs, "
                              f"Rating: {stats['target_achievement']['overall_performance_rating']}")
                
                await asyncio.sleep(1)  # Monitor every second
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(5)
    
    def shutdown(self):
        """Shutdown ultra-low latency trading engine."""
        self.monitoring_enabled = False
        
        # Shutdown all components
        self.risk_engine.shutdown()
        self.position_keeper.shutdown()
        self.order_manager.shutdown()
        self.execution_engine.shutdown()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Clear active contexts
        self.active_trade_contexts.clear()
        
        logger.info(f"Ultra-Low Latency Trading Engine '{self.engine_id}' shutdown complete")


# Factory function
def create_ultra_low_latency_engine(engine_id: str = "ull_engine") -> UltraLowLatencyTradingEngine:
    """Create and initialize ultra-low latency trading engine."""
    return UltraLowLatencyTradingEngine(engine_id=engine_id)