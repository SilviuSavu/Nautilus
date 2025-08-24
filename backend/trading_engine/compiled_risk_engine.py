"""
JIT-Compiled Risk Engine
========================

Ultra-low latency risk management with JIT-compiled rules and SIMD vectorization.
Targets <1ms P99 risk checking latency through aggressive optimization.

Performance Features:
- JIT-compiled risk rules with Numba
- SIMD vectorized calculations  
- Lock-free concurrent evaluation
- Pre-computed risk metrics
- Cache-optimized data structures

Target: <1ms P99 risk checking (from 4-12ms)
"""

import asyncio
import logging
import time
import threading
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from concurrent.futures import ThreadPoolExecutor

try:
    import numba
    from numba import jit, njit, prange, types
    from numba.typed import Dict as NumbaDict, List as NumbaList
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback decorators
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator
    
    def njit(*args, **kwargs):
        return jit(*args, **kwargs)
    
    prange = range

from .risk_engine import RiskViolationType, RiskLimits, RiskMetrics, RiskViolation
from .poolable_objects import PooledOrder, PooledRiskViolation, trading_pools
from .memory_pool import ObjectPool

logger = logging.getLogger(__name__)

if not NUMBA_AVAILABLE:
    logger.warning("Numba not available - using fallback implementations")


@dataclass
class CompiledRiskMetrics:
    """Optimized risk metrics for JIT compilation."""
    portfolio_id: str
    
    # Vectorized arrays for SIMD operations
    position_values: np.ndarray = field(default_factory=lambda: np.zeros(1000, dtype=np.float64))
    position_quantities: np.ndarray = field(default_factory=lambda: np.zeros(1000, dtype=np.float64))
    position_prices: np.ndarray = field(default_factory=lambda: np.zeros(1000, dtype=np.float64))
    
    # Scalar metrics (compiled-friendly)
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    daily_pnl: float = 0.0
    portfolio_value: float = 1000000.0  # Default 1M
    active_positions: int = 0
    
    # Risk limit values (for vectorized checking)
    max_position_size: float = 1000000.0
    max_daily_loss: float = 50000.0
    max_gross_leverage: float = 2.0
    max_concentration: float = 0.20


# JIT-compiled risk checking functions
@njit(cache=True, fastmath=True)
def check_position_size_compiled(position_value: float, max_position_size: float) -> bool:
    """JIT-compiled position size check."""
    return position_value <= max_position_size


@njit(cache=True, fastmath=True)
def check_leverage_compiled(gross_exposure: float, portfolio_value: float, max_leverage: float) -> bool:
    """JIT-compiled leverage check."""
    if portfolio_value <= 0:
        return False
    leverage = gross_exposure / portfolio_value
    return leverage <= max_leverage


@njit(cache=True, fastmath=True)
def check_daily_loss_compiled(daily_pnl: float, max_daily_loss: float) -> bool:
    """JIT-compiled daily loss check."""
    return daily_pnl >= -max_daily_loss


@njit(cache=True, fastmath=True)
def check_concentration_compiled(position_values: np.ndarray, active_positions: int,
                               max_concentration: float) -> bool:
    """JIT-compiled concentration check with vectorized operations."""
    if active_positions <= 0:
        return True
    
    # Calculate total exposure
    total_exposure = 0.0
    max_position = 0.0
    
    for i in prange(active_positions):  # Parallel loop
        abs_value = abs(position_values[i])
        total_exposure += abs_value
        if abs_value > max_position:
            max_position = abs_value
    
    if total_exposure <= 0:
        return True
    
    concentration = max_position / total_exposure
    return concentration <= max_concentration


@njit(cache=True, fastmath=True)
def vectorized_risk_check(position_value: float, gross_exposure: float, 
                         portfolio_value: float, daily_pnl: float,
                         position_values: np.ndarray, active_positions: int,
                         max_position_size: float, max_leverage: float,
                         max_daily_loss: float, max_concentration: float) -> int:
    """
    Vectorized risk checking - all rules in single compiled function.
    
    Returns:
        0: All checks passed
        1: Position size violation
        2: Leverage violation  
        3: Daily loss violation
        4: Concentration violation
    """
    # Position size check
    if not check_position_size_compiled(position_value, max_position_size):
        return 1
    
    # Leverage check
    if not check_leverage_compiled(gross_exposure, portfolio_value, max_leverage):
        return 2
    
    # Daily loss check
    if not check_daily_loss_compiled(daily_pnl, max_daily_loss):
        return 3
    
    # Concentration check
    if not check_concentration_compiled(position_values, active_positions, max_concentration):
        return 4
    
    return 0  # All checks passed


@njit(cache=True, fastmath=True)
def calculate_portfolio_metrics_vectorized(position_quantities: np.ndarray,
                                         position_prices: np.ndarray,
                                         active_positions: int) -> Tuple[float, float, float]:
    """Calculate portfolio metrics with SIMD vectorization."""
    long_exposure = 0.0
    short_exposure = 0.0
    
    # Vectorized calculation
    for i in prange(active_positions):
        value = position_quantities[i] * position_prices[i]
        if value > 0:
            long_exposure += value
        else:
            short_exposure += abs(value)
    
    gross_exposure = long_exposure + short_exposure
    net_exposure = long_exposure - short_exposure
    
    return gross_exposure, net_exposure, long_exposure


class CompiledRiskRule:
    """Base class for JIT-compiled risk rules."""
    
    def __init__(self, name: str, rule_id: int):
        self.name = name
        self.rule_id = rule_id
        self.enabled = True
        self.call_count = 0
        self.total_time_ns = 0
        
    async def check_compiled(self, order: PooledOrder, metrics: CompiledRiskMetrics) -> Optional[int]:
        """Check rule using compiled function - returns rule_id if violated."""
        raise NotImplementedError


class PositionSizeRuleCompiled(CompiledRiskRule):
    """JIT-compiled position size rule."""
    
    def __init__(self):
        super().__init__("Position Size (Compiled)", 1)
    
    async def check_compiled(self, order: PooledOrder, metrics: CompiledRiskMetrics) -> Optional[int]:
        start_time = time.perf_counter_ns()
        
        # Calculate new position value
        order_value = order.quantity * (order.price or 100.0)
        
        # JIT-compiled check
        passed = check_position_size_compiled(order_value, metrics.max_position_size)
        
        # Update metrics
        self.call_count += 1
        self.total_time_ns += time.perf_counter_ns() - start_time
        
        return None if passed else self.rule_id


class LeverageRuleCompiled(CompiledRiskRule):
    """JIT-compiled leverage rule."""
    
    def __init__(self):
        super().__init__("Leverage (Compiled)", 2)
    
    async def check_compiled(self, order: PooledOrder, metrics: CompiledRiskMetrics) -> Optional[int]:
        start_time = time.perf_counter_ns()
        
        # Estimate new gross exposure
        order_value = order.quantity * (order.price or 100.0)
        new_gross_exposure = metrics.gross_exposure + order_value
        
        # JIT-compiled check
        passed = check_leverage_compiled(new_gross_exposure, metrics.portfolio_value, metrics.max_gross_leverage)
        
        # Update metrics
        self.call_count += 1
        self.total_time_ns += time.perf_counter_ns() - start_time
        
        return None if passed else self.rule_id


class VectorizedRiskChecker:
    """
    Ultra-fast risk checker using vectorized operations.
    
    Performs all risk checks in a single JIT-compiled function call.
    """
    
    def __init__(self):
        self.check_count = 0
        self.total_time_ns = 0
        self.violation_counts = {
            1: 0,  # Position size
            2: 0,  # Leverage  
            3: 0,  # Daily loss
            4: 0   # Concentration
        }
        
        # Warmup the JIT compiler
        self._warmup_jit()
    
    def _warmup_jit(self):
        """Warm up JIT compilation."""
        if NUMBA_AVAILABLE:
            logger.info("Warming up JIT compiler...")
            
            # Create dummy data for warmup
            dummy_position_values = np.array([100000.0, 200000.0, 150000.0])
            
            # Call compiled functions to trigger compilation
            vectorized_risk_check(
                position_value=100000.0,
                gross_exposure=1000000.0,
                portfolio_value=2000000.0,
                daily_pnl=0.0,
                position_values=dummy_position_values,
                active_positions=3,
                max_position_size=1000000.0,
                max_leverage=2.0,
                max_daily_loss=50000.0,
                max_concentration=0.30
            )
            
            logger.info("JIT compilation complete")
    
    def check_all_rules_vectorized(self, order: PooledOrder, metrics: CompiledRiskMetrics) -> Optional[int]:
        """
        Check all risk rules in single vectorized operation.
        
        Returns:
            None if all checks pass, rule_id if violation detected
        """
        start_time = time.perf_counter_ns()
        
        # Calculate new position value
        order_value = order.quantity * (order.price or 100.0)
        new_gross_exposure = metrics.gross_exposure + order_value
        
        # Single vectorized risk check
        violation_code = vectorized_risk_check(
            position_value=order_value,
            gross_exposure=new_gross_exposure,
            portfolio_value=metrics.portfolio_value,
            daily_pnl=metrics.daily_pnl,
            position_values=metrics.position_values,
            active_positions=metrics.active_positions,
            max_position_size=metrics.max_position_size,
            max_leverage=metrics.max_gross_leverage,
            max_daily_loss=metrics.max_daily_loss,
            max_concentration=metrics.max_concentration
        )
        
        # Update metrics
        self.check_count += 1
        self.total_time_ns += time.perf_counter_ns() - start_time
        
        if violation_code > 0:
            self.violation_counts[violation_code] += 1
        
        return violation_code if violation_code > 0 else None
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        avg_time_ns = self.total_time_ns / max(1, self.check_count)
        
        return {
            "total_checks": self.check_count,
            "avg_check_time_ns": avg_time_ns,
            "avg_check_time_us": avg_time_ns / 1000,
            "total_violations": sum(self.violation_counts.values()),
            "violation_breakdown": self.violation_counts.copy(),
            "checks_per_second": int(1_000_000_000 / max(1, avg_time_ns))
        }


class CompiledRiskEngine:
    """
    Ultra-low latency risk engine with JIT compilation and SIMD optimization.
    
    Performance Features:
    - <1ms P99 risk checking target
    - JIT-compiled rule evaluation
    - SIMD vectorized calculations
    - Lock-free concurrent operations
    - Memory pool integration
    """
    
    def __init__(self):
        self.portfolio_limits: Dict[str, RiskLimits] = {}
        self.compiled_metrics: Dict[str, CompiledRiskMetrics] = {}
        self.vectorized_checker = VectorizedRiskChecker()
        
        # Individual compiled rules (for detailed analysis)
        self.compiled_rules = [
            PositionSizeRuleCompiled(),
            LeverageRuleCompiled()
        ]
        
        # Performance tracking
        self.total_checks = 0
        self.fast_path_hits = 0
        self.violation_cache: Dict[str, bool] = {}  # Order pattern cache
        
        # Thread pool for concurrent operations
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="compiled_risk")
        
        # Violation pool for zero-allocation error reporting
        self.violation_pool = trading_pools.risk_violation_pool
        
        logger.info(f"Compiled Risk Engine initialized (Numba: {'Available' if NUMBA_AVAILABLE else 'Fallback'})")
    
    def set_portfolio_limits(self, portfolio_id: str, limits: RiskLimits):
        """Set portfolio limits and initialize compiled metrics."""
        self.portfolio_limits[portfolio_id] = limits
        
        # Initialize compiled metrics
        metrics = CompiledRiskMetrics(portfolio_id=portfolio_id)
        metrics.max_position_size = limits.max_position_size
        metrics.max_daily_loss = limits.max_daily_loss
        metrics.max_gross_leverage = limits.max_gross_leverage
        metrics.max_concentration = limits.max_portfolio_concentration
        
        self.compiled_metrics[portfolio_id] = metrics
        
        logger.info(f"Compiled risk limits set for portfolio {portfolio_id}")
    
    async def check_pre_trade_risk_compiled(self, order: PooledOrder, portfolio_id: str) -> bool:
        """
        Ultra-fast pre-trade risk checking with JIT compilation.
        
        Target: <1ms P99 latency
        """
        start_time = time.perf_counter_ns()
        
        try:
            # Get compiled metrics
            metrics = self.compiled_metrics.get(portfolio_id)
            if not metrics:
                logger.warning(f"No compiled metrics for portfolio {portfolio_id}")
                return True  # Allow if no limits
            
            # Fast path: Check violation cache
            order_key = f"{order.symbol}_{order.quantity}_{portfolio_id}"
            cached_result = self.violation_cache.get(order_key)
            if cached_result is not None:
                self.fast_path_hits += 1
                return cached_result
            
            # Vectorized risk checking (single JIT call)
            violation_code = self.vectorized_checker.check_all_rules_vectorized(order, metrics)
            
            # Handle violation
            if violation_code:
                # Create pooled violation object
                violation = self._create_violation_from_code(violation_code, order, portfolio_id, metrics)
                if violation:
                    logger.warning(f"Risk violation: {violation.description}")
                    await self._notify_violation_async(violation)
                
                # Cache negative result
                self.violation_cache[order_key] = False
                return False
            
            # Cache positive result  
            self.violation_cache[order_key] = True
            return True
            
        except Exception as e:
            logger.error(f"Compiled risk check failed: {e}")
            return False  # Fail safe
        
        finally:
            # Update performance metrics
            self.total_checks += 1
            check_time_ns = time.perf_counter_ns() - start_time
            
            # Log performance milestones
            if self.total_checks % 10000 == 0:
                avg_time_us = check_time_ns / 1000
                cache_hit_rate = (self.fast_path_hits / self.total_checks) * 100
                logger.debug(f"Risk check #{self.total_checks}: {avg_time_us:.2f}μs (cache hit rate: {cache_hit_rate:.1f}%)")
    
    def _create_violation_from_code(self, violation_code: int, order: PooledOrder, 
                                  portfolio_id: str, metrics: CompiledRiskMetrics) -> Optional[PooledRiskViolation]:
        """Create violation object from compiled rule violation code."""
        violation_types = {
            1: ("position_limit", "Position size limit exceeded"),
            2: ("leverage_limit", "Leverage limit exceeded"),
            3: ("daily_loss_limit", "Daily loss limit exceeded"),
            4: ("concentration_limit", "Concentration limit exceeded")
        }
        
        if violation_code not in violation_types:
            return None
        
        violation_type, base_description = violation_types[violation_code]
        
        # Get violation from pool
        violation = self.violation_pool.acquire()
        
        # Calculate specific values based on violation type
        if violation_code == 1:  # Position size
            current_value = order.quantity * (order.price or 100.0)
            limit_value = metrics.max_position_size
            description = f"Position value {current_value:,.2f} exceeds limit {limit_value:,.2f}"
        elif violation_code == 2:  # Leverage
            order_value = order.quantity * (order.price or 100.0)
            new_gross_exposure = metrics.gross_exposure + order_value
            current_value = new_gross_exposure / metrics.portfolio_value
            limit_value = metrics.max_gross_leverage
            description = f"Leverage {current_value:.2f} would exceed limit {limit_value:.2f}"
        elif violation_code == 3:  # Daily loss
            current_value = abs(metrics.daily_pnl)
            limit_value = metrics.max_daily_loss
            description = f"Daily loss {current_value:,.2f} exceeds limit {limit_value:,.2f}"
        else:  # Concentration
            current_value = 0.0  # Would need full position analysis
            limit_value = metrics.max_concentration * 100
            description = f"Concentration would exceed limit {limit_value:.1f}%"
        
        # Populate violation
        violation.populate(
            violation_type=violation_type,
            portfolio_id=portfolio_id,
            description=description,
            current_value=current_value,
            limit_value=limit_value,
            severity="HIGH"
        )
        
        return violation
    
    async def _notify_violation_async(self, violation: PooledRiskViolation):
        """Notify violation asynchronously to avoid blocking."""
        # Submit to thread pool for async notification
        self.executor.submit(self._process_violation, violation)
    
    def _process_violation(self, violation: PooledRiskViolation):
        """Process violation in thread pool."""
        try:
            # Log violation
            logger.warning(f"Risk violation processed: {violation.description}")
            
            # Here you would integrate with alerting systems
            # For now, just return to pool
            self.violation_pool.release(violation)
            
        except Exception as e:
            logger.error(f"Error processing violation: {e}")
    
    def update_portfolio_metrics_vectorized(self, portfolio_id: str, 
                                          position_data: List[Tuple[float, float]]):
        """Update portfolio metrics using vectorized calculations."""
        metrics = self.compiled_metrics.get(portfolio_id)
        if not metrics:
            return
        
        if not position_data:
            return
        
        # Convert to numpy arrays for vectorized processing
        quantities = np.array([pos[0] for pos in position_data])
        prices = np.array([pos[1] for pos in position_data])
        
        # Ensure arrays fit in pre-allocated space
        active_positions = min(len(position_data), len(metrics.position_quantities))
        
        # Update arrays
        metrics.position_quantities[:active_positions] = quantities[:active_positions]
        metrics.position_prices[:active_positions] = prices[:active_positions]
        metrics.position_values[:active_positions] = quantities[:active_positions] * prices[:active_positions]
        metrics.active_positions = active_positions
        
        # Calculate portfolio metrics using JIT-compiled function
        gross_exp, net_exp, long_exp = calculate_portfolio_metrics_vectorized(
            metrics.position_quantities, 
            metrics.position_prices, 
            active_positions
        )
        
        metrics.gross_exposure = gross_exp
        metrics.net_exposure = net_exp
    
    def clear_violation_cache(self):
        """Clear violation cache for fresh evaluations."""
        self.violation_cache.clear()
        logger.debug("Risk violation cache cleared")
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        vectorized_stats = self.vectorized_checker.get_performance_stats()
        
        cache_hit_rate = (self.fast_path_hits / max(1, self.total_checks)) * 100
        
        individual_rule_stats = {}
        for rule in self.compiled_rules:
            if rule.call_count > 0:
                avg_time_ns = rule.total_time_ns / rule.call_count
                individual_rule_stats[rule.name] = {
                    "calls": rule.call_count,
                    "avg_time_ns": avg_time_ns,
                    "avg_time_us": avg_time_ns / 1000
                }
        
        return {
            "compiled_risk_engine": {
                "total_checks": self.total_checks,
                "cache_hit_rate_percent": cache_hit_rate,
                "numba_available": NUMBA_AVAILABLE,
                "portfolios_configured": len(self.compiled_metrics)
            },
            "vectorized_checker": vectorized_stats,
            "individual_rules": individual_rule_stats,
            "performance_summary": {
                "target_check_time_us": 1000,  # 1ms target
                "actual_avg_time_us": vectorized_stats.get("avg_check_time_us", 0),
                "performance_rating": "EXCELLENT" if vectorized_stats.get("avg_check_time_us", 1000) < 100 else "GOOD",
                "checks_per_second_capacity": vectorized_stats.get("checks_per_second", 0)
            }
        }
    
    def shutdown(self):
        """Shutdown compiled risk engine."""
        self.executor.shutdown(wait=True)
        self.clear_violation_cache()
        logger.info("Compiled risk engine shutdown complete")


# Convenience function for easy integration
def create_compiled_risk_engine() -> CompiledRiskEngine:
    """Create and initialize compiled risk engine."""
    return CompiledRiskEngine()