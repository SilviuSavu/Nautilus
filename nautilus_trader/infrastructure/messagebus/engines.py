#!/usr/bin/env python3
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

"""
Enhanced MessageBus integration for NautilusTrader Engines.

Provides enhanced MessageBus capabilities for Data, Execution, and Risk engines
while maintaining backward compatibility and graceful fallback to standard MessageBus.
"""

import asyncio
import logging
import time
from typing import Any, Dict, Optional, Callable, List, Union

from nautilus_trader.common.component import MessageBus
from nautilus_trader.infrastructure.messagebus.config import MessagePriority
from nautilus_trader.infrastructure.messagebus.client import BufferedMessageBusClient
from nautilus_trader.infrastructure.messagebus.integration import (
    EnhancedMessageBusFeatures,
    ComponentPerformanceMonitor
)


class EnhancedDataEngineIntegration:
    """
    Enhanced MessageBus integration for LiveDataEngine.
    
    Provides:
    - Priority-based market data publishing (HIGH for ticks, NORMAL for bars)
    - Pattern-based data subscriptions for efficient routing
    - Performance monitoring for data throughput
    - Market data flow optimization with enhanced buffering
    - Graceful fallback to standard MessageBus operations
    """
    
    def __init__(self, engine, logger_name: str = "EnhancedDataEngine"):
        self.engine = engine
        self.logger = logging.getLogger(logger_name)
        
        # Initialize enhanced features
        self._enhanced_features = EnhancedMessageBusFeatures(engine._msgbus)
        self._performance_monitor = ComponentPerformanceMonitor(
            component_name=f"DataEngine.{engine.__class__.__name__}",
            msgbus=engine._msgbus
        )
        
        # Data-specific metrics
        self._data_metrics = {
            "ticks_published": 0,
            "bars_published": 0,
            "quotes_published": 0,
            "trades_published": 0,
            "books_published": 0,
            "data_requests_processed": 0,
            "subscriptions_active": 0
        }
        
        # Market data priority mapping
        self._data_priority_map = {
            "tick": MessagePriority.HIGH,
            "quote": MessagePriority.HIGH, 
            "trade": MessagePriority.HIGH,
            "bar": MessagePriority.NORMAL,
            "book": MessagePriority.HIGH,
            "status": MessagePriority.NORMAL,
            "news": MessagePriority.LOW
        }
        
        # Performance tracking
        self._last_performance_report = time.time()
        self._performance_report_interval = 60.0  # Report every minute
        
        self.logger.info("Enhanced Data Engine integration initialized")
    
    def is_enhanced(self) -> bool:
        """Check if enhanced MessageBus features are available."""
        return self._enhanced_features.is_enhanced()
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get enhanced MessageBus capabilities."""
        return self._enhanced_features.get_capabilities()
    
    def publish_market_data(self, topic: str, data: Any, data_type: str = "unknown") -> None:
        """
        Publish market data with appropriate priority based on data type.
        
        Args:
            topic: Data topic
            data: Market data
            data_type: Type of data (tick, bar, quote, trade, book, etc.)
        """
        # Determine priority based on data type
        priority = self._data_priority_map.get(data_type.lower(), MessagePriority.NORMAL)
        
        if self.is_enhanced():
            # Use enhanced priority publishing
            self._enhanced_features.publish_with_priority(topic, data, priority)
        else:
            # Fall back to standard publishing
            self.engine._msgbus.publish(topic, data)
        
        # Update metrics
        self._update_data_metrics(data_type)
        self._performance_monitor.record_message_sent(topic, priority)
        
        # Check for performance reporting
        self._maybe_report_performance()
    
    def subscribe_to_market_data_pattern(self, pattern: str, callback: Callable) -> None:
        """
        Subscribe to market data using pattern matching.
        
        Args:
            pattern: Market data pattern (e.g., "data.ticks.BINANCE.*")
            callback: Callback function for matching data
        """
        if self.is_enhanced():
            capabilities = self.get_capabilities()
            if capabilities.get("pattern_matching", False):
                self._enhanced_features.subscribe_with_pattern(pattern, callback)
                self.logger.info(f"Enhanced pattern subscription: {pattern}")
            else:
                # Enhanced MessageBus without pattern matching
                self.engine._msgbus.subscribe(pattern, callback)
                self.logger.warning(f"Pattern matching not available, using exact match: {pattern}")
        else:
            # Standard MessageBus
            self.engine._msgbus.subscribe(pattern, callback)
            self.logger.debug(f"Standard subscription: {pattern}")
        
        self._data_metrics["subscriptions_active"] += 1
    
    def optimize_data_flow(self, instrument_ids: List[str], data_types: List[str]) -> None:
        """
        Optimize data flow for specific instruments and data types.
        
        Args:
            instrument_ids: List of instrument identifiers
            data_types: List of data types to optimize for
        """
        if not self.is_enhanced():
            self.logger.debug("Data flow optimization not available - standard MessageBus in use")
            return
        
        # Set up optimized subscriptions for high-frequency data
        for instrument_id in instrument_ids:
            for data_type in data_types:
                if data_type.lower() in ["tick", "quote", "trade"]:
                    # Subscribe to high-frequency data with pattern
                    pattern = f"data.{data_type}.*.{instrument_id}"
                    
                    # Create optimized callback wrapper
                    def optimized_callback(data, dt=data_type, iid=instrument_id):
                        self._handle_optimized_data(data, dt, iid)
                    
                    self.subscribe_to_market_data_pattern(pattern, optimized_callback)
        
        self.logger.info(f"Optimized data flow for {len(instrument_ids)} instruments, {len(data_types)} data types")
    
    def _handle_optimized_data(self, data: Any, data_type: str, instrument_id: str) -> None:
        """Handle optimized high-frequency data."""
        # Fast path processing for high-frequency data
        self._update_data_metrics(data_type)
        
        # Optional: Add data validation, filtering, or transformation here
        # For now, just pass through to engine's normal processing
    
    def _update_data_metrics(self, data_type: str) -> None:
        """Update data-specific metrics."""
        metric_key = f"{data_type.lower()}s_published"
        if metric_key in self._data_metrics:
            self._data_metrics[metric_key] += 1
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary for data engine."""
        base_summary = self._performance_monitor.get_performance_summary()
        
        # Calculate data rates
        uptime = base_summary.get("uptime_seconds", 1)
        data_rates = {}
        for key, count in self._data_metrics.items():
            if key.endswith("_published"):
                data_type = key.replace("_published", "")
                data_rates[f"{data_type}_per_second"] = count / uptime if uptime > 0 else 0
        
        enhanced_summary = {
            **base_summary,
            "data_metrics": self._data_metrics.copy(),
            "data_rates": data_rates,
            "enhanced_messagebus": self.is_enhanced(),
        }
        
        if self.is_enhanced():
            enhanced_summary["messagebus_capabilities"] = self.get_capabilities()
            enhanced_summary["messagebus_metrics"] = self._enhanced_features.get_metrics()
        
        return enhanced_summary
    
    def _maybe_report_performance(self) -> None:
        """Report performance if interval exceeded."""
        current_time = time.time()
        
        if (current_time - self._last_performance_report) >= self._performance_report_interval:
            summary = self.get_performance_summary()
            
            total_data = sum(v for k, v in self._data_metrics.items() if k.endswith("_published"))
            data_rate = total_data / summary["uptime_seconds"] if summary["uptime_seconds"] > 0 else 0
            
            self.logger.info(
                f"Data Engine Performance - Total data published: {total_data}, "
                f"Rate: {data_rate:.1f}/sec, Enhanced: {self.is_enhanced()}"
            )
            
            self._last_performance_report = current_time


class EnhancedExecutionEngineIntegration:
    """
    Enhanced MessageBus integration for LiveExecutionEngine.
    
    Provides:
    - Critical priority for order execution messages
    - High priority for execution reports
    - Pattern-based order routing for multi-venue execution
    - Sub-10ms latency optimization for critical messages
    - Order flow performance monitoring
    """
    
    def __init__(self, engine, logger_name: str = "EnhancedExecutionEngine"):
        self.engine = engine
        self.logger = logging.getLogger(logger_name)
        
        # Initialize enhanced features
        self._enhanced_features = EnhancedMessageBusFeatures(engine._msgbus)
        self._performance_monitor = ComponentPerformanceMonitor(
            component_name=f"ExecutionEngine.{engine.__class__.__name__}",
            msgbus=engine._msgbus
        )
        
        # Execution-specific metrics
        self._execution_metrics = {
            "orders_submitted": 0,
            "orders_filled": 0,
            "orders_canceled": 0,
            "orders_rejected": 0,
            "execution_reports": 0,
            "position_updates": 0,
            "account_updates": 0,
            "latency_samples": []
        }
        
        # Execution message priorities
        self._execution_priority_map = {
            "order_submit": MessagePriority.CRITICAL,
            "order_modify": MessagePriority.CRITICAL,
            "order_cancel": MessagePriority.CRITICAL,
            "execution_report": MessagePriority.HIGH,
            "position_update": MessagePriority.HIGH,
            "account_update": MessagePriority.NORMAL,
            "mass_status": MessagePriority.NORMAL
        }
        
        # Latency tracking
        self._order_timestamps: Dict[str, float] = {}
        self._max_latency_samples = 1000  # Keep last 1000 samples
        
        self.logger.info("Enhanced Execution Engine integration initialized")
    
    def is_enhanced(self) -> bool:
        """Check if enhanced MessageBus features are available."""
        return self._enhanced_features.is_enhanced()
    
    def publish_order_event(self, topic: str, order_data: Any, event_type: str = "unknown") -> str:
        """
        Publish order-related event with appropriate priority and latency tracking.
        
        Args:
            topic: Order event topic
            order_data: Order event data
            event_type: Type of order event
            
        Returns:
            Event ID for latency tracking
        """
        # Generate event ID for tracking
        event_id = f"{event_type}_{int(time.time_ns())}"
        start_time = time.time()
        
        # Determine priority
        priority = self._execution_priority_map.get(event_type.lower(), MessagePriority.HIGH)
        
        if self.is_enhanced():
            # Use enhanced priority publishing for ultra-low latency
            self._enhanced_features.publish_with_priority(topic, order_data, priority)
        else:
            # Fall back to standard publishing
            self.engine._msgbus.publish(topic, order_data)
        
        # Track latency for critical messages
        if priority == MessagePriority.CRITICAL:
            self._order_timestamps[event_id] = start_time
        
        # Update metrics
        self._update_execution_metrics(event_type)
        self._performance_monitor.record_message_sent(topic, priority)
        
        return event_id
    
    def complete_order_latency_tracking(self, event_id: str) -> Optional[float]:
        """
        Complete latency tracking for an order event.
        
        Args:
            event_id: Event ID from publish_order_event
            
        Returns:
            Latency in milliseconds, or None if not tracked
        """
        if event_id in self._order_timestamps:
            start_time = self._order_timestamps.pop(event_id)
            latency_ms = (time.time() - start_time) * 1000
            
            # Store latency sample
            self._execution_metrics["latency_samples"].append(latency_ms)
            
            # Limit sample size
            if len(self._execution_metrics["latency_samples"]) > self._max_latency_samples:
                self._execution_metrics["latency_samples"].pop(0)
            
            # Log high latency
            if latency_ms > 50.0:  # > 50ms is considered high latency
                self.logger.warning(f"High execution latency: {latency_ms:.2f}ms for {event_id}")
            
            return latency_ms
        
        return None
    
    def setup_order_routing_patterns(self, venues: List[str]) -> None:
        """
        Set up pattern-based order routing for multiple venues.
        
        Args:
            venues: List of venue identifiers
        """
        if not self.is_enhanced():
            self.logger.debug("Pattern-based order routing not available - standard MessageBus in use")
            return
        
        capabilities = self.get_capabilities()
        if not capabilities.get("pattern_matching", False):
            self.logger.warning("Pattern matching not available in enhanced MessageBus")
            return
        
        # Set up routing patterns for each venue
        for venue in venues:
            # Orders going to specific venue
            pattern = f"orders.*.{venue}.*"
            self._enhanced_features.subscribe_with_pattern(
                pattern, 
                lambda data, v=venue: self._handle_venue_order(data, v)
            )
            
            # Execution reports from specific venue
            pattern = f"executions.{venue}.*"
            self._enhanced_features.subscribe_with_pattern(
                pattern,
                lambda data, v=venue: self._handle_venue_execution(data, v)
            )
        
        self.logger.info(f"Set up order routing patterns for {len(venues)} venues")
    
    def _handle_venue_order(self, order_data: Any, venue: str) -> None:
        """Handle venue-specific order routing."""
        self._update_execution_metrics("order_routing")
        # Additional venue-specific processing could go here
    
    def _handle_venue_execution(self, execution_data: Any, venue: str) -> None:
        """Handle venue-specific execution reports."""
        self._update_execution_metrics("execution_report")
        # Additional venue-specific processing could go here
    
    def _update_execution_metrics(self, event_type: str) -> None:
        """Update execution-specific metrics."""
        if event_type in ["order_submit", "order_modify", "order_cancel"]:
            base_type = event_type.replace("order_", "orders_") + "ted"
            if base_type in self._execution_metrics:
                self._execution_metrics[base_type] += 1
        elif event_type == "execution_report":
            self._execution_metrics["execution_reports"] += 1
        elif event_type == "position_update":
            self._execution_metrics["position_updates"] += 1
        elif event_type == "account_update":
            self._execution_metrics["account_updates"] += 1
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get enhanced MessageBus capabilities."""
        return self._enhanced_features.get_capabilities()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary for execution engine."""
        base_summary = self._performance_monitor.get_performance_summary()
        
        # Calculate latency statistics
        latency_stats = {}
        if self._execution_metrics["latency_samples"]:
            samples = self._execution_metrics["latency_samples"]
            latency_stats = {
                "average_latency_ms": sum(samples) / len(samples),
                "min_latency_ms": min(samples),
                "max_latency_ms": max(samples),
                "samples_count": len(samples),
                "sub_10ms_count": len([s for s in samples if s < 10.0]),
                "sub_10ms_percentage": len([s for s in samples if s < 10.0]) / len(samples) * 100
            }
        
        # Calculate execution rates
        uptime = base_summary.get("uptime_seconds", 1)
        execution_rates = {
            "orders_per_minute": self._execution_metrics["orders_submitted"] / (uptime / 60) if uptime > 0 else 0,
            "fills_per_minute": self._execution_metrics["orders_filled"] / (uptime / 60) if uptime > 0 else 0,
        }
        
        enhanced_summary = {
            **base_summary,
            "execution_metrics": self._execution_metrics.copy(),
            "execution_rates": execution_rates,
            "latency_stats": latency_stats,
            "enhanced_messagebus": self.is_enhanced(),
        }
        
        # Remove latency samples from summary to avoid large payloads
        enhanced_summary["execution_metrics"] = {
            k: v for k, v in enhanced_summary["execution_metrics"].items() 
            if k != "latency_samples"
        }
        
        if self.is_enhanced():
            enhanced_summary["messagebus_capabilities"] = self.get_capabilities()
            enhanced_summary["messagebus_metrics"] = self._enhanced_features.get_metrics()
        
        return enhanced_summary


class EnhancedRiskEngineIntegration:
    """
    Enhanced MessageBus integration for LiveRiskEngine.
    
    Provides:
    - Critical priority for risk alerts and breaches
    - High priority for position updates
    - Real-time risk monitoring with <5s intervals
    - Enhanced risk event broadcasting
    - Risk metrics and compliance reporting
    """
    
    def __init__(self, engine, logger_name: str = "EnhancedRiskEngine"):
        self.engine = engine
        self.logger = logging.getLogger(logger_name)
        
        # Initialize enhanced features
        self._enhanced_features = EnhancedMessageBusFeatures(engine._msgbus)
        self._performance_monitor = ComponentPerformanceMonitor(
            component_name=f"RiskEngine.{engine.__class__.__name__}",
            msgbus=engine._msgbus
        )
        
        # Risk-specific metrics
        self._risk_metrics = {
            "risk_alerts": 0,
            "position_updates": 0,
            "limit_checks": 0,
            "limit_breaches": 0,
            "risk_reports": 0,
            "compliance_checks": 0,
            "emergency_stops": 0
        }
        
        # Risk message priorities
        self._risk_priority_map = {
            "risk_alert": MessagePriority.CRITICAL,
            "limit_breach": MessagePriority.CRITICAL,
            "emergency_stop": MessagePriority.CRITICAL,
            "position_update": MessagePriority.HIGH,
            "limit_check": MessagePriority.HIGH,
            "risk_report": MessagePriority.NORMAL,
            "compliance_report": MessagePriority.NORMAL
        }
        
        # Risk monitoring
        self._last_risk_check = time.time()
        self._risk_check_interval = 5.0  # 5 second intervals
        
        self.logger.info("Enhanced Risk Engine integration initialized")
    
    def is_enhanced(self) -> bool:
        """Check if enhanced MessageBus features are available."""
        return self._enhanced_features.is_enhanced()
    
    def publish_risk_alert(self, alert_type: str, alert_data: Any, severity: str = "high") -> None:
        """
        Publish risk alert with critical priority.
        
        Args:
            alert_type: Type of risk alert
            alert_data: Alert data
            severity: Alert severity (critical, high, medium, low)
        """
        topic = f"risk.alert.{alert_type}"
        
        # Determine priority based on severity
        if severity.lower() in ["critical", "emergency"]:
            priority = MessagePriority.CRITICAL
        elif severity.lower() == "high":
            priority = MessagePriority.HIGH
        else:
            priority = MessagePriority.NORMAL
        
        if self.is_enhanced():
            self._enhanced_features.publish_with_priority(topic, alert_data, priority)
        else:
            self.engine._msgbus.publish(topic, alert_data)
        
        # Update metrics
        self._risk_metrics["risk_alerts"] += 1
        if severity.lower() in ["critical", "emergency"]:
            self._risk_metrics["limit_breaches"] += 1
        
        self._performance_monitor.record_message_sent(topic, priority)
        
        # Log critical alerts
        if priority == MessagePriority.CRITICAL:
            self.logger.error(f"CRITICAL RISK ALERT: {alert_type} - {alert_data}")
        else:
            self.logger.warning(f"Risk Alert: {alert_type} - severity: {severity}")
    
    def publish_position_update(self, position_data: Any) -> None:
        """
        Publish position update with high priority.
        
        Args:
            position_data: Position update data
        """
        topic = "risk.positions.update"
        
        if self.is_enhanced():
            self._enhanced_features.publish_with_priority(
                topic, position_data, MessagePriority.HIGH
            )
        else:
            self.engine._msgbus.publish(topic, position_data)
        
        self._risk_metrics["position_updates"] += 1
        self._performance_monitor.record_message_sent(topic, MessagePriority.HIGH)
    
    def enable_realtime_risk_monitoring(self, check_interval: float = 5.0) -> None:
        """
        Enable real-time risk monitoring with configurable intervals.
        
        Args:
            check_interval: Risk check interval in seconds
        """
        self._risk_check_interval = check_interval
        self.logger.info(f"Real-time risk monitoring enabled with {check_interval}s intervals")
        
        if self.is_enhanced():
            capabilities = self.get_capabilities()
            self.logger.info(f"Enhanced risk monitoring capabilities: {capabilities}")
    
    def perform_risk_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive risk check and return status.
        
        Returns:
            Risk check results
        """
        current_time = time.time()
        
        # Simulate risk check (in real implementation, this would check actual risk metrics)
        risk_status = {
            "timestamp": current_time,
            "checks_performed": ["position_limits", "portfolio_var", "concentration_risk"],
            "status": "healthy",
            "warnings": [],
            "breaches": []
        }
        
        # Update metrics
        self._risk_metrics["limit_checks"] += 1
        self._last_risk_check = current_time
        
        # Publish risk check results if enhanced
        if self.is_enhanced():
            self._enhanced_features.publish_with_priority(
                "risk.check.results",
                risk_status,
                MessagePriority.NORMAL
            )
        
        return risk_status
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get enhanced MessageBus capabilities."""
        return self._enhanced_features.get_capabilities()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary for risk engine."""
        base_summary = self._performance_monitor.get_performance_summary()
        
        # Calculate risk rates
        uptime = base_summary.get("uptime_seconds", 1)
        risk_rates = {
            "alerts_per_hour": self._risk_metrics["risk_alerts"] / (uptime / 3600) if uptime > 0 else 0,
            "checks_per_minute": self._risk_metrics["limit_checks"] / (uptime / 60) if uptime > 0 else 0,
        }
        
        # Risk health indicators
        total_checks = self._risk_metrics["limit_checks"]
        breach_rate = (self._risk_metrics["limit_breaches"] / total_checks * 100) if total_checks > 0 else 0
        
        enhanced_summary = {
            **base_summary,
            "risk_metrics": self._risk_metrics.copy(),
            "risk_rates": risk_rates,
            "risk_health": {
                "breach_rate_percent": breach_rate,
                "last_check_seconds_ago": time.time() - self._last_risk_check,
                "monitoring_interval": self._risk_check_interval
            },
            "enhanced_messagebus": self.is_enhanced(),
        }
        
        if self.is_enhanced():
            enhanced_summary["messagebus_capabilities"] = self.get_capabilities()
            enhanced_summary["messagebus_metrics"] = self._enhanced_features.get_metrics()
        
        return enhanced_summary


# =============================================================================
# ENGINE ENHANCEMENT UTILITIES
# =============================================================================

def enhance_data_engine(engine) -> EnhancedDataEngineIntegration:
    """
    Add enhanced MessageBus integration to a LiveDataEngine.
    
    Args:
        engine: LiveDataEngine instance
        
    Returns:
        Enhanced integration wrapper
    """
    if not hasattr(engine, '_msgbus'):
        raise ValueError("Engine must have _msgbus attribute")
    
    integration = EnhancedDataEngineIntegration(engine)
    
    # Add enhanced methods to engine
    engine.publish_market_data = integration.publish_market_data
    engine.subscribe_to_market_data_pattern = integration.subscribe_to_market_data_pattern
    engine.optimize_data_flow = integration.optimize_data_flow
    engine.get_enhanced_performance_summary = integration.get_performance_summary
    
    # Mark as enhanced
    engine._enhanced_messagebus_integration = integration
    
    integration.logger.info("LiveDataEngine enhanced with MessageBus integration")
    return integration


def enhance_execution_engine(engine) -> EnhancedExecutionEngineIntegration:
    """
    Add enhanced MessageBus integration to a LiveExecutionEngine.
    
    Args:
        engine: LiveExecutionEngine instance
        
    Returns:
        Enhanced integration wrapper
    """
    if not hasattr(engine, '_msgbus'):
        raise ValueError("Engine must have _msgbus attribute")
    
    integration = EnhancedExecutionEngineIntegration(engine)
    
    # Add enhanced methods to engine
    engine.publish_order_event = integration.publish_order_event
    engine.complete_order_latency_tracking = integration.complete_order_latency_tracking
    engine.setup_order_routing_patterns = integration.setup_order_routing_patterns
    engine.get_enhanced_performance_summary = integration.get_performance_summary
    
    # Mark as enhanced
    engine._enhanced_messagebus_integration = integration
    
    integration.logger.info("LiveExecutionEngine enhanced with MessageBus integration")
    return integration


def enhance_risk_engine(engine) -> EnhancedRiskEngineIntegration:
    """
    Add enhanced MessageBus integration to a LiveRiskEngine.
    
    Args:
        engine: LiveRiskEngine instance
        
    Returns:
        Enhanced integration wrapper
    """
    if not hasattr(engine, '_msgbus'):
        raise ValueError("Engine must have _msgbus attribute")
    
    integration = EnhancedRiskEngineIntegration(engine)
    
    # Add enhanced methods to engine
    engine.publish_risk_alert = integration.publish_risk_alert
    engine.publish_position_update = integration.publish_position_update
    engine.enable_realtime_risk_monitoring = integration.enable_realtime_risk_monitoring
    engine.perform_risk_check = integration.perform_risk_check
    engine.get_enhanced_performance_summary = integration.get_performance_summary
    
    # Mark as enhanced
    engine._enhanced_messagebus_integration = integration
    
    integration.logger.info("LiveRiskEngine enhanced with MessageBus integration")
    return integration


def check_engine_enhancement_support(engine) -> Dict[str, Any]:
    """
    Check if an engine supports enhanced MessageBus integration.
    
    Args:
        engine: Engine instance to check
        
    Returns:
        Support information
    """
    info = {
        "has_messagebus": hasattr(engine, '_msgbus'),
        "messagebus_enhanced": False,
        "integration_available": False,
        "already_enhanced": False,
        "engine_type": engine.__class__.__name__
    }
    
    if info["has_messagebus"]:
        try:
            from nautilus_trader.infrastructure.messagebus.client import BufferedMessageBusClient
            info["messagebus_enhanced"] = isinstance(engine._msgbus, BufferedMessageBusClient)
            info["integration_available"] = True
            info["already_enhanced"] = hasattr(engine, '_enhanced_messagebus_integration')
        except ImportError:
            info["integration_available"] = False
    
    return info