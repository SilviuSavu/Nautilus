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
Enhanced MessageBus integration for NautilusTrader Data and Execution Adapters.

Provides enhanced MessageBus capabilities for adapter classes through mixins, decorators,
and utility functions while maintaining backward compatibility with existing adapters.
"""

import logging
import time
from typing import Any, Dict, Callable, Optional, Union

from nautilus_trader.infrastructure.messagebus.config import MessagePriority
from nautilus_trader.infrastructure.messagebus.integration import (
    MessageBusIntegrationMixin,
    EnhancedMessageBusFeatures,
    ComponentPerformanceMonitor,
)


class EnhancedDataAdapterMixin(MessageBusIntegrationMixin):
    """
    Enhanced MessageBus mixin specifically designed for Data Adapter classes.
    
    Provides Data Adapter-specific enhanced MessageBus features including:
    - Priority-based market data publishing with automatic priority assignment
    - Pattern-based subscriptions for efficient market data routing
    - Market data throughput monitoring and performance tracking
    - Exchange-specific optimizations for high-frequency data streams
    - Latency tracking for market data delivery
    
    Usage:
        class MyDataAdapter(LiveMarketDataClient, EnhancedDataAdapterMixin):
            def __init__(self, msgbus, ...):
                super().__init__(msgbus=msgbus, ...)
                self._setup_enhanced_data_adapter()
    """
    
    def _setup_enhanced_data_adapter(self) -> None:
        """Set up enhanced Data Adapter-specific MessageBus features."""
        # Call parent setup
        self._setup_enhanced_messagebus()
        
        # Data adapter-specific setup
        self._data_performance_monitor = ComponentPerformanceMonitor(
            component_name=f"DataAdapter.{self.__class__.__name__}",
            msgbus=self._msgbus
        )
        
        # Market data-specific message counters
        self._market_data_counters = {
            "ticks_published": 0,
            "quotes_published": 0,
            "trades_published": 0,
            "bars_published": 0,
            "book_updates_published": 0,
            "instruments_published": 0,
            "subscription_requests": 0,
            "data_bytes_processed": 0
        }
        
        # Latency tracking
        self._data_latency_tracker = {
            "last_tick_timestamp": 0,
            "last_quote_timestamp": 0,
            "last_trade_timestamp": 0,
            "avg_processing_latency_ms": 0.0
        }
        
        self._enhanced_logger.info("Enhanced Data Adapter features initialized")
    
    def publish_market_data(self, topic: str, data: Any, data_type: str = "unknown", 
                          priority: Optional[MessagePriority] = None) -> None:
        """
        Publish market data with intelligent priority assignment.
        
        Args:
            topic: Market data topic
            data: Market data object
            data_type: Type of market data (tick, quote, trade, bar, book, instrument)
            priority: Override priority (auto-assigned if None)
        """
        # Auto-assign priority based on data type
        if priority is None:
            priority = self._get_market_data_priority(data_type)
        
        # Track processing start time for latency measurement
        start_time = time.perf_counter()
        
        # Publish with priority
        self.publish_with_priority(topic, data, priority)
        
        # Update counters and latency tracking
        self._update_market_data_metrics(data_type, start_time)
        self._data_performance_monitor.record_message_sent(topic, priority)
    
    def subscribe_to_market_data_stream(self, exchange_pattern: str, callback: Callable) -> None:
        """
        Subscribe to market data using exchange-specific patterns.
        
        Args:
            exchange_pattern: Pattern like "binance.spot.*" or "*.trades.BTCUSDT"
            callback: Callback for matching market data
        """
        self.subscribe_with_pattern(exchange_pattern, callback)
        self._market_data_counters["subscription_requests"] += 1
        self._enhanced_logger.info(f"Subscribed to market data stream: {exchange_pattern}")
    
    def publish_tick_data(self, symbol: str, tick_data: Any, venue: str = None) -> None:
        """Publish tick data with HIGH priority for time-sensitive trading."""
        topic = f"data.ticks.{venue}.{symbol}" if venue else f"data.ticks.{symbol}"
        self.publish_market_data(topic, tick_data, "tick", MessagePriority.HIGH)
    
    def publish_trade_data(self, symbol: str, trade_data: Any, venue: str = None) -> None:
        """Publish trade data with HIGH priority for execution tracking."""
        topic = f"data.trades.{venue}.{symbol}" if venue else f"data.trades.{symbol}"
        self.publish_market_data(topic, trade_data, "trade", MessagePriority.HIGH)
    
    def publish_quote_data(self, symbol: str, quote_data: Any, venue: str = None) -> None:
        """Publish quote data with NORMAL priority for spread analysis."""
        topic = f"data.quotes.{venue}.{symbol}" if venue else f"data.quotes.{symbol}"
        self.publish_market_data(topic, quote_data, "quote", MessagePriority.NORMAL)
    
    def publish_book_update(self, symbol: str, book_data: Any, venue: str = None) -> None:
        """Publish order book updates with HIGH priority for liquidity tracking."""
        topic = f"data.book.{venue}.{symbol}" if venue else f"data.book.{symbol}"
        self.publish_market_data(topic, book_data, "book", MessagePriority.HIGH)
    
    def publish_bar_data(self, symbol: str, bar_data: Any, timeframe: str = None, venue: str = None) -> None:
        """Publish bar/candlestick data with NORMAL priority."""
        topic_parts = ["data.bars"]
        if venue:
            topic_parts.append(venue)
        if timeframe:
            topic_parts.append(timeframe)
        topic_parts.append(symbol)
        topic = ".".join(topic_parts)
        
        self.publish_market_data(topic, bar_data, "bar", MessagePriority.NORMAL)
    
    def publish_instrument_data(self, symbol: str, instrument_data: Any, venue: str = None) -> None:
        """Publish instrument/symbol information with LOW priority."""
        topic = f"data.instruments.{venue}.{symbol}" if venue else f"data.instruments.{symbol}"
        self.publish_market_data(topic, instrument_data, "instrument", MessagePriority.LOW)
    
    def _get_market_data_priority(self, data_type: str) -> MessagePriority:
        """Auto-assign priority based on market data type."""
        priority_map = {
            "tick": MessagePriority.HIGH,       # Time-sensitive price updates
            "trade": MessagePriority.HIGH,      # Execution tracking
            "book": MessagePriority.HIGH,       # Liquidity changes
            "quote": MessagePriority.NORMAL,    # Bid/ask spreads
            "bar": MessagePriority.NORMAL,      # OHLCV data
            "instrument": MessagePriority.LOW,   # Symbol metadata
            "unknown": MessagePriority.NORMAL    # Default fallback
        }
        return priority_map.get(data_type, MessagePriority.NORMAL)
    
    def _update_market_data_metrics(self, data_type: str, start_time: float) -> None:
        """Update performance metrics for market data processing."""
        # Calculate processing latency
        processing_time_ms = (time.perf_counter() - start_time) * 1000.0
        
        # Update latency tracking
        current_time = time.time()
        if data_type == "tick":
            self._data_latency_tracker["last_tick_timestamp"] = current_time
        elif data_type == "quote":
            self._data_latency_tracker["last_quote_timestamp"] = current_time
        elif data_type == "trade":
            self._data_latency_tracker["last_trade_timestamp"] = current_time
        
        # Update average processing latency (exponential moving average)
        alpha = 0.1  # Smoothing factor
        current_avg = self._data_latency_tracker["avg_processing_latency_ms"]
        self._data_latency_tracker["avg_processing_latency_ms"] = (
            alpha * processing_time_ms + (1 - alpha) * current_avg
        )
        
        # Update counters
        counter_map = {
            "tick": "ticks_published",
            "quote": "quotes_published", 
            "trade": "trades_published",
            "bar": "bars_published",
            "book": "book_updates_published",
            "instrument": "instruments_published"
        }
        
        counter_key = counter_map.get(data_type)
        if counter_key:
            self._market_data_counters[counter_key] += 1
    
    def get_market_data_performance_summary(self) -> Dict[str, Any]:
        """Get market data-specific performance summary."""
        base_summary = self.get_performance_summary()
        uptime_minutes = (time.time() - self._data_performance_monitor.start_time) / 60.0
        
        total_data_messages = sum([
            self._market_data_counters["ticks_published"],
            self._market_data_counters["quotes_published"],
            self._market_data_counters["trades_published"],
            self._market_data_counters["bars_published"],
            self._market_data_counters["book_updates_published"],
            self._market_data_counters["instruments_published"]
        ])
        
        market_data_summary = {
            **base_summary,
            "market_data_metrics": self._market_data_counters.copy(),
            "latency_metrics": self._data_latency_tracker.copy(),
            "data_messages_per_minute": total_data_messages / uptime_minutes if uptime_minutes > 0 else 0,
            "avg_processing_latency_ms": self._data_latency_tracker["avg_processing_latency_ms"],
            "high_frequency_data_rate": self._calculate_high_frequency_rate()
        }
        
        return market_data_summary
    
    def _calculate_high_frequency_rate(self) -> float:
        """Calculate rate of high-frequency data (ticks + trades + book updates)."""
        uptime_seconds = time.time() - self._data_performance_monitor.start_time
        if uptime_seconds <= 0:
            return 0.0
        
        high_freq_count = (
            self._market_data_counters["ticks_published"] +
            self._market_data_counters["trades_published"] + 
            self._market_data_counters["book_updates_published"]
        )
        
        return high_freq_count / uptime_seconds


class EnhancedExecutionAdapterMixin(MessageBusIntegrationMixin):
    """
    Enhanced MessageBus mixin specifically designed for Execution Adapter classes.
    
    Provides Execution Adapter-specific enhanced MessageBus features including:
    - Critical priority order event publishing for time-sensitive execution
    - Order latency tracking and execution quality metrics
    - Multi-venue routing patterns for optimal execution
    - Position tracking with enhanced performance monitoring
    - Risk event integration with critical alerting
    
    Usage:
        class MyExecutionAdapter(LiveExecutionClient, EnhancedExecutionAdapterMixin):
            def __init__(self, msgbus, ...):
                super().__init__(msgbus=msgbus, ...)
                self._setup_enhanced_execution_adapter()
    """
    
    def _setup_enhanced_execution_adapter(self) -> None:
        """Set up enhanced Execution Adapter-specific MessageBus features."""
        # Call parent setup
        self._setup_enhanced_messagebus()
        
        # Execution adapter-specific setup
        self._execution_performance_monitor = ComponentPerformanceMonitor(
            component_name=f"ExecutionAdapter.{self.__class__.__name__}",
            msgbus=self._msgbus
        )
        
        # Execution-specific message counters
        self._execution_counters = {
            "orders_submitted": 0,
            "orders_filled": 0,
            "orders_cancelled": 0,
            "orders_rejected": 0,
            "positions_opened": 0,
            "positions_closed": 0,
            "execution_reports": 0,
            "risk_events": 0
        }
        
        # Execution latency tracking
        self._execution_latency_tracker = {
            "last_order_submit_time": {},  # order_id -> timestamp
            "avg_order_latency_ms": 0.0,
            "avg_fill_latency_ms": 0.0,
            "fastest_execution_ms": float('inf'),
            "slowest_execution_ms": 0.0
        }
        
        self._enhanced_logger.info("Enhanced Execution Adapter features initialized")
    
    def publish_order_event(self, topic: str, order_data: Any, event_type: str = "unknown",
                          priority: Optional[MessagePriority] = None) -> str:
        """
        Publish order event with intelligent priority assignment.
        
        Args:
            topic: Order event topic
            order_data: Order event data
            event_type: Type of order event (submit, fill, cancel, reject)
            priority: Override priority (auto-assigned if None)
            
        Returns:
            Event tracking ID for latency measurement
        """
        # Auto-assign priority based on event type
        if priority is None:
            priority = self._get_order_event_priority(event_type)
        
        # Generate tracking ID for latency measurement
        tracking_id = f"{event_type}_{time.time_ns()}"
        
        # Track order submission time for latency measurement
        if event_type == "submit":
            order_id = getattr(order_data, 'client_order_id', str(tracking_id))
            self._execution_latency_tracker["last_order_submit_time"][str(order_id)] = time.perf_counter()
        
        # Publish with priority
        self.publish_with_priority(topic, order_data, priority)
        
        # Update counters and latency tracking
        self._update_execution_metrics(event_type, order_data)
        self._execution_performance_monitor.record_message_sent(topic, priority)
        
        return tracking_id
    
    def publish_order_submitted(self, order_data: Any, venue: str = None) -> str:
        """Publish order submission with CRITICAL priority."""
        topic = f"execution.orders.submit.{venue}" if venue else "execution.orders.submit"
        return self.publish_order_event(topic, order_data, "submit", MessagePriority.CRITICAL)
    
    def publish_order_filled(self, fill_data: Any, venue: str = None) -> str:
        """Publish order fill with CRITICAL priority."""
        topic = f"execution.orders.fill.{venue}" if venue else "execution.orders.fill"
        return self.publish_order_event(topic, fill_data, "fill", MessagePriority.CRITICAL)
    
    def publish_order_cancelled(self, cancel_data: Any, venue: str = None) -> str:
        """Publish order cancellation with HIGH priority."""
        topic = f"execution.orders.cancel.{venue}" if venue else "execution.orders.cancel"
        return self.publish_order_event(topic, cancel_data, "cancel", MessagePriority.HIGH)
    
    def publish_order_rejected(self, reject_data: Any, venue: str = None) -> str:
        """Publish order rejection with CRITICAL priority."""
        topic = f"execution.orders.reject.{venue}" if venue else "execution.orders.reject"
        return self.publish_order_event(topic, reject_data, "reject", MessagePriority.CRITICAL)
    
    def publish_position_event(self, position_data: Any, event_type: str, venue: str = None) -> None:
        """Publish position events with HIGH priority."""
        topic = f"execution.positions.{event_type}.{venue}" if venue else f"execution.positions.{event_type}"
        self.publish_with_priority(topic, position_data, MessagePriority.HIGH)
        
        if event_type == "open":
            self._execution_counters["positions_opened"] += 1
        elif event_type == "close":
            self._execution_counters["positions_closed"] += 1
            
        self._execution_performance_monitor.record_message_sent(topic, MessagePriority.HIGH)
    
    def publish_execution_report(self, report_data: Any, venue: str = None) -> None:
        """Publish execution reports with NORMAL priority."""
        topic = f"execution.reports.{venue}" if venue else "execution.reports"
        self.publish_with_priority(topic, report_data, MessagePriority.NORMAL)
        
        self._execution_counters["execution_reports"] += 1
        self._execution_performance_monitor.record_message_sent(topic, MessagePriority.NORMAL)
    
    def publish_risk_event(self, event_type: str, risk_data: Any, venue: str = None) -> None:
        """Publish risk events with CRITICAL priority."""
        topic = f"execution.risk.{event_type}.{venue}" if venue else f"execution.risk.{event_type}"
        self.publish_with_priority(topic, risk_data, MessagePriority.CRITICAL)
        
        self._execution_counters["risk_events"] += 1
        self._execution_performance_monitor.record_message_sent(topic, MessagePriority.CRITICAL)
        
        # Log critical risk events
        self._enhanced_logger.warning(f"Critical execution risk event: {event_type}")
    
    def subscribe_to_execution_stream(self, venue_pattern: str, callback: Callable) -> None:
        """Subscribe to execution events using venue-specific patterns."""
        self.subscribe_with_pattern(venue_pattern, callback)
        self._enhanced_logger.info(f"Subscribed to execution stream: {venue_pattern}")
    
    def _get_order_event_priority(self, event_type: str) -> MessagePriority:
        """Auto-assign priority based on order event type."""
        priority_map = {
            "submit": MessagePriority.CRITICAL,    # Order submissions are time-critical
            "fill": MessagePriority.CRITICAL,      # Fills need immediate processing
            "reject": MessagePriority.CRITICAL,    # Rejections need immediate attention
            "cancel": MessagePriority.HIGH,        # Cancellations are important but not critical
            "modify": MessagePriority.HIGH,        # Order modifications
            "unknown": MessagePriority.HIGH        # Default fallback
        }
        return priority_map.get(event_type, MessagePriority.HIGH)
    
    def _update_execution_metrics(self, event_type: str, order_data: Any) -> None:
        """Update performance metrics for execution events."""
        current_time = time.perf_counter()
        
        # Update execution counters
        counter_map = {
            "submit": "orders_submitted",
            "fill": "orders_filled",
            "cancel": "orders_cancelled",
            "reject": "orders_rejected"
        }
        
        counter_key = counter_map.get(event_type)
        if counter_key:
            self._execution_counters[counter_key] += 1
        
        # Calculate latency for fills
        if event_type == "fill":
            order_id = str(getattr(order_data, 'client_order_id', ''))
            if order_id in self._execution_latency_tracker["last_order_submit_time"]:
                submit_time = self._execution_latency_tracker["last_order_submit_time"][order_id]
                fill_latency_ms = (current_time - submit_time) * 1000.0
                
                # Update latency metrics
                self._update_latency_stats(fill_latency_ms)
                
                # Clean up tracking
                del self._execution_latency_tracker["last_order_submit_time"][order_id]
    
    def _update_latency_stats(self, latency_ms: float) -> None:
        """Update execution latency statistics."""
        # Update average latencies (exponential moving average)
        alpha = 0.1  # Smoothing factor
        
        current_avg = self._execution_latency_tracker["avg_fill_latency_ms"]
        self._execution_latency_tracker["avg_fill_latency_ms"] = (
            alpha * latency_ms + (1 - alpha) * current_avg
        )
        
        # Update fastest and slowest execution times
        if latency_ms < self._execution_latency_tracker["fastest_execution_ms"]:
            self._execution_latency_tracker["fastest_execution_ms"] = latency_ms
        
        if latency_ms > self._execution_latency_tracker["slowest_execution_ms"]:
            self._execution_latency_tracker["slowest_execution_ms"] = latency_ms
    
    def get_execution_performance_summary(self) -> Dict[str, Any]:
        """Get execution-specific performance summary."""
        base_summary = self.get_performance_summary()
        uptime_minutes = (time.time() - self._execution_performance_monitor.start_time) / 60.0
        
        total_orders = (
            self._execution_counters["orders_submitted"] +
            self._execution_counters["orders_filled"] +
            self._execution_counters["orders_cancelled"] +
            self._execution_counters["orders_rejected"]
        )
        
        fill_rate = (
            self._execution_counters["orders_filled"] / 
            self._execution_counters["orders_submitted"] 
            if self._execution_counters["orders_submitted"] > 0 else 0.0
        )
        
        execution_summary = {
            **base_summary,
            "execution_metrics": self._execution_counters.copy(),
            "latency_metrics": self._execution_latency_tracker.copy(),
            "orders_per_minute": total_orders / uptime_minutes if uptime_minutes > 0 else 0,
            "fill_rate": fill_rate,
            "avg_execution_latency_ms": self._execution_latency_tracker["avg_fill_latency_ms"],
            "execution_quality_score": self._calculate_execution_quality_score()
        }
        
        return execution_summary
    
    def _calculate_execution_quality_score(self) -> float:
        """Calculate execution quality score based on fill rate and latency."""
        if self._execution_counters["orders_submitted"] == 0:
            return 0.0
        
        # Fill rate component (0-50 points)
        fill_rate = self._execution_counters["orders_filled"] / self._execution_counters["orders_submitted"]
        fill_score = fill_rate * 50
        
        # Latency component (0-50 points, lower latency = higher score)
        avg_latency = self._execution_latency_tracker["avg_fill_latency_ms"]
        if avg_latency > 0:
            # Score decreases as latency increases (100ms = 50 points, 1000ms = 5 points)
            latency_score = max(5, 50 - (avg_latency / 20))
        else:
            latency_score = 50
        
        return fill_score + latency_score


# =============================================================================
# ADAPTER ENHANCEMENT DECORATORS
# =============================================================================

def enhanced_data_adapter(cls):
    """
    Class decorator that automatically enhances Data Adapter classes with MessageBus features.
    
    Args:
        cls: Data Adapter class to enhance
        
    Returns:
        Enhanced Data Adapter class
    """
    original_init = cls.__init__
    
    def enhanced_init(self, *args, **kwargs):
        # Call original __init__
        original_init(self, *args, **kwargs)
        
        # Add enhanced features if MessageBus available
        if hasattr(self, '_msgbus') and self._msgbus is not None:
            _add_enhanced_data_adapter_features(self)
    
    cls.__init__ = enhanced_init
    cls.__enhanced_messagebus__ = True
    
    return cls


def enhanced_execution_adapter(cls):
    """
    Class decorator that automatically enhances Execution Adapter classes with MessageBus features.
    
    Args:
        cls: Execution Adapter class to enhance
        
    Returns:
        Enhanced Execution Adapter class
    """
    original_init = cls.__init__
    
    def enhanced_init(self, *args, **kwargs):
        # Call original __init__
        original_init(self, *args, **kwargs)
        
        # Add enhanced features if MessageBus available
        if hasattr(self, '_msgbus') and self._msgbus is not None:
            _add_enhanced_execution_adapter_features(self)
    
    cls.__init__ = enhanced_init
    cls.__enhanced_messagebus__ = True
    
    return cls


def _add_enhanced_data_adapter_features(adapter) -> None:
    """Add enhanced MessageBus features to a Data Adapter instance."""
    # Add core enhanced features from integration module
    adapter._enhanced_msgbus_features = EnhancedMessageBusFeatures(adapter._msgbus)
    adapter._enhanced_logger = logging.getLogger(f"{adapter.__class__.__name__}.Enhanced")
    
    # Add data adapter-specific features
    adapter._data_performance_monitor = ComponentPerformanceMonitor(
        component_name=f"DataAdapter.{adapter.__class__.__name__}",
        msgbus=adapter._msgbus
    )
    
    adapter._market_data_counters = {
        "ticks_published": 0,
        "quotes_published": 0,
        "trades_published": 0,
        "bars_published": 0,
        "book_updates_published": 0,
        "instruments_published": 0,
        "subscription_requests": 0,
        "data_bytes_processed": 0
    }
    
    adapter._data_latency_tracker = {
        "last_tick_timestamp": 0,
        "last_quote_timestamp": 0,
        "last_trade_timestamp": 0,
        "avg_processing_latency_ms": 0.0
    }
    
    # Add enhanced methods from mixin
    def publish_market_data(topic: str, data: Any, data_type: str = "unknown", 
                          priority: Optional[MessagePriority] = None):
        if priority is None:
            priority_map = {
                "tick": MessagePriority.HIGH,
                "trade": MessagePriority.HIGH,
                "book": MessagePriority.HIGH,
                "quote": MessagePriority.NORMAL,
                "bar": MessagePriority.NORMAL,
                "instrument": MessagePriority.LOW,
            }
            priority = priority_map.get(data_type, MessagePriority.NORMAL)
        
        adapter._enhanced_msgbus_features.publish_with_priority(topic, data, priority)
        
        # Update counters
        counter_map = {
            "tick": "ticks_published",
            "quote": "quotes_published", 
            "trade": "trades_published",
            "bar": "bars_published",
            "book": "book_updates_published",
            "instrument": "instruments_published"
        }
        counter_key = counter_map.get(data_type)
        if counter_key:
            adapter._market_data_counters[counter_key] += 1
            
        adapter._data_performance_monitor.record_message_sent(topic, priority)
    
    def publish_tick_data(symbol: str, tick_data: Any, venue: str = None):
        topic = f"data.ticks.{venue}.{symbol}" if venue else f"data.ticks.{symbol}"
        publish_market_data(topic, tick_data, "tick", MessagePriority.HIGH)
    
    def publish_trade_data(symbol: str, trade_data: Any, venue: str = None):
        topic = f"data.trades.{venue}.{symbol}" if venue else f"data.trades.{symbol}"
        publish_market_data(topic, trade_data, "trade", MessagePriority.HIGH)
    
    def get_market_data_performance_summary():
        base_summary = adapter._data_performance_monitor.get_performance_summary()
        return {
            **base_summary,
            "market_data_metrics": adapter._market_data_counters.copy(),
            "latency_metrics": adapter._data_latency_tracker.copy(),
        }
    
    def get_messagebus_info():
        return adapter._enhanced_msgbus_features.get_capabilities()
    
    # Attach methods
    adapter.publish_market_data = publish_market_data
    adapter.publish_tick_data = publish_tick_data
    adapter.publish_trade_data = publish_trade_data
    adapter.get_market_data_performance_summary = get_market_data_performance_summary
    adapter.get_messagebus_info = get_messagebus_info
    
    adapter._enhanced_logger.info("Enhanced Data Adapter features added")


def _add_enhanced_execution_adapter_features(adapter) -> None:
    """Add enhanced MessageBus features to an Execution Adapter instance."""
    # Add core enhanced features
    adapter._enhanced_msgbus_features = EnhancedMessageBusFeatures(adapter._msgbus)
    adapter._enhanced_logger = logging.getLogger(f"{adapter.__class__.__name__}.Enhanced")
    
    # Add execution adapter-specific features
    adapter._execution_performance_monitor = ComponentPerformanceMonitor(
        component_name=f"ExecutionAdapter.{adapter.__class__.__name__}",
        msgbus=adapter._msgbus
    )
    
    adapter._execution_counters = {
        "orders_submitted": 0,
        "orders_filled": 0,
        "orders_cancelled": 0,
        "orders_rejected": 0,
        "positions_opened": 0,
        "positions_closed": 0,
        "execution_reports": 0,
        "risk_events": 0
    }
    
    adapter._execution_latency_tracker = {
        "last_order_submit_time": {},
        "avg_order_latency_ms": 0.0,
        "avg_fill_latency_ms": 0.0,
        "fastest_execution_ms": float('inf'),
        "slowest_execution_ms": 0.0
    }
    
    # Add enhanced methods from mixin
    def publish_order_event(topic: str, order_data: Any, event_type: str = "unknown",
                          priority: Optional[MessagePriority] = None):
        if priority is None:
            priority_map = {
                "submit": MessagePriority.CRITICAL,
                "fill": MessagePriority.CRITICAL,
                "reject": MessagePriority.CRITICAL,
                "cancel": MessagePriority.HIGH,
                "modify": MessagePriority.HIGH,
            }
            priority = priority_map.get(event_type, MessagePriority.HIGH)
        
        adapter._enhanced_msgbus_features.publish_with_priority(topic, order_data, priority)
        
        # Update counters
        counter_map = {
            "submit": "orders_submitted",
            "fill": "orders_filled",
            "cancel": "orders_cancelled",
            "reject": "orders_rejected"
        }
        counter_key = counter_map.get(event_type)
        if counter_key:
            adapter._execution_counters[counter_key] += 1
            
        adapter._execution_performance_monitor.record_message_sent(topic, priority)
        return f"{event_type}_{time.time_ns()}"
    
    def publish_order_submitted(order_data: Any, venue: str = None):
        topic = f"execution.orders.submit.{venue}" if venue else "execution.orders.submit"
        return publish_order_event(topic, order_data, "submit", MessagePriority.CRITICAL)
    
    def publish_order_filled(fill_data: Any, venue: str = None):
        topic = f"execution.orders.fill.{venue}" if venue else "execution.orders.fill"
        return publish_order_event(topic, fill_data, "fill", MessagePriority.CRITICAL)
    
    def get_execution_performance_summary():
        base_summary = adapter._execution_performance_monitor.get_performance_summary()
        return {
            **base_summary,
            "execution_metrics": adapter._execution_counters.copy(),
            "latency_metrics": adapter._execution_latency_tracker.copy(),
        }
    
    def get_messagebus_info():
        return adapter._enhanced_msgbus_features.get_capabilities()
    
    # Attach methods
    adapter.publish_order_event = publish_order_event
    adapter.publish_order_submitted = publish_order_submitted
    adapter.publish_order_filled = publish_order_filled
    adapter.get_execution_performance_summary = get_execution_performance_summary
    adapter.get_messagebus_info = get_messagebus_info
    
    adapter._enhanced_logger.info("Enhanced Execution Adapter features added")


# =============================================================================
# RUNTIME ENHANCEMENT UTILITIES
# =============================================================================

def enhance_data_adapter(adapter) -> bool:
    """
    Enable enhanced MessageBus features for an existing Data Adapter instance.
    
    Args:
        adapter: Data Adapter instance to enhance
        
    Returns:
        True if enhancement successful, False otherwise
    """
    try:
        if not hasattr(adapter, '_msgbus') or adapter._msgbus is None:
            logging.getLogger(__name__).warning("Data Adapter has no MessageBus - cannot enhance")
            return False
        
        _add_enhanced_data_adapter_features(adapter)
        return True
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to enhance Data Adapter: {e}")
        return False


def enhance_execution_adapter(adapter) -> bool:
    """
    Enable enhanced MessageBus features for an existing Execution Adapter instance.
    
    Args:
        adapter: Execution Adapter instance to enhance
        
    Returns:
        True if enhancement successful, False otherwise
    """
    try:
        if not hasattr(adapter, '_msgbus') or adapter._msgbus is None:
            logging.getLogger(__name__).warning("Execution Adapter has no MessageBus - cannot enhance")
            return False
        
        _add_enhanced_execution_adapter_features(adapter)
        return True
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to enhance Execution Adapter: {e}")
        return False


def check_adapter_enhanced_messagebus_support(adapter) -> Dict[str, Any]:
    """
    Check if an adapter has enhanced MessageBus support enabled.
    
    Args:
        adapter: Adapter to check
        
    Returns:
        Dict with support information
    """
    info = {
        "has_messagebus": hasattr(adapter, '_msgbus'),
        "messagebus_enhanced": False,
        "features_enabled": False,
        "adapter_type": "unknown",
        "enhancement_available": False
    }
    
    if info["has_messagebus"]:
        try:
            from nautilus_trader.infrastructure.messagebus.client import BufferedMessageBusClient
            info["messagebus_enhanced"] = isinstance(adapter._msgbus, BufferedMessageBusClient)
            info["features_enabled"] = hasattr(adapter, '_enhanced_msgbus_features')
            info["enhancement_available"] = not info["features_enabled"]
            
            # Determine adapter type
            if hasattr(adapter, 'subscribe_trade_ticks') or 'DataClient' in str(type(adapter)):
                info["adapter_type"] = "data"
            elif hasattr(adapter, 'submit_order') or 'ExecutionClient' in str(type(adapter)):
                info["adapter_type"] = "execution"
                
        except ImportError:
            info["enhancement_available"] = False
    
    return info