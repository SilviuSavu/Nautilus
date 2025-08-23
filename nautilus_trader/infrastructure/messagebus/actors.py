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
Enhanced MessageBus integration for NautilusTrader Actors and Components.

Provides enhanced MessageBus capabilities for Actor and Component classes through
mixins, decorators, and utility functions while maintaining backward compatibility.
"""

import asyncio
import logging
import time
from typing import Any, Dict, Callable, Optional, Union

from nautilus_trader.infrastructure.messagebus.config import MessagePriority
from nautilus_trader.infrastructure.messagebus.integration import (
    MessageBusIntegrationMixin,
    EnhancedMessageBusFeatures,
    ComponentPerformanceMonitor,
)


class EnhancedActorMixin(MessageBusIntegrationMixin):
    """
    Enhanced MessageBus mixin specifically designed for Actor classes.
    
    Provides Actor-specific enhanced MessageBus features including:
    - Priority-based message publishing for trading signals
    - Pattern-based subscriptions for market data
    - Performance monitoring for trading operations
    - Health monitoring for live trading systems
    
    Usage:
        class MyActor(Actor, EnhancedActorMixin):
            def __init__(self, config):
                super().__init__(config)
                self._setup_enhanced_actor()
    """
    
    def _setup_enhanced_actor(self) -> None:
        """Set up enhanced Actor-specific MessageBus features."""
        # Call parent setup
        self._setup_enhanced_messagebus()
        
        # Actor-specific setup
        self._actor_performance_monitor = ComponentPerformanceMonitor(
            component_name=f"Actor.{self.__class__.__name__}",
            msgbus=self._msgbus
        )
        
        # Trading-specific message counters
        self._trading_message_counts = {
            "signals_sent": 0,
            "orders_placed": 0,
            "positions_opened": 0,
            "positions_closed": 0,
            "data_received": 0
        }
        
        self._enhanced_logger.info("Enhanced Actor features initialized")
    
    def publish_signal(self, topic: str, signal_data: Any, priority: MessagePriority = MessagePriority.HIGH) -> None:
        """
        Publish a trading signal with high priority.
        
        Args:
            topic: Signal topic
            signal_data: Signal data
            priority: Message priority (default HIGH for trading signals)
        """
        self.publish_with_priority(topic, signal_data, priority)
        self._trading_message_counts["signals_sent"] += 1
        self._actor_performance_monitor.record_message_sent(topic, priority)
    
    def publish_order_event(self, topic: str, order_data: Any) -> None:
        """
        Publish order-related event with critical priority.
        
        Args:
            topic: Order event topic
            order_data: Order event data
        """
        self.publish_with_priority(topic, order_data, MessagePriority.CRITICAL)
        self._trading_message_counts["orders_placed"] += 1
        self._actor_performance_monitor.record_message_sent(topic, MessagePriority.CRITICAL)
    
    def subscribe_to_market_data_pattern(self, instrument_pattern: str, callback: Callable) -> None:
        """
        Subscribe to market data using instrument pattern.
        
        Args:
            instrument_pattern: Pattern like "data.*.EURUSD.*" or "data.ticks.BINANCE.*"
            callback: Callback for matching market data
        """
        self.subscribe_with_pattern(instrument_pattern, callback)
        self._enhanced_logger.info(f"Subscribed to market data pattern: {instrument_pattern}")
    
    def get_trading_performance_summary(self) -> Dict[str, Any]:
        """Get trading-specific performance summary."""
        base_summary = self.get_performance_summary()
        
        trading_summary = {
            **base_summary,
            "trading_metrics": self._trading_message_counts.copy(),
            "signals_per_minute": self._calculate_rate("signals_sent"),
            "orders_per_minute": self._calculate_rate("orders_placed")
        }
        
        return trading_summary
    
    def _calculate_rate(self, metric_name: str) -> float:
        """Calculate per-minute rate for a trading metric."""
        if hasattr(self, '_actor_performance_monitor'):
            uptime_minutes = (time.time() - self._actor_performance_monitor.start_time) / 60.0
            if uptime_minutes > 0:
                return self._trading_message_counts.get(metric_name, 0) / uptime_minutes
        return 0.0
    
    def log_trading_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Log trading event with enhanced MessageBus context."""
        msgbus_info = self.get_messagebus_info()
        
        log_entry = {
            "event_type": event_type,
            "details": details,
            "messagebus_type": msgbus_info.get("type", "unknown"),
            "timestamp": time.time()
        }
        
        self._enhanced_logger.info(f"Trading Event: {log_entry}")


class EnhancedStrategyMixin(EnhancedActorMixin):
    """
    Enhanced MessageBus mixin specifically designed for Strategy classes.
    
    Extends EnhancedActorMixin with strategy-specific features:
    - Strategy performance tracking
    - Risk event publishing with critical priority
    - Portfolio event monitoring
    - Strategy state synchronization
    """
    
    def _setup_enhanced_strategy(self) -> None:
        """Set up enhanced Strategy-specific features."""
        self._setup_enhanced_actor()
        
        # Strategy-specific counters
        self._strategy_counters = {
            "positions_entered": 0,
            "positions_exited": 0,
            "risk_events": 0,
            "pnl_updates": 0,
            "portfolio_updates": 0
        }
        
        self._enhanced_logger.info("Enhanced Strategy features initialized")
    
    def publish_risk_event(self, event_type: str, risk_data: Any) -> None:
        """
        Publish risk-related event with critical priority.
        
        Args:
            event_type: Type of risk event
            risk_data: Risk event data
        """
        topic = f"risk.{event_type}"
        self.publish_with_priority(topic, risk_data, MessagePriority.CRITICAL)
        
        self._strategy_counters["risk_events"] += 1
        self._actor_performance_monitor.record_message_sent(topic, MessagePriority.CRITICAL)
        
        # Log critical risk events
        self._enhanced_logger.warning(f"Critical risk event published: {event_type}")
    
    def subscribe_to_portfolio_updates(self, callback: Callable) -> None:
        """Subscribe to portfolio updates using pattern matching."""
        pattern = "portfolio.*"
        self.subscribe_with_pattern(pattern, callback)
        self._enhanced_logger.info("Subscribed to portfolio updates")
    
    def publish_strategy_state(self, state_data: Dict[str, Any]) -> None:
        """
        Publish strategy state for monitoring systems.
        
        Args:
            state_data: Strategy state information
        """
        topic = f"strategy.{self.__class__.__name__}.state"
        enhanced_state = {
            **state_data,
            "messagebus_metrics": self.get_messagebus_metrics(),
            "strategy_counters": self._strategy_counters.copy(),
            "timestamp": time.time()
        }
        
        self.publish_with_priority(topic, enhanced_state, MessagePriority.NORMAL)
    
    def get_strategy_performance_summary(self) -> Dict[str, Any]:
        """Get strategy-specific performance summary."""
        base_summary = self.get_trading_performance_summary()
        
        strategy_summary = {
            **base_summary,
            "strategy_metrics": self._strategy_counters.copy(),
            "positions_per_hour": self._calculate_hourly_rate("positions_entered"),
            "risk_events_per_hour": self._calculate_hourly_rate("risk_events")
        }
        
        return strategy_summary
    
    def _calculate_hourly_rate(self, metric_name: str) -> float:
        """Calculate per-hour rate for a strategy metric."""
        if hasattr(self, '_actor_performance_monitor'):
            uptime_hours = (time.time() - self._actor_performance_monitor.start_time) / 3600.0
            if uptime_hours > 0:
                return self._strategy_counters.get(metric_name, 0) / uptime_hours
        return 0.0


# =============================================================================
# COMPONENT ENHANCEMENT DECORATORS
# =============================================================================

def enhanced_actor(cls):
    """
    Class decorator that automatically enhances Actor classes with MessageBus features.
    
    Args:
        cls: Actor class to enhance
        
    Returns:
        Enhanced Actor class
    """
    original_init = cls.__init__
    
    def enhanced_init(self, *args, **kwargs):
        # Call original __init__
        original_init(self, *args, **kwargs)
        
        # Add enhanced features
        if hasattr(self, '_msgbus') and self._msgbus is not None:
            # Manually add enhanced features
            _add_enhanced_actor_features(self)
    
    cls.__init__ = enhanced_init
    cls.__enhanced_messagebus__ = True
    
    return cls


def enhanced_strategy(cls):
    """
    Class decorator that automatically enhances Strategy classes with MessageBus features.
    
    Args:
        cls: Strategy class to enhance
        
    Returns:
        Enhanced Strategy class
    """
    original_init = cls.__init__
    
    def enhanced_init(self, *args, **kwargs):
        # Call original __init__
        original_init(self, *args, **kwargs)
        
        # Add enhanced features
        if hasattr(self, '_msgbus') and self._msgbus is not None:
            _add_enhanced_strategy_features(self)
    
    cls.__init__ = enhanced_init
    cls.__enhanced_messagebus__ = True
    
    return cls


def _add_enhanced_actor_features(actor) -> None:
    """Add enhanced MessageBus features to an Actor instance."""
    # Add core enhanced features
    actor._enhanced_msgbus_features = EnhancedMessageBusFeatures(actor._msgbus)
    actor._enhanced_logger = logging.getLogger(f"{actor.__class__.__name__}.Enhanced")
    actor._actor_performance_monitor = ComponentPerformanceMonitor(
        component_name=f"Actor.{actor.__class__.__name__}",
        msgbus=actor._msgbus
    )
    
    # Add trading message counters
    actor._trading_message_counts = {
        "signals_sent": 0,
        "orders_placed": 0,
        "positions_opened": 0,
        "positions_closed": 0,
        "data_received": 0
    }
    
    # Add enhanced methods
    def publish_signal(topic: str, signal_data: Any, priority: MessagePriority = MessagePriority.HIGH):
        actor._enhanced_msgbus_features.publish_with_priority(topic, signal_data, priority)
        actor._trading_message_counts["signals_sent"] += 1
        actor._actor_performance_monitor.record_message_sent(topic, priority)
    
    def publish_order_event(topic: str, order_data: Any):
        actor._enhanced_msgbus_features.publish_with_priority(topic, order_data, MessagePriority.CRITICAL)
        actor._trading_message_counts["orders_placed"] += 1
        actor._actor_performance_monitor.record_message_sent(topic, MessagePriority.CRITICAL)
    
    def subscribe_to_market_data_pattern(instrument_pattern: str, callback: Callable):
        actor._enhanced_msgbus_features.subscribe_with_pattern(instrument_pattern, callback)
        actor._enhanced_logger.info(f"Subscribed to market data pattern: {instrument_pattern}")
    
    def get_trading_performance_summary():
        base_summary = actor._actor_performance_monitor.get_performance_summary()
        uptime_minutes = (time.time() - actor._actor_performance_monitor.start_time) / 60.0
        
        return {
            **base_summary,
            "trading_metrics": actor._trading_message_counts.copy(),
            "signals_per_minute": actor._trading_message_counts["signals_sent"] / uptime_minutes if uptime_minutes > 0 else 0,
            "orders_per_minute": actor._trading_message_counts["orders_placed"] / uptime_minutes if uptime_minutes > 0 else 0
        }
    
    def get_messagebus_info():
        return actor._enhanced_msgbus_features.get_capabilities()
    
    # Attach methods
    actor.publish_signal = publish_signal
    actor.publish_order_event = publish_order_event
    actor.subscribe_to_market_data_pattern = subscribe_to_market_data_pattern
    actor.get_trading_performance_summary = get_trading_performance_summary
    actor.get_messagebus_info = get_messagebus_info
    
    actor._enhanced_logger.info("Enhanced Actor features added")


def _add_enhanced_strategy_features(strategy) -> None:
    """Add enhanced MessageBus features to a Strategy instance."""
    # Add Actor features first
    _add_enhanced_actor_features(strategy)
    
    # Add strategy-specific features
    strategy._strategy_counters = {
        "positions_entered": 0,
        "positions_exited": 0,
        "risk_events": 0,
        "pnl_updates": 0,
        "portfolio_updates": 0
    }
    
    # Add strategy methods
    def publish_risk_event(event_type: str, risk_data: Any):
        topic = f"risk.{event_type}"
        strategy._enhanced_msgbus_features.publish_with_priority(topic, risk_data, MessagePriority.CRITICAL)
        strategy._strategy_counters["risk_events"] += 1
        strategy._actor_performance_monitor.record_message_sent(topic, MessagePriority.CRITICAL)
        strategy._enhanced_logger.warning(f"Critical risk event published: {event_type}")
    
    def publish_strategy_state(state_data: Dict[str, Any]):
        topic = f"strategy.{strategy.__class__.__name__}.state"
        enhanced_state = {
            **state_data,
            "messagebus_metrics": strategy._enhanced_msgbus_features.get_metrics(),
            "strategy_counters": strategy._strategy_counters.copy(),
            "timestamp": time.time()
        }
        strategy._enhanced_msgbus_features.publish_with_priority(topic, enhanced_state, MessagePriority.NORMAL)
    
    def get_strategy_performance_summary():
        base_summary = strategy.get_trading_performance_summary()
        uptime_hours = (time.time() - strategy._actor_performance_monitor.start_time) / 3600.0
        
        return {
            **base_summary,
            "strategy_metrics": strategy._strategy_counters.copy(),
            "positions_per_hour": strategy._strategy_counters["positions_entered"] / uptime_hours if uptime_hours > 0 else 0,
            "risk_events_per_hour": strategy._strategy_counters["risk_events"] / uptime_hours if uptime_hours > 0 else 0
        }
    
    # Attach strategy methods
    strategy.publish_risk_event = publish_risk_event
    strategy.publish_strategy_state = publish_strategy_state
    strategy.get_strategy_performance_summary = get_strategy_performance_summary
    
    strategy._enhanced_logger.info("Enhanced Strategy features added")


# =============================================================================
# RUNTIME ENHANCEMENT UTILITIES
# =============================================================================

def enable_enhanced_messagebus_for_actor(actor) -> bool:
    """
    Enable enhanced MessageBus features for an existing Actor instance.
    
    Args:
        actor: Actor instance to enhance
        
    Returns:
        True if enhancement successful, False otherwise
    """
    try:
        if not hasattr(actor, '_msgbus') or actor._msgbus is None:
            logging.getLogger(__name__).warning("Actor has no MessageBus - cannot enhance")
            return False
        
        _add_enhanced_actor_features(actor)
        return True
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to enhance Actor: {e}")
        return False


def enable_enhanced_messagebus_for_strategy(strategy) -> bool:
    """
    Enable enhanced MessageBus features for an existing Strategy instance.
    
    Args:
        strategy: Strategy instance to enhance
        
    Returns:
        True if enhancement successful, False otherwise
    """
    try:
        if not hasattr(strategy, '_msgbus') or strategy._msgbus is None:
            logging.getLogger(__name__).warning("Strategy has no MessageBus - cannot enhance")
            return False
        
        _add_enhanced_strategy_features(strategy)
        return True
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to enhance Strategy: {e}")
        return False


def check_enhanced_messagebus_support(component) -> Dict[str, Any]:
    """
    Check if a component has enhanced MessageBus support enabled.
    
    Args:
        component: Component to check
        
    Returns:
        Dict with support information
    """
    info = {
        "has_messagebus": hasattr(component, '_msgbus'),
        "messagebus_enhanced": False,
        "features_enabled": False,
        "enhancement_available": False
    }
    
    if info["has_messagebus"]:
        try:
            from nautilus_trader.infrastructure.messagebus.client import BufferedMessageBusClient
            info["messagebus_enhanced"] = isinstance(component._msgbus, BufferedMessageBusClient)
            info["features_enabled"] = hasattr(component, '_enhanced_msgbus_features')
            info["enhancement_available"] = not info["features_enabled"]
        except ImportError:
            info["enhancement_available"] = False
    
    return info