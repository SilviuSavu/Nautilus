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
Example demonstrating Enhanced MessageBus integration with NautilusTrader components.

This example shows how to:
1. Create enhanced MessageBus-enabled actors and strategies
2. Use priority-based messaging for trading operations
3. Implement pattern-based subscriptions for market data
4. Monitor performance and health of enhanced MessageBus
5. Gracefully handle fallback to standard MessageBus
"""

import asyncio
import logging
import time
from typing import Any, Dict

from nautilus_trader.common.actor import Actor
from nautilus_trader.common.config import ActorConfig
from nautilus_trader.infrastructure.messagebus.config import (
    EnhancedMessageBusConfig,
    ConfigPresets,
    MessagePriority
)
from nautilus_trader.infrastructure.messagebus.actors import (
    enhanced_actor,
    enhanced_strategy,
    enable_enhanced_messagebus_for_actor,
    check_enhanced_messagebus_support
)
from nautilus_trader.infrastructure.messagebus.integration import (
    EnhancedMessageBusFeatures,
    ComponentPerformanceMonitor
)


# =============================================================================
# ENHANCED ACTOR EXAMPLES
# =============================================================================

@enhanced_actor
class EnhancedTradingActor(Actor):
    """
    Example trading actor with enhanced MessageBus features.
    
    Demonstrates:
    - Automatic enhanced MessageBus feature detection
    - Priority-based message publishing for trading signals
    - Pattern-based market data subscriptions
    - Performance monitoring and health checks
    """
    
    def __init__(self, config: ActorConfig):
        super().__init__(config)
        
        # Enhanced features are automatically added by the decorator
        # Check if enhancement worked
        support_info = check_enhanced_messagebus_support(self)
        self.log.info(f"Enhanced MessageBus support: {support_info}")
        
        # Set up trading-specific subscriptions if enhanced features available
        if hasattr(self, 'get_messagebus_info'):
            msgbus_info = self.get_messagebus_info()
            if msgbus_info.get('pattern_matching', False):
                self._setup_enhanced_subscriptions()
            else:
                self._setup_standard_subscriptions()
    
    def _setup_enhanced_subscriptions(self):
        """Set up subscriptions using enhanced pattern matching."""
        self.log.info("Setting up enhanced pattern-based subscriptions")
        
        # Subscribe to all tick data for major currency pairs using patterns
        self.subscribe_to_market_data_pattern(
            "data.ticks.*.EUR*",
            self._on_enhanced_tick_data
        )
        
        # Subscribe to order book updates for specific exchanges
        self.subscribe_to_market_data_pattern(
            "data.book.BINANCE.*",
            self._on_enhanced_book_data
        )
    
    def _setup_standard_subscriptions(self):
        """Set up standard subscriptions as fallback."""
        self.log.info("Setting up standard MessageBus subscriptions")
        # Standard subscription logic here
    
    def _on_enhanced_tick_data(self, data: Any):
        """Handle enhanced tick data with performance monitoring."""
        # Record data received for performance monitoring
        if hasattr(self, '_trading_message_counts'):
            self._trading_message_counts["data_received"] += 1
        
        # Process tick data
        self.log.debug(f"Received enhanced tick data: {data}")
        
        # Generate trading signal with high priority
        if self._should_generate_signal(data):
            signal_data = self._create_trading_signal(data)
            
            # Use enhanced priority messaging if available
            if hasattr(self, 'publish_signal'):
                self.publish_signal(
                    topic=f"signal.{self.id}",
                    signal_data=signal_data,
                    priority=MessagePriority.HIGH
                )
            else:
                # Fallback to standard publish
                self.msgbus.publish(f"signal.{self.id}", signal_data)
    
    def _on_enhanced_book_data(self, data: Any):
        """Handle enhanced order book data."""
        self.log.debug(f"Received enhanced book data: {data}")
        
        # Analyze order book for opportunities
        if self._detect_arbitrage_opportunity(data):
            # Publish critical priority message for time-sensitive arbitrage
            if hasattr(self, 'publish_order_event'):
                self.publish_order_event(
                    topic=f"arbitrage.{self.id}",
                    order_data={"type": "arbitrage_opportunity", "data": data}
                )
    
    def _should_generate_signal(self, data: Any) -> bool:
        """Determine if tick data should generate a trading signal."""
        # Placeholder logic
        return True
    
    def _create_trading_signal(self, data: Any) -> Dict[str, Any]:
        """Create trading signal from tick data."""
        return {
            "signal_type": "momentum",
            "strength": 0.8,
            "data": data,
            "timestamp": time.time()
        }
    
    def _detect_arbitrage_opportunity(self, book_data: Any) -> bool:
        """Detect arbitrage opportunities in order book data."""
        # Placeholder logic
        return False
    
    def on_start(self):
        """Called when the actor is started."""
        self.log.info("Enhanced Trading Actor started")
        
        # Log MessageBus capabilities
        if hasattr(self, 'get_messagebus_info'):
            capabilities = self.get_messagebus_info()
            self.log.info(f"MessageBus capabilities: {capabilities}")
        
        # Start performance monitoring
        if hasattr(self, 'get_trading_performance_summary'):
            # Schedule periodic performance reports
            self._schedule_performance_reports()
    
    def on_stop(self):
        """Called when the actor is stopped."""
        # Log final performance summary
        if hasattr(self, 'get_trading_performance_summary'):
            summary = self.get_trading_performance_summary()
            self.log.info(f"Final performance summary: {summary}")
        
        self.log.info("Enhanced Trading Actor stopped")
    
    def _schedule_performance_reports(self):
        """Schedule periodic performance reports."""
        # This would typically use the actor's timer functionality
        pass


class StandardTradingActor(Actor):
    """
    Standard trading actor that can be enhanced at runtime.
    
    Demonstrates:
    - Runtime enhancement of existing actors
    - Feature detection and graceful degradation
    - Manual integration of enhanced features
    """
    
    def __init__(self, config: ActorConfig):
        super().__init__(config)
        self._enhanced_features_enabled = False
    
    def on_start(self):
        """Called when the actor is started."""
        self.log.info("Standard Trading Actor started")
        
        # Attempt to enable enhanced features at runtime
        if enable_enhanced_messagebus_for_actor(self):
            self._enhanced_features_enabled = True
            self.log.info("Enhanced MessageBus features enabled at runtime")
            self._setup_enhanced_features()
        else:
            self.log.info("Enhanced MessageBus features not available")
            self._setup_standard_features()
    
    def _setup_enhanced_features(self):
        """Set up enhanced MessageBus features."""
        # Enhanced features are now available through added methods
        if hasattr(self, 'get_messagebus_info'):
            capabilities = self.get_messagebus_info()
            self.log.info(f"Runtime-enabled capabilities: {capabilities}")
    
    def _setup_standard_features(self):
        """Set up standard MessageBus features."""
        self.log.info("Using standard MessageBus features")
    
    def send_trading_signal(self, signal_data: Dict[str, Any]):
        """Send a trading signal using the best available method."""
        if self._enhanced_features_enabled and hasattr(self, 'publish_signal'):
            # Use enhanced priority messaging
            self.publish_signal(
                topic=f"signal.{self.id}",
                signal_data=signal_data,
                priority=MessagePriority.HIGH
            )
        else:
            # Use standard messaging
            self.msgbus.publish(f"signal.{self.id}", signal_data)


# =============================================================================
# CONFIGURATION EXAMPLES
# =============================================================================

def create_enhanced_messagebus_config() -> EnhancedMessageBusConfig:
    """Create enhanced MessageBus configuration for high-frequency trading."""
    config = ConfigPresets.high_frequency()
    
    # Add custom subscriptions for this example
    config.add_subscription("data.ticks.*", MessagePriority.HIGH)
    config.add_subscription("data.book.*", MessagePriority.HIGH)
    config.add_subscription("signal.*", MessagePriority.HIGH)
    config.add_subscription("arbitrage.*", MessagePriority.CRITICAL)
    
    # Add custom streams
    config.add_stream("trading-signals")
    config.add_stream("order-events")
    
    return config


def create_development_config() -> EnhancedMessageBusConfig:
    """Create enhanced MessageBus configuration for development."""
    config = ConfigPresets.development()
    
    # Enable all monitoring for development
    config.enable_metrics = True
    config.enable_tracing = True
    config.trace_sample_rate = 1.0  # 100% sampling for development
    
    return config


# =============================================================================
# DEMONSTRATION FUNCTIONS
# =============================================================================

async def demonstrate_enhanced_messagebus_features():
    """Demonstrate enhanced MessageBus features."""
    print("=== Enhanced MessageBus Feature Demonstration ===\n")
    
    # Create enhanced configuration
    config = create_enhanced_messagebus_config()
    print(f"Created enhanced MessageBus config:")
    print(f"  - Connection pool: {config.connection_pool_size}")
    print(f"  - Auto-scaling: {config.auto_scale_enabled} ({config.min_workers}-{config.max_workers} workers)")
    print(f"  - Pattern matching: {config.enable_pattern_matching}")
    print(f"  - Metrics enabled: {config.enable_metrics}")
    print(f"  - Subscriptions: {len(config.subscriptions)}")
    print(f"  - Streams: {len(config.stream_configs)}\n")
    
    # Create mock MessageBus for demonstration
    try:
        from nautilus_trader.infrastructure.messagebus.client import BufferedMessageBusClient
        from nautilus_trader.common.component import TestClock
        from nautilus_trader.core.uuid import UUID4
        from nautilus_trader.model.identifiers import TraderId
        
        # Create enhanced MessageBus
        enhanced_msgbus = BufferedMessageBusClient(config)
        
        # Demonstrate feature detection
        features = EnhancedMessageBusFeatures(enhanced_msgbus)
        capabilities = features.get_capabilities()
        
        print("Enhanced MessageBus capabilities:")
        for key, value in capabilities.items():
            print(f"  - {key}: {value}")
        print()
        
        # Demonstrate performance monitoring
        monitor = ComponentPerformanceMonitor("DemoActor", enhanced_msgbus)
        
        # Simulate some activity
        for i in range(10):
            monitor.record_message_sent("demo.topic", MessagePriority.NORMAL)
            monitor.record_message_received("demo.response")
        
        summary = monitor.get_performance_summary()
        print("Performance monitoring demonstration:")
        print(f"  - Total messages: {summary['total_messages']}")
        print(f"  - Messages per second: {summary['messages_per_second']:.1f}")
        print(f"  - Error rate: {summary['error_rate']:.2%}")
        print()
        
    except ImportError:
        print("Enhanced MessageBus components not available - demonstration limited")
    except Exception as e:
        print(f"Error during demonstration: {e}")


def demonstrate_actor_enhancement():
    """Demonstrate actor enhancement capabilities."""
    print("=== Actor Enhancement Demonstration ===\n")
    
    # Mock actor for demonstration
    class MockActor:
        def __init__(self):
            self._msgbus = None  # Would normally be set by system
            self.__class__.__name__ = "MockActor"
    
    mock_actor = MockActor()
    
    # Check support before enhancement
    support_info = check_enhanced_messagebus_support(mock_actor)
    print("Before enhancement:")
    for key, value in support_info.items():
        print(f"  - {key}: {value}")
    print()
    
    # This would normally work with a real MessageBus
    print("Note: Full demonstration requires running system with MessageBus")
    print("In practice, enhanced features would be automatically available")
    print("when using @enhanced_actor decorator or runtime enhancement functions.\n")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Enhanced MessageBus Integration Example")
    print("=" * 50)
    
    # Run demonstrations
    asyncio.run(demonstrate_enhanced_messagebus_features())
    demonstrate_actor_enhancement()
    
    print("Example completed. See the code for implementation details.")
    print("\nTo use enhanced MessageBus in your actors:")
    print("1. Use @enhanced_actor decorator on your Actor classes")
    print("2. Or call enable_enhanced_messagebus_for_actor() at runtime")
    print("3. Enhanced features will be automatically available when supported")
    print("4. Graceful fallback to standard MessageBus when not available")