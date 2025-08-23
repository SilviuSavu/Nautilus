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
Enhanced MessageBus integration utilities for NautilusTrader components.

Provides utilities for components to detect and utilize enhanced MessageBus features
while maintaining backward compatibility with standard MessageBus implementations.
"""

import asyncio
import logging
import time
from typing import Any, Optional, Dict, Callable, Union

from nautilus_trader.common.component import MessageBus
from nautilus_trader.infrastructure.messagebus.client import BufferedMessageBusClient
from nautilus_trader.infrastructure.messagebus.config import MessagePriority


class MessageBusIntegrationMixin:
    """
    Mixin class providing enhanced MessageBus integration for components.
    
    This mixin can be used by component classes to gain enhanced MessageBus
    capabilities while maintaining full backward compatibility.
    
    Usage:
        class MyComponent(Component, MessageBusIntegrationMixin):
            def __init__(self, ...):
                super().__init__(...)
                self._setup_enhanced_messagebus()
    """
    
    def _setup_enhanced_messagebus(self) -> None:
        """
        Set up enhanced MessageBus integration features.
        
        This method should be called in the component's __init__ method
        after the parent Component.__init__ has been called.
        """
        self._enhanced_msgbus_features = EnhancedMessageBusFeatures(self._msgbus)
        self._enhanced_logger = logging.getLogger(f"{self.__class__.__name__}.Enhanced")
        
        # Log enhanced capabilities
        if self._enhanced_msgbus_features.is_enhanced():
            capabilities = self._enhanced_msgbus_features.get_capabilities()
            self._enhanced_logger.info(f"Enhanced MessageBus detected: {capabilities}")
        else:
            self._enhanced_logger.debug("Standard MessageBus in use")
    
    def publish_with_priority(
        self, 
        topic: str, 
        message: Any, 
        priority: MessagePriority = MessagePriority.NORMAL
    ) -> None:
        """
        Publish message with priority (enhanced MessageBus feature).
        
        Args:
            topic: Message topic
            message: Message to publish
            priority: Message priority (only used if enhanced MessageBus available)
        """
        if self._enhanced_msgbus_features.is_enhanced():
            self._enhanced_msgbus_features.publish_with_priority(topic, message, priority)
        else:
            # Fall back to standard publish
            self._msgbus.publish(topic, message)
    
    def subscribe_with_pattern(self, pattern: str, callback: Callable) -> None:
        """
        Subscribe to topics using pattern matching (enhanced MessageBus feature).
        
        Args:
            pattern: Topic pattern (e.g., "data.*.BINANCE.*")
            callback: Callback function for matching messages
        """
        if self._enhanced_msgbus_features.is_enhanced():
            self._enhanced_msgbus_features.subscribe_with_pattern(pattern, callback)
        else:
            # Fall back to exact topic subscription
            # Note: This is a simplified fallback - real implementation would need topic expansion
            self._msgbus.subscribe(pattern.replace('*', ''), callback)
            self._enhanced_logger.warning(f"Pattern subscription fallback for: {pattern}")
    
    def get_messagebus_metrics(self) -> Dict[str, Any]:
        """
        Get MessageBus performance metrics.
        
        Returns:
            Dict containing metrics (empty dict if not enhanced)
        """
        if self._enhanced_msgbus_features.is_enhanced():
            return self._enhanced_msgbus_features.get_metrics()
        else:
            return {"enhanced": False, "message": "Standard MessageBus - no metrics available"}
    
    def get_messagebus_health(self) -> Dict[str, Any]:
        """
        Get MessageBus health status.
        
        Returns:
            Dict containing health information
        """
        return self._enhanced_msgbus_features.get_health_status()
    
    async def flush_messagebus_buffers(self) -> None:
        """
        Flush MessageBus buffers (enhanced MessageBus feature).
        
        Forces immediate processing of buffered messages.
        """
        if self._enhanced_msgbus_features.is_enhanced():
            await self._enhanced_msgbus_features.flush_buffers()
        else:
            # No-op for standard MessageBus
            pass


class EnhancedMessageBusFeatures:
    """
    Wrapper class providing enhanced MessageBus feature detection and utilities.
    
    This class encapsulates the logic for detecting and utilizing enhanced
    MessageBus features while providing graceful fallbacks.
    """
    
    def __init__(self, msgbus: MessageBus):
        self.msgbus = msgbus
        self.logger = logging.getLogger(f"{__name__}.EnhancedFeatures")
        self._capabilities_cache: Optional[Dict[str, Any]] = None
        self._last_metrics_time = 0.0
        self._metrics_cache: Optional[Dict[str, Any]] = None
        self._cache_duration = 5.0  # Cache metrics for 5 seconds
    
    def is_enhanced(self) -> bool:
        """Check if MessageBus is enhanced (BufferedMessageBusClient)."""
        return isinstance(self.msgbus, BufferedMessageBusClient)
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get MessageBus capabilities (cached)."""
        if self._capabilities_cache is None:
            if self.is_enhanced():
                enhanced_msgbus = self.msgbus
                self._capabilities_cache = {
                    "type": "enhanced",
                    "priority_handling": True,
                    "pattern_matching": enhanced_msgbus.config.enable_pattern_matching,
                    "auto_scaling": enhanced_msgbus.config.auto_scale_enabled,
                    "metrics": enhanced_msgbus.config.enable_metrics,
                    "health_monitoring": True,
                    "worker_count": f"{enhanced_msgbus.config.min_workers}-{enhanced_msgbus.config.max_workers}",
                    "connection_pool": enhanced_msgbus.config.connection_pool_size,
                    "buffer_sizes": {
                        priority.name: enhanced_msgbus.config.get_buffer_config(priority).max_size
                        for priority in MessagePriority
                    }
                }
            else:
                self._capabilities_cache = {
                    "type": "standard",
                    "priority_handling": False,
                    "pattern_matching": False,
                    "auto_scaling": False,
                    "metrics": False,
                    "health_monitoring": False
                }
        
        return self._capabilities_cache
    
    def publish_with_priority(
        self, 
        topic: str, 
        message: Any, 
        priority: MessagePriority = MessagePriority.NORMAL
    ) -> None:
        """Publish message with priority."""
        if self.is_enhanced():
            enhanced_msgbus = self.msgbus
            # Use enhanced publish method with priority
            try:
                # This would be the actual enhanced publish method
                # For now, fall back to standard publish
                enhanced_msgbus.publish(topic, message)
                # TODO: Implement priority handling in BufferedMessageBusClient.publish()
            except Exception as e:
                self.logger.error(f"Enhanced publish failed: {e}, falling back to standard")
                self.msgbus.publish(topic, message)
        else:
            self.msgbus.publish(topic, message)
    
    def subscribe_with_pattern(self, pattern: str, callback: Callable) -> None:
        """Subscribe with pattern matching."""
        if self.is_enhanced():
            enhanced_msgbus = self.msgbus
            try:
                # Use pattern-based subscription if available
                # For now, fall back to standard subscription
                self.msgbus.subscribe(pattern, callback)
                # TODO: Implement pattern subscription in BufferedMessageBusClient
            except Exception as e:
                self.logger.error(f"Pattern subscription failed: {e}")
                self.msgbus.subscribe(pattern, callback)
        else:
            self.msgbus.subscribe(pattern, callback)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics (cached)."""
        current_time = time.time()
        
        if (self._metrics_cache is None or 
            (current_time - self._last_metrics_time) > self._cache_duration):
            
            if self.is_enhanced():
                enhanced_msgbus = self.msgbus
                try:
                    self._metrics_cache = enhanced_msgbus.get_metrics()
                    self._last_metrics_time = current_time
                except Exception as e:
                    self.logger.error(f"Failed to get enhanced metrics: {e}")
                    self._metrics_cache = {
                        "enhanced": True,
                        "error": str(e),
                        "message": "Enhanced MessageBus metrics unavailable"
                    }
            else:
                self._metrics_cache = {
                    "enhanced": False,
                    "message": "Standard MessageBus - no metrics available"
                }
        
        return self._metrics_cache
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status."""
        if self.is_enhanced():
            enhanced_msgbus = self.msgbus
            try:
                # This would need to be an async method in practice
                return {
                    "enhanced": True,
                    "connected": enhanced_msgbus.is_connected(),
                    "healthy": True,  # Simplified check
                    "capabilities": self.get_capabilities()
                }
            except Exception as e:
                return {
                    "enhanced": True,
                    "healthy": False,
                    "error": str(e)
                }
        else:
            return {
                "enhanced": False,
                "healthy": True,
                "message": "Standard MessageBus - basic health check"
            }
    
    async def flush_buffers(self) -> None:
        """Flush message buffers."""
        if self.is_enhanced():
            enhanced_msgbus = self.msgbus
            try:
                # Force flush of all priority buffers
                await enhanced_msgbus._flush_all_buffers()
            except Exception as e:
                self.logger.error(f"Buffer flush failed: {e}")


class ComponentPerformanceMonitor:
    """
    Performance monitoring utility for components using enhanced MessageBus.
    
    Provides automatic performance monitoring and alerting for components
    that utilize enhanced MessageBus features.
    """
    
    def __init__(self, component_name: str, msgbus: MessageBus):
        self.component_name = component_name
        self.msgbus = msgbus
        self.logger = logging.getLogger(f"{__name__}.PerfMonitor.{component_name}")
        self.enhanced_features = EnhancedMessageBusFeatures(msgbus)
        
        # Performance tracking
        self.message_counts: Dict[str, int] = {}
        self.error_counts: Dict[str, int] = {}
        self.start_time = time.time()
        self.last_report_time = time.time()
        self.report_interval = 60.0  # Report every minute
    
    def record_message_sent(self, topic: str, priority: MessagePriority = MessagePriority.NORMAL) -> None:
        """Record a message being sent."""
        key = f"sent.{priority.name}.{topic}"
        self.message_counts[key] = self.message_counts.get(key, 0) + 1
        
        # Auto-report if interval exceeded
        self._maybe_report_performance()
    
    def record_message_received(self, topic: str) -> None:
        """Record a message being received."""
        key = f"received.{topic}"
        self.message_counts[key] = self.message_counts.get(key, 0) + 1
        
        self._maybe_report_performance()
    
    def record_error(self, error_type: str, topic: str = "unknown") -> None:
        """Record an error."""
        key = f"error.{error_type}.{topic}"
        self.error_counts[key] = self.error_counts.get(key, 0) + 1
        
        self._maybe_report_performance()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        current_time = time.time()
        uptime = current_time - self.start_time
        
        total_messages = sum(self.message_counts.values())
        total_errors = sum(self.error_counts.values())
        
        summary = {
            "component": self.component_name,
            "uptime_seconds": uptime,
            "total_messages": total_messages,
            "total_errors": total_errors,
            "messages_per_second": total_messages / uptime if uptime > 0 else 0,
            "error_rate": total_errors / total_messages if total_messages > 0 else 0,
            "enhanced_messagebus": self.enhanced_features.is_enhanced()
        }
        
        if self.enhanced_features.is_enhanced():
            summary["messagebus_metrics"] = self.enhanced_features.get_metrics()
        
        return summary
    
    def _maybe_report_performance(self) -> None:
        """Report performance if interval exceeded."""
        current_time = time.time()
        
        if (current_time - self.last_report_time) >= self.report_interval:
            summary = self.get_performance_summary()
            
            self.logger.info(
                f"Performance Report - Messages: {summary['total_messages']}, "
                f"Rate: {summary['messages_per_second']:.1f}/sec, "
                f"Errors: {summary['total_errors']} ({summary['error_rate']:.2%})"
            )
            
            # Log enhanced MessageBus metrics if available
            if summary["enhanced_messagebus"]:
                msgbus_metrics = summary.get("messagebus_metrics", {})
                if "client_metrics" in msgbus_metrics:
                    client_metrics = msgbus_metrics["client_metrics"]
                    self.logger.info(
                        f"Enhanced MessageBus - Buffer utilization: {client_metrics.get('buffer_utilization', 0):.1%}, "
                        f"Workers: {client_metrics.get('active_workers', 0)}"
                    )
            
            self.last_report_time = current_time


# =============================================================================
# COMPONENT ENHANCEMENT UTILITIES
# =============================================================================

def enhance_component_with_messagebus_features(component) -> None:
    """
    Enhance an existing component with enhanced MessageBus features.
    
    This utility function can be used to add enhanced MessageBus capabilities
    to existing components that don't inherit from MessageBusIntegrationMixin.
    
    Args:
        component: Component instance to enhance
    """
    if not hasattr(component, '_msgbus'):
        raise ValueError("Component must have _msgbus attribute")
    
    # Add enhanced features
    component._enhanced_msgbus_features = EnhancedMessageBusFeatures(component._msgbus)
    component._performance_monitor = ComponentPerformanceMonitor(
        component_name=component.__class__.__name__,
        msgbus=component._msgbus
    )
    
    # Add enhanced methods
    def publish_with_priority(topic: str, message: Any, priority: MessagePriority = MessagePriority.NORMAL):
        component._performance_monitor.record_message_sent(topic, priority)
        component._enhanced_msgbus_features.publish_with_priority(topic, message, priority)
    
    def get_messagebus_info():
        return component._enhanced_msgbus_features.get_capabilities()
    
    def get_performance_summary():
        return component._performance_monitor.get_performance_summary()
    
    # Attach methods to component
    component.publish_with_priority = publish_with_priority
    component.get_messagebus_info = get_messagebus_info
    component.get_performance_summary = get_performance_summary
    
    # Log enhancement
    logger = logging.getLogger(f"{__name__}.Enhancement")
    logger.info(f"Enhanced component {component.__class__.__name__} with MessageBus features")


def create_enhanced_component_decorator(enable_performance_monitoring: bool = True):
    """
    Create a decorator that automatically enhances components with MessageBus features.
    
    Args:
        enable_performance_monitoring: Whether to enable automatic performance monitoring
        
    Returns:
        Decorator function
    """
    def decorator(cls):
        original_init = cls.__init__
        
        def enhanced_init(self, *args, **kwargs):
            # Call original init
            original_init(self, *args, **kwargs)
            
            # Add enhanced features after initialization
            if hasattr(self, '_msgbus') and self._msgbus is not None:
                enhance_component_with_messagebus_features(self)
                
                if enable_performance_monitoring:
                    # Start automatic performance monitoring
                    logger = logging.getLogger(f"{cls.__name__}.Enhanced")
                    logger.info("Enhanced MessageBus features enabled with performance monitoring")
        
        cls.__init__ = enhanced_init
        return cls
    
    return decorator


# Convenience decorator for simple enhancement
enhanced_messagebus_component = create_enhanced_component_decorator(enable_performance_monitoring=True)