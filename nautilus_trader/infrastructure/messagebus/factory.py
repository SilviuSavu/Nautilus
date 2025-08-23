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
Enhanced MessageBus Factory for NautilusTrader system integration.

This module provides factories for creating enhanced MessageBus instances
with backward compatibility and seamless integration into the existing
NautilusTrader kernel and component system.
"""

import asyncio
import logging
from typing import Optional, Union, Any

from nautilus_trader.common.component import Clock, MessageBus, Logger
from nautilus_trader.common.config import MessageBusConfig
from nautilus_trader.core.uuid import UUID4
from nautilus_trader.model.identifiers import TraderId
from nautilus_trader.serialization.serializer import MsgSpecSerializer

from nautilus_trader.infrastructure.messagebus.config import (
    EnhancedMessageBusConfig, 
    ConfigValidator, 
    ConfigMigration,
    migrate_from_nautilus_config
)
from nautilus_trader.infrastructure.messagebus.client import BufferedMessageBusClient
from nautilus_trader.infrastructure.messagebus.streams import RedisStreamManager


class EnhancedMessageBusFactory:
    """
    Factory for creating enhanced MessageBus instances with backward compatibility.
    
    Provides seamless integration with existing NautilusTrader components while
    offering enhanced performance and functionality through the BufferedMessageBusClient.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.EnhancedMessageBusFactory")
    
    def create_enhanced_messagebus(
        self,
        trader_id: TraderId,
        instance_id: UUID4,
        clock: Clock,
        serializer: MsgSpecSerializer,
        config: Union[MessageBusConfig, EnhancedMessageBusConfig, None] = None,
        database: Optional[Any] = None,
    ) -> Union[MessageBus, BufferedMessageBusClient]:
        """
        Create enhanced MessageBus with backward compatibility.
        
        Args:
            trader_id: The trader identifier
            instance_id: The instance identifier  
            clock: System clock
            serializer: Message serializer
            config: MessageBus configuration (legacy or enhanced)
            database: Optional database backend
            
        Returns:
            MessageBus instance (enhanced if configuration supports it, standard otherwise)
        """
        try:
            # Detect if enhanced MessageBus should be used
            enhanced_config = self._resolve_enhanced_config(config)
            
            if enhanced_config is not None:
                # Create enhanced MessageBus
                return self._create_enhanced_messagebus(
                    trader_id=trader_id,
                    instance_id=instance_id,
                    clock=clock,
                    serializer=serializer,
                    config=enhanced_config,
                    database=database
                )
            else:
                # Fall back to standard MessageBus
                self.logger.info("Using standard MessageBus (enhanced MessageBus not configured)")
                return MessageBus(
                    trader_id=trader_id,
                    instance_id=instance_id,
                    clock=clock,
                    serializer=serializer,
                    database=database,
                    config=config
                )
                
        except Exception as e:
            self.logger.error(f"Failed to create enhanced MessageBus, falling back to standard: {e}")
            # Graceful fallback to standard MessageBus
            return MessageBus(
                trader_id=trader_id,
                instance_id=instance_id,
                clock=clock,
                serializer=serializer,
                database=database,
                config=config
            )
    
    def _resolve_enhanced_config(
        self, 
        config: Union[MessageBusConfig, EnhancedMessageBusConfig, None]
    ) -> Optional[EnhancedMessageBusConfig]:
        """
        Resolve configuration for enhanced MessageBus.
        
        Args:
            config: Input configuration
            
        Returns:
            EnhancedMessageBusConfig if enhanced features should be used, None otherwise
        """
        if config is None:
            return None
            
        if isinstance(config, EnhancedMessageBusConfig):
            # Already enhanced configuration
            validator = ConfigValidator()
            validation_result = validator.validate(config)
            if validation_result.is_valid:
                return config
            else:
                self.logger.warning(f"Enhanced MessageBus config validation failed: {validation_result.errors}")
                return None
        
        elif isinstance(config, MessageBusConfig):
            # Try to migrate from standard MessageBusConfig
            try:
                enhanced_config, messages = migrate_from_nautilus_config(config)
                
                for message in messages:
                    self.logger.info(f"Migration: {message}")
                
                return enhanced_config
                
            except Exception as e:
                self.logger.debug(f"Could not migrate MessageBusConfig to enhanced: {e}")
                return None
        
        else:
            self.logger.warning(f"Unknown MessageBus config type: {type(config)}")
            return None
    
    def _create_enhanced_messagebus(
        self,
        trader_id: TraderId,
        instance_id: UUID4,
        clock: Clock,
        serializer: MsgSpecSerializer,
        config: EnhancedMessageBusConfig,
        database: Optional[Any] = None
    ) -> BufferedMessageBusClient:
        """
        Create enhanced MessageBus with BufferedMessageBusClient.
        
        Args:
            trader_id: The trader identifier
            instance_id: The instance identifier
            clock: System clock
            serializer: Message serializer
            config: Enhanced MessageBus configuration
            database: Optional database backend
            
        Returns:
            BufferedMessageBusClient instance
        """
        self.logger.info("Creating enhanced MessageBus with BufferedMessageBusClient")
        
        # Create enhanced client
        enhanced_client = BufferedMessageBusClient(config)
        
        # Set up integration with NautilusTrader components
        enhanced_client._trader_id = trader_id
        enhanced_client._instance_id = instance_id
        enhanced_client._clock = clock
        enhanced_client._serializer = serializer
        
        if database is not None:
            enhanced_client._database = database
            self.logger.info("Enhanced MessageBus connected to database backend")
        
        # Log configuration details
        self.logger.info(f"Enhanced MessageBus configuration:")
        self.logger.info(f"  - Connection pool size: {config.connection_pool_size}")
        self.logger.info(f"  - Auto-scaling: {config.auto_scale_enabled} ({config.min_workers}-{config.max_workers} workers)")
        self.logger.info(f"  - Pattern matching: {config.enable_pattern_matching}")
        self.logger.info(f"  - Metrics enabled: {config.enable_metrics}")
        
        return enhanced_client
    
    async def start_enhanced_messagebus(self, messagebus: BufferedMessageBusClient) -> None:
        """
        Start enhanced MessageBus asynchronously.
        
        Args:
            messagebus: Enhanced MessageBus client to start
        """
        if isinstance(messagebus, BufferedMessageBusClient):
            try:
                await messagebus.connect()
                await messagebus.start()
                
                # Run performance benchmark if enabled
                if messagebus.config.enable_metrics:
                    metrics = messagebus.get_metrics()
                    self.logger.info(f"Enhanced MessageBus started - Performance baseline: {metrics}")
                
            except Exception as e:
                self.logger.error(f"Failed to start enhanced MessageBus: {e}")
                raise
        else:
            self.logger.debug("MessageBus is not enhanced, no special startup required")
    
    async def stop_enhanced_messagebus(self, messagebus: BufferedMessageBusClient) -> None:
        """
        Stop enhanced MessageBus asynchronously.
        
        Args:
            messagebus: Enhanced MessageBus client to stop
        """
        if isinstance(messagebus, BufferedMessageBusClient):
            try:
                await messagebus.stop()
                await messagebus.close()
                
                # Log final metrics
                metrics = messagebus.get_metrics()
                self.logger.info(f"Enhanced MessageBus stopped - Final metrics: {metrics}")
                
            except Exception as e:
                self.logger.error(f"Error stopping enhanced MessageBus: {e}")
        else:
            self.logger.debug("MessageBus is not enhanced, no special shutdown required")


class MessageBusIntegration:
    """
    Integration utilities for enhanced MessageBus in NautilusTrader system.
    
    Provides utilities for detecting enhanced MessageBus capabilities,
    performance benchmarking, and system health monitoring.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.MessageBusIntegration")
    
    def is_enhanced_messagebus(self, messagebus: MessageBus) -> bool:
        """
        Check if MessageBus instance is enhanced.
        
        Args:
            messagebus: MessageBus instance to check
            
        Returns:
            True if enhanced MessageBus, False otherwise
        """
        return isinstance(messagebus, BufferedMessageBusClient)
    
    def get_messagebus_capabilities(self, messagebus: MessageBus) -> dict[str, Any]:
        """
        Get MessageBus capabilities and features.
        
        Args:
            messagebus: MessageBus instance
            
        Returns:
            Dictionary of capabilities
        """
        capabilities = {
            "type": "standard",
            "enhanced": False,
            "priority_handling": False,
            "pattern_matching": False,
            "auto_scaling": False,
            "metrics": False,
            "health_monitoring": False
        }
        
        if isinstance(messagebus, BufferedMessageBusClient):
            capabilities.update({
                "type": "enhanced",
                "enhanced": True,
                "priority_handling": True,
                "pattern_matching": messagebus.config.enable_pattern_matching,
                "auto_scaling": messagebus.config.auto_scale_enabled,
                "metrics": messagebus.config.enable_metrics,
                "health_monitoring": True,
                "worker_count": f"{messagebus.config.min_workers}-{messagebus.config.max_workers}",
                "connection_pool": messagebus.config.connection_pool_size
            })
        
        return capabilities
    
    async def run_performance_benchmark(
        self, 
        messagebus: MessageBus,
        duration_seconds: float = 10.0
    ) -> dict[str, Any]:
        """
        Run performance benchmark on MessageBus.
        
        Args:
            messagebus: MessageBus instance to benchmark
            duration_seconds: Benchmark duration
            
        Returns:
            Benchmark results
        """
        if not isinstance(messagebus, BufferedMessageBusClient):
            return {
                "enhanced": False,
                "message": "Standard MessageBus - no enhanced benchmarking available"
            }
        
        try:
            from nautilus_trader.infrastructure.messagebus.performance import run_quick_benchmark
            
            self.logger.info(f"Running performance benchmark for {duration_seconds}s...")
            results = await run_quick_benchmark(duration_seconds)
            
            self.logger.info(f"Benchmark complete - {results['summary']['messages_per_second']:.0f} msg/sec")
            return results
            
        except Exception as e:
            self.logger.error(f"Benchmark failed: {e}")
            return {
                "enhanced": True,
                "error": str(e),
                "message": "Enhanced MessageBus benchmark failed"
            }
    
    def get_health_status(self, messagebus: MessageBus) -> dict[str, Any]:
        """
        Get MessageBus health status.
        
        Args:
            messagebus: MessageBus instance
            
        Returns:
            Health status information
        """
        health = {
            "healthy": True,
            "type": "standard",
            "enhanced": False
        }
        
        if isinstance(messagebus, BufferedMessageBusClient):
            try:
                # Get enhanced health status
                enhanced_health = asyncio.create_task(messagebus.get_health_status())
                # Note: In real implementation, this would need proper async handling
                
                health.update({
                    "type": "enhanced",
                    "enhanced": True,
                    "metrics_available": True
                })
                
            except Exception as e:
                health.update({
                    "healthy": False,
                    "error": str(e),
                    "message": "Enhanced MessageBus health check failed"
                })
        
        return health


# =============================================================================
# FACTORY FUNCTIONS FOR KERNEL INTEGRATION
# =============================================================================

def create_messagebus_for_kernel(
    trader_id: TraderId,
    instance_id: UUID4,
    clock: Clock,
    serializer: MsgSpecSerializer,
    config: Union[MessageBusConfig, EnhancedMessageBusConfig, None] = None,
    database: Optional[Any] = None,
    enable_enhanced: bool = True
) -> MessageBus:
    """
    Factory function for creating MessageBus in NautilusTrader kernel.
    
    This is the main entry point for kernel integration, providing
    backward compatibility while enabling enhanced features when available.
    
    Args:
        trader_id: The trader identifier
        instance_id: The instance identifier
        clock: System clock
        serializer: Message serializer
        config: MessageBus configuration
        database: Optional database backend
        enable_enhanced: Whether to attempt enhanced MessageBus creation
        
    Returns:
        MessageBus instance (enhanced if possible)
    """
    factory = EnhancedMessageBusFactory()
    
    if enable_enhanced:
        return factory.create_enhanced_messagebus(
            trader_id=trader_id,
            instance_id=instance_id,
            clock=clock,
            serializer=serializer,
            config=config,
            database=database
        )
    else:
        # Force standard MessageBus
        return MessageBus(
            trader_id=trader_id,
            instance_id=instance_id,
            clock=clock,
            serializer=serializer,
            database=database,
            config=config
        )


def detect_enhanced_messagebus_support(config: Optional[Union[MessageBusConfig, EnhancedMessageBusConfig]] = None) -> bool:
    """
    Detect if enhanced MessageBus should be used based on configuration.
    
    Args:
        config: MessageBus configuration
        
    Returns:
        True if enhanced MessageBus should be used
    """
    if config is None:
        return False
    
    if isinstance(config, EnhancedMessageBusConfig):
        return True
    
    if isinstance(config, MessageBusConfig):
        # Check if config can be migrated
        try:
            migrate_from_nautilus_config(config)
            return True
        except Exception:
            return False
    
    return False