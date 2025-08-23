"""
Real-time Data Streaming Service - Sprint 3 Priority 1

Handles real-time data streaming for:
- Engine status and health monitoring
- Market data feeds
- Trade execution updates
- System performance metrics
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import redis

from .websocket_manager import websocket_manager
from .message_protocols import MessageProtocol, EngineStatusMessage, MarketDataMessage, TradeUpdateMessage, SystemHealthMessage

logger = logging.getLogger(__name__)


class StreamingService:
    """
    Core streaming service for real-time data distribution
    
    Integrates with:
    - NautilusTrader engine for live data
    - Redis cache for market data
    - WebSocket manager for client distribution
    """
    
    def __init__(self):
        self.redis_client = None
        self.message_protocol = MessageProtocol()
        self.streaming_tasks: Dict[str, asyncio.Task] = {}
        self.stream_intervals = {
            "engine_status": 2.0,      # 2 seconds
            "market_data": 0.1,        # 100ms for high-frequency data
            "trade_updates": 0.5,      # 500ms for trade updates
            "system_health": 5.0       # 5 seconds for system metrics
        }
        
    async def stream_engine_status(self, connection_id: str) -> None:
        """
        Stream real-time engine status updates
        
        Args:
            connection_id: WebSocket connection identifier
        """
        logger.info(f"Starting engine status stream for connection: {connection_id}")
        
        try:
            while True:
                try:
                    # Get current engine status (mock for development)
                    engine_status = await self._get_mock_engine_status()
                    
                    # Create structured message
                    message = EngineStatusMessage(
                        type="engine_status",
                        data=engine_status,
                        engine_id="nautilus-001"
                    )
                    
                    # Validate and send message
                    if self.message_protocol.validate_message(message.dict()):
                        await websocket_manager.send_personal_message(
                            message.dict(),
                            connection_id
                        )
                    
                    # Wait for next update interval
                    await asyncio.sleep(self.stream_intervals["engine_status"])
                    
                except asyncio.CancelledError:
                    logger.info(f"Engine status stream cancelled for: {connection_id}")
                    break
                except Exception as e:
                    logger.error(f"Error in engine status stream: {e}")
                    await asyncio.sleep(1.0)  # Brief pause on error
                    
        except Exception as e:
            logger.error(f"Engine status streaming failed for {connection_id}: {e}")
            
    async def stream_market_data(self, connection_id: str, symbol: str) -> None:
        """
        Stream real-time market data for a specific symbol
        
        Args:
            connection_id: WebSocket connection identifier
            symbol: Trading symbol to stream
        """
        logger.info(f"Starting market data stream for {symbol} on connection: {connection_id}")
        
        try:
            while True:
                try:
                    # Get latest market data from Redis cache
                    market_data = await self._get_market_data_update(symbol)
                    
                    if market_data:
                        # Create structured message
                        message = MarketDataMessage(
                            type="market_data",
                            symbol=symbol,
                            data=market_data,
                            timestamp=datetime.utcnow(),
                            connection_id=connection_id
                        )
                        
                        # Validate and send message
                        if self.message_protocol.validate_message(message.dict()):
                            await websocket_manager.send_personal_message(
                                message.dict(),
                                connection_id
                            )
                    
                    # High-frequency updates for market data
                    await asyncio.sleep(self.stream_intervals["market_data"])
                    
                except asyncio.CancelledError:
                    logger.info(f"Market data stream cancelled for {symbol}: {connection_id}")
                    break
                except Exception as e:
                    logger.error(f"Error in market data stream for {symbol}: {e}")
                    await asyncio.sleep(0.5)  # Brief pause on error
                    
        except Exception as e:
            logger.error(f"Market data streaming failed for {symbol} on {connection_id}: {e}")
            
    async def stream_trade_updates(self, connection_id: str, user_id: str) -> None:
        """
        Stream real-time trade execution updates for a user
        
        Args:
            connection_id: WebSocket connection identifier
            user_id: User identifier for trade filtering
        """
        logger.info(f"Starting trade updates stream for user {user_id} on connection: {connection_id}")
        
        try:
            while True:
                try:
                    # Get trade updates for user from engine
                    trade_updates = await self._get_trade_updates(user_id)
                    
                    if trade_updates:
                        # Create structured message
                        message = TradeUpdateMessage(
                            type="trade_updates",
                            user_id=user_id,
                            data=trade_updates,
                            timestamp=datetime.utcnow(),
                            connection_id=connection_id
                        )
                        
                        # Validate and send message
                        if self.message_protocol.validate_message(message.dict()):
                            await websocket_manager.send_personal_message(
                                message.dict(),
                                connection_id
                            )
                    
                    # Regular updates for trade status
                    await asyncio.sleep(self.stream_intervals["trade_updates"])
                    
                except asyncio.CancelledError:
                    logger.info(f"Trade updates stream cancelled for user {user_id}: {connection_id}")
                    break
                except Exception as e:
                    logger.error(f"Error in trade updates stream for user {user_id}: {e}")
                    await asyncio.sleep(1.0)  # Brief pause on error
                    
        except Exception as e:
            logger.error(f"Trade updates streaming failed for user {user_id} on {connection_id}: {e}")
            
    async def stream_system_health(self, connection_id: str) -> None:
        """
        Stream real-time system health and performance metrics
        
        Args:
            connection_id: WebSocket connection identifier
        """
        logger.info(f"Starting system health stream for connection: {connection_id}")
        
        try:
            while True:
                try:
                    # Get system health metrics
                    health_data = await self._get_system_health_metrics()
                    
                    # Create structured message
                    message = SystemHealthMessage(
                        type="system_health",
                        data=health_data,
                        timestamp=datetime.utcnow(),
                        connection_id=connection_id
                    )
                    
                    # Validate and send message
                    if self.message_protocol.validate_message(message.dict()):
                        await websocket_manager.send_personal_message(
                            message.dict(),
                            connection_id
                        )
                    
                    # Regular updates for system health
                    await asyncio.sleep(self.stream_intervals["system_health"])
                    
                except asyncio.CancelledError:
                    logger.info(f"System health stream cancelled for: {connection_id}")
                    break
                except Exception as e:
                    logger.error(f"Error in system health stream: {e}")
                    await asyncio.sleep(2.0)  # Brief pause on error
                    
        except Exception as e:
            logger.error(f"System health streaming failed for {connection_id}: {e}")
            
    async def start_custom_stream(self, connection_id: str, stream_config: Dict[str, Any]) -> bool:
        """
        Start a custom data stream with specified configuration
        
        Args:
            connection_id: WebSocket connection identifier
            stream_config: Configuration for custom stream
            
        Returns:
            bool: True if stream started successfully
        """
        try:
            stream_type = stream_config.get("type")
            stream_id = f"{connection_id}_{stream_type}_{datetime.utcnow().timestamp()}"
            
            if stream_type == "portfolio_updates":
                task = asyncio.create_task(
                    self._stream_portfolio_updates(connection_id, stream_config)
                )
            elif stream_type == "risk_metrics":
                task = asyncio.create_task(
                    self._stream_risk_metrics(connection_id, stream_config)
                )
            elif stream_type == "strategy_performance":
                task = asyncio.create_task(
                    self._stream_strategy_performance(connection_id, stream_config)
                )
            else:
                logger.warning(f"Unknown custom stream type: {stream_type}")
                return False
            
            self.streaming_tasks[stream_id] = task
            logger.info(f"Started custom stream {stream_type} for connection: {connection_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start custom stream: {e}")
            return False
            
    async def stop_stream(self, connection_id: str, stream_type: Optional[str] = None) -> int:
        """
        Stop streaming tasks for a connection
        
        Args:
            connection_id: WebSocket connection identifier
            stream_type: Optional specific stream type to stop
            
        Returns:
            int: Number of streams stopped
        """
        stopped_count = 0
        
        # Find tasks to stop
        tasks_to_stop = []
        for stream_id, task in list(self.streaming_tasks.items()):
            if connection_id in stream_id:
                if stream_type is None or stream_type in stream_id:
                    tasks_to_stop.append((stream_id, task))
        
        # Cancel tasks
        for stream_id, task in tasks_to_stop:
            try:
                task.cancel()
                await asyncio.sleep(0.1)  # Allow task to cleanup
                del self.streaming_tasks[stream_id]
                stopped_count += 1
                logger.info(f"Stopped stream: {stream_id}")
            except Exception as e:
                logger.error(f"Error stopping stream {stream_id}: {e}")
        
        return stopped_count
        
    async def get_stream_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about active streams
        
        Returns:
            Dict containing stream statistics
        """
        active_streams = len(self.streaming_tasks)
        stream_types = {}
        
        for stream_id in self.streaming_tasks.keys():
            parts = stream_id.split("_")
            if len(parts) >= 2:
                stream_type = parts[1]
                stream_types[stream_type] = stream_types.get(stream_type, 0) + 1
        
        return {
            "active_streams": active_streams,
            "stream_types": stream_types,
            "stream_intervals": self.stream_intervals,
            "last_updated": datetime.utcnow().isoformat()
        }
        
    # Private helper methods
    
    async def _get_market_data_update(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get latest market data update for symbol"""
        try:
            # Mock market data for development
            import random
            return {
                "symbol": symbol,
                "price": round(150.0 + random.uniform(-5, 5), 2),
                "bid": round(149.95 + random.uniform(-5, 5), 2),
                "ask": round(150.05 + random.uniform(-5, 5), 2),
                "volume": random.randint(1000, 10000),
                "timestamp": datetime.utcnow().isoformat()
            }
                
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            
        return None
        
    async def _get_trade_updates(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get trade updates for user"""
        try:
            # This would integrate with your trading service
            # For now, return simulated data structure
            return {
                "user_id": user_id,
                "active_orders": [],
                "recent_fills": [],
                "position_updates": [],
                "pnl_changes": {
                    "unrealized_pnl": 0.0,
                    "realized_pnl": 0.0,
                    "total_pnl": 0.0
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting trade updates for user {user_id}: {e}")
            
        return None
        
    async def _get_system_health_metrics(self) -> Dict[str, Any]:
        """Get current system health metrics"""
        try:
            # Mock system health for development
            import random
            
            return {
                "engine": await self._get_mock_engine_status(),
                "redis": {"status": "connected", "ping_ms": 1.2},
                "websocket_connections": websocket_manager.get_connection_stats(),
                "streaming_stats": await self.get_stream_statistics(),
                "system_load": {
                    "cpu_percent": round(random.uniform(10, 80), 1),
                    "memory_percent": round(random.uniform(40, 90), 1),
                    "disk_usage": round(random.uniform(15, 50), 1)
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting system health metrics: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
            
    async def _get_mock_engine_status(self) -> Dict[str, Any]:
        """Get mock engine status for development"""
        import random
        return {
            "state": "running",
            "uptime": random.randint(3600, 86400),
            "memory_usage": f"{random.randint(200, 800)}MB",
            "cpu_usage": f"{round(random.uniform(5, 50), 1)}%",
            "active_strategies": random.randint(1, 5),
            "processed_messages": random.randint(1000, 50000)
        }
            
    async def _stream_portfolio_updates(self, connection_id: str, config: Dict[str, Any]) -> None:
        """Stream portfolio updates for custom stream"""
        while True:
            try:
                # Get portfolio data based on config
                portfolio_data = await self._get_portfolio_data(config)
                
                if portfolio_data:
                    await websocket_manager.send_personal_message({
                        "type": "portfolio_updates",
                        "data": portfolio_data,
                        "timestamp": datetime.utcnow().isoformat()
                    }, connection_id)
                
                await asyncio.sleep(config.get("interval", 1.0))
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in portfolio stream: {e}")
                await asyncio.sleep(1.0)
                
    async def _stream_risk_metrics(self, connection_id: str, config: Dict[str, Any]) -> None:
        """Stream risk metrics for custom stream"""
        while True:
            try:
                # Get risk data based on config
                risk_data = await self._get_risk_metrics(config)
                
                if risk_data:
                    await websocket_manager.send_personal_message({
                        "type": "risk_metrics",
                        "data": risk_data,
                        "timestamp": datetime.utcnow().isoformat()
                    }, connection_id)
                
                await asyncio.sleep(config.get("interval", 2.0))
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in risk metrics stream: {e}")
                await asyncio.sleep(1.0)
                
    async def _stream_strategy_performance(self, connection_id: str, config: Dict[str, Any]) -> None:
        """Stream strategy performance for custom stream"""
        while True:
            try:
                # Get strategy performance data based on config
                performance_data = await self._get_strategy_performance(config)
                
                if performance_data:
                    await websocket_manager.send_personal_message({
                        "type": "strategy_performance",
                        "data": performance_data,
                        "timestamp": datetime.utcnow().isoformat()
                    }, connection_id)
                
                await asyncio.sleep(config.get("interval", 3.0))
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in strategy performance stream: {e}")
                await asyncio.sleep(1.0)
                
    async def _get_portfolio_data(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get portfolio data (placeholder for actual implementation)"""
        return {
            "total_value": 1000000.0,
            "positions": [],
            "cash_balance": 100000.0,
            "unrealized_pnl": 5000.0
        }
        
    async def _get_risk_metrics(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get risk metrics (placeholder for actual implementation)"""
        return {
            "var_1d": -5000.0,
            "max_drawdown": -2.5,
            "sharpe_ratio": 1.8,
            "risk_exposure": 0.85
        }
        
    async def _get_strategy_performance(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get strategy performance (placeholder for actual implementation)"""
        return {
            "total_return": 12.5,
            "win_rate": 0.65,
            "profit_factor": 1.8,
            "max_drawdown": -5.2
        }


# Helper function to get Redis client
def get_redis():
    """Get Redis client instance"""
    try:
        from ..redis_cache import redis_cache
        return redis_cache
    except ImportError:
        logger.warning("Redis cache not available, using mock client")
        return None
    except Exception as e:
        logger.warning(f"Redis cache import failed: {e}")
        return None