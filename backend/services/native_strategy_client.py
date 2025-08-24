#!/usr/bin/env python3
"""
Native Strategy Client for Docker Backend
Unix Domain Socket client for communicating with native strategy engine

This component provides:
- Unix socket communication with native strategy engine
- Connection pooling and retry logic
- Async interface for strategy execution
- Fallback to containerized strategy when native unavailable
"""

import asyncio
import json
import logging
import socket
import struct
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List
import contextlib

@dataclass
class TradingSignal:
    """Trading signal structure"""
    signal_id: str
    symbol: str
    signal_type: str  # BUY, SELL, HOLD
    confidence: float
    price: float
    quantity: int
    timestamp: float
    pattern_detected: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

@dataclass
class StrategyRequest:
    """Strategy execution request"""
    request_id: str
    strategy_type: str
    market_data: Dict[str, Any]
    parameters: Dict[str, Any]
    timestamp: float

@dataclass
class StrategyResponse:
    """Strategy execution response"""
    request_id: str
    signals: List[Dict[str, Any]]  # Serialized TradingSignals
    processing_time_ms: float
    hardware_used: str
    patterns_analyzed: int
    timestamp: float
    error: Optional[str] = None

class NativeStrategyConnection:
    """Single connection to native strategy engine"""
    
    def __init__(self, socket_path: str, timeout: float = 30.0):
        self.socket_path = socket_path
        self.timeout = timeout
        self.sock = None
        self.logger = logging.getLogger(__name__)
        
    async def connect(self) -> bool:
        """Connect to native strategy engine"""
        try:
            # Create Unix socket
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.sock.settimeout(self.timeout)
            
            # Connect to native strategy engine
            await asyncio.get_event_loop().run_in_executor(
                None, self.sock.connect, self.socket_path
            )
            
            self.logger.info(f"Connected to native strategy engine at {self.socket_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to native strategy engine: {e}")
            if self.sock:
                self.sock.close()
                self.sock = None
            return False
    
    async def send_request(self, request: StrategyRequest) -> Optional[StrategyResponse]:
        """Send strategy execution request and receive response"""
        if not self.sock:
            return None
            
        try:
            # Serialize request
            request_data = json.dumps(asdict(request)).encode('utf-8')
            
            # Send message length followed by data
            await asyncio.get_event_loop().run_in_executor(
                None, self.sock.send, struct.pack('!I', len(request_data))
            )
            await asyncio.get_event_loop().run_in_executor(
                None, self.sock.send, request_data
            )
            
            # Receive response length
            length_data = await asyncio.get_event_loop().run_in_executor(
                None, self.sock.recv, 4
            )
            if not length_data:
                raise ConnectionError("Connection closed by native strategy engine")
            
            response_length = struct.unpack('!I', length_data)[0]
            
            # Receive response data
            response_data = b''
            while len(response_data) < response_length:
                chunk = await asyncio.get_event_loop().run_in_executor(
                    None, self.sock.recv, response_length - len(response_data)
                )
                if not chunk:
                    raise ConnectionError("Incomplete response from native strategy engine")
                response_data += chunk
            
            # Parse response
            response_json = json.loads(response_data.decode('utf-8'))
            
            return StrategyResponse(**response_json)
            
        except Exception as e:
            self.logger.error(f"Strategy request failed: {e}")
            return None
    
    def disconnect(self):
        """Disconnect from native strategy engine"""
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
            finally:
                self.sock = None
        
        self.logger.info("Disconnected from native strategy engine")
    
    def is_connected(self) -> bool:
        """Check if connection is active"""
        return self.sock is not None

class NativeStrategyClient:
    """High-level client for native strategy engine communication"""
    
    def __init__(self, socket_path: str = "/tmp/nautilus_strategy_engine.sock", 
                 enable_fallback: bool = True):
        self.socket_path = socket_path
        self.enable_fallback = enable_fallback
        self.connection = None
        self.logger = logging.getLogger(__name__)
        
        # Statistics
        self.stats = {
            "requests_sent": 0,
            "responses_received": 0,
            "errors": 0,
            "native_engine_requests": 0,
            "fallback_requests": 0,
            "average_latency_ms": 0.0,
            "total_latency_ms": 0.0,
            "signals_generated": 0
        }
        
    async def initialize(self):
        """Initialize native strategy client"""
        try:
            self.connection = NativeStrategyConnection(self.socket_path)
            connected = await self.connection.connect()
            
            if connected:
                self.logger.info("Native strategy client initialized successfully")
            else:
                self.logger.warning("Failed to initialize native strategy client")
                if not self.enable_fallback:
                    raise RuntimeError("Native strategy engine connection failed and fallback disabled")
        except Exception as e:
            self.logger.error(f"Failed to initialize native strategy client: {e}")
            if not self.enable_fallback:
                raise
    
    async def execute_strategy(self, strategy_type: str, market_data: Dict[str, Any], 
                             parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute trading strategy using native engine or fallback"""
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        self.stats["requests_sent"] += 1
        
        # Try native strategy engine first
        try:
            if self.connection and self.connection.is_connected():
                request = StrategyRequest(
                    request_id=request_id,
                    strategy_type=strategy_type,
                    market_data=market_data,
                    parameters=parameters,
                    timestamp=start_time
                )
                
                response = await self.connection.send_request(request)
                
                if response and not response.error:
                    # Success with native engine
                    processing_time = (time.time() - start_time) * 1000
                    self._update_stats(processing_time, native=True, signal_count=len(response.signals))
                    
                    return {
                        "success": True,
                        "signals": response.signals,
                        "processing_time_ms": response.processing_time_ms,
                        "total_time_ms": processing_time,
                        "hardware_used": response.hardware_used,
                        "patterns_analyzed": response.patterns_analyzed,
                        "source": "native_strategy_engine"
                    }
                elif response and response.error:
                    self.logger.warning(f"Native strategy engine returned error: {response.error}")
                else:
                    self.logger.warning("Native strategy engine connection failed")
                    
        except Exception as e:
            self.logger.warning(f"Native strategy engine request failed: {e}")
        
        # Fallback to simple strategy models if enabled
        if self.enable_fallback:
            self.logger.info(f"Using fallback model for {strategy_type}")
            return await self._fallback_execute_strategy(strategy_type, market_data, parameters, start_time)
        else:
            processing_time = (time.time() - start_time) * 1000
            self._update_stats(processing_time, native=False, error=True)
            
            return {
                "success": False,
                "error": "Native strategy engine unavailable and fallback disabled",
                "processing_time_ms": processing_time,
                "source": "error"
            }
    
    async def _fallback_execute_strategy(self, strategy_type: str, market_data: Dict[str, Any],
                                       parameters: Dict[str, Any], start_time: float) -> Dict[str, Any]:
        """Simple fallback strategy execution when native engine unavailable"""
        try:
            if strategy_type == "momentum_neural":
                signals = self._momentum_strategy(market_data, parameters)
            elif strategy_type == "mean_reversion_neural":
                signals = self._mean_reversion_strategy(market_data, parameters)
            elif strategy_type == "breakout_detection":
                signals = self._breakout_strategy(market_data, parameters)
            elif strategy_type == "pattern_recognition":
                signals = self._pattern_strategy(market_data, parameters)
            elif strategy_type == "multi_timeframe":
                signals = self._multi_timeframe_strategy(market_data, parameters)
            else:
                signals = self._default_strategy(market_data, parameters)
            
            processing_time = (time.time() - start_time) * 1000
            self._update_stats(processing_time, native=False, signal_count=len(signals))
            
            return {
                "success": True,
                "signals": signals,
                "processing_time_ms": processing_time,
                "hardware_used": "cpu_fallback",
                "patterns_analyzed": 1,
                "source": "fallback_strategy"
            }
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self._update_stats(processing_time, native=False, error=True)
            
            return {
                "success": False,
                "error": f"Fallback strategy execution failed: {str(e)}",
                "processing_time_ms": processing_time,
                "source": "fallback_error"
            }
    
    def _momentum_strategy(self, market_data: Dict[str, Any], parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Simple momentum-based strategy"""
        signals = []
        
        try:
            symbol = market_data.get("symbol", "UNKNOWN")
            current_price = market_data.get("current_price", 100.0)
            rsi = market_data.get("rsi", 50.0)
            macd = market_data.get("macd", 0.0)
            volume_ratio = market_data.get("volume_ratio", 1.0)
            
            # Momentum signals
            if rsi > 60 and macd > 0 and volume_ratio > 1.2:
                signals.append({
                    "signal_id": f"momentum_buy_{int(time.time())}",
                    "symbol": symbol,
                    "signal_type": "BUY",
                    "confidence": 0.75,
                    "price": current_price,
                    "quantity": parameters.get("quantity", 100),
                    "timestamp": time.time(),
                    "pattern_detected": "bullish_momentum",
                    "stop_loss": current_price * 0.98,
                    "take_profit": current_price * 1.05
                })
            elif rsi < 40 and macd < 0 and volume_ratio > 1.2:
                signals.append({
                    "signal_id": f"momentum_sell_{int(time.time())}",
                    "symbol": symbol,
                    "signal_type": "SELL",
                    "confidence": 0.75,
                    "price": current_price,
                    "quantity": parameters.get("quantity", 100),
                    "timestamp": time.time(),
                    "pattern_detected": "bearish_momentum",
                    "stop_loss": current_price * 1.02,
                    "take_profit": current_price * 0.95
                })
            
        except Exception as e:
            self.logger.error(f"Momentum strategy failed: {e}")
        
        return signals
    
    def _mean_reversion_strategy(self, market_data: Dict[str, Any], parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Simple mean reversion strategy"""
        signals = []
        
        try:
            symbol = market_data.get("symbol", "UNKNOWN")
            current_price = market_data.get("current_price", 100.0)
            rsi = market_data.get("rsi", 50.0)
            bb_position = market_data.get("bollinger_position", 0.5)  # 0-1 scale
            
            # Mean reversion signals
            if rsi < 30 and bb_position < 0.2:  # Oversold
                signals.append({
                    "signal_id": f"mean_rev_buy_{int(time.time())}",
                    "symbol": symbol,
                    "signal_type": "BUY",
                    "confidence": 0.7,
                    "price": current_price,
                    "quantity": parameters.get("quantity", 100),
                    "timestamp": time.time(),
                    "pattern_detected": "oversold_bounce",
                    "stop_loss": current_price * 0.97,
                    "take_profit": current_price * 1.03
                })
            elif rsi > 70 and bb_position > 0.8:  # Overbought
                signals.append({
                    "signal_id": f"mean_rev_sell_{int(time.time())}",
                    "symbol": symbol,
                    "signal_type": "SELL",
                    "confidence": 0.7,
                    "price": current_price,
                    "quantity": parameters.get("quantity", 100),
                    "timestamp": time.time(),
                    "pattern_detected": "overbought_reversal",
                    "stop_loss": current_price * 1.03,
                    "take_profit": current_price * 0.97
                })
            
        except Exception as e:
            self.logger.error(f"Mean reversion strategy failed: {e}")
        
        return signals
    
    def _breakout_strategy(self, market_data: Dict[str, Any], parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Simple breakout detection strategy"""
        signals = []
        
        try:
            symbol = market_data.get("symbol", "UNKNOWN")
            current_price = market_data.get("current_price", 100.0)
            high_20d = market_data.get("high_20d", current_price * 1.05)
            low_20d = market_data.get("low_20d", current_price * 0.95)
            volume_ratio = market_data.get("volume_ratio", 1.0)
            atr = market_data.get("atr", current_price * 0.02)
            
            # Breakout signals
            if current_price > high_20d and volume_ratio > 1.5:  # Upside breakout
                signals.append({
                    "signal_id": f"breakout_buy_{int(time.time())}",
                    "symbol": symbol,
                    "signal_type": "BUY",
                    "confidence": 0.8,
                    "price": current_price,
                    "quantity": parameters.get("quantity", 150),  # Larger size for breakouts
                    "timestamp": time.time(),
                    "pattern_detected": "upside_breakout",
                    "stop_loss": current_price - atr * 2,
                    "take_profit": current_price + atr * 4
                })
            elif current_price < low_20d and volume_ratio > 1.5:  # Downside breakout
                signals.append({
                    "signal_id": f"breakout_sell_{int(time.time())}",
                    "symbol": symbol,
                    "signal_type": "SELL",
                    "confidence": 0.8,
                    "price": current_price,
                    "quantity": parameters.get("quantity", 150),
                    "timestamp": time.time(),
                    "pattern_detected": "downside_breakout",
                    "stop_loss": current_price + atr * 2,
                    "take_profit": current_price - atr * 4
                })
            
        except Exception as e:
            self.logger.error(f"Breakout strategy failed: {e}")
        
        return signals
    
    def _pattern_strategy(self, market_data: Dict[str, Any], parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Simple pattern recognition strategy"""
        signals = []
        
        try:
            symbol = market_data.get("symbol", "UNKNOWN")
            current_price = market_data.get("current_price", 100.0)
            
            # Look for simple patterns in price action
            recent_prices = market_data.get("recent_prices", [current_price] * 10)
            
            if len(recent_prices) >= 10:
                # Simple trend detection
                trend_slope = (recent_prices[-1] - recent_prices[-10]) / 10
                price_std = (sum((p - sum(recent_prices)/len(recent_prices))**2 for p in recent_prices) / len(recent_prices)) ** 0.5
                
                # Trend strength
                trend_strength = abs(trend_slope) / max(price_std, current_price * 0.001)
                
                if trend_strength > 0.5:
                    if trend_slope > 0:  # Uptrend
                        signals.append({
                            "signal_id": f"pattern_buy_{int(time.time())}",
                            "symbol": symbol,
                            "signal_type": "BUY",
                            "confidence": min(trend_strength, 0.9),
                            "price": current_price,
                            "quantity": parameters.get("quantity", 100),
                            "timestamp": time.time(),
                            "pattern_detected": "uptrend_pattern",
                            "stop_loss": current_price * 0.96,
                            "take_profit": current_price * 1.08
                        })
                    else:  # Downtrend
                        signals.append({
                            "signal_id": f"pattern_sell_{int(time.time())}",
                            "symbol": symbol,
                            "signal_type": "SELL",
                            "confidence": min(trend_strength, 0.9),
                            "price": current_price,
                            "quantity": parameters.get("quantity", 100),
                            "timestamp": time.time(),
                            "pattern_detected": "downtrend_pattern",
                            "stop_loss": current_price * 1.04,
                            "take_profit": current_price * 0.92
                        })
            
        except Exception as e:
            self.logger.error(f"Pattern strategy failed: {e}")
        
        return signals
    
    def _multi_timeframe_strategy(self, market_data: Dict[str, Any], parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Simple multi-timeframe strategy"""
        signals = []
        
        try:
            symbol = market_data.get("symbol", "UNKNOWN")
            current_price = market_data.get("current_price", 100.0)
            
            # Simulate multi-timeframe analysis
            trend_1h = market_data.get("trend_1h", "neutral")
            trend_4h = market_data.get("trend_4h", "neutral")
            trend_1d = market_data.get("trend_1d", "neutral")
            
            # Count bullish/bearish trends
            bullish_count = sum(1 for trend in [trend_1h, trend_4h, trend_1d] if trend == "bullish")
            bearish_count = sum(1 for trend in [trend_1h, trend_4h, trend_1d] if trend == "bearish")
            
            if bullish_count >= 2:  # Majority bullish
                confidence = 0.6 + (bullish_count - 2) * 0.15  # Higher confidence for more alignment
                signals.append({
                    "signal_id": f"multi_buy_{int(time.time())}",
                    "symbol": symbol,
                    "signal_type": "BUY",
                    "confidence": confidence,
                    "price": current_price,
                    "quantity": parameters.get("quantity", 120),
                    "timestamp": time.time(),
                    "pattern_detected": f"multi_timeframe_bullish_{bullish_count}",
                    "stop_loss": current_price * 0.94,
                    "take_profit": current_price * 1.12
                })
            elif bearish_count >= 2:  # Majority bearish
                confidence = 0.6 + (bearish_count - 2) * 0.15
                signals.append({
                    "signal_id": f"multi_sell_{int(time.time())}",
                    "symbol": symbol,
                    "signal_type": "SELL",
                    "confidence": confidence,
                    "price": current_price,
                    "quantity": parameters.get("quantity", 120),
                    "timestamp": time.time(),
                    "pattern_detected": f"multi_timeframe_bearish_{bearish_count}",
                    "stop_loss": current_price * 1.06,
                    "take_profit": current_price * 0.88
                })
            
        except Exception as e:
            self.logger.error(f"Multi-timeframe strategy failed: {e}")
        
        return signals
    
    def _default_strategy(self, market_data: Dict[str, Any], parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Default simple moving average strategy"""
        signals = []
        
        try:
            symbol = market_data.get("symbol", "UNKNOWN")
            current_price = market_data.get("current_price", 100.0)
            sma_10 = market_data.get("sma_10", current_price)
            sma_20 = market_data.get("sma_20", current_price)
            
            # Simple moving average crossover
            if sma_10 > sma_20 * 1.005:  # Golden cross with buffer
                signals.append({
                    "signal_id": f"default_buy_{int(time.time())}",
                    "symbol": symbol,
                    "signal_type": "BUY",
                    "confidence": 0.6,
                    "price": current_price,
                    "quantity": parameters.get("quantity", 100),
                    "timestamp": time.time(),
                    "pattern_detected": "golden_cross",
                    "stop_loss": current_price * 0.97,
                    "take_profit": current_price * 1.05
                })
            elif sma_10 < sma_20 * 0.995:  # Death cross with buffer
                signals.append({
                    "signal_id": f"default_sell_{int(time.time())}",
                    "symbol": symbol,
                    "signal_type": "SELL",
                    "confidence": 0.6,
                    "price": current_price,
                    "quantity": parameters.get("quantity", 100),
                    "timestamp": time.time(),
                    "pattern_detected": "death_cross",
                    "stop_loss": current_price * 1.03,
                    "take_profit": current_price * 0.95
                })
            
        except Exception as e:
            self.logger.error(f"Default strategy failed: {e}")
        
        return signals
    
    def _update_stats(self, processing_time_ms: float, native: bool, signal_count: int = 0, error: bool = False):
        """Update client statistics"""
        if error:
            self.stats["errors"] += 1
        else:
            self.stats["responses_received"] += 1
            self.stats["signals_generated"] += signal_count
            
        if native:
            self.stats["native_engine_requests"] += 1
        else:
            self.stats["fallback_requests"] += 1
        
        # Update latency statistics
        self.stats["total_latency_ms"] += processing_time_ms
        if self.stats["responses_received"] > 0:
            self.stats["average_latency_ms"] = (
                self.stats["total_latency_ms"] / self.stats["responses_received"]
            )
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of native strategy engine connection"""
        try:
            # Try a simple strategy execution to test connectivity
            result = await self.execute_strategy(
                "momentum_neural",
                {
                    "symbol": "TEST",
                    "current_price": 100.0,
                    "rsi": 55.0,
                    "macd": 0.1
                },
                {"quantity": 100}
            )
            
            native_healthy = result.get("source") == "native_strategy_engine"
            
            return {
                "native_engine_healthy": native_healthy,
                "fallback_available": self.enable_fallback,
                "statistics": self.stats.copy()
            }
            
        except Exception as e:
            return {
                "native_engine_healthy": False,
                "fallback_available": self.enable_fallback,
                "error": str(e),
                "statistics": self.stats.copy()
            }
    
    async def cleanup(self):
        """Clean up native strategy client"""
        if self.connection:
            self.connection.disconnect()
            self.connection = None
        self.logger.info("Native strategy client cleaned up")

# Global client instance
_native_strategy_client = None

async def get_native_strategy_client() -> NativeStrategyClient:
    """Get global native strategy client instance"""
    global _native_strategy_client
    
    if _native_strategy_client is None:
        _native_strategy_client = NativeStrategyClient(
            socket_path="/tmp/nautilus_strategy_engine.sock",
            enable_fallback=True
        )
        await _native_strategy_client.initialize()
    
    return _native_strategy_client

async def cleanup_native_strategy_client():
    """Clean up global native strategy client"""
    global _native_strategy_client
    
    if _native_strategy_client is not None:
        await _native_strategy_client.cleanup()
        _native_strategy_client = None