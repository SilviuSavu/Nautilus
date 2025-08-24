#!/usr/bin/env python3
"""
Native ML Client for Docker Backend
Unix Domain Socket client for communicating with native ML engine

This component provides:
- Unix socket communication with native ML engine
- Connection pooling and retry logic
- Async interface for Docker containers
- Fallback to containerized ML when native unavailable
"""

import asyncio
import json
import logging
import socket
import struct
import time
import uuid
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple
import contextlib

@dataclass
class MLRequest:
    """ML inference request structure"""
    request_id: str
    model_type: str
    input_data: Dict[str, Any]
    options: Dict[str, Any]
    timestamp: float

@dataclass
class MLResponse:
    """ML inference response structure"""
    request_id: str
    predictions: Dict[str, Any]
    confidence: float
    processing_time_ms: float
    hardware_used: str
    timestamp: float
    error: Optional[str] = None

class ConnectionState(Enum):
    """Connection state enumeration"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"

class NativeMLConnection:
    """Single connection to native ML engine"""
    
    def __init__(self, socket_path: str, timeout: float = 30.0):
        self.socket_path = socket_path
        self.timeout = timeout
        self.sock = None
        self.state = ConnectionState.DISCONNECTED
        self.last_error = None
        self.logger = logging.getLogger(__name__)
        
    async def connect(self) -> bool:
        """Connect to native ML engine"""
        try:
            self.state = ConnectionState.CONNECTING
            
            # Create Unix socket
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.sock.settimeout(self.timeout)
            
            # Connect to native ML engine
            await asyncio.get_event_loop().run_in_executor(
                None, self.sock.connect, self.socket_path
            )
            
            self.state = ConnectionState.CONNECTED
            self.last_error = None
            self.logger.info(f"Connected to native ML engine at {self.socket_path}")
            return True
            
        except Exception as e:
            self.state = ConnectionState.ERROR
            self.last_error = str(e)
            self.logger.error(f"Failed to connect to native ML engine: {e}")
            if self.sock:
                self.sock.close()
                self.sock = None
            return False
    
    async def send_request(self, request: MLRequest) -> Optional[MLResponse]:
        """Send ML request and receive response"""
        if self.state != ConnectionState.CONNECTED or not self.sock:
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
                raise ConnectionError("Connection closed by native ML engine")
            
            response_length = struct.unpack('!I', length_data)[0]
            
            # Receive response data
            response_data = b''
            while len(response_data) < response_length:
                chunk = await asyncio.get_event_loop().run_in_executor(
                    None, self.sock.recv, response_length - len(response_data)
                )
                if not chunk:
                    raise ConnectionError("Incomplete response from native ML engine")
                response_data += chunk
            
            # Parse response
            response_json = json.loads(response_data.decode('utf-8'))
            
            return MLResponse(**response_json)
            
        except Exception as e:
            self.logger.error(f"Request failed: {e}")
            self.state = ConnectionState.ERROR
            self.last_error = str(e)
            return None
    
    def disconnect(self):
        """Disconnect from native ML engine"""
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
            finally:
                self.sock = None
        
        self.state = ConnectionState.DISCONNECTED
        self.logger.info("Disconnected from native ML engine")
    
    def is_connected(self) -> bool:
        """Check if connection is active"""
        return self.state == ConnectionState.CONNECTED and self.sock is not None

class NativeMLConnectionPool:
    """Connection pool for native ML engine"""
    
    def __init__(self, socket_path: str, max_connections: int = 10, timeout: float = 30.0):
        self.socket_path = socket_path
        self.max_connections = max_connections
        self.timeout = timeout
        self.connections: List[NativeMLConnection] = []
        self.active_connections: List[NativeMLConnection] = []
        self.lock = asyncio.Lock()
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self):
        """Initialize connection pool"""
        self.logger.info(f"Initializing native ML connection pool (max: {self.max_connections})")
        
        # Create initial connections
        initial_connections = min(3, self.max_connections)  # Start with 3 connections
        
        for i in range(initial_connections):
            conn = NativeMLConnection(self.socket_path, self.timeout)
            if await conn.connect():
                self.connections.append(conn)
            else:
                self.logger.warning(f"Failed to create initial connection {i+1}")
        
        self.logger.info(f"Created {len(self.connections)} initial connections")
    
    @contextlib.asynccontextmanager
    async def get_connection(self):
        """Get connection from pool (context manager)"""
        connection = None
        
        try:
            async with self.lock:
                # Try to get available connection
                for conn in self.connections:
                    if conn not in self.active_connections and conn.is_connected():
                        connection = conn
                        self.active_connections.append(conn)
                        break
                
                # Create new connection if needed and under limit
                if not connection and len(self.connections) < self.max_connections:
                    new_conn = NativeMLConnection(self.socket_path, self.timeout)
                    if await new_conn.connect():
                        self.connections.append(new_conn)
                        self.active_connections.append(new_conn)
                        connection = new_conn
                
                # Wait for available connection if at limit
                if not connection:
                    self.logger.warning("No connections available, waiting...")
                    # For now, just try the first connection
                    if self.connections:
                        connection = self.connections[0]
                        if not connection.is_connected():
                            await connection.connect()
                        if connection not in self.active_connections:
                            self.active_connections.append(connection)
            
            if connection:
                yield connection
            else:
                raise RuntimeError("Unable to get ML engine connection")
                
        finally:
            if connection:
                async with self.lock:
                    if connection in self.active_connections:
                        self.active_connections.remove(connection)
    
    async def cleanup(self):
        """Clean up all connections"""
        async with self.lock:
            for conn in self.connections:
                conn.disconnect()
            self.connections.clear()
            self.active_connections.clear()
        
        self.logger.info("Connection pool cleaned up")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        connected_count = sum(1 for conn in self.connections if conn.is_connected())
        
        return {
            "total_connections": len(self.connections),
            "connected_connections": connected_count,
            "active_connections": len(self.active_connections),
            "max_connections": self.max_connections,
            "socket_path": self.socket_path
        }

class NativeMLClient:
    """High-level client for native ML engine communication"""
    
    def __init__(self, socket_path: str = "/tmp/nautilus_ml_engine.sock", 
                 max_connections: int = 10, enable_fallback: bool = True):
        self.socket_path = socket_path
        self.max_connections = max_connections
        self.enable_fallback = enable_fallback
        
        self.pool = NativeMLConnectionPool(socket_path, max_connections)
        self.logger = logging.getLogger(__name__)
        
        # Statistics
        self.stats = {
            "requests_sent": 0,
            "responses_received": 0,
            "errors": 0,
            "native_engine_requests": 0,
            "fallback_requests": 0,
            "average_latency_ms": 0.0,
            "total_latency_ms": 0.0
        }
        
        # Fallback ML service (simple implementations)
        self.fallback_models = self._initialize_fallback_models()
        
    async def initialize(self):
        """Initialize native ML client"""
        try:
            await self.pool.initialize()
            self.logger.info("Native ML client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize native ML client: {e}")
            if not self.enable_fallback:
                raise
    
    def _initialize_fallback_models(self) -> Dict[str, Any]:
        """Initialize simple fallback models when native engine unavailable"""
        return {
            "price_predictor": {
                "type": "simple_momentum",
                "description": "Simple momentum-based price predictor"
            },
            "regime_detector": {
                "type": "volatility_based",
                "description": "Volatility-based market regime detector"
            },
            "risk_classifier": {
                "type": "var_based",
                "description": "VaR-based risk classifier"
            }
        }
    
    async def predict(self, model_type: str, input_data: Dict[str, Any], 
                     options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make ML prediction using native engine or fallback"""
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        if options is None:
            options = {}
        
        self.stats["requests_sent"] += 1
        
        # Try native ML engine first
        try:
            async with self.pool.get_connection() as connection:
                request = MLRequest(
                    request_id=request_id,
                    model_type=model_type,
                    input_data=input_data,
                    options=options,
                    timestamp=start_time
                )
                
                response = await connection.send_request(request)
                
                if response and not response.error:
                    # Success with native engine
                    processing_time = (time.time() - start_time) * 1000
                    self._update_stats(processing_time, native=True)
                    
                    return {
                        "success": True,
                        "predictions": response.predictions,
                        "confidence": response.confidence,
                        "processing_time_ms": response.processing_time_ms,
                        "total_time_ms": processing_time,
                        "hardware_used": response.hardware_used,
                        "source": "native_ml_engine"
                    }
                elif response and response.error:
                    self.logger.warning(f"Native ML engine returned error: {response.error}")
                else:
                    self.logger.warning("Native ML engine connection failed")
                    
        except Exception as e:
            self.logger.warning(f"Native ML engine request failed: {e}")
        
        # Fallback to simple models if enabled
        if self.enable_fallback:
            self.logger.info(f"Using fallback model for {model_type}")
            return await self._fallback_predict(model_type, input_data, start_time)
        else:
            processing_time = (time.time() - start_time) * 1000
            self._update_stats(processing_time, native=False, error=True)
            
            return {
                "success": False,
                "error": "Native ML engine unavailable and fallback disabled",
                "processing_time_ms": processing_time,
                "source": "error"
            }
    
    async def _fallback_predict(self, model_type: str, input_data: Dict[str, Any], 
                               start_time: float) -> Dict[str, Any]:
        """Simple fallback predictions when native engine unavailable"""
        try:
            if model_type == "price_predictor":
                predictions = self._simple_price_prediction(input_data)
            elif model_type == "regime_detector":
                predictions = self._simple_regime_detection(input_data)
            elif model_type == "risk_classifier":
                predictions = self._simple_risk_classification(input_data)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            processing_time = (time.time() - start_time) * 1000
            self._update_stats(processing_time, native=False)
            
            return {
                "success": True,
                "predictions": predictions,
                "confidence": 0.6,  # Lower confidence for fallback
                "processing_time_ms": processing_time,
                "hardware_used": "cpu_fallback",
                "source": "fallback_model"
            }
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self._update_stats(processing_time, native=False, error=True)
            
            return {
                "success": False,
                "error": f"Fallback model failed: {str(e)}",
                "processing_time_ms": processing_time,
                "source": "fallback_error"
            }
    
    def _simple_price_prediction(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simple momentum-based price prediction"""
        # Extract key indicators
        rsi = input_data.get("rsi", 50.0)
        macd = input_data.get("macd", 0.0)
        price_change_1d = input_data.get("price_change_1d", 0.0)
        volume_ratio = input_data.get("volume_ratio", 1.0)
        
        # Simple momentum calculation
        momentum_score = (rsi - 50) / 50 + macd + price_change_1d
        
        # Volume adjustment
        volume_factor = min(volume_ratio, 3.0) / 3.0  # Cap at 3x normal volume
        momentum_score *= (0.7 + 0.3 * volume_factor)
        
        # Calculate price change prediction
        price_change = max(min(momentum_score * 0.02, 0.05), -0.05)  # Cap at ±5%
        volatility = abs(momentum_score * 0.01) + 0.005  # Base volatility
        
        return {
            "price_change": price_change,
            "confidence": abs(momentum_score),
            "volatility": volatility
        }
    
    def _simple_regime_detection(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simple volatility-based regime detection"""
        vix = input_data.get("vix", 20.0)
        market_return_1m = input_data.get("market_return_1m", 0.02)
        market_volatility = input_data.get("market_volatility", 0.15)
        
        # Simple regime classification based on VIX and returns
        if vix > 30 or market_volatility > 0.25:
            regime = "volatile"
            prob_volatile = 0.8
            prob_bull = prob_bear = prob_consolidation = 0.067
        elif market_return_1m > 0.03:
            regime = "bull"
            prob_bull = 0.7
            prob_volatile = prob_bear = prob_consolidation = 0.1
        elif market_return_1m < -0.02:
            regime = "bear"
            prob_bear = 0.7
            prob_bull = prob_volatile = prob_consolidation = 0.1
        else:
            regime = "consolidation"
            prob_consolidation = 0.6
            prob_bull = prob_bear = prob_volatile = 0.133
        
        return {
            "predicted_regime": regime,
            "regime_probabilities": {
                "bull": prob_bull,
                "bear": prob_bear,
                "consolidation": prob_consolidation,
                "volatile": prob_volatile
            }
        }
    
    def _simple_risk_classification(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simple VaR-based risk classification"""
        var_1d = input_data.get("var_1d", 0.02)
        max_drawdown = input_data.get("max_drawdown", 0.05)
        leverage = input_data.get("leverage", 1.0)
        concentration_risk = input_data.get("concentration_risk", 0.1)
        
        # Calculate risk score
        risk_score = (
            var_1d * 2.0 +  # Daily VaR weight
            max_drawdown * 1.5 +  # Drawdown weight
            (leverage - 1.0) * 0.1 +  # Leverage weight
            concentration_risk * 0.5  # Concentration weight
        )
        
        # Classify risk level
        if risk_score < 0.02:
            risk_level = "very_low"
            probs = [0.7, 0.2, 0.05, 0.03, 0.02]
        elif risk_score < 0.05:
            risk_level = "low"
            probs = [0.2, 0.6, 0.15, 0.03, 0.02]
        elif risk_score < 0.1:
            risk_level = "medium"
            probs = [0.1, 0.2, 0.5, 0.15, 0.05]
        elif risk_score < 0.2:
            risk_level = "high"
            probs = [0.05, 0.1, 0.2, 0.5, 0.15]
        else:
            risk_level = "critical"
            probs = [0.02, 0.03, 0.1, 0.25, 0.6]
        
        risk_levels = ["very_low", "low", "medium", "high", "critical"]
        risk_probabilities = {level: prob for level, prob in zip(risk_levels, probs)}
        
        return {
            "risk_level": risk_level,
            "risk_probabilities": risk_probabilities
        }
    
    def _update_stats(self, processing_time_ms: float, native: bool, error: bool = False):
        """Update client statistics"""
        if error:
            self.stats["errors"] += 1
        else:
            self.stats["responses_received"] += 1
            
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
        """Check health of native ML engine connection"""
        try:
            # Try a simple prediction to test connectivity
            result = await self.predict("price_predictor", {"rsi": 50.0}, {"timeout": 5.0})
            
            native_healthy = result.get("source") == "native_ml_engine"
            
            pool_stats = self.pool.get_stats()
            
            return {
                "native_engine_healthy": native_healthy,
                "fallback_available": self.enable_fallback,
                "connection_pool": pool_stats,
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
        """Clean up native ML client"""
        await self.pool.cleanup()
        self.logger.info("Native ML client cleaned up")

# Global client instance
_native_ml_client = None

async def get_native_ml_client() -> NativeMLClient:
    """Get global native ML client instance"""
    global _native_ml_client
    
    if _native_ml_client is None:
        _native_ml_client = NativeMLClient(
            socket_path="/tmp/nautilus_ml_engine.sock",
            max_connections=10,
            enable_fallback=True
        )
        await _native_ml_client.initialize()
    
    return _native_ml_client

async def cleanup_native_ml_client():
    """Clean up global native ML client"""
    global _native_ml_client
    
    if _native_ml_client is not None:
        await _native_ml_client.cleanup()
        _native_ml_client = None