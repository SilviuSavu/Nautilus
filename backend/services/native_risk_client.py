#!/usr/bin/env python3
"""
Native Risk Client for Docker Backend
Unix Domain Socket client for communicating with native risk engine

This component provides:
- Unix socket communication with native risk engine
- Connection pooling and retry logic  
- Async interface for risk calculations
- Fallback to containerized risk when native unavailable
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
class RiskRequest:
    """Risk calculation request structure"""
    request_id: str
    calculation_type: str
    portfolio_data: Dict[str, Any]
    parameters: Dict[str, Any]
    timestamp: float

@dataclass
class RiskResponse:
    """Risk calculation response structure"""
    request_id: str
    risk_metrics: Dict[str, Any]
    calculation_time_ms: float
    hardware_used: str
    simulations_count: int
    timestamp: float
    error: Optional[str] = None

class NativeRiskConnection:
    """Single connection to native risk engine"""
    
    def __init__(self, socket_path: str, timeout: float = 60.0):
        self.socket_path = socket_path
        self.timeout = timeout
        self.sock = None
        self.logger = logging.getLogger(__name__)
        
    async def connect(self) -> bool:
        """Connect to native risk engine"""
        try:
            # Create Unix socket
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.sock.settimeout(self.timeout)
            
            # Connect to native risk engine
            await asyncio.get_event_loop().run_in_executor(
                None, self.sock.connect, self.socket_path
            )
            
            self.logger.info(f"Connected to native risk engine at {self.socket_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to native risk engine: {e}")
            if self.sock:
                self.sock.close()
                self.sock = None
            return False
    
    async def send_request(self, request: RiskRequest) -> Optional[RiskResponse]:
        """Send risk calculation request and receive response"""
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
                raise ConnectionError("Connection closed by native risk engine")
            
            response_length = struct.unpack('!I', length_data)[0]
            
            # Receive response data
            response_data = b''
            while len(response_data) < response_length:
                chunk = await asyncio.get_event_loop().run_in_executor(
                    None, self.sock.recv, response_length - len(response_data)
                )
                if not chunk:
                    raise ConnectionError("Incomplete response from native risk engine")
                response_data += chunk
            
            # Parse response
            response_json = json.loads(response_data.decode('utf-8'))
            
            return RiskResponse(**response_json)
            
        except Exception as e:
            self.logger.error(f"Risk request failed: {e}")
            return None
    
    def disconnect(self):
        """Disconnect from native risk engine"""
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
            finally:
                self.sock = None
        
        self.logger.info("Disconnected from native risk engine")
    
    def is_connected(self) -> bool:
        """Check if connection is active"""
        return self.sock is not None

class NativeRiskClient:
    """High-level client for native risk engine communication"""
    
    def __init__(self, socket_path: str = "/tmp/nautilus_risk_engine.sock", 
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
            "total_latency_ms": 0.0
        }
        
    async def initialize(self):
        """Initialize native risk client"""
        try:
            self.connection = NativeRiskConnection(self.socket_path)
            connected = await self.connection.connect()
            
            if connected:
                self.logger.info("Native risk client initialized successfully")
            else:
                self.logger.warning("Failed to initialize native risk client")
                if not self.enable_fallback:
                    raise RuntimeError("Native risk engine connection failed and fallback disabled")
        except Exception as e:
            self.logger.error(f"Failed to initialize native risk client: {e}")
            if not self.enable_fallback:
                raise
    
    async def calculate_risk(self, calculation_type: str, portfolio_data: Dict[str, Any], 
                           parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate risk using native engine or fallback"""
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        self.stats["requests_sent"] += 1
        
        # Try native risk engine first
        try:
            if self.connection and self.connection.is_connected():
                request = RiskRequest(
                    request_id=request_id,
                    calculation_type=calculation_type,
                    portfolio_data=portfolio_data,
                    parameters=parameters,
                    timestamp=start_time
                )
                
                response = await self.connection.send_request(request)
                
                if response and not response.error:
                    # Success with native engine
                    processing_time = (time.time() - start_time) * 1000
                    self._update_stats(processing_time, native=True)
                    
                    return {
                        "success": True,
                        "risk_metrics": response.risk_metrics,
                        "calculation_time_ms": response.calculation_time_ms,
                        "total_time_ms": processing_time,
                        "hardware_used": response.hardware_used,
                        "simulations_count": response.simulations_count,
                        "source": "native_risk_engine"
                    }
                elif response and response.error:
                    self.logger.warning(f"Native risk engine returned error: {response.error}")
                else:
                    self.logger.warning("Native risk engine connection failed")
                    
        except Exception as e:
            self.logger.warning(f"Native risk engine request failed: {e}")
        
        # Fallback to simple risk models if enabled
        if self.enable_fallback:
            self.logger.info(f"Using fallback model for {calculation_type}")
            return await self._fallback_calculate_risk(calculation_type, portfolio_data, parameters, start_time)
        else:
            processing_time = (time.time() - start_time) * 1000
            self._update_stats(processing_time, native=False, error=True)
            
            return {
                "success": False,
                "error": "Native risk engine unavailable and fallback disabled",
                "processing_time_ms": processing_time,
                "source": "error"
            }
    
    async def _fallback_calculate_risk(self, calculation_type: str, portfolio_data: Dict[str, Any],
                                     parameters: Dict[str, Any], start_time: float) -> Dict[str, Any]:
        """Simple fallback risk calculations when native engine unavailable"""
        try:
            if calculation_type == "monte_carlo_var":
                risk_metrics = self._simple_var_calculation(portfolio_data, parameters)
            elif calculation_type == "portfolio_optimization":
                risk_metrics = self._simple_portfolio_optimization(portfolio_data, parameters)
            elif calculation_type == "stress_testing":
                risk_metrics = self._simple_stress_testing(portfolio_data, parameters)
            elif calculation_type == "option_pricing":
                risk_metrics = self._simple_option_pricing(portfolio_data, parameters)
            else:
                raise ValueError(f"Unknown calculation type: {calculation_type}")
            
            processing_time = (time.time() - start_time) * 1000
            self._update_stats(processing_time, native=False)
            
            return {
                "success": True,
                "risk_metrics": risk_metrics,
                "calculation_time_ms": processing_time,
                "hardware_used": "cpu_fallback",
                "simulations_count": parameters.get("num_simulations", 10000),
                "source": "fallback_model"
            }
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self._update_stats(processing_time, native=False, error=True)
            
            return {
                "success": False,
                "error": f"Fallback risk calculation failed: {str(e)}",
                "processing_time_ms": processing_time,
                "source": "fallback_error"
            }
    
    def _simple_var_calculation(self, portfolio_data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Simple VaR calculation using parametric approach"""
        try:
            positions = portfolio_data.get("positions", [])
            confidence_level = parameters.get("confidence_level", 0.95)
            
            if not positions:
                # Default risk metrics
                return {
                    "var_1d": 0.02,
                    "expected_shortfall": 0.025,
                    "confidence_level": confidence_level,
                    "portfolio_return": 0.08,
                    "portfolio_volatility": 0.15
                }
            
            # Simple parametric VaR
            total_value = sum(pos.get("market_value", 0.0) for pos in positions)
            weighted_volatility = sum(
                pos.get("weight", 0.0) * pos.get("volatility", 0.2) 
                for pos in positions
            )
            
            # Z-score for confidence level
            if confidence_level == 0.95:
                z_score = 1.645
            elif confidence_level == 0.99:
                z_score = 2.326
            else:
                z_score = 1.96  # Default to 97.5%
            
            var_1d = weighted_volatility * z_score
            expected_shortfall = var_1d * 1.25  # Simple ES approximation
            
            return {
                "var_1d": var_1d,
                "expected_shortfall": expected_shortfall,
                "confidence_level": confidence_level,
                "portfolio_return": sum(pos.get("expected_return", 0.08) * pos.get("weight", 0.0) for pos in positions),
                "portfolio_volatility": weighted_volatility,
                "worst_case_loss": var_1d * 2.0,
                "percentile_losses": {
                    "p1": var_1d * 2.33,
                    "p5": var_1d * 1.64,
                    "p10": var_1d * 1.28,
                    "p50": 0.0,
                    "p95": -var_1d * 0.5
                }
            }
            
        except Exception as e:
            self.logger.error(f"Simple VaR calculation failed: {e}")
            return {"error": str(e)}
    
    def _simple_portfolio_optimization(self, portfolio_data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Simple portfolio optimization using equal weights"""
        try:
            positions = portfolio_data.get("positions", [])
            risk_aversion = parameters.get("risk_aversion", 1.0)
            
            if not positions:
                return {"optimal_weights": [], "expected_return": 0.0, "expected_volatility": 0.0}
            
            # Equal weight allocation (simple)
            n_assets = len(positions)
            equal_weight = 1.0 / n_assets
            optimal_weights = [equal_weight] * n_assets
            
            # Calculate portfolio metrics
            expected_return = sum(
                pos.get("expected_return", 0.08) * equal_weight 
                for pos in positions
            )
            
            # Simple volatility (ignoring correlations)
            expected_volatility = (
                sum(pos.get("volatility", 0.2) ** 2 * equal_weight ** 2 for pos in positions) ** 0.5
            )
            
            sharpe_ratio = expected_return / expected_volatility if expected_volatility > 0 else 0
            
            return {
                "optimal_weights": optimal_weights,
                "expected_return": expected_return,
                "expected_volatility": expected_volatility,
                "sharpe_ratio": sharpe_ratio,
                "risk_aversion": risk_aversion
            }
            
        except Exception as e:
            self.logger.error(f"Simple portfolio optimization failed: {e}")
            return {"error": str(e)}
    
    def _simple_stress_testing(self, portfolio_data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Simple stress testing scenarios"""
        try:
            positions = portfolio_data.get("positions", [])
            
            # Simple stress scenarios
            scenarios = [
                {"name": "Market Crash -30%", "market_shock": -0.30},
                {"name": "Volatility Spike +50%", "volatility_shock": 1.50},
                {"name": "Interest Rate +200bp", "rate_shock": 0.02}
            ]
            
            scenario_results = []
            
            for scenario in scenarios:
                if "market_shock" in scenario:
                    # Apply market shock
                    portfolio_loss = sum(
                        pos.get("market_value", 0.0) * scenario["market_shock"] * pos.get("beta", 1.0)
                        for pos in positions
                    )
                else:
                    # Conservative estimate
                    portfolio_loss = sum(pos.get("market_value", 0.0) for pos in positions) * -0.10
                
                scenario_results.append({
                    "scenario_name": scenario["name"],
                    "portfolio_loss": abs(portfolio_loss),
                    "loss_percentage": abs(portfolio_loss) / max(1.0, sum(pos.get("market_value", 0.0) for pos in positions)) * 100
                })
            
            worst_case = max(scenario_results, key=lambda x: x["portfolio_loss"]) if scenario_results else {}
            
            return {
                "scenario_results": scenario_results,
                "worst_case_scenario": worst_case
            }
            
        except Exception as e:
            self.logger.error(f"Simple stress testing failed: {e}")
            return {"error": str(e)}
    
    def _simple_option_pricing(self, portfolio_data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Simple Black-Scholes option pricing"""
        try:
            import math
            
            spot_price = parameters.get("spot_price", 100.0)
            strike_price = parameters.get("strike_price", 100.0)
            risk_free_rate = parameters.get("risk_free_rate", 0.05)
            volatility = parameters.get("volatility", 0.2)
            time_to_expiry = parameters.get("time_to_expiry", 1.0)
            option_type = parameters.get("option_type", "call")
            
            # Black-Scholes formula
            d1 = (math.log(spot_price / strike_price) + (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / (volatility * math.sqrt(time_to_expiry))
            d2 = d1 - volatility * math.sqrt(time_to_expiry)
            
            # Normal CDF approximation
            def norm_cdf(x):
                return 0.5 * (1 + math.erf(x / math.sqrt(2)))
            
            if option_type.lower() == "call":
                option_price = (spot_price * norm_cdf(d1) - 
                              strike_price * math.exp(-risk_free_rate * time_to_expiry) * norm_cdf(d2))
            else:  # put
                option_price = (strike_price * math.exp(-risk_free_rate * time_to_expiry) * norm_cdf(-d2) - 
                              spot_price * norm_cdf(-d1))
            
            return {
                "option_price": max(0.0, option_price),
                "spot_price": spot_price,
                "strike_price": strike_price,
                "volatility": volatility,
                "time_to_expiry": time_to_expiry,
                "option_type": option_type
            }
            
        except Exception as e:
            self.logger.error(f"Simple option pricing failed: {e}")
            return {"error": str(e)}
    
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
        """Check health of native risk engine connection"""
        try:
            # Try a simple calculation to test connectivity
            result = await self.calculate_risk(
                "monte_carlo_var",
                {"positions": [{"weight": 1.0, "volatility": 0.1}]},
                {"confidence_level": 0.95, "num_simulations": 1000}
            )
            
            native_healthy = result.get("source") == "native_risk_engine"
            
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
        """Clean up native risk client"""
        if self.connection:
            self.connection.disconnect()
            self.connection = None
        self.logger.info("Native risk client cleaned up")

# Global client instance
_native_risk_client = None

async def get_native_risk_client() -> NativeRiskClient:
    """Get global native risk client instance"""
    global _native_risk_client
    
    if _native_risk_client is None:
        _native_risk_client = NativeRiskClient(
            socket_path="/tmp/nautilus_risk_engine.sock",
            enable_fallback=True
        )
        await _native_risk_client.initialize()
    
    return _native_risk_client

async def cleanup_native_risk_client():
    """Clean up global native risk client"""
    global _native_risk_client
    
    if _native_risk_client is not None:
        await _native_risk_client.cleanup()
        _native_risk_client = None