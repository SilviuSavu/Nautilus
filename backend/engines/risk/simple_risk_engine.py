#!/usr/bin/env python3
"""
Simple Risk Engine - Containerized Risk Management Service
Demonstrates containerization approach with basic risk checks
"""

import asyncio
import logging
import os
import time
from datetime import datetime
from typing import Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import json
import numpy as np
from fastapi import FastAPI, HTTPException
import uvicorn

# Basic MessageBus integration
from enhanced_messagebus_client import BufferedMessageBusClient, MessageBusConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskLimitType(Enum):
    POSITION_SIZE = "position_size"
    PORTFOLIO_VALUE = "portfolio_value"
    DAILY_LOSS = "daily_loss"
    CONCENTRATION = "concentration"
    LEVERAGE = "leverage"

class BreachSeverity(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

@dataclass
class RiskLimit:
    limit_id: str
    limit_type: RiskLimitType
    limit_value: float
    current_value: float
    enabled: bool = True

@dataclass
class RiskBreach:
    breach_id: str
    limit_id: str
    breach_time: datetime
    severity: BreachSeverity
    breach_value: float
    limit_value: float

class SimpleRiskEngine:
    """
    Simple Risk Engine demonstrating containerization approach
    """
    
    def __init__(self):
        self.app = FastAPI(title="Nautilus Simple Risk Engine", version="1.0.0")
        self.is_running = False
        self.risk_checks_processed = 0
        self.breaches_detected = 0
        self.start_time = time.time()
        
        # Risk state
        self.active_limits: Dict[str, RiskLimit] = {}
        self.active_breaches: Dict[str, RiskBreach] = {}
        
        # MessageBus configuration
        self.messagebus_config = MessageBusConfig(
            redis_host=os.getenv("REDIS_HOST", "redis"),
            redis_port=int(os.getenv("REDIS_PORT", "6379")),
            redis_db=0
        )
        
        self.messagebus = None
        self.setup_routes()
        
    def setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy" if self.is_running else "stopped",
                "risk_checks_processed": self.risk_checks_processed,
                "breaches_detected": self.breaches_detected,
                "active_limits": len(self.active_limits),
                "active_breaches": len(self.active_breaches),
                "uptime_seconds": time.time() - self.start_time,
                "messagebus_connected": self.messagebus is not None and self.messagebus.is_connected
            }
        
        @self.app.get("/metrics")
        async def get_metrics():
            return {
                "risk_checks_per_second": self.risk_checks_processed / max(1, time.time() - self.start_time),
                "total_risk_checks": self.risk_checks_processed,
                "total_breaches": self.breaches_detected,
                "active_limits_count": len(self.active_limits),
                "active_breaches_count": len(self.active_breaches),
                "breach_rate": self.breaches_detected / max(1, self.risk_checks_processed),
                "uptime": time.time() - self.start_time,
                "engine_type": "risk",
                "containerized": True
            }
        
        @self.app.post("/risk/check/{portfolio_id}")
        async def perform_risk_check(portfolio_id: str, position_data: Dict[str, Any]):
            """Perform comprehensive risk check"""
            try:
                # Simulate risk check processing
                result = await self._perform_risk_check(portfolio_id, position_data)
                self.risk_checks_processed += 1
                
                # Check for breaches
                breaches = await self._check_for_breaches(result)
                
                if breaches:
                    self.breaches_detected += len(breaches)
                
                return {
                    "status": "completed",
                    "portfolio_id": portfolio_id,
                    "risk_result": result,
                    "breaches": breaches
                }
                
            except Exception as e:
                logger.error(f"Risk check error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/risk/limits")
        async def create_risk_limit(limit_data: Dict[str, Any]):
            """Create a new risk limit"""
            try:
                limit = RiskLimit(
                    limit_id=limit_data.get("limit_id", f"limit_{int(time.time())}"),
                    limit_type=RiskLimitType(limit_data.get("limit_type", "position_size")),
                    limit_value=float(limit_data.get("limit_value", 100000)),
                    current_value=float(limit_data.get("current_value", 0)),
                    enabled=limit_data.get("enabled", True)
                )
                
                self.active_limits[limit.limit_id] = limit
                
                return {
                    "status": "created",
                    "limit_id": limit.limit_id,
                    "limit_type": limit.limit_type.value,
                    "limit_value": limit.limit_value
                }
                
            except Exception as e:
                logger.error(f"Risk limit creation error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/risk/limits")
        async def get_risk_limits():
            """Get all active risk limits"""
            return {
                "limits": [
                    {
                        "limit_id": limit.limit_id,
                        "limit_type": limit.limit_type.value,
                        "limit_value": limit.limit_value,
                        "current_value": limit.current_value,
                        "enabled": limit.enabled
                    }
                    for limit in self.active_limits.values()
                ],
                "count": len(self.active_limits)
            }
        
        @self.app.get("/risk/breaches")
        async def get_active_breaches():
            """Get all active risk breaches"""
            return {
                "breaches": [
                    {
                        "breach_id": breach.breach_id,
                        "limit_id": breach.limit_id,
                        "breach_time": breach.breach_time.isoformat(),
                        "severity": breach.severity.value,
                        "breach_value": breach.breach_value,
                        "limit_value": breach.limit_value
                    }
                    for breach in self.active_breaches.values()
                ],
                "count": len(self.active_breaches)
            }

    async def start_engine(self):
        """Start the risk engine"""
        try:
            logger.info("Starting Simple Risk Engine...")
            
            # Try to initialize MessageBus
            try:
                self.messagebus = BufferedMessageBusClient(self.messagebus_config)
                await self.messagebus.start()
                logger.info("MessageBus connected successfully")
            except Exception as e:
                logger.warning(f"MessageBus connection failed: {e}. Running without MessageBus.")
                self.messagebus = None
            
            # Initialize some default limits for demonstration
            await self._initialize_default_limits()
            
            self.is_running = True
            logger.info("Simple Risk Engine started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start Risk Engine: {e}")
            raise
    
    async def stop_engine(self):
        """Stop the risk engine"""
        logger.info("Stopping Simple Risk Engine...")
        self.is_running = False
        
        if self.messagebus:
            await self.messagebus.stop()
        
        logger.info("Simple Risk Engine stopped")
    
    async def _perform_risk_check(self, portfolio_id: str, position_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform risk check"""
        # Simulate risk calculation
        await asyncio.sleep(0.0005)  # 0.5ms processing time
        
        # Calculate risk metrics
        positions = position_data.get("positions", [])
        total_exposure = sum(pos.get("market_value", 0) for pos in positions)
        portfolio_value = position_data.get("portfolio_value", 100000)
        
        # Calculate leverage
        leverage = total_exposure / portfolio_value if portfolio_value > 0 else 0
        
        # Calculate concentration
        max_position = max(pos.get("market_value", 0) for pos in positions) if positions else 0
        concentration_ratio = max_position / total_exposure if total_exposure > 0 else 0
        
        # Mock VaR calculation
        var_95 = -abs(np.random.normal(-5000, 2000))
        
        return {
            "portfolio_id": portfolio_id,
            "total_exposure": total_exposure,
            "portfolio_value": portfolio_value,
            "leverage": leverage,
            "concentration_ratio": concentration_ratio,
            "var_95": var_95,
            "position_count": len(positions),
            "check_timestamp": time.time(),
            "processing_time_ms": 0.5
        }
    
    async def _check_for_breaches(self, risk_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for risk breaches"""
        breaches = []
        
        # Simple breach detection
        leverage = risk_result.get("leverage", 0)
        concentration = risk_result.get("concentration_ratio", 0)
        
        # Check leverage breach
        if leverage > 3.0:  # 3x leverage limit
            breach = {
                "breach_id": f"breach_{int(time.time())}",
                "breach_type": "leverage_exceeded",
                "current_value": leverage,
                "limit_value": 3.0,
                "severity": "HIGH" if leverage > 5.0 else "MEDIUM"
            }
            breaches.append(breach)
        
        # Check concentration breach
        if concentration > 0.25:  # 25% concentration limit
            breach = {
                "breach_id": f"breach_{int(time.time())}_conc",
                "breach_type": "concentration_exceeded", 
                "current_value": concentration,
                "limit_value": 0.25,
                "severity": "MEDIUM" if concentration < 0.5 else "HIGH"
            }
            breaches.append(breach)
        
        return breaches
    
    async def _initialize_default_limits(self):
        """Initialize some default risk limits"""
        default_limits = [
            {
                "limit_id": "leverage_limit",
                "limit_type": "leverage",
                "limit_value": 3.0,
                "current_value": 0.0
            },
            {
                "limit_id": "concentration_limit", 
                "limit_type": "concentration",
                "limit_value": 0.25,
                "current_value": 0.0
            },
            {
                "limit_id": "portfolio_value_limit",
                "limit_type": "portfolio_value",
                "limit_value": 1000000,
                "current_value": 0.0
            }
        ]
        
        for limit_data in default_limits:
            limit = RiskLimit(
                limit_id=limit_data["limit_id"],
                limit_type=RiskLimitType(limit_data["limit_type"]),
                limit_value=limit_data["limit_value"],
                current_value=limit_data["current_value"]
            )
            self.active_limits[limit.limit_id] = limit
        
        logger.info(f"Initialized {len(default_limits)} default risk limits")

# Create and start the engine
simple_risk_engine = SimpleRiskEngine()

# Check for hybrid mode
ENABLE_HYBRID = os.getenv("ENABLE_HYBRID", "true").lower() == "true"

if ENABLE_HYBRID:
    try:
        from hybrid_risk_engine import hybrid_risk_engine
        logger.info("Hybrid Risk Engine integration enabled")
        # Use hybrid engine as the primary engine
        app = hybrid_risk_engine.app
        engine_instance = hybrid_risk_engine
    except ImportError as e:
        logger.warning(f"Hybrid Risk Engine not available: {e}. Using simple engine.")
        app = simple_risk_engine.app
        engine_instance = simple_risk_engine
else:
    logger.info("Using Simple Risk Engine (hybrid disabled)")
    app = simple_risk_engine.app
    engine_instance = simple_risk_engine

if __name__ == "__main__":
    # Configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8200"))
    
    logger.info(f"Starting Risk Engine ({type(engine_instance).__name__}) on {host}:{port}")
    
    # Start the engine on startup
    async def lifespan():
        await engine_instance.start_engine()
    
    # Run startup
    asyncio.run(lifespan())
    
    # Start FastAPI server
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )