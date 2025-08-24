"""
Engine Health Monitoring System for Nautilus Hybrid Architecture
Provides comprehensive health checking and performance monitoring for all 9 engines.
"""

import asyncio
import httpx
import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class EngineStatus(Enum):
    """Engine health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class EngineMetrics:
    """Engine performance and health metrics"""
    name: str
    url: str
    port: int
    status: EngineStatus = EngineStatus.UNKNOWN
    last_check: float = 0
    response_time_ms: float = 0
    avg_response_time_ms: float = 0
    success_rate: float = 100.0
    cpu_percent: float = 0
    memory_percent: float = 0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    consecutive_failures: int = 0
    last_error: Optional[str] = None
    uptime_seconds: float = 0
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    response_history: List[float] = field(default_factory=list)
    
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate percentage"""
        return 100.0 - self.success_rate
    
    def update_response_time(self, response_time_ms: float):
        """Update response time metrics"""
        self.response_time_ms = response_time_ms
        self.response_history.append(response_time_ms)
        
        # Keep only last 100 response times
        if len(self.response_history) > 100:
            self.response_history = self.response_history[-100:]
        
        # Calculate average
        if self.response_history:
            self.avg_response_time_ms = sum(self.response_history) / len(self.response_history)


class EngineHealthChecker:
    """
    Comprehensive health checker for all Nautilus engines.
    Monitors health, performance, and availability of all 9 containerized engines.
    """
    
    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.engines: Dict[str, EngineMetrics] = {}
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=2.0, read=10.0),
            limits=httpx.Limits(max_connections=20)
        )
        self.running = False
        self.health_check_task: Optional[asyncio.Task] = None
        
        # Initialize engine configurations
        self._initialize_engines()
    
    def _initialize_engines(self):
        """Initialize all 9 engine configurations"""
        engine_configs = [
            ("analytics", "http://localhost:8100", 8100),
            ("risk", "http://localhost:8200", 8200),
            ("factor", "http://localhost:8300", 8300),
            ("ml", "http://localhost:8400", 8400),
            ("features", "http://localhost:8500", 8500),
            ("websocket", "http://localhost:8600", 8600),
            ("strategy", "http://localhost:8700", 8700),
            ("marketdata", "http://localhost:8800", 8800),
            ("portfolio", "http://localhost:8900", 8900)
        ]
        
        for name, url, port in engine_configs:
            self.engines[name] = EngineMetrics(name=name, url=url, port=port)
        
        logger.info(f"🔍 Initialized health monitoring for {len(self.engines)} engines")
    
    async def start_monitoring(self):
        """Start continuous health monitoring"""
        if self.running:
            logger.warning("🔍 Health monitoring already running")
            return
        
        self.running = True
        self.health_check_task = asyncio.create_task(self._monitor_loop())
        logger.info(f"🔍 Started engine health monitoring (interval: {self.check_interval}s)")
    
    async def stop_monitoring(self):
        """Stop health monitoring"""
        if not self.running:
            return
        
        self.running = False
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
        
        await self.http_client.aclose()
        logger.info("🔍 Stopped engine health monitoring")
    
    async def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                await self._check_all_engines()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"🔍 Health monitoring error: {e}")
                await asyncio.sleep(5)  # Brief pause on error
    
    async def _check_all_engines(self):
        """Check health of all engines concurrently"""
        tasks = []
        for engine_name in self.engines.keys():
            task = asyncio.create_task(self._check_engine_health(engine_name))
            tasks.append(task)
        
        # Wait for all health checks to complete
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _check_engine_health(self, engine_name: str):
        """Check health of a specific engine"""
        engine = self.engines[engine_name]
        start_time = time.time()
        
        try:
            # Perform health check request
            response = await self.http_client.get(f"{engine.url}/health")
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            
            engine.last_check = time.time()
            engine.update_response_time(response_time)
            engine.total_requests += 1
            
            if response.status_code == 200:
                # Successful health check
                engine.successful_requests += 1
                engine.consecutive_failures = 0
                engine.last_error = None
                
                # Parse response for additional metrics
                try:
                    health_data = response.json()
                    await self._parse_health_response(engine, health_data)
                except Exception as e:
                    logger.debug(f"🔍 Failed to parse health response for {engine_name}: {e}")
                
                # Determine status based on response time and health data
                engine.status = self._determine_engine_status(engine, response_time)
                
            else:
                # Non-200 response
                engine.failed_requests += 1
                engine.consecutive_failures += 1
                engine.status = EngineStatus.UNHEALTHY
                engine.last_error = f"HTTP {response.status_code}"
                
        except asyncio.TimeoutError:
            # Timeout
            response_time = (time.time() - start_time) * 1000
            engine.last_check = time.time()
            engine.total_requests += 1
            engine.failed_requests += 1
            engine.consecutive_failures += 1
            engine.status = EngineStatus.UNHEALTHY
            engine.last_error = f"Timeout after {response_time:.0f}ms"
            
        except Exception as e:
            # Other errors (connection refused, etc.)
            response_time = (time.time() - start_time) * 1000
            engine.last_check = time.time()
            engine.total_requests += 1
            engine.failed_requests += 1
            engine.consecutive_failures += 1
            engine.status = EngineStatus.UNHEALTHY
            engine.last_error = f"{type(e).__name__}: {str(e)}"
        
        # Update success rate
        if engine.total_requests > 0:
            engine.success_rate = (engine.successful_requests / engine.total_requests) * 100
        
        # Log status changes
        self._log_status_changes(engine_name, engine)
    
    async def _parse_health_response(self, engine: EngineMetrics, health_data: Dict[str, Any]):
        """Parse health response for additional metrics"""
        try:
            # Standard health fields
            if "uptime_seconds" in health_data:
                engine.uptime_seconds = float(health_data["uptime_seconds"])
            
            # Engine-specific metrics
            if "cpu_percent" in health_data:
                engine.cpu_percent = float(health_data["cpu_percent"])
            
            if "memory_percent" in health_data:
                engine.memory_percent = float(health_data["memory_percent"])
            
            # Store custom metrics
            custom_fields = ["factors_calculated", "ml_predictions", "strategies_running", 
                           "websocket_connections", "risk_calculations", "cache_entries"]
            
            for field in custom_fields:
                if field in health_data:
                    engine.custom_metrics[field] = health_data[field]
                    
        except Exception as e:
            logger.debug(f"🔍 Error parsing health data for {engine.name}: {e}")
    
    def _determine_engine_status(self, engine: EngineMetrics, response_time_ms: float) -> EngineStatus:
        """Determine engine status based on metrics"""
        # Critical engines have stricter requirements
        critical_engines = ["strategy", "risk"]
        is_critical = engine.name in critical_engines
        
        # Response time thresholds
        healthy_threshold = 100 if is_critical else 500
        degraded_threshold = 500 if is_critical else 2000
        
        # Check consecutive failures
        if engine.consecutive_failures >= 3:
            return EngineStatus.UNHEALTHY
        
        # Check response time
        if response_time_ms > degraded_threshold:
            return EngineStatus.UNHEALTHY
        elif response_time_ms > healthy_threshold:
            return EngineStatus.DEGRADED
        
        # Check success rate
        if engine.success_rate < 95:
            return EngineStatus.DEGRADED
        elif engine.success_rate < 90:
            return EngineStatus.UNHEALTHY
        
        return EngineStatus.HEALTHY
    
    def _log_status_changes(self, engine_name: str, engine: EngineMetrics):
        """Log significant status changes"""
        # This would typically store previous status, simplified for now
        if engine.status == EngineStatus.UNHEALTHY and engine.consecutive_failures == 1:
            logger.error(f"🚨 Engine {engine_name} became unhealthy: {engine.last_error}")
        elif engine.status == EngineStatus.HEALTHY and engine.consecutive_failures == 0:
            logger.info(f"✅ Engine {engine_name} recovered to healthy state")
    
    async def get_engine_health(self, engine_name: str) -> Optional[EngineMetrics]:
        """Get health status of a specific engine"""
        return self.engines.get(engine_name)
    
    async def get_all_engine_health(self) -> Dict[str, EngineMetrics]:
        """Get health status of all engines"""
        return self.engines.copy()
    
    async def get_healthy_engines(self, engine_type: Optional[str] = None) -> List[str]:
        """Get list of healthy engines, optionally filtered by type"""
        healthy = []
        for name, metrics in self.engines.items():
            if metrics.status == EngineStatus.HEALTHY:
                if engine_type is None or name == engine_type:
                    healthy.append(name)
        return healthy
    
    async def get_system_health_summary(self) -> Dict[str, Any]:
        """Get overall system health summary"""
        healthy_count = sum(1 for e in self.engines.values() if e.status == EngineStatus.HEALTHY)
        degraded_count = sum(1 for e in self.engines.values() if e.status == EngineStatus.DEGRADED)
        unhealthy_count = sum(1 for e in self.engines.values() if e.status == EngineStatus.UNHEALTHY)
        
        avg_response_time = 0
        total_success_rate = 0
        
        if self.engines:
            avg_response_time = sum(e.avg_response_time_ms for e in self.engines.values()) / len(self.engines)
            total_success_rate = sum(e.success_rate for e in self.engines.values()) / len(self.engines)
        
        # Overall system status
        if unhealthy_count > 2:
            overall_status = "unhealthy"
        elif unhealthy_count > 0 or degraded_count > 3:
            overall_status = "degraded"
        else:
            overall_status = "healthy"
        
        return {
            "overall_status": overall_status,
            "total_engines": len(self.engines),
            "healthy_engines": healthy_count,
            "degraded_engines": degraded_count,
            "unhealthy_engines": unhealthy_count,
            "system_availability": (healthy_count + degraded_count) / len(self.engines) * 100,
            "average_response_time_ms": round(avg_response_time, 2),
            "average_success_rate": round(total_success_rate, 2),
            "last_check": time.time(),
            "engines": {
                name: {
                    "status": metrics.status.value,
                    "response_time_ms": round(metrics.response_time_ms, 2),
                    "success_rate": round(metrics.success_rate, 2),
                    "consecutive_failures": metrics.consecutive_failures,
                    "last_error": metrics.last_error,
                    "uptime_seconds": metrics.uptime_seconds,
                    "custom_metrics": metrics.custom_metrics
                }
                for name, metrics in self.engines.items()
            }
        }
    
    async def force_health_check(self, engine_name: Optional[str] = None):
        """Force immediate health check for specific engine or all engines"""
        if engine_name:
            if engine_name in self.engines:
                await self._check_engine_health(engine_name)
                logger.info(f"🔍 Forced health check for {engine_name}")
            else:
                raise ValueError(f"Unknown engine: {engine_name}")
        else:
            await self._check_all_engines()
            logger.info("🔍 Forced health check for all engines")


# Global health monitor instance
health_monitor = EngineHealthChecker()


async def get_engine_health_summary() -> Dict[str, Any]:
    """Convenience function to get health summary"""
    return await health_monitor.get_system_health_summary()


async def is_engine_healthy(engine_name: str) -> bool:
    """Check if specific engine is healthy"""
    engine = await health_monitor.get_engine_health(engine_name)
    return engine is not None and engine.status == EngineStatus.HEALTHY