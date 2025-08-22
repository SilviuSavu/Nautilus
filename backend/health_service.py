"""
Comprehensive Health Check Service
=================================

Provides detailed health monitoring for all system components.
"""

import asyncio
import time
from typing import Dict, Any, List
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

import redis
import asyncpg
import httpx


class HealthStatus(str, Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded" 
    UNHEALTHY = "unhealthy"


@dataclass
class ServiceHealth:
    """Health information for a service."""
    name: str
    status: HealthStatus
    response_time_ms: float
    last_check: datetime
    error_message: str | None = None
    metadata: Dict[str, Any] | None = None


class HealthCheckService:
    """Comprehensive health monitoring service."""
    
    def __init__(self):
        self.redis_url = "redis://localhost:6379"
        self.postgres_url = "postgresql://nautilus:nautilus123@localhost:5432/nautilus"
        self.ib_gateway_host = "host.docker.internal"
        self.ib_gateway_port = 4002
        
    async def check_redis(self) -> ServiceHealth:
        """Check Redis connectivity and performance."""
        start_time = time.time()
        try:
            r = redis.Redis.from_url(self.redis_url, socket_timeout=5)
            r.ping()
            info = r.info()
            
            response_time = (time.time() - start_time) * 1000
            
            return ServiceHealth(
                name="redis",
                status=HealthStatus.HEALTHY,
                response_time_ms=response_time,
                last_check=datetime.now(),
                metadata={
                    "version": info.get("redis_version"),
                    "memory_used": info.get("used_memory_human"),
                    "connections": info.get("connected_clients"),
                    "uptime_seconds": info.get("uptime_in_seconds")
                }
            )
        except Exception as e:
            return ServiceHealth(
                name="redis",
                status=HealthStatus.UNHEALTHY,
                response_time_ms=(time.time() - start_time) * 1000,
                last_check=datetime.now(),
                error_message=str(e)
            )
    
    async def check_postgres(self) -> ServiceHealth:
        """Check PostgreSQL connectivity and performance."""
        start_time = time.time()
        try:
            conn = await asyncpg.connect(self.postgres_url)
            
            # Test basic query
            result = await conn.fetchval("SELECT version()")
            
            # Test table access
            bars_count = await conn.fetchval("SELECT COUNT(*) FROM bars")
            
            await conn.close()
            
            response_time = (time.time() - start_time) * 1000
            
            return ServiceHealth(
                name="postgres",
                status=HealthStatus.HEALTHY,
                response_time_ms=response_time,
                last_check=datetime.now(),
                metadata={
                    "version": result,
                    "bars_count": bars_count,
                    "connection_success": True
                }
            )
        except Exception as e:
            return ServiceHealth(
                name="postgres", 
                status=HealthStatus.UNHEALTHY,
                response_time_ms=(time.time() - start_time) * 1000,
                last_check=datetime.now(),
                error_message=str(e)
            )
    
    async def check_ib_gateway(self) -> ServiceHealth:
        """Check IB Gateway connectivity."""
        start_time = time.time()
        try:
            # Test basic socket connectivity
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((self.ib_gateway_host, self.ib_gateway_port))
            sock.close()
            
            response_time = (time.time() - start_time) * 1000
            
            if result == 0:
                return ServiceHealth(
                    name="ib_gateway",
                    status=HealthStatus.HEALTHY,
                    response_time_ms=response_time,
                    last_check=datetime.now(),
                    metadata={
                        "host": self.ib_gateway_host,
                        "port": self.ib_gateway_port,
                        "connection_success": True
                    }
                )
            else:
                return ServiceHealth(
                    name="ib_gateway",
                    status=HealthStatus.UNHEALTHY,
                    response_time_ms=response_time,
                    last_check=datetime.now(),
                    error_message=f"Connection failed with code {result}"
                )
                
        except Exception as e:
            return ServiceHealth(
                name="ib_gateway",
                status=HealthStatus.UNHEALTHY,
                response_time_ms=(time.time() - start_time) * 1000,
                last_check=datetime.now(),
                error_message=str(e)
            )
    
    async def check_api_endpoints(self) -> List[ServiceHealth]:
        """Check critical API endpoints."""
        endpoints = [
            ("messagebus", "http://localhost:8000/api/v1/messagebus/status"),
            ("historical_data", "http://localhost:8000/api/v1/historical/backfill/status"),
            ("ib_status", "http://localhost:8000/api/v1/ib/status")
        ]
        
        results = []
        async with httpx.AsyncClient(timeout=5.0) as client:
            for name, url in endpoints:
                start_time = time.time()
                try:
                    response = await client.get(url)
                    response_time = (time.time() - start_time) * 1000
                    
                    status = HealthStatus.HEALTHY if response.status_code == 200 else HealthStatus.DEGRADED
                    
                    results.append(ServiceHealth(
                        name=name,
                        status=status,
                        response_time_ms=response_time,
                        last_check=datetime.now(),
                        metadata={
                            "status_code": response.status_code,
                            "endpoint": url
                        }
                    ))
                except Exception as e:
                    results.append(ServiceHealth(
                        name=name,
                        status=HealthStatus.UNHEALTHY,
                        response_time_ms=(time.time() - start_time) * 1000,
                        last_check=datetime.now(),
                        error_message=str(e),
                        metadata={"endpoint": url}
                    ))
        
        return results
    
    async def get_comprehensive_health(self) -> Dict[str, Any]:
        """Get comprehensive health check for all services."""
        
        # Run all health checks concurrently
        tasks = [
            self.check_redis(),
            self.check_postgres(), 
            self.check_ib_gateway(),
        ]
        
        # Add API endpoint checks
        api_checks_task = self.check_api_endpoints()
        
        # Execute all checks
        redis_health, postgres_health, ib_health = await asyncio.gather(*tasks)
        api_healths = await api_checks_task
        
        # Combine all results
        all_services = [redis_health, postgres_health, ib_health] + api_healths
        
        # Calculate overall status
        unhealthy_count = sum(1 for s in all_services if s.status == HealthStatus.UNHEALTHY)
        degraded_count = sum(1 for s in all_services if s.status == HealthStatus.DEGRADED)
        
        if unhealthy_count > 0:
            overall_status = HealthStatus.UNHEALTHY
        elif degraded_count > 0:
            overall_status = HealthStatus.DEGRADED  
        else:
            overall_status = HealthStatus.HEALTHY
        
        return {
            "overall_status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "services": {
                service.name: {
                    "status": service.status,
                    "response_time_ms": service.response_time_ms,
                    "last_check": service.last_check.isoformat(),
                    "error_message": service.error_message,
                    "metadata": service.metadata
                }
                for service in all_services
            },
            "summary": {
                "total_services": len(all_services),
                "healthy": len([s for s in all_services if s.status == HealthStatus.HEALTHY]),
                "degraded": degraded_count,
                "unhealthy": unhealthy_count,
                "average_response_time_ms": sum(s.response_time_ms for s in all_services) / len(all_services)
            }
        }


# Singleton instance
health_service = HealthCheckService()