"""
Container Performance Monitor for Nautilus Trading Platform
Comprehensive monitoring for all 16+ containerized engines with M4 Max optimization focus:

Monitored Engines:
1. Analytics Engine (8100)
2. Risk Engine (8200) 
3. Factor Engine (8300)
4. ML Engine (8400)
5. Features Engine (8500)
6. WebSocket Engine (8600)
7. Strategy Engine (8700)
8. MarketData Engine (8800)
9. Portfolio Engine (8900)
10. Backend API (8001)
11. Frontend (3000)
12. PostgreSQL (5432)
13. Redis (6379)
14. Prometheus (9090)
15. Grafana (3002)
16. Nginx (80)
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
import aiohttp
import docker
import redis
from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram, Info
from prometheus_client.exposition import generate_latest

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ContainerStatus(Enum):
    RUNNING = "running"
    STOPPED = "stopped"
    RESTARTING = "restarting"
    PAUSED = "paused"
    EXITED = "exited"
    DEAD = "dead"
    UNKNOWN = "unknown"

@dataclass
class ContainerMetrics:
    """Container performance metrics data structure"""
    container_name: str
    container_id: str
    status: ContainerStatus
    cpu_usage_percent: float
    memory_usage_mb: float
    memory_limit_mb: float
    memory_usage_percent: float
    network_rx_mb: float
    network_tx_mb: float
    disk_read_mb: float
    disk_write_mb: float
    restart_count: int
    uptime_seconds: float
    health_status: str
    port_mappings: Dict[str, str]
    environment_vars: Dict[str, str]
    resource_limits: Dict[str, any]
    timestamp: datetime

@dataclass
class EngineHealthMetrics:
    """Engine-specific health metrics"""
    engine_name: str
    endpoint_url: str
    response_time_ms: float
    is_healthy: bool
    status_code: int
    error_message: Optional[str]
    last_successful_check: datetime
    consecutive_failures: int
    custom_metrics: Dict[str, any]

class ContainerPerformanceMonitor:
    """Comprehensive container performance monitoring system"""
    
    def __init__(self, redis_host: str = "redis", redis_port: int = 6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.docker_client = docker.from_env()
        
        # Prometheus metrics setup
        self.registry = CollectorRegistry()
        self._setup_prometheus_metrics()
        
        # Engine configuration
        self.engines = {
            "nautilus-analytics-engine": {
                "port": 8100,
                "health_endpoint": "/health",
                "metrics_endpoint": "/metrics"
            },
            "nautilus-risk-engine": {
                "port": 8200,
                "health_endpoint": "/health",
                "metrics_endpoint": "/metrics"
            },
            "nautilus-factor-engine": {
                "port": 8300,
                "health_endpoint": "/health",
                "metrics_endpoint": "/metrics"
            },
            "nautilus-ml-engine": {
                "port": 8400,
                "health_endpoint": "/health",
                "metrics_endpoint": "/metrics"
            },
            "nautilus-features-engine": {
                "port": 8500,
                "health_endpoint": "/health",
                "metrics_endpoint": "/metrics"
            },
            "nautilus-websocket-engine": {
                "port": 8600,
                "health_endpoint": "/health",
                "metrics_endpoint": "/metrics"
            },
            "nautilus-strategy-engine": {
                "port": 8700,
                "health_endpoint": "/health",
                "metrics_endpoint": "/metrics"
            },
            "nautilus-marketdata-engine": {
                "port": 8800,
                "health_endpoint": "/health",
                "metrics_endpoint": "/metrics"
            },
            "nautilus-portfolio-engine": {
                "port": 8900,
                "health_endpoint": "/health",
                "metrics_endpoint": "/metrics"
            },
            "nautilus-backend": {
                "port": 8001,
                "health_endpoint": "/health",
                "metrics_endpoint": "/metrics"
            },
            "nautilus-frontend": {
                "port": 3000,
                "health_endpoint": "/",
                "metrics_endpoint": None
            },
            "nautilus-postgres": {
                "port": 5432,
                "health_endpoint": None,
                "metrics_endpoint": None
            },
            "nautilus-redis": {
                "port": 6379,
                "health_endpoint": None,
                "metrics_endpoint": None
            },
            "nautilus-prometheus": {
                "port": 9090,
                "health_endpoint": "/-/healthy",
                "metrics_endpoint": "/metrics"
            },
            "nautilus-grafana": {
                "port": 3002,
                "health_endpoint": "/api/health",
                "metrics_endpoint": "/metrics"
            },
            "nautilus-nginx": {
                "port": 80,
                "health_endpoint": "/",
                "metrics_endpoint": None
            }
        }
        
        # Monitoring state
        self.monitoring = False
        self.health_check_failures = {}
        
        logger.info("Container Performance Monitor initialized")
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics for container monitoring"""
        # Container Resource Metrics
        self.container_cpu_gauge = Gauge('nautilus_container_cpu_usage_percent', 
                                       'Container CPU usage percentage', 
                                       ['container_name'], registry=self.registry)
        self.container_memory_gauge = Gauge('nautilus_container_memory_usage_mb', 
                                          'Container memory usage in MB', 
                                          ['container_name'], registry=self.registry)
        self.container_memory_percent_gauge = Gauge('nautilus_container_memory_usage_percent', 
                                                   'Container memory usage percentage', 
                                                   ['container_name'], registry=self.registry)
        
        # Container Network Metrics
        self.container_network_rx_gauge = Gauge('nautilus_container_network_rx_mb', 
                                              'Container network receive MB', 
                                              ['container_name'], registry=self.registry)
        self.container_network_tx_gauge = Gauge('nautilus_container_network_tx_mb', 
                                              'Container network transmit MB', 
                                              ['container_name'], registry=self.registry)
        
        # Container Disk I/O Metrics
        self.container_disk_read_gauge = Gauge('nautilus_container_disk_read_mb', 
                                             'Container disk read MB', 
                                             ['container_name'], registry=self.registry)
        self.container_disk_write_gauge = Gauge('nautilus_container_disk_write_mb', 
                                              'Container disk write MB', 
                                              ['container_name'], registry=self.registry)
        
        # Container Health Metrics
        self.container_uptime_gauge = Gauge('nautilus_container_uptime_seconds', 
                                          'Container uptime in seconds', 
                                          ['container_name'], registry=self.registry)
        self.container_restart_counter = Counter('nautilus_container_restarts_total', 
                                                'Total container restarts', 
                                                ['container_name'], registry=self.registry)
        self.container_health_gauge = Gauge('nautilus_container_health_status', 
                                          'Container health status (1=healthy, 0=unhealthy)', 
                                          ['container_name'], registry=self.registry)
        
        # Engine Health Metrics
        self.engine_response_time_gauge = Gauge('nautilus_engine_response_time_ms', 
                                              'Engine HTTP response time in milliseconds', 
                                              ['engine_name'], registry=self.registry)
        self.engine_health_gauge = Gauge('nautilus_engine_health_status', 
                                       'Engine health status (1=healthy, 0=unhealthy)', 
                                       ['engine_name'], registry=self.registry)
        self.engine_consecutive_failures_gauge = Gauge('nautilus_engine_consecutive_failures', 
                                                      'Engine consecutive health check failures', 
                                                      ['engine_name'], registry=self.registry)
        
        # Performance Counters
        self.metrics_collected_counter = Counter('nautilus_container_metrics_collected_total', 
                                               'Total container metrics collected', 
                                               registry=self.registry)
        self.health_checks_counter = Counter('nautilus_engine_health_checks_total', 
                                           'Total engine health checks performed', 
                                           ['engine_name', 'status'], registry=self.registry)
        
        # Collection Performance
        self.collection_duration_histogram = Histogram('nautilus_container_collection_duration_seconds', 
                                                      'Time taken to collect container metrics', 
                                                      registry=self.registry)
        
        # System Info
        self.system_info = Info('nautilus_monitoring_system_info', 
                              'System information for monitoring setup', 
                              registry=self.registry)
    
    def _get_container_metrics(self, container) -> Optional[ContainerMetrics]:
        """Get detailed metrics for a single container"""
        try:
            # Get container stats
            stats = container.stats(stream=False)
            attrs = container.attrs
            
            # Parse container name
            container_name = container.name
            container_id = container.id[:12]
            
            # Container status
            status = ContainerStatus(container.status.lower())
            
            # CPU metrics
            cpu_stats = stats.get('cpu_stats', {})
            precpu_stats = stats.get('precpu_stats', {})
            
            if cpu_stats and precpu_stats:
                cpu_delta = cpu_stats.get('cpu_usage', {}).get('total_usage', 0) - \
                           precpu_stats.get('cpu_usage', {}).get('total_usage', 0)
                system_delta = cpu_stats.get('system_cpu_usage', 0) - \
                              precpu_stats.get('system_cpu_usage', 0)
                
                if system_delta > 0:
                    cpu_usage_percent = (cpu_delta / system_delta) * 100.0
                else:
                    cpu_usage_percent = 0.0
            else:
                cpu_usage_percent = 0.0
            
            # Memory metrics
            memory_stats = stats.get('memory_stats', {})
            memory_usage = memory_stats.get('usage', 0)
            memory_limit = memory_stats.get('limit', 0)
            memory_usage_mb = memory_usage / (1024 * 1024)
            memory_limit_mb = memory_limit / (1024 * 1024)
            memory_usage_percent = (memory_usage / memory_limit * 100) if memory_limit > 0 else 0.0
            
            # Network metrics
            networks = stats.get('networks', {})
            total_rx_bytes = sum(net.get('rx_bytes', 0) for net in networks.values())
            total_tx_bytes = sum(net.get('tx_bytes', 0) for net in networks.values())
            network_rx_mb = total_rx_bytes / (1024 * 1024)
            network_tx_mb = total_tx_bytes / (1024 * 1024)
            
            # Disk I/O metrics
            blkio_stats = stats.get('blkio_stats', {})
            io_service_bytes = blkio_stats.get('io_service_bytes_recursive', [])
            
            disk_read_bytes = 0
            disk_write_bytes = 0
            for entry in io_service_bytes:
                if entry.get('op') == 'Read':
                    disk_read_bytes += entry.get('value', 0)
                elif entry.get('op') == 'Write':
                    disk_write_bytes += entry.get('value', 0)
            
            disk_read_mb = disk_read_bytes / (1024 * 1024)
            disk_write_mb = disk_write_bytes / (1024 * 1024)
            
            # Container metadata
            restart_count = attrs.get('RestartCount', 0)
            started_at = attrs.get('State', {}).get('StartedAt', '')
            
            # Calculate uptime
            if started_at:
                try:
                    start_time = datetime.fromisoformat(started_at.replace('Z', '+00:00'))
                    uptime_seconds = (datetime.now().replace(tzinfo=start_time.tzinfo) - start_time).total_seconds()
                except:
                    uptime_seconds = 0.0
            else:
                uptime_seconds = 0.0
            
            # Health status
            health = attrs.get('State', {}).get('Health', {})
            health_status = health.get('Status', 'none')
            
            # Port mappings
            port_mappings = {}
            ports = attrs.get('NetworkSettings', {}).get('Ports', {})
            for container_port, host_bindings in ports.items():
                if host_bindings:
                    host_port = host_bindings[0].get('HostPort', '')
                    port_mappings[container_port] = host_port
            
            # Environment variables (filtered)
            env_list = attrs.get('Config', {}).get('Env', [])
            environment_vars = {}
            for env_var in env_list:
                if '=' in env_var:
                    key, value = env_var.split('=', 1)
                    # Only store non-sensitive env vars
                    if not any(sensitive in key.upper() for sensitive in ['KEY', 'SECRET', 'PASSWORD', 'TOKEN']):
                        environment_vars[key] = value[:50]  # Truncate long values
            
            # Resource limits
            host_config = attrs.get('HostConfig', {})
            resource_limits = {
                'cpu_count': host_config.get('CpuCount', 0),
                'cpu_percent': host_config.get('CpuPercent', 0),
                'memory_mb': host_config.get('Memory', 0) / (1024 * 1024) if host_config.get('Memory') else 0
            }
            
            return ContainerMetrics(
                container_name=container_name,
                container_id=container_id,
                status=status,
                cpu_usage_percent=cpu_usage_percent,
                memory_usage_mb=memory_usage_mb,
                memory_limit_mb=memory_limit_mb,
                memory_usage_percent=memory_usage_percent,
                network_rx_mb=network_rx_mb,
                network_tx_mb=network_tx_mb,
                disk_read_mb=disk_read_mb,
                disk_write_mb=disk_write_mb,
                restart_count=restart_count,
                uptime_seconds=uptime_seconds,
                health_status=health_status,
                port_mappings=port_mappings,
                environment_vars=environment_vars,
                resource_limits=resource_limits,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error getting metrics for container {container.name}: {e}")
            return None
    
    async def _check_engine_health(self, engine_name: str, config: Dict) -> EngineHealthMetrics:
        """Check health of a specific engine"""
        start_time = time.time()
        
        # Default values
        response_time_ms = 0.0
        is_healthy = False
        status_code = 0
        error_message = None
        custom_metrics = {}
        
        if config.get('health_endpoint'):
            try:
                url = f"http://localhost:{config['port']}{config['health_endpoint']}"
                timeout = aiohttp.ClientTimeout(total=10)
                
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(url) as response:
                        response_time_ms = (time.time() - start_time) * 1000
                        status_code = response.status
                        is_healthy = 200 <= status_code < 400
                        
                        if not is_healthy:
                            error_message = f"HTTP {status_code}"
                        
                        # Try to get custom metrics if available
                        if is_healthy and config.get('metrics_endpoint'):
                            try:
                                metrics_url = f"http://localhost:{config['port']}{config['metrics_endpoint']}"
                                async with session.get(metrics_url) as metrics_response:
                                    if metrics_response.status == 200:
                                        custom_metrics['metrics_available'] = True
                                    else:
                                        custom_metrics['metrics_available'] = False
                            except:
                                custom_metrics['metrics_available'] = False
                        
            except asyncio.TimeoutError:
                response_time_ms = 10000  # 10 second timeout
                error_message = "Timeout"
            except aiohttp.ClientError as e:
                response_time_ms = (time.time() - start_time) * 1000
                error_message = f"Connection error: {str(e)[:100]}"
            except Exception as e:
                response_time_ms = (time.time() - start_time) * 1000
                error_message = f"Error: {str(e)[:100]}"
        else:
            # For services without health endpoints, check if container is running
            try:
                container = self.docker_client.containers.get(engine_name)
                is_healthy = container.status == 'running'
                response_time_ms = (time.time() - start_time) * 1000
                if not is_healthy:
                    error_message = f"Container status: {container.status}"
            except docker.errors.NotFound:
                error_message = "Container not found"
            except Exception as e:
                error_message = f"Docker error: {str(e)[:100]}"
        
        # Update failure tracking
        if engine_name not in self.health_check_failures:
            self.health_check_failures[engine_name] = {'count': 0, 'last_success': datetime.now()}
        
        if is_healthy:
            self.health_check_failures[engine_name]['count'] = 0
            self.health_check_failures[engine_name]['last_success'] = datetime.now()
        else:
            self.health_check_failures[engine_name]['count'] += 1
        
        return EngineHealthMetrics(
            engine_name=engine_name,
            endpoint_url=f"http://localhost:{config['port']}{config.get('health_endpoint', '')}",
            response_time_ms=response_time_ms,
            is_healthy=is_healthy,
            status_code=status_code,
            error_message=error_message,
            last_successful_check=self.health_check_failures[engine_name]['last_success'],
            consecutive_failures=self.health_check_failures[engine_name]['count'],
            custom_metrics=custom_metrics
        )
    
    async def collect_container_metrics(self) -> List[ContainerMetrics]:
        """Collect metrics from all containers"""
        container_metrics = []
        
        try:
            with self.collection_duration_histogram.time():
                containers = self.docker_client.containers.list(all=True)
                
                for container in containers:
                    if container.name.startswith('nautilus-') or container.name in self.engines:
                        metrics = self._get_container_metrics(container)
                        if metrics:
                            container_metrics.append(metrics)
                            self._update_container_prometheus_metrics(metrics)
                            
                self.metrics_collected_counter.inc(len(container_metrics))
                
        except Exception as e:
            logger.error(f"Error collecting container metrics: {e}")
        
        return container_metrics
    
    async def collect_engine_health_metrics(self) -> List[EngineHealthMetrics]:
        """Collect health metrics from all engines"""
        health_metrics = []
        
        # Create tasks for concurrent health checks
        health_tasks = []
        for engine_name, config in self.engines.items():
            task = asyncio.create_task(self._check_engine_health(engine_name, config))
            health_tasks.append(task)
        
        # Wait for all health checks to complete
        try:
            results = await asyncio.gather(*health_tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    engine_name = list(self.engines.keys())[i]
                    logger.error(f"Health check failed for {engine_name}: {result}")
                else:
                    health_metrics.append(result)
                    self._update_engine_prometheus_metrics(result)
                    
        except Exception as e:
            logger.error(f"Error in health check collection: {e}")
        
        return health_metrics
    
    def _update_container_prometheus_metrics(self, metrics: ContainerMetrics):
        """Update Prometheus metrics for container"""
        try:
            container_name = metrics.container_name
            
            self.container_cpu_gauge.labels(container_name=container_name).set(metrics.cpu_usage_percent)
            self.container_memory_gauge.labels(container_name=container_name).set(metrics.memory_usage_mb)
            self.container_memory_percent_gauge.labels(container_name=container_name).set(metrics.memory_usage_percent)
            
            self.container_network_rx_gauge.labels(container_name=container_name).set(metrics.network_rx_mb)
            self.container_network_tx_gauge.labels(container_name=container_name).set(metrics.network_tx_mb)
            
            self.container_disk_read_gauge.labels(container_name=container_name).set(metrics.disk_read_mb)
            self.container_disk_write_gauge.labels(container_name=container_name).set(metrics.disk_write_mb)
            
            self.container_uptime_gauge.labels(container_name=container_name).set(metrics.uptime_seconds)
            
            # Health status mapping
            health_value = 1 if metrics.health_status in ['healthy', 'none'] and metrics.status == ContainerStatus.RUNNING else 0
            self.container_health_gauge.labels(container_name=container_name).set(health_value)
            
        except Exception as e:
            logger.error(f"Error updating container Prometheus metrics: {e}")
    
    def _update_engine_prometheus_metrics(self, metrics: EngineHealthMetrics):
        """Update Prometheus metrics for engine health"""
        try:
            engine_name = metrics.engine_name
            
            self.engine_response_time_gauge.labels(engine_name=engine_name).set(metrics.response_time_ms)
            self.engine_health_gauge.labels(engine_name=engine_name).set(1 if metrics.is_healthy else 0)
            self.engine_consecutive_failures_gauge.labels(engine_name=engine_name).set(metrics.consecutive_failures)
            
            # Update health check counter
            status = 'success' if metrics.is_healthy else 'failure'
            self.health_checks_counter.labels(engine_name=engine_name, status=status).inc()
            
        except Exception as e:
            logger.error(f"Error updating engine Prometheus metrics: {e}")
    
    def _store_metrics_redis(self, container_metrics: List[ContainerMetrics], 
                           engine_health_metrics: List[EngineHealthMetrics]):
        """Store metrics in Redis"""
        try:
            timestamp = int(time.time())
            
            # Store container metrics
            for metrics in container_metrics:
                metrics_dict = asdict(metrics)
                metrics_dict['timestamp'] = metrics_dict['timestamp'].isoformat()
                
                # Current metrics
                self.redis_client.set(
                    f"container:metrics:{metrics.container_name}",
                    json.dumps(metrics_dict),
                    ex=300
                )
                
                # Time series data
                self.redis_client.zadd(
                    f"container:cpu:{metrics.container_name}",
                    {timestamp: metrics.cpu_usage_percent}
                )
                self.redis_client.zadd(
                    f"container:memory:{metrics.container_name}",
                    {timestamp: metrics.memory_usage_percent}
                )
            
            # Store engine health metrics
            for metrics in engine_health_metrics:
                metrics_dict = asdict(metrics)
                metrics_dict['last_successful_check'] = metrics_dict['last_successful_check'].isoformat()
                
                self.redis_client.set(
                    f"engine:health:{metrics.engine_name}",
                    json.dumps(metrics_dict),
                    ex=300
                )
                
                self.redis_client.zadd(
                    f"engine:response_time:{metrics.engine_name}",
                    {timestamp: metrics.response_time_ms}
                )
            
            # Cleanup old time series data (keep last 24 hours)
            cutoff_time = timestamp - (24 * 60 * 60)
            for engine_name in self.engines.keys():
                self.redis_client.zremrangebyscore(f"container:cpu:{engine_name}", 0, cutoff_time)
                self.redis_client.zremrangebyscore(f"container:memory:{engine_name}", 0, cutoff_time)
                self.redis_client.zremrangebyscore(f"engine:response_time:{engine_name}", 0, cutoff_time)
            
        except Exception as e:
            logger.error(f"Error storing metrics in Redis: {e}")
    
    async def collect_all_metrics(self) -> Tuple[List[ContainerMetrics], List[EngineHealthMetrics]]:
        """Collect both container and engine health metrics"""
        logger.debug("Collecting container and engine metrics...")
        
        # Run both collections concurrently
        container_task = asyncio.create_task(self.collect_container_metrics())
        engine_task = asyncio.create_task(self.collect_engine_health_metrics())
        
        container_metrics, engine_health_metrics = await asyncio.gather(
            container_task, engine_task
        )
        
        # Store in Redis
        self._store_metrics_redis(container_metrics, engine_health_metrics)
        
        logger.debug(f"Collected metrics for {len(container_metrics)} containers and {len(engine_health_metrics)} engines")
        
        return container_metrics, engine_health_metrics
    
    async def start_monitoring(self, interval: float = 30.0):
        """Start continuous container monitoring"""
        logger.info(f"Starting container performance monitoring (interval: {interval}s)")
        self.monitoring = True
        
        while self.monitoring:
            try:
                await self.collect_all_metrics()
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                logger.info("Container monitoring cancelled")
                break
            except Exception as e:
                logger.error(f"Error in container monitoring loop: {e}")
                await asyncio.sleep(interval)
    
    def stop_monitoring(self):
        """Stop monitoring"""
        logger.info("Stopping container performance monitoring")
        self.monitoring = False
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics in text format"""
        return generate_latest(self.registry).decode('utf-8')
    
    def get_container_status_summary(self) -> Dict[str, any]:
        """Get container status summary"""
        try:
            containers = self.docker_client.containers.list(all=True)
            summary = {
                'total_containers': len(containers),
                'running': 0,
                'stopped': 0,
                'nautilus_containers': 0,
                'containers': []
            }
            
            for container in containers:
                is_nautilus = container.name.startswith('nautilus-')
                if is_nautilus:
                    summary['nautilus_containers'] += 1
                
                if container.status == 'running':
                    summary['running'] += 1
                else:
                    summary['stopped'] += 1
                
                summary['containers'].append({
                    'name': container.name,
                    'status': container.status,
                    'is_nautilus': is_nautilus
                })
            
            return summary
        except Exception as e:
            logger.error(f"Error getting container status summary: {e}")
            return {'error': str(e)}

# Example usage
if __name__ == "__main__":
    async def main():
        monitor = ContainerPerformanceMonitor()
        
        # Test collection
        container_metrics, engine_health = await monitor.collect_all_metrics()
        
        print(f"\n=== Container Performance Summary ===")
        print(f"Monitored containers: {len(container_metrics)}")
        print(f"Engine health checks: {len(engine_health)}")
        
        print(f"\n=== Container Status ===")
        for metrics in container_metrics:
            print(f"  {metrics.container_name}: {metrics.status.value}")
            print(f"    CPU: {metrics.cpu_usage_percent:.1f}%")
            print(f"    Memory: {metrics.memory_usage_mb:.1f}MB ({metrics.memory_usage_percent:.1f}%)")
            print(f"    Uptime: {metrics.uptime_seconds:.0f}s")
        
        print(f"\n=== Engine Health ===")
        for health in engine_health:
            status = "✅ HEALTHY" if health.is_healthy else "❌ UNHEALTHY"
            print(f"  {health.engine_name}: {status}")
            print(f"    Response time: {health.response_time_ms:.1f}ms")
            if health.error_message:
                print(f"    Error: {health.error_message}")
        
        # Start monitoring for 60 seconds
        print(f"\n=== Starting continuous monitoring for 60 seconds ===")
        monitoring_task = asyncio.create_task(monitor.start_monitoring(interval=10.0))
        await asyncio.sleep(60)
        monitor.stop_monitoring()
        monitoring_task.cancel()
        
        print("Container Performance Monitor test completed.")
    
    asyncio.run(main())