#!/usr/bin/env python3
"""
Nautilus System Health Monitor
Real-time monitoring and alerting for all 9 containerized engines
"""

import asyncio
import aiohttp
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('system_health.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class EngineHealth:
    name: str
    port: int
    status: str
    response_time_ms: float
    last_check: datetime
    consecutive_failures: int
    total_checks: int
    success_rate: float

@dataclass
class SystemAlert:
    level: str  # INFO, WARNING, CRITICAL
    engine: str
    message: str
    timestamp: datetime
    metric_value: Optional[float] = None

class HealthMonitor:
    def __init__(self):
        self.engines = {
            'analytics': {'port': 8100, 'name': 'Analytics Engine'},
            'risk': {'port': 8200, 'name': 'Risk Engine'},
            'factor': {'port': 8300, 'name': 'Factor Engine'},
            'ml': {'port': 8400, 'name': 'ML Engine'},
            'features': {'port': 8500, 'name': 'Features Engine'},
            'websocket': {'port': 8600, 'name': 'WebSocket Engine'},
            'strategy': {'port': 8700, 'name': 'Strategy Engine'},
            'marketdata': {'port': 8800, 'name': 'MarketData Engine'},
            'portfolio': {'port': 8900, 'name': 'Portfolio Engine'}
        }
        
        self.health_status = {}
        self.alerts = []
        self.alert_thresholds = {
            'response_time_warning': 100.0,  # ms
            'response_time_critical': 500.0,  # ms
            'failure_rate_warning': 20.0,  # %
            'failure_rate_critical': 50.0,  # %
            'consecutive_failures_critical': 5
        }
        
        # Initialize health tracking
        for engine_id, config in self.engines.items():
            self.health_status[engine_id] = EngineHealth(
                name=config['name'],
                port=config['port'],
                status='unknown',
                response_time_ms=0.0,
                last_check=datetime.now(),
                consecutive_failures=0,
                total_checks=0,
                success_rate=0.0
            )
    
    async def check_engine_health(self, session: aiohttp.ClientSession, engine_id: str) -> EngineHealth:
        """Check health of a single engine"""
        engine = self.health_status[engine_id]
        port = self.engines[engine_id]['port']
        
        start_time = time.time()
        try:
            async with session.get(
                f'http://localhost:{port}/health',
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                end_time = time.time()
                response_time_ms = (end_time - start_time) * 1000
                
                if response.status == 200:
                    engine.status = 'healthy'
                    engine.consecutive_failures = 0
                    engine.response_time_ms = response_time_ms
                    
                    # Check for performance alerts
                    await self._check_performance_alerts(engine_id, response_time_ms)
                    
                else:
                    engine.status = f'error_{response.status}'
                    engine.consecutive_failures += 1
                    
        except asyncio.TimeoutError:
            engine.status = 'timeout'
            engine.consecutive_failures += 1
            engine.response_time_ms = 10000  # 10s timeout
            
        except Exception as e:
            engine.status = f'connection_failed'
            engine.consecutive_failures += 1
            engine.response_time_ms = 0.0
        
        # Update statistics
        engine.total_checks += 1
        engine.last_check = datetime.now()
        
        # Calculate success rate
        success_checks = engine.total_checks - engine.consecutive_failures
        if engine.consecutive_failures > 0:
            # Recent failure, need more sophisticated calculation
            success_rate = max(0, (engine.total_checks - engine.consecutive_failures) / engine.total_checks * 100)
        else:
            success_rate = 100.0 if engine.total_checks > 0 else 0.0
        
        engine.success_rate = success_rate
        
        # Generate critical alerts
        await self._check_critical_alerts(engine_id, engine)
        
        return engine
    
    async def _check_performance_alerts(self, engine_id: str, response_time_ms: float):
        """Check for performance-based alerts"""
        if response_time_ms > self.alert_thresholds['response_time_critical']:
            await self._generate_alert(
                'CRITICAL', engine_id,
                f'Response time critically high: {response_time_ms:.2f}ms',
                response_time_ms
            )
        elif response_time_ms > self.alert_thresholds['response_time_warning']:
            await self._generate_alert(
                'WARNING', engine_id,
                f'Response time elevated: {response_time_ms:.2f}ms',
                response_time_ms
            )
    
    async def _check_critical_alerts(self, engine_id: str, engine: EngineHealth):
        """Check for critical system alerts"""
        # Consecutive failures alert
        if engine.consecutive_failures >= self.alert_thresholds['consecutive_failures_critical']:
            await self._generate_alert(
                'CRITICAL', engine_id,
                f'Engine down: {engine.consecutive_failures} consecutive failures',
                engine.consecutive_failures
            )
        
        # Success rate alerts
        if engine.total_checks >= 10:  # Only after sufficient samples
            if engine.success_rate <= (100 - self.alert_thresholds['failure_rate_critical']):
                await self._generate_alert(
                    'CRITICAL', engine_id,
                    f'High failure rate: {100 - engine.success_rate:.1f}%',
                    engine.success_rate
                )
            elif engine.success_rate <= (100 - self.alert_thresholds['failure_rate_warning']):
                await self._generate_alert(
                    'WARNING', engine_id,
                    f'Elevated failure rate: {100 - engine.success_rate:.1f}%',
                    engine.success_rate
                )
    
    async def _generate_alert(self, level: str, engine: str, message: str, metric_value: float):
        """Generate and log an alert"""
        alert = SystemAlert(
            level=level,
            engine=engine,
            message=message,
            timestamp=datetime.now(),
            metric_value=metric_value
        )
        
        self.alerts.append(alert)
        
        # Log the alert
        log_level = getattr(logging, level.upper(), logging.INFO)
        logger.log(log_level, f"[{level}] {engine.upper()}: {message}")
        
        # Keep only last 100 alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
    
    async def run_health_checks(self) -> Dict[str, EngineHealth]:
        """Run health checks on all engines"""
        async with aiohttp.ClientSession() as session:
            tasks = []
            for engine_id in self.engines.keys():
                tasks.append(self.check_engine_health(session, engine_id))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Update health status
            for i, engine_id in enumerate(self.engines.keys()):
                if not isinstance(results[i], Exception):
                    self.health_status[engine_id] = results[i]
        
        return self.health_status
    
    def get_system_summary(self) -> Dict:
        """Generate system health summary"""
        healthy_engines = sum(1 for engine in self.health_status.values() if engine.status == 'healthy')
        total_engines = len(self.health_status)
        
        avg_response_time = sum(engine.response_time_ms for engine in self.health_status.values()) / total_engines
        
        recent_critical_alerts = len([
            alert for alert in self.alerts[-20:] 
            if alert.level == 'CRITICAL' and (datetime.now() - alert.timestamp).seconds < 300
        ])
        
        return {
            'system_status': 'healthy' if healthy_engines == total_engines else 'degraded' if healthy_engines > total_engines // 2 else 'critical',
            'healthy_engines': healthy_engines,
            'total_engines': total_engines,
            'availability_percentage': (healthy_engines / total_engines) * 100,
            'average_response_time_ms': avg_response_time,
            'recent_critical_alerts': recent_critical_alerts,
            'last_check': datetime.now().isoformat()
        }
    
    def generate_dashboard_data(self) -> Dict:
        """Generate comprehensive dashboard data"""
        return {
            'summary': self.get_system_summary(),
            'engines': {
                engine_id: {
                    'name': health.name,
                    'port': health.port,
                    'status': health.status,
                    'response_time_ms': health.response_time_ms,
                    'success_rate': health.success_rate,
                    'consecutive_failures': health.consecutive_failures,
                    'total_checks': health.total_checks,
                    'last_check': health.last_check.isoformat()
                }
                for engine_id, health in self.health_status.items()
            },
            'recent_alerts': [
                {
                    'level': alert.level,
                    'engine': alert.engine,
                    'message': alert.message,
                    'timestamp': alert.timestamp.isoformat(),
                    'metric_value': alert.metric_value
                }
                for alert in self.alerts[-20:]  # Last 20 alerts
            ]
        }
    
    async def continuous_monitoring(self, interval_seconds: int = 30):
        """Run continuous health monitoring"""
        logger.info(f"Starting continuous monitoring (interval: {interval_seconds}s)")
        
        while True:
            try:
                # Run health checks
                await self.run_health_checks()
                
                # Generate summary
                summary = self.get_system_summary()
                logger.info(
                    f"System Status: {summary['system_status'].upper()} | "
                    f"Healthy: {summary['healthy_engines']}/{summary['total_engines']} | "
                    f"Avg Response: {summary['average_response_time_ms']:.1f}ms"
                )
                
                # Save dashboard data
                dashboard_data = self.generate_dashboard_data()
                with open('system_health_dashboard.json', 'w') as f:
                    json.dump(dashboard_data, f, indent=2)
                
                # Log any recent critical issues
                critical_engines = [
                    engine_id for engine_id, health in self.health_status.items()
                    if health.status != 'healthy'
                ]
                if critical_engines:
                    logger.warning(f"Issues detected: {', '.join(critical_engines)}")
                
                await asyncio.sleep(interval_seconds)
                
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(interval_seconds)

async def main():
    """Main monitoring loop"""
    monitor = HealthMonitor()
    
    # Run initial check
    logger.info("Running initial system health check...")
    await monitor.run_health_checks()
    
    summary = monitor.get_system_summary()
    logger.info(f"Initial Status: {summary['system_status']} ({summary['healthy_engines']}/{summary['total_engines']} engines healthy)")
    
    # Start continuous monitoring
    await monitor.continuous_monitoring(interval_seconds=30)

if __name__ == "__main__":
    asyncio.run(main())