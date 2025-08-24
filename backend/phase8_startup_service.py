"""
Phase 8 Autonomous Security Operations Startup Service
Enterprise-grade startup and lifecycle management for autonomous security components.

Provides centralized initialization, health monitoring, and graceful shutdown
for all Phase 8 security services integrated with the Nautilus trading platform.
"""

import asyncio
import logging
import os
import signal
import sys
from contextlib import asynccontextmanager
from typing import Dict, List, Any, Optional, Set, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

# Import Phase 8 components
from phase8_autonomous_operations.security.cognitive_security_operations_center import (
    CognitiveSecurityOperationsCenter, ThreatSeverity, ThreatCategory
)
from phase8_autonomous_operations.threat_intelligence.advanced_threat_intelligence import (
    AdvancedThreatIntelligence, ThreatIntelligenceFeed
)
from phase8_autonomous_operations.security_response.autonomous_security_response import (
    AutonomousSecurityResponse, ResponseAction
)
from phase8_autonomous_operations.fraud_detection.intelligent_fraud_detection import (
    IntelligentFraudDetection, FraudRiskLevel
)
from phase8_autonomous_operations.security_orchestration.automated_security_orchestration import (
    AutomatedSecurityOrchestration, SecurityPlaybook
)

# Import existing Nautilus services
from messagebus_client import messagebus_client, MessageBusMessage, ConnectionState
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Service lifecycle status"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"
    RECOVERING = "recovering"


class HealthStatus(Enum):
    """Service health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ServiceMetrics:
    """Service performance and health metrics"""
    service_name: str
    status: ServiceStatus
    health: HealthStatus
    started_at: Optional[datetime] = None
    last_health_check: Optional[datetime] = None
    uptime_seconds: float = 0
    error_count: int = 0
    last_error: Optional[str] = None
    cpu_usage: float = 0.0
    memory_usage_mb: float = 0.0
    threat_alerts_processed: int = 0
    response_actions_executed: int = 0
    custom_metrics: Dict[str, Any] = field(default_factory=dict)


class Phase8Settings(BaseSettings):
    """Phase 8 configuration settings"""
    
    # Redis Configuration
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_db: int = Field(default=0, env="REDIS_DB")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    
    # Security Configuration
    enable_cognitive_security: bool = Field(default=True, env="ENABLE_COGNITIVE_SECURITY")
    enable_threat_intelligence: bool = Field(default=True, env="ENABLE_THREAT_INTELLIGENCE")
    enable_autonomous_response: bool = Field(default=True, env="ENABLE_AUTONOMOUS_RESPONSE")
    enable_fraud_detection: bool = Field(default=True, env="ENABLE_FRAUD_DETECTION")
    enable_security_orchestration: bool = Field(default=True, env="ENABLE_SECURITY_ORCHESTRATION")
    
    # Monitoring Configuration
    health_check_interval: int = Field(default=30, env="HEALTH_CHECK_INTERVAL")
    metrics_collection_interval: int = Field(default=60, env="METRICS_COLLECTION_INTERVAL")
    auto_recovery_enabled: bool = Field(default=True, env="AUTO_RECOVERY_ENABLED")
    max_recovery_attempts: int = Field(default=3, env="MAX_RECOVERY_ATTEMPTS")
    
    # Threat Intelligence Configuration
    threat_feed_update_interval: int = Field(default=300, env="THREAT_FEED_UPDATE_INTERVAL")  # 5 minutes
    threat_intelligence_sources: List[str] = Field(default_factory=lambda: [
        "misp", "virustotal", "alienvault", "threat_crowd", "internal"
    ], env="THREAT_INTELLIGENCE_SOURCES")
    
    # ML Model Configuration
    fraud_model_update_interval: int = Field(default=3600, env="FRAUD_MODEL_UPDATE_INTERVAL")  # 1 hour
    anomaly_detection_threshold: float = Field(default=0.8, env="ANOMALY_DETECTION_THRESHOLD")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    enable_security_audit_logs: bool = Field(default=True, env="ENABLE_SECURITY_AUDIT_LOGS")
    
    class Config:
        env_prefix = "PHASE8_"


class Phase8StartupService:
    """
    Comprehensive startup and lifecycle management for Phase 8 autonomous security operations.
    
    Manages initialization, monitoring, and shutdown of all security components
    with integration into the existing Nautilus infrastructure.
    """
    
    def __init__(self, settings: Optional[Phase8Settings] = None):
        self.settings = settings or Phase8Settings()
        self.services: Dict[str, Any] = {}
        self.service_metrics: Dict[str, ServiceMetrics] = {}
        self.background_tasks: Set[asyncio.Task] = set()
        self.recovery_attempts: Dict[str, int] = {}
        self.shutdown_event = asyncio.Event()
        self._running = False
        self._redis_client: Optional[redis.Redis] = None
        
        # Configure logging
        self._setup_logging()
        
        logger.info(f"Phase 8 Startup Service initialized with {len(self._get_enabled_services())} enabled services")
    
    def _setup_logging(self):
        """Configure logging for Phase 8 services"""
        log_level = getattr(logging, self.settings.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - [Phase8] %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('phase8_security.log') if self.settings.enable_security_audit_logs else logging.NullHandler()
            ]
        )
    
    def _get_enabled_services(self) -> List[str]:
        """Get list of enabled services based on configuration"""
        enabled_services = []
        if self.settings.enable_cognitive_security:
            enabled_services.append("cognitive_security")
        if self.settings.enable_threat_intelligence:
            enabled_services.append("threat_intelligence")
        if self.settings.enable_autonomous_response:
            enabled_services.append("autonomous_response")
        if self.settings.enable_fraud_detection:
            enabled_services.append("fraud_detection")
        if self.settings.enable_security_orchestration:
            enabled_services.append("security_orchestration")
        return enabled_services
    
    async def _setup_redis_connection(self) -> redis.Redis:
        """Setup Redis connection for Phase 8 services"""
        try:
            self._redis_client = redis.Redis(
                host=self.settings.redis_host,
                port=self.settings.redis_port,
                db=self.settings.redis_db,
                password=self.settings.redis_password,
                decode_responses=True
            )
            await self._redis_client.ping()
            logger.info("Phase 8 Redis connection established")
            return self._redis_client
        except Exception as e:
            logger.error(f"Failed to setup Redis connection: {e}")
            raise
    
    async def _initialize_cognitive_security(self) -> CognitiveSecurityOperationsCenter:
        """Initialize Cognitive Security Operations Center"""
        logger.info("Initializing Cognitive Security Operations Center...")
        
        try:
            csoc = CognitiveSecurityOperationsCenter(
                redis_client=self._redis_client,
                anomaly_threshold=self.settings.anomaly_detection_threshold,
                enable_ml_detection=True,
                enable_behavioral_analysis=True
            )
            
            await csoc.initialize()
            
            # Setup message handlers for security events
            messagebus_client.add_message_handler(csoc.handle_security_event)
            
            self.service_metrics["cognitive_security"] = ServiceMetrics(
                service_name="cognitive_security",
                status=ServiceStatus.RUNNING,
                health=HealthStatus.HEALTHY,
                started_at=datetime.utcnow()
            )
            
            logger.info("✅ Cognitive Security Operations Center initialized successfully")
            return csoc
            
        except Exception as e:
            logger.error(f"Failed to initialize Cognitive Security: {e}")
            self.service_metrics["cognitive_security"] = ServiceMetrics(
                service_name="cognitive_security",
                status=ServiceStatus.ERROR,
                health=HealthStatus.UNHEALTHY,
                last_error=str(e)
            )
            raise
    
    async def _initialize_threat_intelligence(self) -> AdvancedThreatIntelligence:
        """Initialize Advanced Threat Intelligence"""
        logger.info("Initializing Advanced Threat Intelligence...")
        
        try:
            threat_intel = AdvancedThreatIntelligence(
                redis_client=self._redis_client,
                intelligence_sources=self.settings.threat_intelligence_sources,
                update_interval=self.settings.threat_feed_update_interval
            )
            
            await threat_intel.initialize()
            
            self.service_metrics["threat_intelligence"] = ServiceMetrics(
                service_name="threat_intelligence",
                status=ServiceStatus.RUNNING,
                health=HealthStatus.HEALTHY,
                started_at=datetime.utcnow()
            )
            
            logger.info("✅ Advanced Threat Intelligence initialized successfully")
            return threat_intel
            
        except Exception as e:
            logger.error(f"Failed to initialize Threat Intelligence: {e}")
            self.service_metrics["threat_intelligence"] = ServiceMetrics(
                service_name="threat_intelligence",
                status=ServiceStatus.ERROR,
                health=HealthStatus.UNHEALTHY,
                last_error=str(e)
            )
            raise
    
    async def _initialize_autonomous_response(self) -> AutonomousSecurityResponse:
        """Initialize Autonomous Security Response"""
        logger.info("Initializing Autonomous Security Response...")
        
        try:
            autonomous_response = AutonomousSecurityResponse(
                redis_client=self._redis_client,
                enable_auto_mitigation=True,
                enable_adaptive_response=True,
                max_response_time_ms=5000  # 5 second response SLA
            )
            
            await autonomous_response.initialize()
            
            self.service_metrics["autonomous_response"] = ServiceMetrics(
                service_name="autonomous_response",
                status=ServiceStatus.RUNNING,
                health=HealthStatus.HEALTHY,
                started_at=datetime.utcnow()
            )
            
            logger.info("✅ Autonomous Security Response initialized successfully")
            return autonomous_response
            
        except Exception as e:
            logger.error(f"Failed to initialize Autonomous Response: {e}")
            self.service_metrics["autonomous_response"] = ServiceMetrics(
                service_name="autonomous_response",
                status=ServiceStatus.ERROR,
                health=HealthStatus.UNHEALTHY,
                last_error=str(e)
            )
            raise
    
    async def _initialize_fraud_detection(self) -> IntelligentFraudDetection:
        """Initialize Intelligent Fraud Detection"""
        logger.info("Initializing Intelligent Fraud Detection...")
        
        try:
            fraud_detection = IntelligentFraudDetection(
                redis_client=self._redis_client,
                model_update_interval=self.settings.fraud_model_update_interval,
                enable_real_time_scoring=True,
                enable_behavioral_profiling=True
            )
            
            await fraud_detection.initialize()
            
            # Load pre-trained models
            await fraud_detection.load_models()
            
            self.service_metrics["fraud_detection"] = ServiceMetrics(
                service_name="fraud_detection",
                status=ServiceStatus.RUNNING,
                health=HealthStatus.HEALTHY,
                started_at=datetime.utcnow()
            )
            
            logger.info("✅ Intelligent Fraud Detection initialized successfully")
            return fraud_detection
            
        except Exception as e:
            logger.error(f"Failed to initialize Fraud Detection: {e}")
            self.service_metrics["fraud_detection"] = ServiceMetrics(
                service_name="fraud_detection",
                status=ServiceStatus.ERROR,
                health=HealthStatus.UNHEALTHY,
                last_error=str(e)
            )
            raise
    
    async def _initialize_security_orchestration(self) -> AutomatedSecurityOrchestration:
        """Initialize Automated Security Orchestration"""
        logger.info("Initializing Automated Security Orchestration...")
        
        try:
            security_orchestration = AutomatedSecurityOrchestration(
                redis_client=self._redis_client,
                enable_automated_playbooks=True,
                enable_cross_service_coordination=True,
                orchestration_timeout_seconds=300  # 5 minute timeout
            )
            
            await security_orchestration.initialize()
            
            # Load security playbooks
            await security_orchestration.load_playbooks()
            
            self.service_metrics["security_orchestration"] = ServiceMetrics(
                service_name="security_orchestration",
                status=ServiceStatus.RUNNING,
                health=HealthStatus.HEALTHY,
                started_at=datetime.utcnow()
            )
            
            logger.info("✅ Automated Security Orchestration initialized successfully")
            return security_orchestration
            
        except Exception as e:
            logger.error(f"Failed to initialize Security Orchestration: {e}")
            self.service_metrics["security_orchestration"] = ServiceMetrics(
                service_name="security_orchestration",
                status=ServiceStatus.ERROR,
                health=HealthStatus.UNHEALTHY,
                last_error=str(e)
            )
            raise
    
    async def _start_background_monitoring(self):
        """Start background tasks for health monitoring and metrics collection"""
        logger.info("Starting background monitoring tasks...")
        
        # Health check task
        health_check_task = asyncio.create_task(self._health_check_loop())
        self.background_tasks.add(health_check_task)
        health_check_task.add_done_callback(self.background_tasks.discard)
        
        # Metrics collection task
        metrics_task = asyncio.create_task(self._metrics_collection_loop())
        self.background_tasks.add(metrics_task)
        metrics_task.add_done_callback(self.background_tasks.discard)
        
        # Auto-recovery task
        if self.settings.auto_recovery_enabled:
            recovery_task = asyncio.create_task(self._auto_recovery_loop())
            self.background_tasks.add(recovery_task)
            recovery_task.add_done_callback(self.background_tasks.discard)
        
        logger.info(f"Started {len(self.background_tasks)} background monitoring tasks")
    
    async def _health_check_loop(self):
        """Continuous health monitoring loop"""
        while not self.shutdown_event.is_set():
            try:
                for service_name, service in self.services.items():
                    if hasattr(service, 'health_check'):
                        health_status = await service.health_check()
                        if service_name in self.service_metrics:
                            self.service_metrics[service_name].health = health_status
                            self.service_metrics[service_name].last_health_check = datetime.utcnow()
                
                await asyncio.sleep(self.settings.health_check_interval)
                
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(self.settings.health_check_interval)
    
    async def _metrics_collection_loop(self):
        """Continuous metrics collection loop"""
        while not self.shutdown_event.is_set():
            try:
                for service_name, service in self.services.items():
                    if hasattr(service, 'get_metrics'):
                        metrics = await service.get_metrics()
                        if service_name in self.service_metrics:
                            self.service_metrics[service_name].custom_metrics.update(metrics)
                
                # Update uptime metrics
                current_time = datetime.utcnow()
                for metrics in self.service_metrics.values():
                    if metrics.started_at:
                        metrics.uptime_seconds = (current_time - metrics.started_at).total_seconds()
                
                await asyncio.sleep(self.settings.metrics_collection_interval)
                
            except Exception as e:
                logger.error(f"Metrics collection loop error: {e}")
                await asyncio.sleep(self.settings.metrics_collection_interval)
    
    async def _auto_recovery_loop(self):
        """Automatic service recovery loop"""
        while not self.shutdown_event.is_set():
            try:
                for service_name, metrics in self.service_metrics.items():
                    if metrics.health == HealthStatus.UNHEALTHY and metrics.status != ServiceStatus.RECOVERING:
                        recovery_attempts = self.recovery_attempts.get(service_name, 0)
                        
                        if recovery_attempts < self.settings.max_recovery_attempts:
                            logger.warning(f"Attempting recovery for unhealthy service: {service_name}")
                            await self._recover_service(service_name)
                            self.recovery_attempts[service_name] = recovery_attempts + 1
                        else:
                            logger.error(f"Max recovery attempts reached for service: {service_name}")
                
                await asyncio.sleep(self.settings.health_check_interval * 2)  # Check less frequently
                
            except Exception as e:
                logger.error(f"Auto recovery loop error: {e}")
                await asyncio.sleep(self.settings.health_check_interval * 2)
    
    async def _recover_service(self, service_name: str):
        """Recover a failed service"""
        try:
            self.service_metrics[service_name].status = ServiceStatus.RECOVERING
            logger.info(f"Recovering service: {service_name}")
            
            # Stop the service if it exists
            if service_name in self.services:
                service = self.services[service_name]
                if hasattr(service, 'stop'):
                    await service.stop()
            
            # Reinitialize the service
            await asyncio.sleep(2)  # Brief pause before restart
            
            if service_name == "cognitive_security" and self.settings.enable_cognitive_security:
                self.services[service_name] = await self._initialize_cognitive_security()
            elif service_name == "threat_intelligence" and self.settings.enable_threat_intelligence:
                self.services[service_name] = await self._initialize_threat_intelligence()
            elif service_name == "autonomous_response" and self.settings.enable_autonomous_response:
                self.services[service_name] = await self._initialize_autonomous_response()
            elif service_name == "fraud_detection" and self.settings.enable_fraud_detection:
                self.services[service_name] = await self._initialize_fraud_detection()
            elif service_name == "security_orchestration" and self.settings.enable_security_orchestration:
                self.services[service_name] = await self._initialize_security_orchestration()
            
            logger.info(f"✅ Service {service_name} recovered successfully")
            
        except Exception as e:
            logger.error(f"Failed to recover service {service_name}: {e}")
            self.service_metrics[service_name].status = ServiceStatus.ERROR
            self.service_metrics[service_name].last_error = str(e)
    
    async def startup(self):
        """Initialize all Phase 8 security services"""
        try:
            logger.info("🚀 Starting Phase 8 Autonomous Security Operations...")
            
            # Setup Redis connection
            await self._setup_redis_connection()
            
            # Initialize enabled services
            enabled_services = self._get_enabled_services()
            
            for service_name in enabled_services:
                try:
                    if service_name == "cognitive_security":
                        self.services[service_name] = await self._initialize_cognitive_security()
                    elif service_name == "threat_intelligence":
                        self.services[service_name] = await self._initialize_threat_intelligence()
                    elif service_name == "autonomous_response":
                        self.services[service_name] = await self._initialize_autonomous_response()
                    elif service_name == "fraud_detection":
                        self.services[service_name] = await self._initialize_fraud_detection()
                    elif service_name == "security_orchestration":
                        self.services[service_name] = await self._initialize_security_orchestration()
                        
                except Exception as e:
                    logger.error(f"Failed to initialize {service_name}: {e}")
                    # Continue with other services even if one fails
                    continue
            
            # Start background monitoring
            await self._start_background_monitoring()
            
            self._running = True
            
            # Setup signal handlers for graceful shutdown
            self._setup_signal_handlers()
            
            logger.info(f"✅ Phase 8 startup complete! {len(self.services)} services running")
            
        except Exception as e:
            logger.error(f"Phase 8 startup failed: {e}")
            raise
    
    async def shutdown(self):
        """Graceful shutdown of all Phase 8 services"""
        try:
            logger.info("🛑 Shutting down Phase 8 Autonomous Security Operations...")
            
            self._running = False
            self.shutdown_event.set()
            
            # Stop background tasks
            for task in self.background_tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for background tasks to complete
            if self.background_tasks:
                await asyncio.gather(*self.background_tasks, return_exceptions=True)
            
            # Stop services in reverse order
            service_names = list(self.services.keys())
            for service_name in reversed(service_names):
                try:
                    service = self.services[service_name]
                    if hasattr(service, 'shutdown'):
                        await service.shutdown()
                    elif hasattr(service, 'stop'):
                        await service.stop()
                    
                    self.service_metrics[service_name].status = ServiceStatus.STOPPED
                    logger.info(f"✅ Service {service_name} stopped")
                    
                except Exception as e:
                    logger.error(f"Error stopping {service_name}: {e}")
            
            # Close Redis connection
            if self._redis_client:
                await self._redis_client.aclose()
            
            logger.info("✅ Phase 8 shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during Phase 8 shutdown: {e}")
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    def get_service_status(self) -> Dict[str, ServiceMetrics]:
        """Get current status of all services"""
        return self.service_metrics.copy()
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        healthy_services = sum(1 for m in self.service_metrics.values() if m.health == HealthStatus.HEALTHY)
        total_services = len(self.service_metrics)
        
        overall_health = HealthStatus.HEALTHY
        if healthy_services == 0:
            overall_health = HealthStatus.UNHEALTHY
        elif healthy_services < total_services:
            overall_health = HealthStatus.DEGRADED
        
        return {
            "overall_health": overall_health.value,
            "healthy_services": healthy_services,
            "total_services": total_services,
            "uptime_seconds": max(m.uptime_seconds for m in self.service_metrics.values()) if self.service_metrics else 0,
            "last_health_check": max(m.last_health_check for m in self.service_metrics.values() if m.last_health_check) if self.service_metrics else None,
            "services": self.get_service_status()
        }
    
    async def handle_security_event(self, event: MessageBusMessage):
        """Handle security events from the message bus"""
        try:
            # Route security events to appropriate services
            if event.topic.startswith("security."):
                for service_name, service in self.services.items():
                    if hasattr(service, 'handle_security_event'):
                        await service.handle_security_event(event)
            
        except Exception as e:
            logger.error(f"Error handling security event: {e}")


# Global instance
phase8_startup_service = Phase8StartupService()


@asynccontextmanager
async def phase8_lifespan_context():
    """
    Async context manager for Phase 8 lifecycle management.
    Use this in FastAPI lifespan events for proper startup/shutdown.
    """
    try:
        # Startup
        await phase8_startup_service.startup()
        yield phase8_startup_service
    finally:
        # Shutdown
        await phase8_startup_service.shutdown()


# FastAPI integration functions
async def phase8_startup():
    """Startup function for FastAPI lifespan integration"""
    await phase8_startup_service.startup()


async def phase8_shutdown():
    """Shutdown function for FastAPI lifespan integration"""
    await phase8_startup_service.shutdown()


# Health check endpoints
def get_phase8_health():
    """Get Phase 8 system health for health check endpoints"""
    return phase8_startup_service.get_system_health()


def get_phase8_metrics():
    """Get detailed Phase 8 metrics for monitoring endpoints"""
    return {
        "system_health": phase8_startup_service.get_system_health(),
        "service_metrics": phase8_startup_service.get_service_status(),
        "configuration": {
            "enabled_services": phase8_startup_service._get_enabled_services(),
            "auto_recovery_enabled": phase8_startup_service.settings.auto_recovery_enabled,
            "health_check_interval": phase8_startup_service.settings.health_check_interval
        }
    }


# Message bus integration
async def setup_phase8_messagebus_handlers():
    """Setup Phase 8 message bus handlers"""
    if messagebus_client.is_connected:
        messagebus_client.add_message_handler(phase8_startup_service.handle_security_event)
        logger.info("Phase 8 message bus handlers configured")
    else:
        logger.warning("Message bus not connected, Phase 8 handlers not configured")


if __name__ == "__main__":
    """
    Standalone Phase 8 service runner for testing and development.
    In production, use FastAPI integration via phase8_lifespan_context.
    """
    async def main():
        try:
            service = Phase8StartupService()
            await service.startup()
            
            # Keep running until interrupted
            while service._running:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        finally:
            await service.shutdown()
    
    asyncio.run(main())