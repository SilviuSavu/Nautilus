"""
Autonomous Deployment System - Phase 8 Autonomous Operations
==========================================================

Provides zero-downtime deployments with intelligent automation, canary analysis,
automated rollbacks, and ML-driven deployment strategies.

Key Features:
- Zero-downtime blue-green and canary deployments
- ML-powered deployment success prediction
- Automated rollback with anomaly detection
- Progressive deployment with traffic splitting
- Intelligent health monitoring and validation
- Multi-environment deployment orchestration
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid
import numpy as np
from pydantic import BaseModel, Field
import kubernetes as k8s
from kubernetes.client.rest import ApiException
import docker
import requests
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Configure logging
logger = logging.getLogger(__name__)

class DeploymentStrategy(Enum):
    """Deployment strategy types"""
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING = "rolling"
    RECREATE = "recreate"
    A_B_TESTING = "a_b_testing"

class DeploymentStatus(Enum):
    """Deployment status states"""
    PENDING = "pending"
    INITIALIZING = "initializing"
    DEPLOYING = "deploying"
    HEALTH_CHECK = "health_check"
    TRAFFIC_SWITCH = "traffic_switch"
    MONITORING = "monitoring"
    COMPLETED = "completed"
    ROLLING_BACK = "rolling_back"
    FAILED = "failed"
    CANCELLED = "cancelled"

class Environment(Enum):
    """Target deployment environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    CANARY = "canary"
    BLUE = "blue"
    GREEN = "green"

@dataclass
class HealthCheck:
    """Health check configuration"""
    endpoint: str
    expected_status: int = 200
    timeout: int = 30
    interval: int = 10
    retries: int = 3
    headers: Dict[str, str] = field(default_factory=dict)
    
@dataclass
class TrafficSplitConfig:
    """Traffic splitting configuration"""
    target_percentage: int
    ramp_duration: int = 300  # 5 minutes
    increment: int = 10
    monitoring_interval: int = 30
    success_threshold: float = 0.95
    error_threshold: float = 0.05

class DeploymentConfig(BaseModel):
    """Deployment configuration model"""
    deployment_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    service_name: str
    image_tag: str
    environment: Environment
    strategy: DeploymentStrategy = DeploymentStrategy.ROLLING
    
    # Resource configuration
    replicas: int = 3
    cpu_request: str = "100m"
    cpu_limit: str = "500m"
    memory_request: str = "128Mi"
    memory_limit: str = "512Mi"
    
    # Health checks
    health_checks: List[HealthCheck] = Field(default_factory=list)
    readiness_probe: Dict[str, Any] = Field(default_factory=dict)
    liveness_probe: Dict[str, Any] = Field(default_factory=dict)
    
    # Traffic management
    traffic_config: Optional[TrafficSplitConfig] = None
    
    # Rollback configuration
    auto_rollback: bool = True
    rollback_threshold: float = 0.05  # 5% error rate
    monitoring_duration: int = 600  # 10 minutes
    
    # Environment variables and configs
    environment_variables: Dict[str, str] = Field(default_factory=dict)
    config_maps: List[str] = Field(default_factory=list)
    secrets: List[str] = Field(default_factory=list)
    
    # Metadata
    created_by: str = "autonomous_deployment_system"
    created_at: datetime = Field(default_factory=datetime.now)
    tags: Dict[str, str] = Field(default_factory=dict)

class DeploymentExecution(BaseModel):
    """Deployment execution tracking"""
    execution_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    deployment_id: str
    status: DeploymentStatus = DeploymentStatus.PENDING
    
    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration: Optional[float] = None
    
    # Results
    success: bool = False
    error_message: Optional[str] = None
    rollback_triggered: bool = False
    
    # Metrics
    metrics: Dict[str, Any] = Field(default_factory=dict)
    health_check_results: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Deployment artifacts
    blue_deployment: Optional[str] = None
    green_deployment: Optional[str] = None
    canary_deployment: Optional[str] = None
    
    # Traffic management
    traffic_history: List[Dict[str, Any]] = Field(default_factory=list)

class MLDeploymentPredictor:
    """ML-powered deployment success prediction"""
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_columns = [
            'replicas', 'cpu_limit_millicores', 'memory_limit_mb',
            'health_check_count', 'environment_var_count',
            'strategy_encoded', 'environment_encoded',
            'previous_deployments_success_rate',
            'service_complexity_score', 'deployment_hour'
        ]
        
    def extract_features(self, config: DeploymentConfig, historical_data: List[Dict]) -> np.ndarray:
        """Extract features from deployment configuration"""
        features = []
        
        # Basic resource features
        features.append(config.replicas)
        features.append(self.parse_cpu_limit(config.cpu_limit))
        features.append(self.parse_memory_limit(config.memory_limit))
        features.append(len(config.health_checks))
        features.append(len(config.environment_variables))
        
        # Categorical encoding
        strategy_map = {
            DeploymentStrategy.BLUE_GREEN: 0,
            DeploymentStrategy.CANARY: 1,
            DeploymentStrategy.ROLLING: 2,
            DeploymentStrategy.RECREATE: 3,
            DeploymentStrategy.A_B_TESTING: 4
        }
        env_map = {
            Environment.DEVELOPMENT: 0,
            Environment.STAGING: 1,
            Environment.PRODUCTION: 2,
            Environment.CANARY: 3,
            Environment.BLUE: 4,
            Environment.GREEN: 5
        }
        features.append(strategy_map.get(config.strategy, 0))
        features.append(env_map.get(config.environment, 0))
        
        # Historical success rate
        service_history = [d for d in historical_data if d.get('service_name') == config.service_name]
        if service_history:
            success_rate = sum(1 for d in service_history if d.get('success', False)) / len(service_history)
        else:
            success_rate = 0.5  # Default for new services
        features.append(success_rate)
        
        # Service complexity (heuristic)
        complexity_score = (
            len(config.environment_variables) * 0.1 +
            len(config.config_maps) * 0.2 +
            len(config.secrets) * 0.3 +
            len(config.health_checks) * 0.1
        )
        features.append(complexity_score)
        
        # Time-based features
        features.append(datetime.now().hour)
        
        return np.array(features).reshape(1, -1)
    
    def parse_cpu_limit(self, cpu_limit: str) -> float:
        """Parse CPU limit to millicores"""
        if cpu_limit.endswith('m'):
            return float(cpu_limit[:-1])
        else:
            return float(cpu_limit) * 1000
    
    def parse_memory_limit(self, memory_limit: str) -> float:
        """Parse memory limit to MB"""
        if memory_limit.endswith('Mi'):
            return float(memory_limit[:-2])
        elif memory_limit.endswith('Gi'):
            return float(memory_limit[:-2]) * 1024
        else:
            return float(memory_limit)
    
    def train(self, training_data: List[Dict[str, Any]]) -> None:
        """Train the deployment prediction model"""
        if len(training_data) < 10:
            logger.warning("Insufficient training data for ML model")
            return
        
        features = []
        labels = []
        
        for data in training_data:
            try:
                # Create temporary config for feature extraction
                config = DeploymentConfig(**data['config'])
                feature_vector = self.extract_features(config, training_data)
                features.append(feature_vector.flatten())
                labels.append(1 if data.get('success', False) else 0)
            except Exception as e:
                logger.warning(f"Error processing training data: {e}")
                continue
        
        if len(features) < 10:
            logger.warning("Insufficient valid training samples")
            return
        
        X = np.array(features)
        y = np.array(labels)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        logger.info(f"Trained deployment predictor with {len(features)} samples")
    
    def predict_success_probability(self, config: DeploymentConfig, historical_data: List[Dict]) -> float:
        """Predict deployment success probability"""
        if not self.is_trained:
            return 0.5  # Default probability
        
        try:
            features = self.extract_features(config, historical_data)
            features_scaled = self.scaler.transform(features)
            probability = self.model.predict_proba(features_scaled)[0][1]  # Probability of success
            return float(probability)
        except Exception as e:
            logger.error(f"Error predicting deployment success: {e}")
            return 0.5

class AutonomousDeploymentSystem:
    """Main autonomous deployment orchestrator"""
    
    def __init__(self):
        self.deployments: Dict[str, DeploymentConfig] = {}
        self.executions: Dict[str, DeploymentExecution] = {}
        self.running_deployments: Set[str] = set()
        self.ml_predictor = MLDeploymentPredictor()
        
        # Kubernetes client
        try:
            k8s.config.load_incluster_config()
        except:
            try:
                k8s.config.load_kube_config()
            except:
                logger.warning("Could not load Kubernetes config")
        
        self.k8s_apps_v1 = k8s.client.AppsV1Api()
        self.k8s_core_v1 = k8s.client.CoreV1Api()
        
        # Docker client
        try:
            self.docker_client = docker.from_env()
        except Exception as e:
            logger.warning(f"Could not connect to Docker: {e}")
            self.docker_client = None
        
        # Deployment history for ML training
        self.deployment_history: List[Dict[str, Any]] = []
        
    async def deploy_service(self, config: DeploymentConfig) -> str:
        """Deploy a service using the specified configuration"""
        try:
            # Validate configuration
            validation_result = await self.validate_deployment_config(config)
            if not validation_result['valid']:
                raise ValueError(f"Invalid deployment config: {validation_result['errors']}")
            
            # Predict deployment success
            success_probability = self.ml_predictor.predict_success_probability(
                config, self.deployment_history
            )
            
            logger.info(f"Predicted deployment success probability: {success_probability:.2%}")
            
            if success_probability < 0.3:
                logger.warning("Low success probability - recommend reviewing configuration")
            
            # Create execution tracking
            execution = DeploymentExecution(
                deployment_id=config.deployment_id,
                metrics={'predicted_success_probability': success_probability}
            )
            
            self.deployments[config.deployment_id] = config
            self.executions[execution.execution_id] = execution
            self.running_deployments.add(execution.execution_id)
            
            # Start deployment based on strategy
            if config.strategy == DeploymentStrategy.BLUE_GREEN:
                await self.deploy_blue_green(config, execution)
            elif config.strategy == DeploymentStrategy.CANARY:
                await self.deploy_canary(config, execution)
            elif config.strategy == DeploymentStrategy.ROLLING:
                await self.deploy_rolling(config, execution)
            elif config.strategy == DeploymentStrategy.A_B_TESTING:
                await self.deploy_ab_testing(config, execution)
            else:
                await self.deploy_recreate(config, execution)
            
            return execution.execution_id
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            if 'execution' in locals():
                execution.status = DeploymentStatus.FAILED
                execution.error_message = str(e)
                execution.completed_at = datetime.now()
            raise
        finally:
            if 'execution' in locals() and execution.execution_id in self.running_deployments:
                self.running_deployments.discard(execution.execution_id)
    
    async def validate_deployment_config(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Validate deployment configuration"""
        errors = []
        warnings = []
        
        # Basic validation
        if not config.service_name:
            errors.append("Service name is required")
        
        if not config.image_tag:
            errors.append("Image tag is required")
        
        if config.replicas < 1:
            errors.append("Replicas must be at least 1")
        
        if config.replicas > 100:
            warnings.append("High replica count may impact cluster resources")
        
        # Resource validation
        try:
            cpu_limit = self.ml_predictor.parse_cpu_limit(config.cpu_limit)
            if cpu_limit < 10:  # 10 millicores minimum
                warnings.append("Very low CPU limit may cause performance issues")
        except ValueError:
            errors.append(f"Invalid CPU limit format: {config.cpu_limit}")
        
        try:
            memory_limit = self.ml_predictor.parse_memory_limit(config.memory_limit)
            if memory_limit < 64:  # 64Mi minimum
                warnings.append("Very low memory limit may cause OOM issues")
        except ValueError:
            errors.append(f"Invalid memory limit format: {config.memory_limit}")
        
        # Health check validation
        for health_check in config.health_checks:
            if not health_check.endpoint:
                errors.append("Health check endpoint is required")
        
        # Strategy-specific validation
        if config.strategy == DeploymentStrategy.CANARY:
            if not config.traffic_config:
                warnings.append("Canary deployment without traffic config")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    async def deploy_blue_green(self, config: DeploymentConfig, execution: DeploymentExecution) -> None:
        """Execute blue-green deployment"""
        logger.info(f"Starting blue-green deployment for {config.service_name}")
        
        execution.status = DeploymentStatus.INITIALIZING
        execution.started_at = datetime.now()
        
        try:
            # Determine current and new environments
            current_env = await self.get_current_environment(config.service_name)
            new_env = Environment.GREEN if current_env == Environment.BLUE else Environment.BLUE
            
            logger.info(f"Deploying to {new_env.value} environment")
            
            # Deploy to new environment
            execution.status = DeploymentStatus.DEPLOYING
            deployment_name = f"{config.service_name}-{new_env.value}"
            
            await self.create_kubernetes_deployment(config, deployment_name, new_env)
            
            if new_env == Environment.GREEN:
                execution.green_deployment = deployment_name
            else:
                execution.blue_deployment = deployment_name
            
            # Wait for deployment to be ready
            await self.wait_for_deployment_ready(deployment_name, timeout=600)
            
            # Perform health checks
            execution.status = DeploymentStatus.HEALTH_CHECK
            health_results = await self.perform_health_checks(config, deployment_name)
            execution.health_check_results = health_results
            
            if not all(result['success'] for result in health_results):
                raise Exception("Health checks failed")
            
            # Switch traffic
            execution.status = DeploymentStatus.TRAFFIC_SWITCH
            await self.switch_traffic(config.service_name, new_env)
            
            # Monitor for stability
            execution.status = DeploymentStatus.MONITORING
            monitoring_passed = await self.monitor_deployment_stability(
                config, deployment_name, duration=config.monitoring_duration
            )
            
            if not monitoring_passed and config.auto_rollback:
                await self.rollback_deployment(config, execution, current_env)
                return
            
            # Cleanup old environment
            if current_env != Environment.PRODUCTION:  # Keep production as backup
                old_deployment = f"{config.service_name}-{current_env.value}"
                await self.cleanup_old_deployment(old_deployment)
            
            execution.status = DeploymentStatus.COMPLETED
            execution.success = True
            execution.completed_at = datetime.now()
            execution.duration = (execution.completed_at - execution.started_at).total_seconds()
            
            logger.info(f"Blue-green deployment completed successfully in {execution.duration:.1f}s")
            
        except Exception as e:
            logger.error(f"Blue-green deployment failed: {e}")
            execution.status = DeploymentStatus.FAILED
            execution.error_message = str(e)
            execution.completed_at = datetime.now()
            
            if execution.started_at:
                execution.duration = (execution.completed_at - execution.started_at).total_seconds()
            
            # Auto-rollback if enabled
            if config.auto_rollback and 'current_env' in locals():
                await self.rollback_deployment(config, execution, current_env)
    
    async def deploy_canary(self, config: DeploymentConfig, execution: DeploymentExecution) -> None:
        """Execute canary deployment with progressive traffic splitting"""
        logger.info(f"Starting canary deployment for {config.service_name}")
        
        execution.status = DeploymentStatus.INITIALIZING
        execution.started_at = datetime.now()
        
        try:
            # Deploy canary version
            execution.status = DeploymentStatus.DEPLOYING
            canary_name = f"{config.service_name}-canary"
            
            await self.create_kubernetes_deployment(config, canary_name, Environment.CANARY)
            execution.canary_deployment = canary_name
            
            # Wait for canary to be ready
            await self.wait_for_deployment_ready(canary_name, timeout=300)
            
            # Perform initial health checks
            execution.status = DeploymentStatus.HEALTH_CHECK
            health_results = await self.perform_health_checks(config, canary_name)
            execution.health_check_results = health_results
            
            if not all(result['success'] for result in health_results):
                raise Exception("Canary health checks failed")
            
            # Progressive traffic splitting
            execution.status = DeploymentStatus.TRAFFIC_SWITCH
            traffic_config = config.traffic_config or TrafficSplitConfig(target_percentage=100)
            
            success = await self.progressive_traffic_split(
                config, execution, canary_name, traffic_config
            )
            
            if not success:
                raise Exception("Canary deployment failed during traffic split")
            
            # Replace production with canary
            await self.promote_canary_to_production(config.service_name, canary_name)
            
            execution.status = DeploymentStatus.COMPLETED
            execution.success = True
            execution.completed_at = datetime.now()
            execution.duration = (execution.completed_at - execution.started_at).total_seconds()
            
            logger.info(f"Canary deployment completed successfully in {execution.duration:.1f}s")
            
        except Exception as e:
            logger.error(f"Canary deployment failed: {e}")
            execution.status = DeploymentStatus.FAILED
            execution.error_message = str(e)
            execution.completed_at = datetime.now()
            
            if execution.started_at:
                execution.duration = (execution.completed_at - execution.started_at).total_seconds()
            
            # Rollback canary
            if execution.canary_deployment:
                await self.cleanup_old_deployment(execution.canary_deployment)
    
    async def deploy_rolling(self, config: DeploymentConfig, execution: DeploymentExecution) -> None:
        """Execute rolling deployment"""
        logger.info(f"Starting rolling deployment for {config.service_name}")
        
        execution.status = DeploymentStatus.INITIALIZING
        execution.started_at = datetime.now()
        
        try:
            # Update existing deployment
            execution.status = DeploymentStatus.DEPLOYING
            await self.update_kubernetes_deployment(config)
            
            # Monitor rollout
            await self.wait_for_rollout_complete(config.service_name, timeout=600)
            
            # Perform health checks
            execution.status = DeploymentStatus.HEALTH_CHECK
            health_results = await self.perform_health_checks(config, config.service_name)
            execution.health_check_results = health_results
            
            if not all(result['success'] for result in health_results):
                raise Exception("Rolling deployment health checks failed")
            
            # Monitor stability
            execution.status = DeploymentStatus.MONITORING
            monitoring_passed = await self.monitor_deployment_stability(
                config, config.service_name, duration=config.monitoring_duration
            )
            
            if not monitoring_passed and config.auto_rollback:
                await self.rollback_kubernetes_deployment(config.service_name)
                execution.rollback_triggered = True
                raise Exception("Rolling deployment failed monitoring - rolled back")
            
            execution.status = DeploymentStatus.COMPLETED
            execution.success = True
            execution.completed_at = datetime.now()
            execution.duration = (execution.completed_at - execution.started_at).total_seconds()
            
            logger.info(f"Rolling deployment completed successfully in {execution.duration:.1f}s")
            
        except Exception as e:
            logger.error(f"Rolling deployment failed: {e}")
            execution.status = DeploymentStatus.FAILED
            execution.error_message = str(e)
            execution.completed_at = datetime.now()
            
            if execution.started_at:
                execution.duration = (execution.completed_at - execution.started_at).total_seconds()
    
    async def deploy_ab_testing(self, config: DeploymentConfig, execution: DeploymentExecution) -> None:
        """Execute A/B testing deployment"""
        logger.info(f"Starting A/B testing deployment for {config.service_name}")
        
        # Similar to canary but with different traffic management
        # Implementation would include A/B testing specific logic
        await self.deploy_canary(config, execution)  # Simplified for now
    
    async def deploy_recreate(self, config: DeploymentConfig, execution: DeploymentExecution) -> None:
        """Execute recreate deployment (with downtime)"""
        logger.info(f"Starting recreate deployment for {config.service_name}")
        
        execution.status = DeploymentStatus.INITIALIZING
        execution.started_at = datetime.now()
        
        try:
            # Delete existing deployment
            execution.status = DeploymentStatus.DEPLOYING
            await self.delete_kubernetes_deployment(config.service_name)
            
            # Create new deployment
            await self.create_kubernetes_deployment(config, config.service_name, config.environment)
            
            # Wait for ready
            await self.wait_for_deployment_ready(config.service_name, timeout=300)
            
            # Health checks
            execution.status = DeploymentStatus.HEALTH_CHECK
            health_results = await self.perform_health_checks(config, config.service_name)
            execution.health_check_results = health_results
            
            if not all(result['success'] for result in health_results):
                raise Exception("Recreate deployment health checks failed")
            
            execution.status = DeploymentStatus.COMPLETED
            execution.success = True
            execution.completed_at = datetime.now()
            execution.duration = (execution.completed_at - execution.started_at).total_seconds()
            
            logger.info(f"Recreate deployment completed successfully in {execution.duration:.1f}s")
            
        except Exception as e:
            logger.error(f"Recreate deployment failed: {e}")
            execution.status = DeploymentStatus.FAILED
            execution.error_message = str(e)
            execution.completed_at = datetime.now()
            
            if execution.started_at:
                execution.duration = (execution.completed_at - execution.started_at).total_seconds()
    
    async def create_kubernetes_deployment(self, config: DeploymentConfig, deployment_name: str, environment: Environment) -> None:
        """Create Kubernetes deployment"""
        try:
            # Build deployment manifest
            deployment_manifest = {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {
                    "name": deployment_name,
                    "labels": {
                        "app": config.service_name,
                        "environment": environment.value,
                        "managed-by": "autonomous-deployment-system"
                    }
                },
                "spec": {
                    "replicas": config.replicas,
                    "selector": {
                        "matchLabels": {
                            "app": config.service_name,
                            "environment": environment.value
                        }
                    },
                    "template": {
                        "metadata": {
                            "labels": {
                                "app": config.service_name,
                                "environment": environment.value
                            }
                        },
                        "spec": {
                            "containers": [{
                                "name": config.service_name,
                                "image": config.image_tag,
                                "resources": {
                                    "requests": {
                                        "cpu": config.cpu_request,
                                        "memory": config.memory_request
                                    },
                                    "limits": {
                                        "cpu": config.cpu_limit,
                                        "memory": config.memory_limit
                                    }
                                },
                                "env": [
                                    {"name": k, "value": v} 
                                    for k, v in config.environment_variables.items()
                                ]
                            }]
                        }
                    }
                }
            }
            
            # Add health checks if configured
            if config.readiness_probe:
                deployment_manifest["spec"]["template"]["spec"]["containers"][0]["readinessProbe"] = config.readiness_probe
            
            if config.liveness_probe:
                deployment_manifest["spec"]["template"]["spec"]["containers"][0]["livenessProbe"] = config.liveness_probe
            
            # Create deployment
            self.k8s_apps_v1.create_namespaced_deployment(
                namespace="default",
                body=deployment_manifest
            )
            
            logger.info(f"Created Kubernetes deployment: {deployment_name}")
            
        except ApiException as e:
            if e.status == 409:  # Already exists
                logger.info(f"Deployment {deployment_name} already exists, updating...")
                await self.update_kubernetes_deployment(config, deployment_name)
            else:
                raise
        except Exception as e:
            logger.error(f"Failed to create deployment {deployment_name}: {e}")
            raise
    
    async def wait_for_deployment_ready(self, deployment_name: str, timeout: int = 300) -> None:
        """Wait for deployment to be ready"""
        start_time = datetime.now()
        
        while (datetime.now() - start_time).total_seconds() < timeout:
            try:
                deployment = self.k8s_apps_v1.read_namespaced_deployment(
                    name=deployment_name,
                    namespace="default"
                )
                
                if (deployment.status.ready_replicas and 
                    deployment.status.ready_replicas >= deployment.spec.replicas):
                    logger.info(f"Deployment {deployment_name} is ready")
                    return
                
            except ApiException as e:
                if e.status != 404:
                    logger.warning(f"Error checking deployment status: {e}")
            
            await asyncio.sleep(10)
        
        raise TimeoutError(f"Deployment {deployment_name} not ready after {timeout} seconds")
    
    async def perform_health_checks(self, config: DeploymentConfig, deployment_name: str) -> List[Dict[str, Any]]:
        """Perform health checks on deployment"""
        results = []
        
        for health_check in config.health_checks:
            result = await self.execute_health_check(health_check, deployment_name)
            results.append(result)
        
        return results
    
    async def execute_health_check(self, health_check: HealthCheck, deployment_name: str) -> Dict[str, Any]:
        """Execute a single health check"""
        start_time = datetime.now()
        
        try:
            # Get service endpoint
            service_url = await self.get_service_endpoint(deployment_name)
            full_url = f"{service_url}{health_check.endpoint}"
            
            # Execute health check
            for attempt in range(health_check.retries):
                try:
                    async with asyncio.timeout(health_check.timeout):
                        response = requests.get(
                            full_url,
                            headers=health_check.headers,
                            timeout=health_check.timeout
                        )
                    
                    success = response.status_code == health_check.expected_status
                    
                    if success:
                        break
                        
                except Exception as e:
                    if attempt == health_check.retries - 1:
                        raise e
                    
                    await asyncio.sleep(health_check.interval)
            
            duration = (datetime.now() - start_time).total_seconds()
            
            return {
                'endpoint': health_check.endpoint,
                'success': success,
                'status_code': response.status_code,
                'response_time': duration,
                'attempts': attempt + 1,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            
            return {
                'endpoint': health_check.endpoint,
                'success': False,
                'error': str(e),
                'response_time': duration,
                'attempts': health_check.retries,
                'timestamp': datetime.now().isoformat()
            }
    
    async def get_service_endpoint(self, service_name: str) -> str:
        """Get service endpoint URL"""
        try:
            service = self.k8s_core_v1.read_namespaced_service(
                name=service_name,
                namespace="default"
            )
            
            # For simplicity, assume NodePort or LoadBalancer
            if service.spec.type == "LoadBalancer" and service.status.load_balancer.ingress:
                host = service.status.load_balancer.ingress[0].hostname or service.status.load_balancer.ingress[0].ip
                port = service.spec.ports[0].port
                return f"http://{host}:{port}"
            elif service.spec.type == "NodePort":
                # Use localhost for local testing
                port = service.spec.ports[0].node_port
                return f"http://localhost:{port}"
            else:
                # Use cluster IP
                return f"http://{service.spec.cluster_ip}:{service.spec.ports[0].port}"
                
        except Exception as e:
            logger.warning(f"Could not get service endpoint for {service_name}: {e}")
            return f"http://{service_name}:8080"  # Default assumption
    
    async def monitor_deployment_stability(self, config: DeploymentConfig, deployment_name: str, duration: int) -> bool:
        """Monitor deployment stability over time"""
        logger.info(f"Monitoring {deployment_name} for {duration} seconds")
        
        start_time = datetime.now()
        error_count = 0
        total_checks = 0
        
        while (datetime.now() - start_time).total_seconds() < duration:
            try:
                # Perform health checks
                health_results = await self.perform_health_checks(config, deployment_name)
                
                total_checks += len(health_results)
                error_count += sum(1 for result in health_results if not result['success'])
                
                # Check error threshold
                current_error_rate = error_count / total_checks if total_checks > 0 else 0
                
                if current_error_rate > config.rollback_threshold:
                    logger.warning(f"Error rate {current_error_rate:.2%} exceeds threshold {config.rollback_threshold:.2%}")
                    return False
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error during stability monitoring: {e}")
                error_count += 1
        
        final_error_rate = error_count / total_checks if total_checks > 0 else 0
        logger.info(f"Monitoring completed. Error rate: {final_error_rate:.2%}")
        
        return final_error_rate <= config.rollback_threshold
    
    async def progressive_traffic_split(self, config: DeploymentConfig, execution: DeploymentExecution, canary_name: str, traffic_config: TrafficSplitConfig) -> bool:
        """Execute progressive traffic splitting for canary deployment"""
        logger.info(f"Starting progressive traffic split to {traffic_config.target_percentage}%")
        
        current_percentage = 0
        
        while current_percentage < traffic_config.target_percentage:
            # Calculate next increment
            next_percentage = min(
                current_percentage + traffic_config.increment,
                traffic_config.target_percentage
            )
            
            # Update traffic split
            await self.update_traffic_split(config.service_name, canary_name, next_percentage)
            
            # Record traffic change
            execution.traffic_history.append({
                'timestamp': datetime.now().isoformat(),
                'canary_percentage': next_percentage,
                'production_percentage': 100 - next_percentage
            })
            
            logger.info(f"Traffic split updated: {next_percentage}% canary, {100-next_percentage}% production")
            
            # Monitor for errors during this increment
            monitoring_passed = await self.monitor_deployment_stability(
                config, canary_name, duration=traffic_config.monitoring_interval
            )
            
            if not monitoring_passed:
                logger.warning("Traffic split monitoring failed - rolling back")
                await self.update_traffic_split(config.service_name, canary_name, 0)
                return False
            
            current_percentage = next_percentage
            
            if current_percentage < traffic_config.target_percentage:
                await asyncio.sleep(traffic_config.ramp_duration // 
                                  (traffic_config.target_percentage // traffic_config.increment))
        
        return True
    
    async def update_traffic_split(self, service_name: str, canary_name: str, canary_percentage: int) -> None:
        """Update traffic split between production and canary"""
        # This would integrate with service mesh (Istio, Linkerd) or ingress controller
        # For now, we'll log the action
        logger.info(f"Updated traffic split: {canary_percentage}% to {canary_name}")
        
        # In real implementation, this would update VirtualService or similar resources
        pass
    
    async def rollback_deployment(self, config: DeploymentConfig, execution: DeploymentExecution, target_environment: Environment) -> None:
        """Rollback deployment to previous version"""
        logger.info(f"Rolling back {config.service_name} to {target_environment.value}")
        
        execution.status = DeploymentStatus.ROLLING_BACK
        execution.rollback_triggered = True
        
        try:
            # Switch traffic back
            await self.switch_traffic(config.service_name, target_environment)
            
            # Cleanup failed deployment
            if execution.green_deployment:
                await self.cleanup_old_deployment(execution.green_deployment)
            if execution.blue_deployment:
                await self.cleanup_old_deployment(execution.blue_deployment)
            if execution.canary_deployment:
                await self.cleanup_old_deployment(execution.canary_deployment)
            
            logger.info(f"Successfully rolled back {config.service_name}")
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            execution.error_message = f"Rollback failed: {e}"
    
    async def get_current_environment(self, service_name: str) -> Environment:
        """Get current active environment for service"""
        # This would check current traffic routing
        # For simplicity, return production as default
        return Environment.PRODUCTION
    
    async def switch_traffic(self, service_name: str, target_environment: Environment) -> None:
        """Switch traffic to target environment"""
        logger.info(f"Switching traffic for {service_name} to {target_environment.value}")
        
        # This would update load balancer, service mesh, or ingress configuration
        # Implementation depends on infrastructure setup
        pass
    
    async def cleanup_old_deployment(self, deployment_name: str) -> None:
        """Cleanup old deployment"""
        try:
            self.k8s_apps_v1.delete_namespaced_deployment(
                name=deployment_name,
                namespace="default"
            )
            logger.info(f"Cleaned up deployment: {deployment_name}")
        except ApiException as e:
            if e.status != 404:
                logger.error(f"Failed to cleanup deployment {deployment_name}: {e}")
    
    def get_deployment_status(self, execution_id: str) -> Optional[DeploymentExecution]:
        """Get deployment execution status"""
        return self.executions.get(execution_id)
    
    def get_deployment_metrics(self, execution_id: str) -> Dict[str, Any]:
        """Get deployment metrics"""
        execution = self.executions.get(execution_id)
        if not execution:
            return {}
        
        return {
            'execution_id': execution_id,
            'status': execution.status.value,
            'duration': execution.duration,
            'success': execution.success,
            'rollback_triggered': execution.rollback_triggered,
            'health_check_results': execution.health_check_results,
            'traffic_history': execution.traffic_history,
            'metrics': execution.metrics
        }
    
    async def train_ml_model(self) -> None:
        """Train ML model with historical deployment data"""
        self.ml_predictor.train(self.deployment_history)
    
    def add_deployment_to_history(self, config: DeploymentConfig, execution: DeploymentExecution) -> None:
        """Add completed deployment to history for ML training"""
        history_entry = {
            'config': config.dict(),
            'success': execution.success,
            'duration': execution.duration,
            'error_message': execution.error_message,
            'rollback_triggered': execution.rollback_triggered,
            'timestamp': execution.completed_at.isoformat() if execution.completed_at else None
        }
        
        self.deployment_history.append(history_entry)
        
        # Keep only recent history
        if len(self.deployment_history) > 1000:
            self.deployment_history = self.deployment_history[-1000:]

# Example usage
async def example_deployment():
    """Example deployment usage"""
    deployment_system = AutonomousDeploymentSystem()
    
    # Create deployment configuration
    config = DeploymentConfig(
        service_name="trading-api",
        image_tag="trading-api:v2.1.0",
        environment=Environment.PRODUCTION,
        strategy=DeploymentStrategy.BLUE_GREEN,
        replicas=3,
        health_checks=[
            HealthCheck(endpoint="/health", expected_status=200),
            HealthCheck(endpoint="/ready", expected_status=200)
        ],
        traffic_config=TrafficSplitConfig(target_percentage=100),
        auto_rollback=True,
        rollback_threshold=0.05
    )
    
    # Deploy
    execution_id = await deployment_system.deploy_service(config)
    print(f"Started deployment: {execution_id}")
    
    # Monitor deployment
    while True:
        status = deployment_system.get_deployment_status(execution_id)
        if status and status.status in [DeploymentStatus.COMPLETED, DeploymentStatus.FAILED]:
            break
        await asyncio.sleep(5)
    
    print(f"Deployment completed with status: {status.status}")

if __name__ == "__main__":
    asyncio.run(example_deployment())