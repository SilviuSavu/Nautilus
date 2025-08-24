"""
Smart Configuration Manager - Phase 8 Autonomous Operations
==========================================================

Provides intelligent configuration management with drift detection, automatic
remediation, version control, and AI-driven configuration optimization.

Key Features:
- Configuration drift detection and automatic remediation
- AI-powered configuration optimization and validation
- Version control with automatic rollback capabilities
- Environment-specific configuration management
- Real-time configuration monitoring and alerting
- Compliance checking and security validation
"""

import asyncio
import json
import logging
import hashlib
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid
from pathlib import Path
import yaml
import git
from deepdiff import DeepDiff
import jsonschema
from jsonschema import validate, ValidationError
from pydantic import BaseModel, Field
import kubernetes as k8s
from kubernetes.client.rest import ApiException
import consul
import etcd3
from sklearn.ensemble import IsolationForest
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

class ConfigSource(Enum):
    """Configuration source types"""
    FILE_SYSTEM = "filesystem"
    KUBERNETES = "kubernetes"
    CONSUL = "consul"
    ETCD = "etcd"
    DATABASE = "database"
    ENVIRONMENT = "environment"
    VAULT = "vault"

class ConfigStatus(Enum):
    """Configuration status"""
    VALID = "valid"
    INVALID = "invalid"
    DRIFTED = "drifted"
    REMEDIATED = "remediated"
    PENDING_APPROVAL = "pending_approval"
    ROLLBACK_REQUIRED = "rollback_required"

class DriftSeverity(Enum):
    """Configuration drift severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class Environment(Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    DISASTER_RECOVERY = "disaster_recovery"

@dataclass
class ConfigDrift:
    """Configuration drift detection result"""
    drift_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    config_key: str = ""
    expected_value: Any = None
    actual_value: Any = None
    drift_type: str = "value_change"  # value_change, key_missing, key_added
    severity: DriftSeverity = DriftSeverity.MEDIUM
    detected_at: datetime = field(default_factory=datetime.now)
    environment: Environment = Environment.PRODUCTION
    source: ConfigSource = ConfigSource.FILE_SYSTEM
    auto_remediate: bool = True
    remediation_applied: bool = False
    remediation_timestamp: Optional[datetime] = None

@dataclass
class ConfigValidationRule:
    """Configuration validation rule"""
    rule_id: str
    name: str
    description: str
    rule_type: str  # schema, range, enum, regex, custom
    parameters: Dict[str, Any]
    severity: DriftSeverity = DriftSeverity.MEDIUM
    environments: List[Environment] = field(default_factory=lambda: [Environment.PRODUCTION])
    enabled: bool = True

class ConfigurationSchema(BaseModel):
    """Configuration schema definition"""
    schema_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    version: str
    environment: Environment
    schema_definition: Dict[str, Any]
    validation_rules: List[ConfigValidationRule] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    is_active: bool = True

class ConfigurationVersion(BaseModel):
    """Configuration version tracking"""
    version_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    config_path: str
    version_number: str
    content: Dict[str, Any]
    content_hash: str
    author: str = "smart_config_manager"
    commit_message: str = ""
    created_at: datetime = Field(default_factory=datetime.now)
    environment: Environment
    is_active: bool = False
    rollback_safe: bool = True
    approval_status: str = "pending"  # pending, approved, rejected

class AIConfigOptimizer:
    """AI-powered configuration optimizer"""
    
    def __init__(self):
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.performance_history: Dict[str, List[float]] = {}
        self.optimization_patterns: Dict[str, Any] = {}
        self.learning_threshold = 50  # Minimum samples for learning
        
    def analyze_configuration_performance(self, config_key: str, config_value: Any, 
                                        performance_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Analyze configuration impact on system performance"""
        try:
            # Store performance data
            if config_key not in self.performance_history:
                self.performance_history[config_key] = []
            
            # Combine config and performance into feature vector
            feature_vector = self._extract_features(config_value, performance_metrics)
            self.performance_history[config_key].append(feature_vector)
            
            # Keep only recent history
            if len(self.performance_history[config_key]) > 1000:
                self.performance_history[config_key] = self.performance_history[config_key][-1000:]
            
            # Analyze if we have enough data
            if len(self.performance_history[config_key]) < self.learning_threshold:
                return {"status": "insufficient_data", "recommendations": []}
            
            # Detect anomalies
            data = np.array(self.performance_history[config_key])
            anomalies = self.anomaly_detector.fit_predict(data)
            current_anomaly = anomalies[-1] == -1
            
            # Generate recommendations
            recommendations = []
            
            if current_anomaly:
                recommendations.append({
                    "type": "performance_anomaly",
                    "severity": "high",
                    "message": "Current configuration shows anomalous performance patterns",
                    "suggested_action": "review_configuration"
                })
            
            # Pattern analysis
            recent_performance = [fv[-1] for fv in self.performance_history[config_key][-10:]]  # Last metric
            performance_trend = self._calculate_trend(recent_performance)
            
            if performance_trend < -0.1:  # Declining performance
                recommendations.append({
                    "type": "performance_degradation",
                    "severity": "medium",
                    "message": "Configuration showing declining performance trend",
                    "suggested_action": "optimize_configuration"
                })
            
            return {
                "status": "analyzed",
                "anomaly_detected": current_anomaly,
                "performance_trend": performance_trend,
                "recommendations": recommendations
            }
            
        except Exception as e:
            logger.error(f"Error analyzing configuration performance: {e}")
            return {"status": "error", "error": str(e)}
    
    def _extract_features(self, config_value: Any, performance_metrics: Dict[str, float]) -> List[float]:
        """Extract numerical features from configuration and performance data"""
        features = []
        
        # Config value features
        if isinstance(config_value, (int, float)):
            features.append(float(config_value))
        elif isinstance(config_value, str):
            features.append(float(len(config_value)))
        elif isinstance(config_value, bool):
            features.append(1.0 if config_value else 0.0)
        elif isinstance(config_value, dict):
            features.append(float(len(config_value)))
        elif isinstance(config_value, list):
            features.append(float(len(config_value)))
        else:
            features.append(0.0)
        
        # Performance metrics features
        for metric_name in ['cpu_usage', 'memory_usage', 'response_time', 'error_rate', 'throughput']:
            features.append(performance_metrics.get(metric_name, 0.0))
        
        return features
    
    def _calculate_trend(self, data: List[float]) -> float:
        """Calculate trend in performance data"""
        if len(data) < 3:
            return 0.0
        
        x = np.arange(len(data))
        y = np.array(data)
        
        # Simple linear regression slope
        n = len(x)
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - (np.sum(x))**2)
        return slope / np.mean(y) if np.mean(y) != 0 else 0.0
    
    def suggest_optimizations(self, config_path: str, current_config: Dict[str, Any], 
                            performance_metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Suggest configuration optimizations based on performance analysis"""
        optimizations = []
        
        # Analyze each configuration parameter
        for key, value in current_config.items():
            if isinstance(value, (int, float)):
                # Numerical parameter optimization
                opt = self._optimize_numerical_parameter(key, value, performance_metrics)
                if opt:
                    optimizations.append(opt)
            elif isinstance(value, bool):
                # Boolean parameter optimization
                opt = self._optimize_boolean_parameter(key, value, performance_metrics)
                if opt:
                    optimizations.append(opt)
        
        return optimizations
    
    def _optimize_numerical_parameter(self, key: str, value: Union[int, float], 
                                    performance_metrics: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Suggest optimization for numerical parameters"""
        # Common optimization patterns
        if 'timeout' in key.lower() and performance_metrics.get('response_time', 0) > 5.0:
            return {
                "parameter": key,
                "current_value": value,
                "suggested_value": min(value * 1.5, value + 30),
                "reason": "High response times detected, consider increasing timeout",
                "confidence": 0.7
            }
        
        if 'pool' in key.lower() and performance_metrics.get('cpu_usage', 0) > 0.8:
            return {
                "parameter": key,
                "current_value": value,
                "suggested_value": min(value + 5, value * 1.3),
                "reason": "High CPU usage detected, consider increasing pool size",
                "confidence": 0.6
            }
        
        return None
    
    def _optimize_boolean_parameter(self, key: str, value: bool, 
                                  performance_metrics: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Suggest optimization for boolean parameters"""
        if 'cache' in key.lower() and not value and performance_metrics.get('response_time', 0) > 2.0:
            return {
                "parameter": key,
                "current_value": value,
                "suggested_value": True,
                "reason": "High response times detected, consider enabling caching",
                "confidence": 0.8
            }
        
        return None

class SmartConfigurationManager:
    """Main smart configuration manager"""
    
    def __init__(self, config_root: str = "/etc/nautilus/config"):
        self.config_root = Path(config_root)
        self.config_sources: Dict[ConfigSource, Any] = {}
        self.schemas: Dict[str, ConfigurationSchema] = {}
        self.configurations: Dict[str, Dict[str, Any]] = {}
        self.configuration_versions: Dict[str, List[ConfigurationVersion]] = {}
        self.drift_history: List[ConfigDrift] = []
        self.ai_optimizer = AIConfigOptimizer()
        
        # Git repository for version control
        self.git_repo: Optional[git.Repo] = None
        self._initialize_git_repo()
        
        # Configuration sources
        self._initialize_config_sources()
        
        # Monitoring
        self.monitoring_active = False
        self.monitoring_interval = 60  # seconds
        self.last_check: Dict[str, datetime] = {}
        
    def _initialize_git_repo(self) -> None:
        """Initialize Git repository for configuration versioning"""
        try:
            if not self.config_root.exists():
                self.config_root.mkdir(parents=True, exist_ok=True)
            
            git_dir = self.config_root / ".git"
            if git_dir.exists():
                self.git_repo = git.Repo(self.config_root)
            else:
                self.git_repo = git.Repo.init(self.config_root)
                
                # Create initial commit
                gitignore_path = self.config_root / ".gitignore"
                gitignore_path.write_text("*.tmp\n*.log\n__pycache__/\n")
                
                self.git_repo.index.add([".gitignore"])
                self.git_repo.index.commit("Initial commit")
            
            logger.info(f"Git repository initialized at {self.config_root}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Git repository: {e}")
    
    def _initialize_config_sources(self) -> None:
        """Initialize configuration sources"""
        try:
            # Kubernetes client
            try:
                k8s.config.load_incluster_config()
            except:
                try:
                    k8s.config.load_kube_config()
                except:
                    logger.warning("Could not load Kubernetes config")
            
            self.config_sources[ConfigSource.KUBERNETES] = {
                'apps_v1': k8s.client.AppsV1Api(),
                'core_v1': k8s.client.CoreV1Api()
            }
            
            # Consul client (if available)
            try:
                consul_client = consul.Consul()
                self.config_sources[ConfigSource.CONSUL] = consul_client
            except Exception as e:
                logger.warning(f"Could not initialize Consul client: {e}")
            
            # etcd client (if available)
            try:
                etcd_client = etcd3.client()
                self.config_sources[ConfigSource.ETCD] = etcd_client
            except Exception as e:
                logger.warning(f"Could not initialize etcd client: {e}")
            
            # File system source
            self.config_sources[ConfigSource.FILE_SYSTEM] = self.config_root
            
            logger.info(f"Initialized {len(self.config_sources)} configuration sources")
            
        except Exception as e:
            logger.error(f"Error initializing configuration sources: {e}")
    
    async def register_schema(self, schema: ConfigurationSchema) -> str:
        """Register configuration schema"""
        try:
            # Validate schema definition
            validation_result = await self._validate_schema_definition(schema.schema_definition)
            if not validation_result['valid']:
                raise ValueError(f"Invalid schema: {validation_result['errors']}")
            
            self.schemas[schema.schema_id] = schema
            
            # Save schema to file
            schema_file = self.config_root / "schemas" / f"{schema.name}-{schema.environment.value}.json"
            schema_file.parent.mkdir(exist_ok=True)
            
            with open(schema_file, 'w') as f:
                json.dump(schema.dict(), f, indent=2, default=str)
            
            # Commit to Git
            if self.git_repo:
                self.git_repo.index.add([str(schema_file.relative_to(self.config_root))])
                self.git_repo.index.commit(f"Register schema: {schema.name} v{schema.version}")
            
            logger.info(f"Registered schema: {schema.name} ({schema.schema_id})")
            return schema.schema_id
            
        except Exception as e:
            logger.error(f"Error registering schema: {e}")
            raise
    
    async def _validate_schema_definition(self, schema_def: Dict[str, Any]) -> Dict[str, Any]:
        """Validate schema definition"""
        errors = []
        warnings = []
        
        try:
            # Try to validate a sample configuration against the schema
            jsonschema.Draft7Validator.check_schema(schema_def)
        except jsonschema.SchemaError as e:
            errors.append(f"Invalid JSON schema: {e.message}")
        
        # Check for required fields
        if 'type' not in schema_def:
            errors.append("Schema must have 'type' field")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    async def load_configuration(self, config_path: str, source: ConfigSource = ConfigSource.FILE_SYSTEM, 
                               environment: Environment = Environment.PRODUCTION) -> Dict[str, Any]:
        """Load configuration from specified source"""
        try:
            if source == ConfigSource.FILE_SYSTEM:
                return await self._load_from_filesystem(config_path)
            elif source == ConfigSource.KUBERNETES:
                return await self._load_from_kubernetes(config_path)
            elif source == ConfigSource.CONSUL:
                return await self._load_from_consul(config_path)
            elif source == ConfigSource.ETCD:
                return await self._load_from_etcd(config_path)
            else:
                raise ValueError(f"Unsupported configuration source: {source}")
                
        except Exception as e:
            logger.error(f"Error loading configuration from {source.value}: {e}")
            raise
    
    async def _load_from_filesystem(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file system"""
        file_path = self.config_root / config_path
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        # Determine file format
        if file_path.suffix.lower() == '.yaml' or file_path.suffix.lower() == '.yml':
            with open(file_path, 'r') as f:
                return yaml.safe_load(f)
        elif file_path.suffix.lower() == '.json':
            with open(file_path, 'r') as f:
                return json.load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {file_path.suffix}")
    
    async def _load_from_kubernetes(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from Kubernetes ConfigMap or Secret"""
        k8s_client = self.config_sources.get(ConfigSource.KUBERNETES, {}).get('core_v1')
        if not k8s_client:
            raise RuntimeError("Kubernetes client not available")
        
        # Parse config path (namespace/configmap_name or namespace/secret_name)
        parts = config_path.split('/')
        if len(parts) != 2:
            raise ValueError("Kubernetes config path must be namespace/resource_name")
        
        namespace, resource_name = parts
        
        try:
            # Try ConfigMap first
            config_map = k8s_client.read_namespaced_config_map(
                name=resource_name,
                namespace=namespace
            )
            return dict(config_map.data) if config_map.data else {}
        except ApiException as e:
            if e.status == 404:
                try:
                    # Try Secret
                    secret = k8s_client.read_namespaced_secret(
                        name=resource_name,
                        namespace=namespace
                    )
                    # Decode base64 encoded values
                    data = {}
                    if secret.data:
                        import base64
                        for key, value in secret.data.items():
                            data[key] = base64.b64decode(value).decode('utf-8')
                    return data
                except ApiException as secret_e:
                    if secret_e.status == 404:
                        raise FileNotFoundError(f"ConfigMap or Secret not found: {config_path}")
                    raise
            raise
    
    async def save_configuration(self, config_path: str, config_data: Dict[str, Any], 
                               source: ConfigSource = ConfigSource.FILE_SYSTEM,
                               environment: Environment = Environment.PRODUCTION,
                               commit_message: str = "") -> str:
        """Save configuration to specified source"""
        try:
            # Calculate content hash
            content_hash = hashlib.sha256(
                json.dumps(config_data, sort_keys=True).encode()
            ).hexdigest()
            
            # Create version entry
            version = ConfigurationVersion(
                config_path=config_path,
                version_number=f"v{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                content=config_data,
                content_hash=content_hash,
                commit_message=commit_message or f"Update configuration: {config_path}",
                environment=environment
            )
            
            # Save based on source
            if source == ConfigSource.FILE_SYSTEM:
                await self._save_to_filesystem(config_path, config_data, version)
            elif source == ConfigSource.KUBERNETES:
                await self._save_to_kubernetes(config_path, config_data, version)
            elif source == ConfigSource.CONSUL:
                await self._save_to_consul(config_path, config_data, version)
            elif source == ConfigSource.ETCD:
                await self._save_to_etcd(config_path, config_data, version)
            else:
                raise ValueError(f"Unsupported configuration source: {source}")
            
            # Store version
            if config_path not in self.configuration_versions:
                self.configuration_versions[config_path] = []
            
            self.configuration_versions[config_path].append(version)
            self.configurations[config_path] = config_data
            
            logger.info(f"Saved configuration {config_path} version {version.version_number}")
            return version.version_id
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            raise
    
    async def _save_to_filesystem(self, config_path: str, config_data: Dict[str, Any], 
                                 version: ConfigurationVersion) -> None:
        """Save configuration to file system"""
        file_path = self.config_root / config_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine format and save
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            with open(file_path, 'w') as f:
                yaml.safe_dump(config_data, f, default_flow_style=False)
        else:
            with open(file_path, 'w') as f:
                json.dump(config_data, f, indent=2)
        
        # Commit to Git
        if self.git_repo:
            self.git_repo.index.add([str(file_path.relative_to(self.config_root))])
            self.git_repo.index.commit(version.commit_message)
    
    async def detect_drift(self, config_path: str, environment: Environment = Environment.PRODUCTION) -> List[ConfigDrift]:
        """Detect configuration drift"""
        try:
            drift_results = []
            
            # Load expected configuration (from Git)
            expected_config = await self._get_expected_configuration(config_path, environment)
            if not expected_config:
                logger.warning(f"No expected configuration found for {config_path}")
                return drift_results
            
            # Load actual configuration from all sources
            for source in [ConfigSource.FILE_SYSTEM, ConfigSource.KUBERNETES]:
                try:
                    actual_config = await self.load_configuration(config_path, source, environment)
                    
                    # Compare configurations
                    diffs = self._compare_configurations(expected_config, actual_config)
                    
                    for diff in diffs:
                        drift = ConfigDrift(
                            config_key=diff['key'],
                            expected_value=diff['expected'],
                            actual_value=diff['actual'],
                            drift_type=diff['type'],
                            severity=self._assess_drift_severity(diff),
                            environment=environment,
                            source=source
                        )
                        drift_results.append(drift)
                        
                except Exception as e:
                    logger.warning(f"Could not load configuration from {source.value}: {e}")
                    continue
            
            # Store drift history
            self.drift_history.extend(drift_results)
            
            if drift_results:
                logger.warning(f"Detected {len(drift_results)} configuration drifts for {config_path}")
            
            return drift_results
            
        except Exception as e:
            logger.error(f"Error detecting drift: {e}")
            return []
    
    def _compare_configurations(self, expected: Dict[str, Any], actual: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Compare two configurations and return differences"""
        diffs = []
        
        try:
            deep_diff = DeepDiff(expected, actual, ignore_order=True)
            
            # Handle different types of changes
            for change_type, changes in deep_diff.items():
                if change_type == 'values_changed':
                    for key, change in changes.items():
                        diffs.append({
                            'key': key.replace('root[\'', '').replace('\']', '').replace('[', '.').replace(']', ''),
                            'expected': change['old_value'],
                            'actual': change['new_value'],
                            'type': 'value_change'
                        })
                elif change_type == 'dictionary_item_removed':
                    for key in changes:
                        clean_key = key.replace('root[\'', '').replace('\']', '').replace('[', '.').replace(']', '')
                        diffs.append({
                            'key': clean_key,
                            'expected': self._get_nested_value(expected, clean_key),
                            'actual': None,
                            'type': 'key_missing'
                        })
                elif change_type == 'dictionary_item_added':
                    for key in changes:
                        clean_key = key.replace('root[\'', '').replace('\']', '').replace('[', '.').replace(']', '')
                        diffs.append({
                            'key': clean_key,
                            'expected': None,
                            'actual': self._get_nested_value(actual, clean_key),
                            'type': 'key_added'
                        })
                        
        except Exception as e:
            logger.error(f"Error comparing configurations: {e}")
        
        return diffs
    
    def _get_nested_value(self, data: Dict[str, Any], key_path: str) -> Any:
        """Get nested value from dictionary using dot notation"""
        keys = key_path.split('.')
        value = data
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        
        return value
    
    def _assess_drift_severity(self, diff: Dict[str, Any]) -> DriftSeverity:
        """Assess the severity of configuration drift"""
        key = diff['key'].lower()
        
        # Critical drifts
        if any(critical_key in key for critical_key in ['security', 'auth', 'password', 'secret', 'key']):
            return DriftSeverity.CRITICAL
        
        # High severity
        if any(high_key in key for high_key in ['port', 'endpoint', 'database', 'redis']):
            return DriftSeverity.HIGH
        
        # Medium severity
        if any(medium_key in key for medium_key in ['timeout', 'pool', 'limit', 'threshold']):
            return DriftSeverity.MEDIUM
        
        # Default to low
        return DriftSeverity.LOW
    
    async def remediate_drift(self, drift: ConfigDrift, auto_approve: bool = False) -> bool:
        """Remediate configuration drift"""
        try:
            if not drift.auto_remediate and not auto_approve:
                logger.info(f"Drift {drift.drift_id} requires manual approval")
                return False
            
            # Apply remediation based on drift type
            if drift.drift_type == 'value_change':
                success = await self._remediate_value_change(drift)
            elif drift.drift_type == 'key_missing':
                success = await self._remediate_missing_key(drift)
            elif drift.drift_type == 'key_added':
                success = await self._remediate_added_key(drift)
            else:
                logger.warning(f"Unknown drift type: {drift.drift_type}")
                return False
            
            if success:
                drift.remediation_applied = True
                drift.remediation_timestamp = datetime.now()
                logger.info(f"Successfully remediated drift {drift.drift_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error remediating drift {drift.drift_id}: {e}")
            return False
    
    async def _remediate_value_change(self, drift: ConfigDrift) -> bool:
        """Remediate value change drift"""
        try:
            # Load current configuration
            config_data = await self.load_configuration(
                drift.config_key.split('.')[0],  # Extract config path
                drift.source,
                drift.environment
            )
            
            # Update the value
            keys = drift.config_key.split('.')
            self._set_nested_value(config_data, keys, drift.expected_value)
            
            # Save updated configuration
            await self.save_configuration(
                drift.config_key.split('.')[0],
                config_data,
                drift.source,
                drift.environment,
                f"Auto-remediate drift: {drift.config_key}"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error remediating value change: {e}")
            return False
    
    def _set_nested_value(self, data: Dict[str, Any], keys: List[str], value: Any) -> None:
        """Set nested value in dictionary using key path"""
        current = data
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    async def validate_configuration(self, config_path: str, config_data: Dict[str, Any], 
                                   environment: Environment = Environment.PRODUCTION) -> Dict[str, Any]:
        """Validate configuration against schema and rules"""
        try:
            validation_results = {
                'valid': True,
                'errors': [],
                'warnings': [],
                'schema_validation': None,
                'rule_validations': []
            }
            
            # Find matching schema
            schema = self._find_matching_schema(config_path, environment)
            if schema:
                # Schema validation
                try:
                    validate(config_data, schema.schema_definition)
                    validation_results['schema_validation'] = {'valid': True}
                except ValidationError as e:
                    validation_results['valid'] = False
                    validation_results['errors'].append(f"Schema validation failed: {e.message}")
                    validation_results['schema_validation'] = {'valid': False, 'error': e.message}
                
                # Rule validations
                for rule in schema.validation_rules:
                    if environment in rule.environments and rule.enabled:
                        rule_result = await self._validate_rule(config_data, rule)
                        validation_results['rule_validations'].append(rule_result)
                        
                        if not rule_result['valid']:
                            if rule.severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]:
                                validation_results['valid'] = False
                                validation_results['errors'].append(rule_result['message'])
                            else:
                                validation_results['warnings'].append(rule_result['message'])
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating configuration: {e}")
            return {
                'valid': False,
                'errors': [f"Validation error: {e}"],
                'warnings': [],
                'schema_validation': None,
                'rule_validations': []
            }
    
    def _find_matching_schema(self, config_path: str, environment: Environment) -> Optional[ConfigurationSchema]:
        """Find schema matching the configuration path and environment"""
        for schema in self.schemas.values():
            if (schema.environment == environment and 
                schema.is_active and
                config_path.startswith(schema.name.lower())):
                return schema
        return None
    
    async def _validate_rule(self, config_data: Dict[str, Any], rule: ConfigValidationRule) -> Dict[str, Any]:
        """Validate configuration against a specific rule"""
        try:
            if rule.rule_type == 'range':
                return self._validate_range_rule(config_data, rule)
            elif rule.rule_type == 'enum':
                return self._validate_enum_rule(config_data, rule)
            elif rule.rule_type == 'regex':
                return self._validate_regex_rule(config_data, rule)
            elif rule.rule_type == 'custom':
                return await self._validate_custom_rule(config_data, rule)
            else:
                return {'valid': True, 'rule_id': rule.rule_id, 'message': 'Rule type not implemented'}
                
        except Exception as e:
            return {
                'valid': False,
                'rule_id': rule.rule_id,
                'message': f"Rule validation error: {e}"
            }
    
    def _validate_range_rule(self, config_data: Dict[str, Any], rule: ConfigValidationRule) -> Dict[str, Any]:
        """Validate range rule"""
        field_path = rule.parameters.get('field')
        min_value = rule.parameters.get('min')
        max_value = rule.parameters.get('max')
        
        value = self._get_nested_value(config_data, field_path)
        
        if value is None:
            return {'valid': False, 'rule_id': rule.rule_id, 'message': f"Field {field_path} not found"}
        
        if not isinstance(value, (int, float)):
            return {'valid': False, 'rule_id': rule.rule_id, 'message': f"Field {field_path} is not numeric"}
        
        if min_value is not None and value < min_value:
            return {'valid': False, 'rule_id': rule.rule_id, 'message': f"Field {field_path} below minimum {min_value}"}
        
        if max_value is not None and value > max_value:
            return {'valid': False, 'rule_id': rule.rule_id, 'message': f"Field {field_path} above maximum {max_value}"}
        
        return {'valid': True, 'rule_id': rule.rule_id, 'message': 'Range validation passed'}
    
    async def optimize_configuration(self, config_path: str, environment: Environment,
                                   performance_metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Use AI to optimize configuration based on performance metrics"""
        try:
            current_config = await self.load_configuration(config_path, ConfigSource.FILE_SYSTEM, environment)
            
            # Get AI recommendations
            recommendations = self.ai_optimizer.suggest_optimizations(
                config_path, current_config, performance_metrics
            )
            
            # Analyze performance impact of current configuration
            for key, value in current_config.items():
                analysis = self.ai_optimizer.analyze_configuration_performance(
                    f"{config_path}.{key}", value, performance_metrics
                )
                
                if analysis.get('recommendations'):
                    for rec in analysis['recommendations']:
                        rec['config_key'] = f"{config_path}.{key}"
                        recommendations.append(rec)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error optimizing configuration: {e}")
            return []
    
    async def start_monitoring(self) -> None:
        """Start continuous configuration monitoring"""
        self.monitoring_active = True
        
        async def monitor_loop():
            while self.monitoring_active:
                try:
                    await self._monitoring_cycle()
                    await asyncio.sleep(self.monitoring_interval)
                except Exception as e:
                    logger.error(f"Error in monitoring cycle: {e}")
                    await asyncio.sleep(60)  # Wait longer on error
        
        asyncio.create_task(monitor_loop())
        logger.info("Started configuration monitoring")
    
    async def stop_monitoring(self) -> None:
        """Stop configuration monitoring"""
        self.monitoring_active = False
        logger.info("Stopped configuration monitoring")
    
    async def _monitoring_cycle(self) -> None:
        """Execute one monitoring cycle"""
        for config_path in self.configurations.keys():
            try:
                # Check for drift
                drifts = await self.detect_drift(config_path)
                
                # Auto-remediate if configured
                for drift in drifts:
                    if drift.auto_remediate and drift.severity != DriftSeverity.CRITICAL:
                        await self.remediate_drift(drift, auto_approve=True)
                
                self.last_check[config_path] = datetime.now()
                
            except Exception as e:
                logger.error(f"Error monitoring {config_path}: {e}")
    
    def get_drift_summary(self, environment: Environment = Environment.PRODUCTION) -> Dict[str, Any]:
        """Get summary of configuration drifts"""
        env_drifts = [drift for drift in self.drift_history if drift.environment == environment]
        
        summary = {
            'total_drifts': len(env_drifts),
            'by_severity': {},
            'by_source': {},
            'unremediated_count': 0,
            'recent_drifts': []
        }
        
        # Count by severity
        for severity in DriftSeverity:
            summary['by_severity'][severity.value] = len([
                d for d in env_drifts if d.severity == severity
            ])
        
        # Count by source
        for source in ConfigSource:
            summary['by_source'][source.value] = len([
                d for d in env_drifts if d.source == source
            ])
        
        # Count unremediated
        summary['unremediated_count'] = len([
            d for d in env_drifts if not d.remediation_applied
        ])
        
        # Recent drifts (last 24 hours)
        recent_cutoff = datetime.now() - timedelta(hours=24)
        summary['recent_drifts'] = [
            {
                'drift_id': d.drift_id,
                'config_key': d.config_key,
                'severity': d.severity.value,
                'detected_at': d.detected_at.isoformat(),
                'remediated': d.remediation_applied
            }
            for d in env_drifts 
            if d.detected_at > recent_cutoff
        ]
        
        return summary
    
    async def rollback_configuration(self, config_path: str, target_version: str) -> bool:
        """Rollback configuration to a specific version"""
        try:
            versions = self.configuration_versions.get(config_path, [])
            target = next((v for v in versions if v.version_number == target_version), None)
            
            if not target:
                raise ValueError(f"Version {target_version} not found for {config_path}")
            
            if not target.rollback_safe:
                raise ValueError(f"Version {target_version} is not safe for rollback")
            
            # Rollback in Git
            if self.git_repo:
                # Find the commit for this version
                for commit in self.git_repo.iter_commits():
                    if target.commit_message in commit.message:
                        # Create a new commit that reverts to this version
                        self.git_repo.git.checkout(commit.hexsha, '--', config_path)
                        self.git_repo.index.commit(f"Rollback {config_path} to {target_version}")
                        break
            
            # Update current configuration
            self.configurations[config_path] = target.content
            
            # Mark version as active
            for version in versions:
                version.is_active = False
            target.is_active = True
            
            logger.info(f"Successfully rolled back {config_path} to {target_version}")
            return True
            
        except Exception as e:
            logger.error(f"Error rolling back configuration: {e}")
            return False

# Example usage and testing
async def example_usage():
    """Example usage of Smart Configuration Manager"""
    config_manager = SmartConfigurationManager("/tmp/nautilus_config")
    
    # Register a schema
    schema = ConfigurationSchema(
        name="trading-service",
        version="1.0",
        environment=Environment.PRODUCTION,
        schema_definition={
            "type": "object",
            "properties": {
                "database": {
                    "type": "object",
                    "properties": {
                        "host": {"type": "string"},
                        "port": {"type": "integer", "minimum": 1, "maximum": 65535},
                        "pool_size": {"type": "integer", "minimum": 1, "maximum": 100}
                    },
                    "required": ["host", "port"]
                },
                "redis": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "timeout": {"type": "number", "minimum": 0}
                    }
                }
            },
            "required": ["database", "redis"]
        }
    )
    
    await config_manager.register_schema(schema)
    
    # Create sample configuration
    config_data = {
        "database": {
            "host": "localhost",
            "port": 5432,
            "pool_size": 10
        },
        "redis": {
            "url": "redis://localhost:6379",
            "timeout": 5.0
        }
    }
    
    # Save configuration
    version_id = await config_manager.save_configuration(
        "trading-service.json", 
        config_data,
        commit_message="Initial trading service configuration"
    )
    
    print(f"Saved configuration version: {version_id}")
    
    # Validate configuration
    validation_result = await config_manager.validate_configuration(
        "trading-service.json", 
        config_data,
        Environment.PRODUCTION
    )
    
    print(f"Validation result: {validation_result}")
    
    # Start monitoring
    await config_manager.start_monitoring()
    
    # Simulate drift detection
    await asyncio.sleep(2)
    drifts = await config_manager.detect_drift("trading-service.json")
    
    print(f"Detected {len(drifts)} configuration drifts")
    
    # Get drift summary
    summary = config_manager.get_drift_summary()
    print(f"Drift summary: {summary}")
    
    # Stop monitoring
    await config_manager.stop_monitoring()

if __name__ == "__main__":
    asyncio.run(example_usage())