# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2025 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Union, Any, Tuple
from enum import Enum
import json
import os
import logging
import warnings
from pathlib import Path

class MessagePriority(Enum):
    """Message priority levels matching NautilusTrader patterns"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3

class TopicPattern:
    """Topic pattern matching similar to NautilusTrader's implementation"""
    
    def __init__(self, pattern: str):
        self.pattern = pattern
        self.compiled = self._compile_pattern(pattern)
    
    def _compile_pattern(self, pattern: str) -> str:
        """Compile glob pattern to regex"""
        import re
        # Convert glob patterns to regex
        regex_pattern = pattern.replace('.', r'\.')
        regex_pattern = regex_pattern.replace('*', '.*')
        regex_pattern = regex_pattern.replace('?', '.')
        return f"^{regex_pattern}$"
    
    def matches(self, topic: str) -> bool:
        """Check if topic matches pattern"""
        import re
        return bool(re.match(self.compiled, topic))

@dataclass
class BufferConfig:
    """Buffer configuration matching NautilusTrader patterns"""
    max_size: int = 10000
    flush_interval_ms: int = 100
    high_water_mark: int = 8000
    low_water_mark: int = 2000
    enable_compression: bool = True
    compression_threshold: int = 1024

@dataclass
class StreamConfig:
    """Redis stream configuration"""
    name: str
    max_length: int = 100000
    trim_strategy: str = "MAXLEN"
    consumer_group: Optional[str] = None
    consumer_name: Optional[str] = None
    block_timeout_ms: int = 1000
    batch_size: int = 100
    auto_ack: bool = True

@dataclass
class PatternSubscription:
    """Pattern-based subscription configuration"""
    pattern: str
    priority: MessagePriority = MessagePriority.NORMAL
    buffer_config: Optional[BufferConfig] = None
    filters: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        self.topic_pattern = TopicPattern(self.pattern)

@dataclass
class EnhancedMessageBusConfig:
    """Enhanced MessageBus configuration inspired by NautilusTrader"""
    
    # Connection settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    connection_pool_size: int = 20
    connection_timeout: float = 5.0
    command_timeout: float = 30.0
    
    # Buffer settings
    default_buffer_config: BufferConfig = field(default_factory=BufferConfig)
    priority_buffers: Dict[MessagePriority, BufferConfig] = field(default_factory=dict)
    
    # Stream settings
    default_stream_config: StreamConfig = field(default_factory=StreamConfig)
    stream_configs: Dict[str, StreamConfig] = field(default_factory=dict)
    
    # Subscription settings
    subscriptions: List[PatternSubscription] = field(default_factory=list)
    enable_pattern_matching: bool = True
    max_concurrent_handlers: int = 100
    
    # Performance settings
    enable_metrics: bool = True
    metrics_interval_ms: int = 5000
    enable_tracing: bool = False
    trace_sample_rate: float = 0.1
    
    # Health check settings
    health_check_interval: int = 30
    heartbeat_timeout: int = 10
    max_consecutive_failures: int = 3
    
    # Auto-scaling settings
    auto_scale_enabled: bool = True
    scale_up_threshold: float = 0.8  # 80% buffer utilization
    scale_down_threshold: float = 0.3  # 30% buffer utilization
    min_workers: int = 1
    max_workers: int = 10
    
    # Serialization settings
    default_serializer: str = "json"
    compression_enabled: bool = True
    compression_level: int = 6
    
    def add_subscription(self, pattern: str, priority: MessagePriority = MessagePriority.NORMAL, 
                        filters: Optional[Dict[str, str]] = None, 
                        buffer_config: Optional[BufferConfig] = None):
        """Add a pattern subscription"""
        subscription = PatternSubscription(
            pattern=pattern,
            priority=priority,
            buffer_config=buffer_config or self.default_buffer_config,
            filters=filters or {}
        )
        self.subscriptions.append(subscription)
    
    def add_stream(self, name: str, config: Optional[StreamConfig] = None):
        """Add a stream configuration"""
        if config is None:
            config = StreamConfig(name=name)
        self.stream_configs[name] = config
    
    def get_buffer_config(self, priority: MessagePriority) -> BufferConfig:
        """Get buffer config for priority level"""
        return self.priority_buffers.get(priority, self.default_buffer_config)
    
    def get_stream_config(self, stream_name: str) -> StreamConfig:
        """Get stream config by name"""
        return self.stream_configs.get(stream_name, self.default_stream_config)
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        config_dict = {}
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, Enum):
                config_dict[field_name] = field_value.value
            elif isinstance(field_value, (BufferConfig, StreamConfig)):
                config_dict[field_name] = field_value.__dict__
            elif isinstance(field_value, list):
                config_dict[field_name] = [
                    item.__dict__ if hasattr(item, '__dict__') else item 
                    for item in field_value
                ]
            elif isinstance(field_value, dict):
                config_dict[field_name] = {
                    k: v.__dict__ if hasattr(v, '__dict__') else v 
                    for k, v in field_value.items()
                }
            else:
                config_dict[field_name] = field_value
        return config_dict
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'EnhancedMessageBusConfig':
        """Create config from dictionary with validation"""
        # Validate the configuration data
        validator = ConfigValidator()
        validation_result = validator.validate(data)
        
        if not validation_result.is_valid:
            raise ValueError(f"Invalid configuration: {validation_result.errors}")
        
        # Apply warnings for deprecated fields
        if validation_result.warnings:
            for warning_msg in validation_result.warnings:
                warnings.warn(warning_msg, DeprecationWarning)
        
        # Convert nested objects
        processed_data = cls._process_dict_data(data)
        return cls(**processed_data)
    
    @classmethod
    def _process_dict_data(cls, data: Dict) -> Dict:
        """Process dictionary data for object creation"""
        processed = data.copy()
        
        # Convert buffer configs
        if 'default_buffer_config' in processed and isinstance(processed['default_buffer_config'], dict):
            processed['default_buffer_config'] = BufferConfig(**processed['default_buffer_config'])
        
        if 'priority_buffers' in processed:
            priority_buffers = {}
            for priority_str, buffer_dict in processed['priority_buffers'].items():
                if isinstance(priority_str, str):
                    priority = MessagePriority[priority_str.upper()]
                else:
                    priority = priority_str
                if isinstance(buffer_dict, dict):
                    priority_buffers[priority] = BufferConfig(**buffer_dict)
                else:
                    priority_buffers[priority] = buffer_dict
            processed['priority_buffers'] = priority_buffers
        
        # Convert stream configs
        if 'stream_configs' in processed:
            stream_configs = {}
            for name, stream_dict in processed['stream_configs'].items():
                if isinstance(stream_dict, dict):
                    stream_configs[name] = StreamConfig(**stream_dict)
                else:
                    stream_configs[name] = stream_dict
            processed['stream_configs'] = stream_configs
        
        # Convert subscriptions
        if 'subscriptions' in processed:
            subscriptions = []
            for sub_dict in processed['subscriptions']:
                if isinstance(sub_dict, dict):
                    # Convert priority string to enum
                    if 'priority' in sub_dict and isinstance(sub_dict['priority'], str):
                        sub_dict['priority'] = MessagePriority[sub_dict['priority'].upper()]
                    subscriptions.append(PatternSubscription(**sub_dict))
                else:
                    subscriptions.append(sub_dict)
            processed['subscriptions'] = subscriptions
        
        return processed
    
    @classmethod
    def from_file(cls, filepath: str) -> 'EnhancedMessageBusConfig':
        """Load config from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def save_to_file(self, filepath: str):
        """Save config to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

# Predefined configurations for different environments
class ConfigPresets:
    """Predefined configuration presets"""
    
    @staticmethod
    def development() -> EnhancedMessageBusConfig:
        """Development configuration"""
        config = EnhancedMessageBusConfig(
            connection_pool_size=5,
            enable_metrics=True,
            enable_tracing=True,
            trace_sample_rate=1.0,
            auto_scale_enabled=False
        )
        
        # Add common development subscriptions
        config.add_subscription("data.*", MessagePriority.NORMAL)
        config.add_subscription("events.*", MessagePriority.HIGH)
        config.add_subscription("alerts.*", MessagePriority.CRITICAL)
        
        return config
    
    @staticmethod
    def production() -> EnhancedMessageBusConfig:
        """Production configuration"""
        config = EnhancedMessageBusConfig(
            connection_pool_size=50,
            command_timeout=60.0,
            enable_metrics=True,
            enable_tracing=False,
            auto_scale_enabled=True,
            max_workers=20
        )
        
        # High-performance buffer for critical messages
        critical_buffer = BufferConfig(
            max_size=50000,
            flush_interval_ms=10,
            high_water_mark=40000,
            enable_compression=True
        )
        config.priority_buffers[MessagePriority.CRITICAL] = critical_buffer
        
        # Production subscriptions
        config.add_subscription("trading.*", MessagePriority.CRITICAL)
        config.add_subscription("risk.*", MessagePriority.HIGH)
        config.add_subscription("market-data.*", MessagePriority.HIGH)
        config.add_subscription("analytics.*", MessagePriority.NORMAL)
        
        return config
    
    @staticmethod
    def high_frequency() -> EnhancedMessageBusConfig:
        """High-frequency trading configuration"""
        config = EnhancedMessageBusConfig(
            connection_pool_size=100,
            command_timeout=1.0,
            connection_timeout=1.0,
            enable_metrics=True,
            auto_scale_enabled=True,
            max_workers=50
        )
        
        # Ultra-fast buffer for HFT
        hft_buffer = BufferConfig(
            max_size=100000,
            flush_interval_ms=1,
            high_water_mark=80000,
            enable_compression=False  # Speed over size
        )
        
        config.priority_buffers[MessagePriority.CRITICAL] = hft_buffer
        config.default_buffer_config = BufferConfig(flush_interval_ms=10)
        
        # HFT-specific subscriptions
        config.add_subscription("tick.*", MessagePriority.CRITICAL)
        config.add_subscription("order.*", MessagePriority.CRITICAL)
        config.add_subscription("execution.*", MessagePriority.CRITICAL)
        
        return config

# Environment-based configuration loader
def load_config() -> EnhancedMessageBusConfig:
    """Load configuration based on environment"""
    env = os.getenv('NAUTILUS_ENV', 'development').lower()
    config_file = os.getenv('NAUTILUS_MESSAGEBUS_CONFIG')
    
    if config_file and os.path.exists(config_file):
        return EnhancedMessageBusConfig.from_file(config_file)
    
    if env == 'production':
        return ConfigPresets.production()
    elif env == 'hft':
        return ConfigPresets.high_frequency()
    else:
        return ConfigPresets.development()


# =============================================================================
# CONFIGURATION VALIDATION SYSTEM
# =============================================================================

@dataclass
class ValidationResult:
    """Configuration validation result"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


class ConfigValidator:
    """Enhanced MessageBus configuration validator"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.ConfigValidator")
        
        # Define validation rules
        self.required_fields = [
            'redis_host', 'redis_port', 'connection_pool_size'
        ]
        
        self.deprecated_fields = {
            'old_redis_url': 'Use redis_host and redis_port instead',
            'legacy_buffer_size': 'Use default_buffer_config.max_size instead',
        }
        
        # Value constraints
        self.constraints = {
            'redis_port': (1, 65535),
            'connection_pool_size': (1, 1000),
            'connection_timeout': (0.1, 300.0),
            'command_timeout': (0.1, 600.0),
            'max_workers': (1, 200),
            'min_workers': (1, None),
            'trace_sample_rate': (0.0, 1.0),
            'scale_up_threshold': (0.1, 1.0),
            'scale_down_threshold': (0.0, 0.9),
        }
    
    def validate(self, config_data: Union[Dict, EnhancedMessageBusConfig]) -> ValidationResult:
        """Validate configuration"""
        result = ValidationResult(is_valid=True)
        
        # Convert config object to dict if needed
        if isinstance(config_data, EnhancedMessageBusConfig):
            data = config_data.to_dict()
        else:
            data = config_data
        
        # Check required fields
        self._validate_required_fields(data, result)
        
        # Check deprecated fields
        self._validate_deprecated_fields(data, result)
        
        # Check value constraints
        self._validate_constraints(data, result)
        
        # Validate buffer configurations
        self._validate_buffer_configs(data, result)
        
        # Validate stream configurations
        self._validate_stream_configs(data, result)
        
        # Validate subscriptions
        self._validate_subscriptions(data, result)
        
        # Check logical consistency
        self._validate_consistency(data, result)
        
        # Performance recommendations
        self._generate_recommendations(data, result)
        
        return result
    
    def _validate_required_fields(self, data: Dict, result: ValidationResult):
        """Validate required fields are present"""
        for field in self.required_fields:
            if field not in data or data[field] is None:
                result.errors.append(f"Required field '{field}' is missing or None")
                result.is_valid = False
    
    def _validate_deprecated_fields(self, data: Dict, result: ValidationResult):
        """Check for deprecated fields"""
        for deprecated_field, message in self.deprecated_fields.items():
            if deprecated_field in data:
                result.warnings.append(f"Deprecated field '{deprecated_field}': {message}")
    
    def _validate_constraints(self, data: Dict, result: ValidationResult):
        """Validate value constraints"""
        for field, (min_val, max_val) in self.constraints.items():
            if field in data:
                value = data[field]
                if not isinstance(value, (int, float)):
                    continue
                
                if min_val is not None and value < min_val:
                    result.errors.append(f"Field '{field}' value {value} is below minimum {min_val}")
                    result.is_valid = False
                
                if max_val is not None and value > max_val:
                    result.errors.append(f"Field '{field}' value {value} is above maximum {max_val}")
                    result.is_valid = False
    
    def _validate_buffer_configs(self, data: Dict, result: ValidationResult):
        """Validate buffer configurations"""
        # Check default buffer config
        if 'default_buffer_config' in data:
            self._validate_single_buffer_config(data['default_buffer_config'], 'default_buffer_config', result)
        
        # Check priority buffers
        if 'priority_buffers' in data:
            for priority, buffer_config in data['priority_buffers'].items():
                self._validate_single_buffer_config(buffer_config, f'priority_buffers.{priority}', result)
    
    def _validate_single_buffer_config(self, buffer_config: Union[Dict, BufferConfig], 
                                     config_name: str, result: ValidationResult):
        """Validate a single buffer configuration"""
        if isinstance(buffer_config, BufferConfig):
            config_dict = buffer_config.__dict__
        else:
            config_dict = buffer_config
        
        # Check max_size constraints
        max_size = config_dict.get('max_size', 0)
        if max_size <= 0:
            result.errors.append(f"{config_name}.max_size must be positive, got {max_size}")
            result.is_valid = False
        
        # Check water marks
        high_water = config_dict.get('high_water_mark', 0)
        low_water = config_dict.get('low_water_mark', 0)
        
        if high_water >= max_size:
            result.errors.append(f"{config_name}.high_water_mark ({high_water}) must be less than max_size ({max_size})")
            result.is_valid = False
        
        if low_water >= high_water:
            result.errors.append(f"{config_name}.low_water_mark ({low_water}) must be less than high_water_mark ({high_water})")
            result.is_valid = False
        
        # Check flush interval
        flush_interval = config_dict.get('flush_interval_ms', 0)
        if flush_interval <= 0 or flush_interval > 60000:  # Max 1 minute
            result.warnings.append(f"{config_name}.flush_interval_ms ({flush_interval}) may cause performance issues")
    
    def _validate_stream_configs(self, data: Dict, result: ValidationResult):
        """Validate stream configurations"""
        if 'stream_configs' in data:
            for stream_name, stream_config in data['stream_configs'].items():
                self._validate_single_stream_config(stream_config, stream_name, result)
    
    def _validate_single_stream_config(self, stream_config: Union[Dict, StreamConfig], 
                                     stream_name: str, result: ValidationResult):
        """Validate a single stream configuration"""
        if isinstance(stream_config, StreamConfig):
            config_dict = stream_config.__dict__
        else:
            config_dict = stream_config
        
        # Check max_length
        max_length = config_dict.get('max_length', 0)
        if max_length < 0:
            result.errors.append(f"Stream '{stream_name}' max_length cannot be negative")
            result.is_valid = False
        elif max_length > 1000000:  # 1M messages
            result.warnings.append(f"Stream '{stream_name}' max_length ({max_length}) is very large and may impact performance")
        
        # Check batch_size
        batch_size = config_dict.get('batch_size', 0)
        if batch_size <= 0 or batch_size > 10000:
            result.warnings.append(f"Stream '{stream_name}' batch_size ({batch_size}) may cause performance issues")
    
    def _validate_subscriptions(self, data: Dict, result: ValidationResult):
        """Validate subscription patterns"""
        if 'subscriptions' in data:
            patterns = set()
            for i, subscription in enumerate(data['subscriptions']):
                if isinstance(subscription, PatternSubscription):
                    pattern = subscription.pattern
                elif isinstance(subscription, dict):
                    pattern = subscription.get('pattern', '')
                else:
                    result.errors.append(f"Subscription {i} has invalid format")
                    result.is_valid = False
                    continue
                
                if not pattern:
                    result.errors.append(f"Subscription {i} has empty pattern")
                    result.is_valid = False
                    continue
                
                if pattern in patterns:
                    result.warnings.append(f"Duplicate subscription pattern: '{pattern}'")
                
                patterns.add(pattern)
                
                # Validate pattern syntax
                try:
                    TopicPattern(pattern)
                except Exception as e:
                    result.errors.append(f"Invalid pattern '{pattern}': {str(e)}")
                    result.is_valid = False
    
    def _validate_consistency(self, data: Dict, result: ValidationResult):
        """Validate logical consistency between fields"""
        # Check worker scaling settings
        min_workers = data.get('min_workers', 1)
        max_workers = data.get('max_workers', 10)
        
        if min_workers > max_workers:
            result.errors.append(f"min_workers ({min_workers}) cannot be greater than max_workers ({max_workers})")
            result.is_valid = False
        
        # Check scaling thresholds
        scale_up = data.get('scale_up_threshold', 0.8)
        scale_down = data.get('scale_down_threshold', 0.3)
        
        if scale_down >= scale_up:
            result.errors.append(f"scale_down_threshold ({scale_down}) must be less than scale_up_threshold ({scale_up})")
            result.is_valid = False
        
        # Check auto-scaling with worker limits
        if data.get('auto_scale_enabled', True) and min_workers == max_workers:
            result.warnings.append("Auto-scaling is enabled but min_workers equals max_workers")
    
    def _generate_recommendations(self, data: Dict, result: ValidationResult):
        """Generate performance recommendations"""
        # Connection pool size recommendation
        pool_size = data.get('connection_pool_size', 20)
        max_workers = data.get('max_workers', 10)
        
        if pool_size < max_workers * 2:
            result.suggestions.append(f"Consider increasing connection_pool_size to at least {max_workers * 2} for optimal performance")
        
        # Memory usage warnings
        total_buffer_size = 0
        if 'default_buffer_config' in data:
            total_buffer_size += data['default_buffer_config'].get('max_size', 10000)
        
        if 'priority_buffers' in data:
            for buffer_config in data['priority_buffers'].values():
                total_buffer_size += buffer_config.get('max_size', 10000)
        
        estimated_memory_mb = total_buffer_size * 1024 / (1024 * 1024)  # Rough estimate
        if estimated_memory_mb > 1000:  # > 1GB
            result.warnings.append(f"Estimated buffer memory usage: {estimated_memory_mb:.1f}MB - monitor memory consumption")
        
        # Performance recommendations
        if not data.get('enable_metrics', True):
            result.suggestions.append("Consider enabling metrics for performance monitoring")
        
        if data.get('enable_tracing', False) and data.get('trace_sample_rate', 0.1) == 1.0:
            result.warnings.append("100% trace sampling may impact performance in production")


# =============================================================================
# CONFIGURATION MIGRATION SYSTEM
# =============================================================================

class ConfigMigration:
    """Configuration migration utilities"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.ConfigMigration")
        
        # Define migration rules for different versions
        self.migration_rules = {
            '1.0': self._migrate_from_v1_0,
            '1.1': self._migrate_from_v1_1,
            'legacy': self._migrate_from_legacy,
        }
    
    def migrate_config(self, old_config: Dict, source_version: Optional[str] = None) -> Tuple[EnhancedMessageBusConfig, List[str]]:
        """
        Migrate configuration from older version
        
        Args:
            old_config: Old configuration dictionary
            source_version: Source version identifier
            
        Returns:
            Tuple of (migrated_config, migration_messages)
        """
        messages = []
        
        # Auto-detect version if not provided
        if source_version is None:
            source_version = self._detect_version(old_config)
            messages.append(f"Auto-detected source version: {source_version}")
        
        # Apply migration rules
        if source_version in self.migration_rules:
            migrated_data = self.migration_rules[source_version](old_config, messages)
        else:
            self.logger.warning(f"No migration rules for version {source_version}, using as-is")
            migrated_data = old_config
            messages.append(f"No migration rules for version {source_version}")
        
        # Validate migrated configuration
        validator = ConfigValidator()
        validation_result = validator.validate(migrated_data)
        
        if not validation_result.is_valid:
            raise ValueError(f"Migration failed validation: {validation_result.errors}")
        
        # Add validation warnings to messages
        messages.extend(validation_result.warnings)
        
        # Create enhanced config
        try:
            enhanced_config = EnhancedMessageBusConfig.from_dict(migrated_data)
            messages.append("Successfully migrated to EnhancedMessageBusConfig")
            return enhanced_config, messages
        except Exception as e:
            raise ValueError(f"Failed to create EnhancedMessageBusConfig: {str(e)}")
    
    def _detect_version(self, config: Dict) -> str:
        """Auto-detect configuration version"""
        # Check for version-specific fields
        if 'messagebus_version' in config:
            return config['messagebus_version']
        
        # Legacy NautilusTrader MessageBusConfig detection
        if 'database' in config and isinstance(config['database'], dict):
            return 'legacy'
        
        # Check for v1.1 specific fields
        if 'priority_buffers' in config:
            return '1.1'
        
        # Check for v1.0 specific fields
        if 'buffer_config' in config:
            return '1.0'
        
        # Default to legacy
        return 'legacy'
    
    def _migrate_from_legacy(self, old_config: Dict, messages: List[str]) -> Dict:
        """Migrate from legacy NautilusTrader MessageBusConfig"""
        messages.append("Migrating from legacy NautilusTrader MessageBusConfig")
        
        migrated = {}
        
        # Extract database configuration
        if 'database' in old_config:
            db_config = old_config['database']
            migrated.update({
                'redis_host': db_config.get('host', 'localhost'),
                'redis_port': db_config.get('port', 6379),
                'redis_db': db_config.get('database', 0),
                'redis_password': db_config.get('password'),
                'connection_pool_size': db_config.get('pool_size', 20),
            })
            messages.append("Extracted Redis settings from database config")
        
        # Map other legacy fields
        field_mappings = {
            'heartbeat_interval': 'health_check_interval',
            'timeout': 'command_timeout',
            'max_connections': 'connection_pool_size',
        }
        
        for old_field, new_field in field_mappings.items():
            if old_field in old_config:
                migrated[new_field] = old_config[old_field]
                messages.append(f"Mapped {old_field} -> {new_field}")
        
        # Set enhanced defaults
        migrated.update({
            'enable_metrics': True,
            'auto_scale_enabled': True,
            'enable_pattern_matching': True,
            'max_workers': 10,
            'min_workers': 1,
        })
        
        return migrated
    
    def _migrate_from_v1_0(self, old_config: Dict, messages: List[str]) -> Dict:
        """Migrate from version 1.0"""
        messages.append("Migrating from EnhancedMessageBus v1.0")
        
        migrated = old_config.copy()
        
        # Update buffer configuration structure
        if 'buffer_config' in old_config:
            migrated['default_buffer_config'] = old_config['buffer_config']
            del migrated['buffer_config']
            messages.append("Renamed buffer_config to default_buffer_config")
        
        # Add new v1.1 features
        if 'priority_buffers' not in migrated:
            migrated['priority_buffers'] = {}
        
        if 'stream_configs' not in migrated:
            migrated['stream_configs'] = {}
        
        # Add new monitoring settings
        migrated.update({
            'enable_health_checks': migrated.get('enable_health_checks', True),
            'monitoring_interval': 5,
            'auto_cleanup_enabled': True,
        })
        
        return migrated
    
    def _migrate_from_v1_1(self, old_config: Dict, messages: List[str]) -> Dict:
        """Migrate from version 1.1 (minimal changes)"""
        messages.append("Migrating from EnhancedMessageBus v1.1")
        
        migrated = old_config.copy()
        
        # Add any new fields with defaults
        new_fields = {
            'compression_enabled': True,
            'compression_level': 6,
            'default_serializer': 'json',
        }
        
        for field, default_value in new_fields.items():
            if field not in migrated:
                migrated[field] = default_value
                messages.append(f"Added new field {field} with default value")
        
        return migrated


# =============================================================================
# MIGRATION UTILITIES
# =============================================================================

def migrate_from_nautilus_config(nautilus_config: Any) -> Tuple[EnhancedMessageBusConfig, List[str]]:
    """
    Migrate from standard NautilusTrader MessageBusConfig
    
    Args:
        nautilus_config: NautilusTrader MessageBusConfig instance or dict
        
    Returns:
        Tuple of (enhanced_config, migration_messages)
    """
    migration = ConfigMigration()
    
    # Convert config to dict if it's an object
    if hasattr(nautilus_config, '__dict__'):
        config_dict = nautilus_config.__dict__
    elif hasattr(nautilus_config, 'to_dict'):
        config_dict = nautilus_config.to_dict()
    else:
        config_dict = nautilus_config
    
    return migration.migrate_config(config_dict, 'legacy')


def create_migration_guide(old_config_path: str, new_config_path: str) -> str:
    """
    Create a migration guide for configuration upgrade
    
    Args:
        old_config_path: Path to old configuration file
        new_config_path: Path where new configuration will be saved
        
    Returns:
        Migration guide text
    """
    guide = f"""
# Enhanced MessageBus Configuration Migration Guide

## Source Configuration
File: {old_config_path}

## Target Configuration  
File: {new_config_path}

## Migration Steps

1. **Backup Original Configuration**
   ```bash
   cp {old_config_path} {old_config_path}.backup
   ```

2. **Run Migration Script**
   ```python
   from nautilus_trader.infrastructure.messagebus.config import ConfigMigration
   
   migration = ConfigMigration()
   
   # Load old configuration
   with open("{old_config_path}", 'r') as f:
       old_config = json.load(f)
   
   # Migrate configuration
   new_config, messages = migration.migrate_config(old_config)
   
   # Save new configuration
   new_config.save_to_file("{new_config_path}")
   
   # Review migration messages
   for message in messages:
       print(f"MIGRATION: {{message}}")
   ```

3. **Validate New Configuration**
   ```python
   from nautilus_trader.infrastructure.messagebus.config import ConfigValidator
   
   validator = ConfigValidator()
   result = validator.validate(new_config)
   
   if result.is_valid:
       print("✅ Configuration is valid")
   else:
       print("❌ Configuration errors:", result.errors)
   ```

4. **Update Application Code**
   - Replace `MessageBusConfig` imports with `EnhancedMessageBusConfig`
   - Update initialization code to use new configuration
   - Review new features and optimize settings for your use case

## New Features Available After Migration
- Priority-based message buffering
- Pattern-based topic subscriptions  
- Auto-scaling worker management
- Enhanced performance monitoring
- Redis Streams integration
- Health checking and auto-recovery

For more details, see the Enhanced MessageBus documentation.
"""
    
    return guide.strip()