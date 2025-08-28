#!/usr/bin/env python3
"""
Engine Self-Awareness Module
Provides each engine with self-knowledge capabilities and identity management.
"""

import os
import json
import uuid
import asyncio
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum
from datetime import datetime
import platform
import psutil


class EngineRole(Enum):
    """Primary roles engines can fulfill"""
    DATA_PROVIDER = "data_provider"
    DATA_PROCESSOR = "data_processor"
    ANALYTICS = "analytics"
    RISK_MANAGEMENT = "risk_management"
    TRADING_EXECUTION = "trading_execution"
    PORTFOLIO_MANAGEMENT = "portfolio_management"
    MARKET_ANALYSIS = "market_analysis"
    QUANTUM_COMPUTING = "quantum_computing"
    MACHINE_LEARNING = "machine_learning"
    PHYSICS_SIMULATION = "physics_simulation"


class DataFormat(Enum):
    """Data formats engines can handle"""
    JSON = "json"
    PARQUET = "parquet"
    NUMPY_ARRAY = "numpy_array"
    PANDAS_DATAFRAME = "pandas_dataframe"
    TORCH_TENSOR = "torch_tensor"
    BINARY = "binary"
    TIME_SERIES = "time_series"
    ORDER_BOOK = "order_book"
    FINANCIAL_OHLCV = "financial_ohlcv"


class ProcessingCapability(Enum):
    """Processing capabilities engines can provide"""
    REAL_TIME_STREAMING = "real_time_streaming"
    BATCH_PROCESSING = "batch_processing"
    FEATURE_ENGINEERING = "feature_engineering"
    RISK_CALCULATION = "risk_calculation"
    PORTFOLIO_OPTIMIZATION = "portfolio_optimization"
    MARKET_MICROSTRUCTURE = "market_microstructure"
    MACHINE_LEARNING_INFERENCE = "ml_inference"
    QUANTUM_OPTIMIZATION = "quantum_optimization"
    PHYSICS_SIMULATION = "physics_simulation"
    HIGH_FREQUENCY_TRADING = "high_frequency_trading"
    BACKTESTING = "backtesting"
    STRATEGY_EXECUTION = "strategy_execution"
    FACTOR_CALCULATION = "factor_calculation"
    VOLATILITY_MODELING = "volatility_modeling"
    DERIVATIVE_PRICING = "derivative_pricing"
    COLLATERAL_MANAGEMENT = "collateral_management"


@dataclass
class DataSchema:
    """Schema definition for input/output data"""
    name: str
    format: DataFormat
    required_fields: List[str]
    optional_fields: List[str] = field(default_factory=list)
    description: str = ""
    example: Optional[Dict[str, Any]] = None


@dataclass
class PerformanceMetrics:
    """Engine performance characteristics"""
    average_response_time_ms: float
    max_throughput_per_sec: int
    cpu_utilization_pct: float
    memory_usage_mb: int
    hardware_acceleration_factor: float
    availability_pct: float = 99.9
    last_updated: str = ""
    
    def __post_init__(self):
        if not self.last_updated:
            self.last_updated = datetime.now().isoformat()


@dataclass
class PartnershipPreference:
    """Preference for partnering with other engines"""
    engine_id: str
    relationship_type: str  # "primary", "secondary", "optional"
    data_flow_direction: str  # "input", "output", "bidirectional"
    preferred_message_types: List[str]
    latency_requirement_ms: float
    reliability_requirement_pct: float
    description: str


@dataclass
class EngineCapabilities:
    """Complete capability profile of an engine"""
    supported_roles: List[EngineRole]
    processing_capabilities: List[ProcessingCapability]
    input_data_schemas: List[DataSchema]
    output_data_schemas: List[DataSchema]
    partnership_preferences: List[PartnershipPreference]
    hardware_requirements: List[str]
    software_dependencies: List[str]


@dataclass
class EngineHealth:
    """Current health status of the engine"""
    status: str  # "healthy", "degraded", "critical", "offline"
    last_heartbeat: str
    uptime_seconds: float
    error_count_last_hour: int
    warning_count_last_hour: int
    current_load_pct: float
    memory_usage_pct: float
    connection_count: int
    
    def __post_init__(self):
        if not self.last_heartbeat:
            self.last_heartbeat = datetime.now().isoformat()


class EngineIdentity:
    """Complete identity and self-awareness for a Nautilus engine"""
    
    def __init__(
        self,
        engine_id: str,
        engine_name: str,
        engine_port: int,
        capabilities: EngineCapabilities,
        version: str = "1.0.0"
    ):
        self.engine_id = engine_id
        self.engine_name = engine_name
        self.engine_port = engine_port
        self.capabilities = capabilities
        self.version = version
        
        # Generate unique instance ID for this run
        self.instance_id = f"{engine_id}_{uuid.uuid4().hex[:8]}"
        
        # Initialize runtime state
        self.startup_time = datetime.now()
        self.performance_metrics = PerformanceMetrics(
            average_response_time_ms=1.0,
            max_throughput_per_sec=1000,
            cpu_utilization_pct=0.0,
            memory_usage_mb=0,
            hardware_acceleration_factor=1.0
        )
        self.health = EngineHealth(
            status="starting",
            last_heartbeat=datetime.now().isoformat(),
            uptime_seconds=0.0,
            error_count_last_hour=0,
            warning_count_last_hour=0,
            current_load_pct=0.0,
            memory_usage_pct=0.0,
            connection_count=0
        )
        
        # Partnership tracking
        self.active_partnerships: Dict[str, Dict[str, Any]] = {}
        self.partnership_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self._performance_history: List[Dict[str, Any]] = []
        self._last_metrics_update = datetime.now()
    
    def update_health_status(self):
        """Update current health status"""
        try:
            # Get system metrics
            process = psutil.Process()
            
            self.health.last_heartbeat = datetime.now().isoformat()
            self.health.uptime_seconds = (datetime.now() - self.startup_time).total_seconds()
            self.health.current_load_pct = process.cpu_percent()
            self.health.memory_usage_pct = process.memory_percent()
            
            # Determine overall health status
            if self.health.current_load_pct > 90 or self.health.memory_usage_pct > 90:
                self.health.status = "critical"
            elif self.health.current_load_pct > 70 or self.health.memory_usage_pct > 70:
                self.health.status = "degraded"
            else:
                self.health.status = "healthy"
                
        except Exception as e:
            self.health.status = "error"
            self.health.error_count_last_hour += 1
    
    def update_performance_metrics(self, response_time_ms: float, throughput: int):
        """Update performance metrics with new measurements"""
        # Exponential moving average for response time
        alpha = 0.1
        self.performance_metrics.average_response_time_ms = (
            alpha * response_time_ms + 
            (1 - alpha) * self.performance_metrics.average_response_time_ms
        )
        
        # Update max throughput if higher
        if throughput > self.performance_metrics.max_throughput_per_sec:
            self.performance_metrics.max_throughput_per_sec = throughput
        
        self.performance_metrics.last_updated = datetime.now().isoformat()
        
        # Store history
        self._performance_history.append({
            "timestamp": datetime.now().isoformat(),
            "response_time_ms": response_time_ms,
            "throughput": throughput
        })
        
        # Keep only last 1000 entries
        if len(self._performance_history) > 1000:
            self._performance_history = self._performance_history[-1000:]
    
    def add_partnership(self, partner_engine_id: str, relationship_data: Dict[str, Any]):
        """Record a new partnership"""
        self.active_partnerships[partner_engine_id] = {
            "established_at": datetime.now().isoformat(),
            "relationship_data": relationship_data,
            "message_count": 0,
            "last_interaction": datetime.now().isoformat()
        }
        
        # Record in history
        self.partnership_history.append({
            "action": "established",
            "partner_id": partner_engine_id,
            "timestamp": datetime.now().isoformat(),
            "data": relationship_data
        })
    
    def update_partnership_activity(self, partner_engine_id: str):
        """Update partnership activity"""
        if partner_engine_id in self.active_partnerships:
            self.active_partnerships[partner_engine_id]["message_count"] += 1
            self.active_partnerships[partner_engine_id]["last_interaction"] = datetime.now().isoformat()
    
    def get_preferred_partners(self, relationship_type: str = None) -> List[str]:
        """Get list of preferred partners"""
        preferences = self.capabilities.partnership_preferences
        
        if relationship_type:
            preferences = [p for p in preferences if p.relationship_type == relationship_type]
        
        return [p.engine_id for p in preferences]
    
    def can_process_data_format(self, data_format: DataFormat) -> bool:
        """Check if engine can process specific data format"""
        input_formats = [schema.format for schema in self.capabilities.input_data_schemas]
        return data_format in input_formats
    
    def has_capability(self, capability: ProcessingCapability) -> bool:
        """Check if engine has specific processing capability"""
        return capability in self.capabilities.processing_capabilities
    
    def get_compatibility_score(self, other_engine: 'EngineIdentity') -> float:
        """Calculate compatibility score with another engine (0.0-1.0)"""
        score = 0.0
        
        # Check data format compatibility
        my_outputs = {schema.format for schema in self.capabilities.output_data_schemas}
        other_inputs = {schema.format for schema in other_engine.capabilities.input_data_schemas}
        
        format_compatibility = len(my_outputs.intersection(other_inputs)) / max(len(my_outputs), 1)
        score += format_compatibility * 0.4
        
        # Check if other engine is in preferred partners
        preferred_partners = [p.engine_id for p in self.capabilities.partnership_preferences]
        if other_engine.engine_id in preferred_partners:
            score += 0.4
        
        # Check complementary capabilities
        my_capabilities = set(self.capabilities.processing_capabilities)
        other_capabilities = set(other_engine.capabilities.processing_capabilities)
        
        # Reward complementary (non-overlapping) capabilities
        complementary = len(my_capabilities.symmetric_difference(other_capabilities))
        total_capabilities = len(my_capabilities.union(other_capabilities))
        
        if total_capabilities > 0:
            score += (complementary / total_capabilities) * 0.2
        
        return min(score, 1.0)
    
    def to_discovery_announcement(self) -> Dict[str, Any]:
        """Create announcement message for engine discovery"""
        return {
            "engine_id": self.engine_id,
            "engine_name": self.engine_name,
            "instance_id": self.instance_id,
            "port": self.engine_port,
            "version": self.version,
            "status": self.health.status,
            "roles": [role.value for role in self.capabilities.supported_roles],
            "capabilities": [cap.value for cap in self.capabilities.processing_capabilities],
            "input_formats": [schema.format.value for schema in self.capabilities.input_data_schemas],
            "output_formats": [schema.format.value for schema in self.capabilities.output_data_schemas],
            "preferred_partners": [p.engine_id for p in self.capabilities.partnership_preferences],
            "hardware_requirements": self.capabilities.hardware_requirements,
            "performance": {
                "response_time_ms": self.performance_metrics.average_response_time_ms,
                "throughput": self.performance_metrics.max_throughput_per_sec,
                "acceleration_factor": self.performance_metrics.hardware_acceleration_factor
            },
            "health": {
                "status": self.health.status,
                "uptime_seconds": self.health.uptime_seconds,
                "load_pct": self.health.current_load_pct
            },
            "announced_at": datetime.now().isoformat()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert complete identity to dictionary"""
        return {
            "engine_id": self.engine_id,
            "engine_name": self.engine_name,
            "instance_id": self.instance_id,
            "engine_port": self.engine_port,
            "version": self.version,
            "startup_time": self.startup_time.isoformat(),
            "capabilities": asdict(self.capabilities),
            "performance_metrics": asdict(self.performance_metrics),
            "health": asdict(self.health),
            "active_partnerships": self.active_partnerships,
            "partnership_history": self.partnership_history[-50:]  # Last 50 entries
        }
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2, default=str)


# Predefined engine identity templates
def create_ml_engine_identity() -> EngineIdentity:
    """Create identity for ML Engine"""
    capabilities = EngineCapabilities(
        supported_roles=[EngineRole.MACHINE_LEARNING, EngineRole.DATA_PROCESSOR],
        processing_capabilities=[
            ProcessingCapability.MACHINE_LEARNING_INFERENCE,
            ProcessingCapability.REAL_TIME_STREAMING,
            ProcessingCapability.FEATURE_ENGINEERING
        ],
        input_data_schemas=[
            DataSchema(
                name="features",
                format=DataFormat.PANDAS_DATAFRAME,
                required_fields=["timestamp", "symbol"],
                optional_fields=["volume", "price"],
                description="Feature vectors for ML prediction"
            ),
            DataSchema(
                name="market_data",
                format=DataFormat.FINANCIAL_OHLCV,
                required_fields=["open", "high", "low", "close", "volume"],
                description="Market data for real-time inference"
            )
        ],
        output_data_schemas=[
            DataSchema(
                name="predictions",
                format=DataFormat.JSON,
                required_fields=["symbol", "prediction", "confidence", "timestamp"],
                description="ML predictions with confidence scores"
            )
        ],
        partnership_preferences=[
            PartnershipPreference(
                engine_id="FEATURES_ENGINE",
                relationship_type="primary",
                data_flow_direction="input",
                preferred_message_types=["FEATURE_CALCULATION"],
                latency_requirement_ms=10.0,
                reliability_requirement_pct=99.5,
                description="Primary feature provider for ML models"
            ),
            PartnershipPreference(
                engine_id="ANALYTICS_ENGINE",
                relationship_type="secondary",
                data_flow_direction="output",
                preferred_message_types=["ML_PREDICTION"],
                latency_requirement_ms=5.0,
                reliability_requirement_pct=99.0,
                description="Send predictions for analytics processing"
            )
        ],
        hardware_requirements=["neural_engine", "metal_gpu"],
        software_dependencies=["torch", "numpy", "pandas"]
    )
    
    return EngineIdentity(
        engine_id="ML_ENGINE",
        engine_name="ML Engine",
        engine_port=8400,
        capabilities=capabilities,
        version="2025.1.0"
    )


def create_risk_engine_identity() -> EngineIdentity:
    """Create identity for Risk Engine"""
    capabilities = EngineCapabilities(
        supported_roles=[EngineRole.RISK_MANAGEMENT, EngineRole.ANALYTICS],
        processing_capabilities=[
            ProcessingCapability.RISK_CALCULATION,
            ProcessingCapability.REAL_TIME_STREAMING,
            ProcessingCapability.VOLATILITY_MODELING
        ],
        input_data_schemas=[
            DataSchema(
                name="portfolio_positions",
                format=DataFormat.JSON,
                required_fields=["symbol", "quantity", "market_value"],
                description="Current portfolio positions for risk calculation"
            )
        ],
        output_data_schemas=[
            DataSchema(
                name="risk_metrics",
                format=DataFormat.JSON,
                required_fields=["var", "expected_shortfall", "beta", "timestamp"],
                description="Risk metrics and alerts"
            )
        ],
        partnership_preferences=[
            PartnershipPreference(
                engine_id="PORTFOLIO_ENGINE",
                relationship_type="primary",
                data_flow_direction="bidirectional",
                preferred_message_types=["PORTFOLIO_UPDATE", "RISK_METRIC"],
                latency_requirement_ms=5.0,
                reliability_requirement_pct=99.9,
                description="Primary portfolio risk monitoring"
            )
        ],
        hardware_requirements=["performance_cores"],
        software_dependencies=["numpy", "scipy", "pandas"]
    )
    
    return EngineIdentity(
        engine_id="RISK_ENGINE",
        engine_name="Risk Engine",
        engine_port=8200,
        capabilities=capabilities,
        version="2025.1.0"
    )


if __name__ == "__main__":
    # Demo usage
    ml_engine = create_ml_engine_identity()
    risk_engine = create_risk_engine_identity()
    
    print("=== ML Engine Identity ===")
    print(f"Engine ID: {ml_engine.engine_id}")
    print(f"Capabilities: {[cap.value for cap in ml_engine.capabilities.processing_capabilities]}")
    print(f"Preferred Partners: {ml_engine.get_preferred_partners()}")
    
    print("\n=== Risk Engine Identity ===")
    print(f"Engine ID: {risk_engine.engine_id}")
    print(f"Roles: {[role.value for role in risk_engine.capabilities.supported_roles]}")
    
    print(f"\n=== Compatibility Score ===")
    compatibility = ml_engine.get_compatibility_score(risk_engine)
    print(f"ML <-> Risk compatibility: {compatibility:.2f}")
    
    print(f"\n=== Discovery Announcement ===")
    announcement = ml_engine.to_discovery_announcement()
    print(json.dumps(announcement, indent=2))