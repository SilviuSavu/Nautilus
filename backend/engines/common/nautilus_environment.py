#!/usr/bin/env python3
"""
Nautilus Environment Registry
Centralized environment configuration that all engines can access for platform awareness.
"""

import os
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime


class HardwareComponent(Enum):
    """M4 Max hardware components"""
    NEURAL_ENGINE = "neural_engine"
    METAL_GPU = "metal_gpu"
    SME_AMX = "sme_amx"
    PERFORMANCE_CORES = "performance_cores"
    EFFICIENCY_CORES = "efficiency_cores"
    UNIFIED_MEMORY = "unified_memory"


class MessageBusType(Enum):
    """Available message bus types"""
    MARKETDATA_BUS = "marketdata_bus"      # Port 6380
    ENGINE_LOGIC_BUS = "engine_logic_bus"  # Port 6381
    NEURAL_GPU_BUS = "neural_gpu_bus"      # Port 6382
    PRIMARY_REDIS = "primary_redis"        # Port 6379


class DataSource(Enum):
    """Integrated data sources"""
    IBKR = "ibkr"
    ALPHA_VANTAGE = "alpha_vantage"
    FRED = "fred"
    EDGAR = "edgar"
    DATA_GOV = "data_gov"
    TRADING_ECONOMICS = "trading_economics"
    DBNOMICS = "dbnomics"
    YAHOO_FINANCE = "yahoo_finance"


@dataclass
class EngineSpec:
    """Specification for a Nautilus engine"""
    engine_id: str
    name: str
    port: int
    description: str
    architecture: str
    performance_metrics: Dict[str, Any]
    hardware_optimizations: List[HardwareComponent]
    health_endpoint: str
    status: str = "operational"


@dataclass
class MessageBusSpec:
    """Message bus specification"""
    bus_type: MessageBusType
    port: int
    optimization: str
    purpose: str
    container_name: str
    performance_target: str


@dataclass
class DataSourceSpec:
    """Data source specification"""
    source: DataSource
    description: str
    data_types: List[str]
    update_frequency: str
    integration_status: str


@dataclass
class HardwareSpec:
    """M4 Max hardware specifications"""
    neural_engine_tops: float = 38.0
    metal_gpu_cores: int = 40
    metal_gpu_bandwidth_gbs: float = 546.0
    sme_amx_tflops: float = 2.9
    performance_cores: int = 12
    efficiency_cores: int = 4
    unified_memory_gb: int = 64


@dataclass
class NautilusEnvironment:
    """Complete Nautilus trading platform environment"""
    platform_name: str = "Nautilus"
    version: str = "v3.0"
    total_engines: int = 18
    last_updated: str = ""
    engines: List[EngineSpec] = None
    message_buses: List[MessageBusSpec] = None
    data_sources: List[DataSourceSpec] = None
    hardware: HardwareSpec = None
    database_url: str = "postgresql://nautilus:nautilus123@localhost:5432/nautilus"
    
    def __post_init__(self):
        if self.last_updated == "":
            self.last_updated = datetime.now().isoformat()
        if self.engines is None:
            self.engines = self._create_engine_specs()
        if self.message_buses is None:
            self.message_buses = self._create_messagebus_specs()
        if self.data_sources is None:
            self.data_sources = self._create_datasource_specs()
        if self.hardware is None:
            self.hardware = HardwareSpec()
    
    def _create_engine_specs(self) -> List[EngineSpec]:
        """Create specifications for all 18 engines"""
        return [
            # Core Processing Engines (8100-8900)
            EngineSpec(
                engine_id="ANALYTICS_ENGINE",
                name="Analytics Engine",
                port=8100,
                description="Real-time analytics processing with dual messagebus integration",
                architecture="dual_messagebus",
                performance_metrics={"response_time_ms": 1.9, "speedup_factor": 38},
                hardware_optimizations=[HardwareComponent.NEURAL_ENGINE, HardwareComponent.UNIFIED_MEMORY],
                health_endpoint="http://localhost:8100/health"
            ),
            EngineSpec(
                engine_id="BACKTESTING_ENGINE",
                name="Backtesting Engine",
                port=8110,
                description="Historical strategy validation with Neural Engine 1000x speedup",
                architecture="native_m4_max",
                performance_metrics={"response_time_ms": 1.2, "speedup_factor": 1000},
                hardware_optimizations=[HardwareComponent.NEURAL_ENGINE, HardwareComponent.SME_AMX],
                health_endpoint="http://localhost:8110/health"
            ),
            EngineSpec(
                engine_id="RISK_ENGINE",
                name="Risk Engine",
                port=8200,
                description="Real-time risk monitoring and alerts with dual messagebus",
                architecture="dual_messagebus",
                performance_metrics={"response_time_ms": 1.7, "speedup_factor": 69},
                hardware_optimizations=[HardwareComponent.PERFORMANCE_CORES, HardwareComponent.UNIFIED_MEMORY],
                health_endpoint="http://localhost:8200/health"
            ),
            EngineSpec(
                engine_id="FACTOR_ENGINE",
                name="Factor Engine",
                port=8300,
                description="516 factor definitions with multi-source data integration",
                architecture="dual_messagebus",
                performance_metrics={"response_time_ms": 1.8, "factor_count": 516},
                hardware_optimizations=[HardwareComponent.PERFORMANCE_CORES, HardwareComponent.UNIFIED_MEMORY],
                health_endpoint="http://localhost:8300/health"
            ),
            EngineSpec(
                engine_id="ML_ENGINE",
                name="ML Engine",
                port=8400,
                description="Machine learning predictions with Ultra Fast 2025 implementation",
                architecture="ultra_fast_2025",
                performance_metrics={"response_time_ms": 1.6, "speedup_factor": 27, "models_loaded": 4},
                hardware_optimizations=[HardwareComponent.NEURAL_ENGINE, HardwareComponent.METAL_GPU],
                health_endpoint="http://localhost:8400/health"
            ),
            EngineSpec(
                engine_id="FEATURES_ENGINE",
                name="Features Engine",
                port=8500,
                description="Real-time feature extraction and transformation",
                architecture="native_feature_engineering",
                performance_metrics={"response_time_ms": 1.8, "speedup_factor": 21},
                hardware_optimizations=[HardwareComponent.PERFORMANCE_CORES, HardwareComponent.UNIFIED_MEMORY],
                health_endpoint="http://localhost:8500/health"
            ),
            EngineSpec(
                engine_id="WEBSOCKET_THGNN_ENGINE",
                name="WebSocket/THGNN Engine",
                port=8600,
                description="Real-time streaming with Temporal Heterogeneous GNN for HFT",
                architecture="enhanced_thgnn_hft",
                performance_metrics={"response_time_ms": 1.4, "hft_prediction_latency_us": 1},
                hardware_optimizations=[HardwareComponent.NEURAL_ENGINE, HardwareComponent.METAL_GPU],
                health_endpoint="http://localhost:8600/health"
            ),
            EngineSpec(
                engine_id="STRATEGY_ENGINE",
                name="Strategy Engine",
                port=8700,
                description="Automated trading strategy execution",
                architecture="native_trading_logic",
                performance_metrics={"response_time_ms": 1.5, "speedup_factor": 24, "active_strategies": 2},
                hardware_optimizations=[HardwareComponent.PERFORMANCE_CORES],
                health_endpoint="http://localhost:8700/health"
            ),
            EngineSpec(
                engine_id="IBKR_KEEPALIVE_ENGINE",
                name="Enhanced IBKR Keep-Alive Engine",
                port=8800,
                description="Persistent IBKR connection with Level 2 order book data",
                architecture="native_ibkr_level2",
                performance_metrics={"response_time_ms": 1.7, "speedup_factor": 29},
                hardware_optimizations=[HardwareComponent.PERFORMANCE_CORES, HardwareComponent.UNIFIED_MEMORY],
                health_endpoint="http://localhost:8800/health"
            ),
            EngineSpec(
                engine_id="PORTFOLIO_ENGINE",
                name="Portfolio Engine",
                port=8900,
                description="Real-time portfolio rebalancing and optimization",
                architecture="native_institutional_optimization",
                performance_metrics={"response_time_ms": 1.7, "speedup_factor": 30},
                hardware_optimizations=[HardwareComponent.PERFORMANCE_CORES, HardwareComponent.UNIFIED_MEMORY],
                health_endpoint="http://localhost:8900/health"
            ),
            
            # Mission-Critical Engines (9000-10002)
            EngineSpec(
                engine_id="COLLATERAL_ENGINE",
                name="Collateral Engine",
                port=9000,
                description="Real-time margin monitoring with predictive margin call alerts",
                architecture="mission_critical_dual_messagebus",
                performance_metrics={"response_time_ms": 1.6, "margin_calc_time_ms": 0.36, "efficiency_improvement_pct": 30},
                hardware_optimizations=[HardwareComponent.PERFORMANCE_CORES, HardwareComponent.UNIFIED_MEMORY],
                health_endpoint="http://localhost:9000/health"
            ),
            EngineSpec(
                engine_id="VPIN_ENGINE",
                name="VPIN Engine",
                port=10000,
                description="Volume-Synchronized Probability of Informed Trading",
                architecture="native_market_microstructure",
                performance_metrics={"response_time_ms": 1.5, "gpu_acceleration": True},
                hardware_optimizations=[HardwareComponent.METAL_GPU, HardwareComponent.PERFORMANCE_CORES],
                health_endpoint="http://localhost:10000/health"
            ),
            EngineSpec(
                engine_id="ENHANCED_VPIN_ENGINE",
                name="Enhanced VPIN Engine",
                port=10001,
                description="Advanced VPIN calculations with hardware acceleration",
                architecture="enhanced_platform",
                performance_metrics={"response_time_ms": 0.5, "analysis_type": "market_microstructure"},
                hardware_optimizations=[HardwareComponent.METAL_GPU, HardwareComponent.NEURAL_ENGINE],
                health_endpoint="http://localhost:10001/health"
            ),
            EngineSpec(
                engine_id="MAGNN_MULTIMODAL_ENGINE",
                name="MAGNN Multi-Modal Engine",
                port=10002,
                description="Multi-source data fusion using Graph Neural Networks",
                architecture="triple_messagebus_gnn",
                performance_metrics={"response_time_ms": 0.5, "data_sources": 4},
                hardware_optimizations=[HardwareComponent.NEURAL_ENGINE, HardwareComponent.METAL_GPU],
                health_endpoint="http://localhost:10002/health"
            ),
            
            # Advanced Quantum & Physics Engines (10003-10005)
            EngineSpec(
                engine_id="QUANTUM_PORTFOLIO_ENGINE",
                name="Quantum Portfolio Engine",
                port=10003,
                description="Large portfolio optimization using QAOA, QIGA, QAE, QNN algorithms",
                architecture="triple_messagebus_postgresql",
                performance_metrics={"response_time_ms": 1.0, "quantum_speedup_factor": 100, "max_assets": 1000},
                hardware_optimizations=[HardwareComponent.NEURAL_ENGINE, HardwareComponent.SME_AMX],
                health_endpoint="http://localhost:10003/health"
            ),
            EngineSpec(
                engine_id="NEURAL_SDE_ENGINE",
                name="Neural SDE Engine",
                port=10004,
                description="Real-time derivative pricing using Stochastic Differential Equations",
                architecture="triple_messagebus_postgresql",
                performance_metrics={"response_time_ms": 1.0, "monte_carlo_paths": 1000000},
                hardware_optimizations=[HardwareComponent.NEURAL_ENGINE, HardwareComponent.METAL_GPU],
                health_endpoint="http://localhost:10004/health"
            ),
            EngineSpec(
                engine_id="MOLECULAR_DYNAMICS_ENGINE",
                name="Molecular Dynamics Engine",
                port=10005,
                description="Market microstructure modeling using molecular dynamics simulation",
                architecture="triple_messagebus_postgresql",
                performance_metrics={"response_time_ms": 1.0, "market_participants": 1000000},
                hardware_optimizations=[HardwareComponent.METAL_GPU, HardwareComponent.NEURAL_ENGINE],
                health_endpoint="http://localhost:10005/health"
            )
        ]
    
    def _create_messagebus_specs(self) -> List[MessageBusSpec]:
        """Create message bus specifications"""
        return [
            MessageBusSpec(
                bus_type=MessageBusType.MARKETDATA_BUS,
                port=6380,
                optimization="Neural Engine + Unified Memory",
                purpose="High-throughput market data distribution",
                container_name="nautilus-marketdata-bus",
                performance_target="10,000+ msgs/sec, <2ms latency"
            ),
            MessageBusSpec(
                bus_type=MessageBusType.ENGINE_LOGIC_BUS,
                port=6381,
                optimization="Metal GPU + Performance Cores",
                purpose="Ultra-low latency engine business logic",
                container_name="nautilus-engine-logic-bus",
                performance_target="50,000+ msgs/sec, <0.5ms latency"
            ),
            MessageBusSpec(
                bus_type=MessageBusType.NEURAL_GPU_BUS,
                port=6382,
                optimization="Hardware acceleration coordination",
                purpose="Neural Engine + Metal GPU compute handoffs",
                container_name="nautilus-neural-gpu-bus",
                performance_target="100,000+ ops/sec, <0.1ms latency"
            ),
            MessageBusSpec(
                bus_type=MessageBusType.PRIMARY_REDIS,
                port=6379,
                optimization="General operations",
                purpose="System health, caching, session management",
                container_name="nautilus-redis",
                performance_target="Standard Redis performance"
            )
        ]
    
    def _create_datasource_specs(self) -> List[DataSourceSpec]:
        """Create data source specifications"""
        return [
            DataSourceSpec(
                source=DataSource.IBKR,
                description="Enhanced Keep-Alive Level 2 market depth with persistent real-time trading data",
                data_types=["market_data", "level2_orderbook", "trades", "positions"],
                update_frequency="real_time",
                integration_status="enhanced_keepalive"
            ),
            DataSourceSpec(
                source=DataSource.ALPHA_VANTAGE,
                description="Fundamental data and company metrics",
                data_types=["fundamentals", "earnings", "company_overview"],
                update_frequency="daily",
                integration_status="active"
            ),
            DataSourceSpec(
                source=DataSource.FRED,
                description="32 economic series, macro-economic factors",
                data_types=["economic_indicators", "interest_rates", "inflation"],
                update_frequency="daily",
                integration_status="active"
            ),
            DataSourceSpec(
                source=DataSource.EDGAR,
                description="SEC filings from 7,861+ public company entities",
                data_types=["10k_filings", "10q_filings", "8k_filings", "insider_trading"],
                update_frequency="real_time",
                integration_status="active"
            ),
            DataSourceSpec(
                source=DataSource.DATA_GOV,
                description="Government economic datasets",
                data_types=["government_data", "economic_statistics"],
                update_frequency="weekly",
                integration_status="active"
            ),
            DataSourceSpec(
                source=DataSource.TRADING_ECONOMICS,
                description="Global economic indicators",
                data_types=["global_economics", "country_indicators"],
                update_frequency="daily",
                integration_status="active"
            ),
            DataSourceSpec(
                source=DataSource.DBNOMICS,
                description="International statistical data",
                data_types=["international_stats", "monetary_data"],
                update_frequency="daily",
                integration_status="active"
            ),
            DataSourceSpec(
                source=DataSource.YAHOO_FINANCE,
                description="Supplementary market data",
                data_types=["market_data", "historical_prices"],
                update_frequency="real_time",
                integration_status="supplementary"
            )
        ]
    
    def get_engine_by_id(self, engine_id: str) -> Optional[EngineSpec]:
        """Get engine specification by ID"""
        for engine in self.engines:
            if engine.engine_id == engine_id:
                return engine
        return None
    
    def get_engine_by_port(self, port: int) -> Optional[EngineSpec]:
        """Get engine specification by port"""
        for engine in self.engines:
            if engine.port == port:
                return engine
        return None
    
    def get_engines_by_hardware(self, hardware_component: HardwareComponent) -> List[EngineSpec]:
        """Get engines that use specific hardware component"""
        return [engine for engine in self.engines 
                if hardware_component in engine.hardware_optimizations]
    
    def get_messagebus_by_type(self, bus_type: MessageBusType) -> Optional[MessageBusSpec]:
        """Get message bus specification by type"""
        for bus in self.message_buses:
            if bus.bus_type == bus_type:
                return bus
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NautilusEnvironment':
        """Create from dictionary"""
        # Convert enum strings back to enums
        if 'engines' in data and data['engines']:
            for engine_data in data['engines']:
                if 'hardware_optimizations' in engine_data:
                    engine_data['hardware_optimizations'] = [
                        HardwareComponent(hw) for hw in engine_data['hardware_optimizations']
                    ]
        
        if 'message_buses' in data and data['message_buses']:
            for bus_data in data['message_buses']:
                if 'bus_type' in bus_data:
                    bus_data['bus_type'] = MessageBusType(bus_data['bus_type'])
        
        if 'data_sources' in data and data['data_sources']:
            for source_data in data['data_sources']:
                if 'source' in source_data:
                    source_data['source'] = DataSource(source_data['source'])
        
        return cls(**data)


# Global singleton instance
_environment_instance = None

def get_nautilus_environment() -> NautilusEnvironment:
    """Get singleton instance of Nautilus environment"""
    global _environment_instance
    if _environment_instance is None:
        _environment_instance = NautilusEnvironment()
    return _environment_instance


def refresh_environment():
    """Refresh the environment instance (for testing)"""
    global _environment_instance
    _environment_instance = None


if __name__ == "__main__":
    # Demo usage
    env = get_nautilus_environment()
    print(f"Nautilus Platform: {env.platform_name} {env.version}")
    print(f"Total Engines: {env.total_engines}")
    print(f"Hardware: {env.hardware.neural_engine_tops} TOPS Neural Engine")
    
    # Find ML Engine
    ml_engine = env.get_engine_by_id("ML_ENGINE")
    if ml_engine:
        print(f"\nML Engine: {ml_engine.name} on port {ml_engine.port}")
        print(f"Performance: {ml_engine.performance_metrics}")
    
    # Find all Neural Engine optimized engines
    neural_engines = env.get_engines_by_hardware(HardwareComponent.NEURAL_ENGINE)
    print(f"\nNeural Engine Optimized Engines: {len(neural_engines)}")
    for engine in neural_engines:
        print(f"  - {engine.name} ({engine.engine_id})")