"""
Volatility Forecasting Configuration

This module defines configuration settings for all volatility forecasting components.
"""

import os
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, field
from datetime import timedelta


class ModelType(Enum):
    """Types of volatility models available"""
    # Traditional econometric models
    GARCH = "garch"
    EGARCH = "egarch" 
    GJR_GARCH = "gjr_garch"
    TARCH = "tarch"
    FIGARCH = "figarch"
    
    # Stochastic volatility models
    HESTON = "heston"
    SABR = "sabr"
    KARASINSKI_SEPP = "karasinski_sepp"
    LOG_NORMAL_SV = "log_normal_sv"
    
    # Machine learning models
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    CONV_LSTM = "conv_lstm"
    RANDOM_FOREST = "random_forest"
    
    # Real-time estimators
    GARMAN_KLASS = "garman_klass"
    YANG_ZHANG = "yang_zhang"
    ROGERS_SATCHELL = "rogers_satchell"
    HODGES_TOMPKINS = "hodges_tompkins"


class EstimatorType(Enum):
    """Real-time volatility estimator types"""
    CLOSE_TO_CLOSE = "close_to_close"
    PARKINSON = "parkinson"
    GARMAN_KLASS_YZ = "garman_klass_yz"
    YANG_ZHANG = "yang_zhang"
    ROGERS_SATCHELL = "rogers_satchell"
    HODGES_TOMPKINS = "hodges_tompkins"


class EnsembleMethod(Enum):
    """Model ensemble combination methods"""
    EQUAL_WEIGHT = "equal_weight"
    VARIANCE_WEIGHT = "variance_weight"
    BAYESIAN_AVERAGE = "bayesian_average"
    DYNAMIC_WEIGHT = "dynamic_weight"
    REGIME_BASED = "regime_based"


@dataclass
class ModelConfig:
    """Configuration for individual volatility models"""
    model_type: ModelType
    enabled: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)
    update_frequency: timedelta = field(default=timedelta(minutes=5))
    lookback_days: int = 252
    forecast_horizon: int = 5  # days
    use_gpu: bool = True
    confidence_level: float = 0.95


@dataclass
class EnsembleConfig:
    """Configuration for model ensemble"""
    method: EnsembleMethod = EnsembleMethod.DYNAMIC_WEIGHT
    models: List[ModelType] = field(default_factory=list)
    rebalance_frequency: timedelta = field(default=timedelta(hours=1))
    min_models: int = 3
    max_models: int = 8
    weight_decay: float = 0.95
    use_confidence_weighting: bool = True


@dataclass
class HardwareConfig:
    """M4 Max hardware acceleration configuration"""
    use_metal_gpu: bool = True
    use_neural_engine: bool = True
    use_cpu_optimization: bool = True
    auto_hardware_routing: bool = True
    metal_gpu_threshold: int = 1000000  # Switch to GPU for > 1M calculations
    neural_engine_threshold: int = 10000  # Use Neural Engine for ML models > 10K samples
    max_gpu_memory_gb: float = 16.0
    parallel_models: int = 4


@dataclass
class DataConfig:
    """Data source and storage configuration"""
    data_sources: List[str] = field(default_factory=lambda: [
        "ibkr", "alpha_vantage", "fred", "edgar", "data_gov", 
        "trading_economics", "dbnomics", "yfinance"
    ])
    primary_source: str = "ibkr"
    cache_ttl_seconds: int = 300  # 5 minutes
    max_cache_size_mb: int = 1000
    use_arcticdb: bool = True
    redis_stream_key: str = "nautilus-volatility-streams"
    postgres_table: str = "volatility_forecasts"


@dataclass
class APIConfig:
    """API and service configuration"""
    container_port: int = 9000
    host: str = "0.0.0.0"
    workers: int = 4
    log_level: str = "INFO"
    enable_websocket: bool = True
    websocket_max_connections: int = 1000
    rate_limit_per_minute: int = 1000
    max_forecast_horizon_days: int = 30
    min_confidence_threshold: float = 0.5


@dataclass
class VolatilityConfig:
    """Main volatility forecasting configuration"""
    # Component configurations
    models: Dict[ModelType, ModelConfig] = field(default_factory=dict)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    data: DataConfig = field(default_factory=DataConfig)
    api: APIConfig = field(default_factory=APIConfig)
    
    # Global settings
    debug_mode: bool = False
    performance_monitoring: bool = True
    save_intermediate_results: bool = True
    model_validation_enabled: bool = True
    
    @classmethod
    def from_environment(cls) -> 'VolatilityConfig':
        """Create configuration from environment variables"""
        config = cls()
        
        # Hardware configuration from environment
        config.hardware.use_metal_gpu = os.getenv("METAL_ACCELERATION", "1") == "1"
        config.hardware.use_neural_engine = os.getenv("NEURAL_ENGINE_ENABLED", "1") == "1"
        config.hardware.use_cpu_optimization = os.getenv("CPU_OPTIMIZATION", "1") == "1"
        config.hardware.auto_hardware_routing = os.getenv("AUTO_HARDWARE_ROUTING", "1") == "1"
        
        # API configuration
        config.api.container_port = int(os.getenv("VOLATILITY_PORT", "9000"))
        config.api.log_level = os.getenv("LOG_LEVEL", "INFO")
        config.api.enable_websocket = os.getenv("WEBSOCKET_ENABLED", "1") == "1"
        
        # Data configuration
        config.data.use_arcticdb = os.getenv("USE_ARCTICDB", "1") == "1"
        config.data.cache_ttl_seconds = int(os.getenv("CACHE_TTL", "300"))
        
        # Debug settings
        config.debug_mode = os.getenv("VOLATILITY_DEBUG", "0") == "1"
        config.performance_monitoring = os.getenv("PERFORMANCE_MONITORING", "1") == "1"
        
        return config

    def get_model_config(self, model_type: ModelType) -> Optional[ModelConfig]:
        """Get configuration for a specific model"""
        return self.models.get(model_type)
    
    def add_model_config(self, model_type: ModelType, config: ModelConfig) -> None:
        """Add or update model configuration"""
        self.models[model_type] = config
    
    def get_enabled_models(self) -> List[ModelType]:
        """Get list of enabled model types"""
        return [model_type for model_type, config in self.models.items() if config.enabled]
    
    def validate_config(self) -> bool:
        """Validate configuration settings"""
        # Ensure at least one model is enabled
        if not self.get_enabled_models():
            return False
            
        # Validate ensemble configuration
        if self.ensemble.min_models > len(self.get_enabled_models()):
            return False
            
        # Validate hardware settings
        if self.hardware.max_gpu_memory_gb <= 0:
            return False
            
        return True


def get_default_volatility_config() -> VolatilityConfig:
    """Get default volatility forecasting configuration"""
    config = VolatilityConfig()
    
    # Default GARCH model configuration
    config.add_model_config(ModelType.GARCH, ModelConfig(
        model_type=ModelType.GARCH,
        parameters={"p": 1, "q": 1},
        lookback_days=252,
        forecast_horizon=5,
        use_gpu=True
    ))
    
    # Default GJR-GARCH for asymmetric volatility
    config.add_model_config(ModelType.GJR_GARCH, ModelConfig(
        model_type=ModelType.GJR_GARCH,
        parameters={"p": 1, "o": 1, "q": 1},
        lookback_days=252,
        forecast_horizon=5,
        use_gpu=True
    ))
    
    # Real-time estimators
    config.add_model_config(ModelType.YANG_ZHANG, ModelConfig(
        model_type=ModelType.YANG_ZHANG,
        update_frequency=timedelta(seconds=30),
        lookback_days=21,
        forecast_horizon=1,
        use_gpu=False
    ))
    
    # LSTM model for pattern recognition
    config.add_model_config(ModelType.LSTM, ModelConfig(
        model_type=ModelType.LSTM,
        parameters={
            "hidden_size": 128, 
            "num_layers": 2, 
            "dropout": 0.2,
            "sequence_length": 60,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100,
            "use_neural_engine": True
        },
        lookback_days=252,
        forecast_horizon=10,
        use_gpu=True
    ))
    
    # Transformer model for advanced sequence modeling
    config.add_model_config(ModelType.TRANSFORMER, ModelConfig(
        model_type=ModelType.TRANSFORMER,
        parameters={
            "d_model": 128,
            "nhead": 8,
            "num_layers": 6,
            "dropout": 0.1,
            "sequence_length": 60,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 50,
            "use_neural_engine": True
        },
        lookback_days=252,
        forecast_horizon=10,
        use_gpu=True
    ))
    
    # Configure ensemble
    config.ensemble.models = [
        ModelType.GARCH,
        ModelType.GJR_GARCH,
        ModelType.YANG_ZHANG,
        ModelType.LSTM,
        ModelType.TRANSFORMER
    ]
    
    return config