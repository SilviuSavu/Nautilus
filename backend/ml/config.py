"""
Machine Learning Configuration System
Centralized configuration management for all ML components
"""

import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import json


class ModelType(Enum):
    """Supported ML model types"""
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    ENSEMBLE = "ensemble"
    ISOLATION_FOREST = "isolation_forest"
    AUTOENCODER = "autoencoder"
    SVM = "svm"
    LOGISTIC_REGRESSION = "logistic_regression"


class RegimeType(Enum):
    """Market regime types"""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    CRISIS = "crisis"
    RECOVERY = "recovery"


@dataclass
class ModelConfig:
    """Configuration for individual ML models"""
    model_type: ModelType
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    feature_columns: List[str] = field(default_factory=list)
    target_column: str = ""
    validation_split: float = 0.2
    test_split: float = 0.1
    random_state: int = 42
    max_training_samples: int = 100000
    min_training_samples: int = 1000


@dataclass 
class RegimeDetectionConfig:
    """Configuration for market regime detection"""
    lookback_window: int = 60  # days
    prediction_horizon: int = 5  # days
    regime_types: List[RegimeType] = field(default_factory=lambda: list(RegimeType))
    confidence_threshold: float = 0.7
    rebalance_frequency: str = "daily"  # daily, weekly, monthly
    ensemble_models: List[ModelType] = field(default_factory=lambda: [
        ModelType.RANDOM_FOREST,
        ModelType.GRADIENT_BOOSTING,
        ModelType.LSTM
    ])


@dataclass
class FeatureEngineeringConfig:
    """Configuration for feature engineering"""
    technical_indicators: List[str] = field(default_factory=lambda: [
        "sma_5", "sma_20", "sma_50", "sma_200",
        "ema_12", "ema_26",
        "rsi_14", "rsi_30",
        "macd", "macd_signal", "macd_histogram",
        "bollinger_upper", "bollinger_lower", "bollinger_width",
        "atr_14", "atr_30",
        "volume_sma_20", "volume_ratio",
        "price_momentum_5", "price_momentum_20",
        "volatility_10", "volatility_30"
    ])
    
    alternative_data: List[str] = field(default_factory=lambda: [
        "vix_level", "vix_change",
        "treasury_10y", "treasury_2y", "yield_curve_slope",
        "dollar_index", "oil_price", "gold_price",
        "sector_rotation", "market_breadth",
        "earnings_sentiment", "news_sentiment"
    ])
    
    correlation_window: int = 30  # days
    correlation_threshold: float = 0.8
    feature_selection_method: str = "mutual_info"  # mutual_info, chi2, f_score
    max_features: int = 50
    min_feature_importance: float = 0.01


@dataclass
class ModelLifecycleConfig:
    """Configuration for model lifecycle management"""
    retrain_frequency: str = "weekly"  # daily, weekly, monthly
    drift_detection_window: int = 100  # samples
    drift_threshold: float = 0.05
    performance_threshold: float = 0.02  # minimum performance degradation to trigger retrain
    max_model_age_days: int = 30
    model_validation_samples: int = 5000
    a_b_test_duration_days: int = 7
    champion_challenger_ratio: float = 0.8  # 80% champion, 20% challenger


@dataclass
class RiskPredictionConfig:
    """Configuration for risk prediction models"""
    var_confidence_levels: List[float] = field(default_factory=lambda: [0.95, 0.99, 0.999])
    var_horizons: List[int] = field(default_factory=lambda: [1, 5, 10])  # days
    stress_test_scenarios: List[str] = field(default_factory=lambda: [
        "market_crash", "sector_rotation", "interest_rate_spike",
        "liquidity_crisis", "currency_crisis", "geopolitical_shock"
    ])
    monte_carlo_simulations: int = 10000
    correlation_decay_factor: float = 0.94  # RiskMetrics standard
    portfolio_optimization_method: str = "mean_variance"  # mean_variance, black_litterman, risk_parity


@dataclass
class InferenceConfig:
    """Configuration for real-time inference"""
    max_latency_ms: int = 100
    batch_size: int = 32
    model_cache_size: int = 10  # number of models to keep in memory
    feature_cache_ttl: int = 300  # seconds
    prediction_cache_ttl: int = 60  # seconds
    monitoring_interval: int = 30  # seconds
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "prediction_latency": 0.1,  # seconds
        "model_confidence": 0.5,
        "feature_staleness": 300,  # seconds
        "error_rate": 0.05
    })


@dataclass
class MLConfig:
    """Main ML configuration class"""
    # Database and storage
    database_url: str = "postgresql://nautilus:nautilus123@localhost:5432/nautilus"
    redis_url: str = "redis://localhost:6379"
    model_storage_path: str = "/app/models"
    feature_store_path: str = "/app/features"
    
    # Component configurations
    regime_detection: RegimeDetectionConfig = field(default_factory=RegimeDetectionConfig)
    feature_engineering: FeatureEngineeringConfig = field(default_factory=FeatureEngineeringConfig)
    model_lifecycle: ModelLifecycleConfig = field(default_factory=ModelLifecycleConfig)
    risk_prediction: RiskPredictionConfig = field(default_factory=RiskPredictionConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    
    # Global settings
    random_seed: int = 42
    n_jobs: int = -1  # use all available cores
    verbose: bool = True
    log_level: str = "INFO"
    
    # Production settings
    enable_monitoring: bool = True
    enable_explainability: bool = True
    enable_a_b_testing: bool = True
    enable_drift_detection: bool = True
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MLConfig':
        """Create configuration from dictionary"""
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, json_path: str) -> 'MLConfig':
        """Load configuration from JSON file"""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_env(cls) -> 'MLConfig':
        """Create configuration from environment variables"""
        config = cls()
        
        # Override with environment variables if they exist
        if db_url := os.getenv('ML_DATABASE_URL'):
            config.database_url = db_url
        if redis_url := os.getenv('ML_REDIS_URL'):
            config.redis_url = redis_url
        if model_path := os.getenv('ML_MODEL_STORAGE_PATH'):
            config.model_storage_path = model_path
        if feature_path := os.getenv('ML_FEATURE_STORE_PATH'):
            config.feature_store_path = feature_path
        
        # Override monitoring settings
        if monitoring := os.getenv('ML_ENABLE_MONITORING'):
            config.enable_monitoring = monitoring.lower() == 'true'
        if explainability := os.getenv('ML_ENABLE_EXPLAINABILITY'):
            config.enable_explainability = explainability.lower() == 'true'
        if drift_detection := os.getenv('ML_ENABLE_DRIFT_DETECTION'):
            config.enable_drift_detection = drift_detection.lower() == 'true'
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        def convert_value(value):
            if hasattr(value, '__dict__'):
                return {k: convert_value(v) for k, v in value.__dict__.items()}
            elif isinstance(value, Enum):
                return value.value
            elif isinstance(value, list) and value and isinstance(value[0], Enum):
                return [item.value for item in value]
            else:
                return value
        
        return {k: convert_value(v) for k, v in self.__dict__.items()}
    
    def save_json(self, json_path: str) -> None:
        """Save configuration to JSON file"""
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Validate database URL
        if not self.database_url or not self.database_url.startswith(('postgresql://', 'sqlite://')):
            issues.append("Invalid database_url format")
        
        # Validate Redis URL
        if not self.redis_url or not self.redis_url.startswith('redis://'):
            issues.append("Invalid redis_url format")
        
        # Validate paths
        if not self.model_storage_path:
            issues.append("model_storage_path cannot be empty")
        if not self.feature_store_path:
            issues.append("feature_store_path cannot be empty")
        
        # Validate thresholds
        if not (0.0 <= self.regime_detection.confidence_threshold <= 1.0):
            issues.append("regime_detection.confidence_threshold must be between 0 and 1")
        
        if self.model_lifecycle.drift_threshold <= 0:
            issues.append("model_lifecycle.drift_threshold must be positive")
        
        if self.inference.max_latency_ms <= 0:
            issues.append("inference.max_latency_ms must be positive")
        
        # Validate feature engineering
        if self.feature_engineering.max_features <= 0:
            issues.append("feature_engineering.max_features must be positive")
        
        if not (0.0 <= self.feature_engineering.min_feature_importance <= 1.0):
            issues.append("feature_engineering.min_feature_importance must be between 0 and 1")
        
        return issues


# Global configuration instance
_ml_config: Optional[MLConfig] = None

def get_ml_config() -> MLConfig:
    """Get global ML configuration"""
    global _ml_config
    if _ml_config is None:
        _ml_config = MLConfig.from_env()
    return _ml_config

def set_ml_config(config: MLConfig) -> None:
    """Set global ML configuration"""
    global _ml_config
    _ml_config = config

# Default model configurations
DEFAULT_MODEL_CONFIGS = {
    ModelType.RANDOM_FOREST: ModelConfig(
        model_type=ModelType.RANDOM_FOREST,
        hyperparameters={
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'bootstrap': True,
            'oob_score': True
        }
    ),
    
    ModelType.GRADIENT_BOOSTING: ModelConfig(
        model_type=ModelType.GRADIENT_BOOSTING,
        hyperparameters={
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'subsample': 0.8,
            'max_features': 'sqrt'
        }
    ),
    
    ModelType.LSTM: ModelConfig(
        model_type=ModelType.LSTM,
        hyperparameters={
            'units': 50,
            'dropout': 0.2,
            'recurrent_dropout': 0.2,
            'sequence_length': 60,
            'batch_size': 32,
            'epochs': 50,
            'learning_rate': 0.001
        }
    ),
    
    ModelType.ISOLATION_FOREST: ModelConfig(
        model_type=ModelType.ISOLATION_FOREST,
        hyperparameters={
            'n_estimators': 100,
            'contamination': 0.1,
            'max_samples': 'auto',
            'max_features': 1.0,
            'bootstrap': False
        }
    ),
    
    ModelType.SVM: ModelConfig(
        model_type=ModelType.SVM,
        hyperparameters={
            'C': 1.0,
            'kernel': 'rbf',
            'gamma': 'scale',
            'probability': True,
            'cache_size': 200
        }
    )
}