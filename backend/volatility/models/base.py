"""
Base Volatility Model Classes

This module defines the abstract base class for all volatility forecasting models
and common data structures.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum

logger = logging.getLogger(__name__)


class ForecastHorizon(Enum):
    """Forecast horizon types"""
    INTRADAY = "intraday"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class ModelStatus(Enum):
    """Model status indicators"""
    UNINITIALIZED = "uninitialized"
    TRAINING = "training"
    READY = "ready"
    FORECASTING = "forecasting"
    ERROR = "error"
    STALE = "stale"


@dataclass
class VolatilityForecast:
    """Volatility forecast result"""
    symbol: str
    model_name: str
    model_version: str
    forecast_timestamp: datetime
    
    # Forecast values
    forecast_volatility: float  # Annualized volatility
    forecast_variance: float   # Variance
    forecast_std_error: float  # Standard error of forecast
    
    # Forecast horizon and confidence
    horizon_days: int
    confidence_level: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    
    # Model performance metrics
    in_sample_rmse: float
    out_sample_rmse: float
    likelihood: float
    aic: float
    bic: float
    
    # Additional information
    model_parameters: Dict[str, float] = field(default_factory=dict)
    risk_metrics: Dict[str, float] = field(default_factory=dict)
    regime_state: Optional[str] = None
    market_conditions: Dict[str, Any] = field(default_factory=dict)
    computation_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert forecast to dictionary"""
        return {
            "symbol": self.symbol,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "forecast_timestamp": self.forecast_timestamp.isoformat(),
            "forecast_volatility": self.forecast_volatility,
            "forecast_variance": self.forecast_variance,
            "forecast_std_error": self.forecast_std_error,
            "horizon_days": self.horizon_days,
            "confidence_level": self.confidence_level,
            "confidence_interval_lower": self.confidence_interval_lower,
            "confidence_interval_upper": self.confidence_interval_upper,
            "in_sample_rmse": self.in_sample_rmse,
            "out_sample_rmse": self.out_sample_rmse,
            "likelihood": self.likelihood,
            "aic": self.aic,
            "bic": self.bic,
            "model_parameters": self.model_parameters,
            "risk_metrics": self.risk_metrics,
            "regime_state": self.regime_state,
            "market_conditions": self.market_conditions,
            "computation_time_ms": self.computation_time_ms
        }


@dataclass
class ModelMetrics:
    """Model performance and diagnostic metrics"""
    model_name: str
    symbol: str
    evaluation_date: datetime
    
    # Accuracy metrics
    rmse: float
    mae: float
    mape: float  # Mean Absolute Percentage Error
    r_squared: float
    
    # Information criteria
    aic: float
    bic: float
    hqic: float
    log_likelihood: float
    
    # Residual diagnostics
    ljung_box_p_value: float  # Serial correlation test
    arch_lm_p_value: float    # Heteroscedasticity test
    jarque_bera_p_value: float # Normality test
    
    # Forecast evaluation
    directional_accuracy: float  # % of correct directional forecasts
    hit_ratio: float            # % of forecasts within confidence interval
    coverage_ratio: float       # Confidence interval coverage
    
    # Performance metrics
    training_time_ms: float
    prediction_time_ms: float
    memory_usage_mb: float
    cpu_utilization: float
    
    # Model stability
    parameter_stability: float   # Coefficient of variation of parameters
    forecast_stability: float    # Stability of forecasts over time
    
    # Optional fields with defaults
    gpu_utilization: float = 0.0
    
    def is_model_adequate(self) -> bool:
        """Check if model passes basic adequacy tests"""
        # Check residual diagnostics (p-values should be > 0.05 for no issues)
        residuals_ok = (
            self.ljung_box_p_value > 0.05 and
            self.arch_lm_p_value > 0.05 and
            self.jarque_bera_p_value > 0.01  # Less strict for normality
        )
        
        # Check accuracy
        accuracy_ok = self.rmse < 0.1 and self.r_squared > 0.1
        
        # Check forecast quality
        forecast_ok = (
            self.directional_accuracy > 0.5 and
            self.coverage_ratio > 0.85
        )
        
        return residuals_ok and accuracy_ok and forecast_ok


class BaseVolatilityModel(ABC):
    """
    Abstract base class for all volatility forecasting models.
    
    This class defines the interface that all volatility models must implement,
    including training, forecasting, and evaluation methods.
    """
    
    def __init__(self, 
                 symbol: str,
                 model_name: str,
                 model_version: str = "1.0.0",
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the volatility model.
        
        Args:
            symbol: Trading symbol (e.g., 'AAPL', 'SPY')
            model_name: Name of the model (e.g., 'GARCH', 'LSTM')
            model_version: Version of the model implementation
            config: Model-specific configuration parameters
        """
        self.symbol = symbol
        self.model_name = model_name
        self.model_version = model_version
        self.config = config or {}
        
        # Model state
        self.status = ModelStatus.UNINITIALIZED
        self.last_training_date: Optional[datetime] = None
        self.last_forecast_date: Optional[datetime] = None
        self.training_data: Optional[pd.DataFrame] = None
        self.model_object: Optional[Any] = None
        
        # Performance tracking
        self.metrics_history: List[ModelMetrics] = []
        self.forecast_history: List[VolatilityForecast] = []
        
        # Hardware acceleration flags
        self.use_gpu = config.get('use_gpu', False) if config else False
        self.use_neural_engine = config.get('use_neural_engine', False) if config else False
        
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{model_name}")
    
    @abstractmethod
    async def prepare_data(self, 
                          price_data: pd.DataFrame,
                          **kwargs) -> pd.DataFrame:
        """
        Prepare data for model training and forecasting.
        
        Args:
            price_data: Price data with OHLCV columns
            **kwargs: Additional parameters specific to model
            
        Returns:
            Processed data ready for model consumption
        """
        pass
    
    @abstractmethod
    async def train(self, 
                   data: pd.DataFrame,
                   validation_split: float = 0.2,
                   **kwargs) -> Dict[str, Any]:
        """
        Train the volatility model.
        
        Args:
            data: Training data prepared by prepare_data()
            validation_split: Fraction of data to use for validation
            **kwargs: Model-specific training parameters
            
        Returns:
            Training results including metrics and model parameters
        """
        pass
    
    @abstractmethod
    async def forecast(self, 
                      data: Optional[pd.DataFrame] = None,
                      horizon: int = 5,
                      confidence_level: float = 0.95,
                      **kwargs) -> VolatilityForecast:
        """
        Generate volatility forecast.
        
        Args:
            data: Recent data for forecasting (if None, use last training data)
            horizon: Forecast horizon in days
            confidence_level: Confidence level for prediction intervals
            **kwargs: Model-specific forecasting parameters
            
        Returns:
            Volatility forecast with confidence intervals and metrics
        """
        pass
    
    @abstractmethod
    async def evaluate(self, 
                      test_data: pd.DataFrame,
                      forecast_horizon: int = 5) -> ModelMetrics:
        """
        Evaluate model performance on test data.
        
        Args:
            test_data: Out-of-sample test data
            forecast_horizon: Evaluation horizon in days
            
        Returns:
            Model performance metrics
        """
        pass
    
    async def update(self, 
                    new_data: pd.DataFrame,
                    retrain_threshold: float = 0.1) -> bool:
        """
        Update model with new data.
        
        Args:
            new_data: New market data
            retrain_threshold: RMSE threshold for triggering retraining
            
        Returns:
            True if model was retrained, False if just updated
        """
        try:
            # Check if retraining is needed
            if self.needs_retraining(new_data, retrain_threshold):
                # Combine old and new data for retraining
                if self.training_data is not None:
                    combined_data = pd.concat([self.training_data, new_data])
                else:
                    combined_data = new_data
                
                # Retrain model
                self.status = ModelStatus.TRAINING
                await self.train(combined_data)
                self.status = ModelStatus.READY
                return True
            else:
                # Just update with new data
                self.training_data = pd.concat([self.training_data, new_data])
                return False
                
        except Exception as e:
            self.logger.error(f"Error updating model: {e}")
            self.status = ModelStatus.ERROR
            raise
    
    def needs_retraining(self, 
                        new_data: pd.DataFrame,
                        threshold: float) -> bool:
        """
        Determine if model needs retraining based on performance degradation.
        
        Args:
            new_data: New market data
            threshold: Performance degradation threshold
            
        Returns:
            True if retraining is needed
        """
        # Simple heuristics for retraining decision
        if not self.metrics_history:
            return True
            
        # Check if it's been too long since last training
        if self.last_training_date:
            days_since_training = (datetime.utcnow() - self.last_training_date).days
            if days_since_training > 30:  # Retrain monthly
                return True
        
        # Check if recent performance has degraded
        if len(self.metrics_history) >= 2:
            recent_rmse = self.metrics_history[-1].rmse
            baseline_rmse = np.mean([m.rmse for m in self.metrics_history[-5:]])
            if recent_rmse > baseline_rmse * (1 + threshold):
                return True
                
        return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        return {
            "symbol": self.symbol,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "status": self.status.value,
            "last_training_date": self.last_training_date.isoformat() if self.last_training_date else None,
            "last_forecast_date": self.last_forecast_date.isoformat() if self.last_forecast_date else None,
            "config": self.config,
            "use_gpu": self.use_gpu,
            "use_neural_engine": self.use_neural_engine,
            "training_data_size": len(self.training_data) if self.training_data is not None else 0,
            "metrics_history_count": len(self.metrics_history),
            "forecast_history_count": len(self.forecast_history)
        }
    
    def get_latest_metrics(self) -> Optional[ModelMetrics]:
        """Get the most recent model performance metrics"""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_latest_forecast(self) -> Optional[VolatilityForecast]:
        """Get the most recent volatility forecast"""
        return self.forecast_history[-1] if self.forecast_history else None
    
    async def cleanup(self) -> None:
        """Clean up model resources"""
        self.training_data = None
        self.model_object = None
        self.status = ModelStatus.UNINITIALIZED
        self.logger.info(f"Model {self.model_name} for {self.symbol} cleaned up")
    
    def __str__(self) -> str:
        return f"{self.model_name}Model(symbol={self.symbol}, status={self.status.value})"
    
    def __repr__(self) -> str:
        return self.__str__()