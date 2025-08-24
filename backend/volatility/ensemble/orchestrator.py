"""
Ensemble Orchestrator

This module provides the main orchestration logic for the volatility forecasting engine.
It coordinates multiple models, manages their lifecycle, and produces ensemble forecasts.
"""

import asyncio
import logging
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque

# Internal imports
from ..models.base import BaseVolatilityModel, VolatilityForecast, ModelMetrics, ModelStatus
from ..models.garch_models import GARCHModel, EGARCHModel, GJRGARCHModel
from ..models.estimators import GarmanKlassEstimator, YangZhangEstimator, RogersSatchellEstimator
from ..models.stochastic_models import HestonModel, SABRModel
from ..config import VolatilityConfig, ModelType, EnsembleMethod
from .weights import DynamicWeightOptimizer, WeightingStrategy
from .confidence import ConfidenceAggregator

# Deep learning models (with fallback)
try:
    from ..models.deep_learning_models import (
        LSTMVolatilityPredictor, 
        TransformerVolatilityPredictor,
        DEEP_LEARNING_AVAILABLE,
        NEURAL_ENGINE_OPTIMIZATION_AVAILABLE
    )
except ImportError:
    LSTMVolatilityPredictor = None
    TransformerVolatilityPredictor = None
    DEEP_LEARNING_AVAILABLE = False
    NEURAL_ENGINE_OPTIMIZATION_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class EnsembleForecast:
    """Ensemble volatility forecast combining multiple models"""
    symbol: str
    forecast_timestamp: datetime
    ensemble_volatility: float
    ensemble_variance: float
    ensemble_confidence: float
    
    # Individual model contributions
    model_forecasts: Dict[str, VolatilityForecast] = field(default_factory=dict)
    model_weights: Dict[str, float] = field(default_factory=dict)
    model_contributions: Dict[str, float] = field(default_factory=dict)
    
    # Uncertainty quantification
    forecast_uncertainty: float = 0.0
    model_disagreement: float = 0.0
    confidence_interval_lower: float = 0.0
    confidence_interval_upper: float = 0.0
    
    # Performance metrics
    ensemble_method: str = ""
    computation_time_ms: float = 0.0
    models_used: int = 0
    hardware_acceleration: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "symbol": self.symbol,
            "forecast_timestamp": self.forecast_timestamp.isoformat(),
            "ensemble_volatility": self.ensemble_volatility,
            "ensemble_variance": self.ensemble_variance,
            "ensemble_confidence": self.ensemble_confidence,
            "model_weights": self.model_weights,
            "model_contributions": self.model_contributions,
            "forecast_uncertainty": self.forecast_uncertainty,
            "model_disagreement": self.model_disagreement,
            "confidence_interval_lower": self.confidence_interval_lower,
            "confidence_interval_upper": self.confidence_interval_upper,
            "ensemble_method": self.ensemble_method,
            "computation_time_ms": self.computation_time_ms,
            "models_used": self.models_used,
            "hardware_acceleration": self.hardware_acceleration
        }


class EnsembleOrchestrator:
    """
    Main orchestrator for the volatility forecasting engine.
    
    This class manages multiple volatility models, coordinates their training and forecasting,
    and produces ensemble forecasts with optimal weighting strategies.
    """
    
    def __init__(self, config: VolatilityConfig):
        """
        Initialize the ensemble orchestrator.
        
        Args:
            config: Volatility configuration containing model and ensemble settings
        """
        self.config = config
        self.symbol_models: Dict[str, Dict[str, BaseVolatilityModel]] = defaultdict(dict)
        self.weight_optimizer = DynamicWeightOptimizer(config.ensemble)
        self.confidence_aggregator = ConfidenceAggregator(config.ensemble)
        
        # Performance tracking
        self.forecast_history: Dict[str, List[EnsembleForecast]] = defaultdict(list)
        self.performance_metrics: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Real-time state
        self.last_update: Dict[str, datetime] = {}
        self.data_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Hardware acceleration status
        self.hardware_acceleration_enabled = config.hardware.auto_hardware_routing
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Ensemble orchestrator initialized")
    
    async def initialize_models(self, symbol: str) -> Dict[str, BaseVolatilityModel]:
        """
        Initialize all enabled models for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary of initialized models
        """
        try:
            models = {}
            
            # Get enabled model types from config
            enabled_models = self.config.get_enabled_models()
            
            for model_type in enabled_models:
                model_config = self.config.get_model_config(model_type)
                if model_config is None:
                    continue
                    
                # Create model instance
                model = await self._create_model_instance(symbol, model_type, model_config)
                if model is not None:
                    models[model_type.value] = model
                    self.logger.info(f"Initialized {model_type.value} model for {symbol}")
            
            self.symbol_models[symbol] = models
            self.logger.info(f"Initialized {len(models)} models for {symbol}")
            return models
            
        except Exception as e:
            self.logger.error(f"Error initializing models for {symbol}: {e}")
            raise
    
    async def _create_model_instance(self, 
                                   symbol: str, 
                                   model_type: ModelType,
                                   model_config) -> Optional[BaseVolatilityModel]:
        """Create a model instance based on type and configuration"""
        try:
            config_dict = {
                'use_gpu': self.hardware_acceleration_enabled and model_config.use_gpu,
                'lookback_days': model_config.lookback_days,
                'forecast_horizon': model_config.forecast_horizon
            }
            config_dict.update(model_config.parameters)
            
            # GARCH family models
            if model_type == ModelType.GARCH:
                p, q = model_config.parameters.get('p', 1), model_config.parameters.get('q', 1)
                return GARCHModel(symbol, p=p, q=q, config=config_dict)
            
            elif model_type == ModelType.EGARCH:
                p, q = model_config.parameters.get('p', 1), model_config.parameters.get('q', 1)
                return EGARCHModel(symbol, p=p, q=q, config=config_dict)
            
            elif model_type == ModelType.GJR_GARCH:
                p, q = model_config.parameters.get('p', 1), model_config.parameters.get('q', 1)
                return GJRGARCHModel(symbol, p=p, q=q, config=config_dict)
            
            # Real-time estimators
            elif model_type == ModelType.GARMAN_KLASS:
                window = model_config.parameters.get('window', 20)
                return GarmanKlassEstimator(symbol, window=window, config=config_dict)
            
            elif model_type == ModelType.YANG_ZHANG:
                window = model_config.parameters.get('window', 20)
                return YangZhangEstimator(symbol, window=window, config=config_dict)
            
            elif model_type == ModelType.ROGERS_SATCHELL:
                window = model_config.parameters.get('window', 20)
                return RogersSatchellEstimator(symbol, window=window, config=config_dict)
            
            # Stochastic volatility models
            elif model_type == ModelType.HESTON:
                return HestonModel(symbol, config=config_dict)
            
            elif model_type == ModelType.SABR:
                return SABRModel(symbol, config=config_dict)
            
            # Deep learning models (with availability check)
            elif model_type == ModelType.LSTM:
                if not DEEP_LEARNING_AVAILABLE or LSTMVolatilityPredictor is None:
                    self.logger.warning(f"LSTM model requested but PyTorch not available for {symbol}")
                    return None
                config_dict['deep_learning'] = model_config.parameters
                return LSTMVolatilityPredictor(symbol, config=config_dict)
            
            elif model_type == ModelType.TRANSFORMER:
                if not DEEP_LEARNING_AVAILABLE or TransformerVolatilityPredictor is None:
                    self.logger.warning(f"Transformer model requested but PyTorch not available for {symbol}")
                    return None
                config_dict['deep_learning'] = model_config.parameters
                return TransformerVolatilityPredictor(symbol, config=config_dict)
            
            else:
                self.logger.warning(f"Model type {model_type} not implemented yet")
                return None
                
        except Exception as e:
            self.logger.error(f"Error creating {model_type} model for {symbol}: {e}")
            return None
    
    async def train_models(self, 
                          symbol: str,
                          training_data: pd.DataFrame,
                          validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Train all models for a symbol.
        
        Args:
            symbol: Trading symbol
            training_data: Historical price data
            validation_split: Validation data fraction
            
        Returns:
            Training results for all models
        """
        try:
            if symbol not in self.symbol_models:
                await self.initialize_models(symbol)
            
            models = self.symbol_models[symbol]
            training_results = {}
            
            self.logger.info(f"Training {len(models)} models for {symbol}")
            
            # Train models concurrently for better performance
            training_tasks = []
            for model_name, model in models.items():
                task = self._train_single_model(model_name, model, training_data, validation_split)
                training_tasks.append(task)
            
            # Wait for all models to complete training
            results = await asyncio.gather(*training_tasks, return_exceptions=True)
            
            # Process results
            for i, (model_name, _) in enumerate(models.items()):
                result = results[i]
                if isinstance(result, Exception):
                    self.logger.error(f"Training failed for {model_name}: {result}")
                    training_results[model_name] = {'success': False, 'error': str(result)}
                else:
                    training_results[model_name] = result
            
            # Update performance metrics
            self.performance_metrics[symbol]['last_training'] = datetime.utcnow()
            self.performance_metrics[symbol]['models_trained'] = len([r for r in training_results.values() if r.get('success')])
            
            self.logger.info(f"Training completed for {symbol}: {len(training_results)} models")
            return training_results
            
        except Exception as e:
            self.logger.error(f"Error training models for {symbol}: {e}")
            raise
    
    async def _train_single_model(self, 
                                 model_name: str,
                                 model: BaseVolatilityModel,
                                 training_data: pd.DataFrame,
                                 validation_split: float) -> Dict[str, Any]:
        """Train a single model with error handling"""
        try:
            start_time = time.time()
            
            # Prepare data specific to model type
            prepared_data = await model.prepare_data(training_data)
            
            # Train model
            result = await model.train(prepared_data, validation_split)
            
            # Add timing information
            result['total_time_ms'] = (time.time() - start_time) * 1000
            result['model_name'] = model_name
            
            return result
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'model_name': model_name}
    
    async def generate_ensemble_forecast(self, 
                                       symbol: str,
                                       recent_data: Optional[pd.DataFrame] = None,
                                       horizon: int = 5,
                                       confidence_level: float = 0.95) -> EnsembleForecast:
        """
        Generate ensemble volatility forecast combining all models.
        
        Args:
            symbol: Trading symbol
            recent_data: Recent market data for forecasting
            horizon: Forecast horizon in days
            confidence_level: Confidence level for intervals
            
        Returns:
            Ensemble forecast with model contributions and uncertainty
        """
        try:
            start_time = time.time()
            
            if symbol not in self.symbol_models:
                raise ValueError(f"No models initialized for symbol {symbol}")
            
            models = self.symbol_models[symbol]
            if not models:
                raise ValueError(f"No trained models available for {symbol}")
            
            self.logger.info(f"Generating ensemble forecast for {symbol} with {len(models)} models")
            
            # Generate individual model forecasts concurrently
            forecast_tasks = []
            for model_name, model in models.items():
                if model.status == ModelStatus.READY:
                    task = self._generate_single_forecast(model_name, model, recent_data, horizon, confidence_level)
                    forecast_tasks.append(task)
            
            # Wait for all forecasts
            forecast_results = await asyncio.gather(*forecast_tasks, return_exceptions=True)
            
            # Process individual forecasts
            individual_forecasts = {}
            valid_forecasts = []
            
            for i, result in enumerate(forecast_results):
                model_name = list(models.keys())[i]
                if isinstance(result, Exception):
                    self.logger.warning(f"Forecast failed for {model_name}: {result}")
                    continue
                    
                if result is not None:
                    individual_forecasts[model_name] = result
                    valid_forecasts.append(result.forecast_volatility)
            
            if not valid_forecasts:
                raise ValueError(f"No valid forecasts generated for {symbol}")
            
            # Calculate ensemble weights
            weights = await self.weight_optimizer.calculate_weights(
                symbol, individual_forecasts, self.performance_metrics.get(symbol, {})
            )
            
            # Combine forecasts
            ensemble_vol = self._combine_forecasts(individual_forecasts, weights)
            
            # Calculate uncertainty and confidence
            uncertainty_metrics = self.confidence_aggregator.quantify_uncertainty(
                individual_forecasts, weights, confidence_level
            )
            
            # Create ensemble forecast
            ensemble_forecast = EnsembleForecast(
                symbol=symbol,
                forecast_timestamp=datetime.utcnow(),
                ensemble_volatility=ensemble_vol,
                ensemble_variance=ensemble_vol ** 2,
                ensemble_confidence=uncertainty_metrics['ensemble_confidence'],
                model_forecasts=individual_forecasts,
                model_weights=weights,
                model_contributions={name: weights[name] * forecast.forecast_volatility 
                                   for name, forecast in individual_forecasts.items()},
                forecast_uncertainty=uncertainty_metrics['forecast_uncertainty'],
                model_disagreement=uncertainty_metrics['model_disagreement'],
                confidence_interval_lower=uncertainty_metrics['ci_lower'],
                confidence_interval_upper=uncertainty_metrics['ci_upper'],
                ensemble_method=self.config.ensemble.method.value,
                computation_time_ms=(time.time() - start_time) * 1000,
                models_used=len(individual_forecasts),
                hardware_acceleration=self.hardware_acceleration_enabled
            )
            
            # Store forecast
            self.forecast_history[symbol].append(ensemble_forecast)
            self.last_update[symbol] = datetime.utcnow()
            
            self.logger.info(f"Generated ensemble forecast for {symbol}: {ensemble_vol:.4f} volatility")
            return ensemble_forecast
            
        except Exception as e:
            self.logger.error(f"Error generating ensemble forecast for {symbol}: {e}")
            raise
    
    async def _generate_single_forecast(self, 
                                      model_name: str,
                                      model: BaseVolatilityModel,
                                      data: Optional[pd.DataFrame],
                                      horizon: int,
                                      confidence_level: float) -> Optional[VolatilityForecast]:
        """Generate forecast from a single model with error handling"""
        try:
            return await model.forecast(data, horizon, confidence_level)
        except Exception as e:
            self.logger.warning(f"Forecast failed for {model_name}: {e}")
            return None
    
    def _combine_forecasts(self, 
                          forecasts: Dict[str, VolatilityForecast],
                          weights: Dict[str, float]) -> float:
        """
        Combine individual model forecasts using weights.
        
        Args:
            forecasts: Individual model forecasts
            weights: Model weights
            
        Returns:
            Combined ensemble volatility
        """
        weighted_sum = 0.0
        total_weight = 0.0
        
        for model_name, forecast in forecasts.items():
            weight = weights.get(model_name, 0.0)
            weighted_sum += weight * forecast.forecast_volatility
            total_weight += weight
        
        # Normalize if weights don't sum to 1
        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            # Fallback to equal weighting
            return np.mean([f.forecast_volatility for f in forecasts.values()])
    
    async def update_real_time(self, 
                             symbol: str,
                             new_data: Dict[str, Any]) -> Optional[EnsembleForecast]:
        """
        Update models with real-time data and generate new forecast.
        
        Args:
            symbol: Trading symbol
            new_data: New market data (OHLCV)
            
        Returns:
            Updated ensemble forecast if triggered
        """
        try:
            # Add to data buffer
            self.data_buffers[symbol].append({
                **new_data,
                'timestamp': datetime.utcnow()
            })
            
            # Check if update is needed
            if not self._should_update(symbol):
                return None
            
            # Convert buffer to DataFrame
            recent_data = pd.DataFrame(list(self.data_buffers[symbol]))
            
            # Generate new forecast
            forecast = await self.generate_ensemble_forecast(symbol, recent_data)
            
            self.logger.info(f"Real-time forecast updated for {symbol}")
            return forecast
            
        except Exception as e:
            self.logger.error(f"Error in real-time update for {symbol}: {e}")
            return None
    
    def _should_update(self, symbol: str) -> bool:
        """Determine if ensemble should be updated based on timing and data"""
        if symbol not in self.last_update:
            return True
            
        # Check time since last update
        time_since_update = datetime.utcnow() - self.last_update[symbol]
        update_interval = self.config.ensemble.rebalance_frequency
        
        return time_since_update >= update_interval
    
    def get_model_status(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive status of all models for a symbol"""
        if symbol not in self.symbol_models:
            return {'error': f'No models initialized for {symbol}'}
        
        models = self.symbol_models[symbol]
        status = {
            'symbol': symbol,
            'total_models': len(models),
            'models': {},
            'last_forecast': self.last_update.get(symbol),
            'forecast_count': len(self.forecast_history.get(symbol, [])),
            'hardware_acceleration': self.hardware_acceleration_enabled
        }
        
        for model_name, model in models.items():
            status['models'][model_name] = {
                'status': model.status.value,
                'last_training': model.last_training_date,
                'last_forecast': model.last_forecast_date,
                'metrics_available': len(model.metrics_history) > 0,
                'use_gpu': getattr(model, 'use_gpu', False)
            }
        
        return status
    
    def get_latest_ensemble_forecast(self, symbol: str) -> Optional[EnsembleForecast]:
        """Get the most recent ensemble forecast for a symbol"""
        forecasts = self.forecast_history.get(symbol, [])
        return forecasts[-1] if forecasts else None
    
    async def cleanup(self, symbol: Optional[str] = None):
        """Clean up models and resources"""
        if symbol:
            # Clean up specific symbol
            if symbol in self.symbol_models:
                for model in self.symbol_models[symbol].values():
                    await model.cleanup()
                del self.symbol_models[symbol]
            
            # Clear related data
            self.forecast_history.pop(symbol, None)
            self.performance_metrics.pop(symbol, None)
            self.last_update.pop(symbol, None)
            self.data_buffers.pop(symbol, None)
        else:
            # Clean up everything
            for symbol_models in self.symbol_models.values():
                for model in symbol_models.values():
                    await model.cleanup()
            
            self.symbol_models.clear()
            self.forecast_history.clear()
            self.performance_metrics.clear()
            self.last_update.clear()
            self.data_buffers.clear()
        
        self.logger.info(f"Cleaned up models for {symbol or 'all symbols'}")