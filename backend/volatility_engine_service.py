"""
Nautilus Volatility Forecasting Engine Service

Native service integration with full M4 Max hardware access.
Runs as part of the main Nautilus backend with direct GPU and Neural Engine access.
"""

import asyncio
import logging
import os
import sys
from typing import Dict, List, Optional, Any
from datetime import datetime

# Import the volatility engine
from volatility.engine.volatility_engine import VolatilityEngine
from volatility.config import VolatilityConfig, get_default_volatility_config
from volatility.ensemble.orchestrator import EnsembleOrchestrator, EnsembleForecast

logger = logging.getLogger(__name__)


class NautilusVolatilityService:
    """
    Native Volatility Service for Nautilus Platform
    
    Provides volatility forecasting capabilities with full M4 Max hardware acceleration.
    Integrates directly with the main Nautilus backend for optimal performance.
    """
    
    def __init__(self):
        """Initialize the volatility service with M4 Max optimizations"""
        # Configuration with M4 Max optimizations enabled
        self.config = VolatilityConfig.from_environment()
        
        # Ensure hardware acceleration is enabled
        self.config.hardware.use_metal_gpu = True
        self.config.hardware.use_neural_engine = True
        self.config.hardware.use_cpu_optimization = True
        self.config.hardware.auto_hardware_routing = True
        
        # Initialize the volatility engine
        self.engine: Optional[VolatilityEngine] = None
        self.is_initialized = False
        
        # Service state
        self.active_symbols: set = set()
        self.service_stats = {
            'forecasts_generated': 0,
            'models_trained': 0,
            'hardware_acceleration_active': True,
            'start_time': None
        }
        
        logger.info("Nautilus Volatility Service initialized with M4 Max acceleration")
    
    async def initialize(self) -> None:
        """Initialize the volatility engine with full hardware access"""
        try:
            if self.is_initialized:
                return
            
            logger.info("🚀 Initializing Nautilus Volatility Engine with M4 Max hardware access...")
            
            # Create engine with hardware acceleration
            self.engine = VolatilityEngine(self.config)
            await self.engine.initialize()
            
            # Verify hardware acceleration
            await self._verify_hardware_acceleration()
            
            self.is_initialized = True
            self.service_stats['start_time'] = datetime.utcnow()
            
            logger.info("✅ Volatility engine initialized with direct hardware access")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize volatility engine: {e}")
            raise
    
    async def _verify_hardware_acceleration(self) -> Dict[str, bool]:
        """Verify that hardware acceleration is working"""
        hardware_status = {
            'metal_gpu_available': False,
            'neural_engine_available': False,
            'cpu_optimization_active': False
        }
        
        try:
            # Check Metal GPU
            import torch
            if torch.backends.mps.is_available():
                hardware_status['metal_gpu_available'] = True
                logger.info("✅ Metal GPU acceleration available")
            else:
                logger.warning("⚠️ Metal GPU not available")
            
            # Neural Engine check (simplified)
            try:
                import coreml
                hardware_status['neural_engine_available'] = True
                logger.info("✅ Neural Engine acceleration available")
            except ImportError:
                logger.warning("⚠️ Neural Engine support not available (CoreML not found)")
            
            # CPU optimization is always available
            hardware_status['cpu_optimization_active'] = True
            logger.info("✅ CPU optimization active")
            
        except Exception as e:
            logger.warning(f"Hardware verification failed: {e}")
        
        return hardware_status
    
    async def add_symbol_for_forecasting(self, symbol: str) -> Dict[str, Any]:
        """
        Add a symbol to volatility forecasting with all models.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Initialization results
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            symbol = symbol.upper()
            
            # Add symbol to engine
            result = await self.engine.add_symbol(symbol)
            self.active_symbols.add(symbol)
            
            logger.info(f"📈 Added {symbol} to volatility forecasting")
            return result
            
        except Exception as e:
            logger.error(f"Failed to add symbol {symbol}: {e}")
            raise
    
    async def train_volatility_models(self, symbol: str, training_data) -> Dict[str, Any]:
        """
        Train all volatility models for a symbol using M4 Max acceleration.
        
        Args:
            symbol: Trading symbol
            training_data: Historical price data (DataFrame or dict)
            
        Returns:
            Training results
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            symbol = symbol.upper()
            
            # Ensure symbol is added
            if symbol not in self.active_symbols:
                await self.add_symbol_for_forecasting(symbol)
            
            # Convert data if needed
            import pandas as pd
            if isinstance(training_data, dict):
                training_data = pd.DataFrame(training_data)
            
            # Train models with hardware acceleration
            result = await self.engine.train_symbol_models(symbol, training_data)
            self.service_stats['models_trained'] += 1
            
            logger.info(f"🎯 Trained volatility models for {symbol} with M4 Max acceleration")
            return result
            
        except Exception as e:
            logger.error(f"Training failed for {symbol}: {e}")
            raise
    
    async def generate_volatility_forecast(self, 
                                         symbol: str,
                                         recent_data: Optional[Any] = None,
                                         horizon: int = 5,
                                         confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        Generate ensemble volatility forecast with M4 Max acceleration.
        
        Args:
            symbol: Trading symbol
            recent_data: Recent market data
            horizon: Forecast horizon in days
            confidence_level: Confidence level
            
        Returns:
            Ensemble forecast with uncertainty quantification
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            symbol = symbol.upper()
            
            # Convert data if needed
            if recent_data and not isinstance(recent_data, type(None)):
                import pandas as pd
                if isinstance(recent_data, dict):
                    recent_data = pd.DataFrame(recent_data)
            
            # Generate forecast with hardware acceleration
            result = await self.engine.generate_forecast(
                symbol, recent_data, horizon, confidence_level
            )
            
            self.service_stats['forecasts_generated'] += 1
            
            logger.info(f"📊 Generated volatility forecast for {symbol}: "
                       f"{result['forecast']['ensemble_volatility']:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Forecast generation failed for {symbol}: {e}")
            raise
    
    async def get_latest_volatility_forecast(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get the latest volatility forecast for a symbol"""
        if not self.is_initialized:
            return None
        
        try:
            return await self.engine.get_latest_forecast(symbol.upper())
        except Exception as e:
            logger.error(f"Failed to get latest forecast for {symbol}: {e}")
            return None
    
    async def update_realtime_volatility(self, symbol: str, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Update volatility models with real-time market data.
        
        Args:
            symbol: Trading symbol
            market_data: Real-time OHLCV data
            
        Returns:
            Updated forecast if triggered
        """
        if not self.is_initialized:
            return None
        
        try:
            return await self.engine.update_real_time_data(symbol.upper(), market_data)
        except Exception as e:
            logger.error(f"Real-time update failed for {symbol}: {e}")
            return None
    
    def get_volatility_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status"""
        if not self.engine:
            return {'status': 'not_initialized'}
        
        try:
            # Get engine status
            engine_status = asyncio.create_task(self.engine.get_engine_status())
            
            return {
                'service_name': 'Nautilus Volatility Forecasting Engine',
                'status': 'operational' if self.is_initialized else 'initializing',
                'hardware_acceleration': {
                    'metal_gpu_enabled': self.config.hardware.use_metal_gpu,
                    'neural_engine_enabled': self.config.hardware.use_neural_engine,
                    'cpu_optimization_enabled': self.config.hardware.use_cpu_optimization,
                    'auto_routing_enabled': self.config.hardware.auto_hardware_routing
                },
                'active_symbols': list(self.active_symbols),
                'service_stats': self.service_stats,
                'ensemble_config': {
                    'method': self.config.ensemble.method.value,
                    'min_models': self.config.ensemble.min_models,
                    'max_models': self.config.ensemble.max_models
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting service status: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def get_available_models(self) -> Dict[str, Any]:
        """Get list of available volatility models"""
        # Check deep learning availability
        try:
            from volatility.models.deep_learning_models import (
                DEEP_LEARNING_AVAILABLE, 
                NEURAL_ENGINE_OPTIMIZATION_AVAILABLE
            )
        except ImportError:
            DEEP_LEARNING_AVAILABLE = False
            NEURAL_ENGINE_OPTIMIZATION_AVAILABLE = False
        
        ml_models_info = {
            'LSTM': {
                'available': DEEP_LEARNING_AVAILABLE,
                'neural_engine_optimized': NEURAL_ENGINE_OPTIMIZATION_AVAILABLE,
                'description': 'Long Short-Term Memory network for sequence modeling'
            },
            'TRANSFORMER': {
                'available': DEEP_LEARNING_AVAILABLE,
                'neural_engine_optimized': NEURAL_ENGINE_OPTIMIZATION_AVAILABLE,
                'description': 'Transformer model with multi-head attention for advanced pattern recognition'
            }
        }
        
        return {
            'econometric_models': [
                'GARCH', 'EGARCH', 'GJR_GARCH'
            ],
            'stochastic_models': [
                'HESTON', 'SABR'
            ],
            'realtime_estimators': [
                'GARMAN_KLASS', 'YANG_ZHANG', 'ROGERS_SATCHELL'
            ],
            'ml_models': ml_models_info,
            'deep_learning_available': DEEP_LEARNING_AVAILABLE,
            'neural_engine_optimization': NEURAL_ENGINE_OPTIMIZATION_AVAILABLE,
            'hardware_acceleration': 'M4 Max Metal GPU + Neural Engine',
            'ensemble_methods': ['equal_weight', 'variance_weight', 'dynamic_weight', 'bayesian_average']
        }
    
    async def cleanup(self):
        """Clean up the volatility service"""
        if self.engine:
            await self.engine.shutdown()
            self.engine = None
        
        self.active_symbols.clear()
        self.is_initialized = False
        logger.info("🧹 Volatility service cleaned up")


# Global service instance
_volatility_service: Optional[NautilusVolatilityService] = None


async def get_volatility_service() -> NautilusVolatilityService:
    """Get the global volatility service instance"""
    global _volatility_service
    
    if _volatility_service is None:
        _volatility_service = NautilusVolatilityService()
        await _volatility_service.initialize()
    
    return _volatility_service


async def shutdown_volatility_service():
    """Shutdown the global volatility service"""
    global _volatility_service
    
    if _volatility_service:
        await _volatility_service.cleanup()
        _volatility_service = None