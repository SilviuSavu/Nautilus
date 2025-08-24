"""
Integration Tests for Nautilus Volatility Forecasting Engine

Tests the complete volatility engine with M4 Max hardware acceleration.
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

# Import volatility engine components
from volatility.engine.volatility_engine import VolatilityEngine
from volatility.config import VolatilityConfig, ModelType, get_default_volatility_config
from volatility.models.garch_models import GARCHModel
from volatility.models.estimators import GarmanKlassEstimator
from volatility.ensemble.orchestrator import EnsembleOrchestrator

# Configure test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def sample_price_data():
    """Generate sample OHLCV data for testing"""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    
    # Generate realistic price data with volatility clustering
    returns = np.random.normal(0.0005, 0.02, len(dates))
    returns = np.where(np.abs(returns) > 0.03, np.sign(returns) * 0.03, returns)  # Cap extreme moves
    
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Generate OHLC from close prices
    opens = prices * (1 + np.random.normal(0, 0.001, len(prices)))
    highs = np.maximum(opens, prices) * (1 + np.abs(np.random.normal(0, 0.005, len(prices))))
    lows = np.minimum(opens, prices) * (1 - np.abs(np.random.normal(0, 0.005, len(prices))))
    volumes = np.random.lognormal(10, 1, len(prices))
    
    return pd.DataFrame({
        'date': dates,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': prices,
        'volume': volumes
    })


@pytest.fixture
def test_config():
    """Test configuration with limited models for faster testing"""
    config = get_default_volatility_config()
    
    # Enable only essential models for testing
    config.models.clear()
    
    # Add GARCH model
    config.add_model_config(ModelType.GARCH, config.get_model_config(ModelType.GARCH))
    
    # Add Garman-Klass estimator
    config.add_model_config(ModelType.GARMAN_KLASS, config.get_model_config(ModelType.GARMAN_KLASS))
    
    # Configure ensemble for testing
    config.ensemble.min_models = 2
    config.ensemble.max_models = 2
    
    return config


class TestVolatilityEngine:
    """Test the complete volatility engine functionality"""
    
    @pytest.mark.asyncio
    async def test_engine_initialization(self, test_config):
        """Test engine initialization"""
        engine = VolatilityEngine(test_config)
        
        assert engine.config == test_config
        assert not engine.is_running
        
        await engine.initialize()
        assert engine.is_running
        
        await engine.shutdown()
    
    @pytest.mark.asyncio
    async def test_symbol_management(self, test_config, sample_price_data):
        """Test adding and managing symbols"""
        engine = VolatilityEngine(test_config)
        await engine.initialize()
        
        try:
            # Add symbol
            result = await engine.add_symbol("AAPL")
            assert result['status'] == 'success'
            assert 'AAPL' in engine.active_symbols
            
            # Train models
            train_result = await engine.train_symbol_models("AAPL", sample_price_data)
            assert train_result['status'] == 'success'
            
        finally:
            await engine.shutdown()
    
    @pytest.mark.asyncio
    async def test_ensemble_forecasting(self, test_config, sample_price_data):
        """Test ensemble volatility forecasting"""
        engine = VolatilityEngine(test_config)
        await engine.initialize()
        
        try:
            # Setup symbol and train models
            await engine.add_symbol("AAPL")
            await engine.train_symbol_models("AAPL", sample_price_data)
            
            # Generate forecast
            result = await engine.generate_forecast("AAPL", horizon=5)
            
            assert result['status'] == 'success'
            assert 'forecast' in result
            
            forecast = result['forecast']
            assert 'ensemble_volatility' in forecast
            assert 'ensemble_confidence' in forecast
            assert 'model_weights' in forecast
            assert forecast['models_used'] >= 1
            
            # Volatility should be positive
            assert forecast['ensemble_volatility'] > 0
            
            # Confidence should be between 0 and 1
            assert 0 <= forecast['ensemble_confidence'] <= 1
            
        finally:
            await engine.shutdown()
    
    @pytest.mark.asyncio
    async def test_realtime_updates(self, test_config, sample_price_data):
        """Test real-time data updates"""
        engine = VolatilityEngine(test_config)
        await engine.initialize()
        
        try:
            # Setup symbol
            await engine.add_symbol("AAPL")
            await engine.train_symbol_models("AAPL", sample_price_data)
            
            # Generate initial forecast
            initial_forecast = await engine.generate_forecast("AAPL")
            
            # Update with new data
            new_data = {
                'open': 105.0,
                'high': 107.0,
                'low': 104.0,
                'close': 106.0,
                'volume': 1000000
            }
            
            update_result = await engine.update_real_time_data("AAPL", new_data)
            # Update may or may not trigger new forecast depending on timing
            
        finally:
            await engine.shutdown()


class TestModelComponents:
    """Test individual model components"""
    
    @pytest.mark.asyncio
    async def test_garch_model(self, sample_price_data):
        """Test GARCH model functionality"""
        model = GARCHModel("AAPL", p=1, q=1, config={'use_gpu': False})  # Disable GPU for testing
        
        # Prepare data
        prepared_data = await model.prepare_data(sample_price_data)
        assert len(prepared_data) > 0
        assert 'returns' in prepared_data.columns
        
        # Train model
        train_result = await model.train(prepared_data)
        assert train_result['success'] == True
        
        # Generate forecast
        forecast = await model.forecast(horizon=5)
        assert forecast.forecast_volatility > 0
        assert forecast.confidence_interval_lower >= 0
        assert forecast.confidence_interval_upper > forecast.confidence_interval_lower
    
    @pytest.mark.asyncio 
    async def test_garman_klass_estimator(self, sample_price_data):
        """Test Garman-Klass estimator"""
        estimator = GarmanKlassEstimator("AAPL", window=20, config={'use_gpu': False})
        
        # Prepare data
        prepared_data = await estimator.prepare_data(sample_price_data)
        assert len(prepared_data) > 0
        assert 'gk_volatility' in prepared_data.columns
        
        # Train (initialize) estimator
        train_result = await estimator.train(prepared_data)
        assert train_result['success'] == True
        
        # Generate estimate
        forecast = await estimator.forecast()
        assert forecast.forecast_volatility > 0


class TestEnsembleOrchestrator:
    """Test ensemble orchestration"""
    
    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self, test_config):
        """Test orchestrator initialization"""
        orchestrator = EnsembleOrchestrator(test_config)
        
        # Initialize models for a symbol
        models = await orchestrator.initialize_models("AAPL")
        assert len(models) >= 1
        
        # Cleanup
        await orchestrator.cleanup("AAPL")
    
    @pytest.mark.asyncio
    async def test_model_training(self, test_config, sample_price_data):
        """Test model training orchestration"""
        orchestrator = EnsembleOrchestrator(test_config)
        
        # Initialize and train models
        await orchestrator.initialize_models("AAPL")
        train_results = await orchestrator.train_models("AAPL", sample_price_data)
        
        assert len(train_results) >= 1
        for model_name, result in train_results.items():
            if result.get('success'):
                assert 'training_time_ms' in result
        
        # Cleanup
        await orchestrator.cleanup("AAPL")


class TestHardwareAcceleration:
    """Test hardware acceleration capabilities"""
    
    def test_gpu_detection(self):
        """Test Metal GPU detection"""
        try:
            import torch
            gpu_available = torch.backends.mps.is_available()
            logger.info(f"Metal GPU available: {gpu_available}")
        except ImportError:
            logger.warning("PyTorch not available for GPU testing")
    
    def test_neural_engine_detection(self):
        """Test Neural Engine detection"""
        try:
            import coreml
            logger.info("CoreML available for Neural Engine")
        except ImportError:
            logger.warning("CoreML not available for Neural Engine testing")


@pytest.mark.asyncio
async def test_full_integration(test_config, sample_price_data):
    """Full integration test with multiple symbols"""
    engine = VolatilityEngine(test_config)
    await engine.initialize()
    
    try:
        symbols = ["AAPL", "GOOGL"]
        
        # Add symbols and train models
        for symbol in symbols:
            await engine.add_symbol(symbol)
            await engine.train_symbol_models(symbol, sample_price_data)
        
        # Generate forecasts for all symbols
        forecasts = {}
        for symbol in symbols:
            result = await engine.generate_forecast(symbol)
            forecasts[symbol] = result
        
        # Verify all forecasts
        for symbol, result in forecasts.items():
            assert result['status'] == 'success'
            assert result['forecast']['ensemble_volatility'] > 0
        
        # Test engine status
        status = await engine.get_engine_status()
        assert status['status'] == 'running'
        assert len(status['active_symbols']) == len(symbols)
        
        logger.info("✅ Full integration test passed")
        
    finally:
        await engine.shutdown()


@pytest.mark.asyncio
async def test_deep_learning_integration():
    """Test deep learning models integration with the main engine"""
    try:
        from volatility.models.deep_learning_models import (
            DEEP_LEARNING_AVAILABLE,
            NEURAL_ENGINE_OPTIMIZATION_AVAILABLE
        )
        
        if not DEEP_LEARNING_AVAILABLE:
            logger.info("ℹ️  Skipping deep learning integration test - PyTorch not available")
            return
        
        # Create config with deep learning models
        config = get_default_volatility_config()
        engine = VolatilityEngine(config)
        await engine.initialize()
        
        try:
            # Test with a symbol that includes deep learning models
            symbol = "TSLA"
            await engine.add_symbol(symbol)
            
            # Generate sample data for training
            import numpy as np
            import pandas as pd
            
            np.random.seed(123)
            dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
            prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 200)))
            
            sample_data = pd.DataFrame({
                'date': dates,
                'open': prices * (1 + np.random.normal(0, 0.001, 200)),
                'high': prices * (1 + np.abs(np.random.normal(0, 0.005, 200))),
                'low': prices * (1 - np.abs(np.random.normal(0, 0.005, 200))),
                'close': prices,
                'volume': np.random.lognormal(10, 1, 200)
            })
            
            # Train models (including deep learning if available)
            training_result = await engine.train_symbol_models(symbol, sample_data)
            assert training_result['status'] == 'success'
            
            # Generate forecast
            forecast_result = await engine.generate_forecast(symbol, horizon=5)
            assert forecast_result['status'] == 'success'
            
            forecast = forecast_result['forecast']
            assert forecast['ensemble_volatility'] > 0
            assert forecast['models_used'] >= 1
            
            logger.info("✅ Deep learning integration test passed")
            logger.info(f"   Models used: {forecast['models_used']}")
            logger.info(f"   Ensemble volatility: {forecast['ensemble_volatility']:.4f}")
            logger.info(f"   Neural Engine optimization: {NEURAL_ENGINE_OPTIMIZATION_AVAILABLE}")
            
        finally:
            await engine.shutdown()
            
    except ImportError as e:
        logger.info(f"ℹ️  Deep learning integration test skipped: {e}")


if __name__ == "__main__":
    """Run tests directly"""
    pytest.main([__file__, "-v"])