"""
Integration Tests for Deep Learning Volatility Models

Tests LSTM and Transformer models with M4 Max Neural Engine acceleration.
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

# Import volatility components
try:
    from volatility.models.deep_learning_models import (
        LSTMVolatilityPredictor,
        TransformerVolatilityPredictor,
        DEEP_LEARNING_AVAILABLE,
        NEURAL_ENGINE_OPTIMIZATION_AVAILABLE,
        DeepLearningConfig
    )
    from volatility.config import VolatilityConfig, ModelType, get_default_volatility_config
    from volatility.ensemble.orchestrator import EnsembleOrchestrator
    DEEP_LEARNING_TESTS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Deep learning imports failed: {e}")
    DEEP_LEARNING_TESTS_AVAILABLE = False

# Configure test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def sample_price_data():
    """Generate sample OHLCV data for testing deep learning models"""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    
    # Generate realistic price data with volatility clustering
    returns = np.random.normal(0.0005, 0.02, len(dates))
    returns = np.where(np.abs(returns) > 0.03, np.sign(returns) * 0.03, returns)
    
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
def deep_learning_config():
    """Test configuration with reduced complexity for faster testing"""
    return DeepLearningConfig(
        sequence_length=20,  # Shorter sequences for testing
        hidden_size=64,      # Smaller model
        num_layers=1,        # Single layer
        dropout=0.1,
        learning_rate=0.01,  # Higher learning rate
        batch_size=16,       # Smaller batches
        epochs=5,           # Few epochs for testing
        early_stopping_patience=3,
        use_metal_gpu=True,
        use_neural_engine=True
    )


@pytest.mark.skipif(not DEEP_LEARNING_TESTS_AVAILABLE, reason="Deep learning dependencies not available")
class TestDeepLearningModels:
    """Test deep learning volatility models"""
    
    @pytest.mark.asyncio
    async def test_lstm_model_availability(self):
        """Test LSTM model availability and initialization"""
        if not DEEP_LEARNING_AVAILABLE:
            pytest.skip("PyTorch not available")
        
        # Test model creation
        model = LSTMVolatilityPredictor("AAPL", {})
        assert model.symbol == "AAPL"
        assert model.device is not None
        
        logger.info(f"✅ LSTM model initialized on device: {model.device}")
    
    @pytest.mark.asyncio
    async def test_transformer_model_availability(self):
        """Test Transformer model availability and initialization"""
        if not DEEP_LEARNING_AVAILABLE:
            pytest.skip("PyTorch not available")
        
        # Test model creation
        model = TransformerVolatilityPredictor("GOOGL", {})
        assert model.symbol == "GOOGL"
        assert model.device is not None
        
        logger.info(f"✅ Transformer model initialized on device: {model.device}")
    
    @pytest.mark.asyncio
    async def test_lstm_data_preparation(self, sample_price_data):
        """Test LSTM data preparation"""
        if not DEEP_LEARNING_AVAILABLE:
            pytest.skip("PyTorch not available")
        
        model = LSTMVolatilityPredictor("AAPL", {})
        prepared_data = await model.prepare_data(sample_price_data)
        
        assert len(prepared_data) > 0
        assert 'realized_volatility' in prepared_data.columns
        assert 'vol_5d' in prepared_data.columns
        assert 'vol_20d' in prepared_data.columns
        assert 'vol_60d' in prepared_data.columns
        
        logger.info(f"✅ LSTM data preparation: {len(prepared_data)} samples prepared")
    
    @pytest.mark.asyncio
    async def test_lstm_training(self, sample_price_data, deep_learning_config):
        """Test LSTM model training"""
        if not DEEP_LEARNING_AVAILABLE:
            pytest.skip("PyTorch not available")
        
        model = LSTMVolatilityPredictor("AAPL", {'deep_learning': deep_learning_config.__dict__})
        
        # Prepare data
        prepared_data = await model.prepare_data(sample_price_data)
        
        # Train model
        result = await model.train(prepared_data)
        
        assert result['success'] == True
        assert 'training_time_ms' in result
        assert 'final_train_loss' in result
        assert result['device'] is not None
        
        logger.info(f"✅ LSTM training completed in {result['training_time_ms']:.2f}ms")
        logger.info(f"   Final train loss: {result.get('final_train_loss', 'N/A')}")
        logger.info(f"   Device: {result['device']}")
        logger.info(f"   Neural Engine optimized: {result.get('neural_engine_optimized', False)}")
    
    @pytest.mark.asyncio
    async def test_lstm_forecasting(self, sample_price_data, deep_learning_config):
        """Test LSTM forecasting"""
        if not DEEP_LEARNING_AVAILABLE:
            pytest.skip("PyTorch not available")
        
        model = LSTMVolatilityPredictor("AAPL", {'deep_learning': deep_learning_config.__dict__})
        
        # Prepare and train
        prepared_data = await model.prepare_data(sample_price_data)
        train_result = await model.train(prepared_data)
        
        assert train_result['success'] == True
        
        # Generate forecast
        forecast = await model.forecast(horizon=5, confidence_level=0.95)
        
        assert forecast.forecast_volatility > 0
        assert forecast.confidence_interval_lower >= 0
        assert forecast.confidence_interval_upper > forecast.confidence_interval_lower
        assert forecast.forecast_horizon == 5
        assert 0 <= forecast.model_confidence <= 1
        
        logger.info(f"✅ LSTM forecast generated:")
        logger.info(f"   Volatility: {forecast.forecast_volatility:.4f}")
        logger.info(f"   Confidence interval: [{forecast.confidence_interval_lower:.4f}, {forecast.confidence_interval_upper:.4f}]")
        logger.info(f"   Model confidence: {forecast.model_confidence:.2f}")
    
    @pytest.mark.asyncio
    async def test_transformer_training(self, sample_price_data, deep_learning_config):
        """Test Transformer model training"""
        if not DEEP_LEARNING_AVAILABLE:
            pytest.skip("PyTorch not available")
        
        model = TransformerVolatilityPredictor("GOOGL", {'deep_learning': deep_learning_config.__dict__})
        
        # Prepare data
        prepared_data = await model.prepare_data(sample_price_data)
        
        # Train model
        result = await model.train(prepared_data)
        
        assert result['success'] == True
        assert 'training_time_ms' in result
        assert result['model_type'] == 'Transformer'
        
        logger.info(f"✅ Transformer training completed in {result['training_time_ms']:.2f}ms")
        logger.info(f"   Device: {result['device']}")
    
    @pytest.mark.asyncio
    async def test_transformer_forecasting(self, sample_price_data, deep_learning_config):
        """Test Transformer forecasting"""
        if not DEEP_LEARNING_AVAILABLE:
            pytest.skip("PyTorch not available")
        
        model = TransformerVolatilityPredictor("GOOGL", {'deep_learning': deep_learning_config.__dict__})
        
        # Prepare and train
        prepared_data = await model.prepare_data(sample_price_data)
        train_result = await model.train(prepared_data)
        
        assert train_result['success'] == True
        
        # Generate forecast
        forecast = await model.forecast(horizon=3, confidence_level=0.90)
        
        assert forecast.forecast_volatility > 0
        assert forecast.confidence_interval_lower >= 0
        assert forecast.confidence_interval_upper > forecast.confidence_interval_lower
        assert forecast.forecast_horizon == 3
        
        logger.info(f"✅ Transformer forecast generated:")
        logger.info(f"   Volatility: {forecast.forecast_volatility:.4f}")
        logger.info(f"   Confidence interval: [{forecast.confidence_interval_lower:.4f}, {forecast.confidence_interval_upper:.4f}]")


@pytest.mark.skipif(not DEEP_LEARNING_TESTS_AVAILABLE, reason="Deep learning dependencies not available")
class TestEnsembleWithDeepLearning:
    """Test ensemble orchestrator with deep learning models"""
    
    @pytest.mark.asyncio
    async def test_ensemble_with_lstm(self, sample_price_data):
        """Test ensemble orchestrator including LSTM model"""
        if not DEEP_LEARNING_AVAILABLE:
            pytest.skip("PyTorch not available")
        
        # Create configuration with LSTM
        config = get_default_volatility_config()
        orchestrator = EnsembleOrchestrator(config)
        
        # Initialize models for symbol (should include LSTM)
        models = await orchestrator.initialize_models("AAPL")
        
        # Check if LSTM model was created (depends on availability)
        lstm_models = [m for m in models.values() if 'LSTM' in str(type(m))]
        
        if DEEP_LEARNING_AVAILABLE:
            assert len(lstm_models) > 0, "LSTM model should be included when PyTorch is available"
            logger.info(f"✅ LSTM model included in ensemble: {len(lstm_models)} LSTM models")
        else:
            logger.info("ℹ️  LSTM model skipped due to PyTorch unavailability")
        
        logger.info(f"   Total models initialized: {len(models)}")
    
    @pytest.mark.asyncio
    async def test_ensemble_deep_learning_training(self, sample_price_data):
        """Test training ensemble with deep learning models"""
        if not DEEP_LEARNING_AVAILABLE:
            pytest.skip("PyTorch not available")
        
        config = get_default_volatility_config()
        # Reduce epochs for faster testing
        if ModelType.LSTM in config.models:
            config.models[ModelType.LSTM].parameters['epochs'] = 3
        if ModelType.TRANSFORMER in config.models:
            config.models[ModelType.TRANSFORMER].parameters['epochs'] = 3
        
        orchestrator = EnsembleOrchestrator(config)
        
        # Initialize and train models
        await orchestrator.initialize_models("AAPL")
        train_results = await orchestrator.train_models("AAPL", sample_price_data)
        
        # Check training results
        successful_models = [name for name, result in train_results.items() if result.get('success', False)]
        
        logger.info(f"✅ Ensemble training completed:")
        logger.info(f"   Successful models: {len(successful_models)}/{len(train_results)}")
        logger.info(f"   Models trained: {list(successful_models)}")
        
        # Should have at least some successful models
        assert len(successful_models) > 0, "At least one model should train successfully"


class TestHardwareAcceleration:
    """Test M4 Max hardware acceleration features"""
    
    def test_metal_gpu_detection(self):
        """Test Metal GPU detection"""
        try:
            import torch
            gpu_available = torch.backends.mps.is_available()
            logger.info(f"Metal GPU available: {gpu_available}")
            
            if gpu_available:
                logger.info("✅ M4 Max Metal GPU detected and available")
            else:
                logger.info("ℹ️  Metal GPU not available (expected on non-M4 Max systems)")
        except ImportError:
            logger.warning("PyTorch not available for GPU testing")
    
    def test_neural_engine_detection(self):
        """Test Neural Engine detection"""
        if NEURAL_ENGINE_OPTIMIZATION_AVAILABLE:
            logger.info("✅ Neural Engine optimization available (Core ML detected)")
        else:
            logger.info("ℹ️  Neural Engine optimization not available")
    
    def test_deep_learning_capabilities_summary(self):
        """Test and log complete deep learning capabilities"""
        capabilities = {
            "pytorch_available": DEEP_LEARNING_AVAILABLE,
            "neural_engine_optimization": NEURAL_ENGINE_OPTIMIZATION_AVAILABLE,
            "metal_gpu_available": False,
            "lstm_model_available": False,
            "transformer_model_available": False
        }
        
        try:
            import torch
            capabilities["metal_gpu_available"] = torch.backends.mps.is_available()
        except ImportError:
            pass
        
        if DEEP_LEARNING_AVAILABLE:
            try:
                from volatility.models.deep_learning_models import (
                    LSTMVolatilityPredictor,
                    TransformerVolatilityPredictor
                )
                capabilities["lstm_model_available"] = LSTMVolatilityPredictor is not None
                capabilities["transformer_model_available"] = TransformerVolatilityPredictor is not None
            except ImportError:
                pass
        
        logger.info("🚀 Deep Learning Capabilities Summary:")
        logger.info(f"   PyTorch Available: {capabilities['pytorch_available']}")
        logger.info(f"   Metal GPU Available: {capabilities['metal_gpu_available']}")
        logger.info(f"   Neural Engine Optimization: {capabilities['neural_engine_optimization']}")
        logger.info(f"   LSTM Model Available: {capabilities['lstm_model_available']}")
        logger.info(f"   Transformer Model Available: {capabilities['transformer_model_available']}")
        
        return capabilities


if __name__ == "__main__":
    """Run tests directly"""
    pytest.main([__file__, "-v", "-s"])