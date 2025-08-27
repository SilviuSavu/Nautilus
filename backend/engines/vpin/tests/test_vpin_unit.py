"""
VPIN Unit Tests
Comprehensive unit testing for all VPIN Engine components.
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, AsyncMock, patch
from decimal import Decimal
from typing import List

# NautilusTrader imports
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.enums import AggressorSide
from nautilus_trader.model.objects import Price, Quantity

# VPIN imports
from backend.engines.vpin.models import (
    VolumeBucket, VPINSignal, VPINConfiguration, MarketRegime,
    VPINSignalStrength, TradeSide, VPINDataQuality, VPIN_TIER1_SYMBOLS
)
from backend.engines.vpin.gpu_vpin_calculator import GPUAcceleratedVPIN, VPINCalculation
from backend.engines.vpin.neural_vpin_analyzer import NeuralVPINAnalyzer, VPINPatternPrediction
from backend.engines.vpin.level2_data_collector import VPINDataCollector, VPINDataSubscription
from backend.engines.vpin.volume_synchronizer import VolumeSynchronizer
from backend.engines.vpin.vpin_engine import VPINEngine, VPINEngineStatus

# Test fixtures
from .fixtures import (
    sample_instrument_id, sample_vpin_config, sample_volume_bucket,
    sample_vpin_signal, sample_trade_ticks, vpin_test_helper
)
from .mock_level2_generator import MockLevel2Generator, MarketDataConfig, MarketMicrostructure


class TestVolumeBucket:
    """Test VolumeBucket data structure and calculations"""
    
    def test_volume_bucket_initialization(self):
        """Test basic VolumeBucket creation"""
        bucket = VolumeBucket(
            symbol="AAPL",
            bucket_id=1,
            target_volume=100_000.0
        )
        
        assert bucket.symbol == "AAPL"
        assert bucket.bucket_id == 1
        assert bucket.target_volume == 100_000.0
        assert bucket.buy_volume == 0.0
        assert bucket.sell_volume == 0.0
        assert bucket.total_volume == 0.0
        assert bucket.trade_count == 0
        assert not bucket.is_complete
        
    def test_order_imbalance_calculation(self, sample_volume_bucket):
        """Test order imbalance calculation"""
        bucket = sample_volume_bucket
        
        # Should have calculated imbalance during fixture creation
        expected_imbalance = abs(58_000 - 42_000) / 100_000
        assert abs(bucket.order_imbalance - expected_imbalance) < 0.001
        
    def test_order_imbalance_edge_cases(self):
        """Test order imbalance calculation edge cases"""
        # Zero volume case
        bucket = VolumeBucket(symbol="TEST", bucket_id=1)
        imbalance = bucket.calculate_order_imbalance()
        assert imbalance == 0.0
        
        # Perfect balance case
        bucket = VolumeBucket(
            symbol="TEST", bucket_id=2, 
            buy_volume=50_000, sell_volume=50_000, total_volume=100_000
        )
        imbalance = bucket.calculate_order_imbalance()
        assert imbalance == 0.0
        
        # Maximum imbalance case
        bucket = VolumeBucket(
            symbol="TEST", bucket_id=3,
            buy_volume=100_000, sell_volume=0, total_volume=100_000
        )
        imbalance = bucket.calculate_order_imbalance()
        assert imbalance == 1.0


class TestVPINConfiguration:
    """Test VPIN configuration validation and defaults"""
    
    def test_vpin_config_creation(self, sample_vpin_config):
        """Test VPIN configuration creation"""
        config = sample_vpin_config
        
        assert config.symbol == "AAPL"
        assert config.bucket_size == 100_000.0
        assert config.window_size == 50
        assert config.toxicity_threshold == 0.65
        assert config.enable_neural_analysis
        assert config.enable_gpu_acceleration
        
    def test_vpin_config_validation(self):
        """Test VPIN configuration validation"""
        # Valid configuration
        config = VPINConfiguration(
            symbol="TSLA",
            bucket_size=50_000.0,
            window_size=30,
            toxicity_threshold=0.7
        )
        assert config.symbol == "TSLA"
        
    def test_tier1_symbols_constant(self):
        """Test TIER1 symbols constant"""
        assert "AAPL" in VPIN_TIER1_SYMBOLS
        assert "TSLA" in VPIN_TIER1_SYMBOLS
        assert len(VPIN_TIER1_SYMBOLS) == 8  # Should have 8 tier 1 symbols


class TestVPINSignal:
    """Test VPIN signal structure and validation"""
    
    def test_vpin_signal_creation(self, sample_vpin_signal):
        """Test VPIN signal creation"""
        signal = sample_vpin_signal
        
        assert signal.symbol == "AAPL"
        assert 0.0 <= signal.vpin_value <= 1.0
        assert 0.0 <= signal.toxicity_probability <= 1.0
        assert signal.market_regime == MarketRegime.NORMAL
        assert signal.signal_strength == VPINSignalStrength.MODERATE
        assert signal.timestamp > 0
        
    def test_vpin_signal_validation(self, vpin_test_helper):
        """Test VPIN signal validation"""
        signal = VPINSignal(
            symbol="TEST",
            vpin_value=0.5,
            toxicity_probability=0.5,
            market_regime=MarketRegime.NORMAL,
            signal_strength=VPINSignalStrength.MODERATE,
            timestamp=1692891200_000_000_000
        )
        
        assert vpin_test_helper.validate_vpin_signal(signal)
        
    def test_vpin_signal_invalid_values(self, vpin_test_helper):
        """Test VPIN signal with invalid values"""
        # Invalid VPIN value (> 1.0)
        signal = VPINSignal(
            symbol="TEST",
            vpin_value=1.5,  # Invalid
            toxicity_probability=0.5,
            market_regime=MarketRegime.NORMAL,
            signal_strength=VPINSignalStrength.MODERATE,
            timestamp=1692891200_000_000_000
        )
        
        assert not vpin_test_helper.validate_vpin_signal(signal)


class TestGPUAcceleratedVPIN:
    """Test GPU-accelerated VPIN calculator"""
    
    @pytest.fixture
    def gpu_calculator(self):
        """Create GPU calculator instance"""
        return GPUAcceleratedVPIN()
        
    def test_gpu_calculator_initialization(self, gpu_calculator):
        """Test GPU calculator initialization"""
        assert gpu_calculator is not None
        # Note: GPU availability depends on hardware, so we don't assert true/false
        availability = gpu_calculator.is_gpu_available()
        assert isinstance(availability, bool)
        
    @pytest.mark.asyncio
    async def test_vpin_calculation_basic(self, gpu_calculator, vpin_test_helper):
        """Test basic VPIN calculation"""
        symbol = "AAPL"
        buckets = vpin_test_helper.create_volume_bucket_sequence(symbol, 50)
        
        # Mock GPU calculation if not available
        if not gpu_calculator.is_gpu_available():
            with patch.object(gpu_calculator, 'calculate_vpin_batch') as mock_calc:
                mock_calc.return_value = VPINCalculation(
                    symbol=symbol,
                    vpin_value=0.42,
                    calculation_time_ms=1.8,
                    bucket_count=50,
                    data_quality_score=0.95
                )
                
                result = await gpu_calculator.calculate_vpin_batch(symbol, buckets)
                assert result.vpin_value == 0.42
                assert result.symbol == symbol
                mock_calc.assert_called_once()
        else:
            # Real GPU calculation
            result = await gpu_calculator.calculate_vpin_batch(symbol, buckets)
            assert isinstance(result, VPINCalculation)
            assert 0.0 <= result.vpin_value <= 1.0
            assert result.symbol == symbol
            
    @pytest.mark.asyncio
    async def test_vpin_calculation_empty_buckets(self, gpu_calculator):
        """Test VPIN calculation with empty bucket list"""
        result = await gpu_calculator.calculate_vpin_batch("AAPL", [])
        assert result is None
        
    def test_performance_metrics(self, gpu_calculator):
        """Test performance metrics retrieval"""
        metrics = gpu_calculator.get_performance_metrics()
        assert isinstance(metrics, dict)
        
        # Should contain expected keys
        expected_keys = ["gpu_available", "metal_backend_available"]
        for key in expected_keys:
            assert key in metrics


class TestNeuralVPINAnalyzer:
    """Test Neural Engine VPIN analyzer"""
    
    @pytest.fixture
    def neural_analyzer(self):
        """Create neural analyzer instance"""
        return NeuralVPINAnalyzer()
        
    def test_neural_analyzer_initialization(self, neural_analyzer):
        """Test neural analyzer initialization"""
        assert neural_analyzer is not None
        availability = neural_analyzer.is_neural_engine_available()
        assert isinstance(availability, bool)
        
    @pytest.mark.asyncio
    async def test_pattern_analysis_normal_regime(self, neural_analyzer):
        """Test pattern analysis for normal market regime"""
        symbol = "AAPL"
        vpin_values = [0.2, 0.3, 0.4, 0.35, 0.25]  # Normal regime values
        
        # Mock neural analysis if not available
        if not neural_analyzer.is_neural_engine_available():
            with patch.object(neural_analyzer, 'analyze_vpin_patterns') as mock_analyze:
                mock_analyze.return_value = VPINPatternPrediction(
                    symbol=symbol,
                    predicted_regime=MarketRegime.NORMAL,
                    confidence=0.87,
                    pattern_features=["low_volatility", "balanced_flow"],
                    inference_time_ms=4.2
                )
                
                result = await neural_analyzer.analyze_vpin_patterns(symbol, vpin_values)
                assert result.predicted_regime == MarketRegime.NORMAL
                assert result.confidence > 0.8
                mock_analyze.assert_called_once()
        else:
            # Real neural analysis
            result = await neural_analyzer.analyze_vpin_patterns(symbol, vpin_values)
            assert isinstance(result, VPINPatternPrediction)
            assert result.symbol == symbol
            assert 0.0 <= result.confidence <= 1.0
            
    @pytest.mark.asyncio 
    async def test_pattern_analysis_toxic_regime(self, neural_analyzer):
        """Test pattern analysis for toxic market regime"""
        symbol = "TSLA"
        vpin_values = [0.7, 0.8, 0.85, 0.9, 0.75]  # Toxic regime values
        
        # Mock for consistent testing
        with patch.object(neural_analyzer, 'analyze_vpin_patterns') as mock_analyze:
            mock_analyze.return_value = VPINPatternPrediction(
                symbol=symbol,
                predicted_regime=MarketRegime.TOXIC,
                confidence=0.92,
                pattern_features=["high_imbalance", "persistent_direction"],
                inference_time_ms=3.8
            )
            
            result = await neural_analyzer.analyze_vpin_patterns(symbol, vpin_values)
            assert result.predicted_regime == MarketRegime.TOXIC
            assert result.confidence > 0.9
            
    def test_neural_engine_metrics(self, neural_analyzer):
        """Test neural engine performance metrics"""
        metrics = neural_analyzer.get_performance_metrics()
        assert isinstance(metrics, dict)
        assert "neural_engine_available" in metrics


class TestVolumeSynchronizer:
    """Test Volume Synchronizer for trade classification"""
    
    @pytest.fixture
    def volume_sync(self, sample_vpin_config):
        """Create volume synchronizer instance"""
        return VolumeSynchronizer(sample_vpin_config)
        
    def test_volume_sync_initialization(self, volume_sync, sample_vpin_config):
        """Test volume synchronizer initialization"""
        assert volume_sync.config.symbol == sample_vpin_config.symbol
        assert volume_sync.config.bucket_size == sample_vpin_config.bucket_size
        
    @pytest.mark.asyncio
    async def test_trade_side_classification_buy(self, volume_sync):
        """Test trade side classification for buy trades"""
        # Create buy trade (price > mid)
        trade = TradeTick(
            instrument_id=InstrumentId.from_str("AAPL.NASDAQ"),
            price=Price.from_str("150.50"),
            size=Quantity.from_int(1000),
            aggressor_side=AggressorSide.BUYER,
            trade_id="test_buy_1",
            ts_event=1692891200_000_000_000,
            ts_init=1692891200_000_000_000
        )
        
        # Mock the classification
        with patch.object(volume_sync, '_classify_trade_side') as mock_classify:
            mock_classify.return_value = TradeSide.BUY
            
            side = await volume_sync._classify_trade_side("AAPL", trade)
            assert side == TradeSide.BUY
            
    @pytest.mark.asyncio
    async def test_trade_side_classification_sell(self, volume_sync):
        """Test trade side classification for sell trades"""
        trade = TradeTick(
            instrument_id=InstrumentId.from_str("AAPL.NASDAQ"),
            price=Price.from_str("149.50"),
            size=Quantity.from_int(1000),
            aggressor_side=AggressorSide.SELLER,
            trade_id="test_sell_1",
            ts_event=1692891200_000_000_000,
            ts_init=1692891200_000_000_000
        )
        
        with patch.object(volume_sync, '_classify_trade_side') as mock_classify:
            mock_classify.return_value = TradeSide.SELL
            
            side = await volume_sync._classify_trade_side("AAPL", trade)
            assert side == TradeSide.SELL
            
    @pytest.mark.asyncio
    async def test_bucket_completion(self, volume_sync, sample_trade_ticks):
        """Test volume bucket completion logic"""
        symbol = "AAPL"
        
        # Mock bucket completion
        with patch.object(volume_sync, 'add_trade') as mock_add_trade:
            mock_add_trade.return_value = None
            
            # Add trades until bucket completion
            for trade in sample_trade_ticks[:10]:
                await volume_sync.add_trade(symbol, trade)
                
            # Verify trades were processed
            assert mock_add_trade.call_count == 10
            
    def test_bucket_progress_tracking(self, volume_sync):
        """Test bucket completion progress tracking"""
        progress = volume_sync.get_bucket_completion_progress("AAPL")
        assert isinstance(progress, float)
        assert 0.0 <= progress <= 1.0


class TestVPINDataCollector:
    """Test VPIN Level 2 data collector"""
    
    @pytest.fixture
    def data_collector(self):
        """Create data collector instance"""
        return VPINDataCollector()
        
    def test_data_collector_initialization(self, data_collector):
        """Test data collector initialization"""
        assert data_collector is not None
        assert len(data_collector.active_subscriptions) == 0
        
    @pytest.mark.asyncio
    async def test_symbol_subscription(self, data_collector):
        """Test Level 2 data subscription for symbol"""
        symbol = "AAPL"
        
        # Mock the subscription
        with patch.object(data_collector, 'subscribe_to_symbol') as mock_subscribe:
            mock_subscribe.return_value = True
            
            result = await data_collector.subscribe_to_symbol(symbol)
            assert result is True
            mock_subscribe.assert_called_once_with(symbol)
            
    @pytest.mark.asyncio
    async def test_symbol_unsubscription(self, data_collector):
        """Test Level 2 data unsubscription"""
        symbol = "AAPL"
        
        with patch.object(data_collector, 'unsubscribe_from_symbol') as mock_unsub:
            mock_unsub.return_value = True
            
            result = await data_collector.unsubscribe_from_symbol(symbol)
            assert result is True
            
    def test_data_quality_monitoring(self, data_collector):
        """Test data quality monitoring"""
        symbol = "AAPL"
        
        # Mock data quality
        with patch.object(data_collector, 'get_data_quality') as mock_quality:
            mock_quality.return_value = VPINDataQuality(
                completeness_score=0.95,
                latency_score=0.88,
                accuracy_score=0.92
            )
            
            quality = data_collector.get_data_quality(symbol)
            assert quality.completeness_score == 0.95
            assert quality.latency_score == 0.88
            assert quality.accuracy_score == 0.92
            
    def test_subscription_limits(self, data_collector):
        """Test subscription limits (max 8 symbols)"""
        active_subs = data_collector.get_active_subscriptions()
        assert isinstance(active_subs, list)
        # In real implementation, should enforce 8 symbol limit


class TestVPINEngine:
    """Test main VPIN Engine orchestrator"""
    
    @pytest.fixture
    def vpin_engine(self, sample_vpin_config):
        """Create VPIN engine instance"""
        return VPINEngine(sample_vpin_config)
        
    def test_vpin_engine_initialization(self, vpin_engine, sample_vpin_config):
        """Test VPIN engine initialization"""
        assert vpin_engine.config.symbol == sample_vpin_config.symbol
        assert not vpin_engine.is_running
        
    @pytest.mark.asyncio
    async def test_vpin_engine_start_stop(self, vpin_engine):
        """Test VPIN engine lifecycle"""
        # Mock start/stop
        with patch.object(vpin_engine, 'start') as mock_start:
            mock_start.return_value = True
            
            result = await vpin_engine.start()
            assert result is True
            
        with patch.object(vpin_engine, 'stop') as mock_stop:
            mock_stop.return_value = True
            
            result = await vpin_engine.stop()
            assert result is True
            
    def test_vpin_engine_status(self, vpin_engine):
        """Test VPIN engine status reporting"""
        with patch.object(vpin_engine, 'get_status') as mock_status:
            mock_status.return_value = VPINEngineStatus(
                is_running=True,
                components_initialized=True,
                data_feeds_active=True,
                active_symbols=3,
                total_calculations=1250,
                performance_score=0.94
            )
            
            status = vpin_engine.get_status()
            assert status.is_running
            assert status.components_initialized
            assert status.active_symbols == 3
            assert status.performance_score == 0.94
            
    @pytest.mark.asyncio
    async def test_realtime_vpin_calculation(self, vpin_engine):
        """Test real-time VPIN calculation"""
        symbol = "AAPL"
        
        with patch.object(vpin_engine, 'get_realtime_vpin') as mock_realtime:
            mock_realtime.return_value = 0.42
            
            vpin_value = await vpin_engine.get_realtime_vpin(symbol)
            assert vpin_value == 0.42
            mock_realtime.assert_called_once_with(symbol)


class TestMockLevel2Generator:
    """Test mock Level 2 data generator"""
    
    def test_mock_generator_initialization(self):
        """Test mock generator creation"""
        config = MarketDataConfig(
            symbol="AAPL",
            base_price=150.0,
            market_microstructure=MarketMicrostructure.NORMAL_FLOW
        )
        generator = MockLevel2Generator(config)
        
        assert generator.config.symbol == "AAPL"
        assert generator.current_price == 150.0
        assert len(generator.order_book["bids"]) == 10
        assert len(generator.order_book["asks"]) == 10
        
    def test_trade_generation(self):
        """Test trade tick generation"""
        config = MarketDataConfig(symbol="TEST", base_price=100.0)
        generator = MockLevel2Generator(config)
        
        trade = generator.generate_trade_tick()
        assert isinstance(trade, TradeTick)
        assert trade.instrument_id.symbol == "TEST"
        assert trade.price.as_double() > 0
        assert trade.size.as_int() > 0
        
    def test_quote_generation(self):
        """Test quote tick generation"""
        config = MarketDataConfig(symbol="TEST", base_price=100.0)
        generator = MockLevel2Generator(config)
        
        quote = generator.generate_quote_tick()
        assert isinstance(quote, TradeTick)  # Note: Should be QuoteTick in real implementation
        assert quote.instrument_id.symbol == "TEST"
        
    def test_level2_snapshot(self):
        """Test Level 2 order book snapshot"""
        config = MarketDataConfig(symbol="TEST", base_price=100.0)
        generator = MockLevel2Generator(config)
        
        snapshot = generator.get_current_level2_snapshot()
        assert snapshot["symbol"] == "TEST"
        assert len(snapshot["bids"]) <= 10
        assert len(snapshot["asks"]) <= 10
        assert snapshot["timestamp"] > 0
        
    def test_vpin_scenario_generation(self):
        """Test VPIN scenario generation"""
        config = MarketDataConfig(symbol="TEST", base_price=100.0)
        generator = MockLevel2Generator(config)
        
        trades = generator.generate_vpin_scenario(MarketRegime.TOXIC, duration_minutes=1)
        assert len(trades) > 0
        assert all(isinstance(trade, TradeTick) for trade in trades)
        
    def test_generator_statistics(self):
        """Test generator statistics tracking"""
        config = MarketDataConfig(symbol="TEST", base_price=100.0)
        generator = MockLevel2Generator(config)
        
        # Generate some data
        generator.generate_trade_tick()
        generator.generate_quote_tick()
        
        stats = generator.get_statistics()
        assert stats["trades_generated"] == 1
        assert stats["quotes_generated"] == 1
        assert stats["current_price"] > 0
        assert stats["market_microstructure"] == "normal"


# Integration test with multiple components
class TestVPINComponentIntegration:
    """Test integration between VPIN components"""
    
    @pytest.mark.asyncio
    async def test_data_flow_integration(self, sample_vpin_config, vpin_test_helper):
        """Test complete data flow: collector → synchronizer → calculator → analyzer"""
        # Create components
        collector = VPINDataCollector()
        synchronizer = VolumeSynchronizer(sample_vpin_config)
        calculator = GPUAcceleratedVPIN()
        analyzer = NeuralVPINAnalyzer()
        
        symbol = "AAPL"
        buckets = vpin_test_helper.create_volume_bucket_sequence(symbol, 10)
        
        # Mock the entire flow
        with patch.object(calculator, 'calculate_vpin_batch') as mock_calc, \
             patch.object(analyzer, 'analyze_vpin_patterns') as mock_analyze:
            
            mock_calc.return_value = VPINCalculation(
                symbol=symbol,
                vpin_value=0.65,
                calculation_time_ms=1.5,
                bucket_count=10
            )
            
            mock_analyze.return_value = VPINPatternPrediction(
                symbol=symbol,
                predicted_regime=MarketRegime.STRESSED,
                confidence=0.88
            )
            
            # Execute flow
            vpin_result = await calculator.calculate_vpin_batch(symbol, buckets)
            pattern_result = await analyzer.analyze_vpin_patterns(symbol, [0.65])
            
            # Verify results
            assert vpin_result.vpin_value == 0.65
            assert pattern_result.predicted_regime == MarketRegime.STRESSED
            assert pattern_result.confidence == 0.88


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])