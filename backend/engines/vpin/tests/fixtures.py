"""
Test fixtures and utilities for VPIN Engine testing
Provides reusable test data, mocks, and setup functions.
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from decimal import Decimal
from unittest.mock import Mock, MagicMock, AsyncMock

# NautilusTrader imports
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.data import TradeTick, QuoteTick
from nautilus_trader.model.enums import PriceType, AggressorSide
from nautilus_trader.model.objects import Price, Quantity

# VPIN imports  
from backend.engines.vpin.models import (
    VolumeBucket, VPINSignal, VPINConfiguration, MarketRegime,
    VPINSignalStrength, TradeSide, VPINDataQuality, VPIN_TIER1_SYMBOLS
)
from backend.engines.vpin.gpu_vpin_calculator import GPUAcceleratedVPIN
from backend.engines.vpin.neural_vpin_analyzer import NeuralVPINAnalyzer
from backend.engines.vpin.level2_data_collector import VPINDataCollector
from backend.engines.vpin.volume_synchronizer import VolumeSynchronizer
from backend.engines.vpin.vpin_engine import VPINEngine


@pytest.fixture
def sample_instrument_id():
    """Sample InstrumentId for testing"""
    return InstrumentId.from_str("AAPL.NASDAQ")


@pytest.fixture
def sample_vpin_config():
    """Standard VPIN configuration for testing"""
    return VPINConfiguration(
        symbol="AAPL",
        bucket_size=100_000.0,
        window_size=50,
        toxicity_threshold=0.65,
        extreme_toxicity_threshold=0.8,
        min_trades_per_bucket=10,
        enable_neural_analysis=True,
        enable_gpu_acceleration=True
    )


@pytest.fixture
def sample_volume_bucket():
    """Sample VolumeBucket for testing"""
    bucket = VolumeBucket(
        symbol="AAPL",
        bucket_id=1001,
        target_volume=100_000.0,
        buy_volume=58_000.0,
        sell_volume=42_000.0,
        total_volume=100_000.0,
        trade_count=127,
        start_time=1692891200_000_000_000,  # Mock nanosecond timestamp
        end_time=1692891260_000_000_000,    # 60 seconds later
        is_complete=True
    )
    bucket.calculate_order_imbalance()
    return bucket


@pytest.fixture
def sample_vpin_signal():
    """Sample VPINSignal for testing"""
    return VPINSignal(
        symbol="AAPL",
        vpin_value=0.42,
        toxicity_probability=0.42,
        market_regime=MarketRegime.NORMAL,
        signal_strength=VPINSignalStrength.MODERATE,
        regime_confidence=0.87,
        timestamp=1692891200_000_000_000,
        data_quality=VPINDataQuality(
            completeness_score=0.95,
            latency_score=0.88,
            accuracy_score=0.92
        )
    )


@pytest.fixture
def sample_trade_ticks():
    """Generate sample TradeTick data for testing"""
    ticks = []
    base_price = 150.0
    base_time = 1692891200_000_000_000
    
    for i in range(100):
        # Simulate realistic price movement
        price_change = np.random.normal(0, 0.01)
        price = Price.from_str(f"{base_price + price_change:.2f}")
        size = Quantity.from_int(np.random.randint(100, 2000))
        
        tick = TradeTick(
            instrument_id=InstrumentId.from_str("AAPL.NASDAQ"),
            price=price,
            size=size,
            aggressor_side=AggressorSide.BUYER if np.random.random() > 0.5 else AggressorSide.SELLER,
            trade_id=f"trade_{i}",
            ts_event=base_time + (i * 1_000_000_000),  # 1 second apart
            ts_init=base_time + (i * 1_000_000_000)
        )
        ticks.append(tick)
    
    return ticks


@pytest.fixture
def mock_gpu_calculator():
    """Mock GPU-accelerated VPIN calculator"""
    mock = Mock(spec=GPUAcceleratedVPIN)
    mock.calculate_vpin_batch = AsyncMock(return_value=0.42)
    mock.is_gpu_available = Mock(return_value=True)
    mock.get_performance_metrics = Mock(return_value={
        "calculation_time_ms": 1.8,
        "gpu_utilization": 0.85,
        "memory_usage_mb": 256
    })
    return mock


@pytest.fixture
def mock_neural_analyzer():
    """Mock Neural Engine VPIN analyzer"""
    mock = Mock(spec=NeuralVPINAnalyzer)
    mock.analyze_vpin_patterns = AsyncMock(return_value={
        "market_regime": MarketRegime.NORMAL,
        "regime_confidence": 0.87,
        "pattern_prediction": "bullish"
    })
    mock.is_neural_engine_available = Mock(return_value=True)
    mock.get_inference_time = Mock(return_value=4.2)
    return mock


@pytest.fixture
def mock_level2_collector():
    """Mock Level 2 data collector"""
    mock = Mock(spec=VPINDataCollector)
    mock.subscribe_to_symbol = AsyncMock(return_value=True)
    mock.unsubscribe_from_symbol = AsyncMock(return_value=True)
    mock.get_active_subscriptions = Mock(return_value=["AAPL", "TSLA"])
    mock.get_data_quality = Mock(return_value=VPINDataQuality(
        completeness_score=0.95,
        latency_score=0.88,
        accuracy_score=0.92
    ))
    return mock


@pytest.fixture
def mock_volume_synchronizer():
    """Mock Volume Synchronizer"""
    mock = Mock(spec=VolumeSynchronizer)
    mock.add_trade = AsyncMock(return_value=None)
    mock.get_current_bucket = Mock(return_value=None)  # No complete bucket initially
    mock.get_bucket_completion_progress = Mock(return_value=0.76)
    mock.classify_trade_side = AsyncMock(return_value=TradeSide.BUY)
    return mock


@pytest.fixture
def mock_vpin_engine():
    """Mock VPIN Engine for integration testing"""
    mock = Mock(spec=VPINEngine)
    mock.start = AsyncMock(return_value=True)
    mock.stop = AsyncMock(return_value=True)
    mock.get_status = Mock(return_value={
        "is_running": True,
        "active_symbols": 3,
        "total_calculations": 12450,
        "performance_score": 0.94
    })
    mock.subscribe_to_symbol = AsyncMock(return_value=True)
    mock.get_realtime_vpin = AsyncMock(return_value=0.42)
    return mock


@pytest.fixture
def level2_order_book_data():
    """Sample Level 2 order book data"""
    return {
        "symbol": "AAPL",
        "timestamp": 1692891200_000_000_000,
        "bids": [
            {"price": 150.45, "size": 500, "exchange": "NYSE", "mm": "CDRG"},
            {"price": 150.44, "size": 300, "exchange": "NASDAQ", "mm": "ARCA"},
            {"price": 150.43, "size": 800, "exchange": "BATS", "mm": "CITADEL"},
            {"price": 150.42, "size": 200, "exchange": "EDGX", "mm": "VIRTU"},
            {"price": 150.41, "size": 600, "exchange": "IEX", "mm": "IEX"}
        ],
        "asks": [
            {"price": 150.46, "size": 400, "exchange": "NASDAQ", "mm": "ARCA"},
            {"price": 150.47, "size": 700, "exchange": "NYSE", "mm": "CDRG"},
            {"price": 150.48, "size": 300, "exchange": "BATS", "mm": "CITADEL"},
            {"price": 150.49, "size": 500, "exchange": "EDGX", "mm": "VIRTU"},
            {"price": 150.50, "size": 200, "exchange": "IEX", "mm": "IEX"}
        ]
    }


@pytest.fixture
def market_regime_scenarios():
    """Different market regime scenarios for testing"""
    return {
        "normal": {
            "vpin_values": [0.2, 0.3, 0.4, 0.35, 0.25],
            "expected_regime": MarketRegime.NORMAL,
            "volatility": "low"
        },
        "stressed": {
            "vpin_values": [0.5, 0.6, 0.7, 0.65, 0.55],
            "expected_regime": MarketRegime.STRESSED,
            "volatility": "medium"
        },
        "toxic": {
            "vpin_values": [0.7, 0.8, 0.85, 0.9, 0.75],
            "expected_regime": MarketRegime.TOXIC,
            "volatility": "high"
        },
        "extreme": {
            "vpin_values": [0.85, 0.9, 0.95, 0.92, 0.88],
            "expected_regime": MarketRegime.EXTREME,
            "volatility": "very_high"
        }
    }


@pytest.fixture
def performance_targets():
    """Expected performance targets for validation"""
    return {
        "vpin_calculation_time_ms": 2.0,
        "neural_analysis_time_ms": 5.0,
        "api_response_time_ms": 10.0,
        "gpu_utilization_min": 0.8,
        "neural_engine_utilization_min": 0.7,
        "trade_classification_accuracy": 0.95,
        "concurrent_symbols_max": 8,
        "calculations_per_second": 1000
    }


@pytest.fixture
async def async_mock_setup():
    """Setup async mock objects"""
    loop = asyncio.get_event_loop()
    
    # Create async mocks
    async_mocks = {
        "gpu_calculator": AsyncMock(spec=GPUAcceleratedVPIN),
        "neural_analyzer": AsyncMock(spec=NeuralVPINAnalyzer),
        "level2_collector": AsyncMock(spec=VPINDataCollector),
        "volume_synchronizer": AsyncMock(spec=VolumeSynchronizer)
    }
    
    # Configure async mock behaviors
    async_mocks["gpu_calculator"].calculate_vpin_batch.return_value = 0.42
    async_mocks["neural_analyzer"].analyze_vpin_patterns.return_value = {
        "regime": MarketRegime.NORMAL,
        "confidence": 0.87
    }
    async_mocks["level2_collector"].subscribe_to_symbol.return_value = True
    async_mocks["volume_synchronizer"].add_trade.return_value = None
    
    return async_mocks


@pytest.fixture
def test_symbols():
    """List of symbols for multi-symbol testing"""
    return ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "NFLX"]


@pytest.fixture
def hardware_acceleration_config():
    """Hardware acceleration configuration for testing"""
    return {
        "enable_metal_gpu": True,
        "enable_neural_engine": True,
        "enable_cpu_optimization": True,
        "gpu_memory_limit_mb": 512,
        "neural_engine_batch_size": 64,
        "fallback_to_cpu": True
    }


class VPINTestHelper:
    """Helper class for VPIN testing utilities"""
    
    @staticmethod
    def generate_realistic_vpin_sequence(length: int, regime: MarketRegime) -> List[float]:
        """Generate realistic VPIN value sequence for testing"""
        base_values = {
            MarketRegime.NORMAL: 0.3,
            MarketRegime.STRESSED: 0.6,
            MarketRegime.TOXIC: 0.8,
            MarketRegime.EXTREME: 0.9
        }
        
        base = base_values[regime]
        noise_level = 0.1 if regime == MarketRegime.NORMAL else 0.05
        
        values = []
        for i in range(length):
            noise = np.random.normal(0, noise_level)
            value = max(0.0, min(1.0, base + noise))
            values.append(value)
            
        return values
    
    @staticmethod
    def create_volume_bucket_sequence(symbol: str, count: int) -> List[VolumeBucket]:
        """Create sequence of volume buckets for testing"""
        buckets = []
        base_time = 1692891200_000_000_000
        
        for i in range(count):
            buy_vol = np.random.uniform(40_000, 70_000)
            sell_vol = 100_000 - buy_vol
            
            bucket = VolumeBucket(
                symbol=symbol,
                bucket_id=i + 1,
                target_volume=100_000.0,
                buy_volume=buy_vol,
                sell_volume=sell_vol,
                total_volume=100_000.0,
                trade_count=np.random.randint(50, 200),
                start_time=base_time + (i * 60 * 1_000_000_000),
                end_time=base_time + ((i + 1) * 60 * 1_000_000_000),
                is_complete=True
            )
            bucket.calculate_order_imbalance()
            buckets.append(bucket)
            
        return buckets
    
    @staticmethod
    def validate_vpin_signal(signal: VPINSignal) -> bool:
        """Validate VPIN signal structure and values"""
        if not 0.0 <= signal.vpin_value <= 1.0:
            return False
        if not 0.0 <= signal.toxicity_probability <= 1.0:
            return False
        if not 0.0 <= signal.regime_confidence <= 1.0:
            return False
        if signal.timestamp <= 0:
            return False
        return True


@pytest.fixture
def vpin_test_helper():
    """VPIN test helper instance"""
    return VPINTestHelper()


# Test configuration constants
TEST_CONFIG = {
    "DEFAULT_SYMBOL": "AAPL",
    "DEFAULT_BUCKET_SIZE": 100_000.0,
    "DEFAULT_WINDOW_SIZE": 50,
    "TEST_TIMEOUT": 30.0,
    "PERFORMANCE_ITERATIONS": 100,
    "CONCURRENT_SYMBOLS": 8
}