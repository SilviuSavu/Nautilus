"""
MessageBus Integration Tests for Volatility Engine

Tests real-time market data ingestion and volatility model updates via MessageBus.
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import logging

# Import volatility components
try:
    from volatility.streaming.messagebus_client import (
        VolatilityMessageBusClient,
        MarketDataEvent, 
        create_volatility_messagebus_client,
        MESSAGEBUS_AVAILABLE
    )
    from volatility.engine.volatility_engine import VolatilityEngine
    from volatility.config import get_default_volatility_config
    INTEGRATION_TESTS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"MessageBus integration tests unavailable: {e}")
    INTEGRATION_TESTS_AVAILABLE = False

logger = logging.getLogger(__name__)


@pytest.fixture
def messagebus_config():
    """Test MessageBus configuration"""
    return {
        'redis_host': 'localhost',
        'redis_port': 6379,
        'stream_key': 'test-volatility-streams',
        'consumer_group': 'test-volatility-group',
        'enable_streaming': True
    }


@pytest.fixture
def sample_market_data_event():
    """Sample market data event for testing"""
    return MarketDataEvent(
        symbol="AAPL",
        timestamp=datetime.utcnow(),
        data_type="bar",
        open=150.0,
        high=152.0,
        low=149.0,
        close=151.5,
        volume=1000000,
        source="test_data"
    )


@pytest.fixture
def mock_messagebus_client():
    """Mock MessageBus client for testing"""
    mock_client = AsyncMock()
    mock_client.initialize = AsyncMock()
    mock_client.subscribe = AsyncMock()
    mock_client.publish = AsyncMock()
    mock_client.start = AsyncMock()
    mock_client.stop = AsyncMock()
    mock_client.cleanup = AsyncMock()
    return mock_client


@pytest.mark.skipif(not INTEGRATION_TESTS_AVAILABLE, reason="Integration dependencies not available")
class TestVolatilityMessageBusClient:
    """Test volatility MessageBus client functionality"""
    
    def test_client_creation(self, messagebus_config):
        """Test MessageBus client creation"""
        client = create_volatility_messagebus_client(messagebus_config)
        
        if MESSAGEBUS_AVAILABLE:
            assert client is not None
            assert isinstance(client, VolatilityMessageBusClient)
            assert client.config == messagebus_config
            logger.info("✅ MessageBus client created successfully")
        else:
            assert client is None
            logger.info("ℹ️  MessageBus client creation skipped (dependencies unavailable)")
    
    @pytest.mark.asyncio
    async def test_market_data_event_processing(self, messagebus_config, sample_market_data_event):
        """Test market data event processing"""
        if not MESSAGEBUS_AVAILABLE:
            pytest.skip("MessageBus not available")
        
        client = VolatilityMessageBusClient(messagebus_config)
        
        # Mock the redis client to avoid actual connection
        with patch('volatility.streaming.messagebus_client.redis') as mock_redis:
            mock_redis.from_url.return_value = AsyncMock()
            
            # Add symbol for processing
            client.add_symbol("AAPL")
            
            # Process market data event
            await client._process_market_data_event(sample_market_data_event)
            
            # Check that event was buffered
            assert "AAPL" in client.data_buffer
            assert len(client.data_buffer["AAPL"]) == 1
            assert client.data_buffer["AAPL"][0].symbol == "AAPL"
            assert client.data_buffer["AAPL"][0].close == 151.5
            
            logger.info("✅ Market data event processed and buffered correctly")
    
    @pytest.mark.asyncio
    async def test_volatility_trigger_conditions(self, messagebus_config):
        """Test volatility update trigger conditions"""
        if not MESSAGEBUS_AVAILABLE:
            pytest.skip("MessageBus not available")
        
        client = VolatilityMessageBusClient(messagebus_config)
        client.add_symbol("AAPL")
        
        # Create sequence of price events with significant movement
        base_price = 100.0
        price_sequence = [100.0, 100.5, 101.0, 103.0, 102.0]  # 3% move triggers volatility update
        
        trigger_called = False
        
        async def mock_trigger(symbol, event, return_mag, trigger_type="price_movement"):
            nonlocal trigger_called
            trigger_called = True
        
        # Mock the trigger method
        client._trigger_volatility_update = mock_trigger
        
        # Process price sequence
        for i, price in enumerate(price_sequence):
            event = MarketDataEvent(
                symbol="AAPL",
                timestamp=datetime.utcnow() + timedelta(seconds=i),
                data_type="tick",
                price=price,
                close=price,
                source="test"
            )
            await client._process_market_data_event(event)
        
        # Large price move should have triggered volatility update
        assert trigger_called, "Volatility trigger should have been called for large price movement"
        
        logger.info("✅ Volatility trigger conditions working correctly")
    
    @pytest.mark.asyncio
    async def test_symbol_management(self, messagebus_config):
        """Test adding and removing symbols from tracking"""
        if not MESSAGEBUS_AVAILABLE:
            pytest.skip("MessageBus not available")
        
        client = VolatilityMessageBusClient(messagebus_config)
        
        # Test adding symbols
        client.add_symbol("AAPL")
        client.add_symbol("GOOGL")
        assert "AAPL" in client.active_symbols
        assert "GOOGL" in client.active_symbols
        assert len(client.active_symbols) == 2
        
        # Test removing symbols
        client.remove_symbol("AAPL")
        assert "AAPL" not in client.active_symbols
        assert "GOOGL" in client.active_symbols
        assert len(client.active_symbols) == 1
        
        # Test data buffer cleanup
        assert "AAPL" not in client.data_buffer
        
        logger.info("✅ Symbol management working correctly")
    
    @pytest.mark.asyncio
    async def test_streaming_stats(self, messagebus_config):
        """Test streaming performance statistics"""
        if not MESSAGEBUS_AVAILABLE:
            pytest.skip("MessageBus not available")
        
        client = VolatilityMessageBusClient(messagebus_config)
        client.add_symbol("AAPL")
        
        # Simulate some events processing
        client.events_processed = 100
        client.volatility_updates_triggered = 5
        
        stats = await client.get_streaming_stats()
        
        assert 'messagebus_connected' in stats
        assert 'active_symbols' in stats
        assert stats['active_symbols'] == ['AAPL']
        assert stats['events_processed'] == 100
        assert stats['volatility_updates_triggered'] == 5
        assert 'events_per_second' in stats
        
        logger.info("✅ Streaming statistics working correctly")
        logger.info(f"   Events processed: {stats['events_processed']}")
        logger.info(f"   Volatility updates: {stats['volatility_updates_triggered']}")


@pytest.mark.skipif(not INTEGRATION_TESTS_AVAILABLE, reason="Integration dependencies not available")
class TestVolatilityEngineMessageBusIntegration:
    """Test volatility engine integration with MessageBus"""
    
    @pytest.mark.asyncio
    async def test_engine_messagebus_initialization(self, messagebus_config):
        """Test volatility engine MessageBus initialization"""
        config = get_default_volatility_config()
        engine = VolatilityEngine(config)
        
        # Mock MessageBus creation to avoid actual connections
        with patch('volatility.engine.volatility_engine.create_volatility_messagebus_client') as mock_create:
            mock_client = AsyncMock()
            mock_client.initialize = AsyncMock()
            mock_client.start_streaming = AsyncMock()
            mock_create.return_value = mock_client
            
            # Mock Redis to avoid connection
            with patch('volatility.engine.volatility_engine.redis'):
                await engine.initialize()
                
                # Check MessageBus client was created and initialized
                assert engine.messagebus_client is not None
                mock_client.initialize.assert_called_once()
                mock_client.start_streaming.assert_called_once()
                
                logger.info("✅ Engine MessageBus initialization successful")
                
                await engine.shutdown()
    
    @pytest.mark.asyncio 
    async def test_symbol_messagebus_integration(self):
        """Test symbol addition integrates with MessageBus"""
        config = get_default_volatility_config()
        engine = VolatilityEngine(config)
        
        # Mock MessageBus client
        mock_messagebus_client = MagicMock()
        mock_messagebus_client.add_symbol = MagicMock()
        engine.messagebus_client = mock_messagebus_client
        
        # Mock orchestrator to avoid model initialization complexity
        with patch.object(engine.orchestrator, 'initialize_models') as mock_init:
            mock_init.return_value = {"test_model": MagicMock()}
            
            # Add symbol
            result = await engine.add_symbol("AAPL")
            
            # Check MessageBus integration
            mock_messagebus_client.add_symbol.assert_called_once_with("AAPL")
            assert result['status'] == 'success'
            assert "AAPL" in engine.active_symbols
            
            logger.info("✅ Symbol MessageBus integration working")
    
    @pytest.mark.asyncio
    async def test_volatility_trigger_handling(self):
        """Test volatility trigger handling from MessageBus"""
        config = get_default_volatility_config()
        engine = VolatilityEngine(config)
        
        # Setup engine state
        engine.active_symbols.add("AAPL")
        engine.is_running = True
        
        # Mock MessageBus client with recent data
        mock_messagebus_client = MagicMock()
        recent_data = [
            MarketDataEvent(
                symbol="AAPL", timestamp=datetime.utcnow(), data_type="bar",
                open=100.0, high=101.0, low=99.0, close=100.5, volume=1000000
            )
        ]
        mock_messagebus_client.get_recent_data = AsyncMock(return_value=recent_data)
        engine.messagebus_client = mock_messagebus_client
        
        # Mock forecast generation to avoid complex model operations
        with patch.object(engine, 'generate_forecast') as mock_forecast:
            mock_forecast.return_value = {
                'status': 'success',
                'forecast': {
                    'ensemble_volatility': 0.25,
                    'ensemble_confidence': 0.85
                }
            }
            
            # Create trigger event
            trigger_event = {
                'symbol': 'AAPL',
                'trigger_type': 'price_movement',
                'return_magnitude': 0.025
            }
            
            # Handle the trigger
            await engine._handle_volatility_trigger(trigger_event)
            
            # Check that forecast was called
            mock_forecast.assert_called_once()
            
            logger.info("✅ Volatility trigger handling working correctly")
    
    @pytest.mark.asyncio
    async def test_engine_status_includes_messagebus(self):
        """Test engine status includes MessageBus information"""
        config = get_default_volatility_config()
        engine = VolatilityEngine(config)
        engine.is_running = True
        engine.start_time = datetime.utcnow()
        
        # Mock MessageBus client with stats
        mock_messagebus_client = MagicMock()
        mock_messagebus_client.is_running = True
        mock_stats = {
            'events_processed': 1000,
            'volatility_updates_triggered': 50,
            'active_symbols': ['AAPL', 'GOOGL']
        }
        mock_messagebus_client.get_streaming_stats = AsyncMock(return_value=mock_stats)
        engine.messagebus_client = mock_messagebus_client
        
        # Mock orchestrator
        with patch.object(engine.orchestrator, 'get_model_status') as mock_status:
            mock_status.return_value = {'total_models': 5}
            
            status = await engine.get_engine_status()
            
            # Check MessageBus information is included
            assert 'external_services' in status
            assert status['external_services']['messagebus_connected'] == True
            assert 'messagebus_streaming_stats' in status['external_services']
            assert status['external_services']['messagebus_streaming_stats']['events_processed'] == 1000
            
            logger.info("✅ Engine status includes MessageBus information")
            logger.info(f"   MessageBus events processed: {status['external_services']['messagebus_streaming_stats']['events_processed']}")


class TestMessageBusAvailability:
    """Test MessageBus availability and fallback behavior"""
    
    def test_messagebus_availability_check(self):
        """Test MessageBus availability detection"""
        logger.info(f"MessageBus available: {MESSAGEBUS_AVAILABLE}")
        
        if MESSAGEBUS_AVAILABLE:
            logger.info("✅ MessageBus dependencies detected")
        else:
            logger.info("ℹ️  MessageBus dependencies not available - graceful fallback")
    
    def test_graceful_fallback_when_unavailable(self, messagebus_config):
        """Test graceful fallback when MessageBus is unavailable"""
        # This test ensures the system works even without MessageBus
        with patch('volatility.streaming.messagebus_client.MESSAGEBUS_AVAILABLE', False):
            client = create_volatility_messagebus_client(messagebus_config)
            assert client is None
            logger.info("✅ Graceful fallback working when MessageBus unavailable")


if __name__ == "__main__":
    """Run MessageBus integration tests"""
    pytest.main([__file__, "-v", "-s"])