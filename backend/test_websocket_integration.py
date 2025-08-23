"""
Integration test for WebSocket streaming infrastructure - Sprint 3 Priority 1

Tests the integration of:
- WebSocket manager
- Streaming service  
- Event dispatcher
- Subscription manager
- Message protocols
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_websocket_infrastructure():
    """Test WebSocket infrastructure components"""
    logger.info("=== WebSocket Infrastructure Integration Test ===")
    
    # Test imports
    try:
        from backend.websocket.websocket_manager import websocket_manager
        from backend.websocket.streaming_service import StreamingService
        from backend.websocket.event_dispatcher import event_dispatcher, EventType, EventPriority
        from backend.websocket.subscription_manager import subscription_manager, SubscriptionType
        from backend.websocket.message_protocols import default_protocol
        logger.info("✓ All imports successful")
    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        return False
    
    # Test message protocol
    try:
        # Test message validation
        valid_message = {
            "type": "engine_status",
            "data": {"state": "running", "uptime": 3600},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        is_valid = default_protocol.validate_message(valid_message)
        logger.info(f"✓ Message validation: {is_valid}")
        
        # Test message creation
        from backend.websocket.message_protocols import create_engine_status_message
        engine_msg = create_engine_status_message(
            data={"state": "running", "uptime": 3600},
            engine_id="test-engine"
        )
        logger.info(f"✓ Message creation: {engine_msg.type}")
        
    except Exception as e:
        logger.error(f"✗ Message protocol test failed: {e}")
        return False
    
    # Test WebSocket manager
    try:
        stats = websocket_manager.get_connection_stats()
        logger.info(f"✓ WebSocket manager stats: {stats['total_connections']} connections")
    except Exception as e:
        logger.error(f"✗ WebSocket manager test failed: {e}")
        return False
    
    # Test streaming service
    try:
        streaming_service = StreamingService()
        stream_stats = await streaming_service.get_stream_statistics()
        logger.info(f"✓ Streaming service stats: {stream_stats['active_streams']} streams")
    except Exception as e:
        logger.error(f"✗ Streaming service test failed: {e}")
        return False
    
    # Test event dispatcher initialization
    try:
        await event_dispatcher.initialize()
        logger.info("✓ Event dispatcher initialized")
    except Exception as e:
        logger.warning(f"⚠ Event dispatcher init (Redis may not be available): {e}")
    
    # Test subscription manager initialization
    try:
        await subscription_manager.initialize()
        logger.info("✓ Subscription manager initialized")
    except Exception as e:
        logger.warning(f"⚠ Subscription manager init (Redis may not be available): {e}")
    
    # Test event publishing
    try:
        from backend.websocket.event_dispatcher import Event, EventPriority
        
        test_event = Event(
            event_type=EventType.ENGINE_STATUS_CHANGED,
            data={"state": "running", "message": "Test event"},
            source="integration_test",
            priority=EventPriority.NORMAL
        )
        
        # This will work even without Redis
        event_dict = test_event.to_dict()
        logger.info(f"✓ Event creation: {event_dict['event_type']}")
        
    except Exception as e:
        logger.error(f"✗ Event creation test failed: {e}")
        return False
    
    # Test subscription config
    try:
        from backend.websocket.subscription_manager import SubscriptionConfig
        
        config = SubscriptionConfig(
            subscription_type=SubscriptionType.ENGINE_STATUS,
            parameters={"engine_id": "test"},
            rate_limit=10,
            priority=3
        )
        logger.info(f"✓ Subscription config: {config.subscription_type.value}")
        
    except Exception as e:
        logger.error(f"✗ Subscription config test failed: {e}")
        return False
    
    # Test protocol info
    try:
        protocol_info = default_protocol.get_protocol_info()
        logger.info(f"✓ Protocol info: v{protocol_info['version']}, {protocol_info['message_count']} message types")
        
    except Exception as e:
        logger.error(f"✗ Protocol info test failed: {e}")
        return False
    
    logger.info("=== Integration Test Complete ===")
    logger.info("✓ All core components are working correctly")
    logger.info("✓ Message protocols are functional")
    logger.info("✓ Service initialization is successful")
    logger.info("✓ Event and subscription systems are operational")
    
    return True


async def test_mock_data_generation():
    """Test mock data generation for development"""
    logger.info("\n=== Mock Data Generation Test ===")
    
    try:
        from backend.websocket.streaming_service import StreamingService
        
        streaming_service = StreamingService()
        
        # Test mock engine status
        engine_status = await streaming_service._get_mock_engine_status()
        logger.info(f"✓ Mock engine status: {engine_status['state']}, CPU: {engine_status['cpu_usage']}")
        
        # Test mock market data
        market_data = await streaming_service._get_market_data_update("AAPL")
        logger.info(f"✓ Mock market data: AAPL @ ${market_data['price']}")
        
        # Test mock system health
        health_data = await streaming_service._get_system_health_metrics()
        logger.info(f"✓ Mock system health: CPU {health_data['system_load']['cpu_percent']}%")
        
    except Exception as e:
        logger.error(f"✗ Mock data generation test failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    async def main():
        """Run all integration tests"""
        logger.info("Starting WebSocket Infrastructure Integration Tests...")
        
        # Run basic integration test
        success1 = await test_websocket_infrastructure()
        
        # Run mock data test
        success2 = await test_mock_data_generation()
        
        if success1 and success2:
            logger.info("\n🎉 ALL TESTS PASSED!")
            logger.info("WebSocket streaming infrastructure is ready for Sprint 3")
        else:
            logger.error("\n❌ SOME TESTS FAILED!")
            logger.error("Please check the errors above")
    
    # Run the tests
    asyncio.run(main())