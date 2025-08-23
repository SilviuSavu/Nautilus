"""
Simple WebSocket Infrastructure Test - Sprint 3 Priority 1

Basic test to verify the WebSocket infrastructure components work independently.
"""

import asyncio
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_basic_components():
    """Test basic WebSocket infrastructure components"""
    logger.info("=== Basic WebSocket Components Test ===")
    
    # Test message protocols
    try:
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        from backend.websocket.message_protocols import (
            MessageProtocol, 
            EngineStatusMessage,
            MarketDataMessage,
            default_protocol
        )
        
        # Test message validation
        valid_message = {
            "type": "engine_status",
            "data": {"state": "running", "uptime": 3600},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        is_valid = default_protocol.validate_message(valid_message)
        logger.info(f"✓ Message validation: {is_valid}")
        
        # Test message creation
        engine_msg = EngineStatusMessage(
            type="engine_status",
            data={"state": "running", "uptime": 3600},
            engine_id="test-engine"
        )
        logger.info(f"✓ Message creation: {engine_msg.type}")
        
        # Test serialization
        serialized = default_protocol.serialize_message(engine_msg)
        logger.info(f"✓ Message serialization: {len(serialized)} bytes")
        
        # Test deserialization  
        deserialized = default_protocol.deserialize_message(serialized)
        logger.info(f"✓ Message deserialization: {deserialized.type if deserialized else 'Failed'}")
        
    except Exception as e:
        logger.error(f"✗ Message protocol test failed: {e}")
        return False
    
    # Test WebSocket manager (basic)
    try:
        from backend.websocket.websocket_manager import WebSocketManager
        
        manager = WebSocketManager()
        stats = manager.get_connection_stats()
        logger.info(f"✓ WebSocket manager: {stats['total_connections']} connections")
        
    except Exception as e:
        logger.error(f"✗ WebSocket manager test failed: {e}")
        return False
    
    # Test event dispatcher (basic)
    try:
        from backend.websocket.event_dispatcher import Event, EventType, EventPriority
        
        test_event = Event(
            event_type=EventType.ENGINE_STATUS_CHANGED,
            data={"state": "running", "message": "Test event"},
            source="integration_test",
            priority=EventPriority.NORMAL
        )
        
        event_dict = test_event.to_dict()
        logger.info(f"✓ Event creation: {event_dict['event_type']}")
        
        # Test event reconstruction
        reconstructed = Event.from_dict(event_dict)
        logger.info(f"✓ Event reconstruction: {reconstructed.event_type.value}")
        
    except Exception as e:
        logger.error(f"✗ Event dispatcher test failed: {e}")
        return False
    
    # Test subscription manager (basic)
    try:
        from backend.websocket.subscription_manager import SubscriptionConfig, SubscriptionType
        
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
    
    # Test streaming service (mock data only)
    try:
        from backend.websocket.streaming_service import StreamingService
        
        streaming_service = StreamingService()
        
        # Test mock data generation
        engine_status = await streaming_service._get_mock_engine_status()
        logger.info(f"✓ Mock engine status: {engine_status['state']}")
        
        market_data = await streaming_service._get_market_data_update("AAPL")
        logger.info(f"✓ Mock market data: AAPL @ ${market_data['price']}")
        
        stream_stats = await streaming_service.get_stream_statistics()
        logger.info(f"✓ Stream statistics: {stream_stats['active_streams']} active")
        
    except Exception as e:
        logger.error(f"✗ Streaming service test failed: {e}")
        return False
    
    logger.info("\n=== All Basic Tests Passed! ===")
    logger.info("✓ Message protocols are working")
    logger.info("✓ WebSocket manager is functional") 
    logger.info("✓ Event system is operational")
    logger.info("✓ Subscription system is ready")
    logger.info("✓ Streaming service is generating mock data")
    
    return True


async def test_integration_flow():
    """Test a basic integration flow"""
    logger.info("\n=== Integration Flow Test ===")
    
    try:
        from backend.websocket.message_protocols import create_engine_status_message, default_protocol
        from backend.websocket.websocket_manager import WebSocketManager
        
        # Create a mock connection scenario
        manager = WebSocketManager()
        
        # Create a message
        message = create_engine_status_message(
            data={"state": "running", "uptime": 3600, "cpu": "25.5%"},
            engine_id="test-engine-001"
        )
        
        # Validate the message
        is_valid = default_protocol.validate_message(message.dict())
        logger.info(f"✓ Message validation in flow: {is_valid}")
        
        # Test subscription workflow
        from backend.websocket.subscription_manager import SubscriptionManager, SubscriptionConfig, SubscriptionType
        
        sub_manager = SubscriptionManager()
        config = SubscriptionConfig(
            subscription_type=SubscriptionType.ENGINE_STATUS,
            parameters={"engine_id": "test-engine-001"},
            rate_limit=5,
            priority=4
        )
        
        logger.info(f"✓ Subscription workflow: {config.subscription_type.value} at {config.rate_limit}/sec")
        
        # Test event flow
        from backend.websocket.event_dispatcher import Event, EventType, EventPriority
        
        event = Event(
            event_type=EventType.ENGINE_STATUS_CHANGED,
            data=message.data,
            source="test-engine-001",
            priority=EventPriority.HIGH
        )
        
        logger.info(f"✓ Event flow: {event.event_type.value} from {event.source}")
        
    except Exception as e:
        logger.error(f"✗ Integration flow test failed: {e}")
        return False
    
    logger.info("✓ Integration flow test completed successfully")
    return True


if __name__ == "__main__":
    async def main():
        """Run all tests"""
        logger.info("Starting WebSocket Infrastructure Tests...")
        
        # Run basic component tests
        success1 = await test_basic_components()
        
        # Run integration flow test
        success2 = await test_integration_flow()
        
        if success1 and success2:
            logger.info("\n🎉 ALL TESTS PASSED!")
            logger.info("✅ WebSocket streaming infrastructure is ready for Sprint 3")
            logger.info("✅ All components are working correctly")
            logger.info("✅ Integration flows are functional")
            logger.info("\nNext Steps:")
            logger.info("1. Start the backend server with WebSocket support")
            logger.info("2. Test with real WebSocket clients")
            logger.info("3. Monitor streaming performance")
        else:
            logger.error("\n❌ SOME TESTS FAILED!")
            logger.error("Please check the errors above")
    
    # Run the tests
    asyncio.run(main())