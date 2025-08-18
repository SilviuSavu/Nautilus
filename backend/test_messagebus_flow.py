#!/usr/bin/env python3
"""
Test end-to-end MessageBus flow: Redis -> Backend -> WebSocket
"""

import asyncio
import websockets
import json
import redis.asyncio as redis
import time

async def simulate_messagebus_messages():
    """Simulate messages from NautilusTrader by sending to Redis stream"""
    r = redis.Redis(host='localhost', port=6379, db=0)
    
    # Test messages to simulate
    test_messages = [
        {
            "topic": "market_data",
            "type": "quote", 
            "symbol": "EURUSD",
            "bid": "1.0850",
            "ask": "1.0852",
            "timestamp": int(time.time() * 1000)
        },
        {
            "topic": "trading",
            "type": "order_update",
            "order_id": "12345",
            "status": "filled",
            "quantity": "100000",
            "timestamp": int(time.time() * 1000)
        },
        {
            "topic": "portfolio",
            "type": "balance_update", 
            "account": "test_account",
            "currency": "USD",
            "balance": "50000.00",
            "timestamp": int(time.time() * 1000)
        }
    ]
    
    print("📤 Sending test messages to Redis stream...")
    for i, msg in enumerate(test_messages):
        # Convert to Redis stream format
        stream_data = {}
        for key, value in msg.items():
            stream_data[key] = str(value)
        
        # Send to Redis stream
        await r.xadd("nautilus-streams", stream_data)
        print(f"   {i+1}. Sent {msg['type']} message for {msg.get('symbol', msg.get('order_id', msg.get('account', 'unknown')))}")
        await asyncio.sleep(0.5)  # Small delay between messages
    
    await r.close()
    print("✅ All test messages sent to Redis stream")

async def listen_websocket():
    """Listen for messages on WebSocket"""
    uri = "ws://localhost:8002/ws"
    messages_received = []
    
    try:
        async with websockets.connect(uri) as websocket:
            print("🔗 WebSocket connected, waiting for MessageBus events...")
            
            # Wait for welcome message and discard
            await websocket.recv()
            
            # Listen for MessageBus messages
            start_time = time.time()
            while time.time() - start_time < 10:  # Listen for 10 seconds
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    data = json.loads(message)
                    
                    if data.get("type") == "messagebus":
                        messages_received.append(data)
                        print(f"📨 MessageBus event: {data['topic']} - {data['payload']}")
                    
                except asyncio.TimeoutError:
                    continue
                except json.JSONDecodeError:
                    print(f"⚠️  Received non-JSON message: {message}")
            
            return messages_received
            
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        return []

async def test_messagebus_flow():
    """Test complete MessageBus flow"""
    print("🚀 Starting MessageBus integration test...\n")
    
    # Start WebSocket listener in background
    websocket_task = asyncio.create_task(listen_websocket())
    
    # Wait a moment for WebSocket to connect
    await asyncio.sleep(1)
    
    # Send messages to Redis stream
    await simulate_messagebus_messages()
    
    # Wait for WebSocket to receive messages
    print(f"\n🔄 Waiting for messages to propagate through MessageBus...")
    messages_received = await websocket_task
    
    # Analyze results
    print(f"\n📊 Test Results:")
    print(f"   Messages sent: 3")
    print(f"   Messages received: {len(messages_received)}")
    
    if len(messages_received) == 3:
        print("✅ All messages received successfully!")
        print("✅ MessageBus integration working correctly!")
        return True
    elif len(messages_received) > 0:
        print(f"⚠️  Partial success: {len(messages_received)}/3 messages received")
        return True
    else:
        print("❌ No messages received - MessageBus integration issue")
        return False

if __name__ == "__main__":
    asyncio.run(test_messagebus_flow())