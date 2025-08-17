#!/usr/bin/env python3
"""
WebSocket client test for Nautilus Trader MessageBus integration
"""

import asyncio
import websockets
import json

async def test_websocket():
    """Test WebSocket connection and messaging"""
    uri = "ws://localhost:8001/ws"
    
    print("Connecting to WebSocket...")
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ WebSocket connected successfully")
            
            # Wait for welcome message
            welcome = await websocket.recv()
            print(f"📨 Welcome message: {welcome}")
            
            # Send test message
            test_message = "Hello from test client!"
            await websocket.send(test_message)
            print(f"📤 Sent: {test_message}")
            
            # Receive echo
            echo = await websocket.recv()
            print(f"📥 Received: {echo}")
            
            # Keep connection open for a bit to receive any MessageBus messages
            print("🔄 Listening for MessageBus messages for 5 seconds...")
            try:
                while True:
                    message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    print(f"📨 MessageBus event: {message}")
            except asyncio.TimeoutError:
                print("⏰ No additional messages received (timeout)")
            
            print("✅ WebSocket test completed successfully")
            
    except Exception as e:
        print(f"❌ WebSocket test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    asyncio.run(test_websocket())