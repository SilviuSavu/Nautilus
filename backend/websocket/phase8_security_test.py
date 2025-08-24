"""
Phase 8 Security WebSocket Integration Test
Demonstrates real-time security event streaming functionality
"""

import asyncio
import websockets
import json
import logging
from datetime import datetime
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Phase8SecurityWebSocketClient:
    """Test client for Phase 8 Security WebSocket streaming"""
    
    def __init__(self, websocket_url: str, auth_token: str = None):
        self.websocket_url = websocket_url
        self.auth_token = auth_token
        self.websocket = None
        self.connected = False
        self.event_handlers = {}
        
    async def connect(self):
        """Connect to Phase 8 Security WebSocket"""
        try:
            # Build URL with auth token if provided
            url = self.websocket_url
            if self.auth_token:
                url += f"?token={self.auth_token}"
            
            logger.info(f"Connecting to Phase 8 Security WebSocket: {url}")
            self.websocket = await websockets.connect(url)
            self.connected = True
            
            # Start message listener
            asyncio.create_task(self._listen_for_messages())
            
            logger.info("✅ Connected to Phase 8 Security WebSocket")
            return True
            
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from WebSocket"""
        if self.websocket and self.connected:
            await self.websocket.close()
            self.connected = False
            logger.info("Disconnected from Phase 8 Security WebSocket")
    
    async def send_message(self, message: Dict[str, Any]):
        """Send message to WebSocket"""
        if not self.connected:
            logger.error("Not connected to WebSocket")
            return
            
        try:
            await self.websocket.send(json.dumps(message))
            logger.debug(f"Sent message: {message}")
            
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
    
    async def _listen_for_messages(self):
        """Listen for incoming messages"""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    await self._handle_message(data)
                    
                except json.JSONDecodeError:
                    logger.error(f"Received invalid JSON: {message}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed")
            self.connected = False
        except Exception as e:
            logger.error(f"Message listening error: {e}")
            self.connected = False
    
    async def _handle_message(self, data: Dict[str, Any]):
        """Handle incoming WebSocket message"""
        message_type = data.get("type")
        
        if message_type in self.event_handlers:
            await self.event_handlers[message_type](data)
        else:
            await self._default_message_handler(data)
    
    async def _default_message_handler(self, data: Dict[str, Any]):
        """Default message handler"""
        message_type = data.get("type")
        timestamp = data.get("timestamp", datetime.now().isoformat())
        
        if message_type == "connection_established":
            logger.info("🔗 Connection established")
            logger.info(f"Security Mode: {data.get('security_mode')}")
            logger.info(f"Available Event Types: {data.get('available_event_types')}")
            logger.info(f"Security Components: {data.get('security_components')}")
            
        elif message_type == "security_event":
            event_data = data.get("data", {})
            event_type = event_data.get("event_type")
            
            logger.info(f"🚨 Security Event: {event_type}")
            logger.info(f"Timestamp: {timestamp}")
            
            if event_type == "security.analysis.complete":
                await self._handle_security_analysis_event(event_data)
            elif event_type == "security.fraud.alert":
                await self._handle_fraud_alert_event(event_data)
            elif event_type == "security.threat.detected":
                await self._handle_threat_detection_event(event_data)
            elif event_type == "security.response.executed":
                await self._handle_security_response_event(event_data)
            elif event_type == "security.orchestration.update":
                await self._handle_orchestration_event(event_data)
                
        elif message_type == "pong":
            logger.debug("Received pong")
            
        elif message_type == "error":
            logger.error(f"WebSocket Error: {data.get('message')}")
            
        else:
            logger.info(f"Received message type '{message_type}': {data}")
    
    async def _handle_security_analysis_event(self, event_data: Dict[str, Any]):
        """Handle CSOC security analysis event"""
        logger.info("🔍 Security Analysis Complete")
        
        analysis_results = event_data.get("analysis_results", {})
        cognitive_insights = event_data.get("cognitive_insights", {})
        
        logger.info(f"  Threat Detected: {analysis_results.get('threat_detected')}")
        logger.info(f"  Confidence Score: {analysis_results.get('confidence_score')}")
        logger.info(f"  Risk Level: {analysis_results.get('risk_level')}")
        logger.info(f"  Affected Systems: {analysis_results.get('affected_systems')}")
        logger.info(f"  ML Prediction: {cognitive_insights.get('ml_prediction')}")
    
    async def _handle_fraud_alert_event(self, event_data: Dict[str, Any]):
        """Handle fraud detection alert"""
        logger.info("💰 Fraud Alert Generated")
        
        fraud_details = event_data.get("fraud_details", {})
        
        logger.info(f"  Fraud Type: {fraud_details.get('fraud_type')}")
        logger.info(f"  Severity: {fraud_details.get('severity')}")
        logger.info(f"  Risk Score: {fraud_details.get('risk_score')}")
        logger.info(f"  Financial Impact: {fraud_details.get('financial_impact')}")
        logger.info(f"  Investigation Priority: {fraud_details.get('investigation_priority')}")
    
    async def _handle_threat_detection_event(self, event_data: Dict[str, Any]):
        """Handle threat intelligence detection"""
        logger.info("🎯 Threat Detected")
        
        threat_intelligence = event_data.get("threat_intelligence", {})
        context = event_data.get("context", {})
        
        logger.info(f"  Indicator Type: {threat_intelligence.get('indicator_type')}")
        logger.info(f"  Indicator Value: {threat_intelligence.get('indicator_value')}")
        logger.info(f"  Threat Type: {threat_intelligence.get('threat_type')}")
        logger.info(f"  Confidence: {threat_intelligence.get('confidence')}")
        logger.info(f"  Attack Techniques: {context.get('attack_techniques')}")
    
    async def _handle_security_response_event(self, event_data: Dict[str, Any]):
        """Handle autonomous security response"""
        logger.info("⚡ Security Response Executed")
        
        response_details = event_data.get("response_details", {})
        
        logger.info(f"  Action Type: {response_details.get('action_type')}")
        logger.info(f"  Target Entity: {response_details.get('target_entity')}")
        logger.info(f"  Execution Status: {response_details.get('execution_status')}")
        logger.info(f"  Response Time: {response_details.get('response_time_ms')}ms")
        logger.info(f"  Effectiveness: {response_details.get('effectiveness')}")
    
    async def _handle_orchestration_event(self, event_data: Dict[str, Any]):
        """Handle security orchestration update"""
        logger.info("🎛️ Security Orchestration Update")
        
        orchestration_details = event_data.get("orchestration_details", {})
        
        logger.info(f"  Workflow: {orchestration_details.get('workflow_name')}")
        logger.info(f"  Status: {orchestration_details.get('execution_status')}")
        logger.info(f"  Current Step: {orchestration_details.get('current_step')}")
        logger.info(f"  Progress: {orchestration_details.get('steps_completed')}/{orchestration_details.get('total_steps')}")
    
    async def subscribe_to_events(self, event_types: list = None):
        """Subscribe to specific security event types"""
        message = {
            "action": "subscribe",
            "event_types": event_types or [
                "security.analysis.complete",
                "security.threat.detected",
                "security.fraud.alert",
                "security.response.executed",
                "security.orchestration.update"
            ]
        }
        await self.send_message(message)
    
    async def get_security_status(self):
        """Request current security status"""
        message = {"action": "get_security_status"}
        await self.send_message(message)
    
    async def get_event_schema(self):
        """Request event schema"""
        message = {"action": "get_event_schema"}
        await self.send_message(message)
    
    async def send_heartbeat(self):
        """Send heartbeat ping"""
        message = {"action": "ping"}
        await self.send_message(message)


async def run_phase8_security_test():
    """Run Phase 8 Security WebSocket integration test"""
    logger.info("🚀 Starting Phase 8 Security WebSocket Integration Test")
    
    # WebSocket URL (adjust port/host as needed)
    websocket_url = "ws://localhost:8001/ws/phase8/security"
    
    # Create test client
    client = Phase8SecurityWebSocketClient(websocket_url, auth_token="test_token_12345")
    
    try:
        # Connect to WebSocket
        if not await client.connect():
            logger.error("Failed to connect to WebSocket")
            return
        
        # Wait for connection establishment
        await asyncio.sleep(2)
        
        # Subscribe to all security events
        logger.info("📡 Subscribing to security events...")
        await client.subscribe_to_events()
        
        # Request security status
        logger.info("📊 Requesting security status...")
        await client.get_security_status()
        
        # Request event schema
        logger.info("📋 Requesting event schema...")
        await client.get_event_schema()
        
        # Listen for events for 30 seconds
        logger.info("👂 Listening for security events for 30 seconds...")
        
        # Send periodic heartbeats
        for i in range(6):  # 30 seconds / 5 second intervals
            await asyncio.sleep(5)
            await client.send_heartbeat()
            logger.info(f"💓 Sent heartbeat {i+1}/6")
        
        logger.info("✅ Test completed successfully")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        
    finally:
        # Cleanup
        await client.disconnect()


if __name__ == "__main__":
    # Run the test
    asyncio.run(run_phase8_security_test())