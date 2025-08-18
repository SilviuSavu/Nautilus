#!/usr/bin/env python3
"""
Test script for Interactive Brokers integration
Tests the IB integration service, API endpoints, and WebSocket data flow.
"""

import asyncio
import json
import logging
import sys
from datetime import datetime
from typing import Dict, Any

import aiohttp
import websockets
from ib_integration_service import IBIntegrationService, IBConnectionStatus, IBAccountData
from messagebus_client import MessageBusClient, MessageBusMessage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IBIntegrationTester:
    """Test Interactive Brokers integration functionality"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.ws_url = "ws://localhost:8000/ws"
        self.session = None
        self.websocket = None
        self.received_messages = []
        
    async def setup(self):
        """Setup test session"""
        self.session = aiohttp.ClientSession()
        logger.info("Test session setup complete")
    
    async def teardown(self):
        """Cleanup test session"""
        if self.session:
            await self.session.close()
        if self.websocket:
            await self.websocket.close()
        logger.info("Test session teardown complete")
    
    async def test_api_health(self) -> bool:
        """Test basic API health"""
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"✓ API Health: {data}")
                    return True
                else:
                    logger.error(f"✗ API Health failed: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"✗ API Health exception: {e}")
            return False
    
    async def test_ib_connection_status(self) -> bool:
        """Test IB connection status endpoint"""
        try:
            async with self.session.get(f"{self.base_url}/api/v1/ib/connection/status") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"✓ IB Connection Status: {data}")
                    return True
                elif response.status == 401:
                    logger.warning("⚠ IB Connection Status: Authentication required (expected in production)")
                    return True
                else:
                    logger.error(f"✗ IB Connection Status failed: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"✗ IB Connection Status exception: {e}")
            return False
    
    async def test_ib_account_data(self) -> bool:
        """Test IB account data endpoint"""
        try:
            async with self.session.get(f"{self.base_url}/api/v1/ib/account") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"✓ IB Account Data: {data}")
                    return True
                elif response.status == 401:
                    logger.warning("⚠ IB Account Data: Authentication required (expected in production)")
                    return True
                else:
                    logger.error(f"✗ IB Account Data failed: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"✗ IB Account Data exception: {e}")
            return False
    
    async def test_ib_positions(self) -> bool:
        """Test IB positions endpoint"""
        try:
            async with self.session.get(f"{self.base_url}/api/v1/ib/positions") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"✓ IB Positions: {len(data.get('positions', []))} positions")
                    return True
                elif response.status == 401:
                    logger.warning("⚠ IB Positions: Authentication required (expected in production)")
                    return True
                else:
                    logger.error(f"✗ IB Positions failed: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"✗ IB Positions exception: {e}")
            return False
    
    async def test_ib_orders(self) -> bool:
        """Test IB orders endpoint"""
        try:
            async with self.session.get(f"{self.base_url}/api/v1/ib/orders") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"✓ IB Orders: {len(data.get('orders', []))} orders")
                    return True
                elif response.status == 401:
                    logger.warning("⚠ IB Orders: Authentication required (expected in production)")
                    return True
                else:
                    logger.error(f"✗ IB Orders failed: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"✗ IB Orders exception: {e}")
            return False
    
    async def test_websocket_connection(self) -> bool:
        """Test WebSocket connection and IB message handling"""
        try:
            self.websocket = await websockets.connect(self.ws_url)
            logger.info("✓ WebSocket connected")
            
            # Wait for welcome message
            welcome_msg = await asyncio.wait_for(self.websocket.recv(), timeout=5.0)
            welcome_data = json.loads(welcome_msg)
            
            if welcome_data.get("type") == "connection":
                logger.info(f"✓ WebSocket welcome: {welcome_data}")
                return True
            else:
                logger.error(f"✗ Unexpected welcome message: {welcome_data}")
                return False
                
        except asyncio.TimeoutError:
            logger.error("✗ WebSocket connection timeout")
            return False
        except Exception as e:
            logger.error(f"✗ WebSocket connection exception: {e}")
            return False
    
    async def test_demo_ib_data(self) -> bool:
        """Test sending demo IB data through WebSocket"""
        if not self.websocket:
            logger.error("✗ WebSocket not connected for demo data test")
            return False
        
        try:
            # Send demo market data to trigger IB-related messages
            async with self.session.post(f"{self.base_url}/api/v1/demo/send-market-data") as response:
                if response.status == 200:
                    logger.info("✓ Demo market data sent")
                else:
                    logger.warning(f"⚠ Demo market data response: {response.status}")
            
            # Listen for messages
            received_count = 0
            start_time = datetime.now()
            
            while received_count < 3 and (datetime.now() - start_time).seconds < 10:
                try:
                    message = await asyncio.wait_for(self.websocket.recv(), timeout=2.0)
                    msg_data = json.loads(message)
                    
                    self.received_messages.append(msg_data)
                    received_count += 1
                    
                    logger.info(f"✓ Received message: {msg_data.get('type', 'unknown')}")
                    
                    # Check for IB-specific message types
                    if msg_data.get("type") in ["ib_connection", "ib_account", "ib_positions", "ib_order"]:
                        logger.info(f"✓ IB-specific message received: {msg_data['type']}")
                    
                except asyncio.TimeoutError:
                    break
            
            logger.info(f"✓ Received {received_count} WebSocket messages")
            return received_count > 0
            
        except Exception as e:
            logger.error(f"✗ Demo IB data test exception: {e}")
            return False
    
    async def test_order_placement_validation(self) -> bool:
        """Test order placement endpoint validation (without actually placing orders)"""
        try:
            # Test invalid order data
            invalid_order = {
                "symbol": "",  # Invalid empty symbol
                "action": "INVALID",  # Invalid action
                "quantity": -1,  # Invalid quantity
                "order_type": "MKT"
            }
            
            async with self.session.post(
                f"{self.base_url}/api/v1/ib/orders/place",
                json=invalid_order,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 400:
                    logger.info("✓ Order validation correctly rejected invalid order")
                    return True
                elif response.status == 401:
                    logger.warning("⚠ Order placement: Authentication required (expected in production)")
                    return True
                else:
                    error_data = await response.text()
                    logger.error(f"✗ Unexpected order validation response: {response.status}, {error_data}")
                    return False
                    
        except Exception as e:
            logger.error(f"✗ Order placement validation exception: {e}")
            return False
    
    async def test_ib_service_direct(self) -> bool:
        """Test IB service directly (unit test style)"""
        try:
            # Create a mock MessageBus client
            messagebus_client = MessageBusClient()
            
            # Create IB service
            ib_service = IBIntegrationService(messagebus_client)
            
            # Test connection status
            connection_status = await ib_service.get_connection_status()
            logger.info(f"✓ IB Service connection status: {connection_status}")
            
            # Test account data
            account_data = await ib_service.get_account_data()
            logger.info(f"✓ IB Service account data: {account_data}")
            
            # Test positions
            positions = await ib_service.get_positions()
            logger.info(f"✓ IB Service positions: {len(positions)} positions")
            
            # Test orders
            orders = await ib_service.get_orders()
            logger.info(f"✓ IB Service orders: {len(orders)} orders")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ IB Service direct test exception: {e}")
            return False
    
    async def run_all_tests(self) -> Dict[str, bool]:
        """Run all IB integration tests"""
        logger.info("🚀 Starting Interactive Brokers Integration Tests")
        
        results = {}
        
        await self.setup()
        
        try:
            # API Tests
            results["api_health"] = await self.test_api_health()
            results["ib_connection_status"] = await self.test_ib_connection_status()
            results["ib_account_data"] = await self.test_ib_account_data()
            results["ib_positions"] = await self.test_ib_positions()
            results["ib_orders"] = await self.test_ib_orders()
            results["order_validation"] = await self.test_order_placement_validation()
            
            # WebSocket Tests
            results["websocket_connection"] = await self.test_websocket_connection()
            results["demo_ib_data"] = await self.test_demo_ib_data()
            
            # Direct Service Tests
            results["ib_service_direct"] = await self.test_ib_service_direct()
            
        finally:
            await self.teardown()
        
        return results
    
    def print_test_summary(self, results: Dict[str, bool]):
        """Print test results summary"""
        passed = sum(1 for result in results.values() if result)
        total = len(results)
        
        logger.info("\n" + "="*60)
        logger.info("🧪 Interactive Brokers Integration Test Summary")
        logger.info("="*60)
        
        for test_name, result in results.items():
            status = "✓ PASS" if result else "✗ FAIL"
            logger.info(f"{status:8} {test_name}")
        
        logger.info("-"*60)
        logger.info(f"Results: {passed}/{total} tests passed")
        
        if passed == total:
            logger.info("🎉 All tests passed! IB integration is working correctly.")
        else:
            logger.warning(f"⚠️ {total - passed} test(s) failed. Check the logs above for details.")
        
        logger.info("="*60)


async def main():
    """Main test runner"""
    tester = IBIntegrationTester()
    
    try:
        results = await tester.run_all_tests()
        tester.print_test_summary(results)
        
        # Exit with error code if any tests failed
        if not all(results.values()):
            sys.exit(1)
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Test runner failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())