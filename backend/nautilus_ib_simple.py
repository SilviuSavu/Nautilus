"""
Simple Nautilus IB Wrapper using working direct ibapi
Maintains persistent connection and provides Nautilus-compatible interface.
"""

import asyncio
import logging
import os
import threading
import time
from typing import Optional, Dict, Any
from datetime import datetime

from ibapi.client import EClient
from ibapi.wrapper import EWrapper

logger = logging.getLogger(__name__)


class NautilusIBWrapper(EWrapper):
    """Wrapper for IB API callbacks"""
    
    def __init__(self):
        EWrapper.__init__(self)
        self.connected = False
        self.account_id = None
        self.connection_time = None
        
    def connectAck(self):
        """Called when connection is acknowledged"""
        self.connected = True
        self.connection_time = datetime.utcnow()
        logger.info("✅ Connected to IB Gateway")
        
    def managedAccounts(self, accountsList):
        """Called when managed accounts are received"""
        self.account_id = accountsList.split(',')[0] if accountsList else None
        logger.info(f"✅ Account: {self.account_id}")
        
    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        """Handle errors"""
        if errorCode in [2104, 2106, 2158]:  # Informational
            logger.info(f"ℹ️  {errorString}")
        elif errorCode == 326:  # Client ID in use
            logger.warning(f"⚠️  Client ID conflict: {errorString}")
        else:
            logger.error(f"❌ Error {errorCode}: {errorString}")


class NautilusIBClient(EClient):
    """Client for IB API commands"""
    
    def __init__(self, wrapper):
        EClient.__init__(self, wrapper)


class SimpleNautilusIB:
    """Simple Nautilus IB integration using direct ibapi"""
    
    def __init__(self):
        self.host = os.environ.get('IB_HOST', 'host.docker.internal')
        self.port = int(os.environ.get('IB_PORT', '4002'))
        self.client_id = int(os.environ.get('IB_CLIENT_ID', '1001'))
        
        self.wrapper = NautilusIBWrapper()
        self.client = NautilusIBClient(self.wrapper)
        self.client_thread = None
        self._running = False
        
    async def connect(self) -> bool:
        """Connect to IB Gateway"""
        try:
            logger.info(f"Connecting to {self.host}:{self.port} with client_id={self.client_id}")
            
            # Start client thread
            def run_client():
                self.client.connect(self.host, self.port, self.client_id)
                self.client.run()
            
            self.client_thread = threading.Thread(target=run_client, daemon=True)
            self.client_thread.start()
            self._running = True
            
            # Wait for connection
            for _ in range(10):  # 5 seconds timeout
                await asyncio.sleep(0.5)
                if self.wrapper.connected:
                    logger.info("✅ Successfully connected to IB Gateway")
                    return True
            
            logger.error("❌ Connection timeout")
            return False
            
        except Exception as e:
            logger.error(f"❌ Connection error: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from IB Gateway"""
        try:
            if self._running:
                self.client.disconnect()
                self._running = False
                
            # Reset wrapper state
            self.wrapper.connected = False
            self.wrapper.connection_time = None
            
            logger.info("✅ Disconnected from IB Gateway")
            return True
            
        except Exception as e:
            logger.error(f"❌ Disconnect error: {e}")
            return False
    
    def is_connected(self) -> bool:
        """Check if connected"""
        return self.wrapper.connected and self._running
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get connection information"""
        return {
            "connected": self.is_connected(),
            "gateway_type": "IB Gateway (Simple Direct)",
            "account_id": self.wrapper.account_id,
            "connection_time": self.wrapper.connection_time.isoformat() if self.wrapper.connection_time else None,
            "host": self.host,
            "port": self.port,
            "client_id": self.client_id,
            "data_client_connected": self.is_connected(),
            "exec_client_connected": self.is_connected(),
        }


# Global instance
_simple_nautilus_ib: Optional[SimpleNautilusIB] = None


def get_simple_nautilus_ib() -> SimpleNautilusIB:
    """Get or create the global simple Nautilus IB instance"""
    global _simple_nautilus_ib
    
    if _simple_nautilus_ib is None:
        _simple_nautilus_ib = SimpleNautilusIB()
    
    return _simple_nautilus_ib


def reset_simple_nautilus_ib():
    """Reset the global simple Nautilus IB instance"""
    global _simple_nautilus_ib
    if _simple_nautilus_ib and _simple_nautilus_ib._running:
        _simple_nautilus_ib.client.disconnect()
    _simple_nautilus_ib = None