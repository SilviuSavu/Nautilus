"""
IB Integration Service - Nautilus Compatibility Layer
Provides compatibility for existing service dependencies.
"""

from typing import Callable, Optional


class IBIntegrationService:
    """IB Integration service compatibility stub."""
    
    def __init__(self, messagebus_client):
        self.messagebus_client = messagebus_client
        
    def add_connection_handler(self, handler: Callable):
        """Add connection status handler."""
        pass
        
    def add_account_handler(self, handler: Callable):
        """Add account data handler."""
        pass
        
    def add_position_handler(self, handler: Callable):
        """Add position data handler."""
        pass
        
    def add_order_handler(self, handler: Callable):
        """Add order update handler."""
        pass


def get_ib_integration_service(messagebus_client) -> IBIntegrationService:
    """Get IB integration service instance."""
    return IBIntegrationService(messagebus_client)