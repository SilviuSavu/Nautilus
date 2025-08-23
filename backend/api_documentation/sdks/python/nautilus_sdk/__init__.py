"""
Nautilus Trading Platform Python SDK
Official Python client library for the Nautilus Trading Platform API
"""

from .client import NautilusClient
from .auth import AuthenticatedSession
from .exceptions import NautilusException, AuthenticationError, RateLimitError
from .models import *
from .websocket import WebSocketClient

__version__ = "3.0.0"
__author__ = "Nautilus Trading Platform"
__email__ = "api-support@nautilus-trading.com"

__all__ = [
    "NautilusClient",
    "AuthenticatedSession", 
    "WebSocketClient",
    "NautilusException",
    "AuthenticationError",
    "RateLimitError"
]