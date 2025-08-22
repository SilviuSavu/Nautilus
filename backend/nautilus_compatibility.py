"""
Nautilus Trader Compatibility Layer
==================================

Provides compatibility layer for components that may not have full nautilus_trader access.
This allows the system to gracefully handle missing dependencies.
"""

from typing import Any, Optional
from datetime import datetime
from uuid import uuid4

# Try to import nautilus_trader components, fall back to compatible alternatives
try:
    from nautilus_trader.model.data import CustomData
    from nautilus_trader.core.uuid import UUID4
    NAUTILUS_AVAILABLE = True
except ImportError:
    NAUTILUS_AVAILABLE = False
    
    # Fallback CustomData implementation
    class CustomData:
        """Fallback CustomData for when nautilus_trader is not available."""
        def __init__(self, data: dict):
            self.data = data
            self.ts_event = int(datetime.now().timestamp() * 1_000_000_000)
            self.ts_init = self.ts_event
    
    # Fallback UUID4 implementation  
    def UUID4():
        """Fallback UUID4 generator."""
        return str(uuid4())


def create_custom_data(data: dict) -> CustomData:
    """Create CustomData with proper fallback."""
    return CustomData(data)


def generate_uuid() -> str:
    """Generate UUID with proper fallback."""
    if NAUTILUS_AVAILABLE:
        return str(UUID4())
    else:
        return str(uuid4())


def is_nautilus_available() -> bool:
    """Check if nautilus_trader is available."""
    return NAUTILUS_AVAILABLE