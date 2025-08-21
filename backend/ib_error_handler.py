"""
Interactive Brokers Error Handling and Connection Management
Comprehensive error handling, reconnection logic, and connection state management.
"""

import asyncio
import logging
import time
from typing import Any, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


class IBErrorSeverity(Enum):
    """IB Error Severity Levels"""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class IBConnectionState(Enum):
    """IB Connection States"""
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    RECONNECTING = "RECONNECTING"
    FAILED = "FAILED"
    SUSPENDED = "SUSPENDED"


@dataclass
class IBError:
    """IB Error details"""
    error_code: int
    error_message: str
    req_id: int
    severity: IBErrorSeverity
    timestamp: datetime = field(default_factory=datetime.now)
    context: str | None = None
    recoverable: bool = True


@dataclass
class IBConnectionEvent:
    """Connection event details"""
    event_type: str
    state: IBConnectionState
    timestamp: datetime = field(default_factory=datetime.now)
    message: str | None = None
    error: IBError | None = None


class IBErrorHandler:
    """
    Interactive Brokers Error Handler
    
    Provides comprehensive error classification, handling strategies, and automated recovery mechanisms.
    """
    
    def __init__(self, ib_client):
        self.logger = logging.getLogger(__name__)
        self.ib_client = ib_client
        
        # Error tracking
        self.errors: list[IBError] = []
        self.error_counts: dict[int, int] = {}  # error_code -> count
        self.connection_events: list[IBConnectionEvent] = []
        
        # Connection state
        self.connection_state = IBConnectionState.DISCONNECTED
        self.last_connection_time: datetime | None = None
        self.connection_attempts = 0
        self.max_connection_attempts = 5
        
        # Reconnection settings
        self.auto_reconnect = True
        self.reconnect_delay = 5  # seconds
        self.max_reconnect_delay = 300  # 5 minutes
        self.reconnect_backoff_multiplier = 2
        self.current_reconnect_delay = self.reconnect_delay
        
        # Error categorization
        self._setup_error_categories()
        
        # Callbacks
        self.error_callbacks: list[Callable] = []
        self.connection_callbacks: list[Callable] = []
        self.recovery_callbacks: list[Callable] = []
        
        # Recovery task
        self.recovery_task: asyncio.Task | None = None
        self.reconnect_task: asyncio.Task | None = None
    
    def _setup_error_categories(self):
        """Setup error code categorization"""
        
        # Informational messages (not real errors)
        self.info_codes = {
            2104: "Market data farm connection is OK", 2106: "HMDS data farm connection is OK", 2107: "HMDS data farm connection is inactive but should be available upon demand", 2108: "Market data farm connection is inactive but should be available upon demand", 2158: "Sec-def data farm connection is OK", 2119: "Market data farm connection is broken"
        }
        
        # Warning messages
        self.warning_codes = {
            165: "Historical Market Data Service query message", 200: "No security definition has been found for the request", 354: "Requested market data is not subscribed", 2100: "New account data requested", 2101: "New account data received", 2102: "A market data farm is disconnected", 2103: "A market data farm is connected", 300: "Can't find EId with tickerId", 366: "No historical data query found for ticker id", 2109: "Order Event Warning"
        }
        
        # Connection errors (critical, require reconnection)
        self.connection_error_codes = {
            502: "Couldn't connect to TWS", 503: "The TWS is out of date and must be upgraded", 504: "Not connected", 1100: "Connectivity between IB and TWS has been lost", 1101: "Connectivity between IB and TWS has been lost - data maintained", 1102: "Connectivity between IB and TWS has been restored - data lost", 1300: "TWS socket port has been reset and this connection is being dropped", 10168: "Connectivity between TWS and server is broken. It will be restored automatically"
        }
        
        # Market data errors
        self.market_data_error_codes = {
            322: "Error reading request", 354: "Requested market data is not subscribed", 10167: "Requested market data is not subscribed", 10090: "Part of requested market data is not subscribed"
        }
        
        # Order errors
        self.order_error_codes = {
            104: "Order held while securities are located", 105: "Order held for position limits", 110: "The price does not conform to the minimum price variation for this contract", 201: "Order rejected - reason", 202: "Order cancelled", 203: "Security not allowed for this account", 321: "Error validating request", 399: "Order Message"
        }
        
        # Authentication/Permission errors
        self.auth_error_codes = {
            203: "The security is not available or allowed for this account", 321: "Error validating request.-'bd' : cause = You must specify the destination exchange", 10002: "Can't connect as the client id is already in use", 10147: "OrderId that needs to be cancelled cannot be cancelled, state", 10148: "OrderId that needs to be cancelled is not found"
        }
        
        # Non-recoverable errors
        self.fatal_error_codes = {
            503: "The TWS is out of date and must be upgraded", 10002: "Can't connect as the client id is already in use"
        }
    
    def classify_error(self, error_code: int, error_message: str) -> IBErrorSeverity:
        """Classify error severity"""
        if error_code in self.info_codes:
            return IBErrorSeverity.INFO
        elif error_code in self.warning_codes or error_code in self.market_data_error_codes:
            return IBErrorSeverity.WARNING
        elif error_code in self.fatal_error_codes:
            return IBErrorSeverity.CRITICAL
        elif error_code in self.connection_error_codes:
            return IBErrorSeverity.ERROR
        else:
            # Default classification based on error code range
            if 2000 <= error_code < 3000:
                return IBErrorSeverity.INFO
            elif error_code >= 10000:
                return IBErrorSeverity.WARNING
            else:
                return IBErrorSeverity.ERROR
    
    def is_recoverable_error(self, error_code: int) -> bool:
        """Check if error is recoverable"""
        return error_code not in self.fatal_error_codes
    
    async def handle_error(self, req_id: int, error_code: int, error_message: str) -> bool:
        """Handle IB error with recovery logic"""
        try:
            # Classify error
            severity = self.classify_error(error_code, error_message)
            recoverable = self.is_recoverable_error(error_code)
            
            # Create error object
            error = IBError(
                error_code=error_code, error_message=error_message, req_id=req_id, severity=severity, recoverable=recoverable
            )
            
            # Track error
            self.errors.append(error)
            self.error_counts[error_code] = self.error_counts.get(error_code, 0) + 1
            
            # Log error appropriately
            if severity == IBErrorSeverity.INFO:
                self.logger.info(f"IB Info {error_code}: {error_message} (req_id: {req_id})")
            elif severity == IBErrorSeverity.WARNING:
                self.logger.warning(f"IB Warning {error_code}: {error_message} (req_id: {req_id})")
            elif severity == IBErrorSeverity.ERROR:
                self.logger.error(f"IB Error {error_code}: {error_message} (req_id: {req_id})")
            else:  # CRITICAL
                self.logger.critical(f"IB Critical Error {error_code}: {error_message} (req_id: {req_id})")
            
            # Handle specific error types
            recovery_attempted = False
            
            if error_code in self.connection_error_codes:
                recovery_attempted = await self._handle_connection_error(error)
            elif error_code in self.market_data_error_codes:
                recovery_attempted = await self._handle_market_data_error(error)
            elif error_code in self.order_error_codes:
                recovery_attempted = await self._handle_order_error(error)
            
            # Notify callbacks
            await self._notify_error_callbacks(error)
            
            return recovery_attempted
            
        except Exception as e:
            self.logger.error(f"Error in error handler: {e}")
            return False
    
    async def _handle_connection_error(self, error: IBError) -> bool:
        """Handle connection-related errors"""
        try:
            # Update connection state
            if error.error_code in [502, 504, 1100, 1102, 1300]:
                await self._update_connection_state(IBConnectionState.FAILED)
                
                # Trigger reconnection if auto-reconnect is enabled
                if self.auto_reconnect and error.recoverable:
                    await self._schedule_reconnection()
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error handling connection error: {e}")
            return False
    
    async def _handle_market_data_error(self, error: IBError) -> bool:
        """Handle market data-related errors"""
        try:
            # Handle subscription errors
            if error.error_code in [354, 10167, 10090]:
                # Market data not subscribed - could retry subscription
                self.logger.warning(f"Market data subscription error: {error.error_message}")
                return False  # No automatic recovery for subscription errors
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error handling market data error: {e}")
            return False
    
    async def _handle_order_error(self, error: IBError) -> bool:
        """Handle order-related errors"""
        try:
            # Handle order validation errors
            if error.error_code in [201, 321]:
                self.logger.error(f"Order validation error: {error.error_message}")
                # Order errors typically require manual intervention
                return False
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error handling order error: {e}")
            return False
    
    async def _schedule_reconnection(self):
        """Schedule automatic reconnection"""
        if self.reconnect_task and not self.reconnect_task.done():
            self.logger.debug("Reconnection already scheduled")
            return
        
        self.reconnect_task = asyncio.create_task(self._reconnect_with_backoff())
    
    async def _reconnect_with_backoff(self):
        """Reconnect with exponential backoff"""
        try:
            await self._update_connection_state(IBConnectionState.RECONNECTING)
            
            while (self.connection_attempts < self.max_connection_attempts and 
                   self.connection_state != IBConnectionState.CONNECTED):
                
                self.connection_attempts += 1
                
                self.logger.info(f"Reconnection attempt {self.connection_attempts}/{self.max_connection_attempts} "
                               f"in {self.current_reconnect_delay} seconds")
                
                await asyncio.sleep(self.current_reconnect_delay)
                
                # Attempt reconnection
                success = await self._attempt_reconnection()
                
                if success:
                    await self._update_connection_state(IBConnectionState.CONNECTED)
                    self.connection_attempts = 0
                    self.current_reconnect_delay = self.reconnect_delay
                    self.logger.info("Reconnection successful")
                    break
                else:
                    # Increase delay for next attempt (exponential backoff)
                    self.current_reconnect_delay = min(
                        self.current_reconnect_delay * self.reconnect_backoff_multiplier, self.max_reconnect_delay
                    )
            
            if self.connection_attempts >= self.max_connection_attempts:
                await self._update_connection_state(IBConnectionState.FAILED)
                self.logger.error("Maximum reconnection attempts reached")
                
        except Exception as e:
            self.logger.error(f"Error during reconnection: {e}")
            await self._update_connection_state(IBConnectionState.FAILED)
    
    async def _attempt_reconnection(self) -> bool:
        """Attempt to reconnect to IB Gateway"""
        try:
            # Disconnect first if connected
            if hasattr(self.ib_client, 'disconnect_from_ib'):
                self.ib_client.disconnect_from_ib()
            
            await asyncio.sleep(1)  # Brief pause
            
            # Attempt connection
            if hasattr(self.ib_client, 'connect_to_ib'):
                success = self.ib_client.connect_to_ib()
                if success:
                    self.last_connection_time = datetime.now()
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error attempting reconnection: {e}")
            return False
    
    async def _update_connection_state(self, new_state: IBConnectionState):
        """Update connection state and notify callbacks"""
        if self.connection_state != new_state:
            old_state = self.connection_state
            self.connection_state = new_state
            
            # Create connection event
            event = IBConnectionEvent(
                event_type="state_change", state=new_state, message=f"Connection state changed from {old_state.value} to {new_state.value}"
            )
            
            self.connection_events.append(event)
            
            self.logger.info(f"Connection state: {old_state.value} -> {new_state.value}")
            
            # Notify callbacks
            await self._notify_connection_callbacks(event)
    
    async def _notify_error_callbacks(self, error: IBError):
        """Notify error callbacks"""
        for callback in self.error_callbacks:
            try:
                await callback(error)
            except Exception as e:
                self.logger.error(f"Error in error callback: {e}")
    
    async def _notify_connection_callbacks(self, event: IBConnectionEvent):
        """Notify connection callbacks"""
        for callback in self.connection_callbacks:
            try:
                await callback(event)
            except Exception as e:
                self.logger.error(f"Error in connection callback: {e}")
    
    async def _notify_recovery_callbacks(self, recovery_type: str, success: bool):
        """Notify recovery callbacks"""
        for callback in self.recovery_callbacks:
            try:
                await callback(recovery_type, success)
            except Exception as e:
                self.logger.error(f"Error in recovery callback: {e}")
    
    def add_error_callback(self, callback: Callable):
        """Add error callback"""
        self.error_callbacks.append(callback)
    
    def add_connection_callback(self, callback: Callable):
        """Add connection callback"""
        self.connection_callbacks.append(callback)
    
    def add_recovery_callback(self, callback: Callable):
        """Add recovery callback"""
        self.recovery_callbacks.append(callback)
    
    def get_connection_state(self) -> IBConnectionState:
        """Get current connection state"""
        return self.connection_state
    
    def get_recent_errors(self, limit: int = 50) -> list[IBError]:
        """Get recent errors"""
        return self.errors[-limit:] if limit > 0 else self.errors
    
    def get_error_statistics(self) -> dict[str, Any]:
        """Get error statistics"""
        total_errors = len(self.errors)
        if total_errors == 0:
            return {"total_errors": 0}
        
        severity_counts = {}
        for error in self.errors:
            severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1
        
        recent_errors = self.get_recent_errors(10)
        
        return {
            "total_errors": total_errors, "severity_counts": severity_counts, "error_code_counts": dict(self.error_counts), "connection_attempts": self.connection_attempts, "connection_state": self.connection_state.value, "last_connection_time": self.last_connection_time.isoformat() if self.last_connection_time else None, "recent_errors": [
                {
                    "code": error.error_code, "message": error.error_message, "severity": error.severity.value, "timestamp": error.timestamp.isoformat()
                }
                for error in recent_errors
            ]
        }
    
    def set_auto_reconnect(self, enabled: bool):
        """Enable or disable auto-reconnect"""
        self.auto_reconnect = enabled
        self.logger.info(f"Auto-reconnect {'enabled' if enabled else 'disabled'}")
    
    def set_reconnect_settings(self, delay: int = None, max_attempts: int = None, max_delay: int = None, backoff_multiplier: float = None):
        """Configure reconnection settings"""
        if delay is not None:
            self.reconnect_delay = delay
            self.current_reconnect_delay = delay
        if max_attempts is not None:
            self.max_connection_attempts = max_attempts
        if max_delay is not None:
            self.max_reconnect_delay = max_delay
        if backoff_multiplier is not None:
            self.reconnect_backoff_multiplier = backoff_multiplier
        
        self.logger.info(f"Reconnect settings updated: delay={self.reconnect_delay}s, "
                        f"max_attempts={self.max_connection_attempts}, "
                        f"max_delay={self.max_reconnect_delay}s, "
                        f"backoff={self.reconnect_backoff_multiplier}")
    
    async def force_reconnect(self):
        """Force immediate reconnection"""
        self.connection_attempts = 0
        self.current_reconnect_delay = self.reconnect_delay
        await self._schedule_reconnection()
    
    def clear_error_history(self):
        """Clear error history"""
        self.errors.clear()
        self.error_counts.clear()
        self.connection_events.clear()
        self.logger.info("Error history cleared")
    
    async def cleanup(self):
        """Cleanup error handler"""
        if self.recovery_task and not self.recovery_task.done():
            self.recovery_task.cancel()
        if self.reconnect_task and not self.reconnect_task.done():
            self.reconnect_task.cancel()
        
        self.clear_error_history()
        self.logger.info("Error handler cleaned up")


# Global error handler instance
_ib_error_handler: IBErrorHandler | None = None

def get_ib_error_handler(ib_client) -> IBErrorHandler:
    """Get or create the IB error handler singleton"""
    global _ib_error_handler
    
    if _ib_error_handler is None:
        _ib_error_handler = IBErrorHandler(ib_client)
    
    return _ib_error_handler

def reset_ib_error_handler():
    """Reset the error handler singleton (for testing)"""
    global _ib_error_handler
    if _ib_error_handler:
        asyncio.create_task(_ib_error_handler.cleanup())
    _ib_error_handler = None