#!/usr/bin/env python3
"""
Clock Abstraction - Simulated Clock for MessageBus
Matches NautilusTrader's Rust clock implementation for deterministic testing and backtesting.
"""

import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timezone
import threading
from dataclasses import dataclass


@dataclass
class TimeEvent:
    """Time event for timer callbacks"""
    name: str
    timestamp_ns: int
    callback: Optional[Callable] = None


class Clock(ABC):
    """
    Abstract base clock for time operations
    Provides nanosecond precision timestamps like NautilusTrader's Rust implementation
    """
    
    @abstractmethod
    def timestamp(self) -> float:
        """Get current timestamp as float seconds"""
        pass
    
    @abstractmethod
    def timestamp_ms(self) -> int:
        """Get current timestamp as milliseconds"""
        pass
    
    @abstractmethod
    def timestamp_us(self) -> int:
        """Get current timestamp as microseconds"""
        pass
    
    @abstractmethod
    def timestamp_ns(self) -> int:
        """Get current timestamp as nanoseconds"""
        pass
    
    @abstractmethod
    def utc_now(self) -> datetime:
        """Get current UTC datetime"""
        pass


class LiveClock(Clock):
    """
    Live clock using system time
    Production clock for real trading operations
    """
    
    def __init__(self):
        self._timer_callbacks: Dict[str, Callable] = {}
        self._lock = threading.Lock()
    
    def timestamp(self) -> float:
        """Get current timestamp as float seconds"""
        return time.time()
    
    def timestamp_ms(self) -> int:
        """Get current timestamp as milliseconds"""
        return int(time.time() * 1000)
    
    def timestamp_us(self) -> int:
        """Get current timestamp as microseconds"""
        return int(time.time() * 1000000)
    
    def timestamp_ns(self) -> int:
        """Get current timestamp as nanoseconds"""
        return time.time_ns()
    
    def utc_now(self) -> datetime:
        """Get current UTC datetime"""
        return datetime.now(timezone.utc)
    
    def register_timer(self, name: str, callback: Callable[[TimeEvent], None]):
        """Register a timer callback (for compatibility with TestClock)"""
        with self._lock:
            self._timer_callbacks[name] = callback


class TestClock(Clock):
    """
    Simulated clock for testing and backtesting
    Provides controllable time advancement like NautilusTrader's Rust TestClock
    """
    
    def __init__(self, start_time_ns: Optional[int] = None):
        if start_time_ns is None:
            start_time_ns = time.time_ns()
            
        self._time_ns = start_time_ns
        self._timers: Dict[str, TimeEvent] = {}
        self._default_handler: Optional[Callable] = None
        self._lock = threading.Lock()
    
    def timestamp(self) -> float:
        """Get current simulated timestamp as float seconds"""
        return self._time_ns / 1_000_000_000
    
    def timestamp_ms(self) -> int:
        """Get current simulated timestamp as milliseconds"""
        return self._time_ns // 1_000_000
    
    def timestamp_us(self) -> int:
        """Get current simulated timestamp as microseconds"""
        return self._time_ns // 1_000
    
    def timestamp_ns(self) -> int:
        """Get current simulated timestamp as nanoseconds"""
        return self._time_ns
    
    def utc_now(self) -> datetime:
        """Get current simulated UTC datetime"""
        return datetime.fromtimestamp(self.timestamp(), tz=timezone.utc)
    
    def advance_time(self, duration_ns: int) -> List[TimeEvent]:
        """
        Advance simulated time by duration in nanoseconds
        Returns list of triggered time events
        """
        triggered_events = []
        
        with self._lock:
            new_time_ns = self._time_ns + duration_ns
            
            # Find and trigger any timers that should fire
            expired_timers = []
            for name, timer_event in self._timers.items():
                if timer_event.timestamp_ns <= new_time_ns:
                    triggered_events.append(timer_event)
                    expired_timers.append(name)
            
            # Remove expired one-time timers
            for name in expired_timers:
                del self._timers[name]
            
            # Update time
            self._time_ns = new_time_ns
        
        # Execute callbacks outside of lock
        for event in triggered_events:
            if event.callback:
                event.callback(event)
            elif self._default_handler:
                self._default_handler(event)
        
        return triggered_events
    
    def set_time(self, timestamp_ns: int):
        """Set absolute time in nanoseconds"""
        with self._lock:
            self._time_ns = timestamp_ns
    
    def set_time_alert_ns(self, name: str, alert_time_ns: int, callback: Optional[Callable] = None):
        """Set a time alert to trigger at specific time"""
        with self._lock:
            self._timers[name] = TimeEvent(
                name=name,
                timestamp_ns=alert_time_ns,
                callback=callback
            )
    
    def set_timer_ns(self, name: str, interval_ns: int, callback: Optional[Callable] = None):
        """Set a recurring timer (simplified - only single shot for now)"""
        alert_time_ns = self._time_ns + interval_ns
        self.set_time_alert_ns(name, alert_time_ns, callback)
    
    def cancel_timer(self, name: str) -> bool:
        """Cancel a timer by name"""
        with self._lock:
            if name in self._timers:
                del self._timers[name]
                return True
        return False
    
    def cancel_timers(self):
        """Cancel all timers"""
        with self._lock:
            self._timers.clear()
    
    def register_default_handler(self, callback: Callable[[TimeEvent], None]):
        """Register default handler for time events without specific callbacks"""
        self._default_handler = callback
    
    @property
    def timer_names(self) -> List[str]:
        """Get list of active timer names"""
        with self._lock:
            return list(self._timers.keys())
    
    @property
    def timer_count(self) -> int:
        """Get count of active timers"""
        with self._lock:
            return len(self._timers)
    
    def next_time_ns(self) -> Optional[int]:
        """Get timestamp of next scheduled timer event"""
        with self._lock:
            if not self._timers:
                return None
            return min(timer.timestamp_ns for timer in self._timers.values())


def create_clock(clock_type: str = "live", **kwargs) -> Clock:
    """
    Factory function to create clock instances
    
    Args:
        clock_type: "live" for production, "test" for testing/backtesting
        **kwargs: Additional arguments for clock initialization
    
    Returns:
        Clock instance
    """
    if clock_type == "live":
        return LiveClock()
    elif clock_type == "test":
        return TestClock(**kwargs)
    else:
        raise ValueError(f"Unknown clock type: {clock_type}")


# Convenience functions for common time operations
def unix_nanos_to_dt(nanos: int) -> datetime:
    """Convert Unix nanoseconds timestamp to datetime"""
    return datetime.fromtimestamp(nanos / 1_000_000_000, tz=timezone.utc)


def dt_to_unix_nanos(dt: datetime) -> int:
    """Convert datetime to Unix nanoseconds timestamp"""
    return int(dt.timestamp() * 1_000_000_000)


def millis_to_nanos(millis: int) -> int:
    """Convert milliseconds to nanoseconds"""
    return millis * 1_000_000


def nanos_to_millis(nanos: int) -> int:
    """Convert nanoseconds to milliseconds"""
    return nanos // 1_000_000


# Constants for time calculations
NANOS_IN_MICROSECOND = 1_000
NANOS_IN_MILLISECOND = 1_000_000
NANOS_IN_SECOND = 1_000_000_000
NANOS_IN_MINUTE = 60 * NANOS_IN_SECOND
NANOS_IN_HOUR = 60 * NANOS_IN_MINUTE