#!/usr/bin/env python3
"""
Clock Testing Utilities - Comprehensive tests for simulated clock implementation
Tests deterministic behavior, time advancement, and timer functionality.
"""

import pytest
import time
from datetime import datetime, timezone, timedelta
from clock import Clock, LiveClock, TestClock, create_clock, TimeEvent
from clock import unix_nanos_to_dt, dt_to_unix_nanos, millis_to_nanos, nanos_to_millis
from clock import NANOS_IN_SECOND, NANOS_IN_MINUTE, NANOS_IN_HOUR


class TestLiveClock:
    """Test LiveClock functionality"""
    
    def test_live_clock_creation(self):
        """Test LiveClock can be created"""
        clock = LiveClock()
        assert isinstance(clock, Clock)
        assert isinstance(clock, LiveClock)
    
    def test_live_clock_timestamps(self):
        """Test LiveClock provides real timestamps"""
        clock = LiveClock()
        
        # Get timestamps
        ts_float = clock.timestamp()
        ts_ms = clock.timestamp_ms()
        ts_us = clock.timestamp_us()
        ts_ns = clock.timestamp_ns()
        
        # Basic validation
        assert isinstance(ts_float, float)
        assert isinstance(ts_ms, int)
        assert isinstance(ts_us, int)
        assert isinstance(ts_ns, int)
        
        # Timestamps should be in reasonable ranges
        now = time.time()
        assert abs(ts_float - now) < 1.0  # Within 1 second
        assert abs(ts_ms - (now * 1000)) < 1000  # Within 1 second in ms
        
        # Ordering should be correct
        assert ts_ns > ts_us > ts_ms > ts_float
    
    def test_live_clock_utc_now(self):
        """Test LiveClock UTC datetime"""
        clock = LiveClock()
        dt = clock.utc_now()
        
        assert isinstance(dt, datetime)
        assert dt.tzinfo == timezone.utc
        
        # Should be close to current time
        now = datetime.now(timezone.utc)
        diff = abs((dt - now).total_seconds())
        assert diff < 1.0  # Within 1 second


class TestTestClock:
    """Test TestClock (simulated clock) functionality"""
    
    def test_test_clock_creation(self):
        """Test TestClock can be created"""
        clock = TestClock()
        assert isinstance(clock, Clock)
        assert isinstance(clock, TestClock)
    
    def test_test_clock_creation_with_start_time(self):
        """Test TestClock with specific start time"""
        start_time_ns = 1609459200_000_000_000  # 2021-01-01 00:00:00 UTC
        clock = TestClock(start_time_ns=start_time_ns)
        
        assert clock.timestamp_ns() == start_time_ns
        assert clock.timestamp() == 1609459200.0
        assert clock.timestamp_ms() == 1609459200000
    
    def test_test_clock_timestamps(self):
        """Test TestClock timestamp consistency"""
        start_time_ns = 1609459200_000_000_000
        clock = TestClock(start_time_ns=start_time_ns)
        
        # All timestamp methods should be consistent
        ts_float = clock.timestamp()
        ts_ms = clock.timestamp_ms()
        ts_us = clock.timestamp_us()
        ts_ns = clock.timestamp_ns()
        
        assert ts_ns == start_time_ns
        assert ts_us == start_time_ns // 1000
        assert ts_ms == start_time_ns // 1_000_000
        assert ts_float == start_time_ns / 1_000_000_000
    
    def test_test_clock_time_advancement(self):
        """Test TestClock time advancement"""
        start_time_ns = 1609459200_000_000_000
        clock = TestClock(start_time_ns=start_time_ns)
        
        # Advance by 1 second
        advance_ns = NANOS_IN_SECOND
        triggered_events = clock.advance_time(advance_ns)
        
        assert len(triggered_events) == 0  # No timers set
        assert clock.timestamp_ns() == start_time_ns + advance_ns
        assert clock.timestamp() == 1609459201.0
    
    def test_test_clock_set_time(self):
        """Test TestClock absolute time setting"""
        clock = TestClock()
        
        new_time_ns = 1609459200_000_000_000
        clock.set_time(new_time_ns)
        
        assert clock.timestamp_ns() == new_time_ns
        assert clock.timestamp() == 1609459200.0
    
    def test_test_clock_utc_now(self):
        """Test TestClock UTC datetime conversion"""
        start_time_ns = 1609459200_000_000_000  # 2021-01-01 00:00:00 UTC
        clock = TestClock(start_time_ns=start_time_ns)
        
        dt = clock.utc_now()
        
        assert isinstance(dt, datetime)
        assert dt.tzinfo == timezone.utc
        assert dt.year == 2021
        assert dt.month == 1
        assert dt.day == 1
        assert dt.hour == 0
        assert dt.minute == 0
        assert dt.second == 0


class TestTestClockTimers:
    """Test TestClock timer functionality"""
    
    def test_timer_basic(self):
        """Test basic timer functionality"""
        clock = TestClock()
        start_time = clock.timestamp_ns()
        
        # Set timer for 5 seconds
        timer_delay_ns = 5 * NANOS_IN_SECOND
        alert_time_ns = start_time + timer_delay_ns
        
        callback_called = []
        def timer_callback(event: TimeEvent):
            callback_called.append(event)
        
        clock.set_time_alert_ns("test_timer", alert_time_ns, timer_callback)
        
        # Timer should be registered
        assert "test_timer" in clock.timer_names
        assert clock.timer_count == 1
        assert clock.next_time_ns() == alert_time_ns
        
        # Advance time but not enough to trigger
        triggered = clock.advance_time(3 * NANOS_IN_SECOND)
        assert len(triggered) == 0
        assert len(callback_called) == 0
        assert clock.timer_count == 1  # Timer still active
        
        # Advance enough to trigger
        triggered = clock.advance_time(3 * NANOS_IN_SECOND)  # Total: 6 seconds
        assert len(triggered) == 1
        assert len(callback_called) == 1
        assert clock.timer_count == 0  # Timer removed after firing
        
        # Check event details
        event = triggered[0]
        assert event.name == "test_timer"
        assert event.timestamp_ns == alert_time_ns
    
    def test_multiple_timers(self):
        """Test multiple timers with different delays"""
        clock = TestClock()
        start_time = clock.timestamp_ns()
        
        callbacks = []
        
        def make_callback(name):
            def callback(event):
                callbacks.append((name, event.timestamp_ns))
            return callback
        
        # Set multiple timers
        clock.set_time_alert_ns("timer_1s", start_time + 1 * NANOS_IN_SECOND, make_callback("1s"))
        clock.set_time_alert_ns("timer_3s", start_time + 3 * NANOS_IN_SECOND, make_callback("3s"))
        clock.set_time_alert_ns("timer_2s", start_time + 2 * NANOS_IN_SECOND, make_callback("2s"))
        
        assert clock.timer_count == 3
        assert clock.next_time_ns() == start_time + 1 * NANOS_IN_SECOND
        
        # Advance 1.5 seconds - should trigger first timer
        triggered = clock.advance_time(int(1.5 * NANOS_IN_SECOND))
        assert len(triggered) == 1
        assert len(callbacks) == 1
        assert callbacks[0][0] == "1s"
        assert clock.timer_count == 2
        
        # Advance another 1 second - should trigger second timer
        triggered = clock.advance_time(1 * NANOS_IN_SECOND)
        assert len(triggered) == 1
        assert len(callbacks) == 2
        assert callbacks[1][0] == "2s"
        assert clock.timer_count == 1
        
        # Advance another 1 second - should trigger third timer
        triggered = clock.advance_time(1 * NANOS_IN_SECOND)
        assert len(triggered) == 1
        assert len(callbacks) == 3
        assert callbacks[2][0] == "3s"
        assert clock.timer_count == 0
    
    def test_timer_cancellation(self):
        """Test timer cancellation"""
        clock = TestClock()
        start_time = clock.timestamp_ns()
        
        clock.set_time_alert_ns("timer1", start_time + NANOS_IN_SECOND)
        clock.set_time_alert_ns("timer2", start_time + 2 * NANOS_IN_SECOND)
        
        assert clock.timer_count == 2
        
        # Cancel first timer
        result = clock.cancel_timer("timer1")
        assert result is True
        assert clock.timer_count == 1
        assert "timer1" not in clock.timer_names
        assert "timer2" in clock.timer_names
        
        # Try to cancel non-existent timer
        result = clock.cancel_timer("nonexistent")
        assert result is False
        
        # Cancel all timers
        clock.cancel_timers()
        assert clock.timer_count == 0
        assert len(clock.timer_names) == 0
    
    def test_default_handler(self):
        """Test default handler for timers without specific callbacks"""
        clock = TestClock()
        start_time = clock.timestamp_ns()
        
        default_events = []
        def default_handler(event: TimeEvent):
            default_events.append(event)
        
        clock.register_default_handler(default_handler)
        clock.set_time_alert_ns("test_timer", start_time + NANOS_IN_SECOND)
        
        # Trigger timer
        triggered = clock.advance_time(2 * NANOS_IN_SECOND)
        
        assert len(triggered) == 1
        assert len(default_events) == 1
        assert default_events[0].name == "test_timer"
    
    def test_timer_ns_convenience_method(self):
        """Test set_timer_ns convenience method"""
        clock = TestClock()
        start_time = clock.timestamp_ns()
        
        callback_events = []
        def callback(event):
            callback_events.append(event)
        
        # Set timer with interval
        clock.set_timer_ns("interval_timer", 5 * NANOS_IN_SECOND, callback)
        
        assert clock.timer_count == 1
        expected_time = start_time + 5 * NANOS_IN_SECOND
        assert clock.next_time_ns() == expected_time
        
        # Trigger timer
        clock.advance_time(6 * NANOS_IN_SECOND)
        
        assert len(callback_events) == 1
        assert callback_events[0].timestamp_ns == expected_time


class TestClockFactory:
    """Test clock factory function"""
    
    def test_create_live_clock(self):
        """Test creating LiveClock via factory"""
        clock = create_clock("live")
        assert isinstance(clock, LiveClock)
    
    def test_create_test_clock(self):
        """Test creating TestClock via factory"""
        clock = create_clock("test")
        assert isinstance(clock, TestClock)
        
        # With start time
        start_time_ns = 1609459200_000_000_000
        clock = create_clock("test", start_time_ns=start_time_ns)
        assert isinstance(clock, TestClock)
        assert clock.timestamp_ns() == start_time_ns
    
    def test_create_clock_invalid_type(self):
        """Test factory with invalid clock type"""
        with pytest.raises(ValueError, match="Unknown clock type"):
            create_clock("invalid_type")


class TestTimeUtilities:
    """Test time utility functions"""
    
    def test_unix_nanos_conversions(self):
        """Test Unix nanoseconds conversions"""
        # Test timestamp: 2021-01-01 00:00:00 UTC
        nanos = 1609459200_000_000_000
        
        # Convert to datetime
        dt = unix_nanos_to_dt(nanos)
        assert dt.year == 2021
        assert dt.month == 1
        assert dt.day == 1
        assert dt.tzinfo == timezone.utc
        
        # Convert back
        nanos_back = dt_to_unix_nanos(dt)
        assert nanos_back == nanos
    
    def test_time_unit_conversions(self):
        """Test time unit conversion functions"""
        millis = 5000  # 5 seconds
        nanos = millis_to_nanos(millis)
        assert nanos == 5_000_000_000
        
        millis_back = nanos_to_millis(nanos)
        assert millis_back == millis
    
    def test_time_constants(self):
        """Test time constants are correct"""
        assert NANOS_IN_SECOND == 1_000_000_000
        assert NANOS_IN_MINUTE == 60 * NANOS_IN_SECOND
        assert NANOS_IN_HOUR == 60 * NANOS_IN_MINUTE


class TestClockIntegration:
    """Test clock integration scenarios"""
    
    def test_deterministic_sequence(self):
        """Test deterministic event sequence"""
        # Create test clock with known start time
        start_time_ns = 1609459200_000_000_000  # 2021-01-01 00:00:00 UTC
        clock = TestClock(start_time_ns=start_time_ns)
        
        events = []
        def event_handler(event):
            events.append((clock.timestamp_ns(), event.name))
        
        # Setup events at specific times
        clock.set_time_alert_ns("event_10s", start_time_ns + 10 * NANOS_IN_SECOND, event_handler)
        clock.set_time_alert_ns("event_5s", start_time_ns + 5 * NANOS_IN_SECOND, event_handler)
        clock.set_time_alert_ns("event_15s", start_time_ns + 15 * NANOS_IN_SECOND, event_handler)
        
        # Advance time in steps
        clock.advance_time(3 * NANOS_IN_SECOND)  # t=3s
        assert len(events) == 0
        
        clock.advance_time(3 * NANOS_IN_SECOND)  # t=6s
        assert len(events) == 1
        assert events[0][1] == "event_5s"
        
        clock.advance_time(5 * NANOS_IN_SECOND)  # t=11s
        assert len(events) == 2
        assert events[1][1] == "event_10s"
        
        clock.advance_time(5 * NANOS_IN_SECOND)  # t=16s
        assert len(events) == 3
        assert events[2][1] == "event_15s"
        
        # Verify exact timestamps
        assert events[0][0] == start_time_ns + 6 * NANOS_IN_SECOND  # When we checked at 6s
        assert events[1][0] == start_time_ns + 11 * NANOS_IN_SECOND  # When we checked at 11s
        assert events[2][0] == start_time_ns + 16 * NANOS_IN_SECOND  # When we checked at 16s
    
    def test_backtesting_scenario(self):
        """Test backtesting-like scenario with fast time advancement"""
        clock = TestClock()
        start_time = clock.timestamp_ns()
        
        # Simulate daily events for a week
        daily_events = []
        for day in range(7):
            day_time_ns = start_time + day * 24 * NANOS_IN_HOUR
            clock.set_time_alert_ns(f"day_{day}", day_time_ns, 
                                  lambda e, d=day: daily_events.append(d))
        
        # Fast-forward through the week
        clock.advance_time(8 * 24 * NANOS_IN_HOUR)  # 8 days
        
        assert len(daily_events) == 7
        assert daily_events == [0, 1, 2, 3, 4, 5, 6]
        
        # Verify final time
        expected_final_time = start_time + 8 * 24 * NANOS_IN_HOUR
        assert clock.timestamp_ns() == expected_final_time


if __name__ == "__main__":
    # Run specific test for development
    pytest.main([__file__, "-v"])