#!/usr/bin/env python3
"""
Clock Synchronization API Routes for Nautilus Trading Platform
Provides server time synchronization endpoints for Phase 3 frontend integration.
"""

import time
import json
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse

from engines.common.clock import get_global_clock, Clock, LiveClock, TestClock
from api_gateway.clock_timeout_manager import get_global_timeout_manager, RequestPriority

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/clock", tags=["clock"])

# Pydantic models for request/response
class ClockSyncRequest(BaseModel):
    """Request model for clock synchronization"""
    client_timestamp: int = Field(..., description="Client timestamp in milliseconds")
    sync_request_id: str = Field(..., description="Unique sync request identifier")
    client_timezone: Optional[str] = Field(None, description="Client timezone")
    precision_level: Optional[str] = Field("standard", description="Precision level: standard, high, ultra")

class ClockSyncResponse(BaseModel):
    """Response model for clock synchronization"""
    server_timestamp: int = Field(..., description="Server timestamp in milliseconds")
    server_timestamp_ns: int = Field(..., description="Server timestamp in nanoseconds")
    client_timestamp: int = Field(..., description="Original client timestamp")
    sync_request_id: str = Field(..., description="Sync request identifier")
    processing_time_ns: int = Field(..., description="Server processing time in nanoseconds")
    server_timezone: str = Field(..., description="Server timezone")
    clock_type: str = Field(..., description="Clock type: live or test")
    precision_level: str = Field(..., description="Precision level used")

class ClockStatusResponse(BaseModel):
    """Response model for clock status"""
    current_timestamp: int = Field(..., description="Current server timestamp in milliseconds")
    current_timestamp_ns: int = Field(..., description="Current server timestamp in nanoseconds")
    server_timezone: str = Field(..., description="Server timezone")
    clock_type: str = Field(..., description="Clock type: live or test")
    uptime_seconds: float = Field(..., description="Clock uptime in seconds")
    is_synchronized: bool = Field(..., description="Whether clock is synchronized")

class MarketHoursRequest(BaseModel):
    """Request model for market hours information"""
    markets: Optional[List[str]] = Field(None, description="List of markets to query")
    date: Optional[str] = Field(None, description="Date in YYYY-MM-DD format")

class MarketHoursResponse(BaseModel):
    """Response model for market hours"""
    market: str = Field(..., description="Market name")
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    timezone: str = Field(..., description="Market timezone")
    open_time: str = Field(..., description="Market open time")
    close_time: str = Field(..., description="Market close time")
    is_open: bool = Field(..., description="Whether market is currently open")
    session_type: str = Field(..., description="Current session type")
    next_event: Optional[str] = Field(None, description="Next market event (open/close)")
    next_event_timestamp: Optional[int] = Field(None, description="Next event timestamp")

class TimeZoneInfo(BaseModel):
    """Time zone information model"""
    timezone: str = Field(..., description="Timezone name")
    offset_hours: int = Field(..., description="Offset from UTC in hours")
    is_dst: bool = Field(..., description="Whether daylight saving time is active")
    local_time: str = Field(..., description="Local time in timezone")

# Clock service initialization
clock = get_global_clock()
timeout_manager = get_global_timeout_manager()

# Market configurations
MARKET_CONFIGS = {
    'NYSE': {'timezone': 'America/New_York', 'open': '09:30', 'close': '16:00'},
    'NASDAQ': {'timezone': 'America/New_York', 'open': '09:30', 'close': '16:00'},
    'LSE': {'timezone': 'Europe/London', 'open': '08:00', 'close': '16:30'},
    'TSE': {'timezone': 'Asia/Tokyo', 'open': '09:00', 'close': '15:00'},
    'ASX': {'timezone': 'Australia/Sydney', 'open': '10:00', 'close': '16:00'},
    'HKEX': {'timezone': 'Asia/Hong_Kong', 'open': '09:30', 'close': '16:00'},
}

@router.post("/server-time", response_model=ClockSyncResponse)
async def sync_server_time(request: ClockSyncRequest):
    """
    Synchronize with server time for precise client-server timing.
    Used by frontend useClockSync hook for maintaining accurate time synchronization.
    """
    try:
        # Record processing start time
        processing_start_ns = clock.timestamp_ns()
        
        # Get current server time
        server_timestamp_ns = clock.timestamp_ns()
        server_timestamp_ms = server_timestamp_ns // 1_000_000
        
        # Calculate processing time
        processing_time_ns = clock.timestamp_ns() - processing_start_ns
        
        # Determine precision level adjustments
        precision_adjustments = {
            'standard': 0,
            'high': 100_000,     # 0.1ms precision
            'ultra': 1_000       # 1µs precision
        }
        
        precision_ns = precision_adjustments.get(request.precision_level, 0)
        
        response = ClockSyncResponse(
            server_timestamp=server_timestamp_ms,
            server_timestamp_ns=server_timestamp_ns,
            client_timestamp=request.client_timestamp,
            sync_request_id=request.sync_request_id,
            processing_time_ns=processing_time_ns,
            server_timezone='UTC',
            clock_type='test' if isinstance(clock, TestClock) else 'live',
            precision_level=request.precision_level
        )
        
        logger.debug(f"Clock sync request {request.sync_request_id} processed in {processing_time_ns}ns")
        return response
        
    except Exception as e:
        logger.error(f"Clock synchronization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Clock synchronization failed: {str(e)}")

@router.get("/status", response_model=ClockStatusResponse)
async def get_clock_status():
    """
    Get current clock status and server time information.
    Provides health information for clock synchronization monitoring.
    """
    try:
        current_timestamp_ns = clock.timestamp_ns()
        current_timestamp_ms = current_timestamp_ns // 1_000_000
        
        # Calculate uptime (placeholder - would need actual startup tracking)
        uptime_seconds = 3600.0  # Default 1 hour
        
        response = ClockStatusResponse(
            current_timestamp=current_timestamp_ms,
            current_timestamp_ns=current_timestamp_ns,
            server_timezone='UTC',
            clock_type='test' if isinstance(clock, TestClock) else 'live',
            uptime_seconds=uptime_seconds,
            is_synchronized=True  # Always synchronized for now
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to get clock status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get clock status: {str(e)}")

@router.get("/market-hours", response_model=List[MarketHoursResponse])
async def get_market_hours(
    markets: Optional[str] = None,
    date: Optional[str] = None
):
    """
    Get market hours information for specified markets.
    Used by frontend for trading session awareness and scheduling.
    """
    try:
        # Parse markets parameter
        if markets:
            requested_markets = [m.strip().upper() for m in markets.split(',')]
        else:
            requested_markets = list(MARKET_CONFIGS.keys())
        
        # Use current date if not specified
        if date:
            try:
                target_date = datetime.strptime(date, '%Y-%m-%d').date()
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        else:
            target_date = datetime.now(timezone.utc).date()
        
        current_time_ns = clock.timestamp_ns()
        current_time = datetime.fromtimestamp(current_time_ns / 1e9, tz=timezone.utc)
        
        market_hours_list = []
        
        for market in requested_markets:
            if market not in MARKET_CONFIGS:
                continue
            
            market_config = MARKET_CONFIGS[market]
            market_tz = market_config['timezone']
            open_time = market_config['open']
            close_time = market_config['close']
            
            # Create market times for the target date
            market_open = datetime.strptime(f"{target_date} {open_time}", '%Y-%m-%d %H:%M').replace(tzinfo=timezone.utc)
            market_close = datetime.strptime(f"{target_date} {close_time}", '%Y-%m-%d %H:%M').replace(tzinfo=timezone.utc)
            
            # Determine if market is currently open
            is_open = market_open <= current_time < market_close
            
            # Determine session type
            if is_open:
                session_type = 'regular'
                next_event = 'close'
                next_event_timestamp = int(market_close.timestamp() * 1000)
            elif current_time < market_open:
                session_type = 'pre-market'
                next_event = 'open'
                next_event_timestamp = int(market_open.timestamp() * 1000)
            else:
                session_type = 'after-hours'
                # Next open is tomorrow
                next_open = market_open.replace(day=market_open.day + 1)
                # Handle weekends
                while next_open.weekday() >= 5:  # Saturday = 5, Sunday = 6
                    next_open = next_open.replace(day=next_open.day + 1)
                next_event = 'open'
                next_event_timestamp = int(next_open.timestamp() * 1000)
            
            market_hours = MarketHoursResponse(
                market=market,
                date=target_date.strftime('%Y-%m-%d'),
                timezone=market_tz,
                open_time=open_time,
                close_time=close_time,
                is_open=is_open,
                session_type=session_type,
                next_event=next_event,
                next_event_timestamp=next_event_timestamp
            )
            
            market_hours_list.append(market_hours)
        
        return market_hours_list
        
    except Exception as e:
        logger.error(f"Failed to get market hours: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get market hours: {str(e)}")

@router.get("/timezone/{timezone_name}", response_model=TimeZoneInfo)
async def get_timezone_info(timezone_name: str):
    """
    Get information about a specific timezone.
    Useful for multi-timezone trading operations.
    """
    try:
        import zoneinfo
        
        try:
            tz = zoneinfo.ZoneInfo(timezone_name)
        except zoneinfo.ZoneInfoNotFoundError:
            raise HTTPException(status_code=404, detail=f"Timezone '{timezone_name}' not found")
        
        current_time_ns = clock.timestamp_ns()
        current_time_utc = datetime.fromtimestamp(current_time_ns / 1e9, tz=timezone.utc)
        local_time = current_time_utc.astimezone(tz)
        
        # Calculate offset
        offset_seconds = local_time.utcoffset().total_seconds()
        offset_hours = int(offset_seconds // 3600)
        
        # Check if DST is active
        is_dst = local_time.dst() is not None and local_time.dst().total_seconds() > 0
        
        timezone_info = TimeZoneInfo(
            timezone=timezone_name,
            offset_hours=offset_hours,
            is_dst=is_dst,
            local_time=local_time.strftime('%Y-%m-%d %H:%M:%S %Z')
        )
        
        return timezone_info
        
    except ImportError:
        # Fallback for systems without zoneinfo
        raise HTTPException(status_code=501, detail="Timezone support not available")
    except Exception as e:
        logger.error(f"Failed to get timezone info for {timezone_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get timezone info: {str(e)}")

@router.get("/performance-metrics")
async def get_clock_performance_metrics():
    """
    Get clock performance and synchronization metrics.
    Used for monitoring clock accuracy and system performance.
    """
    try:
        # Get timeout manager metrics (includes API performance)
        timeout_metrics = timeout_manager.get_request_metrics()
        
        current_time_ns = clock.timestamp_ns()
        
        performance_metrics = {
            'timestamp': current_time_ns,
            'timestamp_ms': current_time_ns // 1_000_000,
            'clock_type': 'test' if isinstance(clock, TestClock) else 'live',
            'precision_ns': 1_000,  # 1µs precision
            'api_metrics': {
                'total_requests': timeout_metrics.total_requests,
                'completed_requests': timeout_metrics.completed_requests,
                'active_requests': timeout_metrics.active_requests,
                'average_response_time_ms': timeout_metrics.average_response_time_ns / 1e6,
                'p95_response_time_ms': timeout_metrics.p95_response_time_ns / 1e6,
                'p99_response_time_ms': timeout_metrics.p99_response_time_ns / 1e6,
                'rate_limit_violations': timeout_metrics.rate_limit_violations
            }
        }
        
        # Add TestClock specific metrics if available
        if isinstance(clock, TestClock):
            performance_metrics['test_clock'] = {
                'timer_count': clock.timer_count,
                'timer_names': clock.timer_names,
                'next_timer_ns': clock.next_time_ns()
            }
        
        return performance_metrics
        
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {str(e)}")

@router.post("/test-clock/advance")
async def advance_test_clock(duration_ns: int):
    """
    Advance test clock by specified nanoseconds.
    Only available when running with TestClock for testing/backtesting.
    """
    if not isinstance(clock, TestClock):
        raise HTTPException(status_code=400, detail="Test clock operations only available in test mode")
    
    try:
        triggered_events = clock.advance_time(duration_ns)
        
        return {
            'advanced_by_ns': duration_ns,
            'new_timestamp_ns': clock.timestamp_ns(),
            'triggered_events': len(triggered_events),
            'events': [
                {
                    'name': event.name,
                    'timestamp_ns': event.timestamp_ns
                } for event in triggered_events
            ]
        }
        
    except Exception as e:
        logger.error(f"Failed to advance test clock: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to advance test clock: {str(e)}")

@router.post("/test-clock/set-time")
async def set_test_clock_time(timestamp_ns: int):
    """
    Set test clock to specific timestamp.
    Only available when running with TestClock for testing/backtesting.
    """
    if not isinstance(clock, TestClock):
        raise HTTPException(status_code=400, detail="Test clock operations only available in test mode")
    
    try:
        clock.set_time(timestamp_ns)
        
        return {
            'set_timestamp_ns': timestamp_ns,
            'current_timestamp_ns': clock.timestamp_ns(),
            'formatted_time': datetime.fromtimestamp(timestamp_ns / 1e9, tz=timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to set test clock time: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to set test clock time: {str(e)}")

@router.get("/health")
async def clock_health_check():
    """
    Health check endpoint for clock services.
    Used by monitoring systems to verify clock functionality.
    """
    try:
        current_time_ns = clock.timestamp_ns()
        
        health_status = {
            'status': 'healthy',
            'timestamp': current_time_ns,
            'timestamp_ms': current_time_ns // 1_000_000,
            'clock_type': 'test' if isinstance(clock, TestClock) else 'live',
            'uptime': 'operational',
            'last_check': datetime.now(timezone.utc).isoformat()
        }
        
        return health_status
        
    except Exception as e:
        logger.error(f"Clock health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': int(time.time() * 1000)
            }
        )

# Startup event to ensure timeout manager is running
@router.on_event("startup")
async def startup_clock_services():
    """Initialize clock services on startup"""
    logger.info("Starting clock synchronization services")
    timeout_manager.start_monitoring()

@router.on_event("shutdown")  
async def shutdown_clock_services():
    """Cleanup clock services on shutdown"""
    logger.info("Shutting down clock synchronization services")
    timeout_manager.stop_monitoring()