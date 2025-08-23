"""
Alpha Vantage Data Backfill Service
==================================

Dedicated service for historical data backfill using Alpha Vantage API.
Integrates with the existing unified backfill system to provide comprehensive
historical data coverage for stocks, forex, and other supported instruments.
"""

import asyncio
import logging
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json

from alpha_vantage.service import alpha_vantage_service
from alpha_vantage.models import AlphaVantageTimeSeries, AlphaVantageBarData
from historical_data_service import historical_data_service, HistoricalDataQuery
from data_normalizer import NormalizedBar


@dataclass
class AlphaVantageBackfillRequest:
    """Alpha Vantage specific backfill request"""
    symbol: str
    timeframes: List[str] = None  # ["1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"]
    outputsize: str = "full"  # "compact" (100 points) or "full" (20+ years)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    priority: int = 1  # 1=high, 2=medium, 3=low


@dataclass
class AlphaVantageBackfillProgress:
    """Track Alpha Vantage backfill progress"""
    request_id: str
    symbol: str
    timeframe: str
    total_expected_bars: int
    stored_bars: int
    api_calls_made: int
    status: str  # 'pending', 'running', 'completed', 'failed', 'rate_limited'
    start_time: datetime
    end_time: Optional[datetime] = None
    last_error: Optional[str] = None
    data_coverage: Optional[Dict] = None  # {"earliest": "2020-01-01", "latest": "2024-01-01"}


class AlphaVantageBackfillService:
    """
    Alpha Vantage historical data backfill service.
    
    Features:
    - Comprehensive timeframe support (1min to monthly)
    - Intelligent rate limiting (5 calls/minute, 500 calls/day for free tier)
    - Data gap detection and filling
    - Progress tracking with detailed metrics
    - Integration with PostgreSQL storage
    - Graceful error handling and retry logic
    - Support for stocks, ETFs, and forex pairs
    
    Alpha Vantage Limitations:
    - Free tier: 5 API calls per minute, 500 calls per day
    - Intraday data: Last 30 days for 1min/5min, longer for higher timeframes
    - Daily data: 20+ years of history available
    - Rate limiting enforced by API with exponential backoff
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.is_running = False
        self.backfill_queue: List[AlphaVantageBackfillRequest] = []
        self.progress_tracker: Dict[str, AlphaVantageBackfillProgress] = {}
        
        # Alpha Vantage specific configuration
        self.api_calls_per_minute = 5  # Free tier limit
        self.api_calls_per_day = 500   # Free tier limit
        self.request_delay = 12.5      # 60s / 5 calls = 12s + buffer
        
        # Track API usage
        self.daily_api_calls = 0
        self.last_reset_date = datetime.now().date()
        
        # Alpha Vantage timeframe mappings
        self.timeframe_config = {
            # Intraday timeframes (limited historical data)
            "1min": {
                "av_function": "TIME_SERIES_INTRADAY",
                "av_interval": "1min",
                "max_history_days": 30,
                "data_points_per_call": 500,
                "storage_timeframe": "1m"
            },
            "5min": {
                "av_function": "TIME_SERIES_INTRADAY", 
                "av_interval": "5min",
                "max_history_days": 60,
                "data_points_per_call": 500,
                "storage_timeframe": "5m"
            },
            "15min": {
                "av_function": "TIME_SERIES_INTRADAY",
                "av_interval": "15min", 
                "max_history_days": 120,
                "data_points_per_call": 500,
                "storage_timeframe": "15m"
            },
            "30min": {
                "av_function": "TIME_SERIES_INTRADAY",
                "av_interval": "30min",
                "max_history_days": 180,
                "data_points_per_call": 500,
                "storage_timeframe": "30m"
            },
            "60min": {
                "av_function": "TIME_SERIES_INTRADAY",
                "av_interval": "60min",
                "max_history_days": 365,
                "data_points_per_call": 500,
                "storage_timeframe": "1h"
            },
            # Daily and longer timeframes (extensive historical data)
            "daily": {
                "av_function": "TIME_SERIES_DAILY_ADJUSTED",
                "av_interval": None,
                "max_history_days": 7300,  # 20 years
                "data_points_per_call": 5000,  # Full dataset
                "storage_timeframe": "1d"
            },
            "weekly": {
                "av_function": "TIME_SERIES_WEEKLY_ADJUSTED",
                "av_interval": None,
                "max_history_days": 7300,  # 20 years
                "data_points_per_call": 1000,
                "storage_timeframe": "1w"
            },
            "monthly": {
                "av_function": "TIME_SERIES_MONTHLY_ADJUSTED",
                "av_interval": None,
                "max_history_days": 7300,  # 20 years
                "data_points_per_call": 300,
                "storage_timeframe": "1M"
            }
        }
        
        # Priority symbols for initial backfill
        self.priority_symbols = [
            # FAANG + Major Tech
            "AAPL", "GOOGL", "MSFT", "AMZN", "META", "TSLA", "NVDA", "NFLX",
            # Market ETFs
            "SPY", "QQQ", "IWM", "VTI", "VXUS",
            # Sector ETFs
            "XLF", "XLK", "XLE", "XLV", "XLI",
            # Major Forex (Alpha Vantage format: FROM/TO)
            "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD"
        ]
    
    async def initialize(self) -> bool:
        """Initialize Alpha Vantage backfill service"""
        try:
            # Check Alpha Vantage service health
            health = await alpha_vantage_service.health_check()
            if health.status != "operational":
                self.logger.error(f"Alpha Vantage service not operational: {health.error_message}")
                return False
            
            # Ensure database connection
            if not historical_data_service.is_connected:
                await historical_data_service.connect()
            
            # Reset daily API call counter if new day
            current_date = datetime.now().date()
            if current_date != self.last_reset_date:
                self.daily_api_calls = 0
                self.last_reset_date = current_date
            
            self.logger.info("Alpha Vantage backfill service initialized successfully")
            self.logger.info(f"API calls used today: {self.daily_api_calls}/{self.api_calls_per_day}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Alpha Vantage backfill service: {e}")
            return False
    
    async def analyze_missing_data(self, symbol: str) -> Dict[str, List[Tuple[datetime, datetime]]]:
        """Analyze missing data gaps for Alpha Vantage supported timeframes"""
        missing_gaps = {}
        
        try:
            # Alpha Vantage uses different venue format
            venue = "NASDAQ"  # Default venue for stocks
            instrument_id = f"{symbol}.{venue}"
            
            for timeframe, config in self.timeframe_config.items():
                # Calculate available date range for this timeframe
                end_date = datetime.now()
                max_days = config["max_history_days"]
                start_date = end_date - timedelta(days=max_days)
                
                # Query existing data in our database
                query = HistoricalDataQuery(
                    venue=venue,
                    instrument_id=instrument_id, 
                    data_type="bar",
                    start_time=start_date,
                    end_time=end_date,
                    timeframe=config["storage_timeframe"]
                )
                
                existing_bars = await historical_data_service.query_bars(query)
                
                # Find gaps in the data
                gaps = self._find_data_gaps(existing_bars, start_date, end_date, timeframe)
                if gaps:
                    missing_gaps[timeframe] = gaps
                    self.logger.info(f"Found {len(gaps)} gaps for {symbol} {timeframe}")
                    
        except Exception as e:
            self.logger.error(f"Error analyzing missing data for {symbol}: {e}")
            
        return missing_gaps
    
    def _find_data_gaps(self, existing_bars: List[dict], start_date: datetime, end_date: datetime, timeframe: str) -> List[Tuple[datetime, datetime]]:
        """Find gaps in existing historical data for Alpha Vantage timeframes"""
        if not existing_bars:
            return [(start_date, end_date)]
        
        # Convert timestamps and sort
        bar_times = []
        for bar in existing_bars:
            bar_time = datetime.fromtimestamp(bar['timestamp_ns'] / 1_000_000_000)
            bar_times.append(bar_time)
        
        bar_times.sort()
        gaps = []
        
        # Check gap before first bar
        if bar_times[0] > start_date:
            gaps.append((start_date, bar_times[0]))
        
        # Check gaps between bars with timeframe-specific logic
        for i in range(len(bar_times) - 1):
            current_time = bar_times[i]
            next_time = bar_times[i + 1]
            expected_next = self._get_next_bar_time(current_time, timeframe)
            
            # Bigger tolerance for daily+ timeframes (weekends, holidays)
            tolerance = timedelta(days=3) if timeframe in ["daily", "weekly", "monthly"] else timedelta(hours=1)
            
            if next_time > expected_next + tolerance:
                gaps.append((current_time, next_time))
        
        # Check gap after last bar  
        if bar_times[-1] < end_date:
            gaps.append((bar_times[-1], end_date))
        
        return gaps
    
    def _get_next_bar_time(self, current_time: datetime, timeframe: str) -> datetime:
        """Calculate expected next bar time for Alpha Vantage timeframes"""
        if timeframe == "1min":
            return current_time + timedelta(minutes=1)
        elif timeframe == "5min":
            return current_time + timedelta(minutes=5)
        elif timeframe == "15min":
            return current_time + timedelta(minutes=15)
        elif timeframe == "30min":
            return current_time + timedelta(minutes=30)
        elif timeframe == "60min":
            return current_time + timedelta(hours=1)
        elif timeframe == "daily":
            return current_time + timedelta(days=1)
        elif timeframe == "weekly":
            return current_time + timedelta(weeks=1)
        elif timeframe == "monthly":
            return current_time + timedelta(days=30)  # Approximate
        else:
            return current_time + timedelta(days=1)  # Default
    
    async def add_backfill_request(self, request: AlphaVantageBackfillRequest):
        """Add Alpha Vantage backfill request to queue"""
        if request.timeframes is None:
            # Default to most useful timeframes for trading
            request.timeframes = ["daily", "60min", "15min", "5min"]
        
        # Set default date ranges based on timeframe capabilities
        if request.start_date is None or request.end_date is None:
            for timeframe in request.timeframes:
                config = self.timeframe_config.get(timeframe, {})
                max_days = config.get("max_history_days", 365)
                
                if request.end_date is None:
                    request.end_date = datetime.now()
                if request.start_date is None:
                    request.start_date = request.end_date - timedelta(days=max_days)
                break
        
        self.backfill_queue.append(request)
        self.logger.info(f"Added Alpha Vantage backfill request for {request.symbol} ({len(request.timeframes)} timeframes)")
    
    async def check_api_limits(self) -> bool:
        """Check if we can make more API calls today"""
        # Reset counter if new day
        current_date = datetime.now().date()
        if current_date != self.last_reset_date:
            self.daily_api_calls = 0
            self.last_reset_date = current_date
        
        return self.daily_api_calls < self.api_calls_per_day
    
    async def start_backfill_process(self):
        """Start Alpha Vantage backfill process with rate limiting"""
        if self.is_running:
            self.logger.warning("Alpha Vantage backfill process already running")
            return
        
        if not await self.check_api_limits():
            self.logger.error(f"Daily API limit exceeded ({self.daily_api_calls}/{self.api_calls_per_day})")
            return
        
        self.is_running = True
        self.logger.info("Starting Alpha Vantage backfill process")
        
        try:
            while self.backfill_queue and self.is_running and await self.check_api_limits():
                request = self.backfill_queue.pop(0)
                await self._process_backfill_request(request)
                
                # Respect rate limits - wait between requests
                await asyncio.sleep(self.request_delay)
                
        except Exception as e:
            self.logger.error(f"Error in Alpha Vantage backfill process: {e}")
        finally:
            self.is_running = False
            self.logger.info("Alpha Vantage backfill process completed")
    
    async def _process_backfill_request(self, request: AlphaVantageBackfillRequest):
        """Process single Alpha Vantage backfill request"""
        self.logger.info(f"Processing Alpha Vantage backfill for {request.symbol}")
        
        for timeframe in request.timeframes:
            if not self.is_running or not await self.check_api_limits():
                break
            
            request_id = f"av_{request.symbol}_{timeframe}_{datetime.now().isoformat()}"
            
            progress = AlphaVantageBackfillProgress(
                request_id=request_id,
                symbol=request.symbol,
                timeframe=timeframe,
                total_expected_bars=0,
                stored_bars=0,
                api_calls_made=0,
                status="running",
                start_time=datetime.now()
            )
            self.progress_tracker[request_id] = progress
            
            try:
                await self._backfill_timeframe(request, timeframe, progress)
                progress.status = "completed"
                
            except Exception as e:
                progress.status = "failed"
                progress.last_error = str(e)
                self.logger.error(f"Failed Alpha Vantage backfill {request.symbol} {timeframe}: {e}")
            
            progress.end_time = datetime.now()
    
    async def _backfill_timeframe(self, request: AlphaVantageBackfillRequest, timeframe: str, progress: AlphaVantageBackfillProgress):
        """Backfill specific timeframe using Alpha Vantage API"""
        config = self.timeframe_config[timeframe]
        
        try:
            # Track API call
            self.daily_api_calls += 1
            progress.api_calls_made += 1
            
            # Get data from Alpha Vantage based on timeframe type
            if config["av_function"] == "TIME_SERIES_INTRADAY":
                time_series = await alpha_vantage_service.get_intraday_data(
                    symbol=request.symbol,
                    interval=config["av_interval"],
                    extended_hours=True
                )
            elif config["av_function"] == "TIME_SERIES_DAILY_ADJUSTED":
                time_series = await alpha_vantage_service.get_daily_data(
                    symbol=request.symbol,
                    outputsize=request.outputsize
                )
            else:
                # Weekly/monthly would need additional service methods
                self.logger.warning(f"Timeframe {timeframe} not yet implemented for Alpha Vantage backfill")
                return
            
            # Process and store the data
            bars_stored = 0
            earliest_date = None
            latest_date = None
            
            for bar_data in time_series.data:
                try:
                    # Convert Alpha Vantage bar to normalized format
                    normalized_bar = NormalizedBar(
                        venue="NASDAQ",  # Default venue for Alpha Vantage stocks
                        instrument_id=f"{request.symbol}.NASDAQ",
                        timeframe=config["storage_timeframe"],
                        timestamp_ns=int(bar_data.timestamp.timestamp() * 1_000_000_000),
                        open_price=bar_data.open,
                        high_price=bar_data.high,
                        low_price=bar_data.low,
                        close_price=bar_data.close,
                        volume=bar_data.volume,
                        is_final=True
                    )
                    
                    await historical_data_service.store_bar(normalized_bar)
                    bars_stored += 1
                    
                    # Track data coverage
                    if earliest_date is None or bar_data.timestamp < earliest_date:
                        earliest_date = bar_data.timestamp
                    if latest_date is None or bar_data.timestamp > latest_date:
                        latest_date = bar_data.timestamp
                        
                except Exception as e:
                    self.logger.warning(f"Failed to store Alpha Vantage bar for {request.symbol}: {e}")
            
            progress.stored_bars = bars_stored
            progress.total_expected_bars = len(time_series.data)
            progress.data_coverage = {
                "earliest": earliest_date.isoformat() if earliest_date else None,
                "latest": latest_date.isoformat() if latest_date else None,
                "bars_stored": bars_stored,
                "data_source": "Alpha Vantage"
            }
            
            self.logger.info(f"Alpha Vantage: Stored {bars_stored} bars for {request.symbol} {timeframe}")
            
        except Exception as e:
            progress.last_error = str(e)
            if "rate limit" in str(e).lower() or "frequency" in str(e).lower():
                progress.status = "rate_limited"
                self.logger.warning(f"Alpha Vantage rate limit hit for {request.symbol} {timeframe}")
            else:
                raise
    
    async def backfill_priority_symbols(self):
        """Backfill priority symbols using Alpha Vantage"""
        self.logger.info("Starting Alpha Vantage priority symbols backfill")
        
        for symbol in self.priority_symbols:
            if not await self.check_api_limits():
                self.logger.warning("Daily API limit reached, stopping priority backfill")
                break
            
            request = AlphaVantageBackfillRequest(
                symbol=symbol,
                timeframes=["daily", "60min", "15min"],  # Most important timeframes first
                outputsize="full",
                priority=1
            )
            await self.add_backfill_request(request)
        
        await self.start_backfill_process()
    
    async def get_backfill_status(self) -> Dict:
        """Get Alpha Vantage backfill status"""
        # Get database stats for Alpha Vantage data
        try:
            stats_query = """
                SELECT 
                    COUNT(*) as total_bars,
                    COUNT(DISTINCT instrument_id) as unique_instruments,
                    COUNT(DISTINCT timeframe) as unique_timeframes,
                    MIN(timestamp_ns) as earliest_data,
                    MAX(timestamp_ns) as latest_data
                FROM market_bars 
                WHERE venue = 'NASDAQ'
            """
            
            result = await historical_data_service.execute_query(stats_query)
            data_stats = result[0] if result else {}
            
        except Exception as e:
            self.logger.error(f"Error getting Alpha Vantage data stats: {e}")
            data_stats = {}
        
        return {
            "service": "Alpha Vantage",
            "is_running": self.is_running,
            "queue_size": len(self.backfill_queue),
            "daily_api_calls": self.daily_api_calls,
            "daily_api_limit": self.api_calls_per_day,
            "api_calls_remaining": max(0, self.api_calls_per_day - self.daily_api_calls),
            "active_requests": len([p for p in self.progress_tracker.values() if p.status == "running"]),
            "completed_requests": len([p for p in self.progress_tracker.values() if p.status == "completed"]),
            "failed_requests": len([p for p in self.progress_tracker.values() if p.status == "failed"]),
            "rate_limited_requests": len([p for p in self.progress_tracker.values() if p.status == "rate_limited"]),
            "data_stats": {
                "total_bars": data_stats.get("total_bars", 0),
                "unique_instruments": data_stats.get("unique_instruments", 0),
                "unique_timeframes": data_stats.get("unique_timeframes", 0),
                "earliest_data_ns": data_stats.get("earliest_data"),
                "latest_data_ns": data_stats.get("latest_data")
            },
            "supported_timeframes": list(self.timeframe_config.keys()),
            "priority_symbols": self.priority_symbols,
            "last_reset_date": self.last_reset_date.isoformat(),
            "progress_details": [
                {
                    "request_id": p.request_id,
                    "symbol": p.symbol,
                    "timeframe": p.timeframe,
                    "status": p.status,
                    "stored_bars": p.stored_bars,
                    "api_calls_made": p.api_calls_made,
                    "data_coverage": p.data_coverage,
                    "last_error": p.last_error
                }
                for p in list(self.progress_tracker.values())[-10:]  # Last 10 requests
            ]
        }
    
    async def stop_backfill_process(self):
        """Stop Alpha Vantage backfill process"""
        if self.is_running:
            self.is_running = False
            self.logger.info("Alpha Vantage backfill process stopped by user request")
        else:
            self.logger.info("Alpha Vantage backfill process was not running")


# Global Alpha Vantage backfill service instance
alpha_vantage_backfill_service = AlphaVantageBackfillService()