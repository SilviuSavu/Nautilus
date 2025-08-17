"""
Historical Data Backfill Service
Comprehensive system to pull missing historical data from IB Gateway into PostgreSQL.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json

from ib_gateway_client import get_ib_gateway_client
from historical_data_service import historical_data_service, HistoricalDataQuery
from data_normalizer import NormalizedBar
from enums import Venue


@dataclass
class BackfillRequest:
    """Data backfill request specification"""
    symbol: str
    sec_type: str = "STK"
    exchange: str = "SMART"
    currency: str = "USD"
    timeframes: List[str] = None
    start_date: datetime = None
    end_date: datetime = None
    priority: int = 1  # 1=high, 2=medium, 3=low


@dataclass
class BackfillProgress:
    """Track backfill operation progress"""
    request_id: str
    symbol: str
    timeframe: str
    total_periods: int
    completed_periods: int
    success_count: int
    error_count: int
    status: str  # 'pending', 'running', 'completed', 'failed'
    start_time: datetime
    end_time: Optional[datetime] = None
    last_error: Optional[str] = None


class DataBackfillService:
    """
    Service to backfill historical data from IB Gateway to PostgreSQL.
    Handles missing data detection, batch processing, and error recovery.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.ib_client = None
        self.is_running = False
        self.backfill_queue: List[BackfillRequest] = []
        self.progress_tracker: Dict[str, BackfillProgress] = {}
        
        # Rate limiting configuration
        self.max_requests_per_minute = 50  # IB Gateway limit
        self.request_delay = 1.5  # seconds between requests
        self.batch_size = 100  # bars per request
        
        # Standard timeframes with IB Gateway mappings
        self.timeframe_config = {
            "1m": {"ib_size": "1 min", "max_duration_days": 10},
            "5m": {"ib_size": "5 mins", "max_duration_days": 60},
            "15m": {"ib_size": "15 mins", "max_duration_days": 120},
            "30m": {"ib_size": "30 mins", "max_duration_days": 240},
            "1h": {"ib_size": "1 hour", "max_duration_days": 365},
            "2h": {"ib_size": "2 hours", "max_duration_days": 730},
            "4h": {"ib_size": "4 hours", "max_duration_days": 1095},
            "1d": {"ib_size": "1 day", "max_duration_days": 3650},
            "1w": {"ib_size": "1 week", "max_duration_days": 7300},
            "1M": {"ib_size": "1 month", "max_duration_days": 18250}
        }
        
        # Popular instruments to prioritize
        self.priority_instruments = [
            # Major US Stocks
            {"symbol": "AAPL", "sec_type": "STK", "exchange": "SMART", "currency": "USD"},
            {"symbol": "GOOGL", "sec_type": "STK", "exchange": "SMART", "currency": "USD"},
            {"symbol": "MSFT", "sec_type": "STK", "exchange": "SMART", "currency": "USD"},
            {"symbol": "AMZN", "sec_type": "STK", "exchange": "SMART", "currency": "USD"},
            {"symbol": "TSLA", "sec_type": "STK", "exchange": "SMART", "currency": "USD"},
            {"symbol": "NVDA", "sec_type": "STK", "exchange": "SMART", "currency": "USD"},
            {"symbol": "META", "sec_type": "STK", "exchange": "SMART", "currency": "USD"},
            {"symbol": "NFLX", "sec_type": "STK", "exchange": "SMART", "currency": "USD"},
            
            # Major Forex Pairs
            {"symbol": "EUR", "sec_type": "CASH", "exchange": "IDEALPRO", "currency": "USD"},
            {"symbol": "GBP", "sec_type": "CASH", "exchange": "IDEALPRO", "currency": "USD"},
            {"symbol": "JPY", "sec_type": "CASH", "exchange": "IDEALPRO", "currency": "USD"},
            {"symbol": "AUD", "sec_type": "CASH", "exchange": "IDEALPRO", "currency": "USD"},
            
            # Major Indices ETFs
            {"symbol": "SPY", "sec_type": "STK", "exchange": "SMART", "currency": "USD"},
            {"symbol": "QQQ", "sec_type": "STK", "exchange": "SMART", "currency": "USD"},
            {"symbol": "IWM", "sec_type": "STK", "exchange": "SMART", "currency": "USD"},
        ]
    
    async def initialize(self):
        """Initialize the backfill service"""
        try:
            # Connect to PostgreSQL
            if not historical_data_service.is_connected:
                await historical_data_service.connect()
            
            # Get IB Gateway client
            self.ib_client = get_ib_gateway_client()
            
            self.logger.info("Data backfill service initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize backfill service: {e}")
            return False
    
    async def analyze_missing_data(self, symbol: str, sec_type: str = "STK", 
                                 exchange: str = "SMART", currency: str = "USD") -> Dict[str, List[Tuple[datetime, datetime]]]:
        """Analyze missing data gaps for a given instrument"""
        missing_gaps = {}
        
        try:
            venue = exchange
            instrument_id = f"{symbol}.{venue}"
            
            for timeframe in self.timeframe_config.keys():
                # Calculate expected date range
                config = self.timeframe_config[timeframe]
                end_date = datetime.now()
                start_date = end_date - timedelta(days=config["max_duration_days"])
                
                # Query existing data
                query = HistoricalDataQuery(
                    venue=venue,
                    instrument_id=instrument_id,
                    data_type="bar",
                    start_time=start_date,
                    end_time=end_date,
                    timeframe=timeframe
                )
                
                existing_bars = await historical_data_service.query_bars(query)
                
                # Find gaps in the data
                gaps = self._find_data_gaps(existing_bars, start_date, end_date, timeframe)
                if gaps:
                    missing_gaps[timeframe] = gaps
                    
        except Exception as e:
            self.logger.error(f"Error analyzing missing data for {symbol}: {e}")
            
        return missing_gaps
    
    def _find_data_gaps(self, existing_bars: List[Dict], start_date: datetime, 
                       end_date: datetime, timeframe: str) -> List[Tuple[datetime, datetime]]:
        """Find gaps in existing historical data"""
        if not existing_bars:
            # No data at all - entire range is missing
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
        
        # Check gaps between bars
        for i in range(len(bar_times) - 1):
            current_time = bar_times[i]
            next_time = bar_times[i + 1]
            
            # Calculate expected next bar time based on timeframe
            expected_next = self._get_next_bar_time(current_time, timeframe)
            
            # If there's a significant gap, record it
            if next_time > expected_next + timedelta(minutes=5):  # 5 min tolerance
                gaps.append((current_time, next_time))
        
        # Check gap after last bar
        if bar_times[-1] < end_date:
            gaps.append((bar_times[-1], end_date))
        
        return gaps
    
    def _get_next_bar_time(self, current_time: datetime, timeframe: str) -> datetime:
        """Calculate expected next bar time based on timeframe"""
        if timeframe == "1m":
            return current_time + timedelta(minutes=1)
        elif timeframe == "5m":
            return current_time + timedelta(minutes=5)
        elif timeframe == "15m":
            return current_time + timedelta(minutes=15)
        elif timeframe == "30m":
            return current_time + timedelta(minutes=30)
        elif timeframe == "1h":
            return current_time + timedelta(hours=1)
        elif timeframe == "2h":
            return current_time + timedelta(hours=2)
        elif timeframe == "4h":
            return current_time + timedelta(hours=4)
        elif timeframe == "1d":
            return current_time + timedelta(days=1)
        elif timeframe == "1w":
            return current_time + timedelta(weeks=1)
        elif timeframe == "1M":
            # Approximate - add 30 days
            return current_time + timedelta(days=30)
        else:
            return current_time + timedelta(hours=1)  # Default
    
    async def add_backfill_request(self, request: BackfillRequest):
        """Add a backfill request to the queue"""
        if request.timeframes is None:
            request.timeframes = list(self.timeframe_config.keys())
        
        if request.start_date is None:
            # Default to maximum available history for first timeframe
            max_days = max(self.timeframe_config[tf]["max_duration_days"] for tf in request.timeframes)
            request.start_date = datetime.now() - timedelta(days=min(max_days, 365))  # Cap at 1 year
        
        if request.end_date is None:
            request.end_date = datetime.now()
        
        self.backfill_queue.append(request)
        self.logger.info(f"Added backfill request for {request.symbol} ({len(request.timeframes)} timeframes)")
    
    async def start_backfill_process(self):
        """Start the backfill process"""
        if self.is_running:
            self.logger.warning("Backfill process is already running")
            return
        
        if not self.ib_client or not self.ib_client.is_connected():
            self.logger.error("IB Gateway not connected - cannot start backfill")
            return
        
        self.is_running = True
        self.logger.info("Starting historical data backfill process")
        
        try:
            while self.backfill_queue and self.is_running:
                request = self.backfill_queue.pop(0)
                await self._process_backfill_request(request)
                
                # Rate limiting delay
                await asyncio.sleep(self.request_delay)
                
        except Exception as e:
            self.logger.error(f"Error in backfill process: {e}")
        finally:
            self.is_running = False
            self.logger.info("Backfill process completed")
    
    async def _process_backfill_request(self, request: BackfillRequest):
        """Process a single backfill request"""
        self.logger.info(f"Processing backfill for {request.symbol}")
        
        for timeframe in request.timeframes:
            request_id = f"{request.symbol}_{timeframe}_{datetime.now().isoformat()}"
            
            progress = BackfillProgress(
                request_id=request_id,
                symbol=request.symbol,
                timeframe=timeframe,
                total_periods=0,
                completed_periods=0,
                success_count=0,
                error_count=0,
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
                self.logger.error(f"Failed to backfill {request.symbol} {timeframe}: {e}")
            
            progress.end_time = datetime.now()
    
    async def _backfill_timeframe(self, request: BackfillRequest, timeframe: str, progress: BackfillProgress):
        """Backfill data for a specific timeframe"""
        config = self.timeframe_config[timeframe]
        
        # Break down the time range into manageable chunks
        max_duration_days = min(config["max_duration_days"], 30)  # Limit chunks to 30 days
        current_end = request.end_date
        
        while current_end > request.start_date:
            current_start = max(request.start_date, current_end - timedelta(days=max_duration_days))
            
            try:
                # Request data from IB Gateway
                duration_str = f"{(current_end - current_start).days} D"
                
                historical_data = await self.ib_client.request_historical_data(
                    symbol=request.symbol,
                    sec_type=request.sec_type,
                    exchange=request.exchange,
                    currency=request.currency,
                    duration=duration_str,
                    bar_size=config["ib_size"],
                    what_to_show="TRADES"
                )
                
                # Store data in PostgreSQL
                bars_stored = 0
                for bar_data in historical_data.get('bars', []):
                    try:
                        # Convert to normalized format - handle IB Gateway timestamp format
                        time_str = bar_data['time']
                        
                        # IB Gateway returns timestamps like "20250804  15:30:00" 
                        # Convert to proper datetime object
                        if '  ' in time_str:  # Intraday format: "20250804  15:30:00"
                            bar_time = datetime.strptime(time_str, "%Y%m%d  %H:%M:%S")
                        elif len(time_str) == 8:  # Daily format: "20250804"
                            bar_time = datetime.strptime(time_str, "%Y%m%d")
                        else:  # Try ISO format as fallback
                            bar_time = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                        
                        normalized_bar = NormalizedBar(
                            venue=request.exchange,
                            instrument_id=f"{request.symbol}.{request.exchange}",
                            timeframe=timeframe,
                            timestamp_ns=int(bar_time.timestamp() * 1_000_000_000),
                            open_price=float(bar_data['open']),
                            high_price=float(bar_data['high']),
                            low_price=float(bar_data['low']),
                            close_price=float(bar_data['close']),
                            volume=float(bar_data['volume']),
                            is_final=True
                        )
                        
                        await historical_data_service.store_bar(normalized_bar)
                        bars_stored += 1
                        
                    except Exception as e:
                        progress.error_count += 1
                        self.logger.warning(f"Failed to store bar for {request.symbol}: {e}")
                
                progress.success_count += bars_stored
                progress.completed_periods += 1
                
                self.logger.info(f"Stored {bars_stored} bars for {request.symbol} {timeframe} ({current_start} to {current_end})")
                
                # Rate limiting
                await asyncio.sleep(self.request_delay)
                
            except Exception as e:
                progress.error_count += 1
                self.logger.error(f"Error fetching data for {request.symbol} {timeframe}: {e}")
            
            current_end = current_start
    
    async def backfill_priority_instruments(self):
        """Backfill data for priority instruments"""
        self.logger.info("Starting priority instruments backfill")
        
        for instrument in self.priority_instruments:
            request = BackfillRequest(
                symbol=instrument["symbol"],
                sec_type=instrument["sec_type"],
                exchange=instrument["exchange"],
                currency=instrument["currency"],
                timeframes=["1d", "1h", "15m", "5m"],  # Start with most important timeframes
                priority=1
            )
            await self.add_backfill_request(request)
        
        await self.start_backfill_process()
    
    async def get_database_size_gb(self) -> float:
        """Get total size of historical data in PostgreSQL database in GB"""
        try:
            # Get database size using PostgreSQL system functions
            size_query = """
                SELECT 
                    ROUND(
                        (pg_database_size(current_database()) / (1024.0 * 1024.0 * 1024.0))::numeric, 
                        3
                    ) as size_gb
            """
            
            result = await historical_data_service.execute_query(size_query)
            if result and len(result) > 0:
                return float(result[0].get('size_gb', 0.0))
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error calculating database size: {e}")
            return 0.0
    
    async def get_historical_data_stats(self) -> Dict:
        """Get statistics about stored historical data"""
        try:
            # Query to get counts and size info by timeframe and instrument
            stats_query = """
                SELECT 
                    COUNT(*) as total_bars,
                    COUNT(DISTINCT instrument_id) as unique_instruments,
                    COUNT(DISTINCT timeframe) as unique_timeframes,
                    MIN(timestamp_ns) as earliest_data,
                    MAX(timestamp_ns) as latest_data
                FROM market_bars
            """
            
            result = await historical_data_service.execute_query(stats_query)
            if result and len(result) > 0:
                stats = result[0]
                return {
                    "total_bars": stats.get('total_bars', 0),
                    "unique_instruments": stats.get('unique_instruments', 0),
                    "unique_timeframes": stats.get('unique_timeframes', 0),
                    "earliest_data_ns": stats.get('earliest_data'),
                    "latest_data_ns": stats.get('latest_data')
                }
            else:
                return {
                    "total_bars": 0,
                    "unique_instruments": 0, 
                    "unique_timeframes": 0,
                    "earliest_data_ns": None,
                    "latest_data_ns": None
                }
                
        except Exception as e:
            self.logger.error(f"Error getting historical data stats: {e}")
            return {
                "total_bars": 0,
                "unique_instruments": 0,
                "unique_timeframes": 0,
                "earliest_data_ns": None,
                "latest_data_ns": None
            }

    async def get_backfill_status(self) -> Dict:
        """Get current backfill status"""
        # Get database size and stats
        database_size_gb = await self.get_database_size_gb()
        data_stats = await self.get_historical_data_stats()
        
        return {
            "is_running": self.is_running,
            "queue_size": len(self.backfill_queue),
            "active_requests": len([p for p in self.progress_tracker.values() if p.status == "running"]),
            "completed_requests": len([p for p in self.progress_tracker.values() if p.status == "completed"]),
            "failed_requests": len([p for p in self.progress_tracker.values() if p.status == "failed"]),
            "database_size_gb": database_size_gb,
            "total_bars": data_stats["total_bars"],
            "unique_instruments": data_stats["unique_instruments"],
            "unique_timeframes": data_stats["unique_timeframes"],
            "earliest_data_ns": data_stats["earliest_data_ns"],
            "latest_data_ns": data_stats["latest_data_ns"],
            "progress_details": [
                {
                    "request_id": p.request_id,
                    "symbol": p.symbol,
                    "timeframe": p.timeframe,
                    "status": p.status,
                    "success_count": p.success_count,
                    "error_count": p.error_count,
                    "last_error": p.last_error
                }
                for p in list(self.progress_tracker.values())[-10:]  # Last 10 requests
            ]
        }
    
    async def stop_backfill_process(self):
        """Stop the backfill process"""
        self.is_running = False
        self.logger.info("Backfill process stopped")


# Global service instance
backfill_service = DataBackfillService()