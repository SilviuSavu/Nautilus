"""
Data Normalization and Processing Pipeline
Provides standardized data processing for multiple venue formats with validation and quality checks.
"""

import logging
import time
from decimal import Decimal, InvalidOperation
from datetime import datetime
from typing import Any, Union
from dataclasses import dataclass, asdict
from enum import Enum

from enums import Venue, DataType
# Avoiding circular import by redefining the class locally
from dataclasses import dataclass as local_dataclass

@local_dataclass
class NormalizedMarketData:
    """Normalized market data structure"""
    venue: str
    instrument_id: str
    data_type: str
    timestamp: int
    data: dict[str, Any]
    raw_data: dict[str, Any]


class ValidationError(Exception):
    """Data validation error"""
    pass


class NormalizationError(Exception):
    """Data normalization error"""
    pass


@dataclass
class DataQualityMetrics:
    """Data quality metrics for monitoring"""
    total_messages: int = 0
    valid_messages: int = 0
    invalid_messages: int = 0
    normalization_errors: int = 0
    validation_errors: int = 0
    duplicate_messages: int = 0
    out_of_order_messages: int = 0
    last_update: datetime | None = None


@dataclass
class NormalizedTick:
    """Standardized tick data structure"""
    venue: str
    instrument_id: str
    price: Decimal
    size: Decimal
    timestamp_ns: int
    side: str | None = None  # 'buy', 'sell', or None
    trade_id: str | None = None
    sequence: int | None = None


@dataclass
class NormalizedQuote:
    """Standardized quote data structure"""
    venue: str
    instrument_id: str
    bid_price: Decimal
    ask_price: Decimal
    bid_size: Decimal
    ask_size: Decimal
    timestamp_ns: int
    sequence: int | None = None


@dataclass
class NormalizedBar:
    """Standardized OHLCV bar data structure"""
    venue: str
    instrument_id: str
    open_price: Decimal
    high_price: Decimal
    low_price: Decimal
    close_price: Decimal
    volume: Decimal
    timestamp_ns: int
    timeframe: str
    is_final: bool = True


class DataNormalizer:
    """
    Data normalization and processing pipeline that standardizes market data
    from multiple venues into consistent formats with validation and quality monitoring.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._quality_metrics: dict[Venue, DataQualityMetrics] = {}
        self._last_timestamps: dict[str, int] = {}  # instrument_id -> last_timestamp
        self._sequence_numbers: dict[str, int] = {}  # instrument_id -> last_sequence
        self._price_validators: dict[Venue, callable] = {}
        self._size_validators: dict[Venue, callable] = {}
        self._setup_validators()
        
    def _setup_validators(self) -> None:
        """Setup venue-specific validators"""
        # Price validators
        self._price_validators = {
            Venue.BINANCE: lambda p: 0 < p < 1000000, Venue.COINBASE: lambda p: 0 < p < 1000000, Venue.KRAKEN: lambda p: 0 < p < 1000000, Venue.BYBIT: lambda p: 0 < p < 1000000, Venue.OKX: lambda p: 0 < p < 1000000, }
        
        # Size validators  
        self._size_validators = {
            Venue.BINANCE: lambda s: 0 < s < 1000000, Venue.COINBASE: lambda s: 0 < s < 1000000, Venue.KRAKEN: lambda s: 0 < s < 1000000, Venue.BYBIT: lambda s: 0 < s < 1000000, Venue.OKX: lambda s: 0 < s < 1000000, }
        
    def normalize_market_data(self, raw_data: NormalizedMarketData) -> NormalizedTick | NormalizedQuote | NormalizedBar:
        """Main normalization entry point"""
        venue = Venue(raw_data.venue)
        data_type = DataType(raw_data.data_type)
        
        # Initialize metrics if needed
        if venue not in self._quality_metrics:
            self._quality_metrics[venue] = DataQualityMetrics()
            
        metrics = self._quality_metrics[venue]
        metrics.total_messages += 1
        metrics.last_update = datetime.now()
        
        try:
            # Route to appropriate normalizer
            if data_type == DataType.TICK:
                return self._normalize_tick(raw_data, venue)
            elif data_type == DataType.QUOTE:
                return self._normalize_quote(raw_data, venue)
            elif data_type == DataType.BAR:
                return self._normalize_bar(raw_data, venue)
            elif data_type == DataType.TRADE:
                # Treat trades as ticks
                raw_data.data_type = DataType.TICK.value
                return self._normalize_tick(raw_data, venue)
            else:
                raise NormalizationError(f"Unsupported data type: {data_type}")
                
        except (ValidationError, NormalizationError) as e:
            metrics.invalid_messages += 1
            if isinstance(e, ValidationError):
                metrics.validation_errors += 1
            else:
                metrics.normalization_errors += 1
            self.logger.error(f"Normalization failed for {venue.value} {data_type.value}: {e}")
            raise
            
    def _normalize_tick(self, data: NormalizedMarketData, venue: Venue) -> NormalizedTick:
        """Normalize tick data"""
        try:
            # Extract common fields
            price = self._extract_price(data.data, venue)
            size = self._extract_size(data.data, venue)
            timestamp_ns = self._normalize_timestamp(data.timestamp)
            
            # Validate data
            self._validate_price(price, venue)
            self._validate_size(size, venue)
            self._validate_timestamp(timestamp_ns, data.instrument_id)
            
            # Extract optional fields
            side = self._extract_side(data.data, venue)
            trade_id = data.data.get("trade_id") or data.data.get("id")
            sequence = self._extract_sequence(data.data, venue)
            
            normalized_tick = NormalizedTick(
                venue=venue.value, instrument_id=data.instrument_id, price=price, size=size, timestamp_ns=timestamp_ns, side=side, trade_id=str(trade_id) if trade_id else None, sequence=sequence
            )
            
            self._quality_metrics[venue].valid_messages += 1
            return normalized_tick
            
        except Exception as e:
            raise NormalizationError(f"Failed to normalize tick data: {e}")
            
    def _normalize_quote(self, data: NormalizedMarketData, venue: Venue) -> NormalizedQuote:
        """Normalize quote data"""
        try:
            # Extract bid/ask prices and sizes
            bid_price = self._extract_bid_price(data.data, venue)
            ask_price = self._extract_ask_price(data.data, venue)
            bid_size = self._extract_bid_size(data.data, venue)
            ask_size = self._extract_ask_size(data.data, venue)
            timestamp_ns = self._normalize_timestamp(data.timestamp)
            
            # Validate data
            self._validate_price(bid_price, venue)
            self._validate_price(ask_price, venue)
            self._validate_size(bid_size, venue)
            self._validate_size(ask_size, venue)
            self._validate_timestamp(timestamp_ns, data.instrument_id)
            
            # Validate spread
            if ask_price <= bid_price:
                raise ValidationError(f"Invalid spread: bid={bid_price}, ask={ask_price}")
                
            sequence = self._extract_sequence(data.data, venue)
            
            normalized_quote = NormalizedQuote(
                venue=venue.value, instrument_id=data.instrument_id, bid_price=bid_price, ask_price=ask_price, bid_size=bid_size, ask_size=ask_size, timestamp_ns=timestamp_ns, sequence=sequence
            )
            
            self._quality_metrics[venue].valid_messages += 1
            return normalized_quote
            
        except Exception as e:
            raise NormalizationError(f"Failed to normalize quote data: {e}")
            
    def _normalize_bar(self, data: NormalizedMarketData, venue: Venue) -> NormalizedBar:
        """Normalize OHLCV bar data"""
        try:
            # Extract OHLCV data
            open_price = self._extract_decimal(data.data, ["open", "open_price"], venue)
            high_price = self._extract_decimal(data.data, ["high", "high_price"], venue)
            low_price = self._extract_decimal(data.data, ["low", "low_price"], venue)
            close_price = self._extract_decimal(data.data, ["close", "close_price"], venue)
            volume = self._extract_decimal(data.data, ["volume", "vol"], venue)
            timestamp_ns = self._normalize_timestamp(data.timestamp)
            
            # Validate OHLC relationships
            if not (low_price <= open_price <= high_price and 
                   low_price <= close_price <= high_price):
                raise ValidationError(f"Invalid OHLC relationship: O={open_price}, H={high_price}, L={low_price}, C={close_price}")
                
            # Validate all prices
            for price in [open_price, high_price, low_price, close_price]:
                self._validate_price(price, venue)
                
            self._validate_size(volume, venue)
            self._validate_timestamp(timestamp_ns, data.instrument_id)
            
            timeframe = data.data.get("timeframe", "1m")
            is_final = data.data.get("is_final", True)
            
            normalized_bar = NormalizedBar(
                venue=venue.value, instrument_id=data.instrument_id, open_price=open_price, high_price=high_price, low_price=low_price, close_price=close_price, volume=volume, timestamp_ns=timestamp_ns, timeframe=timeframe, is_final=is_final
            )
            
            self._quality_metrics[venue].valid_messages += 1
            return normalized_bar
            
        except Exception as e:
            raise NormalizationError(f"Failed to normalize bar data: {e}")
            
    def _extract_price(self, data: dict[str, Any], venue: Venue) -> Decimal:
        """Extract price from data based on venue format"""
        price_fields = ["price", "px", "p"]
        return self._extract_decimal(data, price_fields, venue)
        
    def _extract_size(self, data: dict[str, Any], venue: Venue) -> Decimal:
        """Extract size/quantity from data based on venue format"""
        size_fields = ["size", "qty", "quantity", "sz", "q"]
        return self._extract_decimal(data, size_fields, venue)
        
    def _extract_bid_price(self, data: dict[str, Any], venue: Venue) -> Decimal:
        """Extract bid price"""
        bid_fields = ["bid", "bid_price", "bidPx", "best_bid"]
        return self._extract_decimal(data, bid_fields, venue)
        
    def _extract_ask_price(self, data: dict[str, Any], venue: Venue) -> Decimal:
        """Extract ask price"""
        ask_fields = ["ask", "ask_price", "askPx", "best_ask"]
        return self._extract_decimal(data, ask_fields, venue)
        
    def _extract_bid_size(self, data: dict[str, Any], venue: Venue) -> Decimal:
        """Extract bid size"""
        bid_size_fields = ["bid_size", "bidSz", "bid_qty"]
        return self._extract_decimal(data, bid_size_fields, venue)
        
    def _extract_ask_size(self, data: dict[str, Any], venue: Venue) -> Decimal:
        """Extract ask size"""
        ask_size_fields = ["ask_size", "askSz", "ask_qty"]
        return self._extract_decimal(data, ask_size_fields, venue)
        
    def _extract_decimal(self, data: dict[str, Any], field_names: list[str], venue: Venue) -> Decimal:
        """Extract decimal value from multiple possible field names"""
        for field in field_names:
            if field in data and data[field] is not None:
                try:
                    return Decimal(str(data[field]))
                except (InvalidOperation, ValueError) as e:
                    continue
                    
        raise ValidationError(f"Could not extract decimal from fields {field_names} in {venue.value} data")
        
    def _extract_side(self, data: dict[str, Any], venue: Venue) -> str | None:
        """Extract trade side"""
        side = data.get("side")
        if side:
            side = side.lower()
            if side in ["buy", "sell", "bid", "ask"]:
                return "buy" if side in ["buy", "bid"] else "sell"
        return None
        
    def _extract_sequence(self, data: dict[str, Any], venue: Venue) -> int | None:
        """Extract sequence number"""
        seq_fields = ["sequence", "seq", "u", "lastUpdateId"]
        for field in seq_fields:
            if field in data:
                try:
                    return int(data[field])
                except (ValueError, TypeError):
                    continue
        return None
        
    def _normalize_timestamp(self, timestamp: int) -> int:
        """Normalize timestamp to nanoseconds"""
        # Convert various timestamp formats to nanoseconds
        if timestamp > 1e18:  # Already in nanoseconds
            return timestamp
        elif timestamp > 1e15:  # Microseconds
            return timestamp * 1000
        elif timestamp > 1e12:  # Milliseconds
            return timestamp * 1000000
        else:  # Seconds
            return timestamp * 1000000000
            
    def _validate_price(self, price: Decimal, venue: Venue) -> None:
        """Validate price value"""
        if price <= 0:
            raise ValidationError(f"Invalid price: {price}")
            
        if venue in self._price_validators:
            if not self._price_validators[venue](float(price)):
                raise ValidationError(f"Price {price} out of range for {venue.value}")
                
    def _validate_size(self, size: Decimal, venue: Venue) -> None:
        """Validate size/quantity value"""
        if size <= 0:
            raise ValidationError(f"Invalid size: {size}")
            
        if venue in self._size_validators:
            if not self._size_validators[venue](float(size)):
                raise ValidationError(f"Size {size} out of range for {venue.value}")
                
    def _validate_timestamp(self, timestamp_ns: int, instrument_id: str) -> None:
        """Validate timestamp and check for out-of-order data"""
        current_time_ns = time.time_ns()
        
        # Check if timestamp is too far in the future (more than 1 minute)
        if timestamp_ns > current_time_ns + 60_000_000_000:
            raise ValidationError(f"Timestamp too far in future: {timestamp_ns}")
            
        # Check if timestamp is too old (more than 1 day)
        if timestamp_ns < current_time_ns - 86400_000_000_000:
            raise ValidationError(f"Timestamp too old: {timestamp_ns}")
            
        # Check for out-of-order data
        last_timestamp = self._last_timestamps.get(instrument_id, 0)
        if timestamp_ns < last_timestamp:
            # Allow some tolerance for out-of-order data (1 second)
            if last_timestamp - timestamp_ns > 1_000_000_000:
                # Track by the actual venue instead of hardcoded BINANCE
                venue_key = next((v for v in self._quality_metrics.keys()), None)
                if venue_key:
                    self._quality_metrics[venue_key].out_of_order_messages += 1
                self.logger.warning(f"Out-of-order data for {instrument_id}: {timestamp_ns} < {last_timestamp}")
                
        self._last_timestamps[instrument_id] = max(timestamp_ns, last_timestamp)
        
    def get_quality_metrics(self, venue: Venue | None = None) -> dict[str, DataQualityMetrics]:
        """Get data quality metrics"""
        if venue:
            return {venue.value: self._quality_metrics.get(venue, DataQualityMetrics())}
        return {v.value: metrics for v, metrics in self._quality_metrics.items()}
        
    def reset_metrics(self, venue: Venue | None = None) -> None:
        """Reset quality metrics"""
        if venue:
            self._quality_metrics[venue] = DataQualityMetrics()
        else:
            self._quality_metrics.clear()


# Global normalizer instance
data_normalizer = DataNormalizer()