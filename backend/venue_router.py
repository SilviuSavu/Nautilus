"""
Venue Router Service
Provides intelligent routing of market data requests to appropriate venue connections
with load balancing and failover capabilities.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from enum import Enum

from enums import Venue, DataType
from market_data_service import MarketDataSubscription


class VenueStatus(Enum):
    """Venue connection status"""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"


@dataclass
class VenueConnection:
    """Venue connection information"""
    venue: Venue
    status: VenueStatus
    last_heartbeat: datetime
    error_count: int = 0
    rate_limit_reset: Optional[datetime] = None
    supported_instruments: Set[str] = None
    supported_data_types: Set[DataType] = None
    connection_priority: int = 1  # Lower number = higher priority


@dataclass
class RoutingRule:
    """Routing rule for directing data requests"""
    venue: Venue
    instrument_pattern: str  # Regex pattern for instruments
    data_types: List[DataType]
    priority: int = 1
    active: bool = True


class VenueRouter:
    """
    Intelligent venue router that manages connections to multiple exchanges
    and routes market data requests based on availability, performance, and rules.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._connections: Dict[Venue, VenueConnection] = {}
        self._routing_rules: List[RoutingRule] = []
        self._subscription_routes: Dict[str, Venue] = {}  # subscription_id -> venue
        self._venue_loads: Dict[Venue, int] = {}  # Track load per venue
        self._setup_default_connections()
        self._setup_default_routing_rules()
        
    def _setup_default_connections(self) -> None:
        """Setup default venue connections"""
        default_venues = [
            (Venue.BINANCE, {"BTC", "ETH", "ADA", "SOL"}, {DataType.TICK, DataType.QUOTE, DataType.BAR}),
            (Venue.COINBASE, {"BTC-USD", "ETH-USD", "ADA-USD"}, {DataType.QUOTE, DataType.TRADE}),
            (Venue.KRAKEN, {"XBTUSD", "ETHUSD"}, {DataType.TICK, DataType.ORDER_BOOK}),
            (Venue.BYBIT, {"BTCUSDT", "ETHUSDT"}, {DataType.TICK, DataType.BAR}),
            (Venue.OKX, {"BTC-USDT", "ETH-USDT"}, {DataType.TICK, DataType.QUOTE}),
        ]
        
        for venue, instruments, data_types in default_venues:
            self._connections[venue] = VenueConnection(
                venue=venue,
                status=VenueStatus.DISCONNECTED,
                last_heartbeat=datetime.now(),
                supported_instruments=set(instruments),
                supported_data_types=set(data_types),
                connection_priority=1
            )
            self._venue_loads[venue] = 0
            
    def _setup_default_routing_rules(self) -> None:
        """Setup default routing rules"""
        self._routing_rules = [
            # Bitcoin routing
            RoutingRule(Venue.BINANCE, r"BTC.*", [DataType.TICK, DataType.QUOTE], priority=1),
            RoutingRule(Venue.COINBASE, r"BTC-USD", [DataType.QUOTE], priority=2),
            RoutingRule(Venue.KRAKEN, r"XBTUSD", [DataType.ORDER_BOOK], priority=1),
            
            # Ethereum routing  
            RoutingRule(Venue.BINANCE, r"ETH.*", [DataType.TICK, DataType.BAR], priority=1),
            RoutingRule(Venue.COINBASE, r"ETH-USD", [DataType.QUOTE], priority=2),
            
            # Default fallback rules
            RoutingRule(Venue.BINANCE, r".*USDT", [DataType.TICK, DataType.QUOTE, DataType.BAR], priority=3),
            RoutingRule(Venue.BYBIT, r".*USDT", [DataType.TICK], priority=4),
        ]
        
    def get_venue_status(self, venue: Venue) -> Optional[VenueConnection]:
        """Get status for a specific venue"""
        return self._connections.get(venue)
        
    def get_all_venue_status(self) -> Dict[Venue, VenueConnection]:
        """Get status for all venues"""
        return self._connections.copy()
        
    def update_venue_status(self, venue: Venue, status: VenueStatus, error_message: str = None) -> None:
        """Update venue connection status"""
        if venue in self._connections:
            connection = self._connections[venue]
            connection.status = status
            connection.last_heartbeat = datetime.now()
            
            if status == VenueStatus.ERROR:
                connection.error_count += 1
                self.logger.warning(f"Venue {venue.value} error: {error_message}")
            elif status == VenueStatus.CONNECTED:
                connection.error_count = 0
                self.logger.info(f"Venue {venue.value} connected")
            elif status == VenueStatus.RATE_LIMITED:
                connection.rate_limit_reset = datetime.now() + timedelta(minutes=1)
                self.logger.warning(f"Venue {venue.value} rate limited")
                
    def route_subscription(self, instrument_id: str, data_type: DataType) -> Optional[Venue]:
        """Route a subscription request to the best available venue"""
        import re
        
        # Find matching routing rules
        matching_rules = []
        for rule in self._routing_rules:
            if not rule.active:
                continue
                
            if data_type in rule.data_types and re.match(rule.instrument_pattern, instrument_id):
                matching_rules.append(rule)
                
        if not matching_rules:
            self.logger.warning(f"No routing rules found for {instrument_id} {data_type.value}")
            return None
            
        # Sort by priority (lower number = higher priority)
        matching_rules.sort(key=lambda x: x.priority)
        
        # Find the best available venue
        for rule in matching_rules:
            venue = rule.venue
            connection = self._connections.get(venue)
            
            if not connection:
                continue
                
            # Check if venue is available
            if self._is_venue_available(connection, instrument_id, data_type):
                # Update load tracking
                self._venue_loads[venue] = self._venue_loads.get(venue, 0) + 1
                self.logger.info(f"Routed {instrument_id} {data_type.value} to {venue.value}")
                return venue
                
        self.logger.warning(f"No available venues for {instrument_id} {data_type.value}")
        return None
        
    def _is_venue_available(self, connection: VenueConnection, instrument_id: str, data_type: DataType) -> bool:
        """Check if venue is available for the requested data"""
        # Check connection status
        if connection.status != VenueStatus.CONNECTED:
            return False
            
        # Check rate limiting
        if (connection.rate_limit_reset and 
            datetime.now() < connection.rate_limit_reset):
            return False
            
        # Check if venue supports the instrument
        if (connection.supported_instruments and 
            instrument_id not in connection.supported_instruments):
            return False
            
        # Check if venue supports the data type
        if (connection.supported_data_types and 
            data_type not in connection.supported_data_types):
            return False
            
        # Check error count (too many errors = unavailable)
        if connection.error_count >= 5:
            return False
            
        return True
        
    def add_routing_rule(self, rule: RoutingRule) -> None:
        """Add a new routing rule"""
        self._routing_rules.append(rule)
        self._routing_rules.sort(key=lambda x: x.priority)
        
    def remove_routing_rule(self, venue: Venue, instrument_pattern: str) -> bool:
        """Remove a routing rule"""
        for rule in self._routing_rules:
            if rule.venue == venue and rule.instrument_pattern == instrument_pattern:
                self._routing_rules.remove(rule)
                return True
        return False
        
    def get_venue_load(self, venue: Venue) -> int:
        """Get current load for a venue"""
        return self._venue_loads.get(venue, 0)
        
    def get_load_balanced_venue(self, candidates: List[Venue]) -> Optional[Venue]:
        """Get the least loaded venue from candidates"""
        if not candidates:
            return None
            
        available_venues = [
            venue for venue in candidates 
            if self._connections.get(venue) and 
            self._connections[venue].status == VenueStatus.CONNECTED
        ]
        
        if not available_venues:
            return None
            
        # Return venue with lowest load
        return min(available_venues, key=lambda v: self._venue_loads.get(v, 0))
        
    def release_subscription(self, subscription_id: str) -> None:
        """Release a subscription and update load tracking"""
        if subscription_id in self._subscription_routes:
            venue = self._subscription_routes[subscription_id]
            self._venue_loads[venue] = max(0, self._venue_loads.get(venue, 0) - 1)
            del self._subscription_routes[subscription_id]
            
    def track_subscription(self, subscription_id: str, venue: Venue) -> None:
        """Track a subscription routing"""
        self._subscription_routes[subscription_id] = venue
        
    def get_routing_stats(self) -> Dict[str, any]:
        """Get routing statistics"""
        total_subscriptions = len(self._subscription_routes)
        venue_distribution = {}
        
        for venue in self._venue_loads:
            venue_distribution[venue.value] = self._venue_loads[venue]
            
        return {
            "total_subscriptions": total_subscriptions,
            "venue_distribution": venue_distribution,
            "active_venues": len([
                v for v in self._connections.values() 
                if v.status == VenueStatus.CONNECTED
            ]),
            "routing_rules": len(self._routing_rules)
        }
        
    async def health_check(self) -> None:
        """Perform health check on all venues"""
        for venue, connection in self._connections.items():
            # Check if venue hasn't sent heartbeat recently
            if datetime.now() - connection.last_heartbeat > timedelta(minutes=5):
                if connection.status == VenueStatus.CONNECTED:
                    self.logger.warning(f"Venue {venue.value} appears stale, marking as disconnected")
                    connection.status = VenueStatus.DISCONNECTED
                    
            # Reset rate limits if expired
            if (connection.rate_limit_reset and 
                datetime.now() >= connection.rate_limit_reset):
                connection.rate_limit_reset = None
                if connection.status == VenueStatus.RATE_LIMITED:
                    connection.status = VenueStatus.DISCONNECTED
                    self.logger.info(f"Rate limit reset for venue {venue.value}")


# Global router instance
venue_router = VenueRouter()