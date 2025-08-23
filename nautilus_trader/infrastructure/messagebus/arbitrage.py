#!/usr/bin/env python3
# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2025 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------

"""
Cross-Venue Arbitrage Message Routing for Enhanced MessageBus.

Provides intelligent routing and prioritization for cross-venue arbitrage
opportunities with ultra-low latency message handling:
- Real-time price spread detection across venues
- Priority-based message routing for arbitrage opportunities
- Cross-venue order book synchronization
- Latency-sensitive arbitrage signal routing
- Multi-venue execution coordination
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from nautilus_trader.infrastructure.messagebus.config import MessagePriority
from nautilus_trader.model.identifiers import Venue


class ArbitrageSignalType(Enum):
    """Types of arbitrage signals."""
    PRICE_SPREAD = "price_spread"           # Simple price difference
    TRIANGULAR = "triangular"               # Three-way arbitrage
    STATISTICAL = "statistical"             # Statistical arbitrage
    LATENCY_ARBITRAGE = "latency_arbitrage" # Speed-based arbitrage
    FUNDING_RATE = "funding_rate"           # Funding rate arbitrage
    CALENDAR_SPREAD = "calendar_spread"     # Time-based spread arbitrage


class ArbitrageUrgency(Enum):
    """Arbitrage opportunity urgency levels."""
    IMMEDIATE = "immediate"     # Requires sub-millisecond execution
    URGENT = "urgent"          # Requires execution within 100ms
    MODERATE = "moderate"      # Can wait up to 1 second
    LOW = "low"               # Opportunity has longer window


class VenueLatencyClass(Enum):
    """Venue latency classifications."""
    ULTRA_LOW = "ultra_low"    # < 1ms
    LOW = "low"               # 1-5ms  
    MEDIUM = "medium"         # 5-20ms
    HIGH = "high"            # 20-100ms
    VERY_HIGH = "very_high"  # > 100ms


@dataclass
class ArbitrageOpportunity:
    """Arbitrage opportunity data."""
    opportunity_id: str
    signal_type: ArbitrageSignalType
    urgency: ArbitrageUrgency
    venues: List[Venue]
    instruments: List[str]
    expected_profit: float
    confidence_score: float
    timestamp: float
    expiration_timestamp: float
    spread_size: float
    volume_available: float
    execution_complexity: int  # 1-10 scale
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class VenueConnection:
    """Venue connection information."""
    venue: Venue
    latency_class: VenueLatencyClass
    avg_latency_ms: float
    connection_quality: float  # 0-1 score
    last_update: float
    is_active: bool = True
    message_priority_offset: int = 0  # Adjustment for slow venues


@dataclass
class CrossVenueState:
    """Cross-venue market state tracking."""
    venue_prices: Dict[Venue, Dict[str, float]]
    venue_volumes: Dict[Venue, Dict[str, float]]
    venue_timestamps: Dict[Venue, float]
    spread_history: deque
    opportunity_count: int = 0
    active_arbitrages: int = 0


class ArbitrageRouter:
    """
    Cross-venue arbitrage message router for Enhanced MessageBus.
    
    Provides intelligent routing, prioritization, and coordination for
    arbitrage opportunities across multiple trading venues.
    """
    
    def __init__(self,
                 min_profit_threshold: float = 0.001,  # 0.1% minimum profit
                 max_opportunity_age_ms: float = 5000,  # 5 second max age
                 enable_predictive_routing: bool = True):
        """
        Initialize arbitrage router.
        
        Args:
            min_profit_threshold: Minimum profit threshold for opportunities
            max_opportunity_age_ms: Maximum age before opportunities expire
            enable_predictive_routing: Enable ML-based predictive routing
        """
        self.min_profit_threshold = min_profit_threshold
        self.max_opportunity_age = max_opportunity_age_ms / 1000.0
        self.enable_predictive_routing = enable_predictive_routing
        
        # Venue management
        self._venue_connections: Dict[Venue, VenueConnection] = {}
        self._venue_latencies: Dict[Venue, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Opportunity tracking
        self._active_opportunities: Dict[str, ArbitrageOpportunity] = {}
        self._opportunity_history: deque = deque(maxlen=1000)
        self._cross_venue_state = CrossVenueState(
            venue_prices=defaultdict(dict),
            venue_volumes=defaultdict(dict),
            venue_timestamps={},
            spread_history=deque(maxlen=500)
        )
        
        # Routing intelligence
        self._routing_rules: List[Callable[[ArbitrageOpportunity], MessagePriority]] = []
        self._predictive_model = None  # Will be initialized if enabled
        
        # Performance tracking
        self._routing_stats = {
            'opportunities_detected': 0,
            'opportunities_executed': 0,
            'total_profit': 0.0,
            'avg_detection_latency_ms': 0.0,
            'avg_execution_latency_ms': 0.0,
            'successful_arbitrages': 0,
            'failed_arbitrages': 0
        }
        
        # Message routing callbacks
        self._arbitrage_handlers: Dict[ArbitrageSignalType, List[Callable]] = defaultdict(list)
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._is_running = False
        
        # Logger
        self._logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Setup default routing rules
        self._setup_default_routing_rules()
        
        if enable_predictive_routing:
            self._initialize_predictive_model()
    
    def _setup_default_routing_rules(self) -> None:
        """Setup default routing rules for arbitrage opportunities."""
        
        def urgent_profit_rule(opportunity: ArbitrageOpportunity) -> MessagePriority:
            """High-profit opportunities get critical priority."""
            if opportunity.expected_profit > self.min_profit_threshold * 10:
                return MessagePriority.CRITICAL
            elif opportunity.expected_profit > self.min_profit_threshold * 3:
                return MessagePriority.HIGH
            else:
                return MessagePriority.NORMAL
        
        def urgency_rule(opportunity: ArbitrageOpportunity) -> MessagePriority:
            """Route based on opportunity urgency."""
            if opportunity.urgency == ArbitrageUrgency.IMMEDIATE:
                return MessagePriority.CRITICAL
            elif opportunity.urgency == ArbitrageUrgency.URGENT:
                return MessagePriority.HIGH
            elif opportunity.urgency == ArbitrageUrgency.MODERATE:
                return MessagePriority.NORMAL
            else:
                return MessagePriority.LOW
        
        def confidence_rule(opportunity: ArbitrageOpportunity) -> MessagePriority:
            """Route based on confidence score."""
            if opportunity.confidence_score > 0.9:
                return MessagePriority.HIGH
            elif opportunity.confidence_score > 0.7:
                return MessagePriority.NORMAL
            else:
                return MessagePriority.LOW
        
        def time_sensitivity_rule(opportunity: ArbitrageOpportunity) -> MessagePriority:
            """Route based on time until expiration."""
            current_time = time.time()
            time_to_expiry = opportunity.expiration_timestamp - current_time
            
            if time_to_expiry < 0.1:  # 100ms
                return MessagePriority.CRITICAL
            elif time_to_expiry < 0.5:  # 500ms
                return MessagePriority.HIGH
            elif time_to_expiry < 2.0:  # 2 seconds
                return MessagePriority.NORMAL
            else:
                return MessagePriority.LOW
        
        # Add default rules
        self._routing_rules.extend([
            urgent_profit_rule,
            urgency_rule,
            confidence_rule,
            time_sensitivity_rule
        ])
    
    def _initialize_predictive_model(self) -> None:
        """Initialize ML-based predictive routing model."""
        try:
            # Simple linear model for demonstration
            # In production, would use more sophisticated ML models
            self._predictive_model = {
                'profit_weight': 2.0,
                'confidence_weight': 1.5,
                'urgency_weight': 1.8,
                'venue_quality_weight': 1.2,
                'historical_success_weight': 1.0
            }
            self._logger.info("Predictive routing model initialized")
        except Exception as e:
            self._logger.error(f"Failed to initialize predictive model: {e}")
            self.enable_predictive_routing = False
    
    def start_routing(self) -> None:
        """Start arbitrage routing service."""
        if not self._is_running:
            self._is_running = True
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            self._logger.info("Arbitrage routing started")
    
    def stop_routing(self) -> None:
        """Stop arbitrage routing service."""
        self._is_running = False
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
        self._logger.info("Arbitrage routing stopped")
    
    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self._is_running:
            try:
                await asyncio.sleep(0.1)  # 100ms monitoring interval
                await self._monitor_opportunities()
                await self._update_venue_states()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Error in monitoring loop: {e}")
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while self._is_running:
            try:
                await asyncio.sleep(1.0)  # 1 second cleanup interval
                await self._cleanup_expired_opportunities()
                await self._update_performance_stats()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Error in cleanup loop: {e}")
    
    async def _monitor_opportunities(self) -> None:
        """Monitor active arbitrage opportunities."""
        current_time = time.time()
        
        for opp_id, opportunity in list(self._active_opportunities.items()):
            # Check if opportunity has expired
            if current_time > opportunity.expiration_timestamp:
                self._logger.debug(f"Arbitrage opportunity {opp_id} expired")
                del self._active_opportunities[opp_id]
                continue
            
            # Check if opportunity should be re-prioritized
            if self._should_reprioritize_opportunity(opportunity, current_time):
                new_priority = self._calculate_message_priority(opportunity)
                await self._route_arbitrage_message(opportunity, new_priority)
    
    def _should_reprioritize_opportunity(self, opportunity: ArbitrageOpportunity, current_time: float) -> bool:
        """Check if opportunity should be re-prioritized."""
        # Re-prioritize if getting close to expiration
        time_to_expiry = opportunity.expiration_timestamp - current_time
        return time_to_expiry < 1.0  # Re-prioritize in final second
    
    async def _update_venue_states(self) -> None:
        """Update cross-venue market states."""
        current_time = time.time()
        
        # Update venue latency classifications
        for venue, latency_history in self._venue_latencies.items():
            if len(latency_history) >= 10:
                avg_latency = np.mean(list(latency_history))
                
                if venue in self._venue_connections:
                    connection = self._venue_connections[venue]
                    connection.avg_latency_ms = avg_latency
                    connection.last_update = current_time
                    
                    # Update latency class
                    if avg_latency < 1.0:
                        connection.latency_class = VenueLatencyClass.ULTRA_LOW
                    elif avg_latency < 5.0:
                        connection.latency_class = VenueLatencyClass.LOW
                    elif avg_latency < 20.0:
                        connection.latency_class = VenueLatencyClass.MEDIUM
                    elif avg_latency < 100.0:
                        connection.latency_class = VenueLatencyClass.HIGH
                    else:
                        connection.latency_class = VenueLatencyClass.VERY_HIGH
    
    async def _cleanup_expired_opportunities(self) -> None:
        """Clean up expired opportunities."""
        current_time = time.time()
        expired_opportunities = []
        
        for opp_id, opportunity in self._active_opportunities.items():
            if current_time > opportunity.expiration_timestamp:
                expired_opportunities.append(opp_id)
        
        for opp_id in expired_opportunities:
            opportunity = self._active_opportunities[opp_id]
            self._opportunity_history.append(opportunity)
            del self._active_opportunities[opp_id]
            
            self._logger.debug(f"Cleaned up expired opportunity {opp_id}")
    
    async def _update_performance_stats(self) -> None:
        """Update routing performance statistics."""
        # Calculate average detection latency
        recent_opportunities = list(self._opportunity_history)[-100:]  # Last 100
        if recent_opportunities:
            detection_latencies = []
            for opp in recent_opportunities:
                if 'detection_latency_ms' in opp.metadata:
                    detection_latencies.append(opp.metadata['detection_latency_ms'])
            
            if detection_latencies:
                self._routing_stats['avg_detection_latency_ms'] = np.mean(detection_latencies)
    
    def register_venue_connection(self, venue: Venue, latency_class: VenueLatencyClass, 
                                avg_latency_ms: float = 0.0) -> None:
        """Register a venue connection for arbitrage routing."""
        connection = VenueConnection(
            venue=venue,
            latency_class=latency_class,
            avg_latency_ms=avg_latency_ms,
            connection_quality=1.0,  # Start with perfect quality
            last_update=time.time(),
            is_active=True
        )
        
        self._venue_connections[venue] = connection
        self._logger.info(f"Registered venue connection: {venue} ({latency_class.value})")
    
    def update_venue_price(self, venue: Venue, instrument: str, price: float, volume: float = 0.0) -> None:
        """Update venue price and detect arbitrage opportunities."""
        current_time = time.time()
        
        # Store price update
        self._cross_venue_state.venue_prices[venue][instrument] = price
        if volume > 0:
            self._cross_venue_state.venue_volumes[venue][instrument] = volume
        self._cross_venue_state.venue_timestamps[venue] = current_time
        
        # Check for arbitrage opportunities
        self._detect_price_arbitrage(instrument, current_time)
    
    def _detect_price_arbitrage(self, instrument: str, timestamp: float) -> None:
        """Detect price arbitrage opportunities across venues."""
        # Get all venues with prices for this instrument
        venues_with_prices = []
        for venue, prices in self._cross_venue_state.venue_prices.items():
            if instrument in prices:
                venues_with_prices.append((venue, prices[instrument]))
        
        if len(venues_with_prices) < 2:
            return  # Need at least 2 venues
        
        # Find best bid and ask
        venues_with_prices.sort(key=lambda x: x[1])  # Sort by price
        
        cheapest_venue, cheapest_price = venues_with_prices[0]
        expensive_venue, expensive_price = venues_with_prices[-1]
        
        # Calculate spread
        spread = expensive_price - cheapest_price
        spread_percent = spread / cheapest_price if cheapest_price > 0 else 0
        
        if spread_percent > self.min_profit_threshold:
            # Create arbitrage opportunity
            opportunity = self._create_arbitrage_opportunity(
                signal_type=ArbitrageSignalType.PRICE_SPREAD,
                venues=[cheapest_venue, expensive_venue],
                instruments=[instrument],
                spread_size=spread_percent,
                timestamp=timestamp
            )
            
            self._register_opportunity(opportunity)
    
    def _create_arbitrage_opportunity(self,
                                    signal_type: ArbitrageSignalType,
                                    venues: List[Venue],
                                    instruments: List[str],
                                    spread_size: float,
                                    timestamp: float) -> ArbitrageOpportunity:
        """Create arbitrage opportunity."""
        opportunity_id = f"arb_{int(timestamp * 1000)}_{signal_type.value}"
        
        # Calculate expected profit
        expected_profit = spread_size * 0.8  # Assume 80% capture rate
        
        # Determine urgency based on spread size and type
        if spread_size > self.min_profit_threshold * 5:
            urgency = ArbitrageUrgency.IMMEDIATE
        elif spread_size > self.min_profit_threshold * 2:
            urgency = ArbitrageUrgency.URGENT
        else:
            urgency = ArbitrageUrgency.MODERATE
        
        # Calculate confidence score
        confidence_score = min(0.95, 0.5 + (spread_size / self.min_profit_threshold) * 0.1)
        
        # Set expiration (shorter for larger spreads)
        expiration_seconds = max(0.5, 5.0 - (spread_size / self.min_profit_threshold))
        expiration_timestamp = timestamp + expiration_seconds
        
        return ArbitrageOpportunity(
            opportunity_id=opportunity_id,
            signal_type=signal_type,
            urgency=urgency,
            venues=venues,
            instruments=instruments,
            expected_profit=expected_profit,
            confidence_score=confidence_score,
            timestamp=timestamp,
            expiration_timestamp=expiration_timestamp,
            spread_size=spread_size,
            volume_available=1000.0,  # Default volume
            execution_complexity=2,    # Simple two-venue arbitrage
            metadata={
                'detection_latency_ms': 0.0,  # Would be measured in real implementation
                'spread_percent': spread_size * 100
            }
        )
    
    def _register_opportunity(self, opportunity: ArbitrageOpportunity) -> None:
        """Register new arbitrage opportunity."""
        self._active_opportunities[opportunity.opportunity_id] = opportunity
        self._routing_stats['opportunities_detected'] += 1
        self._cross_venue_state.opportunity_count += 1
        
        # Route the opportunity message
        priority = self._calculate_message_priority(opportunity)
        asyncio.create_task(self._route_arbitrage_message(opportunity, priority))
        
        self._logger.info(f"Detected arbitrage opportunity: {opportunity.opportunity_id} "
                         f"({opportunity.expected_profit:.4f} profit, {priority.value} priority)")
    
    def _calculate_message_priority(self, opportunity: ArbitrageOpportunity) -> MessagePriority:
        """Calculate message priority for arbitrage opportunity."""
        priorities = []
        
        # Apply all routing rules
        for rule in self._routing_rules:
            try:
                priority = rule(opportunity)
                priorities.append(priority)
            except Exception as e:
                self._logger.error(f"Error in routing rule: {e}")
        
        if not priorities:
            return MessagePriority.NORMAL
        
        # Use predictive model if available
        if self.enable_predictive_routing and self._predictive_model:
            predicted_priority = self._predict_priority(opportunity)
            priorities.append(predicted_priority)
        
        # Return highest priority
        priority_values = {
            MessagePriority.CRITICAL: 4,
            MessagePriority.HIGH: 3,
            MessagePriority.NORMAL: 2,
            MessagePriority.LOW: 1
        }
        
        highest_value = max(priority_values[p] for p in priorities)
        
        for priority, value in priority_values.items():
            if value == highest_value:
                return priority
        
        return MessagePriority.NORMAL
    
    def _predict_priority(self, opportunity: ArbitrageOpportunity) -> MessagePriority:
        """Use predictive model to determine message priority."""
        if not self._predictive_model:
            return MessagePriority.NORMAL
        
        # Calculate prediction score
        score = 0.0
        weights = self._predictive_model
        
        score += opportunity.expected_profit * weights['profit_weight'] * 1000
        score += opportunity.confidence_score * weights['confidence_weight']
        
        # Urgency score
        urgency_scores = {
            ArbitrageUrgency.IMMEDIATE: 4.0,
            ArbitrageUrgency.URGENT: 3.0,
            ArbitrageUrgency.MODERATE: 2.0,
            ArbitrageUrgency.LOW: 1.0
        }
        score += urgency_scores[opportunity.urgency] * weights['urgency_weight']
        
        # Venue quality score
        venue_quality = self._calculate_venue_quality_score(opportunity.venues)
        score += venue_quality * weights['venue_quality_weight']
        
        # Convert score to priority
        if score > 15.0:
            return MessagePriority.CRITICAL
        elif score > 10.0:
            return MessagePriority.HIGH
        elif score > 5.0:
            return MessagePriority.NORMAL
        else:
            return MessagePriority.LOW
    
    def _calculate_venue_quality_score(self, venues: List[Venue]) -> float:
        """Calculate quality score for venues involved."""
        total_quality = 0.0
        
        for venue in venues:
            if venue in self._venue_connections:
                connection = self._venue_connections[venue]
                quality = connection.connection_quality
                
                # Adjust for latency class
                latency_adjustments = {
                    VenueLatencyClass.ULTRA_LOW: 1.0,
                    VenueLatencyClass.LOW: 0.9,
                    VenueLatencyClass.MEDIUM: 0.7,
                    VenueLatencyClass.HIGH: 0.5,
                    VenueLatencyClass.VERY_HIGH: 0.3
                }
                quality *= latency_adjustments.get(connection.latency_class, 0.5)
                total_quality += quality
            else:
                total_quality += 0.5  # Default quality for unknown venues
        
        return total_quality / len(venues) if venues else 0.0
    
    async def _route_arbitrage_message(self, opportunity: ArbitrageOpportunity, priority: MessagePriority) -> None:
        """Route arbitrage message to appropriate handlers."""
        # Get handlers for this signal type
        handlers = self._arbitrage_handlers.get(opportunity.signal_type, [])
        
        if not handlers:
            # Use general arbitrage handlers if specific ones not found
            handlers = self._arbitrage_handlers.get('general', [])
        
        # Create routing message
        message = {
            'type': 'arbitrage_opportunity',
            'opportunity': opportunity,
            'priority': priority,
            'routing_timestamp': time.time(),
            'venues': [str(v) for v in opportunity.venues],
            'instruments': opportunity.instruments,
            'signal_type': opportunity.signal_type.value
        }
        
        # Route to all handlers
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(message)
                else:
                    handler(message)
            except Exception as e:
                self._logger.error(f"Error in arbitrage handler: {e}")
    
    def add_arbitrage_handler(self, signal_type: ArbitrageSignalType, handler: Callable) -> None:
        """Add handler for specific arbitrage signal type."""
        self._arbitrage_handlers[signal_type].append(handler)
        self._logger.info(f"Added arbitrage handler for {signal_type.value}")
    
    def add_routing_rule(self, rule_func: Callable[[ArbitrageOpportunity], MessagePriority]) -> None:
        """Add custom routing rule."""
        self._routing_rules.append(rule_func)
        self._logger.info("Added custom routing rule")
    
    def get_active_opportunities(self) -> List[ArbitrageOpportunity]:
        """Get currently active arbitrage opportunities."""
        return list(self._active_opportunities.values())
    
    def get_opportunity_history(self, limit: int = 100) -> List[ArbitrageOpportunity]:
        """Get historical arbitrage opportunities."""
        return list(self._opportunity_history)[-limit:]
    
    def get_venue_connections(self) -> Dict[Venue, VenueConnection]:
        """Get current venue connections."""
        return dict(self._venue_connections)
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get arbitrage routing statistics."""
        return {
            'total_opportunities_detected': self._routing_stats['opportunities_detected'],
            'active_opportunities': len(self._active_opportunities),
            'total_opportunities_executed': self._routing_stats['opportunities_executed'],
            'success_rate': (
                self._routing_stats['successful_arbitrages'] / 
                max(1, self._routing_stats['opportunities_executed'])
            ),
            'total_profit': self._routing_stats['total_profit'],
            'avg_detection_latency_ms': self._routing_stats['avg_detection_latency_ms'],
            'avg_execution_latency_ms': self._routing_stats['avg_execution_latency_ms'],
            'venue_count': len(self._venue_connections),
            'is_running': self._is_running,
            'predictive_routing_enabled': self.enable_predictive_routing
        }
    
    def record_arbitrage_execution(self, opportunity_id: str, success: bool, profit: float = 0.0) -> None:
        """Record arbitrage execution result."""
        if opportunity_id in self._active_opportunities:
            opportunity = self._active_opportunities[opportunity_id]
            del self._active_opportunities[opportunity_id]
            
            # Update stats
            self._routing_stats['opportunities_executed'] += 1
            if success:
                self._routing_stats['successful_arbitrages'] += 1
                self._routing_stats['total_profit'] += profit
                self._logger.info(f"Successful arbitrage execution: {opportunity_id} (profit: {profit:.4f})")
            else:
                self._routing_stats['failed_arbitrages'] += 1
                self._logger.warning(f"Failed arbitrage execution: {opportunity_id}")
            
            # Store in history
            opportunity.metadata['execution_success'] = success
            opportunity.metadata['actual_profit'] = profit
            self._opportunity_history.append(opportunity)
    
    def update_venue_latency(self, venue: Venue, latency_ms: float) -> None:
        """Update venue latency measurement."""
        self._venue_latencies[venue].append(latency_ms)
        
        if venue in self._venue_connections:
            connection = self._venue_connections[venue]
            # Update connection quality based on latency consistency
            latencies = list(self._venue_latencies[venue])
            if len(latencies) >= 10:
                latency_std = np.std(latencies)
                # Lower std deviation = higher quality
                connection.connection_quality = max(0.1, 1.0 - (latency_std / 100.0))
    
    def get_cross_venue_state(self) -> CrossVenueState:
        """Get current cross-venue market state."""
        return self._cross_venue_state