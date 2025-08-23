"""
Trade Execution Quality Analytics for Sprint 3 Priority 2
Slippage analysis, fill rate monitoring, execution time analysis, and market impact assessment
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import asyncpg
from scipy import stats
import json

logger = logging.getLogger(__name__)

class OrderType(Enum):
    """Order types for execution analysis"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(Enum):
    """Order sides"""
    BUY = "buy"
    SELL = "sell"

class ExecutionQuality(Enum):
    """Execution quality ratings"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"

@dataclass
class SlippageAnalysis:
    """Slippage analysis result"""
    order_id: str
    instrument_id: str
    order_type: OrderType
    side: OrderSide
    intended_price: Decimal
    executed_price: Decimal
    slippage_bps: float
    slippage_amount: Decimal
    market_impact_bps: float
    execution_timestamp: datetime
    market_conditions: Dict[str, Any]

@dataclass
class ExecutionMetrics:
    """Comprehensive execution metrics"""
    period_start: datetime
    period_end: datetime
    total_orders: int
    filled_orders: int
    fill_rate: float
    avg_execution_time_ms: float
    median_execution_time_ms: float
    avg_slippage_bps: float
    median_slippage_bps: float
    total_slippage_cost: Decimal
    market_impact_bps: float
    execution_quality_distribution: Dict[ExecutionQuality, int]
    best_execution_rate: float
    worst_execution_rate: float

@dataclass
class VenueAnalysis:
    """Venue execution analysis"""
    venue: str
    period_start: datetime
    period_end: datetime
    order_count: int
    fill_rate: float
    avg_slippage_bps: float
    avg_execution_time_ms: float
    market_share_pct: float
    execution_quality_score: float
    cost_per_share: Decimal
    venue_ranking: int

@dataclass
class MarketImpactAnalysis:
    """Market impact analysis"""
    order_id: str
    instrument_id: str
    trade_size: Decimal
    order_value: Decimal
    adv_percentage: float  # Percentage of Average Daily Volume
    pre_trade_spread: Decimal
    post_trade_spread: Decimal
    price_impact_bps: float
    temporary_impact_bps: float
    permanent_impact_bps: float
    impact_cost: Decimal
    liquidity_metrics: Dict[str, float]

@dataclass
class TimingAnalysis:
    """Order timing and execution analysis"""
    order_id: str
    instrument_id: str
    order_placement_time: datetime
    first_fill_time: Optional[datetime]
    last_fill_time: Optional[datetime]
    total_execution_time_ms: int
    time_to_first_fill_ms: Optional[int]
    fill_completion_time_ms: Optional[int]
    market_session: str  # pre-market, regular, after-hours
    volatility_regime: str  # low, normal, high
    execution_urgency: str  # patient, normal, aggressive

class ExecutionAnalytics:
    """
    Trade execution quality analysis engine
    Provides comprehensive analysis of trade execution performance
    """
    
    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool
        self.logger = logging.getLogger(__name__)
        
        # Execution quality thresholds (in basis points)
        self.slippage_thresholds = {
            ExecutionQuality.EXCELLENT: 2.0,
            ExecutionQuality.GOOD: 5.0,
            ExecutionQuality.FAIR: 10.0,
            ExecutionQuality.POOR: float('inf')
        }
        
        # Timing thresholds (in milliseconds)
        self.execution_time_thresholds = {
            'excellent': 100,
            'good': 500,
            'fair': 2000,
            'poor': float('inf')
        }
    
    async def analyze_slippage(
        self,
        order_id: str,
        benchmark_method: str = "arrival_price"
    ) -> SlippageAnalysis:
        """
        Analyze slippage for a specific order
        
        Args:
            order_id: Order identifier
            benchmark_method: Benchmark price method (arrival_price, midpoint, vwap)
            
        Returns:
            SlippageAnalysis with detailed slippage metrics
        """
        async with self.db_pool.acquire() as conn:
            try:
                # Get order details
                order_query = """
                    SELECT 
                        o.order_id,
                        o.instrument_id,
                        o.order_type,
                        o.side,
                        o.quantity,
                        o.limit_price,
                        o.created_at,
                        o.filled_at,
                        o.avg_fill_price,
                        o.status,
                        i.symbol,
                        i.multiplier
                    FROM orders o
                    JOIN instruments i ON o.instrument_id = i.instrument_id
                    WHERE o.order_id = $1
                """
                
                order = await conn.fetchrow(order_query, order_id)
                if not order:
                    raise ValueError(f"Order {order_id} not found")
                
                # Get market data at order time
                market_data = await self._get_market_data_at_time(
                    conn, order['instrument_id'], order['created_at']
                )
                
                # Calculate benchmark price
                benchmark_price = await self._calculate_benchmark_price(
                    conn, order, market_data, benchmark_method
                )
                
                executed_price = Decimal(str(order['avg_fill_price']))
                quantity = Decimal(str(order['quantity']))
                multiplier = Decimal(str(order['multiplier'] or 1))
                
                # Calculate slippage
                side_multiplier = 1 if order['side'].lower() == 'buy' else -1
                slippage_per_share = (executed_price - benchmark_price) * side_multiplier
                slippage_bps = float(slippage_per_share / benchmark_price * 10000)
                slippage_amount = slippage_per_share * quantity * multiplier
                
                # Calculate market impact
                market_impact_bps = await self._calculate_market_impact(
                    conn, order, market_data
                )
                
                # Get market conditions
                market_conditions = await self._get_market_conditions(
                    conn, order['instrument_id'], order['created_at']
                )
                
                slippage_analysis = SlippageAnalysis(
                    order_id=order_id,
                    instrument_id=order['instrument_id'],
                    order_type=OrderType(order['order_type'].lower()),
                    side=OrderSide(order['side'].lower()),
                    intended_price=benchmark_price,
                    executed_price=executed_price,
                    slippage_bps=slippage_bps,
                    slippage_amount=slippage_amount,
                    market_impact_bps=market_impact_bps,
                    execution_timestamp=order['filled_at'] or order['created_at'],
                    market_conditions=market_conditions
                )
                
                # Store analysis
                await self._store_slippage_analysis(conn, slippage_analysis)
                
                return slippage_analysis
                
            except Exception as e:
                self.logger.error(f"Error analyzing slippage for order {order_id}: {e}")
                raise
    
    async def calculate_execution_metrics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        strategy_id: Optional[str] = None,
        venue: Optional[str] = None
    ) -> ExecutionMetrics:
        """
        Calculate comprehensive execution metrics for a period
        """
        if not end_date:
            end_date = datetime.utcnow()
        if not start_date:
            start_date = end_date - timedelta(days=30)
        
        async with self.db_pool.acquire() as conn:
            try:
                # Build query filters
                where_conditions = ["o.created_at >= $1", "o.created_at <= $2"]
                params = [start_date, end_date]
                param_count = 2
                
                if strategy_id:
                    param_count += 1
                    where_conditions.append(f"o.strategy_id = ${param_count}")
                    params.append(strategy_id)
                
                if venue:
                    param_count += 1
                    where_conditions.append(f"o.venue = ${param_count}")
                    params.append(venue)
                
                where_clause = " AND ".join(where_conditions)
                
                # Get order statistics
                stats_query = f"""
                    SELECT 
                        COUNT(*) as total_orders,
                        COUNT(CASE WHEN status = 'filled' THEN 1 END) as filled_orders,
                        AVG(EXTRACT(EPOCH FROM (filled_at - created_at)) * 1000) as avg_execution_time_ms,
                        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (filled_at - created_at)) * 1000) as median_execution_time_ms
                    FROM orders o
                    WHERE {where_clause}
                """
                
                stats = await conn.fetchrow(stats_query, *params)
                
                total_orders = stats['total_orders'] or 0
                filled_orders = stats['filled_orders'] or 0
                fill_rate = filled_orders / total_orders if total_orders > 0 else 0
                
                # Get slippage statistics
                slippage_query = f"""
                    SELECT 
                        AVG(slippage_bps) as avg_slippage_bps,
                        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY slippage_bps) as median_slippage_bps,
                        SUM(slippage_amount) as total_slippage_cost,
                        AVG(market_impact_bps) as avg_market_impact_bps
                    FROM execution_slippage_analysis esa
                    JOIN orders o ON esa.order_id = o.order_id
                    WHERE {where_clause}
                """
                
                slippage_stats = await conn.fetchrow(slippage_query, *params)
                
                avg_slippage_bps = float(slippage_stats['avg_slippage_bps'] or 0)
                median_slippage_bps = float(slippage_stats['median_slippage_bps'] or 0)
                total_slippage_cost = Decimal(str(slippage_stats['total_slippage_cost'] or 0))
                market_impact_bps = float(slippage_stats['avg_market_impact_bps'] or 0)
                
                # Get execution quality distribution
                quality_distribution = await self._get_execution_quality_distribution(
                    conn, where_clause, params
                )
                
                # Calculate best/worst execution rates
                best_rate, worst_rate = await self._calculate_execution_rate_extremes(
                    conn, where_clause, params
                )
                
                execution_metrics = ExecutionMetrics(
                    period_start=start_date,
                    period_end=end_date,
                    total_orders=total_orders,
                    filled_orders=filled_orders,
                    fill_rate=fill_rate,
                    avg_execution_time_ms=float(stats['avg_execution_time_ms'] or 0),
                    median_execution_time_ms=float(stats['median_execution_time_ms'] or 0),
                    avg_slippage_bps=avg_slippage_bps,
                    median_slippage_bps=median_slippage_bps,
                    total_slippage_cost=total_slippage_cost,
                    market_impact_bps=market_impact_bps,
                    execution_quality_distribution=quality_distribution,
                    best_execution_rate=best_rate,
                    worst_execution_rate=worst_rate
                )
                
                # Store metrics
                await self._store_execution_metrics(conn, execution_metrics)
                
                return execution_metrics
                
            except Exception as e:
                self.logger.error(f"Error calculating execution metrics: {e}")
                raise
    
    async def analyze_venue_performance(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[VenueAnalysis]:
        """
        Analyze execution performance by venue
        """
        if not end_date:
            end_date = datetime.utcnow()
        if not start_date:
            start_date = end_date - timedelta(days=30)
        
        async with self.db_pool.acquire() as conn:
            try:
                venue_query = """
                    SELECT 
                        o.venue,
                        COUNT(*) as order_count,
                        COUNT(CASE WHEN o.status = 'filled' THEN 1 END) as filled_count,
                        AVG(EXTRACT(EPOCH FROM (o.filled_at - o.created_at)) * 1000) as avg_execution_time_ms,
                        AVG(esa.slippage_bps) as avg_slippage_bps,
                        SUM(o.quantity * o.avg_fill_price) as total_value,
                        SUM(esa.slippage_amount) as total_slippage_cost
                    FROM orders o
                    LEFT JOIN execution_slippage_analysis esa ON o.order_id = esa.order_id
                    WHERE o.created_at >= $1 AND o.created_at <= $2
                    GROUP BY o.venue
                    ORDER BY order_count DESC
                """
                
                venues = await conn.fetch(venue_query, start_date, end_date)
                
                venue_analyses = []
                total_orders = sum(venue['order_count'] for venue in venues)
                
                for i, venue in enumerate(venues):
                    venue_name = venue['venue']
                    order_count = venue['order_count']
                    filled_count = venue['filled_count'] or 0
                    
                    fill_rate = filled_count / order_count if order_count > 0 else 0
                    market_share = order_count / total_orders if total_orders > 0 else 0
                    
                    # Calculate execution quality score
                    quality_score = await self._calculate_venue_quality_score(venue)
                    
                    # Calculate cost per share
                    total_value = Decimal(str(venue['total_value'] or 0))
                    total_slippage = Decimal(str(venue['total_slippage_cost'] or 0))
                    cost_per_share = total_slippage / total_value if total_value > 0 else Decimal('0')
                    
                    venue_analysis = VenueAnalysis(
                        venue=venue_name,
                        period_start=start_date,
                        period_end=end_date,
                        order_count=order_count,
                        fill_rate=fill_rate,
                        avg_slippage_bps=float(venue['avg_slippage_bps'] or 0),
                        avg_execution_time_ms=float(venue['avg_execution_time_ms'] or 0),
                        market_share_pct=market_share * 100,
                        execution_quality_score=quality_score,
                        cost_per_share=cost_per_share,
                        venue_ranking=i + 1
                    )
                    
                    venue_analyses.append(venue_analysis)
                
                # Store venue analyses
                for venue_analysis in venue_analyses:
                    await self._store_venue_analysis(conn, venue_analysis)
                
                return venue_analyses
                
            except Exception as e:
                self.logger.error(f"Error analyzing venue performance: {e}")
                raise
    
    async def analyze_market_impact(
        self,
        order_id: str
    ) -> MarketImpactAnalysis:
        """
        Analyze market impact of a specific order
        """
        async with self.db_pool.acquire() as conn:
            try:
                # Get order details
                order_query = """
                    SELECT 
                        o.*,
                        i.symbol,
                        i.multiplier
                    FROM orders o
                    JOIN instruments i ON o.instrument_id = i.instrument_id
                    WHERE o.order_id = $1
                """
                
                order = await conn.fetchrow(order_query, order_id)
                if not order:
                    raise ValueError(f"Order {order_id} not found")
                
                # Get market data before and after trade
                pre_trade_data = await self._get_market_data_at_time(
                    conn, order['instrument_id'], order['created_at'] - timedelta(minutes=1)
                )
                
                post_trade_data = await self._get_market_data_at_time(
                    conn, order['instrument_id'], order['created_at'] + timedelta(minutes=5)
                )
                
                # Calculate Average Daily Volume (ADV)
                adv = await self._get_average_daily_volume(
                    conn, order['instrument_id'], order['created_at']
                )
                
                trade_size = Decimal(str(order['quantity']))
                order_value = trade_size * Decimal(str(order['avg_fill_price']))
                adv_percentage = float(trade_size / adv * 100) if adv > 0 else 0
                
                # Calculate spreads
                pre_trade_spread = (
                    Decimal(str(pre_trade_data.get('ask_price', 0))) - 
                    Decimal(str(pre_trade_data.get('bid_price', 0)))
                )
                
                post_trade_spread = (
                    Decimal(str(post_trade_data.get('ask_price', 0))) - 
                    Decimal(str(post_trade_data.get('bid_price', 0)))
                )
                
                # Calculate price impact
                pre_mid = (
                    Decimal(str(pre_trade_data.get('bid_price', 0))) + 
                    Decimal(str(pre_trade_data.get('ask_price', 0)))
                ) / 2
                
                post_mid = (
                    Decimal(str(post_trade_data.get('bid_price', 0))) + 
                    Decimal(str(post_trade_data.get('ask_price', 0)))
                ) / 2
                
                # Price impact calculation
                side_multiplier = 1 if order['side'].lower() == 'buy' else -1
                price_impact = (post_mid - pre_mid) * side_multiplier
                price_impact_bps = float(price_impact / pre_mid * 10000) if pre_mid > 0 else 0
                
                # Estimate temporary vs permanent impact (simplified)
                temporary_impact_bps = price_impact_bps * 0.7  # Typically 70% temporary
                permanent_impact_bps = price_impact_bps * 0.3  # 30% permanent
                
                impact_cost = price_impact * trade_size
                
                # Get liquidity metrics
                liquidity_metrics = await self._calculate_liquidity_metrics(
                    conn, order['instrument_id'], order['created_at']
                )
                
                market_impact_analysis = MarketImpactAnalysis(
                    order_id=order_id,
                    instrument_id=order['instrument_id'],
                    trade_size=trade_size,
                    order_value=order_value,
                    adv_percentage=adv_percentage,
                    pre_trade_spread=pre_trade_spread,
                    post_trade_spread=post_trade_spread,
                    price_impact_bps=price_impact_bps,
                    temporary_impact_bps=temporary_impact_bps,
                    permanent_impact_bps=permanent_impact_bps,
                    impact_cost=impact_cost,
                    liquidity_metrics=liquidity_metrics
                )
                
                # Store analysis
                await self._store_market_impact_analysis(conn, market_impact_analysis)
                
                return market_impact_analysis
                
            except Exception as e:
                self.logger.error(f"Error analyzing market impact for order {order_id}: {e}")
                raise
    
    async def analyze_execution_timing(
        self,
        order_id: str
    ) -> TimingAnalysis:
        """
        Analyze execution timing for an order
        """
        async with self.db_pool.acquire() as conn:
            try:
                # Get order and fill details
                timing_query = """
                    SELECT 
                        o.order_id,
                        o.instrument_id,
                        o.created_at as order_placement_time,
                        MIN(f.timestamp) as first_fill_time,
                        MAX(f.timestamp) as last_fill_time,
                        o.filled_at as completion_time
                    FROM orders o
                    LEFT JOIN fills f ON o.order_id = f.order_id
                    WHERE o.order_id = $1
                    GROUP BY o.order_id, o.instrument_id, o.created_at, o.filled_at
                """
                
                timing_data = await conn.fetchrow(timing_query, order_id)
                if not timing_data:
                    raise ValueError(f"Order {order_id} not found")
                
                order_placement_time = timing_data['order_placement_time']
                first_fill_time = timing_data['first_fill_time']
                last_fill_time = timing_data['last_fill_time']
                
                # Calculate timing metrics
                total_execution_time_ms = 0
                time_to_first_fill_ms = None
                fill_completion_time_ms = None
                
                if last_fill_time:
                    total_execution_time_ms = int(
                        (last_fill_time - order_placement_time).total_seconds() * 1000
                    )
                
                if first_fill_time:
                    time_to_first_fill_ms = int(
                        (first_fill_time - order_placement_time).total_seconds() * 1000
                    )
                
                if first_fill_time and last_fill_time:
                    fill_completion_time_ms = int(
                        (last_fill_time - first_fill_time).total_seconds() * 1000
                    )
                
                # Determine market session
                market_session = self._get_market_session(order_placement_time)
                
                # Determine volatility regime
                volatility_regime = await self._get_volatility_regime(
                    conn, timing_data['instrument_id'], order_placement_time
                )
                
                # Determine execution urgency (based on order type and timing)
                execution_urgency = await self._determine_execution_urgency(
                    conn, order_id
                )
                
                timing_analysis = TimingAnalysis(
                    order_id=order_id,
                    instrument_id=timing_data['instrument_id'],
                    order_placement_time=order_placement_time,
                    first_fill_time=first_fill_time,
                    last_fill_time=last_fill_time,
                    total_execution_time_ms=total_execution_time_ms,
                    time_to_first_fill_ms=time_to_first_fill_ms,
                    fill_completion_time_ms=fill_completion_time_ms,
                    market_session=market_session,
                    volatility_regime=volatility_regime,
                    execution_urgency=execution_urgency
                )
                
                # Store analysis
                await self._store_timing_analysis(conn, timing_analysis)
                
                return timing_analysis
                
            except Exception as e:
                self.logger.error(f"Error analyzing execution timing for order {order_id}: {e}")
                raise
    
    # Helper methods
    
    async def _get_market_data_at_time(
        self,
        conn: asyncpg.Connection,
        instrument_id: str,
        timestamp: datetime
    ) -> Dict[str, Any]:
        """Get market data at specific time"""
        timestamp_ns = int(timestamp.timestamp() * 1_000_000_000)
        
        # Get closest quote
        quote_query = """
            SELECT bid_price, ask_price, bid_size, ask_size
            FROM market_quotes
            WHERE instrument_id = $1
            AND timestamp_ns <= $2
            ORDER BY timestamp_ns DESC
            LIMIT 1
        """
        
        quote = await conn.fetchrow(quote_query, instrument_id, timestamp_ns)
        
        if quote:
            return {
                'bid_price': float(quote['bid_price']),
                'ask_price': float(quote['ask_price']),
                'bid_size': float(quote['bid_size']),
                'ask_size': float(quote['ask_size'])
            }
        
        return {}
    
    async def _calculate_benchmark_price(
        self,
        conn: asyncpg.Connection,
        order: dict,
        market_data: Dict[str, Any],
        method: str
    ) -> Decimal:
        """Calculate benchmark price for slippage calculation"""
        if method == "arrival_price":
            # Use mid-price at order arrival
            if 'bid_price' in market_data and 'ask_price' in market_data:
                return (Decimal(str(market_data['bid_price'])) + Decimal(str(market_data['ask_price']))) / 2
            
        elif method == "limit_price" and order['limit_price']:
            return Decimal(str(order['limit_price']))
        
        # Fallback to last trade price
        trade_query = """
            SELECT price
            FROM market_ticks
            WHERE instrument_id = $1
            AND timestamp_ns <= $2
            ORDER BY timestamp_ns DESC
            LIMIT 1
        """
        
        timestamp_ns = int(order['created_at'].timestamp() * 1_000_000_000)
        result = await conn.fetchrow(trade_query, order['instrument_id'], timestamp_ns)
        
        if result:
            return Decimal(str(result['price']))
        
        # Final fallback
        return Decimal(str(order['avg_fill_price']))
    
    async def _calculate_market_impact(
        self,
        conn: asyncpg.Connection,
        order: dict,
        market_data: Dict[str, Any]
    ) -> float:
        """Calculate market impact in basis points"""
        # Simplified market impact calculation
        # In practice, this would be more sophisticated
        trade_size = float(order['quantity'])
        
        # Estimate based on trade size relative to typical order size
        if trade_size > 10000:  # Large order
            return 15.0
        elif trade_size > 5000:  # Medium order
            return 8.0
        elif trade_size > 1000:  # Small order
            return 3.0
        else:  # Very small order
            return 1.0
    
    async def _get_market_conditions(
        self,
        conn: asyncpg.Connection,
        instrument_id: str,
        timestamp: datetime
    ) -> Dict[str, Any]:
        """Get market conditions at execution time"""
        # Get volatility, volume, spread metrics
        timestamp_ns = int(timestamp.timestamp() * 1_000_000_000)
        start_ns = timestamp_ns - (3600 * 1_000_000_000)  # 1 hour before
        
        conditions_query = """
            SELECT 
                AVG((ask_price - bid_price) / ((ask_price + bid_price) / 2)) as avg_spread_pct,
                AVG(bid_size + ask_size) as avg_depth,
                COUNT(*) as quote_updates
            FROM market_quotes
            WHERE instrument_id = $1
            AND timestamp_ns >= $2
            AND timestamp_ns <= $3
        """
        
        result = await conn.fetchrow(conditions_query, instrument_id, start_ns, timestamp_ns)
        
        return {
            'avg_spread_pct': float(result['avg_spread_pct'] or 0) * 100,
            'avg_depth': float(result['avg_depth'] or 0),
            'quote_updates': result['quote_updates'] or 0,
            'liquidity_score': min(float(result['avg_depth'] or 0) / 1000, 1.0)
        }
    
    async def _get_execution_quality_distribution(
        self,
        conn: asyncpg.Connection,
        where_clause: str,
        params: List[Any]
    ) -> Dict[ExecutionQuality, int]:
        """Get distribution of execution quality ratings"""
        quality_query = f"""
            SELECT 
                CASE 
                    WHEN ABS(slippage_bps) <= 2 THEN 'excellent'
                    WHEN ABS(slippage_bps) <= 5 THEN 'good'
                    WHEN ABS(slippage_bps) <= 10 THEN 'fair'
                    ELSE 'poor'
                END as quality,
                COUNT(*) as count
            FROM execution_slippage_analysis esa
            JOIN orders o ON esa.order_id = o.order_id
            WHERE {where_clause}
            GROUP BY quality
        """
        
        results = await conn.fetch(quality_query, *params)
        
        distribution = {quality: 0 for quality in ExecutionQuality}
        
        for result in results:
            quality_str = result['quality']
            count = result['count']
            
            if quality_str == 'excellent':
                distribution[ExecutionQuality.EXCELLENT] = count
            elif quality_str == 'good':
                distribution[ExecutionQuality.GOOD] = count
            elif quality_str == 'fair':
                distribution[ExecutionQuality.FAIR] = count
            else:
                distribution[ExecutionQuality.POOR] = count
        
        return distribution
    
    async def _calculate_execution_rate_extremes(
        self,
        conn: asyncpg.Connection,
        where_clause: str,
        params: List[Any]
    ) -> Tuple[float, float]:
        """Calculate best and worst execution rates"""
        # This is a simplified calculation
        # In practice, you'd define what constitutes "best" and "worst" execution
        
        extremes_query = f"""
            SELECT 
                MIN(slippage_bps) as best_slippage,
                MAX(slippage_bps) as worst_slippage
            FROM execution_slippage_analysis esa
            JOIN orders o ON esa.order_id = o.order_id
            WHERE {where_clause}
        """
        
        result = await conn.fetchrow(extremes_query, *params)
        
        best_rate = 100 - abs(float(result['best_slippage'] or 0))
        worst_rate = 100 - abs(float(result['worst_slippage'] or 0))
        
        return max(best_rate, 0), max(worst_rate, 0)
    
    def _calculate_venue_quality_score(self, venue_data: dict) -> float:
        """Calculate composite quality score for venue"""
        fill_rate = float(venue_data['filled_count'] or 0) / float(venue_data['order_count'] or 1)
        avg_slippage = abs(float(venue_data['avg_slippage_bps'] or 0))
        avg_time = float(venue_data['avg_execution_time_ms'] or 0)
        
        # Normalize and weight factors
        fill_rate_score = fill_rate * 40  # 40% weight
        slippage_score = max(0, (20 - avg_slippage) / 20 * 30)  # 30% weight
        time_score = max(0, (2000 - avg_time) / 2000 * 30)  # 30% weight
        
        return fill_rate_score + slippage_score + time_score
    
    async def _get_average_daily_volume(
        self,
        conn: asyncpg.Connection,
        instrument_id: str,
        reference_date: datetime
    ) -> Decimal:
        """Get Average Daily Volume for instrument"""
        end_date = reference_date.date()
        start_date = end_date - timedelta(days=30)
        
        start_ns = int(datetime.combine(start_date, datetime.min.time()).timestamp() * 1_000_000_000)
        end_ns = int(datetime.combine(end_date, datetime.max.time()).timestamp() * 1_000_000_000)
        
        adv_query = """
            SELECT AVG(volume) as avg_daily_volume
            FROM market_bars
            WHERE instrument_id = $1
            AND timeframe = '1d'
            AND timestamp_ns >= $2
            AND timestamp_ns <= $3
        """
        
        result = await conn.fetchrow(adv_query, instrument_id, start_ns, end_ns)
        return Decimal(str(result['avg_daily_volume'] or 0))
    
    async def _calculate_liquidity_metrics(
        self,
        conn: asyncpg.Connection,
        instrument_id: str,
        timestamp: datetime
    ) -> Dict[str, float]:
        """Calculate liquidity metrics"""
        timestamp_ns = int(timestamp.timestamp() * 1_000_000_000)
        window_ns = 300 * 1_000_000_000  # 5 minutes
        
        liquidity_query = """
            SELECT 
                AVG((ask_price - bid_price) / ((ask_price + bid_price) / 2)) as avg_spread_pct,
                AVG(bid_size + ask_size) as avg_market_depth,
                COUNT(*) as quote_count
            FROM market_quotes
            WHERE instrument_id = $1
            AND timestamp_ns >= $2
            AND timestamp_ns <= $3
        """
        
        result = await conn.fetchrow(
            liquidity_query, 
            instrument_id, 
            timestamp_ns - window_ns, 
            timestamp_ns + window_ns
        )
        
        return {
            'spread_pct': float(result['avg_spread_pct'] or 0) * 100,
            'market_depth': float(result['avg_market_depth'] or 0),
            'quote_frequency': float(result['quote_count'] or 0) / 10,  # per minute
            'liquidity_score': min(float(result['avg_market_depth'] or 0) / 10000, 1.0)
        }
    
    def _get_market_session(self, timestamp: datetime) -> str:
        """Determine market session"""
        hour = timestamp.hour
        
        if hour < 9 or (hour == 9 and timestamp.minute < 30):
            return "pre-market"
        elif hour >= 16:
            return "after-hours"
        else:
            return "regular"
    
    async def _get_volatility_regime(
        self,
        conn: asyncpg.Connection,
        instrument_id: str,
        timestamp: datetime
    ) -> str:
        """Determine volatility regime"""
        # Get recent price data to calculate volatility
        timestamp_ns = int(timestamp.timestamp() * 1_000_000_000)
        start_ns = timestamp_ns - (86400 * 1_000_000_000)  # 24 hours
        
        volatility_query = """
            SELECT close_price
            FROM market_bars
            WHERE instrument_id = $1
            AND timeframe = '1h'
            AND timestamp_ns >= $2
            AND timestamp_ns <= $3
            ORDER BY timestamp_ns
        """
        
        prices = await conn.fetch(volatility_query, instrument_id, start_ns, timestamp_ns)
        
        if len(prices) < 12:  # Need at least 12 hours of data
            return "normal"
        
        # Calculate hourly returns
        returns = []
        for i in range(1, len(prices)):
            prev_price = float(prices[i-1]['close_price'])
            curr_price = float(prices[i]['close_price'])
            
            if prev_price > 0:
                returns.append((curr_price - prev_price) / prev_price)
        
        if not returns:
            return "normal"
        
        volatility = np.std(returns) * np.sqrt(24)  # Annualized daily volatility
        
        if volatility < 0.15:  # Less than 15% annualized
            return "low"
        elif volatility > 0.30:  # Greater than 30% annualized
            return "high"
        else:
            return "normal"
    
    async def _determine_execution_urgency(
        self,
        conn: asyncpg.Connection,
        order_id: str
    ) -> str:
        """Determine execution urgency based on order characteristics"""
        order_query = """
            SELECT order_type, time_in_force
            FROM orders
            WHERE order_id = $1
        """
        
        result = await conn.fetchrow(order_query, order_id)
        
        if not result:
            return "normal"
        
        order_type = result['order_type'].lower()
        time_in_force = result.get('time_in_force', '').lower()
        
        if order_type == 'market' or time_in_force == 'ioc':
            return "aggressive"
        elif order_type == 'limit' and time_in_force in ['gtc', 'day']:
            return "patient"
        else:
            return "normal"
    
    # Storage methods
    
    async def _store_slippage_analysis(
        self,
        conn: asyncpg.Connection,
        analysis: SlippageAnalysis
    ):
        """Store slippage analysis result"""
        query = """
            INSERT INTO execution_slippage_analysis (
                order_id, instrument_id, order_type, side, intended_price,
                executed_price, slippage_bps, slippage_amount, market_impact_bps,
                execution_timestamp, market_conditions
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            ON CONFLICT (order_id) DO UPDATE SET
                slippage_bps = EXCLUDED.slippage_bps,
                slippage_amount = EXCLUDED.slippage_amount,
                market_impact_bps = EXCLUDED.market_impact_bps,
                market_conditions = EXCLUDED.market_conditions
        """
        
        await conn.execute(
            query,
            analysis.order_id,
            analysis.instrument_id,
            analysis.order_type.value,
            analysis.side.value,
            float(analysis.intended_price),
            float(analysis.executed_price),
            analysis.slippage_bps,
            float(analysis.slippage_amount),
            analysis.market_impact_bps,
            analysis.execution_timestamp,
            json.dumps(analysis.market_conditions)
        )
    
    async def _store_execution_metrics(
        self,
        conn: asyncpg.Connection,
        metrics: ExecutionMetrics
    ):
        """Store execution metrics"""
        query = """
            INSERT INTO execution_metrics (
                period_start, period_end, total_orders, filled_orders, fill_rate,
                avg_execution_time_ms, median_execution_time_ms, avg_slippage_bps,
                median_slippage_bps, total_slippage_cost, market_impact_bps,
                execution_quality_distribution, best_execution_rate, worst_execution_rate
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
        """
        
        quality_dist = {k.value: v for k, v in metrics.execution_quality_distribution.items()}
        
        await conn.execute(
            query,
            metrics.period_start,
            metrics.period_end,
            metrics.total_orders,
            metrics.filled_orders,
            metrics.fill_rate,
            metrics.avg_execution_time_ms,
            metrics.median_execution_time_ms,
            metrics.avg_slippage_bps,
            metrics.median_slippage_bps,
            float(metrics.total_slippage_cost),
            metrics.market_impact_bps,
            json.dumps(quality_dist),
            metrics.best_execution_rate,
            metrics.worst_execution_rate
        )
    
    async def _store_venue_analysis(
        self,
        conn: asyncpg.Connection,
        analysis: VenueAnalysis
    ):
        """Store venue analysis result"""
        query = """
            INSERT INTO execution_venue_analysis (
                venue, period_start, period_end, order_count, fill_rate,
                avg_slippage_bps, avg_execution_time_ms, market_share_pct,
                execution_quality_score, cost_per_share, venue_ranking
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            ON CONFLICT (venue, period_start, period_end) DO UPDATE SET
                order_count = EXCLUDED.order_count,
                fill_rate = EXCLUDED.fill_rate,
                avg_slippage_bps = EXCLUDED.avg_slippage_bps,
                avg_execution_time_ms = EXCLUDED.avg_execution_time_ms,
                market_share_pct = EXCLUDED.market_share_pct,
                execution_quality_score = EXCLUDED.execution_quality_score,
                cost_per_share = EXCLUDED.cost_per_share,
                venue_ranking = EXCLUDED.venue_ranking
        """
        
        await conn.execute(
            query,
            analysis.venue,
            analysis.period_start,
            analysis.period_end,
            analysis.order_count,
            analysis.fill_rate,
            analysis.avg_slippage_bps,
            analysis.avg_execution_time_ms,
            analysis.market_share_pct,
            analysis.execution_quality_score,
            float(analysis.cost_per_share),
            analysis.venue_ranking
        )
    
    async def _store_market_impact_analysis(
        self,
        conn: asyncpg.Connection,
        analysis: MarketImpactAnalysis
    ):
        """Store market impact analysis"""
        query = """
            INSERT INTO execution_market_impact (
                order_id, instrument_id, trade_size, order_value, adv_percentage,
                pre_trade_spread, post_trade_spread, price_impact_bps,
                temporary_impact_bps, permanent_impact_bps, impact_cost,
                liquidity_metrics
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            ON CONFLICT (order_id) DO UPDATE SET
                price_impact_bps = EXCLUDED.price_impact_bps,
                temporary_impact_bps = EXCLUDED.temporary_impact_bps,
                permanent_impact_bps = EXCLUDED.permanent_impact_bps,
                impact_cost = EXCLUDED.impact_cost,
                liquidity_metrics = EXCLUDED.liquidity_metrics
        """
        
        await conn.execute(
            query,
            analysis.order_id,
            analysis.instrument_id,
            float(analysis.trade_size),
            float(analysis.order_value),
            analysis.adv_percentage,
            float(analysis.pre_trade_spread),
            float(analysis.post_trade_spread),
            analysis.price_impact_bps,
            analysis.temporary_impact_bps,
            analysis.permanent_impact_bps,
            float(analysis.impact_cost),
            json.dumps(analysis.liquidity_metrics)
        )
    
    async def _store_timing_analysis(
        self,
        conn: asyncpg.Connection,
        analysis: TimingAnalysis
    ):
        """Store timing analysis"""
        query = """
            INSERT INTO execution_timing_analysis (
                order_id, instrument_id, order_placement_time, first_fill_time,
                last_fill_time, total_execution_time_ms, time_to_first_fill_ms,
                fill_completion_time_ms, market_session, volatility_regime,
                execution_urgency
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            ON CONFLICT (order_id) DO UPDATE SET
                first_fill_time = EXCLUDED.first_fill_time,
                last_fill_time = EXCLUDED.last_fill_time,
                total_execution_time_ms = EXCLUDED.total_execution_time_ms,
                time_to_first_fill_ms = EXCLUDED.time_to_first_fill_ms,
                fill_completion_time_ms = EXCLUDED.fill_completion_time_ms,
                market_session = EXCLUDED.market_session,
                volatility_regime = EXCLUDED.volatility_regime,
                execution_urgency = EXCLUDED.execution_urgency
        """
        
        await conn.execute(
            query,
            analysis.order_id,
            analysis.instrument_id,
            analysis.order_placement_time,
            analysis.first_fill_time,
            analysis.last_fill_time,
            analysis.total_execution_time_ms,
            analysis.time_to_first_fill_ms,
            analysis.fill_completion_time_ms,
            analysis.market_session,
            analysis.volatility_regime,
            analysis.execution_urgency
        )

# Global instance
execution_analytics = None

def get_execution_analytics() -> ExecutionAnalytics:
    """Get global execution analytics instance"""
    global execution_analytics
    if execution_analytics is None:
        raise RuntimeError("Execution analytics not initialized. Call init_execution_analytics() first.")
    return execution_analytics

def init_execution_analytics(db_pool: asyncpg.Pool) -> ExecutionAnalytics:
    """Initialize global execution analytics instance"""
    global execution_analytics
    execution_analytics = ExecutionAnalytics(db_pool)
    return execution_analytics