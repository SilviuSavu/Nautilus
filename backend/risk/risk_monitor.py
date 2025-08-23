"""
Real-time Risk Monitor for Sprint 3 Priority 3
Live position tracking, real-time risk calculations, and continuous monitoring
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from dataclasses import dataclass
from enum import Enum
import asyncpg
from concurrent.futures import ThreadPoolExecutor
import json
import redis
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Risk level classifications"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class MonitoringStatus(Enum):
    """Risk monitoring status"""
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"

class AlertPriority(Enum):
    """Alert priority levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class PositionRisk:
    """Real-time position risk metrics"""
    position_id: str
    portfolio_id: str
    strategy_id: str
    instrument_id: str
    symbol: str
    quantity: Decimal
    market_value: Decimal
    unrealized_pnl: Decimal
    cost_basis: Decimal
    current_price: Decimal
    risk_exposure: Decimal
    var_contribution: Decimal
    portfolio_weight: float
    concentration_risk: float
    liquidity_score: float
    beta: Optional[float]
    last_updated: datetime

@dataclass
class PortfolioRisk:
    """Portfolio-level risk metrics"""
    portfolio_id: str
    total_value: Decimal
    total_exposure: Decimal
    net_exposure: Decimal
    gross_exposure: Decimal
    long_exposure: Decimal
    short_exposure: Decimal
    var_95: Decimal
    var_99: Decimal
    expected_shortfall: Decimal
    max_drawdown: float
    concentration_risk: float
    leverage_ratio: float
    beta: float
    correlation_risk: float
    liquidity_risk: float
    positions_count: int
    risk_level: RiskLevel
    last_updated: datetime

@dataclass
class RiskAlert:
    """Risk alert/breach notification"""
    alert_id: str
    portfolio_id: str
    strategy_id: Optional[str]
    position_id: Optional[str]
    alert_type: str
    priority: AlertPriority
    risk_level: RiskLevel
    description: str
    current_value: Decimal
    threshold_value: Decimal
    breach_magnitude: Decimal
    breach_percentage: float
    recommended_action: str
    auto_action_taken: bool
    metadata: Dict[str, Any]
    created_at: datetime
    acknowledged: bool = False
    resolved: bool = False

@dataclass
class MonitoringThresholds:
    """Risk monitoring thresholds configuration"""
    portfolio_id: str
    max_portfolio_var: Decimal
    max_position_weight: float
    max_sector_concentration: float
    max_leverage_ratio: float
    min_liquidity_score: float
    max_correlation_exposure: float
    max_drawdown_limit: float
    position_size_limit: Decimal
    daily_loss_limit: Decimal
    intraday_var_limit: Decimal

class RealTimeRiskMonitor:
    """
    Real-time risk monitoring system with live position tracking,
    continuous risk calculations, and automated alerting
    """
    
    def __init__(
        self, 
        db_pool: asyncpg.Pool, 
        redis_client: Optional[redis.Redis] = None,
        update_interval: int = 30  # seconds
    ):
        self.db_pool = db_pool
        self.redis_client = redis_client
        self.update_interval = update_interval
        self.logger = logging.getLogger(__name__)
        
        # Monitoring state
        self.monitoring_status = MonitoringStatus.STOPPED
        self.active_portfolios: Set[str] = set()
        self.position_cache: Dict[str, PositionRisk] = {}
        self.portfolio_cache: Dict[str, PortfolioRisk] = {}
        self.alert_callbacks: List[callable] = []
        
        # Risk calculation parameters
        self.confidence_levels = [0.95, 0.99]
        self.lookback_days = 252
        self.var_calculation_method = "historical"
        
        # Threading for intensive calculations
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Risk thresholds by portfolio
        self.thresholds: Dict[str, MonitoringThresholds] = {}
        
        # Background task
        self._monitoring_task = None
    
    async def start_monitoring(
        self, 
        portfolio_ids: List[str],
        load_thresholds: bool = True
    ) -> None:
        """
        Start real-time risk monitoring for specified portfolios
        """
        try:
            self.active_portfolios.update(portfolio_ids)
            
            if load_thresholds:
                await self._load_risk_thresholds(portfolio_ids)
            
            # Initialize position and portfolio caches
            await self._initialize_caches()
            
            # Start monitoring task
            if self._monitoring_task is None or self._monitoring_task.done():
                self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            self.monitoring_status = MonitoringStatus.ACTIVE
            self.logger.info(f"Started risk monitoring for {len(portfolio_ids)} portfolios")
            
        except Exception as e:
            self.monitoring_status = MonitoringStatus.ERROR
            self.logger.error(f"Error starting risk monitoring: {e}")
            raise
    
    async def stop_monitoring(self) -> None:
        """
        Stop real-time risk monitoring
        """
        self.monitoring_status = MonitoringStatus.STOPPED
        
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.active_portfolios.clear()
        self.position_cache.clear()
        self.portfolio_cache.clear()
        
        self.logger.info("Risk monitoring stopped")
    
    async def add_alert_callback(self, callback: callable) -> None:
        """
        Add callback function for risk alerts
        """
        self.alert_callbacks.append(callback)
    
    async def get_real_time_risk(self, portfolio_id: str) -> Optional[PortfolioRisk]:
        """
        Get current real-time risk metrics for portfolio
        """
        if portfolio_id not in self.active_portfolios:
            await self.start_monitoring([portfolio_id])
        
        return self.portfolio_cache.get(portfolio_id)
    
    async def get_position_risks(
        self, 
        portfolio_id: str, 
        top_n: Optional[int] = None
    ) -> List[PositionRisk]:
        """
        Get position-level risk metrics for portfolio
        """
        positions = [
            pos for pos in self.position_cache.values() 
            if pos.portfolio_id == portfolio_id
        ]
        
        # Sort by risk exposure
        positions.sort(key=lambda p: float(p.risk_exposure), reverse=True)
        
        if top_n:
            positions = positions[:top_n]
        
        return positions
    
    async def calculate_portfolio_var(
        self,
        portfolio_id: str,
        confidence_level: float = 0.95,
        time_horizon: int = 1
    ) -> Tuple[Decimal, Decimal]:
        """
        Calculate portfolio VaR and Expected Shortfall
        """
        async with self.db_pool.acquire() as conn:
            try:
                # Get portfolio positions
                positions_query = """
                    SELECT 
                        p.position_id,
                        p.instrument_id,
                        p.quantity,
                        p.avg_entry_price,
                        p.unrealized_pnl,
                        i.symbol,
                        i.asset_class,
                        i.multiplier
                    FROM positions p
                    JOIN instruments i ON p.instrument_id = i.instrument_id
                    WHERE p.portfolio_id = $1
                    AND p.quantity != 0
                """
                
                positions = await conn.fetch(positions_query, portfolio_id)
                
                if not positions:
                    return Decimal('0'), Decimal('0')
                
                # Get historical returns for VaR calculation
                returns_data = await self._get_portfolio_returns(conn, positions, time_horizon)
                
                if len(returns_data) < 30:  # Need minimum data
                    return Decimal('0'), Decimal('0')
                
                # Calculate VaR
                var_amount = await self._calculate_var(
                    returns_data, confidence_level, time_horizon
                )
                
                # Calculate Expected Shortfall
                es_amount = await self._calculate_expected_shortfall(
                    returns_data, confidence_level
                )
                
                return Decimal(str(var_amount)), Decimal(str(es_amount))
                
            except Exception as e:
                self.logger.error(f"Error calculating portfolio VaR: {e}")
                return Decimal('0'), Decimal('0')
    
    async def check_risk_breaches(
        self, 
        portfolio_id: str
    ) -> List[RiskAlert]:
        """
        Check for risk limit breaches and generate alerts
        """
        portfolio_risk = await self.get_real_time_risk(portfolio_id)
        if not portfolio_risk:
            return []
        
        thresholds = self.thresholds.get(portfolio_id)
        if not thresholds:
            return []
        
        alerts = []
        
        # VaR breach check
        if portfolio_risk.var_95 < -thresholds.max_portfolio_var:
            alert = await self._create_alert(
                portfolio_id=portfolio_id,
                alert_type="var_breach",
                priority=AlertPriority.ERROR,
                description=f"Portfolio VaR exceeded limit",
                current_value=portfolio_risk.var_95,
                threshold_value=thresholds.max_portfolio_var,
                recommended_action="Reduce position sizes or hedge exposure"
            )
            alerts.append(alert)
        
        # Leverage breach check
        if portfolio_risk.leverage_ratio > thresholds.max_leverage_ratio:
            alert = await self._create_alert(
                portfolio_id=portfolio_id,
                alert_type="leverage_breach",
                priority=AlertPriority.WARNING,
                description=f"Leverage ratio exceeded limit",
                current_value=Decimal(str(portfolio_risk.leverage_ratio)),
                threshold_value=Decimal(str(thresholds.max_leverage_ratio)),
                recommended_action="Reduce leverage by closing positions"
            )
            alerts.append(alert)
        
        # Concentration risk check
        if portfolio_risk.concentration_risk > thresholds.max_sector_concentration:
            alert = await self._create_alert(
                portfolio_id=portfolio_id,
                alert_type="concentration_risk",
                priority=AlertPriority.WARNING,
                description=f"Portfolio concentration risk high",
                current_value=Decimal(str(portfolio_risk.concentration_risk)),
                threshold_value=Decimal(str(thresholds.max_sector_concentration)),
                recommended_action="Diversify positions across sectors"
            )
            alerts.append(alert)
        
        # Daily loss limit check
        daily_pnl = await self._get_daily_pnl(portfolio_id)
        if daily_pnl < -thresholds.daily_loss_limit:
            alert = await self._create_alert(
                portfolio_id=portfolio_id,
                alert_type="daily_loss_limit",
                priority=AlertPriority.CRITICAL,
                description=f"Daily loss limit exceeded",
                current_value=daily_pnl,
                threshold_value=thresholds.daily_loss_limit,
                recommended_action="EMERGENCY: Stop trading and review positions"
            )
            alerts.append(alert)
        
        # Position size checks
        position_alerts = await self._check_position_limits(portfolio_id, thresholds)
        alerts.extend(position_alerts)
        
        # Process alerts
        for alert in alerts:
            await self._process_alert(alert)
        
        return alerts
    
    # Private methods
    
    async def _monitoring_loop(self) -> None:
        """
        Main monitoring loop running continuously
        """
        while self.monitoring_status == MonitoringStatus.ACTIVE:
            try:
                # Update all portfolio risks
                for portfolio_id in self.active_portfolios.copy():
                    await self._update_portfolio_risk(portfolio_id)
                    await self.check_risk_breaches(portfolio_id)
                
                # Cache cleanup
                await self._cleanup_stale_data()
                
                # Wait for next update cycle
                await asyncio.sleep(self.update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)  # Brief pause before retry
    
    async def _initialize_caches(self) -> None:
        """
        Initialize position and portfolio caches
        """
        async with self.db_pool.acquire() as conn:
            for portfolio_id in self.active_portfolios:
                await self._update_position_cache(conn, portfolio_id)
                await self._update_portfolio_risk(portfolio_id)
    
    async def _update_position_cache(
        self, 
        conn: asyncpg.Connection, 
        portfolio_id: str
    ) -> None:
        """
        Update position cache with latest data
        """
        try:
            positions_query = """
                SELECT 
                    p.position_id,
                    p.portfolio_id,
                    p.strategy_id,
                    p.instrument_id,
                    i.symbol,
                    p.quantity,
                    p.avg_entry_price,
                    p.unrealized_pnl,
                    p.market_value,
                    p.updated_at,
                    i.multiplier,
                    i.asset_class
                FROM positions p
                JOIN instruments i ON p.instrument_id = i.instrument_id
                WHERE p.portfolio_id = $1
                AND p.quantity != 0
            """
            
            positions = await conn.fetch(positions_query, portfolio_id)
            
            # Get current market prices
            for position in positions:
                current_price = await self._get_current_price(
                    conn, position['instrument_id']
                )
                
                market_value = Decimal(str(position['quantity'])) * current_price
                cost_basis = (
                    Decimal(str(position['quantity'])) * 
                    Decimal(str(position['avg_entry_price']))
                )
                unrealized_pnl = market_value - cost_basis
                
                # Calculate risk metrics
                risk_exposure = abs(market_value)
                portfolio_weight = await self._calculate_portfolio_weight(
                    portfolio_id, market_value
                )
                
                position_risk = PositionRisk(
                    position_id=position['position_id'],
                    portfolio_id=position['portfolio_id'],
                    strategy_id=position['strategy_id'],
                    instrument_id=position['instrument_id'],
                    symbol=position['symbol'],
                    quantity=Decimal(str(position['quantity'])),
                    market_value=market_value,
                    unrealized_pnl=unrealized_pnl,
                    cost_basis=cost_basis,
                    current_price=current_price,
                    risk_exposure=risk_exposure,
                    var_contribution=Decimal('0'),  # Will be calculated separately
                    portfolio_weight=portfolio_weight,
                    concentration_risk=0.0,  # Will be calculated
                    liquidity_score=await self._get_liquidity_score(position['instrument_id']),
                    beta=await self._get_instrument_beta(position['instrument_id']),
                    last_updated=datetime.utcnow()
                )
                
                self.position_cache[position['position_id']] = position_risk
                
        except Exception as e:
            self.logger.error(f"Error updating position cache: {e}")
    
    async def _update_portfolio_risk(self, portfolio_id: str) -> None:
        """
        Update portfolio-level risk metrics
        """
        try:
            # Get positions for this portfolio
            portfolio_positions = [
                pos for pos in self.position_cache.values()
                if pos.portfolio_id == portfolio_id
            ]
            
            if not portfolio_positions:
                return
            
            # Calculate portfolio metrics
            total_value = sum(pos.market_value for pos in portfolio_positions)
            long_exposure = sum(
                pos.market_value for pos in portfolio_positions 
                if pos.quantity > 0
            )
            short_exposure = abs(sum(
                pos.market_value for pos in portfolio_positions 
                if pos.quantity < 0
            ))
            
            net_exposure = long_exposure - short_exposure
            gross_exposure = long_exposure + short_exposure
            total_exposure = abs(net_exposure)
            
            # Calculate VaR
            var_95, expected_shortfall = await self.calculate_portfolio_var(
                portfolio_id, 0.95
            )
            var_99, _ = await self.calculate_portfolio_var(portfolio_id, 0.99)
            
            # Risk level assessment
            risk_level = self._assess_risk_level(var_95, total_value)
            
            # Additional risk metrics
            concentration_risk = await self._calculate_concentration_risk(portfolio_positions)
            leverage_ratio = float(gross_exposure / total_value) if total_value > 0 else 0
            beta = await self._calculate_portfolio_beta(portfolio_positions)
            correlation_risk = await self._calculate_correlation_risk(portfolio_positions)
            liquidity_risk = await self._calculate_liquidity_risk(portfolio_positions)
            max_drawdown = await self._get_portfolio_max_drawdown(portfolio_id)
            
            portfolio_risk = PortfolioRisk(
                portfolio_id=portfolio_id,
                total_value=total_value,
                total_exposure=total_exposure,
                net_exposure=net_exposure,
                gross_exposure=gross_exposure,
                long_exposure=long_exposure,
                short_exposure=short_exposure,
                var_95=var_95,
                var_99=var_99,
                expected_shortfall=expected_shortfall,
                max_drawdown=max_drawdown,
                concentration_risk=concentration_risk,
                leverage_ratio=leverage_ratio,
                beta=beta,
                correlation_risk=correlation_risk,
                liquidity_risk=liquidity_risk,
                positions_count=len(portfolio_positions),
                risk_level=risk_level,
                last_updated=datetime.utcnow()
            )
            
            self.portfolio_cache[portfolio_id] = portfolio_risk
            
            # Update cache in Redis if available
            if self.redis_client:
                await self._update_redis_cache(portfolio_risk)
                
        except Exception as e:
            self.logger.error(f"Error updating portfolio risk for {portfolio_id}: {e}")
    
    async def _load_risk_thresholds(self, portfolio_ids: List[str]) -> None:
        """
        Load risk thresholds from database
        """
        async with self.db_pool.acquire() as conn:
            query = """
                SELECT * FROM risk_thresholds
                WHERE portfolio_id = ANY($1)
            """
            
            thresholds = await conn.fetch(query, portfolio_ids)
            
            for threshold in thresholds:
                self.thresholds[threshold['portfolio_id']] = MonitoringThresholds(
                    portfolio_id=threshold['portfolio_id'],
                    max_portfolio_var=Decimal(str(threshold['max_portfolio_var'])),
                    max_position_weight=float(threshold['max_position_weight']),
                    max_sector_concentration=float(threshold['max_sector_concentration']),
                    max_leverage_ratio=float(threshold['max_leverage_ratio']),
                    min_liquidity_score=float(threshold['min_liquidity_score']),
                    max_correlation_exposure=float(threshold['max_correlation_exposure']),
                    max_drawdown_limit=float(threshold['max_drawdown_limit']),
                    position_size_limit=Decimal(str(threshold['position_size_limit'])),
                    daily_loss_limit=Decimal(str(threshold['daily_loss_limit'])),
                    intraday_var_limit=Decimal(str(threshold['intraday_var_limit']))
                )
    
    async def _get_current_price(
        self, 
        conn: asyncpg.Connection, 
        instrument_id: str
    ) -> Decimal:
        """
        Get current market price for instrument
        """
        # Try to get latest tick
        tick_query = """
            SELECT price FROM market_ticks
            WHERE instrument_id = $1
            ORDER BY timestamp_ns DESC
            LIMIT 1
        """
        
        result = await conn.fetchrow(tick_query, instrument_id)
        if result:
            return Decimal(str(result['price']))
        
        # Fallback to latest bar close
        bar_query = """
            SELECT close_price FROM market_bars
            WHERE instrument_id = $1
            ORDER BY timestamp_ns DESC
            LIMIT 1
        """
        
        result = await conn.fetchrow(bar_query, instrument_id)
        if result:
            return Decimal(str(result['close_price']))
        
        # Final fallback
        return Decimal('100.0')  # Should not happen in production
    
    async def _create_alert(
        self,
        portfolio_id: str,
        alert_type: str,
        priority: AlertPriority,
        description: str,
        current_value: Decimal,
        threshold_value: Decimal,
        recommended_action: str,
        strategy_id: Optional[str] = None,
        position_id: Optional[str] = None
    ) -> RiskAlert:
        """
        Create risk alert
        """
        breach_magnitude = abs(current_value - threshold_value)
        breach_percentage = float(
            breach_magnitude / abs(threshold_value) * 100
        ) if threshold_value != 0 else 0
        
        risk_level = self._determine_alert_risk_level(priority, breach_percentage)
        
        alert = RiskAlert(
            alert_id=f"{alert_type}_{portfolio_id}_{int(datetime.utcnow().timestamp())}",
            portfolio_id=portfolio_id,
            strategy_id=strategy_id,
            position_id=position_id,
            alert_type=alert_type,
            priority=priority,
            risk_level=risk_level,
            description=description,
            current_value=current_value,
            threshold_value=threshold_value,
            breach_magnitude=breach_magnitude,
            breach_percentage=breach_percentage,
            recommended_action=recommended_action,
            auto_action_taken=False,
            metadata={},
            created_at=datetime.utcnow()
        )
        
        return alert
    
    async def _process_alert(self, alert: RiskAlert) -> None:
        """
        Process and distribute risk alert
        """
        # Store alert in database
        async with self.db_pool.acquire() as conn:
            await self._store_alert(conn, alert)
        
        # Trigger callbacks
        for callback in self.alert_callbacks:
            try:
                await callback(alert)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")
        
        # WebSocket notification (if available)
        if self.redis_client:
            await self._publish_alert_to_websocket(alert)
    
    def _assess_risk_level(
        self, 
        var_amount: Decimal, 
        total_value: Decimal
    ) -> RiskLevel:
        """
        Assess overall risk level based on VaR
        """
        if total_value == 0:
            return RiskLevel.LOW
            
        var_percentage = abs(float(var_amount / total_value)) * 100
        
        if var_percentage < 1:
            return RiskLevel.LOW
        elif var_percentage < 3:
            return RiskLevel.MEDIUM
        elif var_percentage < 5:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL
    
    # Additional helper methods would continue here...
    # (Due to length constraints, showing core structure)

# Global instance
risk_monitor = None

def get_risk_monitor() -> RealTimeRiskMonitor:
    """Get global risk monitor instance"""
    global risk_monitor
    if risk_monitor is None:
        raise RuntimeError("Risk monitor not initialized. Call init_risk_monitor() first.")
    return risk_monitor

def init_risk_monitor(
    db_pool: asyncpg.Pool, 
    redis_client: Optional[redis.Redis] = None
) -> RealTimeRiskMonitor:
    """Initialize global risk monitor instance"""
    global risk_monitor
    risk_monitor = RealTimeRiskMonitor(db_pool, redis_client)
    return risk_monitor