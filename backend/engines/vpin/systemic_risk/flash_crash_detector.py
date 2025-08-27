#!/usr/bin/env python3
"""
Flash Crash Early Warning System - Real-time Market Crash Detection
Implements multi-signal flash crash detection using microstructure indicators, volatility patterns, 
and network effects based on research from flash crash incidents.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict, deque
import statistics
import requests
from scipy import stats
from scipy.stats import zscore
import warnings

logger = logging.getLogger(__name__)

@dataclass
class FlashCrashConfig:
    """Configuration for flash crash detection system"""
    volatility_spike_threshold: float = 5.0  # Standard deviations above normal
    price_drop_threshold: float = 0.10  # 10% rapid price drop threshold
    volume_surge_threshold: float = 3.0  # Volume surge multiplier
    liquidity_drain_threshold: float = 0.80  # 80% reduction in liquidity
    time_window_seconds: int = 300  # 5-minute detection window
    rapid_decline_window_seconds: int = 60  # 1-minute for rapid movements
    network_contagion_threshold: float = 0.7  # Cross-asset contagion threshold
    market_depth_threshold: float = 0.5  # Market depth reduction threshold
    order_imbalance_threshold: float = 0.8  # Severe order imbalance threshold
    consecutive_drops_threshold: int = 5  # Consecutive price drops
    feature_engine_url: str = "http://localhost:8500"

@dataclass
class VolatilityMetrics:
    """Volatility-based flash crash indicators"""
    timestamp: float
    symbol: str
    current_volatility: float
    volatility_zscore: float  # Z-score relative to historical
    volatility_acceleration: float  # Rate of change in volatility
    garch_forecast: float  # GARCH model prediction
    realized_volatility: float  # High-frequency realized volatility
    volatility_risk_level: str  # LOW, MODERATE, HIGH, EXTREME

@dataclass
class LiquidityMetrics:
    """Liquidity-based flash crash indicators"""
    timestamp: float
    symbol: str
    bid_ask_spread: float
    market_depth_score: float  # Depth available at best bid/ask
    liquidity_ratio: float  # Current vs historical liquidity
    order_book_imbalance: float  # Buy/sell order imbalance
    trade_size_impact: float  # Price impact per trade size
    liquidity_risk_level: str  # LOW, MODERATE, HIGH, CRITICAL

@dataclass
class MomentumMetrics:
    """Price momentum and pattern-based indicators"""
    timestamp: float
    symbol: str
    price_momentum_1m: float  # 1-minute price momentum
    price_momentum_5m: float  # 5-minute price momentum
    consecutive_drops: int  # Number of consecutive price drops
    acceleration: float  # Price acceleration (second derivative)
    volume_momentum: float  # Volume momentum
    momentum_divergence: float  # Price vs volume momentum divergence
    momentum_risk_level: str  # LOW, MODERATE, HIGH, SEVERE

@dataclass
class NetworkEffects:
    """Cross-asset and network-based indicators"""
    timestamp: float
    correlated_assets_affected: int  # Number of correlated assets showing stress
    sector_contagion_score: float  # Sector-wide stress propagation
    market_wide_stress_score: float  # Overall market stress level
    correlation_breakdown: Dict[str, float]  # Correlation changes with other assets
    systemic_risk_score: float  # Overall systemic risk assessment
    network_risk_level: str  # LOW, MODERATE, HIGH, SYSTEMIC

@dataclass
class FlashCrashAlert:
    """Flash crash alert with risk assessment"""
    timestamp: float
    symbol: str
    alert_level: str  # WARNING, HIGH, CRITICAL, FLASH_CRASH_IMMINENT
    confidence_score: float  # 0-1 confidence in the alert
    time_to_potential_crash: Optional[int]  # Estimated seconds to potential crash
    primary_indicators: List[str]  # Main indicators triggering alert
    secondary_signals: List[str]  # Supporting signals
    recommended_actions: List[str]  # Recommended immediate actions
    affected_symbols: List[str]  # Other symbols likely to be affected
    risk_assessment: Dict[str, float]  # Detailed risk breakdown
    historical_context: str  # Context based on historical patterns

@dataclass
class FlashCrashSnapshot:
    """Complete flash crash detection snapshot"""
    timestamp: float
    volatility_metrics: Dict[str, VolatilityMetrics]
    liquidity_metrics: Dict[str, LiquidityMetrics]
    momentum_metrics: Dict[str, MomentumMetrics]
    network_effects: NetworkEffects
    active_alerts: List[FlashCrashAlert]
    system_stress_level: str  # NORMAL, ELEVATED, HIGH, CRITICAL
    market_regime: str  # NORMAL, STRESSED, PRE_CRASH, CRASH
    early_warning_signals: List[str]

class FlashCrashDetector:
    """Advanced flash crash detection system with multi-signal analysis"""
    
    def __init__(self, config: FlashCrashConfig = None):
        self.config = config or FlashCrashConfig()
        self.price_history = defaultdict(deque)
        self.volume_history = defaultdict(deque)
        self.volatility_history = defaultdict(deque)
        self.liquidity_history = defaultdict(deque)
        self.order_book_history = defaultdict(deque)
        self.alert_history = deque(maxlen=100)
        self.baseline_metrics = {}  # Baseline metrics for comparison
        self.correlation_matrix = {}
        self.active_monitoring = set()
        
    async def add_market_data(self, symbol: str, market_data: Dict[str, Any]) -> None:
        """Add real-time market data for analysis"""
        
        timestamp = market_data.get('timestamp', datetime.now().timestamp())
        
        # Store price data
        price = float(market_data.get('price', market_data.get('close', 0)))
        if price > 0:
            self.price_history[symbol].append((timestamp, price))
            
        # Store volume data
        volume = float(market_data.get('volume', 0))
        self.volume_history[symbol].append((timestamp, volume))
        
        # Store order book data if available
        if 'bid_price' in market_data and 'ask_price' in market_data:
            order_book_data = {
                'bid_price': float(market_data.get('bid_price', 0)),
                'ask_price': float(market_data.get('ask_price', 0)),
                'bid_size': float(market_data.get('bid_size', 0)),
                'ask_size': float(market_data.get('ask_size', 0))
            }
            self.order_book_history[symbol].append((timestamp, order_book_data))
        
        # Maintain rolling windows
        max_history = 1000
        for history in [self.price_history[symbol], self.volume_history[symbol], 
                       self.order_book_history[symbol]]:
            while len(history) > max_history:
                history.popleft()
        
        self.active_monitoring.add(symbol)
        
        # Update baseline metrics periodically
        await self._update_baseline_metrics(symbol)
    
    async def calculate_volatility_metrics(self, symbol: str) -> VolatilityMetrics:
        """Calculate volatility-based flash crash indicators"""
        
        if symbol not in self.price_history or len(self.price_history[symbol]) < 50:
            return self._create_default_volatility_metrics(symbol)
        
        price_data = list(self.price_history[symbol])
        prices = [p[1] for p in price_data]
        timestamps = [p[0] for p in price_data]
        
        # Calculate returns
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] > 0:
                ret = (prices[i] - prices[i-1]) / prices[i-1]
                returns.append(ret)
        
        if len(returns) < 20:
            return self._create_default_volatility_metrics(symbol)
        
        # Current volatility (rolling standard deviation)
        current_volatility = np.std(returns[-20:]) * np.sqrt(252)  # Annualized
        
        # Historical volatility for comparison
        if len(returns) >= 100:
            historical_volatility = np.std(returns[:-20]) * np.sqrt(252)
            volatility_zscore = (current_volatility - historical_volatility) / (historical_volatility + 1e-8)
        else:
            volatility_zscore = 0.0
        
        # Volatility acceleration (change in volatility)
        if len(returns) >= 40:
            prev_volatility = np.std(returns[-40:-20]) * np.sqrt(252)
            volatility_acceleration = (current_volatility - prev_volatility) / (prev_volatility + 1e-8)
        else:
            volatility_acceleration = 0.0
        
        # Realized volatility (high-frequency)
        realized_volatility = await self._calculate_realized_volatility(returns[-60:])
        
        # Simple GARCH forecast
        garch_forecast = await self._simple_garch_forecast(returns)
        
        # Risk level classification
        volatility_risk_level = self._classify_volatility_risk(volatility_zscore, volatility_acceleration)
        
        # Store in history
        metrics = VolatilityMetrics(
            timestamp=datetime.now().timestamp(),
            symbol=symbol,
            current_volatility=current_volatility,
            volatility_zscore=volatility_zscore,
            volatility_acceleration=volatility_acceleration,
            garch_forecast=garch_forecast,
            realized_volatility=realized_volatility,
            volatility_risk_level=volatility_risk_level
        )
        
        self.volatility_history[symbol].append(metrics)
        if len(self.volatility_history[symbol]) > 200:
            self.volatility_history[symbol].popleft()
        
        return metrics
    
    async def calculate_liquidity_metrics(self, symbol: str) -> LiquidityMetrics:
        """Calculate liquidity-based flash crash indicators"""
        
        if symbol not in self.order_book_history or len(self.order_book_history[symbol]) < 10:
            return self._create_default_liquidity_metrics(symbol)
        
        order_book_data = list(self.order_book_history[symbol])
        recent_books = [ob[1] for ob in order_book_data[-20:]]
        
        # Calculate bid-ask spread
        current_book = recent_books[-1]
        bid_price = current_book['bid_price']
        ask_price = current_book['ask_price']
        
        if bid_price > 0 and ask_price > 0:
            bid_ask_spread = (ask_price - bid_price) / ((ask_price + bid_price) / 2)
        else:
            bid_ask_spread = 0.0
        
        # Market depth score
        bid_size = current_book['bid_size']
        ask_size = current_book['ask_size']
        market_depth_score = min(bid_size, ask_size) / (max(bid_size, ask_size) + 1e-8)
        
        # Liquidity ratio (current vs historical)
        historical_spreads = []
        historical_sizes = []
        
        for book in recent_books[:-5]:  # Historical comparison
            if book['bid_price'] > 0 and book['ask_price'] > 0:
                spread = (book['ask_price'] - book['bid_price']) / ((book['ask_price'] + book['bid_price']) / 2)
                size = (book['bid_size'] + book['ask_size']) / 2
                historical_spreads.append(spread)
                historical_sizes.append(size)
        
        if historical_spreads and historical_sizes:
            avg_historical_spread = statistics.mean(historical_spreads)
            avg_historical_size = statistics.mean(historical_sizes)
            current_size = (bid_size + ask_size) / 2
            
            # Liquidity ratio: higher is better
            liquidity_ratio = (avg_historical_size / (current_size + 1e-8)) * (avg_historical_spread / (bid_ask_spread + 1e-8))
        else:
            liquidity_ratio = 1.0
        
        # Order book imbalance
        total_size = bid_size + ask_size
        if total_size > 0:
            order_book_imbalance = abs(bid_size - ask_size) / total_size
        else:
            order_book_imbalance = 0.0
        
        # Trade size impact (proxy using recent price movements and volumes)
        trade_size_impact = await self._estimate_trade_size_impact(symbol)
        
        # Risk level classification
        liquidity_risk_level = self._classify_liquidity_risk(liquidity_ratio, bid_ask_spread, order_book_imbalance)
        
        metrics = LiquidityMetrics(
            timestamp=datetime.now().timestamp(),
            symbol=symbol,
            bid_ask_spread=bid_ask_spread,
            market_depth_score=market_depth_score,
            liquidity_ratio=liquidity_ratio,
            order_book_imbalance=order_book_imbalance,
            trade_size_impact=trade_size_impact,
            liquidity_risk_level=liquidity_risk_level
        )
        
        self.liquidity_history[symbol].append(metrics)
        if len(self.liquidity_history[symbol]) > 200:
            self.liquidity_history[symbol].popleft()
        
        return metrics
    
    async def calculate_momentum_metrics(self, symbol: str) -> MomentumMetrics:
        """Calculate price momentum and pattern-based indicators"""
        
        if symbol not in self.price_history or len(self.price_history[symbol]) < 20:
            return self._create_default_momentum_metrics(symbol)
        
        price_data = list(self.price_history[symbol])
        volume_data = list(self.volume_history[symbol])
        
        current_time = datetime.now().timestamp()
        
        # Extract recent data
        recent_prices = [(ts, price) for ts, price in price_data if current_time - ts <= 300]  # Last 5 minutes
        recent_volumes = [(ts, vol) for ts, vol in volume_data if current_time - ts <= 300]
        
        if len(recent_prices) < 10:
            return self._create_default_momentum_metrics(symbol)
        
        prices = [p[1] for p in recent_prices]
        volumes = [v[1] for v in recent_volumes if v[1] > 0]
        
        # 1-minute momentum
        one_min_ago_time = current_time - 60
        one_min_prices = [p for ts, p in recent_prices if ts >= one_min_ago_time]
        if len(one_min_prices) >= 2:
            price_momentum_1m = (one_min_prices[-1] - one_min_prices[0]) / (one_min_prices[0] + 1e-8)
        else:
            price_momentum_1m = 0.0
        
        # 5-minute momentum
        if len(prices) >= 2:
            price_momentum_5m = (prices[-1] - prices[0]) / (prices[0] + 1e-8)
        else:
            price_momentum_5m = 0.0
        
        # Consecutive drops detection
        consecutive_drops = self._count_consecutive_drops(prices)
        
        # Price acceleration (second derivative)
        acceleration = self._calculate_price_acceleration(recent_prices)
        
        # Volume momentum
        if len(volumes) >= 2:
            volume_momentum = (volumes[-1] - volumes[0]) / (volumes[0] + 1e-8)
        else:
            volume_momentum = 0.0
        
        # Momentum divergence (price vs volume)
        momentum_divergence = self._calculate_momentum_divergence(price_momentum_1m, volume_momentum)
        
        # Risk level classification
        momentum_risk_level = self._classify_momentum_risk(
            price_momentum_1m, consecutive_drops, acceleration
        )
        
        return MomentumMetrics(
            timestamp=datetime.now().timestamp(),
            symbol=symbol,
            price_momentum_1m=price_momentum_1m,
            price_momentum_5m=price_momentum_5m,
            consecutive_drops=consecutive_drops,
            acceleration=acceleration,
            volume_momentum=volume_momentum,
            momentum_divergence=momentum_divergence,
            momentum_risk_level=momentum_risk_level
        )
    
    async def calculate_network_effects(self, symbols: List[str]) -> NetworkEffects:
        """Calculate cross-asset and network-based indicators"""
        
        if len(symbols) < 2:
            return self._create_default_network_effects()
        
        # Count assets showing stress
        stressed_assets = 0
        total_stress_score = 0.0
        
        correlation_breakdown = {}
        
        for symbol in symbols:
            if symbol in self.active_monitoring:
                # Calculate stress indicators
                volatility_stress = await self._get_volatility_stress_score(symbol)
                liquidity_stress = await self._get_liquidity_stress_score(symbol)
                momentum_stress = await self._get_momentum_stress_score(symbol)
                
                asset_stress = max(volatility_stress, liquidity_stress, momentum_stress)
                total_stress_score += asset_stress
                
                if asset_stress > 0.6:  # High stress threshold
                    stressed_assets += 1
                
                # Calculate correlation changes
                correlation_breakdown[symbol] = await self._calculate_correlation_change(symbol, symbols)
        
        # Calculate metrics
        correlated_assets_affected = stressed_assets
        if len(symbols) > 0:
            market_wide_stress_score = total_stress_score / len(symbols)
        else:
            market_wide_stress_score = 0.0
        
        # Sector contagion score (simplified)
        sector_contagion_score = min(stressed_assets / len(symbols), 1.0) if len(symbols) > 0 else 0.0
        
        # Systemic risk score
        systemic_risk_score = self._calculate_systemic_risk_score(
            market_wide_stress_score, sector_contagion_score, correlated_assets_affected
        )
        
        # Network risk level
        network_risk_level = self._classify_network_risk(systemic_risk_score, correlated_assets_affected)
        
        return NetworkEffects(
            timestamp=datetime.now().timestamp(),
            correlated_assets_affected=correlated_assets_affected,
            sector_contagion_score=sector_contagion_score,
            market_wide_stress_score=market_wide_stress_score,
            correlation_breakdown=correlation_breakdown,
            systemic_risk_score=systemic_risk_score,
            network_risk_level=network_risk_level
        )
    
    async def generate_flash_crash_alert(self, symbol: str, 
                                       volatility_metrics: VolatilityMetrics,
                                       liquidity_metrics: LiquidityMetrics,
                                       momentum_metrics: MomentumMetrics,
                                       network_effects: NetworkEffects) -> Optional[FlashCrashAlert]:
        """Generate flash crash alert based on all indicators"""
        
        # Calculate composite risk scores
        volatility_risk_score = self._get_risk_score(volatility_metrics.volatility_risk_level)
        liquidity_risk_score = self._get_risk_score(liquidity_metrics.liquidity_risk_level)
        momentum_risk_score = self._get_risk_score(momentum_metrics.momentum_risk_level)
        network_risk_score = self._get_risk_score(network_effects.network_risk_level)
        
        # Weighted composite score
        composite_score = (
            0.30 * volatility_risk_score +
            0.25 * liquidity_risk_score +
            0.25 * momentum_risk_score +
            0.20 * network_risk_score
        )
        
        # Determine alert level
        if composite_score < 0.3:
            return None  # No alert needed
        
        alert_level = self._determine_alert_level(composite_score)
        
        # Confidence score calculation
        confidence_score = self._calculate_confidence_score(
            volatility_metrics, liquidity_metrics, momentum_metrics, network_effects
        )
        
        # Time to potential crash estimation
        time_to_crash = await self._estimate_crash_timing(
            volatility_metrics, momentum_metrics
        ) if composite_score > 0.7 else None
        
        # Identify primary indicators
        primary_indicators = []
        if volatility_risk_score > 0.6:
            primary_indicators.append(f"Extreme volatility (Z-score: {volatility_metrics.volatility_zscore:.2f})")
        if liquidity_risk_score > 0.6:
            primary_indicators.append(f"Liquidity crisis (ratio: {liquidity_metrics.liquidity_ratio:.2f})")
        if momentum_risk_score > 0.6:
            primary_indicators.append(f"Severe momentum ({momentum_metrics.consecutive_drops} consecutive drops)")
        if network_risk_score > 0.6:
            primary_indicators.append(f"Network contagion ({network_effects.correlated_assets_affected} assets affected)")
        
        # Secondary signals
        secondary_signals = []
        if volatility_metrics.volatility_acceleration > 2.0:
            secondary_signals.append("Accelerating volatility")
        if liquidity_metrics.order_book_imbalance > 0.8:
            secondary_signals.append("Severe order book imbalance")
        if momentum_metrics.momentum_divergence > 0.5:
            secondary_signals.append("Price-volume momentum divergence")
        
        # Recommended actions
        recommended_actions = self._generate_recommended_actions(alert_level, composite_score)
        
        # Identify likely affected symbols
        affected_symbols = await self._identify_affected_symbols(symbol, network_effects)
        
        # Risk assessment breakdown
        risk_assessment = {
            "volatility_risk": volatility_risk_score,
            "liquidity_risk": liquidity_risk_score,
            "momentum_risk": momentum_risk_score,
            "network_risk": network_risk_score,
            "composite_score": composite_score
        }
        
        # Historical context
        historical_context = await self._get_historical_context(symbol, composite_score)
        
        alert = FlashCrashAlert(
            timestamp=datetime.now().timestamp(),
            symbol=symbol,
            alert_level=alert_level,
            confidence_score=confidence_score,
            time_to_potential_crash=time_to_crash,
            primary_indicators=primary_indicators,
            secondary_signals=secondary_signals,
            recommended_actions=recommended_actions,
            affected_symbols=affected_symbols,
            risk_assessment=risk_assessment,
            historical_context=historical_context
        )
        
        # Store alert
        self.alert_history.append(alert)
        
        return alert
    
    async def generate_system_snapshot(self, symbols: List[str]) -> FlashCrashSnapshot:
        """Generate complete flash crash detection system snapshot"""
        
        # Calculate metrics for all symbols
        volatility_metrics = {}
        liquidity_metrics = {}
        momentum_metrics = {}
        
        for symbol in symbols:
            if symbol in self.active_monitoring:
                volatility_metrics[symbol] = await self.calculate_volatility_metrics(symbol)
                liquidity_metrics[symbol] = await self.calculate_liquidity_metrics(symbol)
                momentum_metrics[symbol] = await self.calculate_momentum_metrics(symbol)
        
        # Calculate network effects
        network_effects = await self.calculate_network_effects(symbols)
        
        # Generate alerts
        active_alerts = []
        for symbol in symbols:
            if symbol in volatility_metrics:
                alert = await self.generate_flash_crash_alert(
                    symbol,
                    volatility_metrics[symbol],
                    liquidity_metrics[symbol],
                    momentum_metrics[symbol],
                    network_effects
                )
                if alert:
                    active_alerts.append(alert)
        
        # Determine system stress level
        system_stress_level = self._determine_system_stress_level(active_alerts, network_effects)
        
        # Determine market regime
        market_regime = self._determine_market_regime(system_stress_level, network_effects)
        
        # Detect early warning signals
        early_warning_signals = self._detect_early_warning_signals(
            volatility_metrics, liquidity_metrics, momentum_metrics, network_effects
        )
        
        return FlashCrashSnapshot(
            timestamp=datetime.now().timestamp(),
            volatility_metrics=volatility_metrics,
            liquidity_metrics=liquidity_metrics,
            momentum_metrics=momentum_metrics,
            network_effects=network_effects,
            active_alerts=active_alerts,
            system_stress_level=system_stress_level,
            market_regime=market_regime,
            early_warning_signals=early_warning_signals
        )
    
    # Helper methods
    async def _update_baseline_metrics(self, symbol: str) -> None:
        """Update baseline metrics for comparison"""
        
        if symbol not in self.baseline_metrics:
            self.baseline_metrics[symbol] = {}
        
        # Update periodically (every 100 data points)
        if len(self.price_history[symbol]) % 100 == 0:
            prices = [p[1] for p in list(self.price_history[symbol])[-500:]]  # Last 500 prices
            if len(prices) > 50:
                self.baseline_metrics[symbol]['avg_volatility'] = np.std(np.diff(prices)) * np.sqrt(252)
                self.baseline_metrics[symbol]['avg_volume'] = statistics.mean(
                    [v[1] for v in list(self.volume_history[symbol])[-500:]]
                )
    
    async def _calculate_realized_volatility(self, returns: List[float]) -> float:
        """Calculate high-frequency realized volatility"""
        if len(returns) < 2:
            return 0.0
        return np.sqrt(np.sum([r**2 for r in returns])) * np.sqrt(252)
    
    async def _simple_garch_forecast(self, returns: List[float]) -> float:
        """Simple GARCH(1,1) volatility forecast"""
        if len(returns) < 20:
            return np.std(returns) * np.sqrt(252) if returns else 0.0
        
        # Simplified GARCH parameters
        alpha0 = 0.001
        alpha1 = 0.05
        beta1 = 0.9
        
        # Initialize with sample variance
        variance = np.var(returns[:10])
        
        # GARCH iteration
        for ret in returns[10:]:
            variance = alpha0 + alpha1 * ret**2 + beta1 * variance
        
        return np.sqrt(variance * 252)  # Annualized
    
    def _classify_volatility_risk(self, zscore: float, acceleration: float) -> str:
        """Classify volatility risk level"""
        if zscore > self.config.volatility_spike_threshold or acceleration > 3.0:
            return "EXTREME"
        elif zscore > 3.0 or acceleration > 2.0:
            return "HIGH"
        elif zscore > 2.0 or acceleration > 1.0:
            return "MODERATE"
        else:
            return "LOW"
    
    def _classify_liquidity_risk(self, liquidity_ratio: float, spread: float, imbalance: float) -> str:
        """Classify liquidity risk level"""
        if (liquidity_ratio > 3.0 or spread > 0.05 or 
            imbalance > self.config.order_imbalance_threshold):
            return "CRITICAL"
        elif liquidity_ratio > 2.0 or spread > 0.02 or imbalance > 0.6:
            return "HIGH"
        elif liquidity_ratio > 1.5 or spread > 0.01 or imbalance > 0.4:
            return "MODERATE"
        else:
            return "LOW"
    
    def _classify_momentum_risk(self, momentum_1m: float, consecutive_drops: int, acceleration: float) -> str:
        """Classify momentum risk level"""
        if (abs(momentum_1m) > self.config.price_drop_threshold or 
            consecutive_drops >= self.config.consecutive_drops_threshold or
            abs(acceleration) > 0.1):
            return "SEVERE"
        elif abs(momentum_1m) > 0.05 or consecutive_drops >= 3 or abs(acceleration) > 0.05:
            return "HIGH"
        elif abs(momentum_1m) > 0.02 or consecutive_drops >= 2:
            return "MODERATE"
        else:
            return "LOW"
    
    def _classify_network_risk(self, systemic_score: float, affected_assets: int) -> str:
        """Classify network risk level"""
        if systemic_score > 0.8 and affected_assets > 5:
            return "SYSTEMIC"
        elif systemic_score > 0.6 or affected_assets > 3:
            return "HIGH"
        elif systemic_score > 0.4 or affected_assets > 1:
            return "MODERATE"
        else:
            return "LOW"
    
    def _count_consecutive_drops(self, prices: List[float]) -> int:
        """Count consecutive price drops"""
        if len(prices) < 2:
            return 0
        
        consecutive = 0
        max_consecutive = 0
        
        for i in range(1, len(prices)):
            if prices[i] < prices[i-1]:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0
        
        return max_consecutive
    
    def _calculate_price_acceleration(self, price_data: List[Tuple[float, float]]) -> float:
        """Calculate price acceleration (second derivative)"""
        if len(price_data) < 3:
            return 0.0
        
        # Calculate velocities (first derivative)
        velocities = []
        for i in range(1, len(price_data)):
            time_diff = price_data[i][0] - price_data[i-1][0]
            price_diff = price_data[i][1] - price_data[i-1][1]
            if time_diff > 0:
                velocity = price_diff / time_diff
                velocities.append(velocity)
        
        if len(velocities) < 2:
            return 0.0
        
        # Calculate acceleration (second derivative)
        accelerations = []
        for i in range(1, len(velocities)):
            acceleration = velocities[i] - velocities[i-1]
            accelerations.append(acceleration)
        
        return statistics.mean(accelerations) if accelerations else 0.0
    
    def _calculate_momentum_divergence(self, price_momentum: float, volume_momentum: float) -> float:
        """Calculate divergence between price and volume momentum"""
        if price_momentum == 0 and volume_momentum == 0:
            return 0.0
        
        # Normalize momentums to compare
        price_sign = 1 if price_momentum > 0 else -1 if price_momentum < 0 else 0
        volume_sign = 1 if volume_momentum > 0 else -1 if volume_momentum < 0 else 0
        
        # Divergence occurs when signs are opposite
        if price_sign != 0 and volume_sign != 0 and price_sign != volume_sign:
            return abs(price_momentum - volume_momentum) / (abs(price_momentum) + abs(volume_momentum) + 1e-8)
        
        return 0.0
    
    async def _estimate_trade_size_impact(self, symbol: str) -> float:
        """Estimate price impact per trade size"""
        if (symbol not in self.price_history or symbol not in self.volume_history or 
            len(self.price_history[symbol]) < 10):
            return 0.0
        
        price_data = list(self.price_history[symbol])[-20:]
        volume_data = list(self.volume_history[symbol])[-20:]
        
        # Calculate price changes and corresponding volumes
        impacts = []
        for i in range(1, min(len(price_data), len(volume_data))):
            if (price_data[i-1][1] > 0 and volume_data[i][1] > 0):
                price_change = abs(price_data[i][1] - price_data[i-1][1]) / price_data[i-1][1]
                volume = volume_data[i][1]
                if volume > 0:
                    impact_per_unit = price_change / np.sqrt(volume)  # Kyle's lambda style
                    impacts.append(impact_per_unit)
        
        return statistics.mean(impacts) if impacts else 0.0
    
    async def _get_volatility_stress_score(self, symbol: str) -> float:
        """Get volatility stress score for network analysis"""
        if symbol not in self.volatility_history or not self.volatility_history[symbol]:
            return 0.0
        
        latest = list(self.volatility_history[symbol])[-1]
        risk_scores = {"LOW": 0.1, "MODERATE": 0.4, "HIGH": 0.7, "EXTREME": 1.0}
        return risk_scores.get(latest.volatility_risk_level, 0.0)
    
    async def _get_liquidity_stress_score(self, symbol: str) -> float:
        """Get liquidity stress score for network analysis"""
        if symbol not in self.liquidity_history or not self.liquidity_history[symbol]:
            return 0.0
        
        latest = list(self.liquidity_history[symbol])[-1]
        risk_scores = {"LOW": 0.1, "MODERATE": 0.4, "HIGH": 0.7, "CRITICAL": 1.0}
        return risk_scores.get(latest.liquidity_risk_level, 0.0)
    
    async def _get_momentum_stress_score(self, symbol: str) -> float:
        """Get momentum stress score for network analysis"""
        # This would use momentum metrics if we had them stored
        # For now, return a simplified calculation based on price history
        if symbol not in self.price_history or len(self.price_history[symbol]) < 5:
            return 0.0
        
        recent_prices = [p[1] for p in list(self.price_history[symbol])[-5:]]
        if len(recent_prices) < 2:
            return 0.0
        
        # Simple momentum stress based on recent price volatility
        returns = [(recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1] 
                  for i in range(1, len(recent_prices)) if recent_prices[i-1] > 0]
        
        if returns:
            volatility = np.std(returns)
            return min(volatility * 10, 1.0)  # Scale to 0-1
        
        return 0.0
    
    async def _calculate_correlation_change(self, symbol: str, all_symbols: List[str]) -> float:
        """Calculate correlation change for network effects"""
        # Simplified correlation calculation
        # In practice, would calculate rolling correlations with other assets
        return 0.0  # Placeholder
    
    def _calculate_systemic_risk_score(self, market_stress: float, contagion: float, affected: int) -> float:
        """Calculate overall systemic risk score"""
        # Weight factors
        stress_weight = 0.4
        contagion_weight = 0.4
        affected_weight = 0.2
        
        # Normalize affected assets (assume max 10 assets)
        normalized_affected = min(affected / 10, 1.0)
        
        systemic_score = (
            stress_weight * market_stress +
            contagion_weight * contagion +
            affected_weight * normalized_affected
        )
        
        return min(systemic_score, 1.0)
    
    def _get_risk_score(self, risk_level: str) -> float:
        """Convert risk level to numerical score"""
        risk_mapping = {
            "LOW": 0.1,
            "MODERATE": 0.4,
            "HIGH": 0.7,
            "SEVERE": 0.9,
            "CRITICAL": 0.95,
            "EXTREME": 1.0,
            "SYSTEMIC": 1.0
        }
        return risk_mapping.get(risk_level, 0.0)
    
    def _determine_alert_level(self, composite_score: float) -> str:
        """Determine alert level from composite score"""
        if composite_score > 0.9:
            return "FLASH_CRASH_IMMINENT"
        elif composite_score > 0.7:
            return "CRITICAL"
        elif composite_score > 0.5:
            return "HIGH"
        else:
            return "WARNING"
    
    def _calculate_confidence_score(self, vol_metrics: VolatilityMetrics, 
                                  liq_metrics: LiquidityMetrics,
                                  mom_metrics: MomentumMetrics,
                                  net_effects: NetworkEffects) -> float:
        """Calculate confidence score for the alert"""
        
        # Multiple independent signals increase confidence
        signal_count = 0
        if vol_metrics.volatility_risk_level in ["HIGH", "EXTREME"]:
            signal_count += 1
        if liq_metrics.liquidity_risk_level in ["HIGH", "CRITICAL"]:
            signal_count += 1
        if mom_metrics.momentum_risk_level in ["HIGH", "SEVERE"]:
            signal_count += 1
        if net_effects.network_risk_level in ["HIGH", "SYSTEMIC"]:
            signal_count += 1
        
        # Base confidence from signal count
        base_confidence = signal_count / 4.0
        
        # Boost confidence if multiple extreme values
        extreme_boost = 0.0
        if vol_metrics.volatility_zscore > 5.0:
            extreme_boost += 0.1
        if liq_metrics.liquidity_ratio > 4.0:
            extreme_boost += 0.1
        if mom_metrics.consecutive_drops >= 5:
            extreme_boost += 0.1
        
        return min(base_confidence + extreme_boost, 1.0)
    
    async def _estimate_crash_timing(self, vol_metrics: VolatilityMetrics,
                                   mom_metrics: MomentumMetrics) -> Optional[int]:
        """Estimate time to potential crash in seconds"""
        
        # Based on acceleration and current momentum
        if mom_metrics.acceleration != 0 and mom_metrics.price_momentum_1m < -0.05:
            # Estimate time for 10% drop at current acceleration
            target_drop = -0.10
            current_momentum = mom_metrics.price_momentum_1m
            acceleration = mom_metrics.acceleration
            
            if acceleration < 0:  # Accelerating downward
                # Quadratic formula: target = current + acceleration * t^2
                discriminant = current_momentum**2 - 4 * acceleration * target_drop
                if discriminant > 0:
                    time_estimate = (-current_momentum - np.sqrt(discriminant)) / (2 * acceleration)
                    return max(int(time_estimate * 60), 30)  # At least 30 seconds
        
        return None
    
    def _generate_recommended_actions(self, alert_level: str, composite_score: float) -> List[str]:
        """Generate recommended immediate actions"""
        
        actions = []
        
        if alert_level == "FLASH_CRASH_IMMINENT":
            actions.extend([
                "IMMEDIATE: Halt trading operations",
                "IMMEDIATE: Reduce position sizes by 75%+",
                "IMMEDIATE: Activate emergency liquidity protocols",
                "IMMEDIATE: Cancel all pending orders",
                "IMMEDIATE: Notify risk management team"
            ])
        elif alert_level == "CRITICAL":
            actions.extend([
                "Reduce position sizes by 50%",
                "Tighten stop-loss orders",
                "Increase monitoring frequency to real-time",
                "Prepare for potential volatility spike",
                "Review correlation with other holdings"
            ])
        elif alert_level == "HIGH":
            actions.extend([
                "Reduce position sizes by 25%",
                "Review risk exposure",
                "Increase monitoring frequency",
                "Consider hedging strategies"
            ])
        else:  # WARNING
            actions.extend([
                "Monitor closely for deterioration",
                "Review position sizing",
                "Prepare contingency plans"
            ])
        
        return actions
    
    async def _identify_affected_symbols(self, primary_symbol: str, network_effects: NetworkEffects) -> List[str]:
        """Identify other symbols likely to be affected"""
        
        affected = []
        
        # Check correlation breakdown
        for symbol, correlation_change in network_effects.correlation_breakdown.items():
            if symbol != primary_symbol and correlation_change > 0.5:
                affected.append(symbol)
        
        # Add high-beta or sector-related symbols (would be configured)
        # This is a placeholder - in practice would use sector mappings
        
        return affected[:5]  # Top 5 most likely affected
    
    async def _get_historical_context(self, symbol: str, composite_score: float) -> str:
        """Get historical context for the alert"""
        
        if composite_score > 0.9:
            return "Extreme conditions rarely seen - similar to major market stress events"
        elif composite_score > 0.7:
            return "High stress conditions - comparable to significant market corrections"
        elif composite_score > 0.5:
            return "Elevated risk conditions - monitor for escalation"
        else:
            return "Early warning stage - conditions bear watching"
    
    def _determine_system_stress_level(self, alerts: List[FlashCrashAlert], 
                                     network_effects: NetworkEffects) -> str:
        """Determine overall system stress level"""
        
        if any(alert.alert_level == "FLASH_CRASH_IMMINENT" for alert in alerts):
            return "CRITICAL"
        elif (len([a for a in alerts if a.alert_level in ["CRITICAL", "HIGH"]]) > 2 or
              network_effects.network_risk_level == "SYSTEMIC"):
            return "HIGH"
        elif len(alerts) > 1 or network_effects.systemic_risk_score > 0.5:
            return "ELEVATED"
        else:
            return "NORMAL"
    
    def _determine_market_regime(self, stress_level: str, network_effects: NetworkEffects) -> str:
        """Determine current market regime"""
        
        if stress_level == "CRITICAL":
            return "CRASH"
        elif stress_level == "HIGH" and network_effects.systemic_risk_score > 0.7:
            return "PRE_CRASH"
        elif stress_level in ["HIGH", "ELEVATED"]:
            return "STRESSED"
        else:
            return "NORMAL"
    
    def _detect_early_warning_signals(self, vol_metrics: Dict[str, VolatilityMetrics],
                                    liq_metrics: Dict[str, LiquidityMetrics],
                                    mom_metrics: Dict[str, MomentumMetrics],
                                    network_effects: NetworkEffects) -> List[str]:
        """Detect early warning signals across the system"""
        
        warnings = []
        
        # Volatility warnings
        extreme_vol_count = sum(1 for m in vol_metrics.values() 
                               if m.volatility_risk_level == "EXTREME")
        if extreme_vol_count > 0:
            warnings.append(f"Extreme volatility detected in {extreme_vol_count} assets")
        
        # Liquidity warnings
        liquidity_crisis_count = sum(1 for m in liq_metrics.values()
                                   if m.liquidity_risk_level == "CRITICAL")
        if liquidity_crisis_count > 0:
            warnings.append(f"Liquidity crisis in {liquidity_crisis_count} assets")
        
        # Momentum warnings
        severe_momentum_count = sum(1 for m in mom_metrics.values()
                                  if m.momentum_risk_level == "SEVERE")
        if severe_momentum_count > 0:
            warnings.append(f"Severe momentum stress in {severe_momentum_count} assets")
        
        # Network warnings
        if network_effects.network_risk_level == "SYSTEMIC":
            warnings.append("Systemic risk detected - network-wide stress")
        
        # Cross-signal warnings
        if (extreme_vol_count > 0 and liquidity_crisis_count > 0 and 
            network_effects.correlated_assets_affected > 2):
            warnings.append("Multi-signal convergence - heightened crash risk")
        
        return warnings
    
    # Default metrics creation
    def _create_default_volatility_metrics(self, symbol: str) -> VolatilityMetrics:
        """Create default volatility metrics"""
        return VolatilityMetrics(
            timestamp=datetime.now().timestamp(),
            symbol=symbol,
            current_volatility=0.0,
            volatility_zscore=0.0,
            volatility_acceleration=0.0,
            garch_forecast=0.0,
            realized_volatility=0.0,
            volatility_risk_level="INSUFFICIENT_DATA"
        )
    
    def _create_default_liquidity_metrics(self, symbol: str) -> LiquidityMetrics:
        """Create default liquidity metrics"""
        return LiquidityMetrics(
            timestamp=datetime.now().timestamp(),
            symbol=symbol,
            bid_ask_spread=0.0,
            market_depth_score=0.0,
            liquidity_ratio=1.0,
            order_book_imbalance=0.0,
            trade_size_impact=0.0,
            liquidity_risk_level="INSUFFICIENT_DATA"
        )
    
    def _create_default_momentum_metrics(self, symbol: str) -> MomentumMetrics:
        """Create default momentum metrics"""
        return MomentumMetrics(
            timestamp=datetime.now().timestamp(),
            symbol=symbol,
            price_momentum_1m=0.0,
            price_momentum_5m=0.0,
            consecutive_drops=0,
            acceleration=0.0,
            volume_momentum=0.0,
            momentum_divergence=0.0,
            momentum_risk_level="INSUFFICIENT_DATA"
        )
    
    def _create_default_network_effects(self) -> NetworkEffects:
        """Create default network effects"""
        return NetworkEffects(
            timestamp=datetime.now().timestamp(),
            correlated_assets_affected=0,
            sector_contagion_score=0.0,
            market_wide_stress_score=0.0,
            correlation_breakdown={},
            systemic_risk_score=0.0,
            network_risk_level="INSUFFICIENT_DATA"
        )

# Global detector instance
flash_crash_detector = FlashCrashDetector()