#!/usr/bin/env python3
"""
Spread-Based Toxicity Analyzer
Detects adverse selection through bid-ask spread analysis with effective/realized spread decomposition.
Research shows 19.85% economically significant increase in spreads following informed trading.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import statistics
import requests
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)

@dataclass
class SpreadToxicityConfig:
    """Configuration for spread-based toxicity analysis"""
    analysis_window_minutes: int = 15
    baseline_window_minutes: int = 60
    significance_threshold: float = 0.05  # 5% increase threshold
    major_threshold: float = 0.1985  # 19.85% research-based threshold
    extreme_threshold: float = 0.30  # 30% extreme threshold
    min_observations: int = 30
    feature_engine_url: str = "http://localhost:8500"

@dataclass
class SpreadMetrics:
    """Individual spread calculation metrics"""
    timestamp: float
    symbol: str
    bid_price: float
    ask_price: float
    trade_price: Optional[float]
    midpoint: float
    quoted_spread: float  # Ask - Bid
    quoted_spread_bps: float  # In basis points
    effective_spread: Optional[float]  # 2 * |Trade Price - Midpoint|
    effective_spread_bps: Optional[float]
    realized_spread: Optional[float]  # 2 * (Trade Price - Future Midpoint) * Direction
    realized_spread_bps: Optional[float]
    price_impact: Optional[float]  # Effective Spread - Realized Spread
    price_impact_bps: Optional[float]

@dataclass
class SpreadToxicityAnalysis:
    """Comprehensive spread toxicity analysis results"""
    timestamp: float
    symbol: str
    current_metrics: SpreadMetrics
    baseline_spread: float  # Average spread over baseline window
    current_spread_increase: float  # Percentage increase from baseline
    toxicity_level: str  # LOW, MODERATE, HIGH, EXTREME
    informed_trading_probability: float  # 0-1 probability of informed trading
    adverse_selection_indicator: float  # Composite indicator
    spread_decomposition: Dict[str, float]  # Temporary vs permanent components
    market_quality_assessment: str  # GOOD, FAIR, POOR, CRITICAL
    statistical_significance: bool
    trend_analysis: str  # INCREASING, DECREASING, STABLE
    alert_triggered: bool

class SpreadToxicityAnalyzer:
    """
    Analyzes bid-ask spreads to detect informed trading and adverse selection.
    Decomposes spreads into temporary (inventory) and permanent (information) components.
    """
    
    def __init__(self, config: SpreadToxicityConfig = None):
        self.config = config or SpreadToxicityConfig()
        self.spread_history = []
        self.toxicity_history = []
        self.baseline_spreads = {}
        
    async def analyze_spread_toxicity(self, symbol: str, market_data: Dict[str, Any]) -> SpreadToxicityAnalysis:
        """Perform comprehensive spread toxicity analysis"""
        
        # Calculate current spread metrics
        current_metrics = await self._calculate_spread_metrics(symbol, market_data)
        
        # Update spread history
        self.spread_history.append(current_metrics)
        self._maintain_history_window()
        
        # Calculate baseline spread (normal conditions)
        baseline_spread = await self._calculate_baseline_spread(symbol)
        
        # Calculate spread increase from baseline
        spread_increase = self._calculate_spread_increase(current_metrics.quoted_spread_bps, baseline_spread)
        
        # Determine toxicity level
        toxicity_level = self._classify_toxicity_level(spread_increase)
        
        # Calculate informed trading probability
        informed_prob = await self._calculate_informed_trading_probability(current_metrics, spread_increase)
        
        # Calculate adverse selection indicator
        adverse_selection = await self._calculate_adverse_selection_indicator(current_metrics, baseline_spread)
        
        # Perform spread decomposition
        decomposition = await self._perform_spread_decomposition(symbol, current_metrics)
        
        # Assess market quality
        market_quality = self._assess_market_quality(current_metrics, toxicity_level)
        
        # Check statistical significance
        is_significant = await self._check_statistical_significance(spread_increase, symbol)
        
        # Analyze trend
        trend = self._analyze_spread_trend(symbol)
        
        # Determine if alert should be triggered
        alert_triggered = self._should_trigger_alert(toxicity_level, spread_increase, is_significant)
        
        # Create analysis result
        analysis = SpreadToxicityAnalysis(
            timestamp=datetime.now().timestamp(),
            symbol=symbol,
            current_metrics=current_metrics,
            baseline_spread=baseline_spread,
            current_spread_increase=spread_increase,
            toxicity_level=toxicity_level,
            informed_trading_probability=informed_prob,
            adverse_selection_indicator=adverse_selection,
            spread_decomposition=decomposition,
            market_quality_assessment=market_quality,
            statistical_significance=is_significant,
            trend_analysis=trend,
            alert_triggered=alert_triggered
        )
        
        # Store in toxicity history
        self.toxicity_history.append(analysis)
        if len(self.toxicity_history) > 200:
            self.toxicity_history.pop(0)
            
        return analysis
    
    async def _calculate_spread_metrics(self, symbol: str, market_data: Dict[str, Any]) -> SpreadMetrics:
        """Calculate comprehensive spread metrics from market data"""
        
        bid_price = float(market_data.get('bid', 0))
        ask_price = float(market_data.get('ask', 0))
        trade_price = market_data.get('last')
        trade_price = float(trade_price) if trade_price is not None else None
        
        # Calculate basic metrics
        midpoint = (bid_price + ask_price) / 2 if bid_price > 0 and ask_price > 0 else 0
        quoted_spread = ask_price - bid_price
        quoted_spread_bps = (quoted_spread / midpoint * 10000) if midpoint > 0 else 0
        
        # Calculate effective spread (requires trade)
        effective_spread = None
        effective_spread_bps = None
        if trade_price is not None and midpoint > 0:
            effective_spread = 2 * abs(trade_price - midpoint)
            effective_spread_bps = (effective_spread / midpoint * 10000)
        
        # Calculate realized spread (requires future midpoint - simplified estimation)
        realized_spread = None
        realized_spread_bps = None
        price_impact = None
        price_impact_bps = None
        
        if effective_spread is not None:
            # Estimate realized spread (would need actual future data in production)
            realized_spread = effective_spread * 0.6  # Typical temporary component
            realized_spread_bps = effective_spread_bps * 0.6
            price_impact = effective_spread - realized_spread
            price_impact_bps = effective_spread_bps - realized_spread_bps
        
        return SpreadMetrics(
            timestamp=datetime.now().timestamp(),
            symbol=symbol,
            bid_price=bid_price,
            ask_price=ask_price,
            trade_price=trade_price,
            midpoint=midpoint,
            quoted_spread=quoted_spread,
            quoted_spread_bps=quoted_spread_bps,
            effective_spread=effective_spread,
            effective_spread_bps=effective_spread_bps,
            realized_spread=realized_spread,
            realized_spread_bps=realized_spread_bps,
            price_impact=price_impact,
            price_impact_bps=price_impact_bps
        )
    
    async def _calculate_baseline_spread(self, symbol: str) -> float:
        """Calculate baseline spread from recent history"""
        
        if symbol not in self.baseline_spreads or len(self.spread_history) < 10:
            # Initialize baseline or insufficient data
            recent_spreads = [m.quoted_spread_bps for m in self.spread_history[-10:] if m.symbol == symbol]
            if recent_spreads:
                baseline = statistics.median(recent_spreads)
                self.baseline_spreads[symbol] = baseline
                return baseline
            return 0.0
        
        # Update baseline with rolling window
        baseline_window = self.config.baseline_window_minutes * 4  # Assuming 15s updates
        symbol_spreads = [m.quoted_spread_bps for m in self.spread_history[-baseline_window:] 
                         if m.symbol == symbol and m.quoted_spread_bps > 0]
        
        if symbol_spreads:
            # Use median for robustness
            baseline = statistics.median(symbol_spreads)
            self.baseline_spreads[symbol] = baseline
            return baseline
        
        return self.baseline_spreads.get(symbol, 0.0)
    
    def _calculate_spread_increase(self, current_spread_bps: float, baseline_spread: float) -> float:
        """Calculate percentage increase from baseline"""
        if baseline_spread <= 0:
            return 0.0
        return (current_spread_bps - baseline_spread) / baseline_spread
    
    def _classify_toxicity_level(self, spread_increase: float) -> str:
        """Classify toxicity level based on spread increase"""
        if spread_increase < self.config.significance_threshold:
            return "LOW"
        elif spread_increase < self.config.major_threshold:
            return "MODERATE"
        elif spread_increase < self.config.extreme_threshold:
            return "HIGH"
        else:
            return "EXTREME"
    
    async def _calculate_informed_trading_probability(self, metrics: SpreadMetrics, spread_increase: float) -> float:
        """Calculate probability of informed trading based on spread metrics"""
        
        # Base probability on spread increase
        base_prob = min(spread_increase / self.config.major_threshold, 1.0) if spread_increase > 0 else 0.0
        
        # Adjust for effective spread vs quoted spread ratio
        if metrics.effective_spread_bps is not None and metrics.quoted_spread_bps > 0:
            eff_quoted_ratio = metrics.effective_spread_bps / metrics.quoted_spread_bps
            if eff_quoted_ratio > 1.2:  # Effective spread significantly higher than quoted
                base_prob *= 1.3
        
        # Adjust for price impact component
        if metrics.price_impact_bps is not None and metrics.effective_spread_bps is not None:
            if metrics.effective_spread_bps > 0:
                impact_ratio = metrics.price_impact_bps / metrics.effective_spread_bps
                if impact_ratio > 0.5:  # High permanent component
                    base_prob *= 1.2
        
        return min(base_prob, 1.0)
    
    async def _calculate_adverse_selection_indicator(self, metrics: SpreadMetrics, baseline: float) -> float:
        """Calculate composite adverse selection indicator"""
        
        # Component 1: Quoted spread increase
        spread_component = metrics.quoted_spread_bps / baseline if baseline > 0 else 1.0
        
        # Component 2: Effective spread premium
        eff_component = 1.0
        if metrics.effective_spread_bps is not None and metrics.quoted_spread_bps > 0:
            eff_component = metrics.effective_spread_bps / metrics.quoted_spread_bps
        
        # Component 3: Price impact ratio
        impact_component = 1.0
        if (metrics.price_impact_bps is not None and metrics.effective_spread_bps is not None 
            and metrics.effective_spread_bps > 0):
            impact_component = 1 + (metrics.price_impact_bps / metrics.effective_spread_bps)
        
        # Combine components (weighted average)
        composite = (0.4 * spread_component + 0.3 * eff_component + 0.3 * impact_component)
        
        # Normalize to 0-1 scale
        return min((composite - 1) * 2, 1.0)
    
    async def _perform_spread_decomposition(self, symbol: str, metrics: SpreadMetrics) -> Dict[str, float]:
        """Decompose spread into temporary and permanent components"""
        
        decomposition = {
            "temporary_component": 0.0,
            "permanent_component": 0.0,
            "inventory_effect": 0.0,
            "information_effect": 0.0,
            "order_processing_cost": 0.0
        }
        
        if metrics.effective_spread_bps is None:
            return decomposition
        
        # Simplified decomposition model
        effective_spread = metrics.effective_spread_bps
        
        # Estimate components based on research findings
        if metrics.realized_spread_bps is not None and metrics.price_impact_bps is not None:
            # Realized spread represents temporary component (inventory effect)
            decomposition["temporary_component"] = metrics.realized_spread_bps
            decomposition["inventory_effect"] = metrics.realized_spread_bps
            
            # Price impact represents permanent component (information effect)  
            decomposition["permanent_component"] = metrics.price_impact_bps
            decomposition["information_effect"] = metrics.price_impact_bps
            
            # Order processing cost (typically small portion)
            decomposition["order_processing_cost"] = effective_spread * 0.1
        else:
            # Use typical proportions from research
            decomposition["temporary_component"] = effective_spread * 0.6
            decomposition["permanent_component"] = effective_spread * 0.3
            decomposition["inventory_effect"] = effective_spread * 0.6
            decomposition["information_effect"] = effective_spread * 0.3
            decomposition["order_processing_cost"] = effective_spread * 0.1
        
        return decomposition
    
    def _assess_market_quality(self, metrics: SpreadMetrics, toxicity_level: str) -> str:
        """Assess overall market quality based on spread metrics"""
        
        if toxicity_level == "EXTREME":
            return "CRITICAL"
        elif toxicity_level == "HIGH":
            return "POOR"
        elif toxicity_level == "MODERATE":
            return "FAIR"
        else:
            # Also check absolute spread levels
            if metrics.quoted_spread_bps > 50:  # Very wide spreads
                return "POOR"
            elif metrics.quoted_spread_bps > 25:
                return "FAIR"
            else:
                return "GOOD"
    
    async def _check_statistical_significance(self, spread_increase: float, symbol: str) -> bool:
        """Check if spread increase is statistically significant"""
        
        # Get recent spread data for symbol
        symbol_spreads = [m.quoted_spread_bps for m in self.spread_history[-50:] 
                         if m.symbol == symbol and m.quoted_spread_bps > 0]
        
        if len(symbol_spreads) < 10:
            return False
        
        # Simple statistical test - compare current to recent distribution
        mean_spread = statistics.mean(symbol_spreads)
        std_spread = statistics.stdev(symbol_spreads) if len(symbol_spreads) > 1 else 0
        
        if std_spread == 0:
            return False
        
        current_spread = self.spread_history[-1].quoted_spread_bps if self.spread_history else 0
        z_score = abs(current_spread - mean_spread) / std_spread
        
        # Significant if z-score > 2 (roughly 95% confidence)
        return bool(z_score > 2.0)  # Convert numpy.bool to Python bool
    
    def _analyze_spread_trend(self, symbol: str) -> str:
        """Analyze recent trend in spreads for symbol"""
        
        symbol_spreads = [m.quoted_spread_bps for m in self.spread_history[-20:] 
                         if m.symbol == symbol]
        
        if len(symbol_spreads) < 5:
            return "INSUFFICIENT_DATA"
        
        # Simple trend analysis using linear regression
        x = np.arange(len(symbol_spreads))
        slope = np.polyfit(x, symbol_spreads, 1)[0]
        
        # Thresholds based on basis points change per observation
        if slope > 2.0:
            return "INCREASING"
        elif slope < -2.0:
            return "DECREASING"
        else:
            return "STABLE"
    
    def _should_trigger_alert(self, toxicity_level: str, spread_increase: float, is_significant: bool) -> bool:
        """Determine if alert should be triggered"""
        
        # Alert conditions
        high_toxicity = toxicity_level in ["HIGH", "EXTREME"]
        major_increase = spread_increase >= self.config.major_threshold  # 19.85% threshold
        statistically_significant = is_significant
        
        return high_toxicity and (major_increase or statistically_significant)
    
    def _maintain_history_window(self):
        """Maintain appropriate history window size"""
        max_history = self.config.baseline_window_minutes * 8  # Conservative window
        if len(self.spread_history) > max_history:
            self.spread_history = self.spread_history[-max_history:]
    
    async def get_comprehensive_spread_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive spread analysis including Feature Engine integration"""
        
        if not self.toxicity_history:
            return {"status": "no_data", "message": "No spread analysis available"}
        
        # Get recent analyses for symbol
        symbol_analyses = [a for a in self.toxicity_history[-50:] if a.symbol == symbol]
        
        if not symbol_analyses:
            return {"status": "no_symbol_data", "message": f"No spread data for {symbol}"}
        
        latest = symbol_analyses[-1]
        
        # Calculate statistics
        recent_increases = [a.current_spread_increase for a in symbol_analyses[-10:]]
        recent_probabilities = [a.informed_trading_probability for a in symbol_analyses[-10:]]
        
        analysis = {
            "current_analysis": {
                "toxicity_level": latest.toxicity_level,
                "spread_increase_percent": latest.current_spread_increase * 100,
                "informed_trading_probability": latest.informed_trading_probability,
                "adverse_selection_indicator": latest.adverse_selection_indicator,
                "market_quality": latest.market_quality_assessment,
                "alert_active": latest.alert_triggered
            },
            "spread_metrics": {
                "quoted_spread_bps": latest.current_metrics.quoted_spread_bps,
                "effective_spread_bps": latest.current_metrics.effective_spread_bps,
                "price_impact_bps": latest.current_metrics.price_impact_bps,
                "baseline_spread_bps": latest.baseline_spread
            },
            "spread_decomposition": latest.spread_decomposition,
            "statistical_analysis": {
                "is_significant": latest.statistical_significance,
                "trend": latest.trend_analysis,
                "mean_recent_increase": statistics.mean(recent_increases) * 100,
                "volatility": statistics.stdev(recent_increases) * 100 if len(recent_increases) > 1 else 0
            },
            "research_benchmarks": {
                "major_threshold_19_85_percent": self.config.major_threshold * 100,
                "current_vs_research_threshold": (latest.current_spread_increase / self.config.major_threshold) * 100,
                "exceeds_research_threshold": latest.current_spread_increase >= self.config.major_threshold
            }
        }
        
        return analysis

# Global analyzer instance
spread_toxicity_analyzer = SpreadToxicityAnalyzer()