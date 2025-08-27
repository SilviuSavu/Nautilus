#!/usr/bin/env python3
"""
Quote Fade Detection System
Detects predatory high-frequency trading strategies through quote fade measurements.
Research shows 7-15% probability of full price fade throughout trading day.
"""

import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import statistics

logger = logging.getLogger(__name__)

@dataclass
class QuoteFadeConfig:
    """Configuration for quote fade detection"""
    fade_detection_window_ms: int = 500  # Window to detect quote fade
    full_fade_threshold: float = 0.9  # 90% of quote size removed = full fade
    partial_fade_threshold: float = 0.3  # 30% reduction = partial fade
    quote_update_timeout_ms: int = 2000  # Max time between quote updates
    min_quote_size_threshold: float = 100  # Minimum quote size to track
    hft_speed_threshold_ms: int = 100  # Threshold for HFT-speed reactions
    max_history_minutes: int = 60  # Maximum history to maintain

@dataclass
class QuoteUpdate:
    """Individual quote update record"""
    timestamp: float
    symbol: str
    side: str  # 'bid' or 'ask'
    price: float
    size: float
    update_type: str  # 'new', 'modify', 'cancel', 'trade'
    
@dataclass
class QuoteFadeEvent:
    """Quote fade event detection"""
    timestamp: float
    symbol: str
    side: str
    fade_type: str  # 'full', 'partial', 'rapid'
    pre_trade_quote_size: float
    post_trade_quote_size: float
    fade_percentage: float
    fade_duration_ms: float
    trade_price: Optional[float]
    trade_size: Optional[float]
    hft_probability: float
    predatory_behavior_score: float

@dataclass
class QuoteFadeAnalysis:
    """Comprehensive quote fade analysis"""
    timestamp: float
    symbol: str
    total_fade_events: int
    full_fade_events: int
    partial_fade_events: int
    rapid_fade_events: int
    fade_probability: float  # Overall fade probability (7-15% benchmark)
    average_fade_percentage: float
    average_fade_duration_ms: float
    hft_predatory_score: float  # 0-1 score for predatory behavior
    market_maker_stress_level: str  # LOW, MODERATE, HIGH, EXTREME
    liquidity_quality_assessment: str
    fade_pattern_analysis: Dict[str, Any]

class QuoteFadeDetector:
    """
    Detects quote fade patterns indicating predatory HFT behavior.
    Quote fade occurs when market makers rapidly withdraw quotes following trades.
    """
    
    def __init__(self, config: QuoteFadeConfig = None):
        self.config = config or QuoteFadeConfig()
        self.quote_history = deque(maxlen=10000)  # Recent quote updates
        self.fade_events = deque(maxlen=1000)  # Detected fade events
        self.current_quotes = {}  # Current best quotes by symbol
        
    async def process_quote_update(self, quote_update: QuoteUpdate) -> Optional[QuoteFadeEvent]:
        """Process new quote update and detect potential fade events"""
        
        # Add to history
        self.quote_history.append(quote_update)
        
        # Update current quotes
        key = f"{quote_update.symbol}_{quote_update.side}"
        self.current_quotes[key] = quote_update
        
        # Check for potential fade event if this was after a trade
        if quote_update.update_type in ['modify', 'cancel']:
            fade_event = await self._detect_fade_event(quote_update)
            if fade_event:
                self.fade_events.append(fade_event)
                return fade_event
        
        return None
    
    async def _detect_fade_event(self, current_update: QuoteUpdate) -> Optional[QuoteFadeEvent]:
        """Detect if current update represents a quote fade event"""
        
        # Look for previous quote on same side within detection window
        cutoff_time = current_update.timestamp - (self.config.fade_detection_window_ms / 1000)
        
        previous_quotes = [
            q for q in reversed(self.quote_history)
            if (q.symbol == current_update.symbol and 
                q.side == current_update.side and
                q.timestamp >= cutoff_time and
                q.timestamp < current_update.timestamp and
                q.size >= self.config.min_quote_size_threshold)
        ]
        
        if not previous_quotes:
            return None
        
        # Get the most recent previous quote
        prev_quote = previous_quotes[0]
        
        # Check if there was a trade between quotes
        trade_occurred = await self._check_for_intervening_trade(
            prev_quote.timestamp, current_update.timestamp, current_update.symbol
        )
        
        if not trade_occurred:
            return None  # No fade without trade
        
        # Calculate fade metrics
        if prev_quote.size == 0:
            return None
            
        fade_percentage = (prev_quote.size - current_update.size) / prev_quote.size
        fade_duration_ms = (current_update.timestamp - prev_quote.timestamp) * 1000
        
        # Classify fade type
        fade_type = self._classify_fade_type(fade_percentage, fade_duration_ms)
        
        if fade_type == 'none':
            return None
        
        # Calculate HFT and predatory behavior probabilities
        hft_probability = self._calculate_hft_probability(fade_duration_ms, fade_percentage)
        predatory_score = self._calculate_predatory_score(
            fade_percentage, fade_duration_ms, current_update
        )
        
        # Get trade information if available
        trade_info = await self._get_trade_information(
            prev_quote.timestamp, current_update.timestamp, current_update.symbol
        )
        
        return QuoteFadeEvent(
            timestamp=current_update.timestamp,
            symbol=current_update.symbol,
            side=current_update.side,
            fade_type=fade_type,
            pre_trade_quote_size=prev_quote.size,
            post_trade_quote_size=current_update.size,
            fade_percentage=fade_percentage,
            fade_duration_ms=fade_duration_ms,
            trade_price=trade_info.get('price'),
            trade_size=trade_info.get('size'),
            hft_probability=hft_probability,
            predatory_behavior_score=predatory_score
        )
    
    async def _check_for_intervening_trade(self, start_time: float, end_time: float, symbol: str) -> bool:
        """Check if there was a trade between two quote updates"""
        
        # Look for trade updates in the time window
        trade_updates = [
            q for q in self.quote_history
            if (q.symbol == symbol and
                q.update_type == 'trade' and
                start_time <= q.timestamp <= end_time)
        ]
        
        return len(trade_updates) > 0
    
    async def _get_trade_information(self, start_time: float, end_time: float, symbol: str) -> Dict[str, Any]:
        """Get trade information from the time window"""
        
        trades = [
            q for q in self.quote_history
            if (q.symbol == symbol and
                q.update_type == 'trade' and
                start_time <= q.timestamp <= end_time)
        ]
        
        if trades:
            # Return most recent trade
            trade = trades[-1]
            return {'price': trade.price, 'size': trade.size}
        
        return {}
    
    def _classify_fade_type(self, fade_percentage: float, fade_duration_ms: float) -> str:
        """Classify the type of fade event"""
        
        if fade_percentage < self.config.partial_fade_threshold:
            return 'none'
        elif fade_percentage >= self.config.full_fade_threshold:
            return 'full'
        elif fade_duration_ms < self.config.hft_speed_threshold_ms:
            return 'rapid'
        else:
            return 'partial'
    
    def _calculate_hft_probability(self, fade_duration_ms: float, fade_percentage: float) -> float:
        """Calculate probability that fade is due to HFT activity"""
        
        # Base probability on speed (HFT typically very fast)
        speed_score = 0.0
        if fade_duration_ms < 50:  # Very fast
            speed_score = 0.9
        elif fade_duration_ms < 100:  # Fast
            speed_score = 0.7
        elif fade_duration_ms < 250:  # Moderate
            speed_score = 0.4
        else:  # Slow
            speed_score = 0.1
        
        # Adjust for fade severity (larger fades more likely HFT)
        severity_multiplier = min(fade_percentage * 2, 1.0)
        
        return min(speed_score * severity_multiplier, 1.0)
    
    def _calculate_predatory_score(self, fade_percentage: float, fade_duration_ms: float, 
                                 quote_update: QuoteUpdate) -> float:
        """Calculate predatory behavior score"""
        
        # Base score on fade severity and speed
        base_score = fade_percentage * 0.6
        
        # Speed component (faster = more predatory)
        if fade_duration_ms < 100:
            base_score += 0.3
        elif fade_duration_ms < 250:
            base_score += 0.15
        
        # Pattern detection (frequent fades from same source)
        frequency_bonus = self._calculate_frequency_bonus(quote_update.symbol)
        base_score += frequency_bonus
        
        return min(base_score, 1.0)
    
    def _calculate_frequency_bonus(self, symbol: str) -> float:
        """Calculate bonus score based on fade frequency"""
        
        # Count recent fade events for this symbol
        recent_cutoff = datetime.now().timestamp() - 300  # 5 minutes
        recent_fades = [
            f for f in self.fade_events
            if f.symbol == symbol and f.timestamp >= recent_cutoff
        ]
        
        # More fades = higher predatory score
        frequency_bonus = min(len(recent_fades) * 0.05, 0.2)
        return frequency_bonus
    
    async def analyze_quote_fade_patterns(self, symbol: str) -> QuoteFadeAnalysis:
        """Perform comprehensive analysis of quote fade patterns"""
        
        # Get recent fade events for symbol
        analysis_window = datetime.now().timestamp() - (self.config.max_history_minutes * 60)
        symbol_fades = [
            f for f in self.fade_events
            if f.symbol == symbol and f.timestamp >= analysis_window
        ]
        
        if not symbol_fades:
            return self._create_default_analysis(symbol)
        
        # Count event types
        full_fades = [f for f in symbol_fades if f.fade_type == 'full']
        partial_fades = [f for f in symbol_fades if f.fade_type == 'partial']
        rapid_fades = [f for f in symbol_fades if f.fade_type == 'rapid']
        
        # Calculate fade probability
        total_quotes = len([q for q in self.quote_history 
                           if q.symbol == symbol and q.timestamp >= analysis_window])
        fade_probability = len(symbol_fades) / max(total_quotes, 1)
        
        # Calculate averages
        avg_fade_percentage = statistics.mean([f.fade_percentage for f in symbol_fades])
        avg_fade_duration = statistics.mean([f.fade_duration_ms for f in symbol_fades])
        
        # Calculate HFT predatory score
        hft_scores = [f.predatory_behavior_score for f in symbol_fades]
        avg_predatory_score = statistics.mean(hft_scores) if hft_scores else 0.0
        
        # Assess market maker stress
        stress_level = self._assess_market_maker_stress(fade_probability, avg_predatory_score)
        
        # Assess liquidity quality
        liquidity_quality = self._assess_liquidity_quality(symbol_fades, fade_probability)
        
        # Pattern analysis
        pattern_analysis = self._analyze_fade_patterns(symbol_fades)
        
        return QuoteFadeAnalysis(
            timestamp=datetime.now().timestamp(),
            symbol=symbol,
            total_fade_events=len(symbol_fades),
            full_fade_events=len(full_fades),
            partial_fade_events=len(partial_fades),
            rapid_fade_events=len(rapid_fades),
            fade_probability=fade_probability,
            average_fade_percentage=avg_fade_percentage,
            average_fade_duration_ms=avg_fade_duration,
            hft_predatory_score=avg_predatory_score,
            market_maker_stress_level=stress_level,
            liquidity_quality_assessment=liquidity_quality,
            fade_pattern_analysis=pattern_analysis
        )
    
    def _assess_market_maker_stress(self, fade_probability: float, predatory_score: float) -> str:
        """Assess market maker stress level"""
        
        # Research benchmark: 7-15% fade probability is normal
        if fade_probability > 0.25 or predatory_score > 0.7:
            return "EXTREME"
        elif fade_probability > 0.15 or predatory_score > 0.5:
            return "HIGH"
        elif fade_probability > 0.07 or predatory_score > 0.3:
            return "MODERATE"
        else:
            return "LOW"
    
    def _assess_liquidity_quality(self, fade_events: List[QuoteFadeEvent], 
                                fade_probability: float) -> str:
        """Assess overall liquidity quality"""
        
        if fade_probability > 0.2:
            return "POOR"
        elif fade_probability > 0.15:
            return "FAIR"
        elif fade_probability < 0.07:
            return "EXCELLENT"
        else:
            return "GOOD"
    
    def _analyze_fade_patterns(self, fade_events: List[QuoteFadeEvent]) -> Dict[str, Any]:
        """Analyze patterns in fade events"""
        
        if not fade_events:
            return {}
        
        # Time-of-day analysis
        hours = [datetime.fromtimestamp(f.timestamp).hour for f in fade_events]
        peak_hour = statistics.mode(hours) if hours else 0
        
        # Side bias analysis
        bid_fades = [f for f in fade_events if f.side == 'bid']
        ask_fades = [f for f in fade_events if f.side == 'ask']
        
        # Speed distribution
        fast_fades = [f for f in fade_events if f.fade_duration_ms < 100]
        
        return {
            "peak_fade_hour": peak_hour,
            "bid_fade_ratio": len(bid_fades) / len(fade_events),
            "ask_fade_ratio": len(ask_fades) / len(fade_events),
            "fast_fade_ratio": len(fast_fades) / len(fade_events),
            "average_hft_probability": statistics.mean([f.hft_probability for f in fade_events]),
            "temporal_clustering": self._detect_temporal_clustering(fade_events)
        }
    
    def _detect_temporal_clustering(self, fade_events: List[QuoteFadeEvent]) -> bool:
        """Detect if fade events show temporal clustering (burst patterns)"""
        
        if len(fade_events) < 5:
            return False
        
        # Sort by timestamp
        sorted_events = sorted(fade_events, key=lambda x: x.timestamp)
        
        # Check for clusters (multiple fades within short time windows)
        cluster_threshold = 60  # 1 minute
        clusters = 0
        
        for i in range(len(sorted_events) - 2):
            window_events = []
            for j in range(i, len(sorted_events)):
                if sorted_events[j].timestamp - sorted_events[i].timestamp <= cluster_threshold:
                    window_events.append(sorted_events[j])
                else:
                    break
            
            if len(window_events) >= 3:  # 3+ fades in 1 minute = cluster
                clusters += 1
        
        return clusters >= 2  # 2+ clusters indicate systematic behavior
    
    def _create_default_analysis(self, symbol: str) -> QuoteFadeAnalysis:
        """Create default analysis when no fade events found"""
        
        return QuoteFadeAnalysis(
            timestamp=datetime.now().timestamp(),
            symbol=symbol,
            total_fade_events=0,
            full_fade_events=0,
            partial_fade_events=0,
            rapid_fade_events=0,
            fade_probability=0.0,
            average_fade_percentage=0.0,
            average_fade_duration_ms=0.0,
            hft_predatory_score=0.0,
            market_maker_stress_level="LOW",
            liquidity_quality_assessment="UNKNOWN",
            fade_pattern_analysis={}
        )
    
    async def get_real_time_fade_metrics(self, symbol: str) -> Dict[str, Any]:
        """Get real-time quote fade metrics for monitoring"""
        
        # Get recent analysis
        analysis = await self.analyze_quote_fade_patterns(symbol)
        
        # Get recent fade events (last 15 minutes)
        recent_cutoff = datetime.now().timestamp() - 900
        recent_fades = [
            f for f in self.fade_events
            if f.symbol == symbol and f.timestamp >= recent_cutoff
        ]
        
        return {
            "real_time_status": {
                "recent_fade_events": len(recent_fades),
                "current_fade_probability": analysis.fade_probability,
                "market_maker_stress": analysis.market_maker_stress_level,
                "liquidity_quality": analysis.liquidity_quality_assessment,
                "hft_predatory_score": analysis.hft_predatory_score
            },
            "research_benchmarks": {
                "normal_fade_range_percent": "7-15%",
                "current_fade_percent": f"{analysis.fade_probability * 100:.1f}%",
                "exceeds_normal_range": analysis.fade_probability > 0.15,
                "market_stress_warning": analysis.market_maker_stress_level in ["HIGH", "EXTREME"]
            },
            "pattern_detection": {
                "temporal_clustering_detected": analysis.fade_pattern_analysis.get('temporal_clustering', False),
                "fast_fade_ratio": analysis.fade_pattern_analysis.get('fast_fade_ratio', 0.0),
                "peak_activity_hour": analysis.fade_pattern_analysis.get('peak_fade_hour', 0)
            },
            "liquidity_impact": {
                "full_fades": analysis.full_fade_events,
                "partial_fades": analysis.partial_fade_events,
                "rapid_fades": analysis.rapid_fade_events,
                "average_fade_severity": analysis.average_fade_percentage
            }
        }

# Global detector instance
quote_fade_detector = QuoteFadeDetector()