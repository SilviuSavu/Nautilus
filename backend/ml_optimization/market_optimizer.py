"""
Market Condition Optimizer

This module implements real-time market condition analysis and AI-driven
performance optimization that adapts system behavior based on market regimes,
volatility patterns, and trading conditions.
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import redis
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import yfinance as yf
import requests


class MarketRegime(Enum):
    """Market regime classifications"""
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRISIS = "crisis"
    RECOVERY = "recovery"
    PRE_MARKET = "pre_market"
    POST_MARKET = "post_market"
    EARNINGS_SEASON = "earnings_season"


class OptimizationStrategy(Enum):
    """Performance optimization strategies"""
    LATENCY_FOCUSED = "latency_focused"
    THROUGHPUT_FOCUSED = "throughput_focused"
    COST_OPTIMIZED = "cost_optimized"
    RELIABILITY_FOCUSED = "reliability_focused"
    ADAPTIVE = "adaptive"


class PerformanceMetric(Enum):
    """Key performance metrics to optimize"""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    RESOURCE_UTILIZATION = "resource_utilization"
    COST_EFFICIENCY = "cost_efficiency"


@dataclass
class MarketCondition:
    """Current market condition assessment"""
    timestamp: datetime
    regime: MarketRegime
    volatility_level: float  # 0-1 scale
    volume_profile: float   # Relative to average
    trend_strength: float   # -1 to 1 (bearish to bullish)
    
    # Specific indicators
    vix_level: float
    vix_percentile: float  # Historical percentile
    market_breadth: float  # Advance/decline ratio
    sector_rotation: float # Sector rotation intensity
    
    # Event indicators
    earnings_intensity: float  # 0-1 based on earnings calendar
    economic_events: List[str] = field(default_factory=list)
    fed_proximity: float = 10.0  # Days to next FOMC meeting
    
    # Confidence in regime classification
    confidence: float = 0.8


@dataclass
class OptimizationSettings:
    """System optimization settings for current market conditions"""
    strategy: OptimizationStrategy
    
    # Resource allocation adjustments
    cpu_multiplier: float = 1.0
    memory_multiplier: float = 1.0
    network_multiplier: float = 1.0
    
    # Performance tuning
    batch_size_adjustment: float = 1.0
    timeout_adjustment: float = 1.0
    cache_ttl_adjustment: float = 1.0
    
    # Risk management
    position_size_multiplier: float = 1.0
    stop_loss_adjustment: float = 1.0
    
    # Trading behavior
    order_frequency_limit: float = 1.0
    market_impact_threshold: float = 1.0
    
    # Quality of service
    latency_target_adjustment: float = 1.0
    reliability_threshold: float = 0.99
    
    # Reasoning
    justification: str = ""
    confidence: float = 0.8


class MarketConditionOptimizer:
    """
    Real-time market condition analyzer and performance optimizer.
    
    This system continuously monitors market conditions and automatically
    adjusts system parameters to optimize performance for current regimes.
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.logger = logging.getLogger(__name__)
        
        # ML Models for market analysis
        self.regime_classifier = None
        self.volatility_predictor = None
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        
        # Market data cache
        self.market_data_cache = {}
        self.last_market_update = None
        
        # Optimization history
        self.optimization_history = []
        self.performance_metrics = {}
        
        # Configuration
        self.update_interval = 60  # seconds
        self.lookback_days = 30
        self.min_confidence_threshold = 0.6
        
        # Initialize components
        self._initialize_models()
        asyncio.create_task(self._start_background_monitoring())
    
    def _initialize_models(self):
        """Initialize ML models for market condition analysis"""
        self.regime_classifier = KMeans(n_clusters=len(MarketRegime), random_state=42)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Keep 95% of variance
    
    async def collect_real_time_market_data(self) -> Dict[str, Any]:
        """Collect comprehensive real-time market data"""
        try:
            market_data = {}
            
            # Major indices
            indices = ["^GSPC", "^DJI", "^IXIC", "^VIX", "^TNX", "DXY-NYB"]
            
            for symbol in indices:
                try:
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period="5d", interval="1h")
                    
                    if not data.empty:
                        latest = data.iloc[-1]
                        prev = data.iloc[-2] if len(data) > 1 else latest
                        
                        market_data[symbol] = {
                            'price': float(latest['Close']),
                            'change': float(latest['Close'] - prev['Close']),
                            'change_pct': float((latest['Close'] / prev['Close'] - 1) * 100),
                            'volume': float(latest['Volume']) if 'Volume' in data.columns else 0,
                            'high': float(latest['High']),
                            'low': float(latest['Low']),
                            'volatility': float(data['Close'].pct_change().std() * 100)
                        }
                except Exception as e:
                    self.logger.warning(f"Failed to fetch {symbol}: {str(e)}")
                    continue
            
            # Sector ETFs for rotation analysis
            sector_etfs = ["XLF", "XLK", "XLE", "XLV", "XLI", "XLY", "XLP", "XLB", "XLRE", "XLU", "XLC"]
            sector_performance = {}
            
            for etf in sector_etfs[:5]:  # Limit to reduce API calls
                try:
                    ticker = yf.Ticker(etf)
                    data = ticker.history(period="5d")
                    if not data.empty:
                        sector_performance[etf] = float((data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100)
                except:
                    continue
            
            market_data['sectors'] = sector_performance
            
            # Economic indicators from FRED (if available)
            try:
                fred_indicators = await self._get_fred_indicators()
                market_data['economic'] = fred_indicators
            except:
                market_data['economic'] = {}
            
            # Options flow and sentiment (simulated)
            market_data['sentiment'] = {
                'put_call_ratio': np.random.uniform(0.6, 1.4),
                'fear_greed_index': np.random.uniform(0, 100),
                'insider_sentiment': np.random.uniform(-1, 1)
            }
            
            # High-frequency indicators
            market_data['hf_indicators'] = {
                'tick_direction': np.random.choice([-1, 0, 1]),
                'bid_ask_spread': np.random.uniform(0.01, 0.05),
                'market_depth': np.random.uniform(0.5, 2.0)
            }
            
            # Update cache
            self.market_data_cache = market_data
            self.last_market_update = datetime.now()
            
            # Store in Redis for other components
            self.redis_client.setex(
                "market:realtime", 
                300,  # 5 minute expiry
                json.dumps(market_data, default=str)
            )
            
            return market_data
            
        except Exception as e:
            self.logger.error(f"Error collecting market data: {str(e)}")
            return self.market_data_cache or {}
    
    async def _get_fred_indicators(self) -> Dict[str, float]:
        """Get economic indicators from FRED (Federal Reserve Economic Data)"""
        # This would typically use the FRED API
        # For demonstration, return simulated economic indicators
        return {
            'unemployment_rate': np.random.uniform(3.5, 6.0),
            'inflation_rate': np.random.uniform(1.0, 4.0),
            'gdp_growth': np.random.uniform(-2.0, 4.0),
            'consumer_confidence': np.random.uniform(80, 130),
            'treasury_10y': np.random.uniform(3.0, 6.0),
            'treasury_2y': np.random.uniform(2.5, 5.5)
        }
    
    def _calculate_market_features(self, market_data: Dict[str, Any]) -> np.ndarray:
        """Extract features from market data for ML analysis"""
        features = []
        
        # Price-based features
        if "^GSPC" in market_data:
            sp500 = market_data["^GSPC"]
            features.extend([
                sp500.get('change_pct', 0),
                sp500.get('volatility', 0),
                sp500.get('volume', 0) / 1e9  # Normalize volume
            ])
        else:
            features.extend([0, 0, 0])
        
        # VIX features
        if "^VIX" in market_data:
            vix = market_data["^VIX"]
            features.extend([
                vix.get('price', 20),
                vix.get('change', 0),
                vix.get('change_pct', 0)
            ])
        else:
            features.extend([20, 0, 0])
        
        # Interest rate features
        if "^TNX" in market_data:
            tnx = market_data["^TNX"]
            features.extend([
                tnx.get('price', 4.5),
                tnx.get('change', 0)
            ])
        else:
            features.extend([4.5, 0])
        
        # Dollar strength
        if "DXY-NYB" in market_data:
            dxy = market_data["DXY-NYB"]
            features.extend([
                dxy.get('price', 100),
                dxy.get('change_pct', 0)
            ])
        else:
            features.extend([100, 0])
        
        # Sector rotation
        sectors = market_data.get('sectors', {})
        if sectors:
            sector_values = list(sectors.values())
            features.extend([
                np.mean(sector_values),
                np.std(sector_values),
                max(sector_values) - min(sector_values)
            ])
        else:
            features.extend([0, 0, 0])
        
        # Sentiment indicators
        sentiment = market_data.get('sentiment', {})
        features.extend([
            sentiment.get('put_call_ratio', 1.0),
            sentiment.get('fear_greed_index', 50) / 100,
            sentiment.get('insider_sentiment', 0)
        ])
        
        # Economic indicators
        economic = market_data.get('economic', {})
        features.extend([
            economic.get('unemployment_rate', 4.0),
            economic.get('inflation_rate', 2.5),
            economic.get('consumer_confidence', 100) / 100
        ])
        
        # Time features
        now = datetime.now()
        features.extend([
            now.hour,
            now.weekday(),
            float(9 <= now.hour <= 16 and now.weekday() < 5)  # market_hours
        ])
        
        return np.array(features).reshape(1, -1)
    
    async def analyze_market_condition(self) -> MarketCondition:
        """Analyze current market conditions and classify regime"""
        try:
            # Get current market data
            market_data = await self.collect_real_time_market_data()
            
            if not market_data:
                return self._get_default_market_condition()
            
            # Extract features
            features = self._calculate_market_features(market_data)
            
            # Classify market regime
            regime = self._classify_market_regime(market_data, features)
            
            # Calculate key metrics
            vix_level = market_data.get("^VIX", {}).get('price', 20.0)
            vix_percentile = self._calculate_vix_percentile(vix_level)
            volatility_level = min(1.0, vix_level / 50.0)  # Normalize to 0-1
            
            # Volume analysis
            sp500_volume = market_data.get("^GSPC", {}).get('volume', 1e9)
            volume_profile = self._analyze_volume_profile(sp500_volume)
            
            # Trend analysis
            trend_strength = self._calculate_trend_strength(market_data)
            
            # Market breadth (simulated)
            market_breadth = self._calculate_market_breadth(market_data)
            
            # Sector rotation intensity
            sector_rotation = self._calculate_sector_rotation(market_data.get('sectors', {}))
            
            # Earnings calendar intensity
            earnings_intensity = self._calculate_earnings_intensity()
            
            # Economic events
            economic_events = self._get_current_economic_events()
            
            # Fed meeting proximity
            fed_proximity = self._calculate_fed_proximity()
            
            # Classification confidence
            confidence = self._calculate_classification_confidence(features, regime)
            
            condition = MarketCondition(
                timestamp=datetime.now(),
                regime=regime,
                volatility_level=volatility_level,
                volume_profile=volume_profile,
                trend_strength=trend_strength,
                vix_level=vix_level,
                vix_percentile=vix_percentile,
                market_breadth=market_breadth,
                sector_rotation=sector_rotation,
                earnings_intensity=earnings_intensity,
                economic_events=economic_events,
                fed_proximity=fed_proximity,
                confidence=confidence
            )
            
            # Store for historical analysis
            await self._store_market_condition(condition)
            
            return condition
            
        except Exception as e:
            self.logger.error(f"Error analyzing market condition: {str(e)}")
            return self._get_default_market_condition()
    
    def _classify_market_regime(self, market_data: Dict[str, Any], features: np.ndarray) -> MarketRegime:
        """Classify current market regime based on multiple indicators"""
        try:
            # Rule-based classification with ML enhancement
            
            # Get key indicators
            vix = market_data.get("^VIX", {}).get('price', 20.0)
            sp500_change = market_data.get("^GSPC", {}).get('change_pct', 0.0)
            volume_ratio = self._get_volume_ratio(market_data)
            
            now = datetime.now()
            is_market_hours = 9 <= now.hour <= 16 and now.weekday() < 5
            
            # Time-based regimes
            if not is_market_hours:
                if now.hour < 9:
                    return MarketRegime.PRE_MARKET
                else:
                    return MarketRegime.POST_MARKET
            
            # Crisis conditions (VIX > 35)
            if vix > 35:
                return MarketRegime.CRISIS
            
            # High volatility (VIX > 25)
            if vix > 25:
                if sp500_change > 1:
                    return MarketRegime.RECOVERY
                else:
                    return MarketRegime.HIGH_VOLATILITY
            
            # Low volatility (VIX < 15)
            if vix < 15:
                return MarketRegime.LOW_VOLATILITY
            
            # Trend-based classification
            if sp500_change > 1.5:
                return MarketRegime.BULL_MARKET
            elif sp500_change < -1.5:
                return MarketRegime.BEAR_MARKET
            elif abs(sp500_change) < 0.5 and volume_ratio < 0.8:
                return MarketRegime.SIDEWAYS
            
            # Earnings season (simplified)
            if self._is_earnings_season():
                return MarketRegime.EARNINGS_SEASON
            
            # Default to sideways if no clear regime
            return MarketRegime.SIDEWAYS
            
        except Exception as e:
            self.logger.error(f"Error classifying market regime: {str(e)}")
            return MarketRegime.SIDEWAYS
    
    def _calculate_vix_percentile(self, current_vix: float) -> float:
        """Calculate VIX percentile relative to historical range"""
        # Historical VIX range: 10-80 (approximate)
        # This would typically use actual historical data
        min_vix = 10.0
        max_vix = 80.0
        
        percentile = (current_vix - min_vix) / (max_vix - min_vix)
        return max(0.0, min(1.0, percentile))
    
    def _analyze_volume_profile(self, current_volume: float) -> float:
        """Analyze current volume relative to average"""
        # This would typically use historical volume data
        # For now, use a simple heuristic
        typical_volume = 3e9  # Typical S&P 500 volume
        
        volume_ratio = current_volume / typical_volume
        return min(3.0, max(0.1, volume_ratio))  # Cap between 0.1x and 3.0x
    
    def _calculate_trend_strength(self, market_data: Dict[str, Any]) -> float:
        """Calculate overall market trend strength (-1 to 1)"""
        try:
            # Simple trend calculation based on multiple indices
            indices_changes = []
            
            for symbol in ["^GSPC", "^DJI", "^IXIC"]:
                if symbol in market_data:
                    change_pct = market_data[symbol].get('change_pct', 0)
                    indices_changes.append(change_pct)
            
            if not indices_changes:
                return 0.0
            
            avg_change = np.mean(indices_changes)
            
            # Normalize to -1 to 1 scale
            return max(-1.0, min(1.0, avg_change / 3.0))  # Assume 3% is max normal move
            
        except:
            return 0.0
    
    def _calculate_market_breadth(self, market_data: Dict[str, Any]) -> float:
        """Calculate market breadth indicator"""
        # Simplified market breadth calculation
        # In reality, this would use advance/decline data
        
        sectors = market_data.get('sectors', {})
        if not sectors:
            return 0.5  # Neutral
        
        positive_sectors = sum(1 for performance in sectors.values() if performance > 0)
        total_sectors = len(sectors)
        
        return positive_sectors / total_sectors if total_sectors > 0 else 0.5
    
    def _calculate_sector_rotation(self, sectors: Dict[str, float]) -> float:
        """Calculate sector rotation intensity"""
        if not sectors or len(sectors) < 2:
            return 0.0
        
        # Rotation intensity = standard deviation of sector performances
        performances = list(sectors.values())
        return np.std(performances) / 100  # Normalize percentage to decimal
    
    def _calculate_earnings_intensity(self) -> float:
        """Calculate earnings season intensity (0-1)"""
        # Simplified earnings calendar intensity
        # This would typically query actual earnings calendar
        
        now = datetime.now()
        
        # Earnings seasons: Jan, Apr, Jul, Oct
        earnings_months = [1, 4, 7, 10]
        current_month = now.month
        
        # Check if we're in an earnings month
        if current_month in earnings_months:
            # Peak intensity in weeks 2-4 of the month
            week_of_month = (now.day - 1) // 7 + 1
            if 2 <= week_of_month <= 4:
                return 0.8
            else:
                return 0.4
        else:
            return 0.1  # Low background level
    
    def _get_current_economic_events(self) -> List[str]:
        """Get list of current economic events"""
        # This would typically query an economic calendar API
        # For now, return simulated events
        
        events = []
        now = datetime.now()
        
        # Common economic events by day of week
        if now.weekday() == 4:  # Friday
            events.append("Employment Report")
        elif now.weekday() == 2:  # Wednesday
            events.append("FOMC Minutes")
        
        # Random events for simulation
        if np.random.random() < 0.3:
            possible_events = [
                "CPI Release", "PPI Release", "GDP Report", "Retail Sales",
                "Industrial Production", "Consumer Confidence", "ISM PMI"
            ]
            events.append(np.random.choice(possible_events))
        
        return events
    
    def _calculate_fed_proximity(self) -> float:
        """Calculate days until next Fed meeting"""
        # FOMC meets 8 times per year, roughly every 6-7 weeks
        # This is a simplified calculation
        
        now = datetime.now()
        
        # Approximate next meeting dates (would use actual calendar)
        meeting_months = [1, 3, 5, 6, 7, 9, 11, 12]
        
        # Find next meeting
        for month in meeting_months:
            if month >= now.month:
                next_meeting = datetime(now.year, month, 15)  # Mid-month approximation
                if next_meeting > now:
                    return (next_meeting - now).days
        
        # If no meeting this year, first meeting next year
        next_meeting = datetime(now.year + 1, 1, 15)
        return (next_meeting - now).days
    
    def _get_volume_ratio(self, market_data: Dict[str, Any]) -> float:
        """Get volume ratio compared to average"""
        sp500_volume = market_data.get("^GSPC", {}).get('volume', 0)
        if sp500_volume == 0:
            return 1.0
        
        # Typical volume (this would use historical average)
        typical_volume = 3e9
        return sp500_volume / typical_volume
    
    def _is_earnings_season(self) -> bool:
        """Check if currently in earnings season"""
        return self._calculate_earnings_intensity() > 0.5
    
    def _calculate_classification_confidence(self, features: np.ndarray, regime: MarketRegime) -> float:
        """Calculate confidence in regime classification"""
        # Simplified confidence calculation
        base_confidence = 0.7
        
        # Adjust based on feature quality
        if features.size > 0:
            feature_quality = min(1.0, np.sum(np.abs(features)) / len(features))
            confidence_adjustment = feature_quality * 0.2
            base_confidence += confidence_adjustment
        
        # Adjust based on regime clarity
        regime_clarity = {
            MarketRegime.CRISIS: 0.9,
            MarketRegime.HIGH_VOLATILITY: 0.8,
            MarketRegime.LOW_VOLATILITY: 0.8,
            MarketRegime.BULL_MARKET: 0.7,
            MarketRegime.BEAR_MARKET: 0.7,
            MarketRegime.SIDEWAYS: 0.6,
            MarketRegime.PRE_MARKET: 0.95,
            MarketRegime.POST_MARKET: 0.95,
            MarketRegime.EARNINGS_SEASON: 0.8,
            MarketRegime.RECOVERY: 0.6
        }
        
        return regime_clarity.get(regime, 0.6)
    
    def _get_default_market_condition(self) -> MarketCondition:
        """Return default market condition when analysis fails"""
        return MarketCondition(
            timestamp=datetime.now(),
            regime=MarketRegime.SIDEWAYS,
            volatility_level=0.3,
            volume_profile=1.0,
            trend_strength=0.0,
            vix_level=20.0,
            vix_percentile=0.3,
            market_breadth=0.5,
            sector_rotation=0.2,
            earnings_intensity=0.1,
            economic_events=[],
            fed_proximity=30.0,
            confidence=0.5
        )
    
    async def optimize_for_market_condition(self, condition: MarketCondition) -> OptimizationSettings:
        """Generate optimization settings based on market condition"""
        try:
            # Determine optimal strategy
            strategy = self._select_optimization_strategy(condition)
            
            # Base settings
            settings = OptimizationSettings(strategy=strategy)
            
            # Adjust based on market regime
            if condition.regime == MarketRegime.CRISIS:
                settings = self._apply_crisis_optimizations(settings, condition)
            elif condition.regime == MarketRegime.HIGH_VOLATILITY:
                settings = self._apply_volatility_optimizations(settings, condition)
            elif condition.regime in [MarketRegime.BULL_MARKET, MarketRegime.BEAR_MARKET]:
                settings = self._apply_trending_optimizations(settings, condition)
            elif condition.regime == MarketRegime.LOW_VOLATILITY:
                settings = self._apply_low_vol_optimizations(settings, condition)
            elif condition.regime == MarketRegime.EARNINGS_SEASON:
                settings = self._apply_earnings_optimizations(settings, condition)
            else:  # SIDEWAYS, PRE_MARKET, POST_MARKET
                settings = self._apply_neutral_optimizations(settings, condition)
            
            # Apply volume-based adjustments
            settings = self._apply_volume_adjustments(settings, condition)
            
            # Apply volatility-based adjustments
            settings = self._apply_volatility_adjustments(settings, condition)
            
            # Validate and bound settings
            settings = self._validate_settings(settings)
            
            # Generate justification
            settings.justification = self._generate_optimization_justification(condition, settings)
            settings.confidence = min(0.95, condition.confidence + 0.1)
            
            # Store optimization decision
            await self._store_optimization_decision(condition, settings)
            
            return settings
            
        except Exception as e:
            self.logger.error(f"Error optimizing for market condition: {str(e)}")
            return self._get_default_optimization_settings()
    
    def _select_optimization_strategy(self, condition: MarketCondition) -> OptimizationStrategy:
        """Select optimization strategy based on market condition"""
        
        if condition.regime == MarketRegime.CRISIS:
            return OptimizationStrategy.RELIABILITY_FOCUSED
        elif condition.volatility_level > 0.7:
            return OptimizationStrategy.LATENCY_FOCUSED
        elif condition.volume_profile > 2.0:
            return OptimizationStrategy.THROUGHPUT_FOCUSED
        elif condition.regime == MarketRegime.LOW_VOLATILITY:
            return OptimizationStrategy.COST_OPTIMIZED
        else:
            return OptimizationStrategy.ADAPTIVE
    
    def _apply_crisis_optimizations(self, settings: OptimizationSettings, condition: MarketCondition) -> OptimizationSettings:
        """Apply optimizations for crisis market conditions"""
        # Increase resources for reliability
        settings.cpu_multiplier = 1.5
        settings.memory_multiplier = 1.3
        settings.network_multiplier = 1.4
        
        # Reduce batch sizes for lower latency
        settings.batch_size_adjustment = 0.5
        
        # Shorter timeouts
        settings.timeout_adjustment = 0.7
        
        # Longer cache TTL to reduce load
        settings.cache_ttl_adjustment = 2.0
        
        # Conservative risk management
        settings.position_size_multiplier = 0.5
        settings.stop_loss_adjustment = 0.8
        
        # Limit order frequency
        settings.order_frequency_limit = 0.6
        
        # Lower latency targets
        settings.latency_target_adjustment = 0.5
        settings.reliability_threshold = 0.995
        
        return settings
    
    def _apply_volatility_optimizations(self, settings: OptimizationSettings, condition: MarketCondition) -> OptimizationSettings:
        """Apply optimizations for high volatility conditions"""
        # Increase CPU for faster processing
        settings.cpu_multiplier = 1.3
        settings.memory_multiplier = 1.2
        
        # Smaller batches for responsiveness
        settings.batch_size_adjustment = 0.7
        
        # Faster timeouts
        settings.timeout_adjustment = 0.8
        
        # Moderate risk adjustments
        settings.position_size_multiplier = 0.8
        settings.stop_loss_adjustment = 0.9
        
        # Higher order frequency for opportunities
        settings.order_frequency_limit = 1.2
        
        # Tighter latency requirements
        settings.latency_target_adjustment = 0.7
        
        return settings
    
    def _apply_trending_optimizations(self, settings: OptimizationSettings, condition: MarketCondition) -> OptimizationSettings:
        """Apply optimizations for trending markets"""
        # Balanced resource allocation
        settings.cpu_multiplier = 1.1
        settings.memory_multiplier = 1.1
        
        # Standard batch sizes
        settings.batch_size_adjustment = 1.0
        
        # Risk adjustments based on trend direction
        if condition.trend_strength > 0:  # Bull market
            settings.position_size_multiplier = 1.1
        else:  # Bear market
            settings.position_size_multiplier = 0.9
            settings.stop_loss_adjustment = 0.9
        
        # Normal trading parameters
        settings.order_frequency_limit = 1.0
        settings.latency_target_adjustment = 1.0
        
        return settings
    
    def _apply_low_vol_optimizations(self, settings: OptimizationSettings, condition: MarketCondition) -> OptimizationSettings:
        """Apply optimizations for low volatility conditions"""
        # Reduce resources for cost efficiency
        settings.cpu_multiplier = 0.8
        settings.memory_multiplier = 0.9
        settings.network_multiplier = 0.9
        
        # Larger batches for efficiency
        settings.batch_size_adjustment = 1.3
        
        # Longer timeouts acceptable
        settings.timeout_adjustment = 1.2
        
        # Longer cache TTL
        settings.cache_ttl_adjustment = 1.5
        
        # Slightly higher position sizes
        settings.position_size_multiplier = 1.05
        
        # Relaxed latency requirements
        settings.latency_target_adjustment = 1.2
        
        return settings
    
    def _apply_earnings_optimizations(self, settings: OptimizationSettings, condition: MarketCondition) -> OptimizationSettings:
        """Apply optimizations for earnings season"""
        # Increased resources for event processing
        settings.cpu_multiplier = 1.2
        settings.memory_multiplier = 1.2
        settings.network_multiplier = 1.3
        
        # Smaller batches for event responsiveness
        settings.batch_size_adjustment = 0.8
        
        # Conservative risk management
        settings.position_size_multiplier = 0.9
        settings.stop_loss_adjustment = 0.9
        
        # Higher order frequency around events
        settings.order_frequency_limit = 1.1
        
        # Tighter latency for event-driven trading
        settings.latency_target_adjustment = 0.8
        
        return settings
    
    def _apply_neutral_optimizations(self, settings: OptimizationSettings, condition: MarketCondition) -> OptimizationSettings:
        """Apply neutral optimizations for sideways/off-hours markets"""
        # Standard resource allocation
        settings.cpu_multiplier = 1.0
        settings.memory_multiplier = 1.0
        settings.network_multiplier = 1.0
        
        # If pre/post market, reduce resources
        if condition.regime in [MarketRegime.PRE_MARKET, MarketRegime.POST_MARKET]:
            settings.cpu_multiplier = 0.7
            settings.memory_multiplier = 0.8
            settings.network_multiplier = 0.6
            settings.order_frequency_limit = 0.5
        
        # Standard parameters
        settings.batch_size_adjustment = 1.0
        settings.timeout_adjustment = 1.0
        settings.cache_ttl_adjustment = 1.0
        settings.position_size_multiplier = 1.0
        settings.latency_target_adjustment = 1.0
        
        return settings
    
    def _apply_volume_adjustments(self, settings: OptimizationSettings, condition: MarketCondition) -> OptimizationSettings:
        """Apply volume-based adjustments"""
        volume_factor = condition.volume_profile
        
        # High volume requires more throughput capacity
        if volume_factor > 1.5:
            settings.cpu_multiplier *= 1.1
            settings.network_multiplier *= 1.2
            settings.batch_size_adjustment *= 1.1
        elif volume_factor < 0.7:
            # Low volume - can reduce resources
            settings.cpu_multiplier *= 0.9
            settings.network_multiplier *= 0.9
            settings.batch_size_adjustment *= 0.9
        
        return settings
    
    def _apply_volatility_adjustments(self, settings: OptimizationSettings, condition: MarketCondition) -> OptimizationSettings:
        """Apply volatility-based adjustments"""
        vol_level = condition.volatility_level
        
        # High volatility needs more resources and faster processing
        if vol_level > 0.6:
            settings.cpu_multiplier *= (1.0 + vol_level * 0.3)
            settings.memory_multiplier *= (1.0 + vol_level * 0.2)
            settings.batch_size_adjustment *= (1.0 - vol_level * 0.3)
            settings.timeout_adjustment *= (1.0 - vol_level * 0.2)
        
        return settings
    
    def _validate_settings(self, settings: OptimizationSettings) -> OptimizationSettings:
        """Validate and bound optimization settings"""
        # Resource multipliers: 0.5x to 3.0x
        settings.cpu_multiplier = max(0.5, min(3.0, settings.cpu_multiplier))
        settings.memory_multiplier = max(0.5, min(3.0, settings.memory_multiplier))
        settings.network_multiplier = max(0.5, min(3.0, settings.network_multiplier))
        
        # Batch size: 0.1x to 5.0x
        settings.batch_size_adjustment = max(0.1, min(5.0, settings.batch_size_adjustment))
        
        # Timeout: 0.1x to 5.0x
        settings.timeout_adjustment = max(0.1, min(5.0, settings.timeout_adjustment))
        
        # Cache TTL: 0.1x to 10.0x
        settings.cache_ttl_adjustment = max(0.1, min(10.0, settings.cache_ttl_adjustment))
        
        # Position size: 0.1x to 2.0x
        settings.position_size_multiplier = max(0.1, min(2.0, settings.position_size_multiplier))
        
        # Stop loss: 0.5x to 1.5x
        settings.stop_loss_adjustment = max(0.5, min(1.5, settings.stop_loss_adjustment))
        
        # Order frequency: 0.1x to 5.0x
        settings.order_frequency_limit = max(0.1, min(5.0, settings.order_frequency_limit))
        
        # Latency target: 0.1x to 3.0x
        settings.latency_target_adjustment = max(0.1, min(3.0, settings.latency_target_adjustment))
        
        # Reliability: 90% to 99.9%
        settings.reliability_threshold = max(0.9, min(0.999, settings.reliability_threshold))
        
        return settings
    
    def _generate_optimization_justification(self, condition: MarketCondition, settings: OptimizationSettings) -> str:
        """Generate justification text for optimization decisions"""
        reasons = []
        
        # Regime-based reasoning
        regime_explanations = {
            MarketRegime.CRISIS: "Crisis conditions require maximum reliability and reduced risk exposure",
            MarketRegime.HIGH_VOLATILITY: "High volatility demands faster processing and tighter risk controls",
            MarketRegime.LOW_VOLATILITY: "Low volatility allows cost optimization and resource reduction",
            MarketRegime.BULL_MARKET: "Bull market optimizations for trend-following strategies",
            MarketRegime.BEAR_MARKET: "Bear market adjustments with defensive risk management",
            MarketRegime.EARNINGS_SEASON: "Earnings season requires enhanced event processing capabilities",
            MarketRegime.PRE_MARKET: "Pre-market hours allow reduced resource allocation",
            MarketRegime.POST_MARKET: "Post-market optimization for maintenance and cost reduction",
            MarketRegime.SIDEWAYS: "Sideways market with balanced resource allocation"
        }
        
        reasons.append(regime_explanations.get(condition.regime, "Standard market conditions"))
        
        # Volatility-based reasoning
        if condition.volatility_level > 0.6:
            reasons.append(f"High volatility ({condition.volatility_level:.1%}) requires enhanced processing capacity")
        elif condition.volatility_level < 0.3:
            reasons.append(f"Low volatility ({condition.volatility_level:.1%}) enables resource optimization")
        
        # Volume-based reasoning
        if condition.volume_profile > 1.5:
            reasons.append(f"High trading volume ({condition.volume_profile:.1f}x) needs increased throughput")
        elif condition.volume_profile < 0.7:
            reasons.append(f"Low trading volume ({condition.volume_profile:.1f}x) allows resource scaling down")
        
        # Event-based reasoning
        if condition.economic_events:
            reasons.append(f"Economic events scheduled: {', '.join(condition.economic_events)}")
        
        if condition.earnings_intensity > 0.5:
            reasons.append(f"Earnings season intensity at {condition.earnings_intensity:.1%}")
        
        # Strategy reasoning
        strategy_explanations = {
            OptimizationStrategy.LATENCY_FOCUSED: "Prioritizing latency reduction",
            OptimizationStrategy.THROUGHPUT_FOCUSED: "Optimizing for maximum throughput",
            OptimizationStrategy.COST_OPTIMIZED: "Cost-efficient resource allocation",
            OptimizationStrategy.RELIABILITY_FOCUSED: "Maximum reliability and fault tolerance",
            OptimizationStrategy.ADAPTIVE: "Balanced adaptive optimization"
        }
        
        reasons.append(strategy_explanations.get(settings.strategy, "Standard optimization"))
        
        return "; ".join(reasons)
    
    def _get_default_optimization_settings(self) -> OptimizationSettings:
        """Return default optimization settings"""
        return OptimizationSettings(
            strategy=OptimizationStrategy.ADAPTIVE,
            justification="Default settings due to analysis error",
            confidence=0.5
        )
    
    async def _store_market_condition(self, condition: MarketCondition):
        """Store market condition for historical analysis"""
        condition_data = {
            "timestamp": condition.timestamp.isoformat(),
            "regime": condition.regime.value,
            "volatility_level": condition.volatility_level,
            "volume_profile": condition.volume_profile,
            "trend_strength": condition.trend_strength,
            "vix_level": condition.vix_level,
            "confidence": condition.confidence
        }
        
        # Store in Redis
        self.redis_client.lpush("market:conditions:history", json.dumps(condition_data))
        self.redis_client.ltrim("market:conditions:history", 0, 999)  # Keep last 1000
        
        # Store current condition
        self.redis_client.setex("market:condition:current", 300, json.dumps(condition_data))
    
    async def _store_optimization_decision(self, condition: MarketCondition, settings: OptimizationSettings):
        """Store optimization decision for analysis"""
        decision_data = {
            "timestamp": datetime.now().isoformat(),
            "market_regime": condition.regime.value,
            "strategy": settings.strategy.value,
            "cpu_multiplier": settings.cpu_multiplier,
            "memory_multiplier": settings.memory_multiplier,
            "confidence": settings.confidence,
            "justification": settings.justification
        }
        
        self.redis_client.lpush("optimization:decisions:history", json.dumps(decision_data))
        self.redis_client.ltrim("optimization:decisions:history", 0, 499)  # Keep last 500
    
    async def _start_background_monitoring(self):
        """Start background monitoring loop"""
        await asyncio.sleep(5)  # Initial delay
        
        while True:
            try:
                # Analyze market conditions
                condition = await self.analyze_market_condition()
                
                # Generate optimization settings
                settings = await self.optimize_for_market_condition(condition)
                
                self.logger.info(
                    f"Market Analysis: {condition.regime.value}, "
                    f"Volatility: {condition.volatility_level:.2f}, "
                    f"Strategy: {settings.strategy.value}, "
                    f"CPU: {settings.cpu_multiplier:.2f}x"
                )
                
                # Apply optimizations (would integrate with actual system)
                await self._apply_optimizations(settings)
                
            except Exception as e:
                self.logger.error(f"Error in background monitoring: {str(e)}")
            
            # Wait for next update
            await asyncio.sleep(self.update_interval)
    
    async def _apply_optimizations(self, settings: OptimizationSettings):
        """Apply optimization settings to the system"""
        # This would integrate with actual system components
        # For now, just log the optimizations
        
        optimizations = {
            "strategy": settings.strategy.value,
            "cpu_multiplier": settings.cpu_multiplier,
            "memory_multiplier": settings.memory_multiplier,
            "batch_size_adjustment": settings.batch_size_adjustment,
            "timeout_adjustment": settings.timeout_adjustment,
            "applied_at": datetime.now().isoformat()
        }
        
        # Store current optimizations
        self.redis_client.setex("optimization:current", 600, json.dumps(optimizations))
        
        self.logger.debug(f"Applied optimizations: {optimizations}")
    
    async def get_current_market_insights(self) -> Dict[str, Any]:
        """Get comprehensive market insights for dashboard"""
        try:
            condition = await self.analyze_market_condition()
            settings = await self.optimize_for_market_condition(condition)
            
            # Performance metrics (would be calculated from actual data)
            performance_impact = {
                "latency_improvement": (2.0 - settings.latency_target_adjustment) * 50,  # % improvement
                "throughput_increase": (settings.cpu_multiplier - 1.0) * 100,  # % increase
                "cost_impact": (settings.cpu_multiplier + settings.memory_multiplier - 2.0) * 10,  # $ impact
                "reliability_score": settings.reliability_threshold * 100  # %
            }
            
            return {
                "market_condition": {
                    "regime": condition.regime.value,
                    "volatility_level": condition.volatility_level,
                    "volume_profile": condition.volume_profile,
                    "trend_strength": condition.trend_strength,
                    "vix_level": condition.vix_level,
                    "confidence": condition.confidence,
                    "last_updated": condition.timestamp.isoformat()
                },
                "optimization_strategy": {
                    "strategy": settings.strategy.value,
                    "cpu_multiplier": settings.cpu_multiplier,
                    "memory_multiplier": settings.memory_multiplier,
                    "justification": settings.justification,
                    "confidence": settings.confidence
                },
                "performance_impact": performance_impact,
                "recommendations": [
                    "Monitor VIX levels for volatility regime changes",
                    "Track volume patterns for capacity planning", 
                    "Review optimization effectiveness hourly",
                    "Adjust risk parameters based on market regime"
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Error getting market insights: {str(e)}")
            return {"error": "Failed to generate market insights"}


async def main():
    """Test the Market Condition Optimizer"""
    logging.basicConfig(level=logging.INFO)
    
    optimizer = MarketConditionOptimizer()
    
    print("🌐 Testing Market Condition Optimizer")
    print("=" * 45)
    
    # Test market data collection
    print("\n📊 Collecting Real-time Market Data...")
    market_data = await optimizer.collect_real_time_market_data()
    
    if market_data:
        print("Market Data Collected:")
        for symbol, data in list(market_data.items())[:3]:  # Show first 3
            if isinstance(data, dict) and 'price' in data:
                print(f"  {symbol}: ${data['price']:.2f} ({data['change_pct']:+.2f}%)")
    
    # Test market condition analysis
    print("\n🔍 Analyzing Market Conditions...")
    condition = await optimizer.analyze_market_condition()
    
    print(f"Market Regime: {condition.regime.value}")
    print(f"Volatility Level: {condition.volatility_level:.2%}")
    print(f"Volume Profile: {condition.volume_profile:.1f}x")
    print(f"Trend Strength: {condition.trend_strength:+.2f}")
    print(f"VIX Level: {condition.vix_level:.1f}")
    print(f"Classification Confidence: {condition.confidence:.2%}")
    
    if condition.economic_events:
        print(f"Economic Events: {', '.join(condition.economic_events)}")
    
    # Test optimization
    print("\n⚡ Generating Optimization Settings...")
    settings = await optimizer.optimize_for_market_condition(condition)
    
    print(f"Strategy: {settings.strategy.value}")
    print(f"Resource Adjustments:")
    print(f"  CPU Multiplier: {settings.cpu_multiplier:.2f}x")
    print(f"  Memory Multiplier: {settings.memory_multiplier:.2f}x")
    print(f"  Network Multiplier: {settings.network_multiplier:.2f}x")
    print(f"Performance Adjustments:")
    print(f"  Batch Size: {settings.batch_size_adjustment:.2f}x")
    print(f"  Timeout: {settings.timeout_adjustment:.2f}x")
    print(f"  Latency Target: {settings.latency_target_adjustment:.2f}x")
    print(f"Risk Management:")
    print(f"  Position Size: {settings.position_size_multiplier:.2f}x")
    print(f"  Order Frequency: {settings.order_frequency_limit:.2f}x")
    print(f"Optimization Confidence: {settings.confidence:.2%}")
    print(f"\nJustification: {settings.justification}")
    
    # Test comprehensive insights
    print("\n📈 Market Insights Dashboard...")
    insights = await optimizer.get_current_market_insights()
    
    if "performance_impact" in insights:
        impact = insights["performance_impact"]
        print("Expected Performance Impact:")
        print(f"  Latency Improvement: {impact['latency_improvement']:+.1f}%")
        print(f"  Throughput Increase: {impact['throughput_increase']:+.1f}%")
        print(f"  Cost Impact: ${impact['cost_impact']:+.1f}/hour")
        print(f"  Reliability Score: {impact['reliability_score']:.1f}%")
    
    if "recommendations" in insights:
        print("\nRecommendations:")
        for i, rec in enumerate(insights["recommendations"], 1):
            print(f"  {i}. {rec}")
    
    print("\n✅ Market Condition Optimizer test completed!")


if __name__ == "__main__":
    asyncio.run(main())