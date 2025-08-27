#!/usr/bin/env python3
"""
Kyle's Lambda Calculator - Price Sensitivity and Adverse Selection Measurement
Implements Kyle's Lambda as an inverse proxy for liquidity - higher values indicate greater adverse selection.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
import statistics
import requests

logger = logging.getLogger(__name__)

@dataclass
class KylesLambdaConfig:
    """Configuration for Kyle's Lambda calculation"""
    regression_window: int = 100  # Number of observations for regression
    min_observations: int = 50   # Minimum observations required
    update_frequency_seconds: int = 30  # Update frequency
    outlier_threshold: float = 3.0  # Standard deviations for outlier detection
    feature_engine_url: str = "http://localhost:8500"

@dataclass
class KylesLambdaMetrics:
    """Kyle's Lambda calculation results"""
    timestamp: float
    symbol: str
    kyles_lambda: float
    regression_r_squared: float
    confidence_interval_95: Tuple[float, float]
    observations_count: int
    adverse_selection_level: str  # LOW, MODERATE, HIGH, EXTREME
    liquidity_proxy: float  # Inverse of Kyle's Lambda (higher = more liquid)
    price_impact_per_dollar: float  # Expected price impact per $1M volume
    regression_p_value: float
    recent_trend: str  # INCREASING, DECREASING, STABLE
    market_condition: str  # NORMAL, STRESSED, ILLIQUID
    statistical_significance: bool
    
@dataclass
class TradeObservation:
    """Individual trade observation for regression"""
    timestamp: float
    return_bps: float  # Return in basis points
    signed_sqrt_dollar_volume: float  # Signed square root of dollar volume
    raw_dollar_volume: float
    trade_direction: int  # 1 for buy, -1 for sell
    price: float
    volume: float

class KylesLambdaCalculator:
    """Kyle's Lambda implementation - measures price sensitivity to order flow"""
    
    def __init__(self, config: KylesLambdaConfig = None):
        self.config = config or KylesLambdaConfig()
        self.trade_observations = []
        self.lambda_history = []
        self.last_calculation_time = 0
        
    async def calculate_kyles_lambda(self, symbol: str, trade_data: List[Dict[str, Any]]) -> KylesLambdaMetrics:
        """
        Calculate Kyle's Lambda using regression of returns on signed square-root dollar volume
        Formula: Return = λ * SignedSqrtDollarVolume + ε
        Where λ is Kyle's Lambda (price impact parameter)
        """
        
        # Process trade data into observations
        observations = await self._process_trade_data(symbol, trade_data)
        
        if len(observations) < self.config.min_observations:
            return self._create_default_metrics(symbol, len(observations))
        
        # Prepare regression data
        returns = np.array([obs.return_bps for obs in observations])
        signed_volumes = np.array([obs.signed_sqrt_dollar_volume for obs in observations])
        
        # Remove outliers
        returns, signed_volumes = self._remove_outliers(returns, signed_volumes)
        
        if len(returns) < self.config.min_observations:
            return self._create_default_metrics(symbol, len(returns))
        
        # Perform regression: Return = λ * SignedSqrtDollarVolume + ε
        regression = LinearRegression()
        X = signed_volumes.reshape(-1, 1)
        y = returns
        
        regression.fit(X, y)
        
        # Extract Kyle's Lambda (the regression coefficient)
        kyles_lambda = regression.coef_[0]
        r_squared = regression.score(X, y)
        
        # Calculate confidence intervals
        residuals = y - regression.predict(X)
        mse = np.mean(residuals**2)
        se_lambda = np.sqrt(mse / np.sum((signed_volumes - np.mean(signed_volumes))**2))
        t_value = 1.96  # 95% confidence interval
        ci_lower = kyles_lambda - t_value * se_lambda
        ci_upper = kyles_lambda + t_value * se_lambda
        
        # Statistical significance (t-test)
        t_stat = kyles_lambda / se_lambda if se_lambda > 0 else 0
        p_value = 2 * (1 - abs(t_stat))  # Simplified p-value approximation
        is_significant = bool(abs(t_stat) > 1.96)  # Convert numpy.bool to Python bool
        
        # Classify adverse selection level
        adverse_selection_level = self._classify_adverse_selection(kyles_lambda)
        
        # Calculate liquidity proxy (inverse of Kyle's Lambda)
        liquidity_proxy = 1 / abs(kyles_lambda) if abs(kyles_lambda) > 1e-8 else float('inf')
        
        # Calculate price impact per $1M volume
        price_impact_per_million = abs(kyles_lambda) * np.sqrt(1_000_000)
        
        # Analyze trend
        recent_trend = self._analyze_trend()
        
        # Assess market condition
        market_condition = self._assess_market_condition(kyles_lambda, r_squared)
        
        # Create metrics object
        metrics = KylesLambdaMetrics(
            timestamp=datetime.now().timestamp(),
            symbol=symbol,
            kyles_lambda=kyles_lambda,
            regression_r_squared=r_squared,
            confidence_interval_95=(ci_lower, ci_upper),
            observations_count=len(observations),
            adverse_selection_level=adverse_selection_level,
            liquidity_proxy=liquidity_proxy,
            price_impact_per_dollar=price_impact_per_million,
            regression_p_value=p_value,
            recent_trend=recent_trend,
            market_condition=market_condition,
            statistical_significance=is_significant
        )
        
        # Store in history
        self.lambda_history.append(metrics)
        if len(self.lambda_history) > 200:  # Keep last 200 calculations
            self.lambda_history.pop(0)
            
        return metrics
    
    async def _process_trade_data(self, symbol: str, trade_data: List[Dict[str, Any]]) -> List[TradeObservation]:
        """Process raw trade data into structured observations"""
        observations = []
        
        # Sort by timestamp
        sorted_trades = sorted(trade_data, key=lambda x: x.get('timestamp', 0))
        
        for i, trade in enumerate(sorted_trades[1:], 1):  # Skip first trade (need previous for return)
            prev_trade = sorted_trades[i-1]
            
            # Calculate return in basis points
            current_price = float(trade.get('price', 0))
            prev_price = float(prev_trade.get('price', 1))
            
            if prev_price > 0:
                return_bps = (current_price / prev_price - 1) * 10000
            else:
                continue
                
            # Calculate dollar volume
            volume = float(trade.get('volume', 0))
            dollar_volume = current_price * volume
            
            # Determine trade direction (simplified - would need more sophisticated logic)
            trade_direction = self._determine_trade_direction(trade, prev_trade)
            
            # Calculate signed square root dollar volume
            signed_sqrt_dollar_volume = trade_direction * np.sqrt(dollar_volume)
            
            observation = TradeObservation(
                timestamp=float(trade.get('timestamp', datetime.now().timestamp())),
                return_bps=return_bps,
                signed_sqrt_dollar_volume=signed_sqrt_dollar_volume,
                raw_dollar_volume=dollar_volume,
                trade_direction=trade_direction,
                price=current_price,
                volume=volume
            )
            
            observations.append(observation)
        
        return observations[-self.config.regression_window:]  # Keep only recent window
    
    def _determine_trade_direction(self, current_trade: Dict, prev_trade: Dict) -> int:
        """
        Determine trade direction using tick rule
        Returns 1 for buy-initiated, -1 for sell-initiated
        """
        current_price = float(current_trade.get('price', 0))
        prev_price = float(prev_trade.get('price', 0))
        
        if current_price > prev_price:
            return 1  # Buy-initiated (uptick)
        elif current_price < prev_price:
            return -1  # Sell-initiated (downtick)
        else:
            # Use previous direction or default to buy
            return getattr(self, '_last_direction', 1)
    
    def _remove_outliers(self, returns: np.ndarray, volumes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Remove statistical outliers using z-score method"""
        
        # Calculate z-scores for both returns and volumes
        returns_z = np.abs((returns - np.mean(returns)) / np.std(returns)) if np.std(returns) > 0 else np.zeros_like(returns)
        volumes_z = np.abs((volumes - np.mean(volumes)) / np.std(volumes)) if np.std(volumes) > 0 else np.zeros_like(volumes)
        
        # Keep observations where both metrics are within threshold
        mask = (returns_z < self.config.outlier_threshold) & (volumes_z < self.config.outlier_threshold)
        
        return returns[mask], volumes[mask]
    
    def _classify_adverse_selection(self, kyles_lambda: float) -> str:
        """Classify the level of adverse selection based on Kyle's Lambda magnitude"""
        abs_lambda = abs(kyles_lambda)
        
        if abs_lambda < 0.01:
            return "LOW"
        elif abs_lambda < 0.05:
            return "MODERATE" 
        elif abs_lambda < 0.15:
            return "HIGH"
        else:
            return "EXTREME"
    
    def _analyze_trend(self) -> str:
        """Analyze the recent trend in Kyle's Lambda"""
        if len(self.lambda_history) < 5:
            return "INSUFFICIENT_DATA"
        
        recent_lambdas = [m.kyles_lambda for m in self.lambda_history[-5:]]
        
        # Simple trend analysis using linear regression
        x = np.arange(len(recent_lambdas))
        slope = np.polyfit(x, recent_lambdas, 1)[0]
        
        if slope > 0.01:
            return "INCREASING"
        elif slope < -0.01:
            return "DECREASING"
        else:
            return "STABLE"
    
    def _assess_market_condition(self, kyles_lambda: float, r_squared: float) -> str:
        """Assess overall market condition based on Kyle's Lambda and regression quality"""
        abs_lambda = abs(kyles_lambda)
        
        if r_squared < 0.1:
            return "NOISY"  # Poor regression fit
        elif abs_lambda > 0.1:
            return "STRESSED"  # High adverse selection
        elif abs_lambda < 0.02 and r_squared > 0.3:
            return "NORMAL"  # Low adverse selection, good fit
        else:
            return "TRANSITIONAL"
    
    def _create_default_metrics(self, symbol: str, obs_count: int) -> KylesLambdaMetrics:
        """Create default metrics when insufficient data"""
        return KylesLambdaMetrics(
            timestamp=datetime.now().timestamp(),
            symbol=symbol,
            kyles_lambda=0.0,
            regression_r_squared=0.0,
            confidence_interval_95=(0.0, 0.0),
            observations_count=obs_count,
            adverse_selection_level="UNKNOWN",
            liquidity_proxy=float('inf'),
            price_impact_per_dollar=0.0,
            regression_p_value=1.0,
            recent_trend="INSUFFICIENT_DATA",
            market_condition="INSUFFICIENT_DATA",
            statistical_significance=False
        )
    
    async def get_enhanced_lambda_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive Kyle's Lambda analysis with Feature Engine integration"""
        
        if not self.lambda_history:
            return {"status": "no_data", "message": "No Kyle's Lambda calculations available"}
        
        latest_metrics = self.lambda_history[-1]
        
        # Calculate statistics over history
        lambda_values = [m.kyles_lambda for m in self.lambda_history[-20:]]  # Last 20 calculations
        r_squared_values = [m.regression_r_squared for m in self.lambda_history[-20:]]
        
        analysis = {
            "current_metrics": {
                "kyles_lambda": latest_metrics.kyles_lambda,
                "adverse_selection_level": latest_metrics.adverse_selection_level,
                "liquidity_proxy": latest_metrics.liquidity_proxy,
                "price_impact_per_million": latest_metrics.price_impact_per_dollar,
                "market_condition": latest_metrics.market_condition,
                "statistical_significance": latest_metrics.statistical_significance
            },
            "historical_statistics": {
                "mean_lambda": statistics.mean(lambda_values),
                "std_lambda": statistics.stdev(lambda_values) if len(lambda_values) > 1 else 0,
                "min_lambda": min(lambda_values),
                "max_lambda": max(lambda_values),
                "mean_r_squared": statistics.mean(r_squared_values),
                "trend": latest_metrics.recent_trend
            },
            "liquidity_assessment": {
                "liquidity_level": "HIGH" if latest_metrics.liquidity_proxy > 100 else "MODERATE" if latest_metrics.liquidity_proxy > 20 else "LOW",
                "market_impact_warning": latest_metrics.price_impact_per_dollar > 50,  # High impact threshold
                "adverse_selection_warning": latest_metrics.adverse_selection_level in ["HIGH", "EXTREME"]
            },
            "regression_quality": {
                "r_squared": latest_metrics.regression_r_squared,
                "observations_count": latest_metrics.observations_count,
                "confidence_interval": latest_metrics.confidence_interval_95,
                "p_value": latest_metrics.regression_p_value
            }
        }
        
        return analysis

# Global calculator instance
kyles_lambda_calculator = KylesLambdaCalculator()