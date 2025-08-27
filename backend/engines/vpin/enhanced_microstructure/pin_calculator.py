#!/usr/bin/env python3
"""
PIN (Probability of Informed Trading) Calculator
Original model from Easley, Kiefer, O'Hara, and Paperman (1996) that VPIN evolved from.
Includes both classic PIN and enhanced Two-Step PIN for better estimation precision.
"""

import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import statistics
from scipy.optimize import minimize
from scipy.stats import poisson
import warnings

logger = logging.getLogger(__name__)

@dataclass
class PINConfig:
    """Configuration for PIN calculation"""
    estimation_method: str = "classic"  # "classic" or "two_step"
    convergence_tolerance: float = 1e-6
    max_iterations: int = 1000
    initial_parameters: Optional[Dict[str, float]] = None
    daily_aggregation: bool = True  # Aggregate to daily data
    min_days_required: int = 20
    
@dataclass
class DailyTradingData:
    """Daily aggregated trading data for PIN calculation"""
    date: datetime
    symbol: str
    buy_volume: float
    sell_volume: float
    total_volume: float
    buy_count: int
    sell_count: int
    total_trades: int
    opening_price: float
    closing_price: float
    high_price: float
    low_price: float

@dataclass
class PINParameters:
    """PIN model parameters from maximum likelihood estimation"""
    alpha: float      # Probability of information event
    delta: float      # Probability of bad news given information event
    epsilon_b: float  # Arrival rate of uninformed buy orders
    epsilon_s: float  # Arrival rate of uninformed sell orders
    mu: float         # Arrival rate of informed orders
    pin: float        # Computed PIN value
    log_likelihood: float
    convergence_achieved: bool
    iterations: int

@dataclass
class PINAnalysis:
    """Comprehensive PIN analysis results"""
    timestamp: float
    symbol: str
    pin_value: float
    pin_classification: str  # LOW, MODERATE, HIGH, EXTREME
    parameters: PINParameters
    two_step_pin: Optional[float]  # Two-Step PIN if available
    comparison_with_vpin: Optional[float]  # VPIN comparison if available
    information_intensity: float  # mu / (epsilon_b + epsilon_s + mu)
    order_imbalance_measure: float
    market_liquidity_indicator: float
    statistical_confidence: float
    days_analyzed: int
    estimation_quality: str  # EXCELLENT, GOOD, FAIR, POOR

class PINCalculator:
    """
    PIN (Probability of Informed Trading) calculator using maximum likelihood estimation.
    Implements both classic PIN and Two-Step PIN methodologies.
    """
    
    def __init__(self, config: PINConfig = None):
        self.config = config or PINConfig()
        self.daily_data = []
        self.pin_history = []
        
    async def calculate_pin(self, symbol: str, trading_data: List[Dict[str, Any]]) -> PINAnalysis:
        """Calculate PIN using maximum likelihood estimation"""
        
        # Aggregate data to daily level
        daily_data = await self._aggregate_to_daily(symbol, trading_data)
        
        if len(daily_data) < self.config.min_days_required:
            return self._create_default_analysis(symbol, len(daily_data))
        
        # Calculate classic PIN
        pin_params = await self._estimate_classic_pin(daily_data)
        
        # Calculate Two-Step PIN if requested
        two_step_pin = None
        if self.config.estimation_method == "two_step" or self.config.estimation_method == "both":
            two_step_pin = await self._estimate_two_step_pin(daily_data, pin_params)
        
        # Create comprehensive analysis
        analysis = await self._create_comprehensive_analysis(
            symbol, daily_data, pin_params, two_step_pin
        )
        
        # Store in history
        self.pin_history.append(analysis)
        if len(self.pin_history) > 100:  # Keep last 100 calculations
            self.pin_history.pop(0)
            
        return analysis
    
    async def calculate_classic_pin(self, symbol: str, trading_data: List[Dict[str, Any]]) -> PINAnalysis:
        """Calculate classic PIN using maximum likelihood estimation"""
        # Set method to classic and delegate to main method
        original_method = self.config.estimation_method
        self.config.estimation_method = "classic"
        result = await self.calculate_pin(symbol, trading_data)
        self.config.estimation_method = original_method
        return result
    
    async def calculate_two_step_pin(self, symbol: str, trading_data: List[Dict[str, Any]]) -> PINAnalysis:
        """Calculate Two-Step PIN for enhanced precision"""
        # Set method to two_step and delegate to main method
        original_method = self.config.estimation_method
        self.config.estimation_method = "two_step"
        result = await self.calculate_pin(symbol, trading_data)
        self.config.estimation_method = original_method
        return result
    
    async def _aggregate_to_daily(self, symbol: str, trading_data: List[Dict[str, Any]]) -> List[DailyTradingData]:
        """Aggregate tick-by-tick data to daily trading data"""
        
        # Group by date
        daily_groups = {}
        
        for trade in trading_data:
            timestamp = float(trade.get('timestamp', 0))
            trade_date = datetime.fromtimestamp(timestamp).date()
            
            if trade_date not in daily_groups:
                daily_groups[trade_date] = []
            daily_groups[trade_date].append(trade)
        
        # Aggregate each day
        daily_data = []
        for date, trades in daily_groups.items():
            if len(trades) < 10:  # Skip days with too few trades
                continue
                
            # Classify trades as buys/sells (simplified tick rule)
            buys = []
            sells = []
            
            prev_price = None
            for trade in sorted(trades, key=lambda x: x.get('timestamp', 0)):
                price = float(trade.get('price', 0))
                volume = float(trade.get('volume', 0))
                
                if prev_price is not None:
                    if price > prev_price:
                        buys.append(trade)
                    elif price < prev_price:
                        sells.append(trade)
                    # Equal prices inherit previous classification or default to buy
                
                prev_price = price
            
            # Calculate daily aggregates
            buy_volume = sum(float(t.get('volume', 0)) for t in buys)
            sell_volume = sum(float(t.get('volume', 0)) for t in sells)
            total_volume = buy_volume + sell_volume
            
            prices = [float(t.get('price', 0)) for t in trades]
            
            daily_data.append(DailyTradingData(
                date=datetime.combine(date, datetime.min.time()),
                symbol=symbol,
                buy_volume=buy_volume,
                sell_volume=sell_volume,
                total_volume=total_volume,
                buy_count=len(buys),
                sell_count=len(sells),
                total_trades=len(trades),
                opening_price=prices[0] if prices else 0,
                closing_price=prices[-1] if prices else 0,
                high_price=max(prices) if prices else 0,
                low_price=min(prices) if prices else 0
            ))
        
        return daily_data
    
    async def _estimate_classic_pin(self, daily_data: List[DailyTradingData]) -> PINParameters:
        """Estimate classic PIN using maximum likelihood estimation"""
        
        # Prepare data arrays
        buys = np.array([d.buy_count for d in daily_data])
        sells = np.array([d.sell_count for d in daily_data])
        
        # Initial parameter estimates
        if self.config.initial_parameters:
            initial = [
                self.config.initial_parameters.get('alpha', 0.5),
                self.config.initial_parameters.get('delta', 0.5),
                self.config.initial_parameters.get('epsilon_b', np.mean(buys)),
                self.config.initial_parameters.get('epsilon_s', np.mean(sells)),
                self.config.initial_parameters.get('mu', np.std(buys) + np.std(sells))
            ]
        else:
            # Method of moments initial estimates
            mean_buys = np.mean(buys)
            mean_sells = np.mean(sells)
            var_buys = np.var(buys)
            var_sells = np.var(sells)
            
            initial = [
                0.5,  # alpha
                0.5,  # delta
                mean_buys,  # epsilon_b
                mean_sells,  # epsilon_s
                max(np.sqrt(var_buys - mean_buys), 0.1)  # mu
            ]
        
        # Define bounds for parameters
        bounds = [
            (0.001, 0.999),  # alpha
            (0.001, 0.999),  # delta
            (0.1, None),     # epsilon_b
            (0.1, None),     # epsilon_s
            (0.1, None)      # mu
        ]
        
        try:
            # Minimize negative log-likelihood
            result = minimize(
                self._negative_log_likelihood,
                initial,
                args=(buys, sells),
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'maxiter': self.config.max_iterations,
                    'ftol': self.config.convergence_tolerance
                }
            )
            
            if result.success:
                alpha, delta, epsilon_b, epsilon_s, mu = result.x
                pin = (alpha * mu) / (alpha * mu + epsilon_b + epsilon_s)
                
                return PINParameters(
                    alpha=alpha,
                    delta=delta,
                    epsilon_b=epsilon_b,
                    epsilon_s=epsilon_s,
                    mu=mu,
                    pin=pin,
                    log_likelihood=-result.fun,
                    convergence_achieved=True,
                    iterations=result.nit
                )
            else:
                logger.warning(f"PIN optimization failed: {result.message}")
                
        except Exception as e:
            logger.error(f"PIN calculation error: {e}")
        
        # Return default parameters if optimization fails
        return PINParameters(
            alpha=0.5,
            delta=0.5,
            epsilon_b=np.mean(buys),
            epsilon_s=np.mean(sells),
            mu=1.0,
            pin=0.5,
            log_likelihood=0.0,
            convergence_achieved=False,
            iterations=0
        )
    
    def _negative_log_likelihood(self, params: np.ndarray, buys: np.ndarray, sells: np.ndarray) -> float:
        """Calculate negative log-likelihood for PIN model"""
        
        alpha, delta, epsilon_b, epsilon_s, mu = params
        
        # Ensure parameters are positive and alpha, delta are probabilities
        if alpha <= 0 or alpha >= 1 or delta <= 0 or delta >= 1:
            return np.inf
        if epsilon_b <= 0 or epsilon_s <= 0 or mu <= 0:
            return np.inf
        
        log_likelihood = 0.0
        
        for B, S in zip(buys, sells):
            try:
                # Three states in PIN model:
                # 1. No information event (probability 1-alpha)
                L1 = poisson.pmf(B, epsilon_b) * poisson.pmf(S, epsilon_s)
                
                # 2. Bad news (probability alpha * delta)
                L2 = poisson.pmf(B, epsilon_b) * poisson.pmf(S, epsilon_s + mu)
                
                # 3. Good news (probability alpha * (1-delta))
                L3 = poisson.pmf(B, epsilon_b + mu) * poisson.pmf(S, epsilon_s)
                
                # Total likelihood for this day
                likelihood = (1 - alpha) * L1 + alpha * delta * L2 + alpha * (1 - delta) * L3
                
                if likelihood > 0:
                    log_likelihood += np.log(likelihood)
                else:
                    return np.inf
                    
            except (OverflowError, ZeroDivisionError, ValueError):
                return np.inf
        
        return -log_likelihood  # Return negative for minimization
    
    async def _estimate_two_step_pin(self, daily_data: List[DailyTradingData], 
                                   classic_params: PINParameters) -> Optional[float]:
        """Estimate Two-Step PIN for enhanced precision"""
        
        try:
            # Two-Step PIN uses factorization approach to improve estimation
            # Step 1: Estimate arrival rates using method of moments
            buys = np.array([d.buy_count for d in daily_data])
            sells = np.array([d.sell_count for d in daily_data])
            
            # Enhanced initial estimates
            mean_B = np.mean(buys)
            mean_S = np.mean(sells)
            var_B = np.var(buys)
            var_S = np.var(sells)
            
            # Use classic PIN as starting point and refine
            alpha_2step = classic_params.alpha
            delta_2step = classic_params.delta
            
            # Refined estimation using moment conditions
            if var_B > mean_B and var_S > mean_S:
                # Overdispersion suggests informed trading
                mu_estimate = np.sqrt(max(var_B - mean_B, var_S - mean_S))
                epsilon_b_estimate = mean_B - alpha_2step * (1 - delta_2step) * mu_estimate
                epsilon_s_estimate = mean_S - alpha_2step * delta_2step * mu_estimate
                
                if epsilon_b_estimate > 0 and epsilon_s_estimate > 0:
                    two_step_pin = (alpha_2step * mu_estimate) / (
                        alpha_2step * mu_estimate + epsilon_b_estimate + epsilon_s_estimate
                    )
                    return max(0.0, min(1.0, two_step_pin))
            
            # Fallback to classic PIN
            return classic_params.pin
            
        except Exception as e:
            logger.warning(f"Two-Step PIN calculation failed: {e}")
            return classic_params.pin
    
    async def _create_comprehensive_analysis(self, symbol: str, daily_data: List[DailyTradingData],
                                           pin_params: PINParameters, 
                                           two_step_pin: Optional[float]) -> PINAnalysis:
        """Create comprehensive PIN analysis"""
        
        # Classify PIN level
        pin_classification = self._classify_pin_level(pin_params.pin)
        
        # Calculate additional metrics
        info_intensity = pin_params.mu / (pin_params.epsilon_b + pin_params.epsilon_s + pin_params.mu)
        
        # Order imbalance measure
        total_buy_volume = sum(d.buy_volume for d in daily_data)
        total_sell_volume = sum(d.sell_volume for d in daily_data)
        total_volume = total_buy_volume + total_sell_volume
        order_imbalance = (total_buy_volume - total_sell_volume) / total_volume if total_volume > 0 else 0
        
        # Liquidity indicator (inverse of PIN)
        liquidity_indicator = 1 - pin_params.pin
        
        # Statistical confidence based on convergence and likelihood
        confidence = self._calculate_statistical_confidence(pin_params)
        
        # Estimation quality
        quality = self._assess_estimation_quality(pin_params, len(daily_data))
        
        return PINAnalysis(
            timestamp=datetime.now().timestamp(),
            symbol=symbol,
            pin_value=pin_params.pin,
            pin_classification=pin_classification,
            parameters=pin_params,
            two_step_pin=two_step_pin,
            comparison_with_vpin=None,  # Would be filled by external comparison
            information_intensity=info_intensity,
            order_imbalance_measure=order_imbalance,
            market_liquidity_indicator=liquidity_indicator,
            statistical_confidence=confidence,
            days_analyzed=len(daily_data),
            estimation_quality=quality
        )
    
    def _classify_pin_level(self, pin_value: float) -> str:
        """Classify PIN level based on research benchmarks"""
        if pin_value < 0.1:
            return "LOW"
        elif pin_value < 0.25:
            return "MODERATE"
        elif pin_value < 0.4:
            return "HIGH"
        else:
            return "EXTREME"
    
    def _calculate_statistical_confidence(self, params: PINParameters) -> float:
        """Calculate statistical confidence in PIN estimate"""
        if not params.convergence_achieved:
            return 0.0
        
        # Base confidence on convergence and parameter reasonableness
        base_confidence = 0.7 if params.convergence_achieved else 0.3
        
        # Adjust for parameter reasonableness
        if 0.01 < params.alpha < 0.99 and 0.01 < params.delta < 0.99:
            base_confidence += 0.2
        
        # Adjust for likelihood quality
        if params.log_likelihood > -1000:  # Reasonable likelihood
            base_confidence += 0.1
            
        return min(base_confidence, 1.0)
    
    def _assess_estimation_quality(self, params: PINParameters, days_count: int) -> str:
        """Assess quality of PIN estimation"""
        if not params.convergence_achieved:
            return "POOR"
        
        if days_count >= 50 and params.log_likelihood > -500:
            return "EXCELLENT"
        elif days_count >= 30 and params.log_likelihood > -1000:
            return "GOOD"
        elif days_count >= 20:
            return "FAIR"
        else:
            return "POOR"
    
    def _create_default_analysis(self, symbol: str, days_count: int) -> PINAnalysis:
        """Create default analysis when insufficient data"""
        default_params = PINParameters(
            alpha=0.0, delta=0.0, epsilon_b=0.0, epsilon_s=0.0, mu=0.0,
            pin=0.0, log_likelihood=0.0, convergence_achieved=False, iterations=0
        )
        
        return PINAnalysis(
            timestamp=datetime.now().timestamp(),
            symbol=symbol,
            pin_value=0.0,
            pin_classification="INSUFFICIENT_DATA",
            parameters=default_params,
            two_step_pin=None,
            comparison_with_vpin=None,
            information_intensity=0.0,
            order_imbalance_measure=0.0,
            market_liquidity_indicator=1.0,
            statistical_confidence=0.0,
            days_analyzed=days_count,
            estimation_quality="INSUFFICIENT_DATA"
        )
    
    async def get_pin_analysis_summary(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive PIN analysis summary"""
        
        if not self.pin_history:
            return {"status": "no_data", "message": "No PIN calculations available"}
        
        # Get recent analyses for symbol
        symbol_analyses = [a for a in self.pin_history if a.symbol == symbol]
        
        if not symbol_analyses:
            return {"status": "no_symbol_data", "message": f"No PIN data for {symbol}"}
        
        latest = symbol_analyses[-1]
        
        # Calculate trend if multiple analyses available
        trend = "STABLE"
        if len(symbol_analyses) >= 3:
            recent_pins = [a.pin_value for a in symbol_analyses[-3:]]
            if recent_pins[-1] > recent_pins[0] * 1.1:
                trend = "INCREASING"
            elif recent_pins[-1] < recent_pins[0] * 0.9:
                trend = "DECREASING"
        
        return {
            "current_pin_analysis": {
                "pin_value": latest.pin_value,
                "classification": latest.pin_classification,
                "two_step_pin": latest.two_step_pin,
                "information_intensity": latest.information_intensity,
                "liquidity_indicator": latest.market_liquidity_indicator,
                "estimation_quality": latest.estimation_quality
            },
            "model_parameters": {
                "alpha_information_probability": latest.parameters.alpha,
                "delta_bad_news_probability": latest.parameters.delta,
                "epsilon_b_uninformed_buys": latest.parameters.epsilon_b,
                "epsilon_s_uninformed_sells": latest.parameters.epsilon_s,
                "mu_informed_arrival_rate": latest.parameters.mu
            },
            "statistical_analysis": {
                "convergence_achieved": latest.parameters.convergence_achieved,
                "log_likelihood": latest.parameters.log_likelihood,
                "statistical_confidence": latest.statistical_confidence,
                "days_analyzed": latest.days_analyzed,
                "iterations": latest.parameters.iterations
            },
            "market_insights": {
                "order_imbalance": latest.order_imbalance_measure,
                "trend": trend,
                "adverse_selection_risk": "HIGH" if latest.pin_value > 0.3 else "MODERATE" if latest.pin_value > 0.15 else "LOW"
            }
        }

# Global calculator instance
pin_calculator = PINCalculator()