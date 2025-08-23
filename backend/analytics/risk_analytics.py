"""
Advanced Risk Analytics Engine for Sprint 3 Priority 2
Risk metrics and calculations including VaR, exposure analysis, correlation, and stress testing
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
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf, EmpiricalCovariance
import json

logger = logging.getLogger(__name__)

class VaRMethod(Enum):
    """Value at Risk calculation methods"""
    HISTORICAL = "historical"
    PARAMETRIC = "parametric"
    MONTE_CARLO = "monte_carlo"
    EXTREME_VALUE = "extreme_value"

class StressScenario(Enum):
    """Predefined stress testing scenarios"""
    MARKET_CRASH = "market_crash"
    INTEREST_RATE_SHOCK = "interest_rate_shock"
    VOLATILITY_SPIKE = "volatility_spike"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    SECTOR_ROTATION = "sector_rotation"
    CURRENCY_CRISIS = "currency_crisis"

@dataclass
class VaRResult:
    """Value at Risk calculation result"""
    portfolio_id: str
    method: VaRMethod
    confidence_level: float
    time_horizon: int
    var_amount: Decimal
    expected_shortfall: Decimal
    calculation_timestamp: datetime
    observations_used: int
    model_parameters: Optional[Dict[str, Any]] = None

@dataclass
class ExposureAnalysis:
    """Portfolio exposure analysis result"""
    portfolio_id: str
    timestamp: datetime
    total_exposure: Decimal
    net_exposure: Decimal
    gross_exposure: Decimal
    long_exposure: Decimal
    short_exposure: Decimal
    exposure_by_asset_class: Dict[str, Decimal]
    exposure_by_sector: Dict[str, Decimal]
    exposure_by_currency: Dict[str, Decimal]
    concentration_metrics: Dict[str, float]

@dataclass
class CorrelationAnalysis:
    """Correlation analysis result"""
    portfolio_id: str
    timestamp: datetime
    correlation_matrix: np.ndarray
    asset_ids: List[str]
    average_correlation: float
    max_correlation: float
    min_correlation: float
    eigenvalues: List[float]
    diversification_ratio: float

@dataclass
class StressTestResult:
    """Stress test scenario result"""
    portfolio_id: str
    scenario: StressScenario
    scenario_name: str
    portfolio_impact: Decimal
    impact_percentage: float
    positions_affected: int
    worst_position_impact: Decimal
    var_breach_probability: float
    recovery_time_estimate: Optional[int]
    stress_factors: Dict[str, float]

class RiskAnalytics:
    """
    Advanced risk analytics engine for comprehensive portfolio risk management
    """
    
    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool
        self.logger = logging.getLogger(__name__)
        
        # Risk parameters
        self.confidence_levels = [0.90, 0.95, 0.99]
        self.time_horizons = [1, 5, 10, 22]  # Days
        self.lookback_days = 252  # 1 year for calculations
        
        # Stress test scenarios
        self.stress_scenarios = {
            StressScenario.MARKET_CRASH: {
                "equity_shock": -0.20,
                "bond_shock": 0.05,
                "vol_multiplier": 2.0,
                "correlation_increase": 0.3
            },
            StressScenario.INTEREST_RATE_SHOCK: {
                "rate_shock": 0.02,
                "bond_shock": -0.15,
                "equity_shock": -0.10,
                "currency_shock": 0.05
            },
            StressScenario.VOLATILITY_SPIKE: {
                "vol_multiplier": 3.0,
                "correlation_increase": 0.5,
                "equity_shock": -0.15
            },
            StressScenario.LIQUIDITY_CRISIS: {
                "bid_ask_widening": 3.0,
                "equity_shock": -0.25,
                "credit_spread_widening": 0.03
            }
        }
    
    async def calculate_var(
        self,
        portfolio_id: str,
        method: VaRMethod = VaRMethod.HISTORICAL,
        confidence_level: float = 0.95,
        time_horizon: int = 1,
        lookback_days: int = 252
    ) -> VaRResult:
        """
        Calculate Value at Risk using specified method
        
        Args:
            portfolio_id: Portfolio identifier
            method: VaR calculation method
            confidence_level: Confidence level (0.90, 0.95, 0.99)
            time_horizon: Time horizon in days
            lookback_days: Historical data lookback period
            
        Returns:
            VaRResult with calculated VaR and Expected Shortfall
        """
        async with self.db_pool.acquire() as conn:
            try:
                # Get portfolio returns
                end_date = datetime.utcnow()
                start_date = end_date - timedelta(days=lookback_days)
                
                portfolio_returns = await self._get_portfolio_returns(
                    conn, portfolio_id, start_date, end_date
                )
                
                if len(portfolio_returns) < 30:
                    raise ValueError(f"Insufficient data for VaR calculation: {len(portfolio_returns)} observations")
                
                returns_array = np.array(portfolio_returns)
                
                # Calculate VaR based on method
                if method == VaRMethod.HISTORICAL:
                    var_amount, expected_shortfall, params = await self._calculate_historical_var(
                        returns_array, confidence_level, time_horizon
                    )
                elif method == VaRMethod.PARAMETRIC:
                    var_amount, expected_shortfall, params = await self._calculate_parametric_var(
                        returns_array, confidence_level, time_horizon
                    )
                elif method == VaRMethod.MONTE_CARLO:
                    var_amount, expected_shortfall, params = await self._calculate_monte_carlo_var(
                        returns_array, confidence_level, time_horizon
                    )
                else:
                    raise ValueError(f"Unsupported VaR method: {method}")
                
                # Get current portfolio value for scaling
                portfolio_value = await self._get_portfolio_value(conn, portfolio_id)
                
                var_result = VaRResult(
                    portfolio_id=portfolio_id,
                    method=method,
                    confidence_level=confidence_level,
                    time_horizon=time_horizon,
                    var_amount=Decimal(str(var_amount * portfolio_value)),
                    expected_shortfall=Decimal(str(expected_shortfall * portfolio_value)),
                    calculation_timestamp=datetime.utcnow(),
                    observations_used=len(returns_array),
                    model_parameters=params
                )
                
                # Store result
                await self._store_var_result(conn, var_result)
                
                return var_result
                
            except Exception as e:
                self.logger.error(f"Error calculating VaR: {e}")
                raise
    
    async def analyze_portfolio_exposure(
        self,
        portfolio_id: str
    ) -> ExposureAnalysis:
        """
        Analyze portfolio exposure across different dimensions
        """
        async with self.db_pool.acquire() as conn:
            try:
                exposure_query = """
                    SELECT 
                        p.instrument_id,
                        p.quantity,
                        p.avg_entry_price,
                        i.asset_class,
                        i.currency,
                        i.multiplier,
                        i.symbol,
                        i.metadata::json->>'sector' as sector,
                        (p.quantity * p.avg_entry_price * i.multiplier) as position_value,
                        CASE WHEN p.quantity > 0 THEN 'LONG' ELSE 'SHORT' END as side
                    FROM positions p
                    JOIN instruments i ON p.instrument_id = i.instrument_id
                    WHERE p.portfolio_id = $1 AND p.quantity != 0
                """
                
                positions = await conn.fetch(exposure_query, portfolio_id)
                
                total_long = Decimal('0')
                total_short = Decimal('0')
                exposure_by_asset_class = {}
                exposure_by_sector = {}
                exposure_by_currency = {}
                position_values = []
                
                for position in positions:
                    position_value = Decimal(str(position['position_value']))
                    abs_value = abs(position_value)
                    
                    position_values.append(float(abs_value))
                    
                    # Long/Short exposure
                    if position['side'] == 'LONG':
                        total_long += abs_value
                    else:
                        total_short += abs_value
                    
                    # Asset class exposure
                    asset_class = position['asset_class']
                    exposure_by_asset_class[asset_class] = (
                        exposure_by_asset_class.get(asset_class, Decimal('0')) + abs_value
                    )
                    
                    # Sector exposure
                    sector = position['sector'] or 'Unknown'
                    exposure_by_sector[sector] = (
                        exposure_by_sector.get(sector, Decimal('0')) + abs_value
                    )
                    
                    # Currency exposure
                    currency = position['currency']
                    exposure_by_currency[currency] = (
                        exposure_by_currency.get(currency, Decimal('0')) + abs_value
                    )
                
                gross_exposure = total_long + total_short
                net_exposure = total_long - total_short
                
                # Concentration metrics
                concentration_metrics = self._calculate_concentration_metrics(position_values)
                
                exposure_analysis = ExposureAnalysis(
                    portfolio_id=portfolio_id,
                    timestamp=datetime.utcnow(),
                    total_exposure=gross_exposure,
                    net_exposure=net_exposure,
                    gross_exposure=gross_exposure,
                    long_exposure=total_long,
                    short_exposure=total_short,
                    exposure_by_asset_class={k: float(v) for k, v in exposure_by_asset_class.items()},
                    exposure_by_sector={k: float(v) for k, v in exposure_by_sector.items()},
                    exposure_by_currency={k: float(v) for k, v in exposure_by_currency.items()},
                    concentration_metrics=concentration_metrics
                )
                
                # Store analysis
                await self._store_exposure_analysis(conn, exposure_analysis)
                
                return exposure_analysis
                
            except Exception as e:
                self.logger.error(f"Error analyzing portfolio exposure: {e}")
                raise
    
    async def calculate_correlation_analysis(
        self,
        portfolio_id: str,
        lookback_days: int = 90
    ) -> CorrelationAnalysis:
        """
        Calculate correlation analysis for portfolio holdings
        """
        async with self.db_pool.acquire() as conn:
            try:
                # Get portfolio instruments
                instruments_query = """
                    SELECT DISTINCT p.instrument_id, i.symbol
                    FROM positions p
                    JOIN instruments i ON p.instrument_id = i.instrument_id
                    WHERE p.portfolio_id = $1 AND p.quantity != 0
                """
                
                instruments = await conn.fetch(instruments_query, portfolio_id)
                
                if len(instruments) < 2:
                    raise ValueError("Need at least 2 instruments for correlation analysis")
                
                # Get return data for each instrument
                end_date = datetime.utcnow()
                start_date = end_date - timedelta(days=lookback_days)
                
                returns_matrix = []
                asset_ids = []
                
                for instrument in instruments:
                    instrument_id = instrument['instrument_id']
                    returns = await self._get_instrument_returns(
                        conn, instrument_id, start_date, end_date
                    )
                    
                    if len(returns) >= 20:  # Minimum observations
                        returns_matrix.append(returns)
                        asset_ids.append(instrument_id)
                
                if len(returns_matrix) < 2:
                    raise ValueError("Insufficient return data for correlation analysis")
                
                # Align return series and calculate correlation
                min_length = min(len(series) for series in returns_matrix)
                aligned_returns = np.array([series[-min_length:] for series in returns_matrix])
                
                # Calculate correlation matrix
                correlation_matrix = np.corrcoef(aligned_returns)
                
                # Calculate metrics
                mask = np.triu(np.ones_like(correlation_matrix), k=1)
                correlations = correlation_matrix[mask == 1]
                
                avg_correlation = np.mean(correlations)
                max_correlation = np.max(correlations)
                min_correlation = np.min(correlations)
                
                # Eigenvalue analysis
                eigenvalues = np.linalg.eigvals(correlation_matrix)
                eigenvalues = sorted(eigenvalues, reverse=True)
                
                # Diversification ratio (approximate)
                portfolio_weights = np.ones(len(asset_ids)) / len(asset_ids)
                portfolio_var = np.dot(portfolio_weights, np.dot(correlation_matrix, portfolio_weights))
                individual_var = np.mean(np.diag(correlation_matrix))
                diversification_ratio = individual_var / portfolio_var if portfolio_var > 0 else 1.0
                
                correlation_analysis = CorrelationAnalysis(
                    portfolio_id=portfolio_id,
                    timestamp=datetime.utcnow(),
                    correlation_matrix=correlation_matrix,
                    asset_ids=asset_ids,
                    average_correlation=float(avg_correlation),
                    max_correlation=float(max_correlation),
                    min_correlation=float(min_correlation),
                    eigenvalues=[float(e) for e in eigenvalues],
                    diversification_ratio=float(diversification_ratio)
                )
                
                # Store analysis
                await self._store_correlation_analysis(conn, correlation_analysis)
                
                return correlation_analysis
                
            except Exception as e:
                self.logger.error(f"Error calculating correlation analysis: {e}")
                raise
    
    async def run_stress_test(
        self,
        portfolio_id: str,
        scenario: StressScenario,
        custom_factors: Optional[Dict[str, float]] = None
    ) -> StressTestResult:
        """
        Run stress test scenario on portfolio
        """
        async with self.db_pool.acquire() as conn:
            try:
                # Get portfolio positions
                positions_query = """
                    SELECT 
                        p.instrument_id,
                        p.quantity,
                        p.avg_entry_price,
                        i.asset_class,
                        i.currency,
                        i.multiplier,
                        i.symbol,
                        i.metadata::json->>'sector' as sector,
                        (p.quantity * p.avg_entry_price * i.multiplier) as position_value
                    FROM positions p
                    JOIN instruments i ON p.instrument_id = i.instrument_id
                    WHERE p.portfolio_id = $1 AND p.quantity != 0
                """
                
                positions = await conn.fetch(positions_query, portfolio_id)
                
                # Get stress factors
                stress_factors = custom_factors or self.stress_scenarios[scenario]
                
                total_impact = Decimal('0')
                positions_affected = 0
                worst_position_impact = Decimal('0')
                position_impacts = []
                
                for position in positions:
                    position_value = Decimal(str(position['position_value']))
                    asset_class = position['asset_class']
                    
                    # Apply stress factors based on asset class
                    stress_factor = 0.0
                    
                    if asset_class == 'STK':  # Equity
                        stress_factor = stress_factors.get('equity_shock', 0.0)
                    elif asset_class == 'BOND':  # Fixed Income
                        stress_factor = stress_factors.get('bond_shock', 0.0)
                    elif asset_class == 'CASH':  # FX
                        stress_factor = stress_factors.get('currency_shock', 0.0)
                    elif asset_class == 'FUT':  # Futures/Commodities
                        stress_factor = stress_factors.get('commodity_shock', -0.10)
                    
                    # Apply volatility multiplier if present
                    if 'vol_multiplier' in stress_factors and stress_factor != 0:
                        stress_factor *= stress_factors['vol_multiplier']
                    
                    position_impact = position_value * Decimal(str(stress_factor))
                    total_impact += position_impact
                    position_impacts.append(float(position_impact))
                    
                    if abs(position_impact) > abs(worst_position_impact):
                        worst_position_impact = position_impact
                    
                    if stress_factor != 0:
                        positions_affected += 1
                
                # Calculate impact percentage
                portfolio_value = await self._get_portfolio_value(conn, portfolio_id)
                impact_percentage = float(total_impact / portfolio_value * 100) if portfolio_value > 0 else 0
                
                # Estimate VaR breach probability (simplified)
                var_breach_probability = min(abs(impact_percentage) / 10, 1.0)  # Rough approximation
                
                # Estimate recovery time (simplified)
                recovery_time = None
                if abs(impact_percentage) > 5:
                    recovery_time = int(abs(impact_percentage) * 2)  # Rough estimate in days
                
                stress_result = StressTestResult(
                    portfolio_id=portfolio_id,
                    scenario=scenario,
                    scenario_name=scenario.value,
                    portfolio_impact=total_impact,
                    impact_percentage=impact_percentage,
                    positions_affected=positions_affected,
                    worst_position_impact=worst_position_impact,
                    var_breach_probability=var_breach_probability,
                    recovery_time_estimate=recovery_time,
                    stress_factors=stress_factors
                )
                
                # Store result
                await self._store_stress_test_result(conn, stress_result)
                
                return stress_result
                
            except Exception as e:
                self.logger.error(f"Error running stress test: {e}")
                raise
    
    async def calculate_portfolio_beta(
        self,
        portfolio_id: str,
        benchmark_symbol: str = "SPY",
        lookback_days: int = 252
    ) -> Dict[str, Any]:
        """
        Calculate portfolio beta against benchmark
        """
        async with self.db_pool.acquire() as conn:
            try:
                end_date = datetime.utcnow()
                start_date = end_date - timedelta(days=lookback_days)
                
                # Get portfolio and benchmark returns
                portfolio_returns = await self._get_portfolio_returns(
                    conn, portfolio_id, start_date, end_date
                )
                
                benchmark_returns = await self._get_benchmark_returns(
                    conn, benchmark_symbol, start_date, end_date
                )
                
                if len(portfolio_returns) < 30 or len(benchmark_returns) < 30:
                    raise ValueError("Insufficient data for beta calculation")
                
                # Align return series
                min_length = min(len(portfolio_returns), len(benchmark_returns))
                port_returns = np.array(portfolio_returns[-min_length:])
                bench_returns = np.array(benchmark_returns[-min_length:])
                
                # Calculate beta using linear regression
                covariance = np.cov(port_returns, bench_returns)[0, 1]
                benchmark_variance = np.var(bench_returns)
                
                beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
                
                # Calculate R-squared
                correlation = np.corrcoef(port_returns, bench_returns)[0, 1]
                r_squared = correlation ** 2
                
                # Calculate tracking error
                excess_returns = port_returns - bench_returns
                tracking_error = np.std(excess_returns) * np.sqrt(252)  # Annualized
                
                return {
                    'portfolio_id': portfolio_id,
                    'benchmark': benchmark_symbol,
                    'beta': float(beta),
                    'correlation': float(correlation),
                    'r_squared': float(r_squared),
                    'tracking_error': float(tracking_error),
                    'observations': min_length,
                    'calculation_date': datetime.utcnow().isoformat()
                }
                
            except Exception as e:
                self.logger.error(f"Error calculating portfolio beta: {e}")
                raise
    
    # Helper methods
    
    async def _calculate_historical_var(
        self,
        returns: np.ndarray,
        confidence_level: float,
        time_horizon: int
    ) -> Tuple[float, float, Dict[str, Any]]:
        """Calculate Historical VaR"""
        # Scale returns for time horizon
        scaled_returns = returns * np.sqrt(time_horizon)
        
        # Calculate VaR as percentile
        var_percentile = (1 - confidence_level) * 100
        var_amount = -np.percentile(scaled_returns, var_percentile)
        
        # Calculate Expected Shortfall (CVaR)
        tail_returns = scaled_returns[scaled_returns <= -var_amount]
        expected_shortfall = -np.mean(tail_returns) if len(tail_returns) > 0 else var_amount
        
        params = {
            'method': 'historical',
            'percentile': var_percentile,
            'tail_observations': len(tail_returns)
        }
        
        return var_amount, expected_shortfall, params
    
    async def _calculate_parametric_var(
        self,
        returns: np.ndarray,
        confidence_level: float,
        time_horizon: int
    ) -> Tuple[float, float, Dict[str, Any]]:
        """Calculate Parametric VaR assuming normal distribution"""
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Scale for time horizon
        scaled_mean = mean_return * time_horizon
        scaled_std = std_return * np.sqrt(time_horizon)
        
        # Calculate VaR using normal distribution
        z_score = stats.norm.ppf(1 - confidence_level)
        var_amount = -(scaled_mean + z_score * scaled_std)
        
        # Expected Shortfall for normal distribution
        es_z_score = stats.norm.pdf(z_score) / (1 - confidence_level)
        expected_shortfall = -(scaled_mean - es_z_score * scaled_std)
        
        params = {
            'method': 'parametric',
            'mean': float(scaled_mean),
            'std': float(scaled_std),
            'z_score': float(z_score)
        }
        
        return var_amount, expected_shortfall, params
    
    async def _calculate_monte_carlo_var(
        self,
        returns: np.ndarray,
        confidence_level: float,
        time_horizon: int,
        num_simulations: int = 10000
    ) -> Tuple[float, float, Dict[str, Any]]:
        """Calculate Monte Carlo VaR"""
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Generate random returns
        np.random.seed(42)  # For reproducibility
        simulated_returns = np.random.normal(
            mean_return * time_horizon,
            std_return * np.sqrt(time_horizon),
            num_simulations
        )
        
        # Calculate VaR and ES
        var_percentile = (1 - confidence_level) * 100
        var_amount = -np.percentile(simulated_returns, var_percentile)
        
        tail_returns = simulated_returns[simulated_returns <= -var_amount]
        expected_shortfall = -np.mean(tail_returns) if len(tail_returns) > 0 else var_amount
        
        params = {
            'method': 'monte_carlo',
            'simulations': num_simulations,
            'mean': float(mean_return),
            'std': float(std_return)
        }
        
        return var_amount, expected_shortfall, params
    
    def _calculate_concentration_metrics(self, position_values: List[float]) -> Dict[str, float]:
        """Calculate portfolio concentration metrics"""
        if not position_values:
            return {}
        
        total_value = sum(position_values)
        weights = [v / total_value for v in position_values] if total_value > 0 else []
        
        # Herfindahl-Hirschman Index
        hhi = sum(w ** 2 for w in weights)
        
        # Effective number of positions
        effective_positions = 1 / hhi if hhi > 0 else 0
        
        # Largest position weight
        max_weight = max(weights) if weights else 0
        
        # Top 5 concentration
        sorted_weights = sorted(weights, reverse=True)
        top5_concentration = sum(sorted_weights[:5])
        
        return {
            'herfindahl_index': hhi,
            'effective_positions': effective_positions,
            'max_position_weight': max_weight,
            'top5_concentration': top5_concentration,
            'position_count': len(weights)
        }
    
    async def _get_portfolio_returns(
        self,
        conn: asyncpg.Connection,
        portfolio_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[float]:
        """Get portfolio daily returns"""
        # Implementation depends on your trade/position data structure
        # This is a simplified version
        query = """
            SELECT 
                date_trunc('day', timestamp) as trade_date,
                SUM(realized_pnl) as daily_pnl
            FROM trades
            WHERE portfolio_id = $1
            AND timestamp >= $2
            AND timestamp <= $3
            GROUP BY date_trunc('day', timestamp)
            ORDER BY trade_date
        """
        
        results = await conn.fetch(query, portfolio_id, start_date, end_date)
        
        # Convert to returns (simplified - assumes constant capital base)
        returns = []
        for result in results:
            daily_pnl = float(result['daily_pnl'] or 0)
            returns.append(daily_pnl / 100000)  # Normalize by assumed portfolio size
        
        return returns
    
    async def _get_instrument_returns(
        self,
        conn: asyncpg.Connection,
        instrument_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[float]:
        """Get instrument daily returns"""
        query = """
            SELECT 
                close_price,
                LAG(close_price) OVER (ORDER BY timestamp_ns) as prev_close
            FROM market_bars
            WHERE instrument_id = $1
            AND timestamp_ns >= $2
            AND timestamp_ns <= $3
            AND timeframe = '1d'
            ORDER BY timestamp_ns
        """
        
        start_ns = int(start_date.timestamp() * 1_000_000_000)
        end_ns = int(end_date.timestamp() * 1_000_000_000)
        
        results = await conn.fetch(query, instrument_id, start_ns, end_ns)
        
        returns = []
        for result in results:
            if result['prev_close'] and result['close_price']:
                ret = (float(result['close_price']) - float(result['prev_close'])) / float(result['prev_close'])
                returns.append(ret)
        
        return returns
    
    async def _get_benchmark_returns(
        self,
        conn: asyncpg.Connection,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[float]:
        """Get benchmark returns"""
        # Similar to _get_instrument_returns but for benchmark
        query = """
            SELECT 
                close_price,
                LAG(close_price) OVER (ORDER BY timestamp_ns) as prev_close
            FROM market_bars
            WHERE instrument_id LIKE $1
            AND timestamp_ns >= $2
            AND timestamp_ns <= $3
            AND timeframe = '1d'
            ORDER BY timestamp_ns
        """
        
        start_ns = int(start_date.timestamp() * 1_000_000_000)
        end_ns = int(end_date.timestamp() * 1_000_000_000)
        
        results = await conn.fetch(query, f"%{symbol}%", start_ns, end_ns)
        
        returns = []
        for result in results:
            if result['prev_close'] and result['close_price']:
                ret = (float(result['close_price']) - float(result['prev_close'])) / float(result['prev_close'])
                returns.append(ret)
        
        return returns
    
    async def _get_portfolio_value(self, conn: asyncpg.Connection, portfolio_id: str) -> float:
        """Get current portfolio value"""
        query = """
            SELECT SUM(ABS(quantity * avg_entry_price * multiplier)) as total_value
            FROM positions p
            JOIN instruments i ON p.instrument_id = i.instrument_id
            WHERE p.portfolio_id = $1 AND p.quantity != 0
        """
        
        result = await conn.fetchrow(query, portfolio_id)
        return float(result['total_value'] or 0)
    
    # Storage methods
    
    async def _store_var_result(self, conn: asyncpg.Connection, var_result: VaRResult):
        """Store VaR calculation result"""
        query = """
            INSERT INTO risk_var_calculations (
                portfolio_id, method, confidence_level, time_horizon,
                var_amount, expected_shortfall, calculation_timestamp,
                observations_used, model_parameters
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        """
        
        await conn.execute(
            query,
            var_result.portfolio_id,
            var_result.method.value,
            var_result.confidence_level,
            var_result.time_horizon,
            float(var_result.var_amount),
            float(var_result.expected_shortfall),
            var_result.calculation_timestamp,
            var_result.observations_used,
            json.dumps(var_result.model_parameters) if var_result.model_parameters else None
        )
    
    async def _store_exposure_analysis(self, conn: asyncpg.Connection, exposure: ExposureAnalysis):
        """Store exposure analysis result"""
        query = """
            INSERT INTO risk_exposure_analysis (
                portfolio_id, timestamp, total_exposure, net_exposure,
                gross_exposure, long_exposure, short_exposure,
                exposure_by_asset_class, exposure_by_sector, exposure_by_currency,
                concentration_metrics
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        """
        
        await conn.execute(
            query,
            exposure.portfolio_id,
            exposure.timestamp,
            float(exposure.total_exposure),
            float(exposure.net_exposure),
            float(exposure.gross_exposure),
            float(exposure.long_exposure),
            float(exposure.short_exposure),
            json.dumps(exposure.exposure_by_asset_class),
            json.dumps(exposure.exposure_by_sector),
            json.dumps(exposure.exposure_by_currency),
            json.dumps(exposure.concentration_metrics)
        )
    
    async def _store_correlation_analysis(self, conn: asyncpg.Connection, correlation: CorrelationAnalysis):
        """Store correlation analysis result"""
        query = """
            INSERT INTO risk_correlation_analysis (
                portfolio_id, timestamp, correlation_matrix, asset_ids,
                average_correlation, max_correlation, min_correlation,
                eigenvalues, diversification_ratio
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        """
        
        await conn.execute(
            query,
            correlation.portfolio_id,
            correlation.timestamp,
            json.dumps(correlation.correlation_matrix.tolist()),
            json.dumps(correlation.asset_ids),
            correlation.average_correlation,
            correlation.max_correlation,
            correlation.min_correlation,
            json.dumps(correlation.eigenvalues),
            correlation.diversification_ratio
        )
    
    async def _store_stress_test_result(self, conn: asyncpg.Connection, stress_result: StressTestResult):
        """Store stress test result"""
        query = """
            INSERT INTO risk_stress_tests (
                portfolio_id, scenario, scenario_name, portfolio_impact,
                impact_percentage, positions_affected, worst_position_impact,
                var_breach_probability, recovery_time_estimate, stress_factors,
                test_timestamp
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        """
        
        await conn.execute(
            query,
            stress_result.portfolio_id,
            stress_result.scenario.value,
            stress_result.scenario_name,
            float(stress_result.portfolio_impact),
            stress_result.impact_percentage,
            stress_result.positions_affected,
            float(stress_result.worst_position_impact),
            stress_result.var_breach_probability,
            stress_result.recovery_time_estimate,
            json.dumps(stress_result.stress_factors),
            datetime.utcnow()
        )

# Global instance
risk_analytics = None

def get_risk_analytics() -> RiskAnalytics:
    """Get global risk analytics instance"""
    global risk_analytics
    if risk_analytics is None:
        raise RuntimeError("Risk analytics not initialized. Call init_risk_analytics() first.")
    return risk_analytics

def init_risk_analytics(db_pool: asyncpg.Pool) -> RiskAnalytics:
    """Initialize global risk analytics instance"""
    global risk_analytics
    risk_analytics = RiskAnalytics(db_pool)
    return risk_analytics