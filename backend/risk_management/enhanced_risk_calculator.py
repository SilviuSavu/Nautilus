"""
Enhanced Risk Calculator with Advanced Portfolio-Level Metrics
Scenario analysis, stress testing, and Monte Carlo simulations
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from decimal import Decimal
import numpy as np
import pandas as pd
from scipy import stats, optimize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import json

# Import base risk calculator
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from risk_calculator import RiskCalculator, VaRCalculator, CorrelationCalculator, RiskMetricsCalculator

logger = logging.getLogger(__name__)

class ScenarioAnalysis:
    """Advanced scenario analysis and stress testing"""
    
    def __init__(self):
        self.predefined_scenarios = {
            'market_crash_2008': {
                'name': '2008 Financial Crisis',
                'shocks': {
                    'equity_indices': -0.40,  # 40% drop
                    'volatility': 2.5,        # 250% increase
                    'credit_spreads': 0.05,   # 500bp widening
                    'yield_curve': -0.02      # 200bp drop
                }
            },
            'covid_2020': {
                'name': 'COVID-19 Pandemic',
                'shocks': {
                    'equity_indices': -0.35,
                    'volatility': 2.0,
                    'credit_spreads': 0.03,
                    'oil_price': -0.60
                }
            },
            'interest_rate_shock': {
                'name': 'Interest Rate Shock',
                'shocks': {
                    'yield_curve': 0.02,      # 200bp increase
                    'duration_risk': 1.5,    # 50% increase
                    'credit_spreads': 0.01    # 100bp widening
                }
            },
            'currency_crisis': {
                'name': 'Currency Crisis',
                'shocks': {
                    'fx_rates': {'USD': 0.0, 'EUR': -0.15, 'GBP': -0.20, 'JPY': 0.10},
                    'emerging_markets': -0.25,
                    'commodities': -0.15
                }
            }
        }
    
    async def run_scenario_analysis(
        self, 
        portfolio_returns: np.ndarray,
        scenario_name: str,
        custom_shocks: Optional[Dict[str, float]] = None,
        monte_carlo_runs: int = 10000
    ) -> Dict[str, Any]:
        """Run comprehensive scenario analysis"""
        try:
            # Get scenario definition
            if scenario_name in self.predefined_scenarios:
                scenario = self.predefined_scenarios[scenario_name]
            else:
                scenario = {'name': scenario_name, 'shocks': custom_shocks or {}}
            
            # Apply shocks to portfolio returns
            shocked_returns = await self._apply_scenario_shocks(portfolio_returns, scenario['shocks'])
            
            # Calculate scenario metrics
            scenario_metrics = await self._calculate_scenario_metrics(
                portfolio_returns, shocked_returns, monte_carlo_runs
            )
            
            result = {
                'scenario_name': scenario['name'],
                'scenario_shocks': scenario['shocks'],
                'scenario_metrics': scenario_metrics,
                'monte_carlo_runs': monte_carlo_runs,
                'calculated_at': datetime.utcnow().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in scenario analysis: {e}")
            raise
    
    async def _apply_scenario_shocks(
        self, 
        portfolio_returns: np.ndarray, 
        shocks: Dict[str, Any]
    ) -> np.ndarray:
        """Apply scenario shocks to portfolio returns"""
        try:
            shocked_returns = portfolio_returns.copy()
            
            # Apply equity market shock
            if 'equity_indices' in shocks:
                equity_shock = shocks['equity_indices']
                # Simulate correlation with equity markets (assume 70% correlation)
                correlation = 0.7
                shocked_returns = shocked_returns + (equity_shock * correlation)
            
            # Apply volatility shock
            if 'volatility' in shocks:
                vol_multiplier = shocks['volatility']
                current_vol = np.std(portfolio_returns)
                shocked_vol = current_vol * vol_multiplier
                vol_ratio = shocked_vol / current_vol
                shocked_returns = shocked_returns * vol_ratio
            
            # Apply credit spread shock (affects portfolio beta)
            if 'credit_spreads' in shocks:
                credit_shock = shocks['credit_spreads']
                # Assume negative correlation with credit spreads
                shocked_returns = shocked_returns - (credit_shock * 0.5)
            
            return shocked_returns
            
        except Exception as e:
            logger.error(f"Error applying scenario shocks: {e}")
            return portfolio_returns
    
    async def _calculate_scenario_metrics(
        self,
        baseline_returns: np.ndarray,
        shocked_returns: np.ndarray,
        monte_carlo_runs: int
    ) -> Dict[str, Any]:
        """Calculate comprehensive scenario metrics"""
        try:
            # Basic statistics
            baseline_mean = np.mean(baseline_returns)
            shocked_mean = np.mean(shocked_returns)
            
            baseline_vol = np.std(baseline_returns, ddof=1)
            shocked_vol = np.std(shocked_returns, ddof=1)
            
            # Calculate percentile impacts
            percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            baseline_percentiles = np.percentile(baseline_returns, percentiles)
            shocked_percentiles = np.percentile(shocked_returns, percentiles)
            
            # Monte Carlo simulation for tail risks
            mc_results = await self._monte_carlo_scenario_simulation(
                shocked_returns, monte_carlo_runs
            )
            
            return {
                'return_impact': {
                    'baseline_mean': float(baseline_mean),
                    'shocked_mean': float(shocked_mean),
                    'return_change': float(shocked_mean - baseline_mean),
                    'return_change_pct': float((shocked_mean - baseline_mean) / abs(baseline_mean) * 100) if baseline_mean != 0 else 0
                },
                'volatility_impact': {
                    'baseline_vol': float(baseline_vol),
                    'shocked_vol': float(shocked_vol),
                    'vol_change': float(shocked_vol - baseline_vol),
                    'vol_change_pct': float((shocked_vol - baseline_vol) / baseline_vol * 100) if baseline_vol != 0 else 0
                },
                'percentile_impacts': {
                    f'p{p}': {
                        'baseline': float(baseline_percentiles[i]),
                        'shocked': float(shocked_percentiles[i]),
                        'change': float(shocked_percentiles[i] - baseline_percentiles[i])
                    }
                    for i, p in enumerate(percentiles)
                },
                'monte_carlo_results': mc_results,
                'tail_risk_metrics': {
                    'expected_shortfall_1pct': float(np.mean(shocked_returns[shocked_returns <= np.percentile(shocked_returns, 1)])),
                    'maximum_loss': float(np.min(shocked_returns)),
                    'tail_expectation': float(np.mean(shocked_returns[shocked_returns <= np.percentile(shocked_returns, 5)]))
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating scenario metrics: {e}")
            return {}
    
    async def _monte_carlo_scenario_simulation(
        self, 
        shocked_returns: np.ndarray, 
        num_runs: int
    ) -> Dict[str, Any]:
        """Run Monte Carlo simulation on shocked returns"""
        try:
            # Fit distribution to shocked returns
            mean_return = np.mean(shocked_returns)
            std_return = np.std(shocked_returns, ddof=1)
            
            # Generate Monte Carlo scenarios
            np.random.seed(42)  # For reproducibility
            mc_returns = np.random.normal(mean_return, std_return, num_runs)
            
            # Calculate statistics
            mc_percentiles = np.percentile(mc_returns, [1, 5, 10, 25, 50, 75, 90, 95, 99])
            
            return {
                'mean': float(np.mean(mc_returns)),
                'std': float(np.std(mc_returns)),
                'skewness': float(stats.skew(mc_returns)),
                'kurtosis': float(stats.kurtosis(mc_returns)),
                'percentiles': {
                    'p1': float(mc_percentiles[0]),
                    'p5': float(mc_percentiles[1]),
                    'p10': float(mc_percentiles[2]),
                    'p25': float(mc_percentiles[3]),
                    'p50': float(mc_percentiles[4]),
                    'p75': float(mc_percentiles[5]),
                    'p90': float(mc_percentiles[6]),
                    'p95': float(mc_percentiles[7]),
                    'p99': float(mc_percentiles[8])
                },
                'var_95': float(np.percentile(mc_returns, 5)),
                'var_99': float(np.percentile(mc_returns, 1)),
                'expected_shortfall_95': float(np.mean(mc_returns[mc_returns <= np.percentile(mc_returns, 5)])),
                'expected_shortfall_99': float(np.mean(mc_returns[mc_returns <= np.percentile(mc_returns, 1)]))
            }
            
        except Exception as e:
            logger.error(f"Error in Monte Carlo simulation: {e}")
            return {}

class AdvancedRiskMetrics:
    """Advanced portfolio-level risk metrics and analytics"""
    
    def __init__(self):
        self.base_calculator = RiskCalculator()
    
    async def calculate_portfolio_factor_exposures(
        self, 
        returns_data: Dict[str, np.ndarray],
        factor_returns: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """Calculate portfolio exposures to risk factors"""
        try:
            # Convert to DataFrames
            portfolio_df = pd.DataFrame(returns_data)
            factor_df = pd.DataFrame(factor_returns)
            
            # Align data
            common_dates = portfolio_df.index.intersection(factor_df.index)
            portfolio_aligned = portfolio_df.loc[common_dates]
            factor_aligned = factor_df.loc[common_dates]
            
            factor_exposures = {}
            
            for asset in portfolio_aligned.columns:
                asset_returns = portfolio_aligned[asset].dropna()
                
                # Run factor regression for each asset
                exposures = {}
                for factor_name in factor_aligned.columns:
                    factor_data = factor_aligned[factor_name].loc[asset_returns.index]
                    
                    if len(asset_returns) > 10 and len(factor_data) > 10:
                        # Linear regression
                        slope, intercept, r_value, p_value, std_err = stats.linregress(
                            factor_data.values, asset_returns.values
                        )
                        
                        exposures[factor_name] = {
                            'beta': float(slope),
                            'alpha': float(intercept),
                            'r_squared': float(r_value ** 2),
                            'p_value': float(p_value),
                            'std_error': float(std_err)
                        }
                
                factor_exposures[asset] = exposures
            
            return {
                'factor_exposures': factor_exposures,
                'analysis_period': f"{common_dates[0]} to {common_dates[-1]}",
                'calculated_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating factor exposures: {e}")
            return {}
    
    async def calculate_risk_attribution(
        self, 
        returns_data: Dict[str, np.ndarray],
        portfolio_weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """Calculate risk attribution across portfolio components"""
        try:
            # Convert to DataFrame
            df = pd.DataFrame(returns_data)
            
            # Calculate covariance matrix
            cov_matrix = df.cov().values
            
            # Convert weights to array
            assets = list(returns_data.keys())
            weights = np.array([portfolio_weights.get(asset, 0) for asset in assets])
            
            # Portfolio variance
            portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))
            portfolio_vol = np.sqrt(portfolio_var)
            
            # Marginal risk contributions
            marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
            
            # Component risk contributions
            component_contrib = weights * marginal_contrib
            
            # Percentage contributions
            contrib_pct = component_contrib / portfolio_var * 100
            
            attribution_results = {}
            for i, asset in enumerate(assets):
                attribution_results[asset] = {
                    'weight': float(weights[i]),
                    'marginal_risk': float(marginal_contrib[i]),
                    'component_risk': float(component_contrib[i]),
                    'risk_contribution_pct': float(contrib_pct[i]),
                    'individual_vol': float(np.sqrt(cov_matrix[i, i]))
                }
            
            return {
                'portfolio_volatility': float(portfolio_vol),
                'portfolio_variance': float(portfolio_var),
                'risk_attribution': attribution_results,
                'diversification_ratio': float(
                    sum(weights[i] * np.sqrt(cov_matrix[i, i]) for i in range(len(weights))) / portfolio_vol
                ),
                'calculated_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk attribution: {e}")
            return {}
    
    async def calculate_liquidity_risk(
        self, 
        positions: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate portfolio liquidity risk metrics"""
        try:
            liquidity_metrics = {}
            total_position_value = 0
            weighted_liquidity_score = 0
            
            for symbol, position in positions.items():
                market_value = float(position.get('market_value', 0))
                total_position_value += abs(market_value)
                
                # Mock liquidity scoring (would use real market data)
                daily_volume = position.get('avg_daily_volume', 1000000)  # Mock
                position_size = abs(market_value)
                
                # Days to liquidate (simplified)
                days_to_liquidate = position_size / (daily_volume * 0.1)  # Assume 10% of volume
                
                # Bid-ask spread impact
                bid_ask_spread = position.get('bid_ask_spread', 0.001)  # Mock 10bp
                spread_cost = market_value * bid_ask_spread
                
                # Liquidity score (0-100, higher is more liquid)
                liquidity_score = max(0, 100 - (days_to_liquidate * 10) - (bid_ask_spread * 10000))
                
                liquidity_metrics[symbol] = {
                    'market_value': market_value,
                    'days_to_liquidate': min(days_to_liquidate, 30),  # Cap at 30 days
                    'spread_cost': spread_cost,
                    'liquidity_score': liquidity_score
                }
                
                weighted_liquidity_score += liquidity_score * abs(market_value)
            
            # Portfolio-level metrics
            if total_position_value > 0:
                weighted_liquidity_score /= total_position_value
            
            # Calculate concentration in illiquid positions
            illiquid_value = sum(
                abs(metrics['market_value']) 
                for metrics in liquidity_metrics.values() 
                if metrics['liquidity_score'] < 50
            )
            
            illiquid_concentration = illiquid_value / total_position_value if total_position_value > 0 else 0
            
            return {
                'individual_positions': liquidity_metrics,
                'portfolio_liquidity_score': float(weighted_liquidity_score),
                'illiquid_concentration_pct': float(illiquid_concentration * 100),
                'total_spread_cost': float(sum(m['spread_cost'] for m in liquidity_metrics.values())),
                'avg_days_to_liquidate': float(
                    np.average(
                        [m['days_to_liquidate'] for m in liquidity_metrics.values()],
                        weights=[abs(m['market_value']) for m in liquidity_metrics.values()]
                    ) if liquidity_metrics else 0
                ),
                'calculated_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating liquidity risk: {e}")
            return {}
    
    async def calculate_concentration_metrics(
        self, 
        positions: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate comprehensive concentration risk metrics"""
        try:
            if not positions:
                return {}
            
            # Extract position values
            position_values = []
            total_value = 0
            
            for symbol, position in positions.items():
                value = abs(float(position.get('market_value', 0)))
                position_values.append(value)
                total_value += value
            
            if total_value == 0:
                return {}
            
            # Calculate weights
            weights = np.array(position_values) / total_value
            
            # Herfindahl-Hirschman Index
            hhi = np.sum(weights ** 2)
            
            # Effective number of positions
            effective_positions = 1 / hhi if hhi > 0 else 0
            
            # Top N concentration
            sorted_weights = np.sort(weights)[::-1]  # Descending order
            
            concentration_metrics = {
                'herfindahl_index': float(hhi),
                'effective_num_positions': float(effective_positions),
                'largest_position_pct': float(sorted_weights[0] * 100) if len(sorted_weights) > 0 else 0,
                'top_3_concentration_pct': float(np.sum(sorted_weights[:3]) * 100) if len(sorted_weights) >= 3 else float(np.sum(sorted_weights) * 100),
                'top_5_concentration_pct': float(np.sum(sorted_weights[:5]) * 100) if len(sorted_weights) >= 5 else float(np.sum(sorted_weights) * 100),
                'top_10_concentration_pct': float(np.sum(sorted_weights[:10]) * 100) if len(sorted_weights) >= 10 else float(np.sum(sorted_weights) * 100),
                'gini_coefficient': float(self._calculate_gini_coefficient(weights)),
                'entropy': float(-np.sum(weights * np.log(weights + 1e-10))),  # Add small value to avoid log(0)
                'total_positions': len(positions),
                'calculated_at': datetime.utcnow().isoformat()
            }
            
            return concentration_metrics
            
        except Exception as e:
            logger.error(f"Error calculating concentration metrics: {e}")
            return {}
    
    def _calculate_gini_coefficient(self, weights: np.ndarray) -> float:
        """Calculate Gini coefficient for concentration measurement"""
        try:
            # Sort weights
            sorted_weights = np.sort(weights)
            n = len(sorted_weights)
            
            if n == 0:
                return 0
            
            # Calculate Gini coefficient
            cumsum = np.cumsum(sorted_weights)
            gini = (2 * np.sum((np.arange(1, n + 1) * sorted_weights))) / (n * cumsum[-1]) - (n + 1) / n
            
            return max(0, min(1, gini))  # Ensure between 0 and 1
            
        except Exception as e:
            logger.error(f"Error calculating Gini coefficient: {e}")
            return 0

class EnhancedRiskCalculator:
    """
    Enhanced risk calculator integrating advanced portfolio analytics,
    scenario analysis, and comprehensive risk metrics
    """
    
    def __init__(self):
        # Base calculators
        self.base_calculator = RiskCalculator()
        self.var_calculator = VaRCalculator()
        self.corr_calculator = CorrelationCalculator()
        self.metrics_calculator = RiskMetricsCalculator()
        
        # Advanced modules
        self.scenario_analysis = ScenarioAnalysis()
        self.advanced_metrics = AdvancedRiskMetrics()
        
        # Configuration
        self.calculation_cache = {}
        self.cache_ttl_minutes = 5
    
    async def comprehensive_portfolio_analysis(
        self,
        portfolio_id: str,
        returns_data: Dict[str, np.ndarray],
        positions: Dict[str, Dict[str, Any]],
        portfolio_weights: Optional[Dict[str, float]] = None,
        benchmark_returns: Optional[np.ndarray] = None,
        factor_returns: Optional[Dict[str, np.ndarray]] = None,
        analysis_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive portfolio risk analysis including:
        - Traditional risk metrics (VaR, volatility, etc.)
        - Advanced analytics (factor exposures, risk attribution)
        - Scenario analysis and stress testing
        - Concentration and liquidity risk
        """
        try:
            logger.info(f"Starting comprehensive portfolio analysis for {portfolio_id}")
            
            analysis_config = analysis_config or {}
            
            # Check cache
            cache_key = f"comprehensive_{portfolio_id}_{hash(str(returns_data.keys()))}"
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                return cached_result
            
            result = {
                'portfolio_id': portfolio_id,
                'analysis_timestamp': datetime.utcnow().isoformat(),
                'traditional_metrics': {},
                'advanced_analytics': {},
                'scenario_analysis': {},
                'risk_attribution': {},
                'concentration_risk': {},
                'liquidity_risk': {},
                'factor_exposures': {},
                'summary': {}
            }
            
            # 1. Traditional Risk Metrics
            logger.info("Calculating traditional risk metrics")
            traditional_metrics = await self.base_calculator.comprehensive_risk_analysis(
                returns_data=returns_data,
                portfolio_weights=portfolio_weights,
                benchmark_returns=benchmark_returns
            )
            result['traditional_metrics'] = traditional_metrics
            
            # 2. Risk Attribution
            if portfolio_weights:
                logger.info("Calculating risk attribution")
                attribution = await self.advanced_metrics.calculate_risk_attribution(
                    returns_data, portfolio_weights
                )
                result['risk_attribution'] = attribution
            
            # 3. Concentration Risk
            logger.info("Calculating concentration risk")
            concentration = await self.advanced_metrics.calculate_concentration_metrics(positions)
            result['concentration_risk'] = concentration
            
            # 4. Liquidity Risk
            logger.info("Calculating liquidity risk")
            liquidity = await self.advanced_metrics.calculate_liquidity_risk(positions)
            result['liquidity_risk'] = liquidity
            
            # 5. Factor Exposures
            if factor_returns:
                logger.info("Calculating factor exposures")
                exposures = await self.advanced_metrics.calculate_portfolio_factor_exposures(
                    returns_data, factor_returns
                )
                result['factor_exposures'] = exposures
            
            # 6. Scenario Analysis
            if analysis_config.get('include_scenarios', True):
                logger.info("Running scenario analysis")
                portfolio_returns = await self._calculate_portfolio_returns(returns_data, portfolio_weights)
                
                scenarios = analysis_config.get('scenarios', ['market_crash_2008', 'interest_rate_shock'])
                scenario_results = {}
                
                for scenario_name in scenarios:
                    try:
                        scenario_result = await self.scenario_analysis.run_scenario_analysis(
                            portfolio_returns, scenario_name
                        )
                        scenario_results[scenario_name] = scenario_result
                    except Exception as e:
                        logger.error(f"Error in scenario {scenario_name}: {e}")
                        scenario_results[scenario_name] = {'error': str(e)}
                
                result['scenario_analysis'] = scenario_results
            
            # 7. Advanced Analytics Summary
            result['advanced_analytics'] = await self._calculate_advanced_analytics_summary(
                result, returns_data, positions
            )
            
            # 8. Overall Summary
            result['summary'] = await self._generate_analysis_summary(result)
            
            # Cache result
            self._cache_result(cache_key, result)
            
            logger.info(f"Completed comprehensive portfolio analysis for {portfolio_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error in comprehensive portfolio analysis: {e}")
            raise
    
    async def _calculate_portfolio_returns(
        self, 
        returns_data: Dict[str, np.ndarray],
        weights: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """Calculate portfolio returns from individual asset returns"""
        try:
            df = pd.DataFrame(returns_data)
            
            if weights is None:
                # Equal weights
                weights = {symbol: 1.0/len(returns_data) for symbol in returns_data.keys()}
            
            # Calculate weighted returns
            portfolio_returns = np.zeros(len(df))
            for symbol, weight in weights.items():
                if symbol in df.columns:
                    portfolio_returns += df[symbol].fillna(0) * weight
            
            return portfolio_returns
            
        except Exception as e:
            logger.error(f"Error calculating portfolio returns: {e}")
            return np.array([])
    
    async def _calculate_advanced_analytics_summary(
        self,
        analysis_result: Dict[str, Any],
        returns_data: Dict[str, np.ndarray],
        positions: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate summary of advanced analytics"""
        try:
            summary = {
                'risk_score': 0,
                'diversification_score': 0,
                'liquidity_score': 0,
                'concentration_score': 0,
                'overall_rating': 'unknown'
            }
            
            # Risk Score (0-100, lower is better)
            traditional = analysis_result.get('traditional_metrics', {})
            var_metrics = traditional.get('var_metrics', {})
            
            var_1d_95 = float(var_metrics.get('var_1d_95_historical', 0))
            portfolio_vol = float(traditional.get('portfolio_metrics', {}).get('volatility', 0))
            
            # Normalize risk score (simplified)
            risk_score = min(100, (var_1d_95 / 1000) * 10 + (portfolio_vol * 100))
            summary['risk_score'] = risk_score
            
            # Diversification Score
            attribution = analysis_result.get('risk_attribution', {})
            diversification_ratio = attribution.get('diversification_ratio', 1.0)
            diversification_score = max(0, min(100, (2 - diversification_ratio) * 100))
            summary['diversification_score'] = diversification_score
            
            # Liquidity Score
            liquidity = analysis_result.get('liquidity_risk', {})
            liquidity_score = liquidity.get('portfolio_liquidity_score', 50)
            summary['liquidity_score'] = liquidity_score
            
            # Concentration Score
            concentration = analysis_result.get('concentration_risk', {})
            hhi = concentration.get('herfindahl_index', 0.5)
            concentration_score = max(0, min(100, (1 - hhi) * 100))
            summary['concentration_score'] = concentration_score
            
            # Overall Rating
            overall_score = (diversification_score + liquidity_score + concentration_score) / 3 - risk_score / 4
            
            if overall_score >= 70:
                summary['overall_rating'] = 'excellent'
            elif overall_score >= 50:
                summary['overall_rating'] = 'good'
            elif overall_score >= 30:
                summary['overall_rating'] = 'fair'
            else:
                summary['overall_rating'] = 'poor'
            
            return summary
            
        except Exception as e:
            logger.error(f"Error calculating advanced analytics summary: {e}")
            return {}
    
    async def _generate_analysis_summary(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary of the analysis"""
        try:
            traditional = analysis_result.get('traditional_metrics', {})
            advanced = analysis_result.get('advanced_analytics', {})
            concentration = analysis_result.get('concentration_risk', {})
            liquidity = analysis_result.get('liquidity_risk', {})
            
            summary = {
                'key_metrics': {
                    'var_1d_95': traditional.get('var_metrics', {}).get('var_1d_95_historical', 0),
                    'portfolio_volatility': traditional.get('portfolio_metrics', {}).get('volatility', 0),
                    'sharpe_ratio': traditional.get('portfolio_metrics', {}).get('sharpe_ratio', 0),
                    'max_drawdown': traditional.get('portfolio_metrics', {}).get('max_drawdown', 0),
                    'concentration_hhi': concentration.get('herfindahl_index', 0),
                    'liquidity_score': liquidity.get('portfolio_liquidity_score', 0)
                },
                'risk_assessment': {
                    'overall_rating': advanced.get('overall_rating', 'unknown'),
                    'risk_score': advanced.get('risk_score', 0),
                    'diversification_score': advanced.get('diversification_score', 0),
                    'concentration_score': advanced.get('concentration_score', 0),
                    'liquidity_score': advanced.get('liquidity_score', 0)
                },
                'alerts': [],
                'recommendations': []
            }
            
            # Generate alerts based on thresholds
            alerts = []
            recommendations = []
            
            # High concentration alert
            if concentration.get('largest_position_pct', 0) > 25:
                alerts.append({
                    'severity': 'warning',
                    'message': f"High concentration risk: Largest position is {concentration.get('largest_position_pct', 0):.1f}% of portfolio"
                })
                recommendations.append("Consider reducing position sizes to improve diversification")
            
            # Low liquidity alert
            if liquidity.get('illiquid_concentration_pct', 0) > 20:
                alerts.append({
                    'severity': 'warning',
                    'message': f"High illiquid concentration: {liquidity.get('illiquid_concentration_pct', 0):.1f}% in illiquid positions"
                })
                recommendations.append("Consider increasing allocation to more liquid assets")
            
            # High VaR alert
            var_1d_95 = float(traditional.get('var_metrics', {}).get('var_1d_95_historical', 0))
            if var_1d_95 > 10000:  # Threshold example
                alerts.append({
                    'severity': 'critical',
                    'message': f"High VaR: Daily 95% VaR is ${var_1d_95:,.0f}"
                })
                recommendations.append("Consider reducing portfolio risk through position sizing or hedging")
            
            summary['alerts'] = alerts
            summary['recommendations'] = recommendations
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating analysis summary: {e}")
            return {}
    
    def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached calculation result if still valid"""
        try:
            if cache_key in self.calculation_cache:
                cached_entry = self.calculation_cache[cache_key]
                cache_time = cached_entry['timestamp']
                
                cache_age_minutes = (datetime.utcnow() - cache_time).total_seconds() / 60
                if cache_age_minutes < self.cache_ttl_minutes:
                    logger.debug(f"Using cached result: {cache_key}")
                    return cached_entry['result']
                else:
                    # Cache expired
                    del self.calculation_cache[cache_key]
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking calculation cache: {e}")
            return None
    
    def _cache_result(self, cache_key: str, result: Dict[str, Any]):
        """Cache calculation result"""
        try:
            self.calculation_cache[cache_key] = {
                'result': result,
                'timestamp': datetime.utcnow()
            }
            
            # Cleanup old cache entries
            if len(self.calculation_cache) > 50:  # Keep max 50 cached results
                oldest_key = min(self.calculation_cache.keys(), 
                               key=lambda k: self.calculation_cache[k]['timestamp'])
                del self.calculation_cache[oldest_key]
                
        except Exception as e:
            logger.error(f"Error caching result: {e}")
    
    async def run_custom_scenario(
        self,
        portfolio_returns: np.ndarray,
        scenario_name: str,
        custom_shocks: Dict[str, float],
        monte_carlo_runs: int = 10000
    ) -> Dict[str, Any]:
        """Run custom scenario analysis"""
        return await self.scenario_analysis.run_scenario_analysis(
            portfolio_returns, scenario_name, custom_shocks, monte_carlo_runs
        )
    
    async def calculate_marginal_var(
        self,
        returns_data: Dict[str, np.ndarray],
        portfolio_weights: Dict[str, float],
        confidence_level: float = 0.95
    ) -> Dict[str, float]:
        """Calculate marginal VaR for each position"""
        try:
            base_portfolio_returns = await self._calculate_portfolio_returns(returns_data, portfolio_weights)
            base_var = await self.var_calculator.historical_var(base_portfolio_returns, confidence_level)
            
            marginal_vars = {}
            
            for asset in returns_data.keys():
                # Create modified weights (small increase in asset weight)
                delta = 0.01  # 1% increase
                modified_weights = portfolio_weights.copy()
                
                if asset in modified_weights:
                    modified_weights[asset] += delta
                else:
                    modified_weights[asset] = delta
                
                # Normalize weights
                total_weight = sum(modified_weights.values())
                modified_weights = {k: v/total_weight for k, v in modified_weights.items()}
                
                # Calculate new portfolio VaR
                modified_returns = await self._calculate_portfolio_returns(returns_data, modified_weights)
                modified_var = await self.var_calculator.historical_var(modified_returns, confidence_level)
                
                # Marginal VaR
                marginal_vars[asset] = (modified_var - base_var) / delta
            
            return marginal_vars
            
        except Exception as e:
            logger.error(f"Error calculating marginal VaR: {e}")
            return {}
    
    async def get_calculation_status(self) -> Dict[str, Any]:
        """Get status of the enhanced risk calculator"""
        return {
            'cached_calculations': len(self.calculation_cache),
            'cache_ttl_minutes': self.cache_ttl_minutes,
            'available_scenarios': list(self.scenario_analysis.predefined_scenarios.keys()),
            'base_calculator_available': self.base_calculator is not None,
            'advanced_metrics_available': self.advanced_metrics is not None,
            'scenario_analysis_available': self.scenario_analysis is not None
        }

# Global instance
enhanced_risk_calculator = EnhancedRiskCalculator()