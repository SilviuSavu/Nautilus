"""
Advanced Risk Calculation Engine for Portfolio Risk Management
Implements various VaR methodologies, correlation analysis, and risk metrics
Enhanced integration with Sprint 3 risk management system
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from decimal import Decimal
from datetime import datetime, timedelta
import logging
from scipy import stats
from sklearn.covariance import LedoitWolf
import asyncio

logger = logging.getLogger(__name__)

# Integration with enhanced risk management
try:
    from risk_management.enhanced_risk_calculator import enhanced_risk_calculator
    from risk_management.risk_monitor import risk_monitor
    ENHANCED_INTEGRATION_AVAILABLE = True
except ImportError:
    logger.warning("Enhanced risk management integration not available")
    ENHANCED_INTEGRATION_AVAILABLE = False

class VaRCalculator:
    """Value at Risk calculation methods"""
    
    def __init__(self):
        self.supported_methods = ['historical', 'parametric', 'monte_carlo']
        
    async def historical_var(
        self, 
        returns: np.ndarray, 
        confidence_level: float = 0.95,
        time_horizon: int = 1
    ) -> float:
        """
        Calculate Historical Value at Risk
        
        Args:
            returns: Array of historical returns
            confidence_level: Confidence level (0.90, 0.95, 0.99)
            time_horizon: Time horizon in days
            
        Returns:
            VaR value as positive number
        """
        try:
            if len(returns) < 30:
                logger.warning(f"Limited data for VaR calculation: {len(returns)} observations")
                
            # Remove any NaN or infinite values
            clean_returns = returns[np.isfinite(returns)]
            
            if len(clean_returns) == 0:
                raise ValueError("No valid returns data for VaR calculation")
            
            # Scale returns to time horizon if needed
            if time_horizon > 1:
                scaled_returns = clean_returns * np.sqrt(time_horizon)
            else:
                scaled_returns = clean_returns
            
            # Calculate percentile for VaR
            var_percentile = (1 - confidence_level) * 100
            var_value = np.percentile(scaled_returns, var_percentile)
            
            # Return as positive value (loss)
            return float(abs(var_value))
            
        except Exception as e:
            logger.error(f"Error calculating historical VaR: {e}")
            raise
    
    async def parametric_var(
        self, 
        returns: np.ndarray, 
        confidence_level: float = 0.95,
        time_horizon: int = 1,
        distribution: str = 'normal'
    ) -> float:
        """
        Calculate Parametric Value at Risk assuming normal distribution
        
        Args:
            returns: Array of historical returns
            confidence_level: Confidence level
            time_horizon: Time horizon in days
            distribution: Distribution assumption ('normal', 't_student')
            
        Returns:
            VaR value as positive number
        """
        try:
            clean_returns = returns[np.isfinite(returns)]
            
            if len(clean_returns) < 10:
                raise ValueError("Insufficient data for parametric VaR")
            
            # Calculate statistics
            mean_return = np.mean(clean_returns)
            std_return = np.std(clean_returns, ddof=1)  # Sample standard deviation
            
            # Scale for time horizon
            if time_horizon > 1:
                mean_return_scaled = mean_return * time_horizon
                std_return_scaled = std_return * np.sqrt(time_horizon)
            else:
                mean_return_scaled = mean_return
                std_return_scaled = std_return
            
            # Calculate VaR based on distribution
            if distribution == 'normal':
                z_score = stats.norm.ppf(1 - confidence_level)
            elif distribution == 't_student':
                # Use t-distribution with degrees of freedom = n-1
                df = len(clean_returns) - 1
                z_score = stats.t.ppf(1 - confidence_level, df)
            else:
                raise ValueError(f"Unsupported distribution: {distribution}")
            
            # Calculate VaR (negative of the left tail)
            var_value = -(mean_return_scaled + z_score * std_return_scaled)
            
            return float(max(0, var_value))  # Ensure non-negative
            
        except Exception as e:
            logger.error(f"Error calculating parametric VaR: {e}")
            raise
    
    async def monte_carlo_var(
        self, 
        returns: np.ndarray, 
        confidence_level: float = 0.95,
        time_horizon: int = 1,
        num_simulations: int = 10000,
        random_seed: Optional[int] = None
    ) -> float:
        """
        Calculate Monte Carlo Value at Risk
        
        Args:
            returns: Array of historical returns
            confidence_level: Confidence level
            time_horizon: Time horizon in days
            num_simulations: Number of Monte Carlo simulations
            random_seed: Random seed for reproducibility
            
        Returns:
            VaR value as positive number
        """
        try:
            if random_seed:
                np.random.seed(random_seed)
                
            clean_returns = returns[np.isfinite(returns)]
            
            if len(clean_returns) < 10:
                raise ValueError("Insufficient data for Monte Carlo VaR")
            
            # Fit distribution parameters
            mean_return = np.mean(clean_returns)
            std_return = np.std(clean_returns, ddof=1)
            
            # Generate random scenarios
            simulated_returns = np.random.normal(
                loc=mean_return, 
                scale=std_return, 
                size=(num_simulations, time_horizon)
            )
            
            # Calculate cumulative returns for each simulation
            if time_horizon == 1:
                portfolio_returns = simulated_returns.flatten()
            else:
                # Compound returns over time horizon
                portfolio_returns = np.prod(1 + simulated_returns, axis=1) - 1
            
            # Calculate VaR
            var_percentile = (1 - confidence_level) * 100
            var_value = np.percentile(portfolio_returns, var_percentile)
            
            return float(abs(var_value))
            
        except Exception as e:
            logger.error(f"Error calculating Monte Carlo VaR: {e}")
            raise
    
    async def expected_shortfall(
        self, 
        returns: np.ndarray, 
        confidence_level: float = 0.95,
        time_horizon: int = 1
    ) -> float:
        """
        Calculate Expected Shortfall (Conditional VaR)
        
        Args:
            returns: Array of historical returns
            confidence_level: Confidence level
            time_horizon: Time horizon in days
            
        Returns:
            Expected Shortfall as positive number
        """
        try:
            clean_returns = returns[np.isfinite(returns)]
            
            if time_horizon > 1:
                scaled_returns = clean_returns * np.sqrt(time_horizon)
            else:
                scaled_returns = clean_returns
            
            # Calculate VaR threshold
            var_percentile = (1 - confidence_level) * 100
            var_threshold = np.percentile(scaled_returns, var_percentile)
            
            # Expected shortfall is the mean of returns worse than VaR
            tail_returns = scaled_returns[scaled_returns <= var_threshold]
            
            if len(tail_returns) == 0:
                return float(abs(var_threshold))
            
            expected_shortfall = np.mean(tail_returns)
            return float(abs(expected_shortfall))
            
        except Exception as e:
            logger.error(f"Error calculating expected shortfall: {e}")
            raise

class CorrelationCalculator:
    """Portfolio correlation and dependence analysis"""
    
    def __init__(self):
        self.methods = ['pearson', 'spearman', 'kendall', 'ledoit_wolf']
    
    async def correlation_matrix(
        self, 
        returns_data: Dict[str, np.ndarray],
        method: str = 'pearson',
        min_periods: int = 30
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Calculate correlation matrix for portfolio assets
        
        Args:
            returns_data: Dictionary of asset returns {symbol: returns_array}
            method: Correlation method ('pearson', 'spearman', 'kendall', 'ledoit_wolf')
            min_periods: Minimum number of overlapping observations
            
        Returns:
            Tuple of (correlation_matrix, asset_symbols)
        """
        try:
            if not returns_data:
                raise ValueError("No returns data provided")
            
            # Convert to DataFrame for easier handling
            df = pd.DataFrame(returns_data)
            
            # Remove assets with insufficient data
            valid_assets = []
            for column in df.columns:
                valid_data = df[column].dropna()
                if len(valid_data) >= min_periods:
                    valid_assets.append(column)
            
            if len(valid_assets) < 2:
                raise ValueError("Need at least 2 assets with sufficient data")
            
            df_valid = df[valid_assets]
            
            # Calculate correlation matrix based on method
            if method == 'pearson':
                corr_matrix = df_valid.corr(method='pearson').values
            elif method == 'spearman':
                corr_matrix = df_valid.corr(method='spearman').values
            elif method == 'kendall':
                corr_matrix = df_valid.corr(method='kendall').values
            elif method == 'ledoit_wolf':
                # Robust correlation estimation using Ledoit-Wolf shrinkage
                lw = LedoitWolf()
                cov_matrix = lw.fit(df_valid.dropna().values).covariance_
                # Convert covariance to correlation
                std_devs = np.sqrt(np.diag(cov_matrix))
                corr_matrix = cov_matrix / np.outer(std_devs, std_devs)
            else:
                raise ValueError(f"Unsupported correlation method: {method}")
            
            return corr_matrix, valid_assets
            
        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {e}")
            raise
    
    async def rolling_correlation(
        self, 
        returns1: np.ndarray, 
        returns2: np.ndarray,
        window: int = 30
    ) -> np.ndarray:
        """
        Calculate rolling correlation between two return series
        
        Args:
            returns1: First return series
            returns2: Second return series
            window: Rolling window size
            
        Returns:
            Array of rolling correlations
        """
        try:
            df = pd.DataFrame({'asset1': returns1, 'asset2': returns2})
            rolling_corr = df['asset1'].rolling(window=window).corr(df['asset2'])
            return rolling_corr.values
            
        except Exception as e:
            logger.error(f"Error calculating rolling correlation: {e}")
            raise
    
    async def portfolio_beta(
        self, 
        portfolio_returns: np.ndarray, 
        market_returns: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Calculate portfolio beta, alpha, and R-squared vs market
        
        Args:
            portfolio_returns: Portfolio return series
            market_returns: Market benchmark return series
            
        Returns:
            Tuple of (beta, alpha, r_squared)
        """
        try:
            # Ensure same length
            min_length = min(len(portfolio_returns), len(market_returns))
            port_returns = portfolio_returns[-min_length:]
            mkt_returns = market_returns[-min_length:]
            
            # Remove any NaN values
            mask = np.isfinite(port_returns) & np.isfinite(mkt_returns)
            port_clean = port_returns[mask]
            mkt_clean = mkt_returns[mask]
            
            if len(port_clean) < 10:
                raise ValueError("Insufficient data for beta calculation")
            
            # Linear regression: portfolio = alpha + beta * market + error
            slope, intercept, r_value, p_value, std_err = stats.linregress(mkt_clean, port_clean)
            
            beta = float(slope)
            alpha = float(intercept)
            r_squared = float(r_value ** 2)
            
            return beta, alpha, r_squared
            
        except Exception as e:
            logger.error(f"Error calculating portfolio beta: {e}")
            raise

class RiskMetricsCalculator:
    """Advanced risk metrics and portfolio analytics"""
    
    def __init__(self):
        self.var_calculator = VaRCalculator()
        self.corr_calculator = CorrelationCalculator()
    
    async def portfolio_volatility(
        self, 
        returns: np.ndarray, 
        annualized: bool = True
    ) -> float:
        """
        Calculate portfolio volatility
        
        Args:
            returns: Portfolio return series
            annualized: Whether to annualize the volatility
            
        Returns:
            Portfolio volatility
        """
        try:
            clean_returns = returns[np.isfinite(returns)]
            vol = np.std(clean_returns, ddof=1)
            
            if annualized:
                # Assume daily returns, annualize with 252 trading days
                vol = vol * np.sqrt(252)
            
            return float(vol)
            
        except Exception as e:
            logger.error(f"Error calculating portfolio volatility: {e}")
            raise
    
    async def sharpe_ratio(
        self, 
        returns: np.ndarray, 
        risk_free_rate: float = 0.02,
        annualized: bool = True
    ) -> float:
        """
        Calculate Sharpe ratio
        
        Args:
            returns: Portfolio return series
            risk_free_rate: Risk-free rate (annual)
            annualized: Whether returns are annualized
            
        Returns:
            Sharpe ratio
        """
        try:
            clean_returns = returns[np.isfinite(returns)]
            
            if annualized:
                mean_return = np.mean(clean_returns) * 252  # Annualized
                vol = np.std(clean_returns, ddof=1) * np.sqrt(252)  # Annualized
            else:
                mean_return = np.mean(clean_returns)
                vol = np.std(clean_returns, ddof=1)
            
            if vol == 0:
                return 0.0
            
            sharpe = (mean_return - risk_free_rate) / vol
            return float(sharpe)
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            raise
    
    async def maximum_drawdown(
        self, 
        returns: np.ndarray
    ) -> Tuple[float, int, int]:
        """
        Calculate maximum drawdown
        
        Args:
            returns: Return series
            
        Returns:
            Tuple of (max_drawdown, start_index, end_index)
        """
        try:
            clean_returns = returns[np.isfinite(returns)]
            
            # Calculate cumulative returns
            cumulative = np.cumprod(1 + clean_returns)
            
            # Calculate running maximum
            running_max = np.maximum.accumulate(cumulative)
            
            # Calculate drawdown
            drawdown = (cumulative - running_max) / running_max
            
            # Find maximum drawdown
            max_dd_idx = np.argmin(drawdown)
            max_drawdown = drawdown[max_dd_idx]
            
            # Find start of drawdown period
            start_idx = np.argmax(running_max[:max_dd_idx + 1] == running_max[max_dd_idx])
            
            return float(abs(max_drawdown)), int(start_idx), int(max_dd_idx)
            
        except Exception as e:
            logger.error(f"Error calculating maximum drawdown: {e}")
            raise
    
    async def tracking_error(
        self, 
        portfolio_returns: np.ndarray, 
        benchmark_returns: np.ndarray,
        annualized: bool = True
    ) -> float:
        """
        Calculate tracking error vs benchmark
        
        Args:
            portfolio_returns: Portfolio return series
            benchmark_returns: Benchmark return series
            annualized: Whether to annualize the result
            
        Returns:
            Tracking error
        """
        try:
            min_length = min(len(portfolio_returns), len(benchmark_returns))
            port_returns = portfolio_returns[-min_length:]
            bench_returns = benchmark_returns[-min_length:]
            
            # Calculate excess returns
            excess_returns = port_returns - bench_returns
            
            # Remove NaN values
            clean_excess = excess_returns[np.isfinite(excess_returns)]
            
            # Calculate tracking error
            te = np.std(clean_excess, ddof=1)
            
            if annualized:
                te = te * np.sqrt(252)
            
            return float(te)
            
        except Exception as e:
            logger.error(f"Error calculating tracking error: {e}")
            raise
    
    async def information_ratio(
        self, 
        portfolio_returns: np.ndarray, 
        benchmark_returns: np.ndarray
    ) -> float:
        """
        Calculate information ratio
        
        Args:
            portfolio_returns: Portfolio return series
            benchmark_returns: Benchmark return series
            
        Returns:
            Information ratio
        """
        try:
            min_length = min(len(portfolio_returns), len(benchmark_returns))
            port_returns = portfolio_returns[-min_length:]
            bench_returns = benchmark_returns[-min_length:]
            
            # Calculate excess returns
            excess_returns = port_returns - bench_returns
            clean_excess = excess_returns[np.isfinite(excess_returns)]
            
            if len(clean_excess) == 0:
                return 0.0
            
            # Calculate active return and tracking error
            active_return = np.mean(clean_excess) * 252  # Annualized
            tracking_error = np.std(clean_excess, ddof=1) * np.sqrt(252)  # Annualized
            
            if tracking_error == 0:
                return 0.0
            
            info_ratio = active_return / tracking_error
            return float(info_ratio)
            
        except Exception as e:
            logger.error(f"Error calculating information ratio: {e}")
            raise

class RiskCalculator:
    """Main risk calculator orchestrating all risk calculations"""
    
    def __init__(self):
        self.var_calculator = VaRCalculator()
        self.corr_calculator = CorrelationCalculator()
        self.metrics_calculator = RiskMetricsCalculator()
    
    async def comprehensive_risk_analysis(
        self, 
        returns_data: Dict[str, np.ndarray],
        portfolio_weights: Optional[Dict[str, float]] = None,
        confidence_levels: List[float] = [0.95, 0.99],
        time_horizons: List[int] = [1, 7, 30],
        benchmark_returns: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive risk analysis
        
        Args:
            returns_data: Dictionary of asset returns
            portfolio_weights: Portfolio weights (equal weight if None)
            confidence_levels: VaR confidence levels
            time_horizons: Time horizons for VaR
            benchmark_returns: Benchmark returns for relative metrics
            
        Returns:
            Dictionary containing all risk metrics
        """
        try:
            results = {
                'var_metrics': {},
                'correlation_analysis': {},
                'portfolio_metrics': {},
                'benchmark_analysis': {}
            }
            
            # Calculate portfolio returns
            portfolio_returns = await self._calculate_portfolio_returns(
                returns_data, portfolio_weights
            )
            
            # VaR Analysis
            for conf_level in confidence_levels:
                for horizon in time_horizons:
                    key = f'var_{horizon}d_{int(conf_level*100)}'
                    
                    # Historical VaR
                    hist_var = await self.var_calculator.historical_var(
                        portfolio_returns, conf_level, horizon
                    )
                    results['var_metrics'][f'{key}_historical'] = hist_var
                    
                    # Parametric VaR
                    param_var = await self.var_calculator.parametric_var(
                        portfolio_returns, conf_level, horizon
                    )
                    results['var_metrics'][f'{key}_parametric'] = param_var
                    
                    # Expected Shortfall
                    es = await self.var_calculator.expected_shortfall(
                        portfolio_returns, conf_level, horizon
                    )
                    results['var_metrics'][f'es_{horizon}d_{int(conf_level*100)}'] = es
            
            # Correlation Analysis
            if len(returns_data) > 1:
                corr_matrix, symbols = await self.corr_calculator.correlation_matrix(
                    returns_data
                )
                results['correlation_analysis']['matrix'] = corr_matrix.tolist()
                results['correlation_analysis']['symbols'] = symbols
            
            # Portfolio Metrics
            volatility = await self.metrics_calculator.portfolio_volatility(portfolio_returns)
            sharpe = await self.metrics_calculator.sharpe_ratio(portfolio_returns)
            max_dd, dd_start, dd_end = await self.metrics_calculator.maximum_drawdown(portfolio_returns)
            
            results['portfolio_metrics']['volatility'] = volatility
            results['portfolio_metrics']['sharpe_ratio'] = sharpe
            results['portfolio_metrics']['max_drawdown'] = max_dd
            results['portfolio_metrics']['max_drawdown_start'] = dd_start
            results['portfolio_metrics']['max_drawdown_end'] = dd_end
            
            # Benchmark Analysis
            if benchmark_returns is not None:
                beta, alpha, r_squared = await self.corr_calculator.portfolio_beta(
                    portfolio_returns, benchmark_returns
                )
                tracking_error = await self.metrics_calculator.tracking_error(
                    portfolio_returns, benchmark_returns
                )
                info_ratio = await self.metrics_calculator.information_ratio(
                    portfolio_returns, benchmark_returns
                )
                
                results['benchmark_analysis']['beta'] = beta
                results['benchmark_analysis']['alpha'] = alpha
                results['benchmark_analysis']['r_squared'] = r_squared
                results['benchmark_analysis']['tracking_error'] = tracking_error
                results['benchmark_analysis']['information_ratio'] = info_ratio
            
            return results
            
        except Exception as e:
            logger.error(f"Error in comprehensive risk analysis: {e}")
            raise
    
    async def _calculate_portfolio_returns(
        self, 
        returns_data: Dict[str, np.ndarray],
        weights: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """Calculate portfolio returns given individual asset returns and weights"""
        try:
            # Convert to DataFrame
            df = pd.DataFrame(returns_data)
            
            # Use equal weights if not provided
            if weights is None:
                weights = {symbol: 1.0/len(returns_data) for symbol in returns_data.keys()}
            
            # Calculate weighted returns
            portfolio_returns = np.zeros(len(df))
            for symbol, weight in weights.items():
                if symbol in df.columns:
                    portfolio_returns += df[symbol].fillna(0) * weight
            
            return portfolio_returns
            
        except Exception as e:
            logger.error(f"Error calculating portfolio returns: {e}")
            raise
    
    async def calculate_enhanced_risk_metrics(
        self,
        portfolio_id: str,
        returns_data: Dict[str, np.ndarray],
        positions: Dict[str, Dict[str, Any]],
        portfolio_weights: Optional[Dict[str, float]] = None,
        include_scenarios: bool = False
    ) -> Dict[str, Any]:
        """
        Calculate enhanced risk metrics using Sprint 3 risk management system
        Integrates with enhanced_risk_calculator for advanced analytics
        """
        try:
            if not ENHANCED_INTEGRATION_AVAILABLE:
                logger.warning("Enhanced integration not available, falling back to basic analysis")
                return await self.comprehensive_risk_analysis(
                    returns_data, portfolio_weights
                )
            
            # Use enhanced risk calculator for comprehensive analysis
            analysis_config = {
                'include_scenarios': include_scenarios,
                'scenarios': ['market_crash_2008', 'interest_rate_shock'] if include_scenarios else []
            }
            
            enhanced_result = await enhanced_risk_calculator.comprehensive_portfolio_analysis(
                portfolio_id=portfolio_id,
                returns_data=returns_data,
                positions=positions,
                portfolio_weights=portfolio_weights,
                analysis_config=analysis_config
            )
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Error calculating enhanced risk metrics: {e}")
            # Fallback to basic analysis
            return await self.comprehensive_risk_analysis(
                returns_data, portfolio_weights
            )
    
    async def get_real_time_risk_snapshot(self, portfolio_id: str) -> Optional[Dict[str, Any]]:
        """
        Get real-time risk snapshot from risk monitor
        """
        try:
            if not ENHANCED_INTEGRATION_AVAILABLE:
                return None
            
            snapshot = await risk_monitor.get_current_risk_metrics(portfolio_id)
            if snapshot:
                return snapshot.to_dict()
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting real-time risk snapshot: {e}")
            return None
    
    async def calculate_position_risk_attribution(
        self,
        returns_data: Dict[str, np.ndarray],
        portfolio_weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Calculate risk attribution at position level
        """
        try:
            if not ENHANCED_INTEGRATION_AVAILABLE:
                logger.warning("Enhanced integration not available for risk attribution")
                return {}
            
            attribution = await enhanced_risk_calculator.advanced_metrics.calculate_risk_attribution(
                returns_data, portfolio_weights
            )
            
            return attribution
            
        except Exception as e:
            logger.error(f"Error calculating risk attribution: {e}")
            return {}
    
    async def run_stress_test_scenario(
        self,
        portfolio_returns: np.ndarray,
        scenario_name: str,
        custom_shocks: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Run stress test scenario using enhanced calculator
        """
        try:
            if not ENHANCED_INTEGRATION_AVAILABLE:
                logger.warning("Enhanced integration not available for stress testing")
                return {}
            
            stress_result = await enhanced_risk_calculator.run_custom_scenario(
                portfolio_returns=portfolio_returns,
                scenario_name=scenario_name,
                custom_shocks=custom_shocks or {}
            )
            
            return stress_result
            
        except Exception as e:
            logger.error(f"Error running stress test: {e}")
            return {}
    
    async def calculate_marginal_risk_contributions(
        self,
        returns_data: Dict[str, np.ndarray],
        portfolio_weights: Dict[str, float],
        confidence_level: float = 0.95
    ) -> Dict[str, float]:
        """
        Calculate marginal VaR contributions for portfolio optimization
        """
        try:
            if not ENHANCED_INTEGRATION_AVAILABLE:
                logger.warning("Enhanced integration not available for marginal VaR")
                return {}
            
            marginal_vars = await enhanced_risk_calculator.calculate_marginal_var(
                returns_data=returns_data,
                portfolio_weights=portfolio_weights,
                confidence_level=confidence_level
            )
            
            return marginal_vars
            
        except Exception as e:
            logger.error(f"Error calculating marginal risk contributions: {e}")
            return {}
    
    def get_integration_status(self) -> Dict[str, Any]:
        """
        Get status of risk management system integration
        """
        status = {
            'enhanced_integration_available': ENHANCED_INTEGRATION_AVAILABLE,
            'risk_monitor_available': ENHANCED_INTEGRATION_AVAILABLE and risk_monitor is not None,
            'enhanced_calculator_available': ENHANCED_INTEGRATION_AVAILABLE and enhanced_risk_calculator is not None
        }
        
        if ENHANCED_INTEGRATION_AVAILABLE:
            try:
                # Get detailed status from enhanced components
                status.update({
                    'enhanced_calculator_status': enhanced_risk_calculator.get_calculation_status() if enhanced_risk_calculator else {},
                    'risk_monitor_status': risk_monitor.get_monitoring_status() if risk_monitor else {}
                })
            except Exception as e:
                logger.error(f"Error getting detailed integration status: {e}")
                status['integration_error'] = str(e)
        
        return status

# Global calculator instance
risk_calculator = RiskCalculator()