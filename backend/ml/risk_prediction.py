"""
Enhanced Risk Prediction with ML-based Portfolio Optimization

Implements advanced risk prediction models and portfolio optimization:
- ML-enhanced VaR calculation with multiple methodologies
- Portfolio optimization using modern portfolio theory and Black-Litterman
- Monte Carlo simulation for stress testing scenarios
- Dynamic hedging recommendations with ML-driven strategies
- Real-time risk factor decomposition and attribution
- Alternative risk measures (CVaR, Expected Shortfall, Maximum Drawdown)
"""

import asyncio
import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

# ML and optimization imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.cluster import KMeans
import scipy.optimize as optimize
import scipy.stats as stats
from scipy.linalg import inv, pinv

# Database imports
import asyncpg
import redis.asyncio as redis

# Internal imports
from .config import get_ml_config, ModelType
from .utils import MLMetrics
try:
    from ..risk.limit_engine import LimitType, LimitScope
except ImportError:
    # Fallback enums for standalone operation
    from enum import Enum
    class LimitType(Enum):
        POSITION_SIZE = "position_size"
        VAR_LIMIT = "var_limit"
        EXPOSURE_LIMIT = "exposure_limit"
    class LimitScope(Enum):
        PORTFOLIO = "portfolio"
        SYMBOL = "symbol"

logger = logging.getLogger(__name__)


class RiskModelType(Enum):
    """Types of risk models"""
    PARAMETRIC_VAR = "parametric_var"
    HISTORICAL_VAR = "historical_var"
    MONTE_CARLO_VAR = "monte_carlo_var"
    GARCH_VAR = "garch_var"
    ML_VAR = "ml_var"
    EXPECTED_SHORTFALL = "expected_shortfall"
    MAXIMUM_DRAWDOWN = "maximum_drawdown"


class OptimizationMethod(Enum):
    """Portfolio optimization methods"""
    MEAN_VARIANCE = "mean_variance"
    BLACK_LITTERMAN = "black_litterman"
    RISK_PARITY = "risk_parity"
    MINIMUM_VARIANCE = "minimum_variance"
    MAXIMUM_SHARPE = "maximum_sharpe"
    MAXIMUM_DIVERSIFICATION = "maximum_diversification"
    HIERARCHICAL_RISK_PARITY = "hierarchical_risk_parity"


class StressTestScenario(Enum):
    """Predefined stress test scenarios"""
    MARKET_CRASH = "market_crash"
    SECTOR_ROTATION = "sector_rotation"
    INTEREST_RATE_SPIKE = "interest_rate_spike"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    CURRENCY_CRISIS = "currency_crisis"
    GEOPOLITICAL_SHOCK = "geopolitical_shock"
    INFLATION_SPIKE = "inflation_spike"
    CREDIT_CRUNCH = "credit_crunch"


@dataclass
class RiskMeasures:
    """Comprehensive risk measure results"""
    portfolio_id: str
    calculation_date: datetime
    
    # Value at Risk measures
    var_95: float  # 95% VaR
    var_99: float  # 99% VaR
    var_99_9: float  # 99.9% VaR
    
    # Expected Shortfall (Conditional VaR)
    es_95: float
    es_99: float
    
    # Volatility measures
    portfolio_volatility: float
    
    # Downside risk measures
    downside_deviation: float
    maximum_drawdown: float
    calmar_ratio: float
    
    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    
    # Optional fields (must come after required fields)
    tracking_error: Optional[float] = None
    information_ratio: Optional[float] = None
    
    # Risk attribution
    component_var: Dict[str, float] = field(default_factory=dict)
    marginal_var: Dict[str, float] = field(default_factory=dict)
    
    # Concentration risk
    concentration_ratio: float = 0.0
    effective_number_assets: float = 0.0
    
    # Model confidence and metadata
    model_confidence: float = 0.8
    calculation_method: RiskModelType = RiskModelType.HISTORICAL_VAR
    simulation_runs: Optional[int] = None
    lookback_days: int = 252
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class PortfolioOptimizationResult:
    """Portfolio optimization result"""
    portfolio_id: str
    optimization_date: datetime
    optimization_method: OptimizationMethod
    
    # Optimal weights
    optimal_weights: Dict[str, float]
    current_weights: Dict[str, float]
    weight_changes: Dict[str, float]
    
    # Expected portfolio characteristics
    expected_return: float
    expected_volatility: float
    expected_sharpe_ratio: float
    
    # Risk metrics
    portfolio_var_95: float
    portfolio_var_99: float
    maximum_drawdown: float
    
    # Optimization constraints satisfied
    constraints_satisfied: bool
    
    # Diversification metrics
    diversification_ratio: float
    effective_number_assets: float
    
    # Transaction costs and implementation
    estimated_transaction_costs: float
    turnover: float
    rebalancing_required: bool
    
    # Optional/default fields
    constraint_violations: List[str] = field(default_factory=list)
    
    # Confidence and stability
    optimization_confidence: float = 0.8
    solution_stability: float = 0.8
    
    # Alternative solutions (Pareto frontier)
    pareto_frontier: Optional[List[Dict[str, float]]] = None
    
    # Metadata
    optimization_time_seconds: float = 0.0
    convergence_status: str = "success"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StressTestResult:
    """Stress test scenario result"""
    portfolio_id: str
    scenario: StressTestScenario
    test_date: datetime
    
    # Scenario parameters
    scenario_parameters: Dict[str, float]
    scenario_description: str
    
    # Portfolio impact
    portfolio_pnl: float
    portfolio_pnl_percent: float
    
    # Position-level impacts
    position_impacts: Dict[str, float]  # symbol -> P&L impact
    
    # Risk measures under stress
    stressed_var_95: float
    stressed_var_99: float
    stressed_max_drawdown: float
    
    # Recovery analysis
    recovery_time_estimate: Optional[int] = None  # days
    recovery_probability: Optional[float] = None
    
    # Hedging recommendations
    hedging_suggestions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Correlation breakdown
    correlation_stress: Dict[str, float] = field(default_factory=dict)
    
    # Confidence and methodology
    scenario_probability: float = 0.1  # 10% probability
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HedgingRecommendation:
    """Dynamic hedging recommendation"""
    portfolio_id: str
    recommendation_date: datetime
    
    # Hedge type and rationale
    hedge_type: str  # delta, vega, correlation, tail
    hedge_rationale: str
    urgency: str  # low, medium, high, critical
    
    # Recommended instruments
    hedge_instruments: List[Dict[str, Any]]  # instrument specs and quantities
    
    # Expected hedge performance
    hedge_effectiveness: float  # 0-1 scale
    cost_of_hedge: float  # basis points
    hedge_ratio: float
    
    # Risk reduction
    var_reduction: float
    max_drawdown_reduction: float
    
    # Implementation details
    optimal_entry_price: Optional[float] = None
    stop_loss_level: Optional[float] = None
    profit_target: Optional[float] = None
    
    # Time horizon
    recommended_duration_days: int = 30
    rebalance_frequency: str = "weekly"
    
    # Alternative hedging strategies
    alternative_strategies: List[Dict[str, Any]] = field(default_factory=list)
    
    # Model confidence
    recommendation_confidence: float = 0.8
    
    metadata: Dict[str, Any] = field(default_factory=dict)


class MLRiskModel:
    """
    Machine Learning-based Risk Model
    Uses ensemble methods for enhanced VaR prediction
    """
    
    def __init__(
        self,
        lookback_window: int = 252,
        prediction_horizon: int = 1
    ):
        self.lookback_window = lookback_window
        self.prediction_horizon = prediction_horizon
        
        # ML models for different risk measures
        self.var_model = None
        self.volatility_model = None
        self.correlation_model = None
        
        # Model components
        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.model_performance = {}
        
        # Training data
        self.last_training_date = None
        self.training_features = []
        
        self.logger = logging.getLogger(f"{__name__}.MLRiskModel")
    
    def train(
        self,
        return_data: pd.DataFrame,
        market_features: pd.DataFrame,
        target_var_levels: pd.Series
    ) -> Dict[str, float]:
        """Train ML risk models"""
        try:
            # Prepare features
            X = self._prepare_features(return_data, market_features)
            
            # Train VaR prediction model
            y_var = target_var_levels.values
            self.var_model = self._train_var_model(X, y_var)
            
            # Train volatility prediction model
            y_vol = return_data.std(axis=1).values
            self.volatility_model = self._train_volatility_model(X, y_vol)
            
            # Calculate training performance
            performance = self._evaluate_models(X, y_var, y_vol)
            self.model_performance = performance
            
            self.last_training_date = datetime.utcnow()
            
            self.logger.info(f"ML risk models trained. Performance: {performance}")
            
            return performance
            
        except Exception as e:
            self.logger.error(f"Error training ML risk models: {e}")
            raise
    
    def _prepare_features(
        self,
        return_data: pd.DataFrame,
        market_features: pd.DataFrame
    ) -> np.ndarray:
        """Prepare features for ML models"""
        try:
            features = []
            
            # Return-based features
            features.append(return_data.mean(axis=1).values)  # Mean return
            features.append(return_data.std(axis=1).values)   # Volatility
            features.append(return_data.skew(axis=1).values)  # Skewness
            features.append(return_data.kurtosis(axis=1).values)  # Kurtosis
            
            # Rolling statistics
            for window in [5, 10, 20]:
                rolling_vol = return_data.rolling(window).std().mean(axis=1)
                features.append(rolling_vol.fillna(0).values)
            
            # Market features (if available)
            if not market_features.empty:
                for col in market_features.columns:
                    if market_features[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                        features.append(market_features[col].fillna(0).values)
            
            # Combine and scale features
            X = np.column_stack(features)
            X_scaled = self.scaler.fit_transform(X)
            
            # Store feature names for importance analysis
            self.training_features = [
                'mean_return', 'volatility', 'skewness', 'kurtosis',
                'vol_5d', 'vol_10d', 'vol_20d'
            ] + list(market_features.columns)
            
            return X_scaled
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {e}")
            raise
    
    def _train_var_model(self, X: np.ndarray, y: np.ndarray) -> Any:
        """Train VaR prediction model"""
        try:
            # Ensemble of models
            models = {
                'rf': RandomForestRegressor(n_estimators=100, random_state=42),
                'gbr': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'ridge': Ridge(alpha=1.0)
            }
            
            best_model = None
            best_score = -np.inf
            
            for name, model in models.items():
                # Cross-validation
                scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
                avg_score = scores.mean()
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_model = model
            
            # Train best model on full data
            best_model.fit(X, y)
            
            # Calculate feature importance (if available)
            if hasattr(best_model, 'feature_importances_'):
                self.feature_importance['var'] = dict(
                    zip(self.training_features[:len(best_model.feature_importances_)], 
                        best_model.feature_importances_)
                )
            
            return best_model
            
        except Exception as e:
            self.logger.error(f"Error training VaR model: {e}")
            raise
    
    def _train_volatility_model(self, X: np.ndarray, y: np.ndarray) -> Any:
        """Train volatility prediction model"""
        try:
            # Use Random Forest for volatility prediction
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X, y)
            
            # Store feature importance
            if hasattr(model, 'feature_importances_'):
                self.feature_importance['volatility'] = dict(
                    zip(self.training_features[:len(model.feature_importances_)], 
                        model.feature_importances_)
                )
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error training volatility model: {e}")
            raise
    
    def _evaluate_models(self, X: np.ndarray, y_var: np.ndarray, y_vol: np.ndarray) -> Dict[str, float]:
        """Evaluate trained models"""
        try:
            performance = {}
            
            if self.var_model:
                var_pred = self.var_model.predict(X)
                performance['var_r2'] = float(np.corrcoef(y_var, var_pred)[0, 1] ** 2)
                performance['var_mse'] = float(np.mean((y_var - var_pred) ** 2))
            
            if self.volatility_model:
                vol_pred = self.volatility_model.predict(X)
                performance['vol_r2'] = float(np.corrcoef(y_vol, vol_pred)[0, 1] ** 2)
                performance['vol_mse'] = float(np.mean((y_vol - vol_pred) ** 2))
            
            return performance
            
        except Exception as e:
            self.logger.error(f"Error evaluating models: {e}")
            return {}
    
    def predict_var(
        self,
        current_returns: pd.DataFrame,
        market_features: pd.DataFrame,
        confidence_levels: List[float] = [0.95, 0.99]
    ) -> Dict[float, float]:
        """Predict VaR using trained ML model"""
        try:
            if not self.var_model:
                raise ValueError("VaR model not trained")
            
            # Prepare current features
            X_current = self._prepare_features(current_returns, market_features)
            
            # Get prediction
            var_prediction = self.var_model.predict(X_current[-1:])
            
            # Convert to confidence levels (simplified mapping)
            var_estimates = {}
            base_var = var_prediction[0]
            
            for conf_level in confidence_levels:
                # Scale VaR based on confidence level
                z_score = stats.norm.ppf(conf_level)
                var_estimates[conf_level] = abs(base_var * z_score / 1.645)  # Normalize to 95%
            
            return var_estimates
            
        except Exception as e:
            self.logger.error(f"Error predicting VaR: {e}")
            return {0.95: 0.0, 0.99: 0.0}
    
    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """Get feature importance for all models"""
        return self.feature_importance.copy()


class PortfolioOptimizer:
    """
    Advanced Portfolio Optimization Engine
    Supports multiple optimization methods and constraints
    """
    
    def __init__(self):
        self.optimization_methods = {
            OptimizationMethod.MEAN_VARIANCE: self._mean_variance_optimization,
            OptimizationMethod.BLACK_LITTERMAN: self._black_litterman_optimization,
            OptimizationMethod.RISK_PARITY: self._risk_parity_optimization,
            OptimizationMethod.MINIMUM_VARIANCE: self._minimum_variance_optimization,
            OptimizationMethod.MAXIMUM_SHARPE: self._maximum_sharpe_optimization
        }
        
        self.logger = logging.getLogger(f"{__name__}.PortfolioOptimizer")
    
    def optimize_portfolio(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        method: OptimizationMethod = OptimizationMethod.MEAN_VARIANCE,
        constraints: Optional[Dict[str, Any]] = None,
        current_weights: Optional[pd.Series] = None
    ) -> PortfolioOptimizationResult:
        """Optimize portfolio using specified method"""
        try:
            start_time = datetime.utcnow()
            
            # Default constraints
            default_constraints = {
                'weight_bounds': (0.0, 1.0),  # Long-only
                'max_weight': 0.3,           # Max 30% in any asset
                'min_weight': 0.01,          # Min 1% in selected assets
                'max_turnover': 0.5,         # Max 50% turnover
                'target_risk': None,         # Target risk level
                'risk_free_rate': 0.02       # 2% risk-free rate
            }
            
            if constraints:
                default_constraints.update(constraints)
            
            # Get optimization method
            optimization_func = self.optimization_methods.get(method)
            if not optimization_func:
                raise ValueError(f"Unsupported optimization method: {method}")
            
            # Run optimization
            result = optimization_func(
                expected_returns, covariance_matrix, 
                default_constraints, current_weights
            )
            
            # Calculate additional metrics
            result = self._enhance_optimization_result(
                result, expected_returns, covariance_matrix, 
                current_weights, start_time
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in portfolio optimization: {e}")
            raise
    
    def _mean_variance_optimization(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        constraints: Dict[str, Any],
        current_weights: Optional[pd.Series]
    ) -> PortfolioOptimizationResult:
        """Mean-variance optimization (Markowitz)"""
        try:
            n_assets = len(expected_returns)
            risk_free_rate = constraints['risk_free_rate']
            
            # Objective function: maximize Sharpe ratio
            def objective(weights):
                portfolio_return = np.sum(weights * expected_returns)
                portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
                sharpe_ratio = (portfolio_return - risk_free_rate) / np.sqrt(portfolio_variance)
                return -sharpe_ratio  # Minimize negative Sharpe
            
            # Constraints
            constraint_list = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}  # Weights sum to 1
            ]
            
            # Bounds
            bounds = [constraints['weight_bounds']] * n_assets
            
            # Initial guess
            x0 = np.ones(n_assets) / n_assets
            
            # Optimization
            result = optimize.minimize(
                objective, x0, method='SLSQP',
                bounds=bounds, constraints=constraint_list,
                options={'maxiter': 1000}
            )
            
            if not result.success:
                raise RuntimeError(f"Optimization failed: {result.message}")
            
            optimal_weights = dict(zip(expected_returns.index, result.x))
            
            return PortfolioOptimizationResult(
                portfolio_id="",  # To be set by caller
                optimization_date=datetime.utcnow(),
                optimization_method=OptimizationMethod.MEAN_VARIANCE,
                optimal_weights=optimal_weights,
                current_weights=current_weights.to_dict() if current_weights is not None else {},
                weight_changes={},  # To be calculated
                expected_return=0.0,  # To be calculated
                expected_volatility=0.0,  # To be calculated
                expected_sharpe_ratio=0.0,  # To be calculated
                portfolio_var_95=0.0,  # To be calculated
                portfolio_var_99=0.0,  # To be calculated
                maximum_drawdown=0.0,  # To be calculated
                constraints_satisfied=True,
                diversification_ratio=0.0,  # To be calculated
                effective_number_assets=0.0,  # To be calculated
                estimated_transaction_costs=0.0,  # To be calculated
                turnover=0.0,  # To be calculated
                rebalancing_required=False,  # To be determined
                convergence_status="success" if result.success else "failed"
            )
            
        except Exception as e:
            self.logger.error(f"Error in mean-variance optimization: {e}")
            raise
    
    def _black_litterman_optimization(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        constraints: Dict[str, Any],
        current_weights: Optional[pd.Series]
    ) -> PortfolioOptimizationResult:
        """Black-Litterman optimization with views"""
        try:
            # Simplified Black-Litterman implementation
            # In practice, this would incorporate investor views
            
            # Market cap weights (equal weight as proxy)
            n_assets = len(expected_returns)
            market_weights = np.ones(n_assets) / n_assets
            
            # Risk aversion parameter
            risk_aversion = 3.0
            
            # Implied expected returns (reverse optimization)
            implied_returns = risk_aversion * np.dot(covariance_matrix, market_weights)
            
            # For simplicity, use implied returns as Black-Litterman returns
            bl_returns = pd.Series(implied_returns, index=expected_returns.index)
            
            # Apply mean-variance optimization with BL returns
            return self._mean_variance_optimization(
                bl_returns, covariance_matrix, constraints, current_weights
            )
            
        except Exception as e:
            self.logger.error(f"Error in Black-Litterman optimization: {e}")
            raise
    
    def _risk_parity_optimization(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        constraints: Dict[str, Any],
        current_weights: Optional[pd.Series]
    ) -> PortfolioOptimizationResult:
        """Risk parity optimization"""
        try:
            n_assets = len(expected_returns)
            
            # Objective: minimize sum of squared risk contributions
            def objective(weights):
                portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
                marginal_contrib = np.dot(covariance_matrix, weights)
                risk_contrib = weights * marginal_contrib / portfolio_variance
                target_risk_contrib = 1.0 / n_assets  # Equal risk contribution
                return np.sum((risk_contrib - target_risk_contrib) ** 2)
            
            # Constraints
            constraint_list = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}
            ]
            
            # Bounds
            bounds = [(0.01, 1.0)] * n_assets  # Long-only
            
            # Initial guess (equal weights)
            x0 = np.ones(n_assets) / n_assets
            
            # Optimization
            result = optimize.minimize(
                objective, x0, method='SLSQP',
                bounds=bounds, constraints=constraint_list,
                options={'maxiter': 1000}
            )
            
            optimal_weights = dict(zip(expected_returns.index, result.x))
            
            return PortfolioOptimizationResult(
                portfolio_id="",
                optimization_date=datetime.utcnow(),
                optimization_method=OptimizationMethod.RISK_PARITY,
                optimal_weights=optimal_weights,
                current_weights=current_weights.to_dict() if current_weights is not None else {},
                weight_changes={},
                expected_return=0.0,
                expected_volatility=0.0,
                expected_sharpe_ratio=0.0,
                portfolio_var_95=0.0,
                portfolio_var_99=0.0,
                maximum_drawdown=0.0,
                constraints_satisfied=result.success,
                diversification_ratio=0.0,
                effective_number_assets=0.0,
                estimated_transaction_costs=0.0,
                turnover=0.0,
                rebalancing_required=False,
                convergence_status="success" if result.success else "failed"
            )
            
        except Exception as e:
            self.logger.error(f"Error in risk parity optimization: {e}")
            raise
    
    def _minimum_variance_optimization(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        constraints: Dict[str, Any],
        current_weights: Optional[pd.Series]
    ) -> PortfolioOptimizationResult:
        """Minimum variance optimization"""
        try:
            n_assets = len(expected_returns)
            
            # Objective: minimize portfolio variance
            def objective(weights):
                return np.dot(weights, np.dot(covariance_matrix, weights))
            
            # Constraints
            constraint_list = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}
            ]
            
            # Bounds
            bounds = [constraints['weight_bounds']] * n_assets
            
            # Initial guess
            x0 = np.ones(n_assets) / n_assets
            
            # Optimization
            result = optimize.minimize(
                objective, x0, method='SLSQP',
                bounds=bounds, constraints=constraint_list
            )
            
            optimal_weights = dict(zip(expected_returns.index, result.x))
            
            return PortfolioOptimizationResult(
                portfolio_id="",
                optimization_date=datetime.utcnow(),
                optimization_method=OptimizationMethod.MINIMUM_VARIANCE,
                optimal_weights=optimal_weights,
                current_weights=current_weights.to_dict() if current_weights is not None else {},
                weight_changes={},
                expected_return=0.0,
                expected_volatility=0.0,
                expected_sharpe_ratio=0.0,
                portfolio_var_95=0.0,
                portfolio_var_99=0.0,
                maximum_drawdown=0.0,
                constraints_satisfied=result.success,
                diversification_ratio=0.0,
                effective_number_assets=0.0,
                estimated_transaction_costs=0.0,
                turnover=0.0,
                rebalancing_required=False,
                convergence_status="success" if result.success else "failed"
            )
            
        except Exception as e:
            self.logger.error(f"Error in minimum variance optimization: {e}")
            raise
    
    def _maximum_sharpe_optimization(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        constraints: Dict[str, Any],
        current_weights: Optional[pd.Series]
    ) -> PortfolioOptimizationResult:
        """Maximum Sharpe ratio optimization"""
        try:
            # This is equivalent to mean-variance optimization
            return self._mean_variance_optimization(
                expected_returns, covariance_matrix, constraints, current_weights
            )
            
        except Exception as e:
            self.logger.error(f"Error in maximum Sharpe optimization: {e}")
            raise
    
    def _enhance_optimization_result(
        self,
        result: PortfolioOptimizationResult,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        current_weights: Optional[pd.Series],
        start_time: datetime
    ) -> PortfolioOptimizationResult:
        """Enhance optimization result with additional metrics"""
        try:
            weights = np.array(list(result.optimal_weights.values()))
            
            # Portfolio expected return and volatility
            result.expected_return = np.sum(weights * expected_returns)
            result.expected_volatility = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
            
            # Sharpe ratio
            risk_free_rate = 0.02  # Default
            result.expected_sharpe_ratio = (result.expected_return - risk_free_rate) / result.expected_volatility
            
            # VaR estimates (simplified)
            result.portfolio_var_95 = 1.645 * result.expected_volatility  # 95% VaR
            result.portfolio_var_99 = 2.326 * result.expected_volatility  # 99% VaR
            
            # Diversification ratio
            individual_vols = np.sqrt(np.diag(covariance_matrix))
            weighted_vol = np.sum(weights * individual_vols)
            result.diversification_ratio = weighted_vol / result.expected_volatility
            
            # Effective number of assets (Herfindahl index)
            result.effective_number_assets = 1.0 / np.sum(weights ** 2)
            
            # Weight changes and turnover
            if current_weights is not None:
                current_weights_aligned = current_weights.reindex(expected_returns.index, fill_value=0.0)
                weight_changes = {}
                total_turnover = 0.0
                
                for symbol in expected_returns.index:
                    old_weight = current_weights_aligned.get(symbol, 0.0)
                    new_weight = result.optimal_weights[symbol]
                    change = new_weight - old_weight
                    weight_changes[symbol] = change
                    total_turnover += abs(change)
                
                result.weight_changes = weight_changes
                result.turnover = total_turnover
                result.rebalancing_required = total_turnover > 0.05  # 5% threshold
            
            # Transaction costs (simplified model)
            result.estimated_transaction_costs = result.turnover * 0.0010  # 10 bps per dollar traded
            
            # Optimization time
            result.optimization_time_seconds = (datetime.utcnow() - start_time).total_seconds()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error enhancing optimization result: {e}")
            return result


class StressTestEngine:
    """
    Advanced Stress Testing Engine with Monte Carlo Simulation
    """
    
    def __init__(self):
        self.scenario_generators = {
            StressTestScenario.MARKET_CRASH: self._market_crash_scenario,
            StressTestScenario.SECTOR_ROTATION: self._sector_rotation_scenario,
            StressTestScenario.INTEREST_RATE_SPIKE: self._interest_rate_spike_scenario,
            StressTestScenario.LIQUIDITY_CRISIS: self._liquidity_crisis_scenario,
            StressTestScenario.CURRENCY_CRISIS: self._currency_crisis_scenario,
            StressTestScenario.GEOPOLITICAL_SHOCK: self._geopolitical_shock_scenario
        }
        
        self.logger = logging.getLogger(f"{__name__}.StressTestEngine")
    
    def run_stress_test(
        self,
        portfolio_weights: pd.Series,
        asset_returns: pd.DataFrame,
        scenario: StressTestScenario,
        simulation_runs: int = 10000
    ) -> StressTestResult:
        """Run stress test for specified scenario"""
        try:
            # Generate scenario parameters
            scenario_generator = self.scenario_generators.get(scenario)
            if not scenario_generator:
                raise ValueError(f"Unsupported stress test scenario: {scenario}")
            
            scenario_params = scenario_generator()
            
            # Run Monte Carlo simulation
            portfolio_outcomes = self._monte_carlo_simulation(
                portfolio_weights, asset_returns, scenario_params, simulation_runs
            )
            
            # Calculate impact metrics
            portfolio_pnl = np.mean(portfolio_outcomes)
            portfolio_pnl_percent = portfolio_pnl / np.sum(portfolio_weights.values) * 100
            
            # Position-level impacts
            position_impacts = self._calculate_position_impacts(
                portfolio_weights, asset_returns, scenario_params
            )
            
            # Risk measures under stress
            stressed_var_95 = np.percentile(portfolio_outcomes, 5)  # 5th percentile
            stressed_var_99 = np.percentile(portfolio_outcomes, 1)  # 1st percentile
            
            # Maximum drawdown simulation
            stressed_max_drawdown = self._simulate_max_drawdown(portfolio_outcomes)
            
            # Generate hedging recommendations
            hedging_suggestions = self._generate_hedging_suggestions(
                scenario, portfolio_weights, position_impacts
            )
            
            return StressTestResult(
                portfolio_id="",  # To be set by caller
                scenario=scenario,
                test_date=datetime.utcnow(),
                scenario_parameters=scenario_params,
                scenario_description=self._get_scenario_description(scenario),
                portfolio_pnl=portfolio_pnl,
                portfolio_pnl_percent=portfolio_pnl_percent,
                position_impacts=position_impacts,
                stressed_var_95=stressed_var_95,
                stressed_var_99=stressed_var_99,
                stressed_max_drawdown=stressed_max_drawdown,
                hedging_suggestions=hedging_suggestions,
                scenario_probability=self._get_scenario_probability(scenario),
                confidence_interval=(np.percentile(portfolio_outcomes, 2.5), 
                                   np.percentile(portfolio_outcomes, 97.5))
            )
            
        except Exception as e:
            self.logger.error(f"Error running stress test: {e}")
            raise
    
    def _monte_carlo_simulation(
        self,
        portfolio_weights: pd.Series,
        asset_returns: pd.DataFrame,
        scenario_params: Dict[str, float],
        n_simulations: int
    ) -> np.ndarray:
        """Run Monte Carlo simulation for stress scenario"""
        try:
            # Calculate mean returns and covariance
            mean_returns = asset_returns.mean()
            cov_matrix = asset_returns.cov()
            
            # Apply stress scenario adjustments
            stressed_mean_returns = self._apply_stress_to_returns(mean_returns, scenario_params)
            stressed_cov_matrix = self._apply_stress_to_covariance(cov_matrix, scenario_params)
            
            # Generate random returns
            np.random.seed(42)  # For reproducibility
            random_returns = np.random.multivariate_normal(
                stressed_mean_returns, stressed_cov_matrix, n_simulations
            )
            
            # Calculate portfolio returns
            weights_array = np.array([portfolio_weights.get(symbol, 0.0) for symbol in mean_returns.index])
            portfolio_returns = np.dot(random_returns, weights_array)
            
            return portfolio_returns
            
        except Exception as e:
            self.logger.error(f"Error in Monte Carlo simulation: {e}")
            return np.array([0.0])
    
    def _apply_stress_to_returns(
        self, 
        mean_returns: pd.Series, 
        scenario_params: Dict[str, float]
    ) -> np.ndarray:
        """Apply stress scenario to expected returns"""
        try:
            stressed_returns = mean_returns.copy()
            
            # Apply market-wide stress
            if 'market_stress' in scenario_params:
                stressed_returns *= (1 + scenario_params['market_stress'])
            
            # Apply sector-specific stress
            if 'sector_stress' in scenario_params:
                # This would be more sophisticated in practice
                stressed_returns *= (1 + scenario_params['sector_stress'])
            
            return stressed_returns.values
            
        except Exception as e:
            self.logger.error(f"Error applying stress to returns: {e}")
            return mean_returns.values
    
    def _apply_stress_to_covariance(
        self, 
        cov_matrix: pd.DataFrame, 
        scenario_params: Dict[str, float]
    ) -> np.ndarray:
        """Apply stress scenario to covariance matrix"""
        try:
            stressed_cov = cov_matrix.copy()
            
            # Increase volatilities
            if 'volatility_multiplier' in scenario_params:
                vol_multiplier = scenario_params['volatility_multiplier']
                stressed_cov *= vol_multiplier
            
            # Increase correlations (flight to quality effect)
            if 'correlation_stress' in scenario_params:
                corr_stress = scenario_params['correlation_stress']
                
                # Convert to correlation matrix
                vol_vector = np.sqrt(np.diag(stressed_cov.values))
                corr_matrix = stressed_cov.values / np.outer(vol_vector, vol_vector)
                
                # Increase off-diagonal correlations
                stressed_corr = corr_matrix * (1 + corr_stress)
                np.fill_diagonal(stressed_corr, 1.0)  # Keep diagonal as 1
                
                # Convert back to covariance
                stressed_cov = pd.DataFrame(
                    stressed_corr * np.outer(vol_vector, vol_vector),
                    index=cov_matrix.index,
                    columns=cov_matrix.columns
                )
            
            return stressed_cov.values
            
        except Exception as e:
            self.logger.error(f"Error applying stress to covariance: {e}")
            return cov_matrix.values
    
    def _calculate_position_impacts(
        self,
        portfolio_weights: pd.Series,
        asset_returns: pd.DataFrame,
        scenario_params: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate position-level impacts"""
        try:
            position_impacts = {}
            
            # Apply scenario stress to individual positions
            for symbol in portfolio_weights.index:
                if symbol in asset_returns.columns:
                    # Simplified impact calculation
                    base_return = asset_returns[symbol].mean()
                    stressed_return = base_return * (1 + scenario_params.get('market_stress', 0.0))
                    
                    position_weight = portfolio_weights[symbol]
                    impact = position_weight * stressed_return
                    position_impacts[symbol] = impact
            
            return position_impacts
            
        except Exception as e:
            self.logger.error(f"Error calculating position impacts: {e}")
            return {}
    
    def _simulate_max_drawdown(self, portfolio_returns: np.ndarray) -> float:
        """Simulate maximum drawdown"""
        try:
            # Convert returns to cumulative returns
            cumulative_returns = np.cumsum(portfolio_returns)
            
            # Calculate running maximum
            running_max = np.maximum.accumulate(cumulative_returns)
            
            # Calculate drawdowns
            drawdowns = cumulative_returns - running_max
            
            # Return maximum drawdown (most negative)
            max_drawdown = np.min(drawdowns)
            
            return max_drawdown
            
        except Exception as e:
            self.logger.error(f"Error simulating max drawdown: {e}")
            return 0.0
    
    def _generate_hedging_suggestions(
        self,
        scenario: StressTestScenario,
        portfolio_weights: pd.Series,
        position_impacts: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Generate hedging suggestions based on stress test results"""
        try:
            suggestions = []
            
            # Find worst-performing positions
            worst_positions = sorted(position_impacts.items(), key=lambda x: x[1])[:3]
            
            for symbol, impact in worst_positions:
                if impact < -0.01:  # Significant negative impact
                    suggestion = {
                        'instrument': f'{symbol}_PUT',
                        'type': 'protective_put',
                        'notional': abs(impact) * 0.5,  # Hedge 50% of exposure
                        'rationale': f'Hedge against downside risk in {symbol}',
                        'cost_estimate': abs(impact) * 0.02  # 2% cost estimate
                    }
                    suggestions.append(suggestion)
            
            # Scenario-specific hedging
            if scenario == StressTestScenario.MARKET_CRASH:
                suggestions.append({
                    'instrument': 'VIX_CALL',
                    'type': 'volatility_hedge',
                    'notional': sum(portfolio_weights.values) * 0.1,
                    'rationale': 'Hedge against market volatility spike',
                    'cost_estimate': sum(portfolio_weights.values) * 0.015
                })
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Error generating hedging suggestions: {e}")
            return []
    
    # Scenario generators
    
    def _market_crash_scenario(self) -> Dict[str, float]:
        """Generate market crash scenario parameters"""
        return {
            'market_stress': -0.20,  # 20% market decline
            'volatility_multiplier': 2.0,  # Double volatility
            'correlation_stress': 0.3,  # 30% increase in correlations
            'liquidity_stress': 0.5   # 50% liquidity reduction
        }
    
    def _sector_rotation_scenario(self) -> Dict[str, float]:
        """Generate sector rotation scenario parameters"""
        return {
            'sector_stress': -0.10,  # 10% sector decline
            'volatility_multiplier': 1.3,
            'correlation_stress': -0.2  # Correlations decrease
        }
    
    def _interest_rate_spike_scenario(self) -> Dict[str, float]:
        """Generate interest rate spike scenario"""
        return {
            'market_stress': -0.05,  # 5% market decline
            'volatility_multiplier': 1.2,
            'duration_effect': -0.15  # Impact on duration-sensitive assets
        }
    
    def _liquidity_crisis_scenario(self) -> Dict[str, float]:
        """Generate liquidity crisis scenario"""
        return {
            'market_stress': -0.15,  # 15% decline
            'volatility_multiplier': 1.8,
            'correlation_stress': 0.4,  # Flight to quality
            'liquidity_stress': 0.7
        }
    
    def _currency_crisis_scenario(self) -> Dict[str, float]:
        """Generate currency crisis scenario"""
        return {
            'market_stress': -0.08,
            'volatility_multiplier': 1.4,
            'fx_stress': 0.15  # Currency volatility
        }
    
    def _geopolitical_shock_scenario(self) -> Dict[str, float]:
        """Generate geopolitical shock scenario"""
        return {
            'market_stress': -0.12,
            'volatility_multiplier': 1.6,
            'correlation_stress': 0.25
        }
    
    def _get_scenario_description(self, scenario: StressTestScenario) -> str:
        """Get human-readable scenario description"""
        descriptions = {
            StressTestScenario.MARKET_CRASH: "Severe market downturn with 20% decline and doubled volatility",
            StressTestScenario.SECTOR_ROTATION: "Major sector rotation with reduced correlations",
            StressTestScenario.INTEREST_RATE_SPIKE: "Sudden interest rate increase affecting duration-sensitive assets",
            StressTestScenario.LIQUIDITY_CRISIS: "Liquidity dry-up with increased correlations",
            StressTestScenario.CURRENCY_CRISIS: "Currency volatility and devaluation effects",
            StressTestScenario.GEOPOLITICAL_SHOCK: "Geopolitical crisis causing market uncertainty"
        }
        return descriptions.get(scenario, "Unknown scenario")
    
    def _get_scenario_probability(self, scenario: StressTestScenario) -> float:
        """Get estimated scenario probability"""
        probabilities = {
            StressTestScenario.MARKET_CRASH: 0.05,       # 5% per year
            StressTestScenario.SECTOR_ROTATION: 0.20,    # 20% per year
            StressTestScenario.INTEREST_RATE_SPIKE: 0.15, # 15% per year
            StressTestScenario.LIQUIDITY_CRISIS: 0.10,   # 10% per year
            StressTestScenario.CURRENCY_CRISIS: 0.08,    # 8% per year
            StressTestScenario.GEOPOLITICAL_SHOCK: 0.12  # 12% per year
        }
        return probabilities.get(scenario, 0.10)


class RiskPredictor:
    """
    Main risk prediction system integrating all components
    """
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for this ML component"""
        try:
            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "component": self.__class__.__name__,
                "initialized": hasattr(self, 'initialized') and getattr(self, 'initialized', True)
            }
        except Exception as e:
            return {
                "status": "unhealthy", 
                "timestamp": datetime.utcnow().isoformat(),
                "component": self.__class__.__name__,
                "error": str(e)
            }

    def __init__(
        self,
        database_url: str = None,
        redis_url: str = None
    ):
        self.config = get_ml_config()
        self.database_url = database_url or self.config.database_url
        self.redis_url = redis_url or self.config.redis_url
        
        # Core components
        self.db_connection: Optional[asyncpg.Connection] = None
        self.redis_client: Optional[redis.Redis] = None
        
        # Risk models
        self.ml_risk_model = MLRiskModel()
        self.portfolio_optimizer = PortfolioOptimizer()
        self.stress_test_engine = StressTestEngine()
        
        # Cache for computed results
        self.risk_cache: Dict[str, Any] = {}
        
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self) -> None:
        """Initialize the risk predictor"""
        try:
            # Initialize database connection
            self.db_connection = await asyncpg.connect(self.database_url)
            
            # Initialize Redis
            self.redis_client = redis.from_url(self.redis_url, decode_responses=False)
            await self.redis_client.ping()
            
            # Create database tables
            await self._create_database_tables()
            
            self.logger.info("Risk predictor initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize risk predictor: {e}")
            raise
    
    async def calculate_portfolio_risk(
        self,
        portfolio_id: str,
        portfolio_weights: pd.Series,
        lookback_days: int = 252
    ) -> RiskMeasures:
        """Calculate comprehensive portfolio risk measures"""
        try:
            # Get historical data
            asset_returns = await self._get_asset_returns(list(portfolio_weights.index), lookback_days)
            
            if asset_returns.empty:
                raise ValueError("No return data available for risk calculation")
            
            # Calculate portfolio returns
            portfolio_returns = (asset_returns * portfolio_weights).sum(axis=1)
            
            # Calculate risk measures
            risk_measures = self._calculate_comprehensive_risk_measures(
                portfolio_id, portfolio_returns, asset_returns, portfolio_weights
            )
            
            # Save to database and cache
            await self._save_risk_measures(risk_measures)
            self.risk_cache[portfolio_id] = risk_measures
            
            return risk_measures
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio risk: {e}")
            raise
    
    def _calculate_comprehensive_risk_measures(
        self,
        portfolio_id: str,
        portfolio_returns: pd.Series,
        asset_returns: pd.DataFrame,
        portfolio_weights: pd.Series
    ) -> RiskMeasures:
        """Calculate comprehensive risk measures"""
        try:
            # Basic statistics
            returns_array = portfolio_returns.dropna().values
            
            # VaR calculations
            var_95 = np.percentile(returns_array, 5)
            var_99 = np.percentile(returns_array, 1)
            var_99_9 = np.percentile(returns_array, 0.1)
            
            # Expected Shortfall
            es_95 = returns_array[returns_array <= var_95].mean()
            es_99 = returns_array[returns_array <= var_99].mean()
            
            # Volatility
            portfolio_volatility = portfolio_returns.std() * np.sqrt(252)  # Annualized
            
            # Downside deviation
            downside_returns = returns_array[returns_array < 0]
            downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0.0
            
            # Maximum drawdown
            cumulative_returns = (1 + portfolio_returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - running_max) / running_max
            maximum_drawdown = drawdowns.min()
            
            # Risk-adjusted returns
            risk_free_rate = 0.02  # 2%
            excess_return = portfolio_returns.mean() * 252 - risk_free_rate
            sharpe_ratio = excess_return / portfolio_volatility if portfolio_volatility > 0 else 0.0
            
            sortino_ratio = excess_return / downside_deviation if downside_deviation > 0 else 0.0
            calmar_ratio = (portfolio_returns.mean() * 252) / abs(maximum_drawdown) if maximum_drawdown < 0 else 0.0
            
            # Component VaR and Marginal VaR
            component_var, marginal_var = self._calculate_component_var(
                asset_returns, portfolio_weights, var_95
            )
            
            # Concentration measures
            concentration_ratio = (portfolio_weights ** 2).sum()  # Herfindahl index
            effective_number_assets = 1.0 / concentration_ratio if concentration_ratio > 0 else len(portfolio_weights)
            
            return RiskMeasures(
                portfolio_id=portfolio_id,
                calculation_date=datetime.utcnow(),
                var_95=abs(var_95),
                var_99=abs(var_99),
                var_99_9=abs(var_99_9),
                es_95=abs(es_95) if not np.isnan(es_95) else 0.0,
                es_99=abs(es_99) if not np.isnan(es_99) else 0.0,
                portfolio_volatility=portfolio_volatility,
                downside_deviation=downside_deviation,
                maximum_drawdown=abs(maximum_drawdown),
                calmar_ratio=calmar_ratio,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                component_var=component_var,
                marginal_var=marginal_var,
                concentration_ratio=concentration_ratio,
                effective_number_assets=effective_number_assets,
                calculation_method=RiskModelType.HISTORICAL_VAR,
                lookback_days=len(portfolio_returns)
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating comprehensive risk measures: {e}")
            raise
    
    def _calculate_component_var(
        self,
        asset_returns: pd.DataFrame,
        portfolio_weights: pd.Series,
        portfolio_var: float
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Calculate component VaR and marginal VaR"""
        try:
            # Align weights with returns
            aligned_weights = portfolio_weights.reindex(asset_returns.columns, fill_value=0.0)
            
            # Calculate covariance matrix
            cov_matrix = asset_returns.cov().values
            weights_array = aligned_weights.values
            
            # Portfolio variance
            portfolio_variance = np.dot(weights_array, np.dot(cov_matrix, weights_array))
            portfolio_vol = np.sqrt(portfolio_variance)
            
            # Marginal VaR (derivative of portfolio VaR w.r.t. weights)
            marginal_var_array = np.dot(cov_matrix, weights_array) / portfolio_vol * 1.645  # 95% z-score
            
            # Component VaR
            component_var_array = marginal_var_array * weights_array
            
            # Convert to dictionaries
            symbols = asset_returns.columns.tolist()
            component_var = dict(zip(symbols, component_var_array))
            marginal_var = dict(zip(symbols, marginal_var_array))
            
            return component_var, marginal_var
            
        except Exception as e:
            self.logger.error(f"Error calculating component VaR: {e}")
            return {}, {}
    
    async def optimize_portfolio(
        self,
        portfolio_id: str,
        expected_returns: pd.Series,
        method: OptimizationMethod = OptimizationMethod.MEAN_VARIANCE,
        constraints: Optional[Dict[str, Any]] = None,
        current_weights: Optional[pd.Series] = None
    ) -> PortfolioOptimizationResult:
        """Optimize portfolio allocation"""
        try:
            # Get covariance matrix
            asset_returns = await self._get_asset_returns(list(expected_returns.index))
            covariance_matrix = asset_returns.cov()
            
            # Run optimization
            result = self.portfolio_optimizer.optimize_portfolio(
                expected_returns, covariance_matrix, method, constraints, current_weights
            )
            
            result.portfolio_id = portfolio_id
            
            # Save result to database
            await self._save_optimization_result(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error optimizing portfolio: {e}")
            raise
    
    async def run_stress_test(
        self,
        portfolio_id: str,
        portfolio_weights: pd.Series,
        scenario: StressTestScenario,
        simulation_runs: int = 10000
    ) -> StressTestResult:
        """Run stress test for portfolio"""
        try:
            # Get asset returns
            asset_returns = await self._get_asset_returns(list(portfolio_weights.index))
            
            # Run stress test
            result = self.stress_test_engine.run_stress_test(
                portfolio_weights, asset_returns, scenario, simulation_runs
            )
            
            result.portfolio_id = portfolio_id
            
            # Save result to database
            await self._save_stress_test_result(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error running stress test: {e}")
            raise
    
    async def _get_asset_returns(
        self,
        symbols: List[str],
        lookback_days: int = 252
    ) -> pd.DataFrame:
        """Get historical asset returns"""
        try:
            query = """
                WITH daily_prices AS (
                    SELECT 
                        symbol,
                        DATE(timestamp) as date,
                        LAST_VALUE(close_price) OVER (
                            PARTITION BY symbol, DATE(timestamp) 
                            ORDER BY timestamp 
                            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                        ) as close_price
                    FROM market_data 
                    WHERE symbol = ANY($1)
                        AND timestamp >= NOW() - INTERVAL '%d days'
                ),
                unique_daily_prices AS (
                    SELECT DISTINCT symbol, date, close_price
                    FROM daily_prices
                ),
                returns AS (
                    SELECT 
                        symbol,
                        date,
                        (close_price / LAG(close_price) OVER (PARTITION BY symbol ORDER BY date) - 1) as daily_return
                    FROM unique_daily_prices
                )
                SELECT * FROM returns 
                WHERE daily_return IS NOT NULL
                ORDER BY date, symbol
            """ % lookback_days
            
            rows = await self.db_connection.fetch(query, symbols)
            
            if not rows:
                return pd.DataFrame()
            
            # Convert to DataFrame and pivot
            df = pd.DataFrame([dict(row) for row in rows])
            df['date'] = pd.to_datetime(df['date'])
            
            returns_df = df.pivot(index='date', columns='symbol', values='daily_return')
            returns_df = returns_df.fillna(0)  # Fill NaN with 0 returns
            
            return returns_df
            
        except Exception as e:
            self.logger.error(f"Error getting asset returns: {e}")
            return pd.DataFrame()
    
    # Database operations
    
    async def _create_database_tables(self) -> None:
        """Create database tables for risk prediction"""
        try:
            # Risk measures table
            await self.db_connection.execute("""
                CREATE TABLE IF NOT EXISTS ml_risk_measures (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    portfolio_id VARCHAR(100) NOT NULL,
                    calculation_date TIMESTAMPTZ NOT NULL,
                    var_95 DECIMAL(12,6),
                    var_99 DECIMAL(12,6),
                    var_99_9 DECIMAL(12,6),
                    es_95 DECIMAL(12,6),
                    es_99 DECIMAL(12,6),
                    portfolio_volatility DECIMAL(8,6),
                    tracking_error DECIMAL(8,6),
                    downside_deviation DECIMAL(8,6),
                    maximum_drawdown DECIMAL(8,6),
                    calmar_ratio DECIMAL(8,6),
                    sharpe_ratio DECIMAL(8,6),
                    sortino_ratio DECIMAL(8,6),
                    information_ratio DECIMAL(8,6),
                    component_var JSONB,
                    marginal_var JSONB,
                    concentration_ratio DECIMAL(8,6),
                    effective_number_assets DECIMAL(8,2),
                    model_confidence DECIMAL(5,4),
                    calculation_method VARCHAR(50),
                    simulation_runs INTEGER,
                    lookback_days INTEGER,
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                
                CREATE INDEX IF NOT EXISTS idx_risk_measures_portfolio_date
                    ON ml_risk_measures(portfolio_id, calculation_date);
            """)
            
            # Portfolio optimization results table
            await self.db_connection.execute("""
                CREATE TABLE IF NOT EXISTS ml_portfolio_optimizations (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    portfolio_id VARCHAR(100) NOT NULL,
                    optimization_date TIMESTAMPTZ NOT NULL,
                    optimization_method VARCHAR(50) NOT NULL,
                    optimal_weights JSONB NOT NULL,
                    current_weights JSONB,
                    weight_changes JSONB,
                    expected_return DECIMAL(8,6),
                    expected_volatility DECIMAL(8,6),
                    expected_sharpe_ratio DECIMAL(8,6),
                    portfolio_var_95 DECIMAL(12,6),
                    portfolio_var_99 DECIMAL(12,6),
                    maximum_drawdown DECIMAL(8,6),
                    constraints_satisfied BOOLEAN,
                    constraint_violations TEXT[],
                    diversification_ratio DECIMAL(8,6),
                    effective_number_assets DECIMAL(8,2),
                    estimated_transaction_costs DECIMAL(8,6),
                    turnover DECIMAL(8,6),
                    rebalancing_required BOOLEAN,
                    optimization_confidence DECIMAL(5,4),
                    solution_stability DECIMAL(5,4),
                    optimization_time_seconds DECIMAL(10,3),
                    convergence_status VARCHAR(20),
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                
                CREATE INDEX IF NOT EXISTS idx_portfolio_optimizations_portfolio_date
                    ON ml_portfolio_optimizations(portfolio_id, optimization_date);
            """)
            
            # Stress test results table
            await self.db_connection.execute("""
                CREATE TABLE IF NOT EXISTS ml_stress_test_results (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    portfolio_id VARCHAR(100) NOT NULL,
                    scenario VARCHAR(50) NOT NULL,
                    test_date TIMESTAMPTZ NOT NULL,
                    scenario_parameters JSONB,
                    scenario_description TEXT,
                    portfolio_pnl DECIMAL(12,2),
                    portfolio_pnl_percent DECIMAL(8,4),
                    position_impacts JSONB,
                    stressed_var_95 DECIMAL(12,6),
                    stressed_var_99 DECIMAL(12,6),
                    stressed_max_drawdown DECIMAL(8,6),
                    recovery_time_estimate INTEGER,
                    recovery_probability DECIMAL(5,4),
                    hedging_suggestions JSONB,
                    correlation_stress JSONB,
                    scenario_probability DECIMAL(5,4),
                    confidence_interval JSONB,
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                
                CREATE INDEX IF NOT EXISTS idx_stress_test_portfolio_scenario
                    ON ml_stress_test_results(portfolio_id, scenario);
            """)
            
            self.logger.info("Risk prediction database tables created successfully")
            
        except Exception as e:
            self.logger.error(f"Error creating database tables: {e}")
            raise
    
    async def _save_risk_measures(self, risk_measures: RiskMeasures) -> None:
        """Save risk measures to database"""
        try:
            await self.db_connection.execute("""
                INSERT INTO ml_risk_measures (
                    portfolio_id, calculation_date, var_95, var_99, var_99_9,
                    es_95, es_99, portfolio_volatility, tracking_error,
                    downside_deviation, maximum_drawdown, calmar_ratio,
                    sharpe_ratio, sortino_ratio, information_ratio,
                    component_var, marginal_var, concentration_ratio,
                    effective_number_assets, model_confidence,
                    calculation_method, simulation_runs, lookback_days, metadata
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15,
                    $16, $17, $18, $19, $20, $21, $22, $23, $24
                )
            """,
                risk_measures.portfolio_id, risk_measures.calculation_date,
                risk_measures.var_95, risk_measures.var_99, risk_measures.var_99_9,
                risk_measures.es_95, risk_measures.es_99,
                risk_measures.portfolio_volatility, risk_measures.tracking_error,
                risk_measures.downside_deviation, risk_measures.maximum_drawdown,
                risk_measures.calmar_ratio, risk_measures.sharpe_ratio,
                risk_measures.sortino_ratio, risk_measures.information_ratio,
                json.dumps(risk_measures.component_var),
                json.dumps(risk_measures.marginal_var),
                risk_measures.concentration_ratio, risk_measures.effective_number_assets,
                risk_measures.model_confidence, risk_measures.calculation_method.value,
                risk_measures.simulation_runs, risk_measures.lookback_days,
                json.dumps(risk_measures.metadata)
            )
            
        except Exception as e:
            self.logger.error(f"Error saving risk measures: {e}")
    
    async def _save_optimization_result(self, result: PortfolioOptimizationResult) -> None:
        """Save optimization result to database"""
        try:
            await self.db_connection.execute("""
                INSERT INTO ml_portfolio_optimizations (
                    portfolio_id, optimization_date, optimization_method,
                    optimal_weights, current_weights, weight_changes,
                    expected_return, expected_volatility, expected_sharpe_ratio,
                    portfolio_var_95, portfolio_var_99, maximum_drawdown,
                    constraints_satisfied, constraint_violations,
                    diversification_ratio, effective_number_assets,
                    estimated_transaction_costs, turnover, rebalancing_required,
                    optimization_confidence, solution_stability,
                    optimization_time_seconds, convergence_status, metadata
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15,
                    $16, $17, $18, $19, $20, $21, $22, $23, $24
                )
            """,
                result.portfolio_id, result.optimization_date,
                result.optimization_method.value,
                json.dumps(result.optimal_weights),
                json.dumps(result.current_weights),
                json.dumps(result.weight_changes),
                result.expected_return, result.expected_volatility,
                result.expected_sharpe_ratio, result.portfolio_var_95,
                result.portfolio_var_99, result.maximum_drawdown,
                result.constraints_satisfied, result.constraint_violations,
                result.diversification_ratio, result.effective_number_assets,
                result.estimated_transaction_costs, result.turnover,
                result.rebalancing_required, result.optimization_confidence,
                result.solution_stability, result.optimization_time_seconds,
                result.convergence_status, json.dumps(result.metadata)
            )
            
        except Exception as e:
            self.logger.error(f"Error saving optimization result: {e}")
    
    async def _save_stress_test_result(self, result: StressTestResult) -> None:
        """Save stress test result to database"""
        try:
            await self.db_connection.execute("""
                INSERT INTO ml_stress_test_results (
                    portfolio_id, scenario, test_date, scenario_parameters,
                    scenario_description, portfolio_pnl, portfolio_pnl_percent,
                    position_impacts, stressed_var_95, stressed_var_99,
                    stressed_max_drawdown, recovery_time_estimate,
                    recovery_probability, hedging_suggestions, correlation_stress,
                    scenario_probability, confidence_interval, metadata
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18
                )
            """,
                result.portfolio_id, result.scenario.value, result.test_date,
                json.dumps(result.scenario_parameters), result.scenario_description,
                result.portfolio_pnl, result.portfolio_pnl_percent,
                json.dumps(result.position_impacts), result.stressed_var_95,
                result.stressed_var_99, result.stressed_max_drawdown,
                result.recovery_time_estimate, result.recovery_probability,
                json.dumps(result.hedging_suggestions),
                json.dumps(result.correlation_stress), result.scenario_probability,
                json.dumps(list(result.confidence_interval)),
                json.dumps(result.metadata)
            )
            
        except Exception as e:
            self.logger.error(f"Error saving stress test result: {e}")
    
    # Public API methods
    
    async def get_portfolio_risk_summary(self, portfolio_id: str) -> Dict[str, Any]:
        """Get portfolio risk summary"""
        try:
            # Get latest risk measures from database
            row = await self.db_connection.fetchrow("""
                SELECT * FROM ml_risk_measures 
                WHERE portfolio_id = $1 
                ORDER BY calculation_date DESC 
                LIMIT 1
            """, portfolio_id)
            
            if row:
                return dict(row)
            else:
                return {"portfolio_id": portfolio_id, "error": "No risk measures found"}
                
        except Exception as e:
            self.logger.error(f"Error getting portfolio risk summary: {e}")
            return {"portfolio_id": portfolio_id, "error": str(e)}


# Global risk predictor instance
risk_predictor_instance = None

def get_risk_predictor() -> RiskPredictor:
    """Get global risk predictor instance"""
    global risk_predictor_instance
    if risk_predictor_instance is None:
        raise RuntimeError("Risk predictor not initialized. Call init_risk_predictor() first.")
    return risk_predictor_instance

def init_risk_predictor() -> RiskPredictor:
    """Initialize global risk predictor instance"""
    global risk_predictor_instance
    risk_predictor_instance = RiskPredictor()
    return risk_predictor_instance