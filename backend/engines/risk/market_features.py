#!/usr/bin/env python3
"""
Market Features Extraction for Supervised k-NN Portfolio Optimization
Extracts comprehensive market state features for similarity-based portfolio construction
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
from scipy import stats
from sklearn.decomposition import PCA
import warnings

logger = logging.getLogger(__name__)

@dataclass
class MarketFeatures:
    """Comprehensive market state features for k-NN optimization"""
    
    # Return characteristics
    returns_mean: float
    returns_volatility: float
    returns_skewness: float
    returns_kurtosis: float
    returns_jarque_bera_stat: float
    returns_jarque_bera_pvalue: float
    
    # Correlation structure
    average_correlation: float
    max_correlation: float
    min_correlation: float
    correlation_std: float
    max_eigenvalue: float
    eigenvalue_dispersion: float
    eigenvalue_ratio: float  # Max eigenvalue / second max eigenvalue
    
    # Volatility regime
    volatility_regime: float
    volatility_persistence: float
    volatility_clustering: float
    
    # Momentum features
    momentum_1m: float
    momentum_3m: float
    momentum_6m: float
    momentum_12m: float
    momentum_strength: float
    
    # Mean reversion indicators
    mean_reversion_strength: float
    autocorrelation_1d: float
    autocorrelation_5d: float
    
    # Risk-adjusted metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    drawdown_duration: float
    
    # Market stress indicators
    tail_risk: float  # 5th percentile return
    extreme_return_frequency: float
    downside_volatility: float
    
    # Regime detection features
    bull_bear_indicator: float
    volatility_regime_prob: float
    trend_strength: float
    
    # Cross-sectional features
    asset_dispersion: float  # Cross-sectional return dispersion
    concentration_risk: float
    sector_momentum: Dict[str, float]
    
    # Time-varying features
    feature_timestamp: datetime
    lookback_window: int
    market_phase: str  # 'bull', 'bear', 'sideways', 'volatile'

class MarketFeatureExtractor:
    """
    Advanced market feature extraction for supervised portfolio optimization
    
    Extracts multi-dimensional features capturing:
    - Return distribution characteristics
    - Correlation structure dynamics
    - Market regime information
    - Risk factor exposures
    - Cross-sectional patterns
    """
    
    def __init__(self, 
                 lookback_windows: List[int] = None,
                 volatility_threshold: float = 0.02,
                 trend_threshold: float = 0.05,
                 sector_mapping: Dict[str, str] = None):
        """
        Initialize feature extractor
        
        Args:
            lookback_windows: List of lookback periods for different features
            volatility_threshold: Threshold for volatility regime classification
            trend_threshold: Threshold for trend strength calculation
            sector_mapping: Mapping of assets to sectors
        """
        self.lookback_windows = lookback_windows or [21, 63, 126, 252]
        self.volatility_threshold = volatility_threshold
        self.trend_threshold = trend_threshold
        self.sector_mapping = sector_mapping or {}
        self._cache = {}
        
    def extract_features(self, 
                        returns_data: pd.DataFrame,
                        benchmark_returns: Optional[pd.Series] = None,
                        market_data: Optional[Dict[str, Any]] = None,
                        timestamp: Optional[datetime] = None) -> MarketFeatures:
        """
        Extract comprehensive market features from returns data
        
        Args:
            returns_data: DataFrame with asset returns (dates x assets)
            benchmark_returns: Optional benchmark returns series
            market_data: Optional additional market data (VIX, term spreads, etc.)
            timestamp: Timestamp for the feature extraction
            
        Returns:
            MarketFeatures object containing all extracted features
        """
        try:
            timestamp = timestamp or datetime.now()
            
            # Ensure we have enough data
            if len(returns_data) < self.lookback_windows[0]:
                logger.warning(f"Insufficient data for feature extraction: {len(returns_data)} < {self.lookback_windows[0]}")
                return self._create_default_features(timestamp)
            
            # Extract return characteristics
            return_features = self._extract_return_characteristics(returns_data)
            
            # Extract correlation structure features
            correlation_features = self._extract_correlation_features(returns_data)
            
            # Extract volatility regime features
            volatility_features = self._extract_volatility_features(returns_data)
            
            # Extract momentum features
            momentum_features = self._extract_momentum_features(returns_data)
            
            # Extract mean reversion features
            mean_reversion_features = self._extract_mean_reversion_features(returns_data)
            
            # Extract risk-adjusted metrics
            risk_features = self._extract_risk_metrics(returns_data, benchmark_returns)
            
            # Extract tail risk and stress features
            tail_risk_features = self._extract_tail_risk_features(returns_data)
            
            # Extract regime detection features
            regime_features = self._extract_regime_features(returns_data, market_data)
            
            # Extract cross-sectional features
            cross_sectional_features = self._extract_cross_sectional_features(returns_data)
            
            # Combine all features
            features = MarketFeatures(
                # Return characteristics
                **return_features,
                
                # Correlation structure
                **correlation_features,
                
                # Volatility regime
                **volatility_features,
                
                # Momentum features
                **momentum_features,
                
                # Mean reversion
                **mean_reversion_features,
                
                # Risk-adjusted metrics
                **risk_features,
                
                # Tail risk
                **tail_risk_features,
                
                # Regime features
                **regime_features,
                
                # Cross-sectional features
                **cross_sectional_features,
                
                # Metadata
                feature_timestamp=timestamp,
                lookback_window=len(returns_data),
                market_phase=self._classify_market_phase(returns_data, market_data)
            )
            
            logger.debug(f"Successfully extracted features for {timestamp}")
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return self._create_default_features(timestamp or datetime.now())
    
    def _extract_return_characteristics(self, returns_data: pd.DataFrame) -> Dict[str, float]:
        """Extract basic return distribution characteristics"""
        # Portfolio-level returns (equal weight for simplicity)
        portfolio_returns = returns_data.mean(axis=1)
        
        # Basic statistics
        mean_return = float(portfolio_returns.mean())
        volatility = float(portfolio_returns.std())
        skewness = float(portfolio_returns.skew())
        kurtosis = float(portfolio_returns.kurtosis())
        
        # Normality test (Jarque-Bera)
        try:
            jb_stat, jb_pvalue = stats.jarque_bera(portfolio_returns.dropna())
        except:
            jb_stat, jb_pvalue = 0.0, 1.0
        
        return {
            'returns_mean': mean_return,
            'returns_volatility': volatility,
            'returns_skewness': skewness,
            'returns_kurtosis': kurtosis,
            'returns_jarque_bera_stat': float(jb_stat),
            'returns_jarque_bera_pvalue': float(jb_pvalue)
        }
    
    def _extract_correlation_features(self, returns_data: pd.DataFrame) -> Dict[str, float]:
        """Extract correlation structure features"""
        # Compute correlation matrix
        corr_matrix = returns_data.corr()
        
        # Remove diagonal (self-correlations)
        mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
        correlations = corr_matrix.values[mask]
        
        # Basic correlation statistics
        avg_correlation = float(np.mean(correlations))
        max_correlation = float(np.max(correlations))
        min_correlation = float(np.min(correlations))
        correlation_std = float(np.std(correlations))
        
        # Eigenvalue analysis
        try:
            eigenvalues = np.linalg.eigvals(corr_matrix)
            eigenvalues = np.real(eigenvalues[eigenvalues > 0])  # Keep only positive real eigenvalues
            eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending
            
            max_eigenvalue = float(eigenvalues[0]) if len(eigenvalues) > 0 else 1.0
            eigenvalue_dispersion = float(np.std(eigenvalues)) if len(eigenvalues) > 0 else 0.0
            eigenvalue_ratio = float(eigenvalues[0] / eigenvalues[1]) if len(eigenvalues) > 1 else 1.0
        except:
            max_eigenvalue = 1.0
            eigenvalue_dispersion = 0.0
            eigenvalue_ratio = 1.0
        
        return {
            'average_correlation': avg_correlation,
            'max_correlation': max_correlation,
            'min_correlation': min_correlation,
            'correlation_std': correlation_std,
            'max_eigenvalue': max_eigenvalue,
            'eigenvalue_dispersion': eigenvalue_dispersion,
            'eigenvalue_ratio': eigenvalue_ratio
        }
    
    def _extract_volatility_features(self, returns_data: pd.DataFrame) -> Dict[str, float]:
        """Extract volatility regime and clustering features"""
        portfolio_returns = returns_data.mean(axis=1)
        
        # Rolling volatility for regime detection
        vol_window = min(21, len(portfolio_returns) // 3)
        rolling_vol = portfolio_returns.rolling(vol_window).std()
        
        # Current vs historical volatility regime
        current_vol = rolling_vol.iloc[-1] if len(rolling_vol) > 0 else portfolio_returns.std()
        avg_vol = rolling_vol.mean()
        volatility_regime = float(current_vol / avg_vol) if avg_vol > 0 else 1.0
        
        # Volatility persistence (autocorrelation of squared returns)
        squared_returns = portfolio_returns ** 2
        try:
            vol_persistence = float(squared_returns.autocorr(1))
            if np.isnan(vol_persistence):
                vol_persistence = 0.0
        except:
            vol_persistence = 0.0
        
        # Volatility clustering (GARCH-like measure)
        try:
            vol_clustering = float(rolling_vol.std() / rolling_vol.mean()) if rolling_vol.mean() > 0 else 0.0
        except:
            vol_clustering = 0.0
        
        return {
            'volatility_regime': volatility_regime,
            'volatility_persistence': vol_persistence,
            'volatility_clustering': vol_clustering
        }
    
    def _extract_momentum_features(self, returns_data: pd.DataFrame) -> Dict[str, float]:
        """Extract momentum features across different horizons"""
        portfolio_returns = returns_data.mean(axis=1)
        
        # Cumulative returns over different periods
        momentum_1m = float(portfolio_returns.tail(21).sum()) if len(portfolio_returns) >= 21 else 0.0
        momentum_3m = float(portfolio_returns.tail(63).sum()) if len(portfolio_returns) >= 63 else 0.0
        momentum_6m = float(portfolio_returns.tail(126).sum()) if len(portfolio_returns) >= 126 else 0.0
        momentum_12m = float(portfolio_returns.tail(252).sum()) if len(portfolio_returns) >= 252 else 0.0
        
        # Momentum strength (consistency of directional moves)
        try:
            recent_returns = portfolio_returns.tail(21) if len(portfolio_returns) >= 21 else portfolio_returns
            positive_days = (recent_returns > 0).sum()
            momentum_strength = float(positive_days / len(recent_returns)) if len(recent_returns) > 0 else 0.5
        except:
            momentum_strength = 0.5
        
        return {
            'momentum_1m': momentum_1m,
            'momentum_3m': momentum_3m,
            'momentum_6m': momentum_6m,
            'momentum_12m': momentum_12m,
            'momentum_strength': momentum_strength
        }
    
    def _extract_mean_reversion_features(self, returns_data: pd.DataFrame) -> Dict[str, float]:
        """Extract mean reversion indicators"""
        portfolio_returns = returns_data.mean(axis=1)
        
        # Mean reversion strength (negative autocorrelation)
        try:
            autocorr_1d = float(portfolio_returns.autocorr(1))
            autocorr_5d = float(portfolio_returns.autocorr(5)) if len(portfolio_returns) >= 10 else 0.0
            
            if np.isnan(autocorr_1d):
                autocorr_1d = 0.0
            if np.isnan(autocorr_5d):
                autocorr_5d = 0.0
            
            # Mean reversion strength (more negative = more mean reverting)
            mean_reversion_strength = -min(autocorr_1d, 0.0)
            
        except:
            autocorr_1d = 0.0
            autocorr_5d = 0.0
            mean_reversion_strength = 0.0
        
        return {
            'mean_reversion_strength': mean_reversion_strength,
            'autocorrelation_1d': autocorr_1d,
            'autocorrelation_5d': autocorr_5d
        }
    
    def _extract_risk_metrics(self, returns_data: pd.DataFrame, 
                            benchmark_returns: Optional[pd.Series] = None) -> Dict[str, float]:
        """Extract risk-adjusted performance metrics"""
        portfolio_returns = returns_data.mean(axis=1)
        
        # Basic risk metrics
        annual_return = float(portfolio_returns.mean() * 252)
        annual_vol = float(portfolio_returns.std() * np.sqrt(252))
        
        # Sharpe ratio (assuming 2% risk-free rate)
        sharpe_ratio = (annual_return - 0.02) / annual_vol if annual_vol > 0 else 0.0
        
        # Sortino ratio (downside deviation)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_vol = float(downside_returns.std() * np.sqrt(252)) if len(downside_returns) > 0 else annual_vol
        sortino_ratio = (annual_return - 0.02) / downside_vol if downside_vol > 0 else 0.0
        
        # Maximum drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = float(drawdown.min())
        
        # Drawdown duration (simplified)
        drawdown_periods = (drawdown < -0.05).sum()  # Periods with >5% drawdown
        drawdown_duration = float(drawdown_periods / len(portfolio_returns)) if len(portfolio_returns) > 0 else 0.0
        
        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'drawdown_duration': drawdown_duration
        }
    
    def _extract_tail_risk_features(self, returns_data: pd.DataFrame) -> Dict[str, float]:
        """Extract tail risk and extreme event features"""
        portfolio_returns = returns_data.mean(axis=1)
        
        # Tail risk (5th percentile)
        tail_risk = float(np.percentile(portfolio_returns, 5))
        
        # Extreme return frequency (returns beyond 2 standard deviations)
        std_dev = portfolio_returns.std()
        extreme_threshold = 2 * std_dev
        extreme_returns = portfolio_returns[np.abs(portfolio_returns) > extreme_threshold]
        extreme_return_frequency = float(len(extreme_returns) / len(portfolio_returns)) if len(portfolio_returns) > 0 else 0.0
        
        # Downside volatility
        negative_returns = portfolio_returns[portfolio_returns < 0]
        downside_volatility = float(negative_returns.std()) if len(negative_returns) > 0 else 0.0
        
        return {
            'tail_risk': tail_risk,
            'extreme_return_frequency': extreme_return_frequency,
            'downside_volatility': downside_volatility
        }
    
    def _extract_regime_features(self, returns_data: pd.DataFrame, 
                               market_data: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Extract market regime detection features"""
        portfolio_returns = returns_data.mean(axis=1)
        
        # Bull/bear indicator (based on cumulative returns and trend)
        cumulative_return = float(portfolio_returns.sum())
        
        # Simple trend analysis
        if len(portfolio_returns) >= 21:
            early_period = portfolio_returns.iloc[:len(portfolio_returns)//2].sum()
            late_period = portfolio_returns.iloc[len(portfolio_returns)//2:].sum()
            trend_strength = float((late_period - early_period) / len(portfolio_returns))
        else:
            trend_strength = cumulative_return / len(portfolio_returns) if len(portfolio_returns) > 0 else 0.0
        
        # Bull/bear indicator (-1 = strong bear, +1 = strong bull)
        bull_bear_indicator = np.tanh(trend_strength * 10)  # Scale and bound between -1 and 1
        
        # Volatility regime probability (based on current vs historical volatility)
        current_vol = portfolio_returns.std()
        if len(portfolio_returns) >= 63:
            historical_vol = portfolio_returns.iloc[:-21].std() if len(portfolio_returns) > 21 else current_vol
        else:
            historical_vol = current_vol
        
        # Probability of high volatility regime
        vol_ratio = current_vol / historical_vol if historical_vol > 0 else 1.0
        volatility_regime_prob = float(1 / (1 + np.exp(-5 * (vol_ratio - 1))))  # Sigmoid transformation
        
        return {
            'bull_bear_indicator': float(bull_bear_indicator),
            'volatility_regime_prob': volatility_regime_prob,
            'trend_strength': trend_strength
        }
    
    def _extract_cross_sectional_features(self, returns_data: pd.DataFrame) -> Dict[str, float]:
        """Extract cross-sectional features across assets"""
        # Asset dispersion (cross-sectional standard deviation of returns)
        latest_returns = returns_data.iloc[-1] if len(returns_data) > 0 else pd.Series([0])
        asset_dispersion = float(latest_returns.std())
        
        # Concentration risk (based on return volatilities)
        asset_vols = returns_data.std()
        if len(asset_vols) > 0:
            max_vol = asset_vols.max()
            avg_vol = asset_vols.mean()
            concentration_risk = float(max_vol / avg_vol) if avg_vol > 0 else 1.0
        else:
            concentration_risk = 1.0
        
        # Sector momentum (if sector mapping is provided)
        sector_momentum = {}
        if self.sector_mapping:
            for sector in set(self.sector_mapping.values()):
                sector_assets = [asset for asset, s in self.sector_mapping.items() if s == sector]
                sector_assets = [asset for asset in sector_assets if asset in returns_data.columns]
                
                if sector_assets:
                    sector_returns = returns_data[sector_assets].mean(axis=1)
                    sector_momentum[sector] = float(sector_returns.tail(21).sum()) if len(sector_returns) >= 21 else 0.0
        
        # If no sector data, create default
        if not sector_momentum:
            sector_momentum = {'default': 0.0}
        
        return {
            'asset_dispersion': asset_dispersion,
            'concentration_risk': concentration_risk,
            'sector_momentum': sector_momentum
        }
    
    def _classify_market_phase(self, returns_data: pd.DataFrame, 
                             market_data: Optional[Dict[str, Any]] = None) -> str:
        """Classify current market phase"""
        portfolio_returns = returns_data.mean(axis=1)
        
        # Simple classification based on returns and volatility
        cumulative_return = portfolio_returns.sum()
        volatility = portfolio_returns.std()
        
        # Thresholds
        high_vol_threshold = self.volatility_threshold * 1.5
        trend_up_threshold = self.trend_threshold
        trend_down_threshold = -self.trend_threshold
        
        if cumulative_return > trend_up_threshold:
            if volatility > high_vol_threshold:
                return "volatile_bull"
            else:
                return "bull"
        elif cumulative_return < trend_down_threshold:
            if volatility > high_vol_threshold:
                return "volatile_bear"
            else:
                return "bear"
        else:
            if volatility > high_vol_threshold:
                return "volatile"
            else:
                return "sideways"
    
    def _create_default_features(self, timestamp: datetime) -> MarketFeatures:
        """Create default features when extraction fails"""
        return MarketFeatures(
            # Return characteristics
            returns_mean=0.0,
            returns_volatility=0.02,
            returns_skewness=0.0,
            returns_kurtosis=3.0,
            returns_jarque_bera_stat=0.0,
            returns_jarque_bera_pvalue=1.0,
            
            # Correlation structure
            average_correlation=0.3,
            max_correlation=0.5,
            min_correlation=0.1,
            correlation_std=0.1,
            max_eigenvalue=1.0,
            eigenvalue_dispersion=0.0,
            eigenvalue_ratio=1.0,
            
            # Volatility regime
            volatility_regime=1.0,
            volatility_persistence=0.0,
            volatility_clustering=0.0,
            
            # Momentum features
            momentum_1m=0.0,
            momentum_3m=0.0,
            momentum_6m=0.0,
            momentum_12m=0.0,
            momentum_strength=0.5,
            
            # Mean reversion
            mean_reversion_strength=0.0,
            autocorrelation_1d=0.0,
            autocorrelation_5d=0.0,
            
            # Risk-adjusted metrics
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            max_drawdown=0.0,
            drawdown_duration=0.0,
            
            # Tail risk
            tail_risk=-0.02,
            extreme_return_frequency=0.05,
            downside_volatility=0.02,
            
            # Regime features
            bull_bear_indicator=0.0,
            volatility_regime_prob=0.5,
            trend_strength=0.0,
            
            # Cross-sectional features
            asset_dispersion=0.02,
            concentration_risk=1.0,
            sector_momentum={'default': 0.0},
            
            # Metadata
            feature_timestamp=timestamp,
            lookback_window=0,
            market_phase="unknown"
        )
    
    def compute_feature_importance(self, features_list: List[MarketFeatures]) -> Dict[str, float]:
        """
        Compute feature importance based on variance and predictive power
        
        Returns dictionary of feature names to importance scores
        """
        if len(features_list) < 2:
            return {}
        
        # Convert features to DataFrame for analysis
        feature_dicts = [asdict(features) for features in features_list]
        
        # Remove non-numeric fields
        numeric_features = {}
        for feature_dict in feature_dicts:
            for key, value in feature_dict.items():
                if isinstance(value, (int, float)) and not key.endswith('_timestamp'):
                    if key not in numeric_features:
                        numeric_features[key] = []
                    numeric_features[key].append(value)
        
        # Compute importance based on coefficient of variation
        importance_scores = {}
        for feature_name, values in numeric_features.items():
            values = np.array(values)
            if len(values) > 1 and np.std(values) > 0:
                # Use coefficient of variation as a simple importance measure
                cv = np.std(values) / (np.mean(np.abs(values)) + 1e-8)
                importance_scores[feature_name] = float(cv)
            else:
                importance_scores[feature_name] = 0.0
        
        # Normalize to sum to 1
        total_importance = sum(importance_scores.values())
        if total_importance > 0:
            importance_scores = {k: v / total_importance for k, v in importance_scores.items()}
        
        return importance_scores

def extract_market_features(returns_data: pd.DataFrame,
                           benchmark_returns: Optional[pd.Series] = None,
                           market_data: Optional[Dict[str, Any]] = None,
                           lookback_windows: List[int] = None,
                           sector_mapping: Dict[str, str] = None) -> MarketFeatures:
    """
    Convenience function to extract market features with default settings
    
    Args:
        returns_data: Asset returns DataFrame
        benchmark_returns: Optional benchmark returns
        market_data: Optional additional market data
        lookback_windows: Lookback periods for feature calculation
        sector_mapping: Asset to sector mapping
        
    Returns:
        MarketFeatures object
    """
    extractor = MarketFeatureExtractor(
        lookback_windows=lookback_windows,
        sector_mapping=sector_mapping
    )
    
    return extractor.extract_features(
        returns_data=returns_data,
        benchmark_returns=benchmark_returns,
        market_data=market_data
    )

def create_feature_weights_for_knn() -> Dict[str, float]:
    """
    Create default feature weights optimized for k-NN portfolio optimization
    
    Returns:
        Dictionary of feature names to weights
    """
    return {
        # High importance - core risk and return characteristics
        'returns_volatility': 2.5,
        'sharpe_ratio': 2.5,
        'max_drawdown': 2.0,
        'average_correlation': 2.0,
        
        # Medium-high importance - market regime and structure
        'volatility_regime': 1.8,
        'bull_bear_indicator': 1.8,
        'momentum_strength': 1.5,
        'trend_strength': 1.5,
        
        # Medium importance - distribution characteristics
        'returns_skewness': 1.2,
        'returns_kurtosis': 1.2,
        'tail_risk': 1.3,
        'downside_volatility': 1.3,
        
        # Lower importance - momentum and technical indicators
        'momentum_1m': 1.0,
        'momentum_3m': 1.0,
        'momentum_6m': 0.8,
        'momentum_12m': 0.7,
        'mean_reversion_strength': 1.0,
        
        # Structural features - medium importance
        'max_eigenvalue': 1.2,
        'eigenvalue_ratio': 1.1,
        'concentration_risk': 1.4,
        'asset_dispersion': 1.1
    }