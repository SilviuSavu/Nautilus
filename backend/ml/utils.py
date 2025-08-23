"""
ML Framework Utilities
Common utility functions and classes used across the ML framework.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class MLMetrics:
    """Standard ML metrics container"""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    mse: float = 0.0
    rmse: float = 0.0
    mae: float = 0.0
    r2_score: float = 0.0
    auc: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary"""
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'mse': self.mse,
            'rmse': self.rmse,
            'mae': self.mae,
            'r2_score': self.r2_score,
            'auc': self.auc
        }


class DataValidator:
    """Data validation utilities for ML workflows"""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
        """Validate DataFrame has required columns"""
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        return True
    
    @staticmethod
    def validate_numeric_data(data: Union[np.ndarray, pd.Series]) -> bool:
        """Validate that data is numeric and finite"""
        if isinstance(data, pd.Series):
            data = data.values
        
        if not np.issubdtype(data.dtype, np.number):
            logger.error("Data is not numeric")
            return False
        
        if np.isnan(data).any():
            logger.warning("Data contains NaN values")
        
        if np.isinf(data).any():
            logger.error("Data contains infinite values")
            return False
        
        return True
    
    @staticmethod
    def validate_price_data(prices: pd.Series) -> bool:
        """Validate price data for financial calculations"""
        if not DataValidator.validate_numeric_data(prices):
            return False
        
        if (prices <= 0).any():
            logger.error("Price data contains non-positive values")
            return False
        
        return True


class TimeSeriesUtils:
    """Utilities for time series data processing"""
    
    @staticmethod
    def calculate_returns(prices: pd.Series, method: str = 'simple') -> pd.Series:
        """Calculate returns from price series"""
        if method == 'simple':
            returns = prices.pct_change()
        elif method == 'log':
            returns = np.log(prices / prices.shift(1))
        else:
            raise ValueError(f"Unknown return calculation method: {method}")
        
        return returns.dropna()
    
    @staticmethod
    def calculate_rolling_statistics(
        data: pd.Series, 
        window: int,
        statistics: List[str] = ['mean', 'std', 'skew', 'kurt']
    ) -> pd.DataFrame:
        """Calculate rolling statistics for time series"""
        results = {}
        
        rolling = data.rolling(window)
        
        if 'mean' in statistics:
            results['mean'] = rolling.mean()
        if 'std' in statistics:
            results['std'] = rolling.std()
        if 'skew' in statistics:
            results['skew'] = rolling.skew()
        if 'kurt' in statistics:
            results['kurt'] = rolling.kurt()
        if 'min' in statistics:
            results['min'] = rolling.min()
        if 'max' in statistics:
            results['max'] = rolling.max()
        if 'median' in statistics:
            results['median'] = rolling.median()
        
        return pd.DataFrame(results)
    
    @staticmethod
    def detect_outliers(data: pd.Series, method: str = 'iqr', threshold: float = 1.5) -> pd.Series:
        """Detect outliers in time series data"""
        if method == 'iqr':
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            return (data < lower_bound) | (data > upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs((data - data.mean()) / data.std())
            return z_scores > threshold
        
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")


class FeatureUtils:
    """Utilities for feature engineering and selection"""
    
    @staticmethod
    def calculate_technical_indicators(
        prices: pd.Series,
        high: Optional[pd.Series] = None,
        low: Optional[pd.Series] = None,
        volume: Optional[pd.Series] = None
    ) -> Dict[str, pd.Series]:
        """Calculate common technical indicators"""
        indicators = {}
        
        # Price-based indicators
        indicators['sma_20'] = prices.rolling(20).mean()
        indicators['sma_50'] = prices.rolling(50).mean()
        indicators['ema_20'] = prices.ewm(span=20).mean()
        indicators['ema_50'] = prices.ewm(span=50).mean()
        
        # Volatility indicators
        returns = TimeSeriesUtils.calculate_returns(prices)
        indicators['volatility_20'] = returns.rolling(20).std() * np.sqrt(252)
        indicators['volatility_50'] = returns.rolling(50).std() * np.sqrt(252)
        
        # Momentum indicators
        indicators['roc_10'] = prices.pct_change(periods=10)
        indicators['roc_20'] = prices.pct_change(periods=20)
        
        # Bollinger Bands
        sma_20 = indicators['sma_20']
        std_20 = prices.rolling(20).std()
        indicators['bb_upper'] = sma_20 + (2 * std_20)
        indicators['bb_lower'] = sma_20 - (2 * std_20)
        indicators['bb_width'] = (indicators['bb_upper'] - indicators['bb_lower']) / sma_20
        
        # RSI
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        indicators['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        indicators['macd'] = ema_12 - ema_26
        indicators['macd_signal'] = indicators['macd'].ewm(span=9).mean()
        
        # High/Low based indicators if available
        if high is not None and low is not None:
            # True Range
            prev_close = prices.shift(1)
            true_range = pd.concat([
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            ], axis=1).max(axis=1)
            indicators['atr'] = true_range.rolling(14).mean()
            
            # Stochastic Oscillator
            lowest_low = low.rolling(14).min()
            highest_high = high.rolling(14).max()
            indicators['stoch_k'] = 100 * (prices - lowest_low) / (highest_high - lowest_low)
            indicators['stoch_d'] = indicators['stoch_k'].rolling(3).mean()
        
        # Volume-based indicators if available
        if volume is not None:
            indicators['volume_sma'] = volume.rolling(20).mean()
            indicators['volume_ratio'] = volume / indicators['volume_sma']
            
            # On-Balance Volume
            price_change = prices.diff()
            obv = np.where(price_change > 0, volume, np.where(price_change < 0, -volume, 0))
            indicators['obv'] = pd.Series(obv, index=prices.index).cumsum()
        
        return indicators
    
    @staticmethod
    def calculate_correlation_features(
        data: pd.DataFrame,
        target_column: str,
        threshold: float = 0.1
    ) -> pd.DataFrame:
        """Calculate correlation-based features"""
        correlations = data.corr()[target_column].abs().sort_values(ascending=False)
        relevant_features = correlations[correlations > threshold].index.tolist()
        
        if target_column in relevant_features:
            relevant_features.remove(target_column)
        
        return data[relevant_features]


class ModelUtils:
    """Utilities for ML model operations"""
    
    @staticmethod
    def split_time_series_data(
        data: pd.DataFrame,
        target_column: str,
        train_size: float = 0.7,
        validation_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Split time series data maintaining temporal order"""
        n_samples = len(data)
        train_end = int(n_samples * train_size)
        val_end = int(n_samples * (train_size + validation_size))
        
        # Split features
        X_train = data.iloc[:train_end].drop(columns=[target_column])
        X_val = data.iloc[train_end:val_end].drop(columns=[target_column])
        X_test = data.iloc[val_end:].drop(columns=[target_column])
        
        # Split target
        y_train = data.iloc[:train_end][target_column]
        y_val = data.iloc[train_end:val_end][target_column]
        y_test = data.iloc[val_end:][target_column]
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    @staticmethod
    def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> MLMetrics:
        """Calculate classification metrics"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        try:
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted')
            recall = recall_score(y_true, y_pred, average='weighted')
            f1 = f1_score(y_true, y_pred, average='weighted')
            
            # AUC only for binary classification
            auc = 0.0
            if len(np.unique(y_true)) == 2:
                auc = roc_auc_score(y_true, y_pred)
            
            return MLMetrics(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                auc=auc
            )
        except Exception as e:
            logger.error(f"Error calculating classification metrics: {e}")
            return MLMetrics()
    
    @staticmethod
    def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> MLMetrics:
        """Calculate regression metrics"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        try:
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            return MLMetrics(
                mse=mse,
                rmse=rmse,
                mae=mae,
                r2_score=r2
            )
        except Exception as e:
            logger.error(f"Error calculating regression metrics: {e}")
            return MLMetrics()


class RiskUtils:
    """Utilities for risk calculations"""
    
    @staticmethod
    def calculate_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk"""
        if len(returns) == 0:
            return 0.0
        
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    @staticmethod
    def calculate_expected_shortfall(returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        var = RiskUtils.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()
    
    @staticmethod
    def calculate_maximum_drawdown(returns: pd.Series) -> float:
        """Calculate Maximum Drawdown"""
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - running_max) / running_max
        return drawdowns.min()
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe Ratio"""
        if returns.std() == 0:
            return 0.0
        return (returns.mean() - risk_free_rate) / returns.std() * np.sqrt(252)
    
    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """Calculate Sortino Ratio"""
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        return (returns.mean() - risk_free_rate) / downside_returns.std() * np.sqrt(252)


class CacheUtils:
    """Utilities for caching ML computations"""
    
    @staticmethod
    def generate_cache_key(prefix: str, **kwargs) -> str:
        """Generate cache key from parameters"""
        import hashlib
        import json
        
        # Sort kwargs for consistent key generation
        sorted_kwargs = json.dumps(kwargs, sort_keys=True)
        hash_obj = hashlib.md5(sorted_kwargs.encode())
        return f"{prefix}:{hash_obj.hexdigest()}"
    
    @staticmethod
    def is_cache_valid(timestamp: datetime, ttl_seconds: int) -> bool:
        """Check if cached data is still valid"""
        return (datetime.utcnow() - timestamp).total_seconds() < ttl_seconds


class FeatureStore:
    """Simple in-memory feature store for caching computed features"""
    
    def __init__(self):
        self._features = {}
        self._metadata = {}
    
    def store_features(self, key: str, features: Dict[str, Any], metadata: Optional[Dict] = None):
        """Store features with optional metadata"""
        self._features[key] = features
        if metadata:
            self._metadata[key] = metadata
    
    def get_features(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve features by key"""
        return self._features.get(key)
    
    def get_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """Get metadata for stored features"""
        return self._metadata.get(key)
    
    def list_keys(self) -> List[str]:
        """List all stored feature keys"""
        return list(self._features.keys())
    
    def clear(self):
        """Clear all stored features"""
        self._features.clear()
        self._metadata.clear()


class ModelRegistry:
    """Simple model registry for tracking ML models"""
    
    def __init__(self):
        self._models = {}
        self._metadata = {}
    
    def register_model(
        self, 
        name: str, 
        model: Any, 
        version: str = "1.0",
        metadata: Optional[Dict] = None
    ):
        """Register a model in the registry"""
        key = f"{name}:{version}"
        self._models[key] = model
        self._metadata[key] = metadata or {}
        self._metadata[key]['registered_at'] = datetime.utcnow()
    
    def get_model(self, name: str, version: Optional[str] = None) -> Optional[Any]:
        """Get a model from the registry"""
        if version:
            key = f"{name}:{version}"
            return self._models.get(key)
        
        # Get latest version if no version specified
        model_keys = [k for k in self._models.keys() if k.startswith(f"{name}:")]
        if model_keys:
            # Sort by version and get latest
            latest_key = sorted(model_keys)[-1]
            return self._models[latest_key]
        
        return None
    
    def list_models(self) -> List[str]:
        """List all registered models"""
        return list(self._models.keys())
    
    def get_model_metadata(self, name: str, version: Optional[str] = None) -> Optional[Dict]:
        """Get model metadata"""
        if version:
            key = f"{name}:{version}"
            return self._metadata.get(key)
        
        # Get latest version metadata
        model_keys = [k for k in self._models.keys() if k.startswith(f"{name}:")]
        if model_keys:
            latest_key = sorted(model_keys)[-1]
            return self._metadata[latest_key]
        
        return None


# Export all utility classes
__all__ = [
    'MLMetrics',
    'DataValidator', 
    'TimeSeriesUtils',
    'FeatureUtils',
    'ModelUtils',
    'RiskUtils',
    'CacheUtils',
    'FeatureStore',
    'ModelRegistry'
]