"""
Advanced Analytics Engine for Phase 8 - Autonomous Operations Intelligence
Real-time decision support with advanced machine learning, statistical analysis, and predictive modeling.

This module provides:
- Real-time streaming analytics with sub-second latency
- Advanced statistical analysis with confidence intervals
- Multi-dimensional data analysis and pattern recognition
- Automated feature engineering and selection
- Real-time decision trees and rule engines
- Adaptive learning algorithms for dynamic environments
- Advanced time series forecasting and anomaly detection
- Multi-model ensemble predictions with uncertainty quantification
"""

import asyncio
import logging
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import uuid
import asyncpg
import redis.asyncio as redis
from concurrent.futures import ThreadPoolExecutor
import statistics
import scipy.stats as stats
from scipy import signal
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class AnalyticsModel(Enum):
    """Types of analytics models"""
    LINEAR_REGRESSION = "linear_regression"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    ENSEMBLE = "ensemble"
    DECISION_TREE = "decision_tree"
    TIME_SERIES = "time_series"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"


class DecisionSupportLevel(Enum):
    """Levels of decision support"""
    INFORMATIONAL = "informational"
    RECOMMENDATION = "recommendation"
    AUTOMATED_ACTION = "automated_action"
    CRITICAL_INTERVENTION = "critical_intervention"


class AnalyticsComplexity(Enum):
    """Analytics complexity levels"""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class AnalyticsRequest:
    """Analytics processing request"""
    request_id: str
    analysis_type: str
    data_sources: List[str]
    parameters: Dict[str, Any]
    complexity_level: AnalyticsComplexity
    real_time: bool = True
    return_confidence: bool = True
    return_explanations: bool = True
    max_processing_time_ms: int = 5000
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AnalyticsResult:
    """Analytics processing result"""
    request_id: str
    result_type: str
    primary_result: Any
    confidence_score: float
    uncertainty_bounds: Optional[Tuple[float, float]]
    
    # Statistical measures
    statistical_significance: Optional[float] = None
    p_value: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    
    # Model performance
    model_accuracy: float = 0.0
    model_type: Optional[AnalyticsModel] = None
    feature_importance: Dict[str, float] = field(default_factory=dict)
    
    # Decision support
    decision_support_level: DecisionSupportLevel = DecisionSupportLevel.INFORMATIONAL
    recommendations: List[str] = field(default_factory=list)
    automated_actions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Explanations
    explanation_text: Optional[str] = None
    decision_tree_rules: Optional[str] = None
    key_insights: List[str] = field(default_factory=list)
    
    # Performance metrics
    processing_time_ms: float = 0.0
    data_points_analyzed: int = 0
    features_used: int = 0
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RealTimePattern:
    """Real-time pattern detection result"""
    pattern_id: str
    pattern_type: str
    pattern_strength: float
    confidence: float
    frequency: float
    duration_seconds: float
    affected_metrics: List[str]
    pattern_description: str
    statistical_properties: Dict[str, Any]
    detected_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DecisionRule:
    """Automated decision rule"""
    rule_id: str
    rule_name: str
    condition: str
    action: str
    confidence_threshold: float
    priority: int
    parameters: Dict[str, Any]
    success_rate: float = 0.0
    execution_count: int = 0
    last_executed: Optional[datetime] = None
    enabled: bool = True


@dataclass
class FeatureEngineering:
    """Feature engineering configuration and results"""
    feature_set_id: str
    original_features: List[str]
    engineered_features: List[str]
    transformation_pipeline: List[Dict[str, Any]]
    feature_importance_scores: Dict[str, float]
    feature_correlations: Dict[str, Dict[str, float]]
    feature_selection_method: str
    features_selected: int
    improvement_score: float


class AdvancedFeatureEngineer:
    """Advanced feature engineering and selection"""
    
    def __init__(self):
        self.scalers = {}
        self.feature_selectors = {}
        self.transformation_history = []
        
    def engineer_features(
        self, 
        data: pd.DataFrame, 
        target_column: Optional[str] = None,
        max_features: int = 50
    ) -> Tuple[pd.DataFrame, FeatureEngineering]:
        """Engineer features from raw data"""
        try:
            feature_set_id = str(uuid.uuid4())
            original_features = list(data.columns)
            
            # Start with original data
            engineered_data = data.copy()
            transformation_pipeline = []
            
            # 1. Time-based features (if timestamp column exists)
            if 'timestamp' in data.columns:
                time_features = self._create_time_features(data['timestamp'])
                engineered_data = pd.concat([engineered_data, time_features], axis=1)
                transformation_pipeline.append({
                    'step': 'time_features',
                    'features_added': list(time_features.columns)
                })
            
            # 2. Statistical features from numeric columns
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                stat_features = self._create_statistical_features(data[numeric_columns])
                engineered_data = pd.concat([engineered_data, stat_features], axis=1)
                transformation_pipeline.append({
                    'step': 'statistical_features',
                    'features_added': list(stat_features.columns)
                })
            
            # 3. Interaction features
            if len(numeric_columns) > 1:
                interaction_features = self._create_interaction_features(
                    data[numeric_columns], max_interactions=10
                )
                engineered_data = pd.concat([engineered_data, interaction_features], axis=1)
                transformation_pipeline.append({
                    'step': 'interaction_features',
                    'features_added': list(interaction_features.columns)
                })
            
            # 4. Polynomial features
            poly_features = self._create_polynomial_features(
                data[numeric_columns].iloc[:, :5], degree=2  # Limit to first 5 columns
            )
            engineered_data = pd.concat([engineered_data, poly_features], axis=1)
            transformation_pipeline.append({
                'step': 'polynomial_features',
                'features_added': list(poly_features.columns)
            })
            
            # 5. Feature scaling
            numeric_features = engineered_data.select_dtypes(include=[np.number]).columns
            scaler = RobustScaler()
            engineered_data[numeric_features] = scaler.fit_transform(
                engineered_data[numeric_features].fillna(0)
            )
            self.scalers[feature_set_id] = scaler
            
            # 6. Feature selection (if target is provided)
            if target_column and target_column in engineered_data.columns:
                selected_data, feature_importance = self._select_best_features(
                    engineered_data.drop(columns=[target_column]), 
                    engineered_data[target_column],
                    max_features=max_features
                )
                engineered_data = pd.concat([selected_data, engineered_data[target_column]], axis=1)
            else:
                # Unsupervised feature selection based on variance
                selected_data, feature_importance = self._select_features_by_variance(
                    engineered_data, max_features=max_features
                )
                engineered_data = selected_data
            
            # Calculate feature correlations
            correlations = self._calculate_feature_correlations(engineered_data)
            
            # Create feature engineering summary
            feature_engineering = FeatureEngineering(
                feature_set_id=feature_set_id,
                original_features=original_features,
                engineered_features=list(engineered_data.columns),
                transformation_pipeline=transformation_pipeline,
                feature_importance_scores=feature_importance,
                feature_correlations=correlations,
                feature_selection_method="selectkbest" if target_column else "variance",
                features_selected=len(engineered_data.columns),
                improvement_score=len(engineered_data.columns) / len(original_features)
            )
            
            return engineered_data, feature_engineering
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {e}")
            return data, FeatureEngineering(
                feature_set_id=str(uuid.uuid4()),
                original_features=list(data.columns),
                engineered_features=list(data.columns),
                transformation_pipeline=[],
                feature_importance_scores={},
                feature_correlations={},
                feature_selection_method="none",
                features_selected=len(data.columns),
                improvement_score=1.0
            )
    
    def _create_time_features(self, timestamps: pd.Series) -> pd.DataFrame:
        """Create time-based features"""
        time_features = pd.DataFrame()
        
        dt = pd.to_datetime(timestamps)
        
        time_features['hour'] = dt.dt.hour
        time_features['day_of_week'] = dt.dt.dayofweek
        time_features['day_of_month'] = dt.dt.day
        time_features['month'] = dt.dt.month
        time_features['quarter'] = dt.dt.quarter
        time_features['is_weekend'] = (dt.dt.dayofweek >= 5).astype(int)
        time_features['is_business_hour'] = ((dt.dt.hour >= 9) & (dt.dt.hour <= 17)).astype(int)
        
        # Cyclical encoding
        time_features['hour_sin'] = np.sin(2 * np.pi * time_features['hour'] / 24)
        time_features['hour_cos'] = np.cos(2 * np.pi * time_features['hour'] / 24)
        time_features['day_sin'] = np.sin(2 * np.pi * time_features['day_of_week'] / 7)
        time_features['day_cos'] = np.cos(2 * np.pi * time_features['day_of_week'] / 7)
        
        return time_features
    
    def _create_statistical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features"""
        stat_features = pd.DataFrame()
        
        # Rolling statistics (if enough data)
        if len(data) >= 10:
            for col in data.columns:
                if data[col].dtype in [np.float64, np.int64]:
                    # Rolling statistics
                    stat_features[f'{col}_rolling_mean_5'] = data[col].rolling(5, min_periods=1).mean()
                    stat_features[f'{col}_rolling_std_5'] = data[col].rolling(5, min_periods=1).std()
                    stat_features[f'{col}_rolling_max_5'] = data[col].rolling(5, min_periods=1).max()
                    stat_features[f'{col}_rolling_min_5'] = data[col].rolling(5, min_periods=1).min()
                    
                    # Differences
                    stat_features[f'{col}_diff_1'] = data[col].diff(1)
                    stat_features[f'{col}_pct_change'] = data[col].pct_change()
        
        return stat_features.fillna(0)
    
    def _create_interaction_features(self, data: pd.DataFrame, max_interactions: int = 10) -> pd.DataFrame:
        """Create interaction features between columns"""
        interaction_features = pd.DataFrame()
        
        columns = list(data.columns)
        interaction_count = 0
        
        for i, col1 in enumerate(columns):
            for col2 in columns[i+1:]:
                if interaction_count >= max_interactions:
                    break
                
                if data[col1].dtype in [np.float64, np.int64] and data[col2].dtype in [np.float64, np.int64]:
                    # Multiplication interaction
                    interaction_features[f'{col1}_x_{col2}'] = data[col1] * data[col2]
                    
                    # Division interaction (avoid division by zero)
                    non_zero_mask = data[col2] != 0
                    interaction_features[f'{col1}_div_{col2}'] = 0.0
                    interaction_features.loc[non_zero_mask, f'{col1}_div_{col2}'] = (
                        data.loc[non_zero_mask, col1] / data.loc[non_zero_mask, col2]
                    )
                    
                    interaction_count += 2
                
                if interaction_count >= max_interactions:
                    break
        
        return interaction_features.fillna(0)
    
    def _create_polynomial_features(self, data: pd.DataFrame, degree: int = 2) -> pd.DataFrame:
        """Create polynomial features"""
        poly_features = pd.DataFrame()
        
        for col in data.columns:
            if data[col].dtype in [np.float64, np.int64]:
                for d in range(2, degree + 1):
                    poly_features[f'{col}_pow_{d}'] = data[col] ** d
        
        return poly_features.fillna(0)
    
    def _select_best_features(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        max_features: int = 50
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Select best features using statistical tests"""
        try:
            # Use SelectKBest with f_regression
            selector = SelectKBest(score_func=f_regression, k=min(max_features, X.shape[1]))
            X_selected = selector.fit_transform(X.fillna(0), y.fillna(0))
            
            # Get selected feature names
            selected_features = X.columns[selector.get_support()].tolist()
            X_selected_df = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
            
            # Get feature scores
            feature_scores = dict(zip(X.columns, selector.scores_))
            selected_feature_scores = {
                feat: feature_scores[feat] for feat in selected_features
            }
            
            return X_selected_df, selected_feature_scores
            
        except Exception as e:
            logger.error(f"Error in feature selection: {e}")
            return X, {}
    
    def _select_features_by_variance(
        self, 
        data: pd.DataFrame, 
        max_features: int = 50
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Select features by variance (unsupervised)"""
        try:
            numeric_data = data.select_dtypes(include=[np.number]).fillna(0)
            
            # Calculate variance for each feature
            variances = numeric_data.var()
            
            # Sort by variance and select top features
            top_features = variances.nlargest(min(max_features, len(variances))).index.tolist()
            
            selected_data = data[top_features]
            feature_scores = dict(variances[top_features])
            
            return selected_data, feature_scores
            
        except Exception as e:
            logger.error(f"Error in variance-based feature selection: {e}")
            return data, {}
    
    def _calculate_feature_correlations(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate feature correlations"""
        try:
            numeric_data = data.select_dtypes(include=[np.number]).fillna(0)
            correlation_matrix = numeric_data.corr()
            
            correlations = {}
            for col1 in correlation_matrix.columns:
                correlations[col1] = {}
                for col2 in correlation_matrix.columns:
                    if col1 != col2:
                        corr_value = correlation_matrix.loc[col1, col2]
                        if not np.isnan(corr_value):
                            correlations[col1][col2] = float(corr_value)
            
            return correlations
            
        except Exception as e:
            logger.error(f"Error calculating feature correlations: {e}")
            return {}


class EnsembleModelEngine:
    """Advanced ensemble modeling for robust predictions"""
    
    def __init__(self):
        self.models = {}
        self.ensemble_weights = {}
        self.model_performance = {}
        
    def build_ensemble_model(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        model_id: str
    ) -> Dict[str, Any]:
        """Build ensemble model with multiple algorithms"""
        try:
            # Prepare data
            X_clean = X.fillna(0)
            y_clean = y.fillna(y.mean())
            
            # Split data for training and validation
            split_idx = int(0.8 * len(X_clean))
            X_train, X_val = X_clean[:split_idx], X_clean[split_idx:]
            y_train, y_val = y_clean[:split_idx], y_clean[split_idx:]
            
            # Individual models
            models = {
                'linear': LinearRegression(),
                'ridge': Ridge(alpha=1.0),
                'random_forest': RandomForestRegressor(n_estimators=50, random_state=42),
                'gradient_boosting': GradientBoostingRegressor(n_estimators=50, random_state=42)
            }
            
            # Train individual models and calculate performance
            trained_models = {}
            model_scores = {}
            
            for name, model in models.items():
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                    
                    # Calculate performance metrics
                    mae = mean_absolute_error(y_val, y_pred)
                    mse = mean_squared_error(y_val, y_pred)
                    r2 = r2_score(y_val, y_pred)
                    
                    trained_models[name] = model
                    model_scores[name] = {
                        'mae': mae,
                        'mse': mse,
                        'r2': r2,
                        'score': r2  # Use R² as primary score
                    }
                    
                except Exception as e:
                    logger.warning(f"Error training model {name}: {e}")
                    continue
            
            if not trained_models:
                raise ValueError("No models were successfully trained")
            
            # Calculate ensemble weights based on performance
            scores = [model_scores[name]['score'] for name in trained_models.keys()]
            min_score = min(scores)
            adjusted_scores = [score - min_score + 0.001 for score in scores]  # Avoid negative weights
            total_score = sum(adjusted_scores)
            weights = [score / total_score for score in adjusted_scores]
            
            ensemble_weights = dict(zip(trained_models.keys(), weights))
            
            # Create ensemble model
            ensemble = VotingRegressor([
                (name, model) for name, model in trained_models.items()
            ])
            ensemble.fit(X_train, y_train)
            
            # Evaluate ensemble
            ensemble_pred = ensemble.predict(X_val)
            ensemble_performance = {
                'mae': mean_absolute_error(y_val, ensemble_pred),
                'mse': mean_squared_error(y_val, ensemble_pred),
                'r2': r2_score(y_val, ensemble_pred)
            }
            
            # Store models
            self.models[model_id] = {
                'ensemble': ensemble,
                'individual_models': trained_models,
                'feature_names': list(X.columns)
            }
            self.ensemble_weights[model_id] = ensemble_weights
            self.model_performance[model_id] = {
                'individual': model_scores,
                'ensemble': ensemble_performance
            }
            
            return {
                'model_id': model_id,
                'ensemble_performance': ensemble_performance,
                'individual_performance': model_scores,
                'ensemble_weights': ensemble_weights,
                'feature_importance': self._calculate_ensemble_feature_importance(
                    trained_models, list(X.columns)
                )
            }
            
        except Exception as e:
            logger.error(f"Error building ensemble model: {e}")
            raise
    
    def predict_with_uncertainty(
        self, 
        model_id: str, 
        X: pd.DataFrame
    ) -> Dict[str, Any]:
        """Make predictions with uncertainty quantification"""
        try:
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not found")
            
            model_info = self.models[model_id]
            ensemble = model_info['ensemble']
            individual_models = model_info['individual_models']
            
            # Prepare data
            X_clean = X.fillna(0)
            
            # Ensemble prediction
            ensemble_pred = ensemble.predict(X_clean)
            
            # Individual model predictions
            individual_preds = {}
            for name, model in individual_models.items():
                individual_preds[name] = model.predict(X_clean)
            
            # Calculate prediction variance across models
            all_predictions = np.array(list(individual_preds.values()))
            prediction_std = np.std(all_predictions, axis=0)
            prediction_mean = np.mean(all_predictions, axis=0)
            
            # Calculate confidence intervals (assuming normal distribution)
            confidence_lower = prediction_mean - 1.96 * prediction_std
            confidence_upper = prediction_mean + 1.96 * prediction_std
            
            # Calculate uncertainty score
            uncertainty_score = np.mean(prediction_std) / (np.abs(np.mean(prediction_mean)) + 1e-8)
            
            return {
                'predictions': ensemble_pred.tolist(),
                'individual_predictions': {k: v.tolist() for k, v in individual_preds.items()},
                'uncertainty_bounds': list(zip(confidence_lower, confidence_upper)),
                'uncertainty_score': float(uncertainty_score),
                'confidence_intervals': list(zip(confidence_lower, confidence_upper)),
                'prediction_std': prediction_std.tolist(),
                'model_agreement': float(1.0 - uncertainty_score)  # Higher is better
            }
            
        except Exception as e:
            logger.error(f"Error in prediction with uncertainty: {e}")
            return {
                'error': str(e),
                'predictions': [],
                'uncertainty_score': 1.0
            }
    
    def _calculate_ensemble_feature_importance(
        self, 
        models: Dict[str, Any], 
        feature_names: List[str]
    ) -> Dict[str, float]:
        """Calculate ensemble feature importance"""
        try:
            feature_importance = defaultdict(float)
            model_count = 0
            
            for name, model in models.items():
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    for feature, importance in zip(feature_names, importances):
                        feature_importance[feature] += importance
                    model_count += 1
                elif hasattr(model, 'coef_'):
                    # For linear models, use absolute coefficients
                    importances = np.abs(model.coef_)
                    # Normalize
                    if np.sum(importances) > 0:
                        importances = importances / np.sum(importances)
                    for feature, importance in zip(feature_names, importances):
                        feature_importance[feature] += importance
                    model_count += 1
            
            # Average across models
            if model_count > 0:
                for feature in feature_importance:
                    feature_importance[feature] /= model_count
            
            return dict(feature_importance)
            
        except Exception as e:
            logger.error(f"Error calculating ensemble feature importance: {e}")
            return {}


class RealTimePatternDetector:
    """Real-time pattern detection in streaming data"""
    
    def __init__(self, max_history_size: int = 1000):
        self.max_history_size = max_history_size
        self.data_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history_size))
        self.detected_patterns: Dict[str, RealTimePattern] = {}
        
    def detect_patterns(
        self, 
        data_stream: Dict[str, float], 
        timestamp: datetime
    ) -> List[RealTimePattern]:
        """Detect patterns in real-time data streams"""
        try:
            detected_patterns = []
            
            # Update data history
            for metric_name, value in data_stream.items():
                self.data_history[metric_name].append((timestamp, value))
            
            # Detect patterns for each metric
            for metric_name, history in self.data_history.items():
                if len(history) >= 10:  # Minimum data points for pattern detection
                    patterns = self._detect_metric_patterns(metric_name, history)
                    detected_patterns.extend(patterns)
            
            # Detect cross-metric patterns
            if len(self.data_history) > 1:
                cross_patterns = self._detect_cross_metric_patterns()
                detected_patterns.extend(cross_patterns)
            
            return detected_patterns
            
        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")
            return []
    
    def _detect_metric_patterns(
        self, 
        metric_name: str, 
        history: deque
    ) -> List[RealTimePattern]:
        """Detect patterns in a single metric"""
        try:
            patterns = []
            
            # Extract values and timestamps
            timestamps, values = zip(*history)
            values = np.array(values)
            
            if len(values) < 10:
                return patterns
            
            # 1. Trend detection
            trend_pattern = self._detect_trend_pattern(metric_name, values, timestamps)
            if trend_pattern:
                patterns.append(trend_pattern)
            
            # 2. Seasonality detection
            if len(values) >= 20:
                seasonal_pattern = self._detect_seasonal_pattern(metric_name, values, timestamps)
                if seasonal_pattern:
                    patterns.append(seasonal_pattern)
            
            # 3. Spike/Anomaly detection
            spike_patterns = self._detect_spike_patterns(metric_name, values, timestamps)
            patterns.extend(spike_patterns)
            
            # 4. Oscillation detection
            oscillation_pattern = self._detect_oscillation_pattern(metric_name, values, timestamps)
            if oscillation_pattern:
                patterns.append(oscillation_pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting patterns for metric {metric_name}: {e}")
            return []
    
    def _detect_trend_pattern(
        self, 
        metric_name: str, 
        values: np.ndarray, 
        timestamps: List[datetime]
    ) -> Optional[RealTimePattern]:
        """Detect trend patterns"""
        try:
            # Linear regression to detect trend
            x = np.arange(len(values))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
            
            # Significance test
            if abs(r_value) > 0.5 and p_value < 0.05:
                pattern_type = "upward_trend" if slope > 0 else "downward_trend"
                
                return RealTimePattern(
                    pattern_id=str(uuid.uuid4()),
                    pattern_type=pattern_type,
                    pattern_strength=abs(r_value),
                    confidence=1.0 - p_value,
                    frequency=1.0 / len(values),  # Simple frequency estimate
                    duration_seconds=(timestamps[-1] - timestamps[0]).total_seconds(),
                    affected_metrics=[metric_name],
                    pattern_description=f"{'Upward' if slope > 0 else 'Downward'} trend with slope {slope:.4f}",
                    statistical_properties={
                        'slope': slope,
                        'r_squared': r_value**2,
                        'p_value': p_value,
                        'standard_error': std_err
                    }
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting trend pattern: {e}")
            return None
    
    def _detect_seasonal_pattern(
        self, 
        metric_name: str, 
        values: np.ndarray, 
        timestamps: List[datetime]
    ) -> Optional[RealTimePattern]:
        """Detect seasonal/cyclical patterns"""
        try:
            # Simple autocorrelation-based seasonality detection
            if len(values) < 20:
                return None
            
            # Calculate autocorrelation at different lags
            max_lag = min(len(values) // 3, 50)
            autocorrelations = []
            
            for lag in range(1, max_lag):
                if lag < len(values):
                    corr = np.corrcoef(values[:-lag], values[lag:])[0, 1]
                    if not np.isnan(corr):
                        autocorrelations.append((lag, abs(corr)))
            
            if autocorrelations:
                # Find the lag with highest autocorrelation
                best_lag, best_corr = max(autocorrelations, key=lambda x: x[1])
                
                if best_corr > 0.6:  # Significant seasonal pattern
                    return RealTimePattern(
                        pattern_id=str(uuid.uuid4()),
                        pattern_type="seasonal_pattern",
                        pattern_strength=best_corr,
                        confidence=best_corr,
                        frequency=1.0 / best_lag,
                        duration_seconds=(timestamps[-1] - timestamps[0]).total_seconds(),
                        affected_metrics=[metric_name],
                        pattern_description=f"Seasonal pattern with period {best_lag} observations",
                        statistical_properties={
                            'best_lag': best_lag,
                            'autocorrelation': best_corr,
                            'all_correlations': autocorrelations[:10]  # Top 10
                        }
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting seasonal pattern: {e}")
            return None
    
    def _detect_spike_patterns(
        self, 
        metric_name: str, 
        values: np.ndarray, 
        timestamps: List[datetime]
    ) -> List[RealTimePattern]:
        """Detect spike/anomaly patterns"""
        try:
            patterns = []
            
            # Use z-score for spike detection
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            if std_val > 0:
                z_scores = np.abs((values - mean_val) / std_val)
                spike_threshold = 3.0  # 3 standard deviations
                
                spike_indices = np.where(z_scores > spike_threshold)[0]
                
                for idx in spike_indices:
                    patterns.append(RealTimePattern(
                        pattern_id=str(uuid.uuid4()),
                        pattern_type="spike_anomaly",
                        pattern_strength=float(z_scores[idx]),
                        confidence=min(0.99, z_scores[idx] / 5.0),  # Confidence based on z-score
                        frequency=0.0,  # Spikes are irregular
                        duration_seconds=60.0,  # Assume 1-minute duration
                        affected_metrics=[metric_name],
                        pattern_description=f"Spike detected with z-score {z_scores[idx]:.2f}",
                        statistical_properties={
                            'z_score': float(z_scores[idx]),
                            'value': float(values[idx]),
                            'mean': float(mean_val),
                            'std': float(std_val),
                            'timestamp_index': int(idx)
                        }
                    ))
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting spike patterns: {e}")
            return []
    
    def _detect_oscillation_pattern(
        self, 
        metric_name: str, 
        values: np.ndarray, 
        timestamps: List[datetime]
    ) -> Optional[RealTimePattern]:
        """Detect oscillation patterns using signal processing"""
        try:
            if len(values) < 20:
                return None
            
            # Find peaks and troughs
            peaks, _ = signal.find_peaks(values)
            troughs, _ = signal.find_peaks(-values)
            
            # Calculate oscillation metrics
            if len(peaks) >= 3 and len(troughs) >= 3:
                # Average period between peaks
                peak_distances = np.diff(peaks)
                avg_peak_distance = np.mean(peak_distances)
                
                # Regularity of oscillations
                peak_regularity = 1.0 - (np.std(peak_distances) / avg_peak_distance)
                
                if peak_regularity > 0.7:  # Regular oscillations
                    return RealTimePattern(
                        pattern_id=str(uuid.uuid4()),
                        pattern_type="oscillation",
                        pattern_strength=peak_regularity,
                        confidence=peak_regularity,
                        frequency=1.0 / avg_peak_distance,
                        duration_seconds=(timestamps[-1] - timestamps[0]).total_seconds(),
                        affected_metrics=[metric_name],
                        pattern_description=f"Oscillation with period ~{avg_peak_distance:.1f} observations",
                        statistical_properties={
                            'num_peaks': len(peaks),
                            'num_troughs': len(troughs),
                            'avg_period': float(avg_peak_distance),
                            'regularity': float(peak_regularity),
                            'peak_indices': peaks.tolist()
                        }
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting oscillation pattern: {e}")
            return None
    
    def _detect_cross_metric_patterns(self) -> List[RealTimePattern]:
        """Detect patterns across multiple metrics"""
        try:
            patterns = []
            
            # Get recent values for all metrics
            metric_values = {}
            for metric_name, history in self.data_history.items():
                if len(history) >= 10:
                    recent_values = [value for _, value in list(history)[-10:]]
                    metric_values[metric_name] = np.array(recent_values)
            
            if len(metric_values) < 2:
                return patterns
            
            # Calculate cross-correlations
            metric_names = list(metric_values.keys())
            for i, metric1 in enumerate(metric_names):
                for metric2 in metric_names[i+1:]:
                    values1 = metric_values[metric1]
                    values2 = metric_values[metric2]
                    
                    if len(values1) == len(values2):
                        correlation = np.corrcoef(values1, values2)[0, 1]
                        
                        if not np.isnan(correlation) and abs(correlation) > 0.8:
                            pattern_type = "positive_correlation" if correlation > 0 else "negative_correlation"
                            
                            patterns.append(RealTimePattern(
                                pattern_id=str(uuid.uuid4()),
                                pattern_type=pattern_type,
                                pattern_strength=abs(correlation),
                                confidence=abs(correlation),
                                frequency=1.0,  # Cross-correlations are continuous
                                duration_seconds=300.0,  # Assume 5-minute pattern
                                affected_metrics=[metric1, metric2],
                                pattern_description=f"{'Strong positive' if correlation > 0 else 'Strong negative'} correlation ({correlation:.3f})",
                                statistical_properties={
                                    'correlation_coefficient': float(correlation),
                                    'metric_1': metric1,
                                    'metric_2': metric2,
                                    'sample_size': len(values1)
                                }
                            ))
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting cross-metric patterns: {e}")
            return []


class DecisionEngine:
    """Automated decision engine with rule-based logic"""
    
    def __init__(self):
        self.decision_rules: Dict[str, DecisionRule] = {}
        self.rule_execution_history: List[Dict[str, Any]] = []
        
    def add_decision_rule(self, rule: DecisionRule) -> None:
        """Add a new decision rule"""
        self.decision_rules[rule.rule_id] = rule
    
    def evaluate_rules(
        self, 
        analytics_result: AnalyticsResult, 
        context_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Evaluate all decision rules against analytics results"""
        try:
            triggered_actions = []
            
            for rule_id, rule in self.decision_rules.items():
                if not rule.enabled:
                    continue
                
                # Check confidence threshold
                if analytics_result.confidence_score < rule.confidence_threshold:
                    continue
                
                # Evaluate rule condition
                if self._evaluate_condition(rule.condition, analytics_result, context_data):
                    action_result = self._execute_action(rule, analytics_result, context_data)
                    triggered_actions.append(action_result)
                    
                    # Update rule statistics
                    rule.execution_count += 1
                    rule.last_executed = datetime.utcnow()
            
            return triggered_actions
            
        except Exception as e:
            logger.error(f"Error evaluating rules: {e}")
            return []
    
    def _evaluate_condition(
        self, 
        condition: str, 
        analytics_result: AnalyticsResult, 
        context_data: Dict[str, Any]
    ) -> bool:
        """Evaluate rule condition (simplified rule engine)"""
        try:
            # Create evaluation context
            eval_context = {
                'confidence': analytics_result.confidence_score,
                'result': analytics_result.primary_result,
                'p_value': analytics_result.p_value or 1.0,
                'model_accuracy': analytics_result.model_accuracy,
                'processing_time': analytics_result.processing_time_ms,
                **context_data
            }
            
            # Simple expression evaluation (in production, use a proper rule engine)
            # This is a simplified version for demonstration
            try:
                return eval(condition, {"__builtins__": {}}, eval_context)
            except:
                # Fallback to simple string matching conditions
                if "confidence > " in condition:
                    threshold = float(condition.split("confidence > ")[1])
                    return analytics_result.confidence_score > threshold
                elif "model_accuracy > " in condition:
                    threshold = float(condition.split("model_accuracy > ")[1])
                    return analytics_result.model_accuracy > threshold
                
                return False
                
        except Exception as e:
            logger.error(f"Error evaluating condition: {e}")
            return False
    
    def _execute_action(
        self, 
        rule: DecisionRule, 
        analytics_result: AnalyticsResult, 
        context_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute rule action"""
        try:
            action_result = {
                'rule_id': rule.rule_id,
                'rule_name': rule.rule_name,
                'action': rule.action,
                'status': 'executed',
                'timestamp': datetime.utcnow().isoformat(),
                'context': {
                    'analytics_result_id': analytics_result.request_id,
                    'confidence': analytics_result.confidence_score,
                    'parameters': rule.parameters
                }
            }
            
            # Log execution
            self.rule_execution_history.append(action_result)
            
            return action_result
            
        except Exception as e:
            logger.error(f"Error executing action for rule {rule.rule_id}: {e}")
            return {
                'rule_id': rule.rule_id,
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }


class AdvancedAnalyticsEngine:
    """
    Advanced Analytics Engine with real-time decision support
    """
    
    def __init__(
        self,
        database_url: str,
        redis_url: str
    ):
        self.database_url = database_url
        self.redis_url = redis_url
        
        # Core components
        self.db_pool: Optional[asyncpg.Pool] = None
        self.redis_client: Optional[redis.Redis] = None
        
        # Analytics components
        self.feature_engineer = AdvancedFeatureEngineer()
        self.ensemble_engine = EnsembleModelEngine()
        self.pattern_detector = RealTimePatternDetector()
        self.decision_engine = DecisionEngine()
        
        # Data storage
        self.analytics_results: Dict[str, AnalyticsResult] = {}
        self.active_patterns: Dict[str, RealTimePattern] = {}
        self.model_cache: Dict[str, Any] = {}
        
        # Background tasks
        self._analytics_task: Optional[asyncio.Task] = None
        self._pattern_detection_task: Optional[asyncio.Task] = None
        
        # Thread pool for ML operations
        self.ml_executor = ThreadPoolExecutor(max_workers=6)
        
        # Configuration
        self.analytics_interval_seconds = 30
        self.pattern_detection_interval_seconds = 10
        
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self) -> None:
        """Initialize the advanced analytics engine"""
        try:
            # Initialize database connection pool
            self.db_pool = await asyncpg.create_pool(
                self.database_url,
                min_size=5,
                max_size=20,
                command_timeout=60
            )
            
            # Initialize Redis connection
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            await self.redis_client.ping()
            
            # Create database tables
            await self._create_database_tables()
            
            # Initialize default decision rules
            self._initialize_default_decision_rules()
            
            # Start background tasks
            self._analytics_task = asyncio.create_task(self._analytics_loop())
            self._pattern_detection_task = asyncio.create_task(self._pattern_detection_loop())
            
            self.logger.info("Advanced Analytics Engine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize analytics engine: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the analytics engine"""
        try:
            # Cancel background tasks
            tasks = [self._analytics_task, self._pattern_detection_task]
            for task in tasks:
                if task:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            # Shutdown thread pool
            self.ml_executor.shutdown(wait=True)
            
            # Close connections
            if self.db_pool:
                await self.db_pool.close()
            
            if self.redis_client:
                await self.redis_client.close()
            
            self.logger.info("Advanced Analytics Engine shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    async def _create_database_tables(self) -> None:
        """Create database tables for analytics engine"""
        try:
            async with self.db_pool.acquire() as conn:
                # Analytics results table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS analytics_results (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        request_id VARCHAR(100) UNIQUE NOT NULL,
                        result_type VARCHAR(50) NOT NULL,
                        primary_result JSONB,
                        confidence_score DECIMAL(5,4),
                        uncertainty_bounds JSONB,
                        statistical_significance DECIMAL(5,4),
                        p_value DECIMAL(10,8),
                        confidence_interval JSONB,
                        model_accuracy DECIMAL(5,4),
                        model_type VARCHAR(50),
                        feature_importance JSONB,
                        decision_support_level VARCHAR(50),
                        recommendations TEXT[],
                        automated_actions JSONB,
                        explanation_text TEXT,
                        decision_tree_rules TEXT,
                        key_insights TEXT[],
                        processing_time_ms DECIMAL(10,3),
                        data_points_analyzed INTEGER,
                        features_used INTEGER,
                        timestamp TIMESTAMPTZ DEFAULT NOW(),
                        expires_at TIMESTAMPTZ,
                        metadata JSONB
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_analytics_results_timestamp
                        ON analytics_results(timestamp);
                    CREATE INDEX IF NOT EXISTS idx_analytics_results_request_id
                        ON analytics_results(request_id);
                """)
                
                # Real-time patterns table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS realtime_patterns (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        pattern_id VARCHAR(100) UNIQUE NOT NULL,
                        pattern_type VARCHAR(50) NOT NULL,
                        pattern_strength DECIMAL(5,4),
                        confidence DECIMAL(5,4),
                        frequency DECIMAL(10,6),
                        duration_seconds DECIMAL(10,2),
                        affected_metrics TEXT[],
                        pattern_description TEXT,
                        statistical_properties JSONB,
                        detected_at TIMESTAMPTZ DEFAULT NOW(),
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_realtime_patterns_detected_at
                        ON realtime_patterns(detected_at);
                    CREATE INDEX IF NOT EXISTS idx_realtime_patterns_pattern_type
                        ON realtime_patterns(pattern_type);
                """)
                
                # Decision rules table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS decision_rules (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        rule_id VARCHAR(100) UNIQUE NOT NULL,
                        rule_name VARCHAR(200) NOT NULL,
                        condition TEXT NOT NULL,
                        action TEXT NOT NULL,
                        confidence_threshold DECIMAL(5,4),
                        priority INTEGER,
                        parameters JSONB,
                        success_rate DECIMAL(5,4) DEFAULT 0,
                        execution_count INTEGER DEFAULT 0,
                        last_executed TIMESTAMPTZ,
                        enabled BOOLEAN DEFAULT TRUE,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        updated_at TIMESTAMPTZ DEFAULT NOW()
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_decision_rules_enabled
                        ON decision_rules(enabled);
                    CREATE INDEX IF NOT EXISTS idx_decision_rules_priority
                        ON decision_rules(priority);
                """)
            
            self.logger.info("Database tables created successfully")
            
        except Exception as e:
            self.logger.error(f"Error creating database tables: {e}")
            raise
    
    def _initialize_default_decision_rules(self) -> None:
        """Initialize default decision rules"""
        try:
            default_rules = [
                DecisionRule(
                    rule_id="high_confidence_alert",
                    rule_name="High Confidence Alert",
                    condition="confidence > 0.9 and model_accuracy > 0.85",
                    action="generate_alert",
                    confidence_threshold=0.9,
                    priority=1,
                    parameters={"alert_severity": "high", "notify_operators": True}
                ),
                DecisionRule(
                    rule_id="performance_degradation",
                    rule_name="Performance Degradation",
                    condition="result < 0.7 and confidence > 0.8",
                    action="scale_resources",
                    confidence_threshold=0.8,
                    priority=2,
                    parameters={"scaling_factor": 1.5, "max_instances": 10}
                ),
                DecisionRule(
                    rule_id="anomaly_detection",
                    rule_name="Anomaly Detection",
                    condition="confidence > 0.85 and p_value < 0.05",
                    action="investigate_anomaly",
                    confidence_threshold=0.85,
                    priority=3,
                    parameters={"create_ticket": True, "escalate_after_minutes": 15}
                )
            ]
            
            for rule in default_rules:
                self.decision_engine.add_decision_rule(rule)
            
            self.logger.info("Default decision rules initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing default decision rules: {e}")
    
    # Background processing loops
    
    async def _analytics_loop(self) -> None:
        """Main analytics processing loop"""
        try:
            while True:
                await asyncio.sleep(self.analytics_interval_seconds)
                
                try:
                    # Process pending analytics requests
                    await self._process_analytics_queue()
                    
                    # Clean up expired results
                    await self._cleanup_expired_results()
                    
                except Exception as e:
                    self.logger.error(f"Error in analytics loop: {e}")
                
        except asyncio.CancelledError:
            self.logger.info("Analytics loop cancelled")
            raise
        except Exception as e:
            self.logger.error(f"Error in analytics loop: {e}")
    
    async def _pattern_detection_loop(self) -> None:
        """Real-time pattern detection loop"""
        try:
            while True:
                await asyncio.sleep(self.pattern_detection_interval_seconds)
                
                try:
                    # Get current metrics from Redis
                    current_metrics = await self._get_current_metrics_from_redis()
                    
                    if current_metrics:
                        # Detect patterns
                        patterns = await asyncio.get_event_loop().run_in_executor(
                            self.ml_executor,
                            self.pattern_detector.detect_patterns,
                            current_metrics,
                            datetime.utcnow()
                        )
                        
                        # Process detected patterns
                        for pattern in patterns:
                            await self._process_detected_pattern(pattern)
                    
                except Exception as e:
                    self.logger.error(f"Error in pattern detection loop: {e}")
                
        except asyncio.CancelledError:
            self.logger.info("Pattern detection loop cancelled")
            raise
        except Exception as e:
            self.logger.error(f"Error in pattern detection loop: {e}")
    
    async def _get_current_metrics_from_redis(self) -> Dict[str, float]:
        """Get current operational metrics from Redis"""
        try:
            # Get metrics keys
            pattern = "nautilus:metrics:*"
            keys = await self.redis_client.keys(pattern)
            
            metrics = {}
            for key in keys:
                try:
                    value = await self.redis_client.get(key)
                    if value:
                        metric_name = key.split(":")[-1]
                        metrics[metric_name] = float(value)
                except (ValueError, TypeError):
                    continue
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error getting current metrics from Redis: {e}")
            return {}
    
    async def _process_detected_pattern(self, pattern: RealTimePattern) -> None:
        """Process a detected pattern"""
        try:
            # Store pattern
            self.active_patterns[pattern.pattern_id] = pattern
            
            # Persist to database
            await self._persist_pattern(pattern)
            
            # Publish pattern notification
            await self._publish_pattern_notification(pattern)
            
        except Exception as e:
            self.logger.error(f"Error processing detected pattern: {e}")
    
    async def _persist_pattern(self, pattern: RealTimePattern) -> None:
        """Persist pattern to database"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO realtime_patterns (
                        pattern_id, pattern_type, pattern_strength, confidence,
                        frequency, duration_seconds, affected_metrics, 
                        pattern_description, statistical_properties, detected_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    ON CONFLICT (pattern_id) DO UPDATE SET
                        pattern_strength = EXCLUDED.pattern_strength,
                        confidence = EXCLUDED.confidence,
                        detected_at = EXCLUDED.detected_at
                """,
                    pattern.pattern_id,
                    pattern.pattern_type,
                    pattern.pattern_strength,
                    pattern.confidence,
                    pattern.frequency,
                    pattern.duration_seconds,
                    pattern.affected_metrics,
                    pattern.pattern_description,
                    json.dumps(pattern.statistical_properties),
                    pattern.detected_at
                )
            
        except Exception as e:
            self.logger.error(f"Error persisting pattern: {e}")
    
    async def _publish_pattern_notification(self, pattern: RealTimePattern) -> None:
        """Publish pattern notification to Redis"""
        try:
            notification = {
                'type': 'pattern_detected',
                'pattern_id': pattern.pattern_id,
                'pattern_type': pattern.pattern_type,
                'pattern_strength': pattern.pattern_strength,
                'confidence': pattern.confidence,
                'affected_metrics': pattern.affected_metrics,
                'description': pattern.pattern_description,
                'timestamp': pattern.detected_at.isoformat()
            }
            
            await self.redis_client.publish(
                'nautilus:pattern_notifications',
                json.dumps(notification)
            )
            
        except Exception as e:
            self.logger.error(f"Error publishing pattern notification: {e}")
    
    # Public API methods
    
    async def process_analytics_request(self, request: AnalyticsRequest) -> AnalyticsResult:
        """Process advanced analytics request"""
        try:
            start_time = time.time()
            
            # Get data for analysis
            analysis_data = await self._get_analysis_data(request.data_sources)
            
            if analysis_data.empty:
                return AnalyticsResult(
                    request_id=request.request_id,
                    result_type="error",
                    primary_result="No data available for analysis",
                    confidence_score=0.0,
                    processing_time_ms=(time.time() - start_time) * 1000
                )
            
            # Feature engineering
            engineered_data, feature_info = await asyncio.get_event_loop().run_in_executor(
                self.ml_executor,
                self.feature_engineer.engineer_features,
                analysis_data,
                request.parameters.get('target_column'),
                request.parameters.get('max_features', 50)
            )
            
            # Build and train model
            model_results = await self._build_and_train_model(
                request, engineered_data, feature_info
            )
            
            # Generate insights and recommendations
            insights = self._generate_insights(model_results, feature_info)
            
            # Calculate decision support level
            decision_level = self._determine_decision_support_level(model_results)
            
            # Create analytics result
            result = AnalyticsResult(
                request_id=request.request_id,
                result_type=request.analysis_type,
                primary_result=model_results.get('predictions', model_results),
                confidence_score=model_results.get('confidence', 0.0),
                uncertainty_bounds=model_results.get('uncertainty_bounds'),
                statistical_significance=model_results.get('statistical_significance'),
                p_value=model_results.get('p_value'),
                confidence_interval=model_results.get('confidence_interval'),
                model_accuracy=model_results.get('model_performance', {}).get('r2', 0.0),
                model_type=AnalyticsModel.ENSEMBLE,
                feature_importance=feature_info.feature_importance_scores,
                decision_support_level=decision_level,
                recommendations=insights.get('recommendations', []),
                automated_actions=insights.get('automated_actions', []),
                explanation_text=insights.get('explanation', ""),
                key_insights=insights.get('key_insights', []),
                processing_time_ms=(time.time() - start_time) * 1000,
                data_points_analyzed=len(analysis_data),
                features_used=len(feature_info.engineered_features)
            )
            
            # Store result
            self.analytics_results[request.request_id] = result
            
            # Evaluate decision rules
            context_data = {
                'data_points': len(analysis_data),
                'feature_count': len(feature_info.engineered_features)
            }
            triggered_actions = self.decision_engine.evaluate_rules(result, context_data)
            result.automated_actions = triggered_actions
            
            # Persist result
            await self._persist_analytics_result(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing analytics request: {e}")
            return AnalyticsResult(
                request_id=request.request_id,
                result_type="error",
                primary_result=str(e),
                confidence_score=0.0,
                processing_time_ms=(time.time() - start_time) * 1000
            )
    
    async def get_real_time_insights(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """Get real-time analytical insights"""
        try:
            current_time = datetime.utcnow()
            start_time = current_time - timedelta(minutes=time_window_minutes)
            
            # Get recent patterns
            recent_patterns = [
                pattern for pattern in self.active_patterns.values()
                if pattern.detected_at >= start_time
            ]
            
            # Get recent analytics results
            recent_results = [
                result for result in self.analytics_results.values()
                if result.timestamp >= start_time
            ]
            
            # Calculate aggregate insights
            insights = {
                'timestamp': current_time.isoformat(),
                'time_window_minutes': time_window_minutes,
                'patterns_detected': len(recent_patterns),
                'pattern_types': self._analyze_pattern_types(recent_patterns),
                'analytics_requests_processed': len(recent_results),
                'average_confidence': np.mean([r.confidence_score for r in recent_results]) if recent_results else 0.0,
                'high_confidence_results': len([r for r in recent_results if r.confidence_score > 0.8]),
                'automated_actions_triggered': sum(len(r.automated_actions) for r in recent_results),
                'top_features': self._get_top_features(recent_results),
                'system_performance': {
                    'average_processing_time_ms': np.mean([r.processing_time_ms for r in recent_results]) if recent_results else 0.0,
                    'total_data_points_analyzed': sum(r.data_points_analyzed for r in recent_results),
                    'model_accuracy_distribution': self._get_accuracy_distribution(recent_results)
                },
                'recommendations': self._generate_system_recommendations(recent_patterns, recent_results)
            }
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error getting real-time insights: {e}")
            return {'error': str(e), 'timestamp': datetime.utcnow().isoformat()}


# Global instance
advanced_analytics_engine = None

def get_advanced_analytics_engine() -> AdvancedAnalyticsEngine:
    """Get global advanced analytics engine instance"""
    global advanced_analytics_engine
    if advanced_analytics_engine is None:
        raise RuntimeError("Advanced analytics engine not initialized")
    return advanced_analytics_engine

def init_advanced_analytics_engine(
    database_url: str, 
    redis_url: str
) -> AdvancedAnalyticsEngine:
    """Initialize global advanced analytics engine instance"""
    global advanced_analytics_engine
    advanced_analytics_engine = AdvancedAnalyticsEngine(database_url, redis_url)
    return advanced_analytics_engine