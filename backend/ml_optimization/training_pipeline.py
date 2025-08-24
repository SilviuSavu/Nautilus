"""
ML Training and Retraining Pipeline

This module implements automated training and retraining of ML models
for predictive auto-scaling and performance optimization, with continuous
learning from trading performance and market conditions.
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
import joblib
from pathlib import Path
import hashlib

# ML imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')


class ModelType(Enum):
    """Types of ML models for different tasks"""
    LOAD_PREDICTOR = "load_predictor"
    PATTERN_CLASSIFIER = "pattern_classifier"
    VOLATILITY_PREDICTOR = "volatility_predictor"
    REGIME_CLASSIFIER = "regime_classifier"
    PERFORMANCE_OPTIMIZER = "performance_optimizer"


class TrainingMode(Enum):
    """Training modes for different scenarios"""
    INITIAL_TRAINING = "initial_training"
    INCREMENTAL_UPDATE = "incremental_update"
    FULL_RETRAIN = "full_retrain"
    A_B_TEST = "a_b_test"
    EMERGENCY_RETRAIN = "emergency_retrain"


class ModelStatus(Enum):
    """Model status lifecycle"""
    TRAINING = "training"
    VALIDATION = "validation"
    TESTING = "testing"
    PRODUCTION = "production"
    CHALLENGER = "challenger"
    RETIRED = "retired"
    FAILED = "failed"


@dataclass
class ModelMetrics:
    """Performance metrics for ML models"""
    model_type: ModelType
    model_version: str
    timestamp: datetime
    
    # Training metrics
    train_mse: float
    train_mae: float
    train_r2: float
    
    # Validation metrics
    val_mse: float
    val_mae: float
    val_r2: float
    
    # Cross-validation metrics
    cv_mean: float
    cv_std: float
    
    # Production metrics (if available)
    prod_accuracy: Optional[float] = None
    prod_latency: Optional[float] = None
    drift_score: Optional[float] = None
    
    # Feature importance
    feature_importance: Dict[str, float] = field(default_factory=dict)


@dataclass
class TrainingJob:
    """Training job specification"""
    job_id: str
    model_type: ModelType
    training_mode: TrainingMode
    priority: int  # 1-10, higher is more urgent
    
    # Data specifications
    data_start_date: datetime
    data_end_date: datetime
    min_samples: int = 1000
    max_samples: int = 100000
    
    # Training parameters
    algorithms: List[str] = field(default_factory=lambda: ['random_forest', 'gradient_boosting'])
    hyperparameter_tuning: bool = True
    cross_validation_folds: int = 5
    
    # Validation requirements
    min_val_r2: float = 0.3
    max_val_mse: float = 1.0
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "pending"
    error_message: Optional[str] = None


class MLTrainingPipeline:
    """
    Automated ML training and retraining pipeline for optimizing
    trading system performance based on historical data and outcomes.
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379", model_storage_path: str = "/tmp/models"):
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.logger = logging.getLogger(__name__)
        
        # Model storage
        self.model_storage_path = Path(model_storage_path)
        self.model_storage_path.mkdir(parents=True, exist_ok=True)
        
        # Active models
        self.production_models: Dict[ModelType, Any] = {}
        self.challenger_models: Dict[ModelType, Any] = {}
        self.model_metrics: Dict[str, ModelMetrics] = {}
        
        # Training queue
        self.training_queue: List[TrainingJob] = []
        self.active_jobs: Dict[str, TrainingJob] = {}
        
        # Configuration
        self.retrain_schedule = {
            ModelType.LOAD_PREDICTOR: timedelta(hours=6),
            ModelType.PATTERN_CLASSIFIER: timedelta(hours=12),
            ModelType.VOLATILITY_PREDICTOR: timedelta(hours=8),
            ModelType.REGIME_CLASSIFIER: timedelta(days=1),
            ModelType.PERFORMANCE_OPTIMIZER: timedelta(hours=4)
        }
        
        self.drift_thresholds = {
            ModelType.LOAD_PREDICTOR: 0.05,
            ModelType.PATTERN_CLASSIFIER: 0.1,
            ModelType.VOLATILITY_PREDICTOR: 0.08,
            ModelType.REGIME_CLASSIFIER: 0.15,
            ModelType.PERFORMANCE_OPTIMIZER: 0.06
        }
        
        # Algorithm configurations
        self.algorithm_configs = {
            'random_forest': {
                'class': RandomForestRegressor,
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'gradient_boosting': {
                'class': GradientBoostingRegressor,
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'ridge': {
                'class': Ridge,
                'params': {
                    'alpha': [0.1, 1.0, 10.0, 100.0],
                    'solver': ['auto', 'svd', 'cholesky']
                }
            },
            'svr': {
                'class': SVR,
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                }
            },
            'mlp': {
                'class': MLPRegressor,
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive']
                }
            }
        }
        
        # Start background training loop
        asyncio.create_task(self._start_training_loop())
    
    async def collect_training_data(
        self, 
        model_type: ModelType, 
        start_date: datetime, 
        end_date: datetime,
        max_samples: int = 50000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Collect training data for specified model type and date range"""
        try:
            self.logger.info(f"Collecting training data for {model_type.value} from {start_date} to {end_date}")
            
            if model_type == ModelType.LOAD_PREDICTOR:
                return await self._collect_load_prediction_data(start_date, end_date, max_samples)
            elif model_type == ModelType.PATTERN_CLASSIFIER:
                return await self._collect_pattern_classification_data(start_date, end_date, max_samples)
            elif model_type == ModelType.VOLATILITY_PREDICTOR:
                return await self._collect_volatility_prediction_data(start_date, end_date, max_samples)
            elif model_type == ModelType.REGIME_CLASSIFIER:
                return await self._collect_regime_classification_data(start_date, end_date, max_samples)
            elif model_type == ModelType.PERFORMANCE_OPTIMIZER:
                return await self._collect_performance_optimization_data(start_date, end_date, max_samples)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
                
        except Exception as e:
            self.logger.error(f"Error collecting training data for {model_type.value}: {str(e)}")
            # Return synthetic data as fallback
            return self._generate_synthetic_training_data(model_type, max_samples)
    
    async def _collect_load_prediction_data(
        self, 
        start_date: datetime, 
        end_date: datetime, 
        max_samples: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Collect data for load prediction model"""
        # Get historical metrics from Redis
        features = []
        targets = []
        
        # This would typically query time-series database
        # For now, generate realistic synthetic data
        n_samples = min(max_samples, int((end_date - start_date).total_seconds() / 300))  # 5-minute intervals
        
        current_time = start_date
        for _ in range(n_samples):
            if current_time >= end_date:
                break
                
            # Create features based on time and market conditions
            hour = current_time.hour
            day_of_week = current_time.weekday()
            is_market_hours = float(9 <= hour <= 16 and day_of_week < 5)
            
            # Simulate market conditions
            vix = np.random.uniform(15, 35)
            volume_ratio = np.random.uniform(0.5, 2.0)
            volatility = np.random.uniform(0.01, 0.05)
            
            # Feature vector
            feature_vector = [
                hour, day_of_week, is_market_hours,
                vix, volume_ratio, volatility,
                np.random.uniform(0, 100),  # current_cpu
                np.random.uniform(0, 100),  # current_memory
                np.random.uniform(0, 1000),  # connections
                np.random.uniform(0, 500),  # request_rate
                np.random.uniform(0, 3),  # market_events
                np.sin(hour * np.pi / 12),  # time_cycle
                np.cos(day_of_week * 2 * np.pi / 7)  # weekly_cycle
            ]
            
            # Target: resource load (0-1)
            base_load = 0.3 + 0.4 * is_market_hours + 0.2 * (vix - 20) / 20
            noise = np.random.normal(0, 0.1)
            target = max(0, min(1, base_load + noise))
            
            features.append(feature_vector)
            targets.append(target)
            
            current_time += timedelta(minutes=5)
        
        return np.array(features), np.array(targets)
    
    async def _collect_pattern_classification_data(
        self, 
        start_date: datetime, 
        end_date: datetime, 
        max_samples: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Collect data for pattern classification model"""
        features = []
        targets = []
        
        n_samples = min(max_samples, 10000)
        
        for _ in range(n_samples):
            # Market condition features
            vix = np.random.uniform(10, 50)
            spy_change = np.random.normal(0, 1.5)
            volume_ratio = np.random.uniform(0.3, 3.0)
            breadth = np.random.uniform(0.2, 0.8)
            
            # Time features
            hour = np.random.randint(0, 24)
            is_market_hours = float(9 <= hour <= 16)
            
            feature_vector = [
                vix, spy_change, volume_ratio, breadth,
                hour, is_market_hours,
                np.random.uniform(-0.5, 0.5),  # trend
                np.random.uniform(0, 5),  # earnings_events
                np.random.uniform(0, 30)  # fed_proximity
            ]
            
            # Pattern classification (0-1 scale)
            if vix > 35:
                pattern_score = 0.9  # Crisis
            elif vix > 25:
                pattern_score = 0.7  # High volatility
            elif abs(spy_change) > 2:
                pattern_score = 0.6  # Strong trend
            elif volume_ratio > 2:
                pattern_score = 0.5  # High activity
            else:
                pattern_score = 0.2  # Normal/low activity
            
            features.append(feature_vector)
            targets.append(pattern_score)
        
        return np.array(features), np.array(targets)
    
    async def _collect_volatility_prediction_data(
        self, 
        start_date: datetime, 
        end_date: datetime, 
        max_samples: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Collect data for volatility prediction model"""
        features = []
        targets = []
        
        n_samples = min(max_samples, 8000)
        
        for _ in range(n_samples):
            # Historical volatility features
            vol_1d = np.random.uniform(0.005, 0.08)
            vol_5d = np.random.uniform(0.008, 0.06)
            vol_20d = np.random.uniform(0.01, 0.05)
            
            # Market features
            vix = np.random.uniform(10, 50)
            spy_return = np.random.normal(0, 0.02)
            volume = np.random.uniform(0.5, 2.5)
            
            # Economic features
            earnings_count = np.random.poisson(2)
            econ_events = np.random.poisson(1)
            
            feature_vector = [
                vol_1d, vol_5d, vol_20d,
                vix, spy_return, volume,
                earnings_count, econ_events,
                np.random.uniform(3, 6),  # interest_rate
                np.random.uniform(-2, 4)  # gdp_growth
            ]
            
            # Predict next day volatility
            future_vol = vol_1d * (1 + np.random.normal(0, 0.3)) + vix * 0.001 + abs(spy_return) * 0.5
            future_vol = max(0.005, min(0.1, future_vol))
            
            features.append(feature_vector)
            targets.append(future_vol)
        
        return np.array(features), np.array(targets)
    
    async def _collect_regime_classification_data(
        self, 
        start_date: datetime, 
        end_date: datetime, 
        max_samples: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Collect data for regime classification model"""
        features = []
        targets = []
        
        n_samples = min(max_samples, 5000)
        
        for _ in range(n_samples):
            # Market indicators
            vix = np.random.uniform(10, 60)
            spy_20d_return = np.random.normal(0, 0.15)
            trend_strength = np.random.uniform(-1, 1)
            breadth = np.random.uniform(0.1, 0.9)
            volume_trend = np.random.uniform(0.5, 2.0)
            
            # Economic indicators
            unemployment = np.random.uniform(3, 10)
            inflation = np.random.uniform(0, 6)
            interest_rates = np.random.uniform(0, 8)
            
            feature_vector = [
                vix, spy_20d_return, trend_strength,
                breadth, volume_trend,
                unemployment, inflation, interest_rates,
                np.random.uniform(80, 130),  # consumer_confidence
                np.random.uniform(-0.5, 0.5)  # yield_curve_slope
            ]
            
            # Regime classification (0-1 encoding)
            if vix > 40:
                regime = 0.9  # Crisis
            elif vix > 25 and spy_20d_return < -0.1:
                regime = 0.8  # Bear volatile
            elif vix < 15 and spy_20d_return > 0.1:
                regime = 0.1  # Bull calm
            elif spy_20d_return > 0.05:
                regime = 0.2  # Bull
            elif spy_20d_return < -0.05:
                regime = 0.7  # Bear
            else:
                regime = 0.5  # Sideways
            
            features.append(feature_vector)
            targets.append(regime)
        
        return np.array(features), np.array(targets)
    
    async def _collect_performance_optimization_data(
        self, 
        start_date: datetime, 
        end_date: datetime, 
        max_samples: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Collect data for performance optimization model"""
        features = []
        targets = []
        
        n_samples = min(max_samples, 12000)
        
        for _ in range(n_samples):
            # System configuration features
            cpu_allocation = np.random.uniform(1, 4)
            memory_allocation = np.random.uniform(2, 8)
            batch_size = np.random.randint(16, 256)
            timeout = np.random.uniform(0.1, 2.0)
            
            # Market condition features
            volatility = np.random.uniform(0.01, 0.08)
            volume = np.random.uniform(0.3, 2.5)
            regime_score = np.random.uniform(0, 1)
            
            # Load features
            connection_count = np.random.randint(10, 1000)
            request_rate = np.random.uniform(10, 500)
            
            feature_vector = [
                cpu_allocation, memory_allocation, batch_size, timeout,
                volatility, volume, regime_score,
                connection_count, request_rate,
                np.random.randint(0, 24),  # hour
                np.random.uniform(0, 1)  # is_market_hours
            ]
            
            # Performance score (0-1, higher is better)
            base_performance = 0.7
            
            # CPU impact
            cpu_optimal = 2.5
            cpu_penalty = abs(cpu_allocation - cpu_optimal) * 0.1
            
            # Batch size impact
            if batch_size < 32:
                batch_penalty = (32 - batch_size) * 0.002
            elif batch_size > 128:
                batch_penalty = (batch_size - 128) * 0.001
            else:
                batch_penalty = 0
            
            # Market condition impact
            if volatility > 0.05:
                vol_penalty = (volatility - 0.05) * 2
            else:
                vol_penalty = 0
            
            performance = base_performance - cpu_penalty - batch_penalty - vol_penalty
            performance = max(0.1, min(1.0, performance + np.random.normal(0, 0.05)))
            
            features.append(feature_vector)
            targets.append(performance)
        
        return np.array(features), np.array(targets)
    
    def _generate_synthetic_training_data(self, model_type: ModelType, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic training data as fallback"""
        np.random.seed(42)
        
        if model_type == ModelType.LOAD_PREDICTOR:
            n_features = 13
        elif model_type == ModelType.PATTERN_CLASSIFIER:
            n_features = 9
        elif model_type == ModelType.VOLATILITY_PREDICTOR:
            n_features = 10
        elif model_type == ModelType.REGIME_CLASSIFIER:
            n_features = 10
        elif model_type == ModelType.PERFORMANCE_OPTIMIZER:
            n_features = 11
        else:
            n_features = 10
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.uniform(0, 1, n_samples)
        
        return X, y
    
    def _preprocess_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
        """Preprocess training data"""
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Remove outliers using IQR method
        Q1 = np.percentile(y, 25)
        Q3 = np.percentile(y, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_mask = (y >= lower_bound) & (y <= upper_bound)
        X_cleaned = X_scaled[outlier_mask]
        y_cleaned = y[outlier_mask]
        
        self.logger.info(f"Removed {len(y) - len(y_cleaned)} outliers ({(len(y) - len(y_cleaned))/len(y)*100:.1f}%)")
        
        return X_cleaned, y_cleaned, scaler
    
    def _select_features(self, X: np.ndarray, y: np.ndarray, max_features: int = 20) -> Tuple[np.ndarray, List[int]]:
        """Select most important features"""
        if X.shape[1] <= max_features:
            return X, list(range(X.shape[1]))
        
        selector = SelectKBest(score_func=f_regression, k=max_features)
        X_selected = selector.fit_transform(X, y)
        selected_features = selector.get_support(indices=True).tolist()
        
        self.logger.info(f"Selected {len(selected_features)} features from {X.shape[1]}")
        return X_selected, selected_features
    
    async def train_model(self, job: TrainingJob) -> ModelMetrics:
        """Train a model based on the training job specification"""
        try:
            job.started_at = datetime.now()
            job.status = "running"
            self.active_jobs[job.job_id] = job
            
            self.logger.info(f"Starting training job {job.job_id} for {job.model_type.value}")
            
            # Collect training data
            X, y = await self.collect_training_data(
                job.model_type,
                job.data_start_date,
                job.data_end_date,
                job.max_samples
            )
            
            if len(X) < job.min_samples:
                raise ValueError(f"Insufficient training data: {len(X)} < {job.min_samples}")
            
            # Preprocess data
            X_processed, y_processed, scaler = self._preprocess_data(X, y)
            
            # Feature selection
            X_selected, selected_features = self._select_features(X_processed, y_processed)
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X_selected, y_processed, 
                test_size=0.2, 
                random_state=42
            )
            
            # Train models with different algorithms
            best_model = None
            best_score = float('-inf')
            best_algorithm = None
            all_results = {}
            
            for algorithm in job.algorithms:
                if algorithm not in self.algorithm_configs:
                    self.logger.warning(f"Unknown algorithm: {algorithm}")
                    continue
                
                try:
                    model_result = await self._train_single_algorithm(
                        algorithm, X_train, X_val, y_train, y_val, 
                        job.hyperparameter_tuning, job.cross_validation_folds
                    )
                    
                    all_results[algorithm] = model_result
                    
                    if model_result['val_r2'] > best_score:
                        best_score = model_result['val_r2']
                        best_model = model_result['model']
                        best_algorithm = algorithm
                        
                except Exception as e:
                    self.logger.error(f"Error training {algorithm}: {str(e)}")
                    continue
            
            if best_model is None:
                raise ValueError("No models trained successfully")
            
            # Validate model meets requirements
            best_result = all_results[best_algorithm]
            if best_result['val_r2'] < job.min_val_r2 or best_result['val_mse'] > job.max_val_mse:
                self.logger.warning(
                    f"Model quality below threshold: R2={best_result['val_r2']:.3f}, "
                    f"MSE={best_result['val_mse']:.3f}"
                )
            
            # Create model version
            model_version = self._generate_model_version(job.model_type, best_algorithm)
            
            # Save model and preprocessing components
            model_package = {
                'model': best_model,
                'scaler': scaler,
                'selected_features': selected_features,
                'algorithm': best_algorithm,
                'training_job_id': job.job_id,
                'created_at': datetime.now().isoformat()
            }
            
            model_path = self._save_model(job.model_type, model_version, model_package)
            
            # Calculate feature importance
            feature_importance = self._get_feature_importance(best_model, selected_features)
            
            # Create metrics
            metrics = ModelMetrics(
                model_type=job.model_type,
                model_version=model_version,
                timestamp=datetime.now(),
                train_mse=best_result['train_mse'],
                train_mae=best_result['train_mae'],
                train_r2=best_result['train_r2'],
                val_mse=best_result['val_mse'],
                val_mae=best_result['val_mae'],
                val_r2=best_result['val_r2'],
                cv_mean=best_result['cv_mean'],
                cv_std=best_result['cv_std'],
                feature_importance=feature_importance
            )
            
            # Store metrics
            self.model_metrics[model_version] = metrics
            await self._store_model_metrics(metrics)
            
            # Update job status
            job.completed_at = datetime.now()
            job.status = "completed"
            
            self.logger.info(
                f"Training job {job.job_id} completed successfully. "
                f"Best algorithm: {best_algorithm}, R2: {best_result['val_r2']:.3f}"
            )
            
            return metrics
            
        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            job.completed_at = datetime.now()
            self.logger.error(f"Training job {job.job_id} failed: {str(e)}")
            raise
        
        finally:
            if job.job_id in self.active_jobs:
                del self.active_jobs[job.job_id]
    
    async def _train_single_algorithm(
        self, 
        algorithm: str, 
        X_train: np.ndarray, 
        X_val: np.ndarray,
        y_train: np.ndarray, 
        y_val: np.ndarray,
        hyperparameter_tuning: bool,
        cv_folds: int
    ) -> Dict[str, Any]:
        """Train a single algorithm and return results"""
        
        config = self.algorithm_configs[algorithm]
        model_class = config['class']
        param_grid = config['params']
        
        if hyperparameter_tuning and len(param_grid) > 0:
            # Hyperparameter tuning with GridSearchCV
            model = GridSearchCV(
                model_class(random_state=42),
                param_grid,
                cv=min(cv_folds, len(X_train) // 100),  # Ensure enough samples per fold
                scoring='r2',
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            best_model = model.best_estimator_
            
        else:
            # Use default parameters
            best_model = model_class(random_state=42)
            best_model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = best_model.predict(X_train)
        y_val_pred = best_model.predict(X_val)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        
        val_mse = mean_squared_error(y_val, y_val_pred)
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv_folds, scoring='r2')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        return {
            'model': best_model,
            'algorithm': algorithm,
            'train_mse': train_mse,
            'train_mae': train_mae,
            'train_r2': train_r2,
            'val_mse': val_mse,
            'val_mae': val_mae,
            'val_r2': val_r2,
            'cv_mean': cv_mean,
            'cv_std': cv_std
        }
    
    def _generate_model_version(self, model_type: ModelType, algorithm: str) -> str:
        """Generate unique model version identifier"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_input = f"{model_type.value}_{algorithm}_{timestamp}"
        hash_suffix = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        return f"{model_type.value}_{algorithm}_{timestamp}_{hash_suffix}"
    
    def _save_model(self, model_type: ModelType, version: str, model_package: Dict[str, Any]) -> Path:
        """Save model package to disk"""
        model_dir = self.model_storage_path / model_type.value
        model_dir.mkdir(exist_ok=True)
        
        model_path = model_dir / f"{version}.joblib"
        joblib.dump(model_package, model_path)
        
        # Update latest symlink
        latest_path = model_dir / "latest.joblib"
        if latest_path.exists():
            latest_path.unlink()
        latest_path.symlink_to(model_path.name)
        
        self.logger.info(f"Saved model to {model_path}")
        return model_path
    
    def _get_feature_importance(self, model: Any, selected_features: List[int]) -> Dict[str, float]:
        """Extract feature importance from trained model"""
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_)
            else:
                # No feature importance available
                return {}
            
            # Create feature importance dictionary
            feature_names = [f"feature_{i}" for i in selected_features]
            importance_dict = dict(zip(feature_names, importances))
            
            # Normalize to sum to 1
            total_importance = sum(importance_dict.values())
            if total_importance > 0:
                importance_dict = {k: v/total_importance for k, v in importance_dict.items()}
            
            return importance_dict
            
        except Exception as e:
            self.logger.error(f"Error extracting feature importance: {str(e)}")
            return {}
    
    async def _store_model_metrics(self, metrics: ModelMetrics):
        """Store model metrics in Redis"""
        metrics_data = {
            "model_type": metrics.model_type.value,
            "model_version": metrics.model_version,
            "timestamp": metrics.timestamp.isoformat(),
            "train_r2": metrics.train_r2,
            "val_r2": metrics.val_r2,
            "cv_mean": metrics.cv_mean,
            "cv_std": metrics.cv_std,
            "feature_importance": metrics.feature_importance
        }
        
        # Store individual metrics
        self.redis_client.setex(
            f"model:metrics:{metrics.model_version}",
            86400,  # 24 hour expiry
            json.dumps(metrics_data)
        )
        
        # Update model type metrics list
        self.redis_client.lpush(
            f"model:metrics:list:{metrics.model_type.value}",
            json.dumps(metrics_data)
        )
        self.redis_client.ltrim(f"model:metrics:list:{metrics.model_type.value}", 0, 49)  # Keep last 50
    
    def schedule_training_job(
        self, 
        model_type: ModelType, 
        mode: TrainingMode = TrainingMode.FULL_RETRAIN,
        priority: int = 5
    ) -> str:
        """Schedule a new training job"""
        
        job_id = f"{model_type.value}_{mode.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Determine data range
        if mode == TrainingMode.INITIAL_TRAINING:
            data_start_date = datetime.now() - timedelta(days=30)
            data_end_date = datetime.now()
        elif mode == TrainingMode.INCREMENTAL_UPDATE:
            data_start_date = datetime.now() - timedelta(days=7)
            data_end_date = datetime.now()
        else:  # FULL_RETRAIN
            data_start_date = datetime.now() - timedelta(days=60)
            data_end_date = datetime.now()
        
        job = TrainingJob(
            job_id=job_id,
            model_type=model_type,
            training_mode=mode,
            priority=priority,
            data_start_date=data_start_date,
            data_end_date=data_end_date
        )
        
        # Add to queue (insert based on priority)
        inserted = False
        for i, existing_job in enumerate(self.training_queue):
            if job.priority > existing_job.priority:
                self.training_queue.insert(i, job)
                inserted = True
                break
        
        if not inserted:
            self.training_queue.append(job)
        
        self.logger.info(f"Scheduled training job {job_id} with priority {priority}")
        return job_id
    
    async def check_retrain_schedule(self):
        """Check if models need retraining based on schedule"""
        now = datetime.now()
        
        for model_type, interval in self.retrain_schedule.items():
            # Get last training time
            last_training_key = f"model:last_training:{model_type.value}"
            last_training_str = self.redis_client.get(last_training_key)
            
            if last_training_str:
                last_training = datetime.fromisoformat(last_training_str)
                if now - last_training < interval:
                    continue  # Not yet due for retraining
            
            # Schedule retraining
            job_id = self.schedule_training_job(
                model_type, 
                TrainingMode.FULL_RETRAIN,
                priority=3
            )
            
            # Update last training time
            self.redis_client.set(last_training_key, now.isoformat())
            
            self.logger.info(f"Scheduled periodic retraining for {model_type.value}: {job_id}")
    
    async def check_model_drift(self):
        """Check for model drift and trigger retraining if needed"""
        for model_type in ModelType:
            try:
                drift_score = await self._calculate_drift_score(model_type)
                threshold = self.drift_thresholds.get(model_type, 0.1)
                
                if drift_score > threshold:
                    self.logger.warning(
                        f"Model drift detected for {model_type.value}: "
                        f"{drift_score:.3f} > {threshold:.3f}"
                    )
                    
                    # Schedule emergency retraining
                    job_id = self.schedule_training_job(
                        model_type,
                        TrainingMode.EMERGENCY_RETRAIN,
                        priority=8
                    )
                    
                    self.logger.info(f"Scheduled emergency retraining: {job_id}")
                
            except Exception as e:
                self.logger.error(f"Error checking drift for {model_type.value}: {str(e)}")
    
    async def _calculate_drift_score(self, model_type: ModelType) -> float:
        """Calculate drift score for a model type"""
        # This would typically compare recent performance to historical performance
        # For now, return a simulated drift score
        
        # Get recent performance metrics
        recent_accuracy = np.random.uniform(0.7, 0.9)
        historical_accuracy = 0.8  # Baseline
        
        # Simple drift calculation
        drift_score = abs(recent_accuracy - historical_accuracy)
        
        # Add some randomness for simulation
        drift_score += np.random.uniform(0, 0.05)
        
        return drift_score
    
    async def _start_training_loop(self):
        """Start background training loop"""
        await asyncio.sleep(10)  # Initial delay
        
        while True:
            try:
                # Check retrain schedule
                await self.check_retrain_schedule()
                
                # Check model drift
                await self.check_model_drift()
                
                # Process training queue
                if self.training_queue and len(self.active_jobs) < 2:  # Limit concurrent jobs
                    job = self.training_queue.pop(0)
                    
                    try:
                        asyncio.create_task(self.train_model(job))
                    except Exception as e:
                        self.logger.error(f"Error starting training job {job.job_id}: {str(e)}")
                
                # Wait before next iteration
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in training loop: {str(e)}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def get_training_status(self) -> Dict[str, Any]:
        """Get current training pipeline status"""
        return {
            "active_jobs": len(self.active_jobs),
            "queued_jobs": len(self.training_queue),
            "production_models": len(self.production_models),
            "challenger_models": len(self.challenger_models),
            "recent_jobs": [
                {
                    "job_id": job.job_id,
                    "model_type": job.model_type.value,
                    "status": job.status,
                    "started_at": job.started_at.isoformat() if job.started_at else None,
                    "priority": job.priority
                }
                for job in list(self.active_jobs.values())
            ],
            "queue_summary": [
                {
                    "model_type": job.model_type.value,
                    "mode": job.training_mode.value,
                    "priority": job.priority,
                    "scheduled_at": job.created_at.isoformat()
                }
                for job in self.training_queue[:5]  # Show first 5 in queue
            ]
        }


async def main():
    """Test the ML Training Pipeline"""
    logging.basicConfig(level=logging.INFO)
    
    pipeline = MLTrainingPipeline()
    
    print("🎯 Testing ML Training Pipeline")
    print("=" * 40)
    
    # Schedule training jobs for all model types
    model_types = [
        ModelType.LOAD_PREDICTOR,
        ModelType.PATTERN_CLASSIFIER,
        ModelType.VOLATILITY_PREDICTOR
    ]
    
    scheduled_jobs = []
    for model_type in model_types:
        job_id = pipeline.schedule_training_job(
            model_type,
            TrainingMode.INITIAL_TRAINING,
            priority=6
        )
        scheduled_jobs.append(job_id)
        print(f"✅ Scheduled {model_type.value} training job: {job_id}")
    
    # Get training status
    print(f"\n📊 Training Pipeline Status:")
    status = await pipeline.get_training_status()
    print(f"Active Jobs: {status['active_jobs']}")
    print(f"Queued Jobs: {status['queued_jobs']}")
    print(f"Production Models: {status['production_models']}")
    
    if status['queue_summary']:
        print("\nQueued Jobs:")
        for job in status['queue_summary']:
            print(f"  - {job['model_type']} ({job['mode']}) Priority: {job['priority']}")
    
    # Train one model to demonstrate
    print(f"\n🏋️ Training Load Predictor Model...")
    
    try:
        # Create a simple training job
        job = TrainingJob(
            job_id="test_load_predictor",
            model_type=ModelType.LOAD_PREDICTOR,
            training_mode=TrainingMode.INITIAL_TRAINING,
            priority=10,
            data_start_date=datetime.now() - timedelta(days=7),
            data_end_date=datetime.now(),
            algorithms=['random_forest', 'gradient_boosting'],
            hyperparameter_tuning=False,  # Faster for demo
            min_samples=100
        )
        
        metrics = await pipeline.train_model(job)
        
        print(f"✅ Training completed successfully!")
        print(f"Model Version: {metrics.model_version}")
        print(f"Validation R²: {metrics.val_r2:.3f}")
        print(f"Validation MSE: {metrics.val_mse:.4f}")
        print(f"Cross-validation: {metrics.cv_mean:.3f} ± {metrics.cv_std:.3f}")
        
        if metrics.feature_importance:
            print("Top Features:")
            sorted_features = sorted(
                metrics.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )
            for feature, importance in sorted_features[:5]:
                print(f"  {feature}: {importance:.3f}")
                
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
    
    # Final status
    print(f"\n📈 Final Pipeline Status:")
    final_status = await pipeline.get_training_status()
    print(f"Active Jobs: {final_status['active_jobs']}")
    print(f"Queued Jobs: {final_status['queued_jobs']}")
    
    print("\n✅ ML Training Pipeline test completed!")


if __name__ == "__main__":
    asyncio.run(main())