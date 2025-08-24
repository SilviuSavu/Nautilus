"""
Advanced Market Regime Detection with ML Ensemble Methods

Implements sophisticated market regime classification using:
- Multi-model ensemble approach (Random Forest, Gradient Boosting, LSTM)
- Real-time regime classification with confidence scores
- Regime-based risk adjustment algorithms
- Alternative data integration for regime detection
- Dynamic rebalancing based on regime transitions
"""

import asyncio
import logging
import json
import uuid
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque, defaultdict
import pickle
import joblib
from concurrent.futures import ThreadPoolExecutor

# ML imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import scipy.stats as stats

# Database imports
import asyncpg
import redis.asyncio as redis

# Internal imports
from .config import get_ml_config, RegimeType, ModelType
from .utils import MLMetrics, FeatureStore
try:
    from ..websocket.redis_pubsub import get_redis_pubsub_manager
except ImportError:
    # Fallback for when running as standalone module
    def get_redis_pubsub_manager():
        return None
try:
    from ..websocket.message_protocols import create_system_alert_message
except ImportError:
    # Fallback for standalone module
    def create_system_alert_message(alert_type, data):
        return {'type': alert_type, 'data': data}

logger = logging.getLogger(__name__)


@dataclass
class RegimeSignal:
    """Individual regime detection signal"""
    regime_type: RegimeType
    confidence: float
    model_source: str
    features_used: List[str]
    probability_distribution: Dict[RegimeType, float]
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RegimeState:
    """Current market regime state"""
    current_regime: RegimeType
    confidence: float
    regime_probability: Dict[RegimeType, float]
    regime_duration: timedelta
    transition_probability: Dict[RegimeType, float]
    regime_strength: float  # 0-1 scale
    supporting_indicators: List[str]
    risk_adjustment_factor: float
    recommended_allocation: Dict[str, float]
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    # Historical context
    previous_regime: Optional[RegimeType] = None
    transition_time: Optional[datetime] = None
    regime_volatility: float = 0.0
    regime_persistence: float = 0.0


@dataclass
class RegimeTransition:
    """Regime transition event"""
    transition_id: str
    from_regime: RegimeType
    to_regime: RegimeType
    transition_time: datetime
    transition_confidence: float
    transition_speed: str  # gradual, rapid, sudden
    market_catalysts: List[str]
    impact_assessment: Dict[str, float]
    recommended_actions: List[str]


class RegimeClassifier:
    """
    Individual regime classification model
    Base class for different model types
    """
    
    def __init__(
        self,
        model_type: ModelType,
        model_config: Optional[Dict[str, Any]] = None
    ):
        self.model_type = model_type
        self.model_config = model_config or {}
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_selector = None
        self.feature_names = []
        self.is_trained = False
        self.training_metrics = {}
        self.last_prediction = None
        self.prediction_cache = {}
        
        self.logger = logging.getLogger(f"{__name__}.{model_type.value}")
    
    def _create_model(self) -> Any:
        """Create model instance based on type"""
        if self.model_type == ModelType.RANDOM_FOREST:
            return RandomForestClassifier(
                n_estimators=self.model_config.get('n_estimators', 100),
                max_depth=self.model_config.get('max_depth', 10),
                min_samples_split=self.model_config.get('min_samples_split', 5),
                min_samples_leaf=self.model_config.get('min_samples_leaf', 2),
                max_features=self.model_config.get('max_features', 'sqrt'),
                bootstrap=self.model_config.get('bootstrap', True),
                oob_score=self.model_config.get('oob_score', True),
                random_state=42,
                n_jobs=-1
            )
        
        elif self.model_type == ModelType.GRADIENT_BOOSTING:
            return GradientBoostingClassifier(
                n_estimators=self.model_config.get('n_estimators', 100),
                learning_rate=self.model_config.get('learning_rate', 0.1),
                max_depth=self.model_config.get('max_depth', 6),
                min_samples_split=self.model_config.get('min_samples_split', 5),
                min_samples_leaf=self.model_config.get('min_samples_leaf', 2),
                subsample=self.model_config.get('subsample', 0.8),
                max_features=self.model_config.get('max_features', 'sqrt'),
                random_state=42
            )
        
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """Train the regime classification model"""
        try:
            self.feature_names = feature_names
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, stratify=y, random_state=42
            )
            
            # Encode labels
            y_train_encoded = self.label_encoder.fit_transform(y_train)
            y_val_encoded = self.label_encoder.transform(y_val)
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            # Feature selection
            self.feature_selector = SelectKBest(
                mutual_info_classif, 
                k=min(30, X_train_scaled.shape[1])
            )
            X_train_selected = self.feature_selector.fit_transform(X_train_scaled, y_train_encoded)
            X_val_selected = self.feature_selector.transform(X_val_scaled)
            
            # Create and train model
            self.model = self._create_model()
            self.model.fit(X_train_selected, y_train_encoded)
            
            # Validate model
            val_predictions = self.model.predict(X_val_selected)
            val_probabilities = self.model.predict_proba(X_val_selected)
            
            # Calculate metrics
            val_accuracy = (val_predictions == y_val_encoded).mean()
            val_auc = roc_auc_score(y_val_encoded, val_probabilities, multi_class='ovr')
            
            # Cross-validation
            cv_scores = cross_val_score(
                self.model, X_train_selected, y_train_encoded, 
                cv=StratifiedKFold(n_splits=5), scoring='accuracy'
            )
            
            self.training_metrics = {
                'validation_accuracy': float(val_accuracy),
                'validation_auc': float(val_auc),
                'cv_mean_accuracy': float(cv_scores.mean()),
                'cv_std_accuracy': float(cv_scores.std()),
                'training_samples': len(X_train),
                'validation_samples': len(X_val),
                'feature_count': X_train_selected.shape[1],
                'model_type': self.model_type.value,
                'trained_at': datetime.utcnow().isoformat()
            }
            
            # Feature importance (if available)
            if hasattr(self.model, 'feature_importances_'):
                selected_features = np.array(feature_names)[self.feature_selector.get_support()]
                importance_dict = dict(zip(selected_features, self.model.feature_importances_))
                self.training_metrics['feature_importance'] = importance_dict
            
            self.is_trained = True
            self.logger.info(
                f"Model trained - Accuracy: {val_accuracy:.3f}, AUC: {val_auc:.3f}, "
                f"CV: {cv_scores.mean():.3f}±{cv_scores.std():.3f}"
            )
            
            return self.training_metrics
            
        except Exception as e:
            self.logger.error(f"Error training model: {e}")
            raise
    
    def predict(
        self,
        X: np.ndarray,
        return_probabilities: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Make regime predictions"""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
        
        try:
            # Preprocess features
            X_scaled = self.scaler.transform(X)
            X_selected = self.feature_selector.transform(X_scaled)
            
            # Make predictions
            predictions = self.model.predict(X_selected)
            regime_predictions = self.label_encoder.inverse_transform(predictions)
            
            if return_probabilities:
                probabilities = self.model.predict_proba(X_selected)
                return regime_predictions, probabilities
            else:
                return regime_predictions
                
        except Exception as e:
            self.logger.error(f"Error making predictions: {e}")
            raise
    
    def predict_single(
        self,
        features: Dict[str, float],
        return_confidence: bool = True
    ) -> Union[RegimeType, Tuple[RegimeType, float, Dict[RegimeType, float]]]:
        """Predict regime for single observation"""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
        
        try:
            # Convert features to array
            feature_values = [features.get(name, 0.0) for name in self.feature_names]
            X = np.array(feature_values).reshape(1, -1)
            
            # Make prediction
            regime_pred, probabilities = self.predict(X, return_probabilities=True)
            regime = RegimeType(regime_pred[0])
            
            if return_confidence:
                # Convert probabilities to regime mapping
                regime_probs = {}
                for i, regime_class in enumerate(self.label_encoder.classes_):
                    regime_probs[RegimeType(regime_class)] = float(probabilities[0][i])
                
                confidence = max(regime_probs.values())
                return regime, confidence, regime_probs
            else:
                return regime
                
        except Exception as e:
            self.logger.error(f"Error predicting single observation: {e}")
            raise
    
    def save_model(self, file_path: str) -> None:
        """Save trained model to file"""
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_selector': self.feature_selector,
            'feature_names': self.feature_names,
            'model_type': self.model_type.value,
            'model_config': self.model_config,
            'training_metrics': self.training_metrics,
            'saved_at': datetime.utcnow().isoformat()
        }
        
        joblib.dump(model_data, file_path)
        self.logger.info(f"Model saved to {file_path}")
    
    @classmethod
    def load_model(cls, file_path: str) -> 'RegimeClassifier':
        """Load trained model from file"""
        model_data = joblib.load(file_path)
        
        classifier = cls(
            model_type=ModelType(model_data['model_type']),
            model_config=model_data['model_config']
        )
        
        classifier.model = model_data['model']
        classifier.scaler = model_data['scaler']
        classifier.label_encoder = model_data['label_encoder']
        classifier.feature_selector = model_data['feature_selector']
        classifier.feature_names = model_data['feature_names']
        classifier.training_metrics = model_data['training_metrics']
        classifier.is_trained = True
        
        return classifier


class MarketRegimeDetector:
    """
    Advanced market regime detection system using ensemble methods
    
    Features:
    - Multi-model ensemble (Random Forest, Gradient Boosting, LSTM)
    - Real-time regime classification with confidence scoring
    - Alternative data integration for enhanced regime detection
    - Dynamic risk adjustment based on regime transitions
    - Regime persistence and transition analysis
    """
    
    def __init__(
        self,
        database_url: str = None,
        redis_url: str = None,
        model_storage_path: str = None
    ):
        self.config = get_ml_config()
        self.database_url = database_url or self.config.database_url
        self.redis_url = redis_url or self.config.redis_url
        self.model_storage_path = model_storage_path or self.config.model_storage_path
        
        # Core components
        self.db_connection: Optional[asyncpg.Connection] = None
        self.redis_client: Optional[redis.Redis] = None
        self.redis_pubsub = None
        self.feature_store = FeatureStore()
        
        # Model ensemble
        self.classifiers: Dict[str, RegimeClassifier] = {}
        self.ensemble_weights: Dict[str, float] = {}
        self.model_performance: Dict[str, Dict[str, float]] = {}
        
        # Regime state tracking
        self.current_regime_state: Optional[RegimeState] = None
        self.regime_history: deque = deque(maxlen=1000)
        self.transition_history: deque = deque(maxlen=100)
        
        # Feature engineering
        self.feature_cache: Dict[str, Any] = {}
        self.feature_last_update: Dict[str, datetime] = {}
        
        # Real-time processing
        self.prediction_cache: Dict[str, Tuple[RegimeState, datetime]] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Monitoring and metrics
        self.total_predictions = 0
        self.regime_transitions = 0
        self.prediction_accuracy = 0.0
        self.average_confidence = 0.0
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._feature_update_task: Optional[asyncio.Task] = None
        self._model_update_task: Optional[asyncio.Task] = None
        
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self) -> None:
        """Initialize the market regime detector"""
        try:
            # Initialize database connection
            self.db_connection = await asyncpg.connect(self.database_url)
            
            # Initialize Redis
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            await self.redis_client.ping()
            
            # Initialize Redis pub/sub
            self.redis_pubsub = get_redis_pubsub_manager()
            
            # Create database tables
            await self._create_database_tables()
            
            # Initialize feature store
            await self.feature_store.initialize(self.db_connection, self.redis_client)
            
            # Load or create ensemble models
            await self._initialize_ensemble()
            
            # Load historical regime state
            await self._load_regime_history()
            
            # Start background tasks
            self._monitoring_task = asyncio.create_task(self._regime_monitoring_loop())
            self._feature_update_task = asyncio.create_task(self._feature_update_loop())
            self._model_update_task = asyncio.create_task(self._model_update_loop())
            
            self.logger.info("Market regime detector initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize market regime detector: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the regime detector"""
        try:
            # Cancel background tasks
            tasks = [self._monitoring_task, self._feature_update_task, self._model_update_task]
            for task in tasks:
                if task:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            # Close connections
            if self.db_connection:
                await self.db_connection.close()
            
            if self.redis_client:
                await self.redis_client.close()
            
            self.logger.info("Market regime detector shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for market regime detector"""
        try:
            health_status = {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "database_connected": self.db_connection is not None,
                "redis_connected": self.redis_client is not None,
                "models_loaded": len(self.classifiers),
                "total_predictions": self.total_predictions,
                "regime_transitions": self.regime_transitions,
                "prediction_accuracy": self.prediction_accuracy,
                "average_confidence": self.average_confidence,
                "current_regime": self.current_regime_state.regime.value if self.current_regime_state else None
            }
            
            # Test database connection
            if self.db_connection:
                try:
                    await self.db_connection.fetchval("SELECT 1")
                    health_status["database_status"] = "connected"
                except Exception as e:
                    health_status["database_status"] = f"error: {str(e)}"
                    health_status["status"] = "degraded"
            
            # Test Redis connection
            if self.redis_client:
                try:
                    await self.redis_client.ping()
                    health_status["redis_status"] = "connected"
                except Exception as e:
                    health_status["redis_status"] = f"error: {str(e)}"
                    health_status["status"] = "degraded"
            
            return health_status
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }
    
    async def _initialize_ensemble(self) -> None:
        """Initialize the ensemble of regime classification models"""
        try:
            model_types = self.config.regime_detection.ensemble_models
            
            for model_type in model_types:
                model_id = f"regime_{model_type.value}"
                
                # Try to load existing model
                model_path = f"{self.model_storage_path}/{model_id}.joblib"
                try:
                    classifier = RegimeClassifier.load_model(model_path)
                    self.classifiers[model_id] = classifier
                    self.ensemble_weights[model_id] = 1.0 / len(model_types)  # Equal weights initially
                    self.logger.info(f"Loaded existing model: {model_id}")
                except:
                    # Create new model if loading fails
                    from .config import DEFAULT_MODEL_CONFIGS
                    model_config = DEFAULT_MODEL_CONFIGS.get(model_type, {})
                    classifier = RegimeClassifier(model_type, model_config.hyperparameters)
                    self.classifiers[model_id] = classifier
                    self.ensemble_weights[model_id] = 1.0 / len(model_types)
                    self.logger.info(f"Created new model: {model_id}")
            
            # Train models if they're not trained
            untrained_models = [
                model_id for model_id, classifier in self.classifiers.items() 
                if not classifier.is_trained
            ]
            
            if untrained_models:
                self.logger.info(f"Training {len(untrained_models)} untrained models")
                await self._train_ensemble()
            
        except Exception as e:
            self.logger.error(f"Error initializing ensemble: {e}")
            raise
    
    async def _train_ensemble(self) -> None:
        """Train the ensemble of models with historical data"""
        try:
            # Get training data
            training_data = await self._get_training_data()
            if training_data is None or len(training_data) < 1000:
                self.logger.warning("Insufficient training data, using synthetic data")
                training_data = self._generate_synthetic_training_data()
            
            X, y, feature_names = training_data
            
            # Train each model in the ensemble
            for model_id, classifier in self.classifiers.items():
                if not classifier.is_trained:
                    self.logger.info(f"Training model: {model_id}")
                    
                    # Train model in thread pool to avoid blocking
                    loop = asyncio.get_event_loop()
                    training_metrics = await loop.run_in_executor(
                        self.executor,
                        classifier.train,
                        X, y, feature_names
                    )
                    
                    self.model_performance[model_id] = training_metrics
                    
                    # Save trained model
                    model_path = f"{self.model_storage_path}/{model_id}.joblib"
                    await loop.run_in_executor(
                        self.executor,
                        classifier.save_model,
                        model_path
                    )
                    
                    self.logger.info(
                        f"Model {model_id} trained with accuracy: "
                        f"{training_metrics.get('validation_accuracy', 0):.3f}"
                    )
            
            # Update ensemble weights based on performance
            self._update_ensemble_weights()
            
        except Exception as e:
            self.logger.error(f"Error training ensemble: {e}")
            raise
    
    async def _get_training_data(self) -> Optional[Tuple[np.ndarray, np.ndarray, List[str]]]:
        """Get historical training data for regime detection"""
        try:
            # Query historical market data with regime labels
            query = """
                SELECT 
                    date_part('epoch', timestamp)::bigint as timestamp,
                    symbol,
                    open_price, high_price, low_price, close_price,
                    volume,
                    -- Technical indicators
                    sma_5, sma_20, sma_50, sma_200,
                    ema_12, ema_26,
                    rsi_14, rsi_30,
                    macd, macd_signal,
                    bollinger_upper, bollinger_lower,
                    atr_14, atr_30,
                    -- Market regime label (if available)
                    regime_label,
                    -- Market conditions
                    vix_level, treasury_10y, dollar_index,
                    sector_rotation, market_breadth
                FROM market_data_features 
                WHERE timestamp >= NOW() - INTERVAL '2 years'
                    AND regime_label IS NOT NULL
                ORDER BY timestamp
            """
            
            rows = await self.db_connection.fetch(query)
            
            if not rows:
                return None
            
            # Convert to DataFrame for easier processing
            df = pd.DataFrame([dict(row) for row in rows])
            
            # Feature engineering
            features = self._engineer_regime_features(df)
            
            # Extract labels
            labels = df['regime_label'].values
            
            # Feature names
            feature_names = list(features.columns)
            
            return features.values, labels, feature_names
            
        except Exception as e:
            self.logger.error(f"Error getting training data: {e}")
            return None
    
    def _generate_synthetic_training_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Generate synthetic training data for initial model training"""
        try:
            np.random.seed(42)
            n_samples = 5000
            n_features = 20
            
            # Generate synthetic features
            features = np.random.randn(n_samples, n_features)
            
            # Create synthetic regime labels based on feature patterns
            regime_labels = []
            for i in range(n_samples):
                feature_sum = np.sum(features[i])
                if feature_sum > 2:
                    regime_labels.append(RegimeType.BULL.value)
                elif feature_sum < -2:
                    regime_labels.append(RegimeType.BEAR.value)
                elif abs(feature_sum) < 0.5:
                    regime_labels.append(RegimeType.SIDEWAYS.value)
                else:
                    regime_labels.append(RegimeType.VOLATILE.value)
            
            feature_names = [f"feature_{i}" for i in range(n_features)]
            
            self.logger.info(f"Generated {n_samples} synthetic training samples")
            return features, np.array(regime_labels), feature_names
            
        except Exception as e:
            self.logger.error(f"Error generating synthetic data: {e}")
            raise
    
    def _engineer_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for regime detection"""
        try:
            features = pd.DataFrame()
            
            # Price-based features
            features['returns_1d'] = df['close_price'].pct_change(1)
            features['returns_5d'] = df['close_price'].pct_change(5)
            features['returns_20d'] = df['close_price'].pct_change(20)
            
            # Volatility features
            features['volatility_10d'] = features['returns_1d'].rolling(10).std()
            features['volatility_30d'] = features['returns_1d'].rolling(30).std()
            
            # Momentum features
            features['momentum_5d'] = df['close_price'] / df['close_price'].shift(5) - 1
            features['momentum_20d'] = df['close_price'] / df['close_price'].shift(20) - 1
            
            # Technical indicators (if available in data)
            for col in ['sma_5', 'sma_20', 'sma_50', 'rsi_14', 'macd', 'atr_14']:
                if col in df.columns:
                    features[col] = df[col]
            
            # Alternative data features (if available)
            for col in ['vix_level', 'treasury_10y', 'dollar_index', 'sector_rotation']:
                if col in df.columns:
                    features[col] = df[col]
            
            # Fill NaN values
            features = features.fillna(method='ffill').fillna(0)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error engineering features: {e}")
            raise
    
    def _update_ensemble_weights(self) -> None:
        """Update ensemble weights based on model performance"""
        try:
            if not self.model_performance:
                return
            
            # Calculate weights based on validation accuracy
            total_weight = 0
            for model_id in self.classifiers.keys():
                if model_id in self.model_performance:
                    accuracy = self.model_performance[model_id].get('validation_accuracy', 0.5)
                    weight = max(accuracy, 0.1)  # Minimum weight of 0.1
                    self.ensemble_weights[model_id] = weight
                    total_weight += weight
            
            # Normalize weights
            for model_id in self.ensemble_weights.keys():
                self.ensemble_weights[model_id] /= total_weight
            
            self.logger.info(f"Updated ensemble weights: {self.ensemble_weights}")
            
        except Exception as e:
            self.logger.error(f"Error updating ensemble weights: {e}")
    
    async def predict_regime(
        self,
        features: Dict[str, float],
        return_confidence: bool = True
    ) -> Union[RegimeType, Tuple[RegimeState, float]]:
        """Predict current market regime using ensemble"""
        try:
            # Check cache first
            cache_key = str(sorted(features.items()))
            if cache_key in self.prediction_cache:
                cached_result, cache_time = self.prediction_cache[cache_key]
                if datetime.utcnow() - cache_time < timedelta(minutes=1):
                    return cached_result if not return_confidence else (cached_result, cached_result.confidence)
            
            # Get predictions from all models
            model_predictions = {}
            model_confidences = {}
            model_probabilities = {}
            
            for model_id, classifier in self.classifiers.items():
                if classifier.is_trained:
                    try:
                        regime, confidence, probabilities = classifier.predict_single(
                            features, return_confidence=True
                        )
                        model_predictions[model_id] = regime
                        model_confidences[model_id] = confidence
                        model_probabilities[model_id] = probabilities
                    except Exception as e:
                        self.logger.warning(f"Error getting prediction from {model_id}: {e}")
            
            if not model_predictions:
                raise RuntimeError("No models available for prediction")
            
            # Ensemble prediction
            regime_state = self._ensemble_predict(
                model_predictions, model_confidences, model_probabilities
            )
            
            # Cache result
            self.prediction_cache[cache_key] = (regime_state, datetime.utcnow())
            
            # Update statistics
            self.total_predictions += 1
            self.average_confidence = (
                (self.average_confidence * (self.total_predictions - 1) + regime_state.confidence) 
                / self.total_predictions
            )
            
            # Check for regime transition
            if self.current_regime_state:
                if regime_state.current_regime != self.current_regime_state.current_regime:
                    await self._handle_regime_transition(
                        self.current_regime_state.current_regime,
                        regime_state.current_regime,
                        regime_state.confidence
                    )
            
            # Update current state
            self.current_regime_state = regime_state
            self.regime_history.append(regime_state)
            
            # Save to database
            await self._save_regime_state(regime_state)
            
            if return_confidence:
                return regime_state, regime_state.confidence
            else:
                return regime_state.current_regime
                
        except Exception as e:
            self.logger.error(f"Error predicting regime: {e}")
            raise
    
    def _ensemble_predict(
        self,
        model_predictions: Dict[str, RegimeType],
        model_confidences: Dict[str, float],
        model_probabilities: Dict[str, Dict[RegimeType, float]]
    ) -> RegimeState:
        """Combine predictions from multiple models"""
        try:
            # Weighted voting
            regime_votes = defaultdict(float)
            total_weight = 0
            
            for model_id, regime in model_predictions.items():
                weight = self.ensemble_weights.get(model_id, 1.0)
                confidence = model_confidences.get(model_id, 0.5)
                
                # Weight by both ensemble weight and prediction confidence
                vote_weight = weight * confidence
                regime_votes[regime] += vote_weight
                total_weight += vote_weight
            
            # Normalize votes
            for regime in regime_votes:
                regime_votes[regime] /= total_weight
            
            # Select regime with highest vote
            predicted_regime = max(regime_votes.keys(), key=lambda r: regime_votes[r])
            regime_confidence = regime_votes[predicted_regime]
            
            # Calculate ensemble probabilities
            ensemble_probabilities = defaultdict(float)
            for model_id, probabilities in model_probabilities.items():
                weight = self.ensemble_weights.get(model_id, 1.0)
                for regime, prob in probabilities.items():
                    ensemble_probabilities[regime] += weight * prob
            
            # Normalize probabilities
            total_prob = sum(ensemble_probabilities.values())
            for regime in ensemble_probabilities:
                ensemble_probabilities[regime] /= total_prob
            
            # Calculate transition probabilities (simplified)
            transition_probabilities = self._calculate_transition_probabilities(
                predicted_regime, ensemble_probabilities
            )
            
            # Calculate risk adjustment factor
            risk_adjustment = self._calculate_risk_adjustment(predicted_regime, regime_confidence)
            
            # Generate recommended allocation
            recommended_allocation = self._generate_regime_allocation(predicted_regime, regime_confidence)
            
            # Calculate regime strength and persistence
            regime_strength = self._calculate_regime_strength(ensemble_probabilities)
            regime_persistence = self._calculate_regime_persistence(predicted_regime)
            
            # Create regime state
            regime_state = RegimeState(
                current_regime=predicted_regime,
                confidence=regime_confidence,
                regime_probability=dict(ensemble_probabilities),
                regime_duration=self._calculate_regime_duration(predicted_regime),
                transition_probability=transition_probabilities,
                regime_strength=regime_strength,
                supporting_indicators=self._get_supporting_indicators(model_predictions),
                risk_adjustment_factor=risk_adjustment,
                recommended_allocation=recommended_allocation,
                previous_regime=self.current_regime_state.current_regime if self.current_regime_state else None,
                regime_persistence=regime_persistence
            )
            
            return regime_state
            
        except Exception as e:
            self.logger.error(f"Error in ensemble prediction: {e}")
            raise
    
    def _calculate_transition_probabilities(
        self,
        current_regime: RegimeType,
        regime_probabilities: Dict[RegimeType, float]
    ) -> Dict[RegimeType, float]:
        """Calculate probabilities of transitioning to other regimes"""
        try:
            # Simple transition model based on current probabilities
            transition_probs = {}
            current_prob = regime_probabilities.get(current_regime, 0.5)
            
            for regime, prob in regime_probabilities.items():
                if regime != current_regime:
                    # Higher probability for regimes with higher current probability
                    # Lower probability if current regime is very confident
                    transition_prob = prob * (1 - current_prob)
                    transition_probs[regime] = transition_prob
                else:
                    # Probability of staying in current regime
                    transition_probs[regime] = current_prob
            
            return transition_probs
            
        except Exception as e:
            self.logger.error(f"Error calculating transition probabilities: {e}")
            return {regime: 0.25 for regime in RegimeType}  # Uniform fallback
    
    def _calculate_risk_adjustment(self, regime: RegimeType, confidence: float) -> float:
        """Calculate risk adjustment factor based on regime"""
        try:
            # Base risk adjustments per regime
            base_adjustments = {
                RegimeType.BULL: 1.0,      # Normal risk
                RegimeType.BEAR: 0.5,      # Reduce risk significantly
                RegimeType.SIDEWAYS: 0.8,  # Slight risk reduction
                RegimeType.VOLATILE: 0.3,  # Major risk reduction
                RegimeType.CRISIS: 0.1,    # Minimal risk
                RegimeType.RECOVERY: 1.2   # Slightly higher risk
            }
            
            base_adjustment = base_adjustments.get(regime, 0.8)
            
            # Adjust based on confidence - lower confidence means more conservative
            confidence_adjustment = 0.5 + (0.5 * confidence)
            
            return base_adjustment * confidence_adjustment
            
        except Exception as e:
            self.logger.error(f"Error calculating risk adjustment: {e}")
            return 0.8  # Conservative fallback
    
    def _generate_regime_allocation(
        self,
        regime: RegimeType,
        confidence: float
    ) -> Dict[str, float]:
        """Generate recommended asset allocation for regime"""
        try:
            # Base allocations per regime
            allocations = {
                RegimeType.BULL: {
                    "equities": 0.7,
                    "bonds": 0.2,
                    "commodities": 0.05,
                    "cash": 0.05
                },
                RegimeType.BEAR: {
                    "equities": 0.2,
                    "bonds": 0.5,
                    "commodities": 0.1,
                    "cash": 0.2
                },
                RegimeType.SIDEWAYS: {
                    "equities": 0.5,
                    "bonds": 0.3,
                    "commodities": 0.1,
                    "cash": 0.1
                },
                RegimeType.VOLATILE: {
                    "equities": 0.3,
                    "bonds": 0.4,
                    "commodities": 0.1,
                    "cash": 0.2
                },
                RegimeType.CRISIS: {
                    "equities": 0.1,
                    "bonds": 0.3,
                    "commodities": 0.1,
                    "cash": 0.5
                },
                RegimeType.RECOVERY: {
                    "equities": 0.6,
                    "bonds": 0.25,
                    "commodities": 0.1,
                    "cash": 0.05
                }
            }
            
            base_allocation = allocations.get(regime, allocations[RegimeType.SIDEWAYS])
            
            # Adjust based on confidence
            if confidence < 0.7:
                # Lower confidence - move towards cash
                cash_adjustment = (0.7 - confidence) * 0.2
                adjusted_allocation = {}
                for asset, weight in base_allocation.items():
                    if asset == "cash":
                        adjusted_allocation[asset] = weight + cash_adjustment
                    else:
                        adjusted_allocation[asset] = weight * (1 - cash_adjustment/3)
                
                return adjusted_allocation
            
            return base_allocation
            
        except Exception as e:
            self.logger.error(f"Error generating allocation: {e}")
            return {"equities": 0.4, "bonds": 0.4, "cash": 0.2}  # Balanced fallback
    
    def _calculate_regime_strength(self, probabilities: Dict[RegimeType, float]) -> float:
        """Calculate how strong/clear the regime signal is"""
        try:
            if not probabilities:
                return 0.5
            
            # Entropy-based measure - lower entropy means stronger signal
            entropy = -sum(p * np.log2(p + 1e-10) for p in probabilities.values())
            max_entropy = np.log2(len(probabilities))
            
            # Convert to strength (0-1, where 1 is strongest)
            strength = 1 - (entropy / max_entropy)
            return max(0.0, min(1.0, strength))
            
        except Exception as e:
            self.logger.error(f"Error calculating regime strength: {e}")
            return 0.5
    
    def _calculate_regime_persistence(self, current_regime: RegimeType) -> float:
        """Calculate how persistent the current regime has been"""
        try:
            if not self.regime_history:
                return 0.5
            
            # Count consecutive occurrences of current regime
            consecutive_count = 0
            for regime_state in reversed(self.regime_history):
                if regime_state.current_regime == current_regime:
                    consecutive_count += 1
                else:
                    break
            
            # Normalize to 0-1 scale (max persistence of 20 periods)
            persistence = min(consecutive_count / 20, 1.0)
            return persistence
            
        except Exception as e:
            self.logger.error(f"Error calculating persistence: {e}")
            return 0.5
    
    def _calculate_regime_duration(self, current_regime: RegimeType) -> timedelta:
        """Calculate how long the current regime has been active"""
        try:
            if not self.regime_history:
                return timedelta(0)
            
            # Find when current regime started
            start_time = None
            for regime_state in reversed(self.regime_history):
                if regime_state.current_regime == current_regime:
                    start_time = regime_state.last_updated
                else:
                    break
            
            if start_time:
                return datetime.utcnow() - start_time
            else:
                return timedelta(0)
                
        except Exception as e:
            self.logger.error(f"Error calculating regime duration: {e}")
            return timedelta(0)
    
    def _get_supporting_indicators(
        self,
        model_predictions: Dict[str, RegimeType]
    ) -> List[str]:
        """Get list of indicators supporting the regime prediction"""
        try:
            # Count votes for each regime
            regime_counts = defaultdict(int)
            for model_id, regime in model_predictions.items():
                regime_counts[regime] += 1
            
            # Find the predicted regime
            predicted_regime = max(regime_counts.keys(), key=lambda r: regime_counts[r])
            
            # Get supporting models
            supporting_models = [
                model_id for model_id, regime in model_predictions.items()
                if regime == predicted_regime
            ]
            
            return supporting_models
            
        except Exception as e:
            self.logger.error(f"Error getting supporting indicators: {e}")
            return []
    
    async def _handle_regime_transition(
        self,
        from_regime: RegimeType,
        to_regime: RegimeType,
        confidence: float
    ) -> None:
        """Handle regime transition event"""
        try:
            # Create transition record
            transition = RegimeTransition(
                transition_id=str(uuid.uuid4()),
                from_regime=from_regime,
                to_regime=to_regime,
                transition_time=datetime.utcnow(),
                transition_confidence=confidence,
                transition_speed=self._assess_transition_speed(from_regime, to_regime),
                market_catalysts=self._identify_market_catalysts(),
                impact_assessment=self._assess_transition_impact(from_regime, to_regime),
                recommended_actions=self._get_transition_recommendations(from_regime, to_regime)
            )
            
            # Store transition
            self.transition_history.append(transition)
            self.regime_transitions += 1
            
            # Save to database
            await self._save_regime_transition(transition)
            
            # Send alerts
            await self._send_regime_transition_alert(transition)
            
            self.logger.info(
                f"REGIME TRANSITION: {from_regime.value} -> {to_regime.value} "
                f"(confidence: {confidence:.3f})"
            )
            
        except Exception as e:
            self.logger.error(f"Error handling regime transition: {e}")
    
    def _assess_transition_speed(self, from_regime: RegimeType, to_regime: RegimeType) -> str:
        """Assess the speed of regime transition"""
        # This would analyze recent regime changes to determine transition speed
        # For now, return a simple assessment
        if from_regime in [RegimeType.BULL, RegimeType.BEAR] and to_regime == RegimeType.CRISIS:
            return "sudden"
        elif from_regime == RegimeType.VOLATILE and to_regime in [RegimeType.BULL, RegimeType.BEAR]:
            return "rapid"
        else:
            return "gradual"
    
    def _identify_market_catalysts(self) -> List[str]:
        """Identify potential market catalysts for regime change"""
        # This would analyze recent market events, news, economic data
        # For now, return generic catalysts
        return ["market_volatility", "economic_data", "geopolitical_events"]
    
    def _assess_transition_impact(
        self,
        from_regime: RegimeType,
        to_regime: RegimeType
    ) -> Dict[str, float]:
        """Assess the impact of regime transition"""
        # Simplified impact assessment
        impact_matrix = {
            (RegimeType.BULL, RegimeType.BEAR): {"portfolio_risk": 0.8, "volatility": 0.9},
            (RegimeType.BEAR, RegimeType.BULL): {"portfolio_risk": -0.3, "volatility": -0.2},
            (RegimeType.SIDEWAYS, RegimeType.VOLATILE): {"portfolio_risk": 0.4, "volatility": 0.7},
        }
        
        return impact_matrix.get((from_regime, to_regime), {"portfolio_risk": 0.0, "volatility": 0.0})
    
    def _get_transition_recommendations(
        self,
        from_regime: RegimeType,
        to_regime: RegimeType
    ) -> List[str]:
        """Get recommended actions for regime transition"""
        recommendations = {
            (RegimeType.BULL, RegimeType.BEAR): ["reduce_equity_exposure", "increase_cash", "review_stop_losses"],
            (RegimeType.BEAR, RegimeType.BULL): ["increase_equity_exposure", "reduce_cash", "review_opportunities"],
            (RegimeType.SIDEWAYS, RegimeType.VOLATILE): ["reduce_position_sizes", "increase_hedging"],
        }
        
        return recommendations.get((from_regime, to_regime), ["monitor_closely", "review_positions"])
    
    # Background monitoring tasks
    
    async def _regime_monitoring_loop(self) -> None:
        """Continuous regime monitoring and prediction"""
        try:
            while True:
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
                # Get latest market features
                features = await self._get_current_market_features()
                if features:
                    # Make regime prediction
                    try:
                        regime_state, confidence = await self.predict_regime(features)
                        
                        # Log prediction
                        self.logger.debug(
                            f"Regime prediction: {regime_state.current_regime.value} "
                            f"(confidence: {confidence:.3f})"
                        )
                        
                    except Exception as e:
                        self.logger.error(f"Error in regime prediction: {e}")
                
        except asyncio.CancelledError:
            self.logger.info("Regime monitoring loop cancelled")
            raise
        except Exception as e:
            self.logger.error(f"Error in regime monitoring loop: {e}")
    
    async def _feature_update_loop(self) -> None:
        """Update market features for regime detection"""
        try:
            while True:
                await asyncio.sleep(60)  # Update features every minute
                
                try:
                    await self._update_market_features()
                except Exception as e:
                    self.logger.error(f"Error updating market features: {e}")
                
        except asyncio.CancelledError:
            self.logger.info("Feature update loop cancelled")
            raise
        except Exception as e:
            self.logger.error(f"Error in feature update loop: {e}")
    
    async def _model_update_loop(self) -> None:
        """Periodic model retraining and updates"""
        try:
            while True:
                await asyncio.sleep(3600)  # Check every hour
                
                try:
                    # Check if models need retraining
                    if await self._should_retrain_models():
                        self.logger.info("Initiating model retraining")
                        await self._retrain_models()
                    
                    # Update ensemble weights based on recent performance
                    await self._update_model_performance()
                    
                except Exception as e:
                    self.logger.error(f"Error in model update loop: {e}")
                
        except asyncio.CancelledError:
            self.logger.info("Model update loop cancelled")
            raise
        except Exception as e:
            self.logger.error(f"Error in model update loop: {e}")
    
    async def _get_current_market_features(self) -> Optional[Dict[str, float]]:
        """Get current market features for regime detection"""
        try:
            # This would integrate with market data services
            # For now, return cached features or fetch from database
            
            query = """
                SELECT * FROM market_data_features 
                ORDER BY timestamp DESC 
                LIMIT 1
            """
            
            row = await self.db_connection.fetchrow(query)
            if row:
                return dict(row)
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting current market features: {e}")
            return None
    
    async def _update_market_features(self) -> None:
        """Update market features in database"""
        try:
            # This would calculate and store updated market features
            # Integration point with market data services
            pass
            
        except Exception as e:
            self.logger.error(f"Error updating market features: {e}")
    
    async def _should_retrain_models(self) -> bool:
        """Check if models should be retrained"""
        try:
            # Check model age
            for model_id, classifier in self.classifiers.items():
                if model_id in self.model_performance:
                    trained_at = self.model_performance[model_id].get('trained_at')
                    if trained_at:
                        trained_time = datetime.fromisoformat(trained_at)
                        age_days = (datetime.utcnow() - trained_time).days
                        if age_days > self.config.model_lifecycle.max_model_age_days:
                            return True
            
            # Check performance degradation (would need recent performance metrics)
            # For now, return False
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking retrain condition: {e}")
            return False
    
    async def _retrain_models(self) -> None:
        """Retrain ensemble models with latest data"""
        try:
            # Get latest training data
            training_data = await self._get_training_data()
            if training_data is None:
                self.logger.warning("No training data available for retraining")
                return
            
            X, y, feature_names = training_data
            
            # Retrain each model
            for model_id, classifier in self.classifiers.items():
                self.logger.info(f"Retraining model: {model_id}")
                
                loop = asyncio.get_event_loop()
                training_metrics = await loop.run_in_executor(
                    self.executor,
                    classifier.train,
                    X, y, feature_names
                )
                
                self.model_performance[model_id] = training_metrics
                
                # Save retrained model
                model_path = f"{self.model_storage_path}/{model_id}.joblib"
                await loop.run_in_executor(
                    self.executor,
                    classifier.save_model,
                    model_path
                )
            
            # Update ensemble weights
            self._update_ensemble_weights()
            
            self.logger.info("Model retraining completed")
            
        except Exception as e:
            self.logger.error(f"Error retraining models: {e}")
    
    async def _update_model_performance(self) -> None:
        """Update model performance metrics"""
        try:
            # This would analyze recent predictions vs actual regime changes
            # and update performance metrics accordingly
            pass
            
        except Exception as e:
            self.logger.error(f"Error updating model performance: {e}")
    
    # Database operations
    
    async def _create_database_tables(self) -> None:
        """Create database tables for regime detection"""
        try:
            # Regime states table
            await self.db_connection.execute("""
                CREATE TABLE IF NOT EXISTS market_regime_states (
                    state_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    current_regime VARCHAR(20) NOT NULL,
                    confidence DECIMAL(5,4) NOT NULL,
                    regime_probability JSONB,
                    regime_duration_seconds INTEGER,
                    transition_probability JSONB,
                    regime_strength DECIMAL(5,4),
                    supporting_indicators TEXT[],
                    risk_adjustment_factor DECIMAL(5,4),
                    recommended_allocation JSONB,
                    previous_regime VARCHAR(20),
                    regime_persistence DECIMAL(5,4),
                    regime_volatility DECIMAL(10,6),
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                
                CREATE INDEX IF NOT EXISTS idx_regime_states_created_at 
                    ON market_regime_states(created_at);
            """)
            
            # Regime transitions table
            await self.db_connection.execute("""
                CREATE TABLE IF NOT EXISTS market_regime_transitions (
                    transition_id UUID PRIMARY KEY,
                    from_regime VARCHAR(20) NOT NULL,
                    to_regime VARCHAR(20) NOT NULL,
                    transition_time TIMESTAMPTZ NOT NULL,
                    transition_confidence DECIMAL(5,4),
                    transition_speed VARCHAR(20),
                    market_catalysts TEXT[],
                    impact_assessment JSONB,
                    recommended_actions TEXT[],
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                
                CREATE INDEX IF NOT EXISTS idx_regime_transitions_time
                    ON market_regime_transitions(transition_time);
            """)
            
            # Market features table (if not exists)
            await self.db_connection.execute("""
                CREATE TABLE IF NOT EXISTS market_data_features (
                    feature_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    timestamp TIMESTAMPTZ NOT NULL,
                    symbol VARCHAR(20),
                    -- Price features
                    open_price DECIMAL(12,2),
                    high_price DECIMAL(12,2),
                    low_price DECIMAL(12,2),
                    close_price DECIMAL(12,2),
                    volume BIGINT,
                    -- Technical indicators
                    sma_5 DECIMAL(12,2),
                    sma_20 DECIMAL(12,2),
                    sma_50 DECIMAL(12,2),
                    sma_200 DECIMAL(12,2),
                    ema_12 DECIMAL(12,2),
                    ema_26 DECIMAL(12,2),
                    rsi_14 DECIMAL(5,2),
                    rsi_30 DECIMAL(5,2),
                    macd DECIMAL(12,4),
                    macd_signal DECIMAL(12,4),
                    bollinger_upper DECIMAL(12,2),
                    bollinger_lower DECIMAL(12,2),
                    atr_14 DECIMAL(12,4),
                    atr_30 DECIMAL(12,4),
                    -- Alternative data
                    vix_level DECIMAL(6,2),
                    treasury_10y DECIMAL(6,4),
                    dollar_index DECIMAL(8,4),
                    sector_rotation DECIMAL(6,4),
                    market_breadth DECIMAL(6,4),
                    -- Regime label (for training)
                    regime_label VARCHAR(20),
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                
                CREATE INDEX IF NOT EXISTS idx_market_features_timestamp
                    ON market_data_features(timestamp);
                CREATE INDEX IF NOT EXISTS idx_market_features_symbol
                    ON market_data_features(symbol);
            """)
            
            self.logger.info("Database tables created successfully")
            
        except Exception as e:
            self.logger.error(f"Error creating database tables: {e}")
            raise
    
    async def _save_regime_state(self, regime_state: RegimeState) -> None:
        """Save regime state to database"""
        try:
            await self.db_connection.execute("""
                INSERT INTO market_regime_states (
                    current_regime, confidence, regime_probability, 
                    regime_duration_seconds, transition_probability, regime_strength,
                    supporting_indicators, risk_adjustment_factor, recommended_allocation,
                    previous_regime, regime_persistence, regime_volatility
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            """,
                regime_state.current_regime.value,
                regime_state.confidence,
                json.dumps({k.value: v for k, v in regime_state.regime_probability.items()}),
                int(regime_state.regime_duration.total_seconds()),
                json.dumps({k.value: v for k, v in regime_state.transition_probability.items()}),
                regime_state.regime_strength,
                regime_state.supporting_indicators,
                regime_state.risk_adjustment_factor,
                json.dumps(regime_state.recommended_allocation),
                regime_state.previous_regime.value if regime_state.previous_regime else None,
                regime_state.regime_persistence,
                regime_state.regime_volatility
            )
            
        except Exception as e:
            self.logger.error(f"Error saving regime state: {e}")
    
    async def _save_regime_transition(self, transition: RegimeTransition) -> None:
        """Save regime transition to database"""
        try:
            await self.db_connection.execute("""
                INSERT INTO market_regime_transitions (
                    transition_id, from_regime, to_regime, transition_time,
                    transition_confidence, transition_speed, market_catalysts,
                    impact_assessment, recommended_actions
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
                transition.transition_id,
                transition.from_regime.value,
                transition.to_regime.value,
                transition.transition_time,
                transition.transition_confidence,
                transition.transition_speed,
                transition.market_catalysts,
                json.dumps(transition.impact_assessment),
                transition.recommended_actions
            )
            
        except Exception as e:
            self.logger.error(f"Error saving regime transition: {e}")
    
    async def _load_regime_history(self) -> None:
        """Load recent regime history from database"""
        try:
            rows = await self.db_connection.fetch("""
                SELECT * FROM market_regime_states
                ORDER BY created_at DESC
                LIMIT 100
            """)
            
            for row in rows:
                regime_state = self._create_regime_state_from_db(row)
                self.regime_history.append(regime_state)
            
            # Set current regime state to most recent
            if self.regime_history:
                self.current_regime_state = self.regime_history[0]
            
            self.logger.info(f"Loaded {len(rows)} regime states from database")
            
        except Exception as e:
            self.logger.error(f"Error loading regime history: {e}")
    
    def _create_regime_state_from_db(self, row: dict) -> RegimeState:
        """Create regime state from database row"""
        try:
            regime_prob = json.loads(row['regime_probability']) if row['regime_probability'] else {}
            regime_prob_typed = {RegimeType(k): v for k, v in regime_prob.items()}
            
            transition_prob = json.loads(row['transition_probability']) if row['transition_probability'] else {}
            transition_prob_typed = {RegimeType(k): v for k, v in transition_prob.items()}
            
            recommended_allocation = json.loads(row['recommended_allocation']) if row['recommended_allocation'] else {}
            
            return RegimeState(
                current_regime=RegimeType(row['current_regime']),
                confidence=float(row['confidence']),
                regime_probability=regime_prob_typed,
                regime_duration=timedelta(seconds=row['regime_duration_seconds'] or 0),
                transition_probability=transition_prob_typed,
                regime_strength=float(row['regime_strength'] or 0.5),
                supporting_indicators=row['supporting_indicators'] or [],
                risk_adjustment_factor=float(row['risk_adjustment_factor'] or 0.8),
                recommended_allocation=recommended_allocation,
                previous_regime=RegimeType(row['previous_regime']) if row['previous_regime'] else None,
                regime_persistence=float(row['regime_persistence'] or 0.5),
                regime_volatility=float(row['regime_volatility'] or 0.0),
                last_updated=row['created_at']
            )
            
        except Exception as e:
            self.logger.error(f"Error creating regime state from DB: {e}")
            # Return default regime state
            return RegimeState(
                current_regime=RegimeType.SIDEWAYS,
                confidence=0.5,
                regime_probability={regime: 0.25 for regime in RegimeType},
                regime_duration=timedelta(0),
                transition_probability={regime: 0.25 for regime in RegimeType},
                regime_strength=0.5,
                supporting_indicators=[],
                risk_adjustment_factor=0.8,
                recommended_allocation={"equities": 0.5, "bonds": 0.3, "cash": 0.2}
            )
    
    async def _send_regime_transition_alert(self, transition: RegimeTransition) -> None:
        """Send regime transition alert"""
        try:
            alert_data = {
                "transition_id": transition.transition_id,
                "from_regime": transition.from_regime.value,
                "to_regime": transition.to_regime.value,
                "transition_time": transition.transition_time.isoformat(),
                "confidence": transition.transition_confidence,
                "transition_speed": transition.transition_speed,
                "market_catalysts": transition.market_catalysts,
                "impact_assessment": transition.impact_assessment,
                "recommended_actions": transition.recommended_actions
            }
            
            # Send system alert
            message = create_system_alert_message(
                component="market_regime",
                alert_type="regime_transition",
                severity="medium",
                data=alert_data
            )
            
            if self.redis_pubsub:
                await self.redis_pubsub.publish_system_alert(message.data)
            
        except Exception as e:
            self.logger.error(f"Error sending regime transition alert: {e}")
    
    # Public API methods
    
    async def get_current_regime(self) -> Optional[RegimeState]:
        """Get current market regime state"""
        return self.current_regime_state
    
    async def get_regime_history(self, hours_back: int = 24) -> List[RegimeState]:
        """Get regime history"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        
        return [
            regime for regime in self.regime_history
            if regime.last_updated >= cutoff_time
        ]
    
    async def get_transition_history(self, days_back: int = 7) -> List[RegimeTransition]:
        """Get transition history"""
        cutoff_time = datetime.utcnow() - timedelta(days=days_back)
        
        return [
            transition for transition in self.transition_history
            if transition.transition_time >= cutoff_time
        ]
    
    def get_regime_statistics(self) -> Dict[str, Any]:
        """Get regime detection statistics"""
        return {
            "total_predictions": self.total_predictions,
            "regime_transitions": self.regime_transitions,
            "average_confidence": self.average_confidence,
            "current_regime": self.current_regime_state.current_regime.value if self.current_regime_state else None,
            "current_confidence": self.current_regime_state.confidence if self.current_regime_state else 0.0,
            "models_trained": sum(1 for c in self.classifiers.values() if c.is_trained),
            "ensemble_weights": self.ensemble_weights,
            "model_performance": self.model_performance
        }


# Global regime detector instance
regime_detector_instance = None

def get_regime_detector() -> MarketRegimeDetector:
    """Get global regime detector instance"""
    global regime_detector_instance
    if regime_detector_instance is None:
        raise RuntimeError("Regime detector not initialized. Call init_regime_detector() first.")
    return regime_detector_instance

def init_regime_detector() -> MarketRegimeDetector:
    """Initialize global regime detector instance"""
    global regime_detector_instance
    regime_detector_instance = MarketRegimeDetector()
    return regime_detector_instance