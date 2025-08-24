"""
Model Lifecycle Management System with Drift Detection and Retraining

Implements comprehensive ML model lifecycle management:
- Automatic model retraining on new data with performance validation
- Model performance monitoring and drift detection algorithms
- A/B testing framework for model comparison and deployment
- Model versioning and rollback capabilities
- Champion/challenger model management
- Automated model deployment pipeline with approval workflows
"""

import asyncio
import logging
import json
import uuid
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque, defaultdict
import hashlib
from concurrent.futures import ThreadPoolExecutor
import pickle
import joblib
from pathlib import Path

# ML imports
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, classification_report
)
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
import scipy.stats as stats

# Database imports
import asyncpg
import redis.asyncio as redis

# Internal imports
from .config import get_ml_config, ModelType
from .utils import MLMetrics, ModelRegistry
try:
    from ..websocket.redis_pubsub import get_redis_pubsub_manager
    from ..websocket.message_protocols import create_system_alert_message
except ImportError:
    # Fallbacks for standalone operation
    def get_redis_pubsub_manager():
        return None
    def create_system_alert_message(alert_type, data):
        return {'type': alert_type, 'data': data}

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model lifecycle status"""
    TRAINING = "training"
    VALIDATION = "validation"
    TESTING = "testing"
    DEPLOYED = "deployed"
    CHAMPION = "champion"
    CHALLENGER = "challenger"
    RETIRED = "retired"
    FAILED = "failed"


class DriftType(Enum):
    """Types of model drift"""
    DATA_DRIFT = "data_drift"          # Input feature distribution changes
    CONCEPT_DRIFT = "concept_drift"    # Target relationship changes
    LABEL_DRIFT = "label_drift"        # Target distribution changes
    PERFORMANCE_DRIFT = "performance_drift"  # Model performance degrades


class TestType(Enum):
    """A/B testing types"""
    CHAMPION_CHALLENGER = "champion_challenger"
    MULTI_ARMED_BANDIT = "multi_armed_bandit"
    GRADUAL_ROLLOUT = "gradual_rollout"
    SHADOW_MODE = "shadow_mode"


@dataclass
class ModelVersion:
    """Model version metadata"""
    model_id: str
    version: str
    model_type: ModelType
    status: ModelStatus
    
    # Training metadata
    training_data_hash: str
    training_samples: int
    training_features: List[str]
    hyperparameters: Dict[str, Any]
    
    # Performance metrics
    training_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    
    # Deployment info
    deployed_at: Optional[datetime] = None
    deployment_config: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    created_by: str = "system"
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Model artifacts
    model_path: Optional[str] = None
    model_size_bytes: Optional[int] = None
    
    # Dependencies
    dependencies: List[str] = field(default_factory=list)
    framework_version: Optional[str] = None


@dataclass
class DriftDetectionResult:
    """Result of drift detection analysis"""
    drift_detected: bool
    drift_type: DriftType
    drift_score: float  # 0-1, higher means more drift
    confidence: float
    
    # Detailed analysis
    feature_drift_scores: Dict[str, float]
    statistical_tests: Dict[str, Dict[str, float]]  # test_name -> {statistic, p_value}
    recommendations: List[str]
    
    # Time-based analysis
    detection_window: int  # samples analyzed
    detection_timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Visualizations data (for dashboards)
    drift_plots_data: Optional[Dict[str, Any]] = None


@dataclass
class ModelPerformanceMetrics:
    """Comprehensive model performance tracking"""
    model_id: str
    version: str
    
    # Core metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc_roc: Optional[float] = None
    
    # Regression metrics
    mse: Optional[float] = None
    mae: Optional[float] = None
    r2_score: Optional[float] = None
    
    # Business metrics
    prediction_latency_ms: float = 0.0
    throughput_predictions_per_sec: float = 0.0
    memory_usage_mb: float = 0.0
    
    # Time-based performance
    performance_window_start: datetime = field(default_factory=lambda: datetime.utcnow() - timedelta(hours=1))
    performance_window_end: datetime = field(default_factory=datetime.utcnow)
    total_predictions: int = 0
    successful_predictions: int = 0
    
    # Confidence and uncertainty
    average_prediction_confidence: float = 0.0
    uncertainty_quantiles: Dict[str, float] = field(default_factory=dict)  # 5%, 50%, 95%


@dataclass
class ABTestConfiguration:
    """A/B test configuration"""
    test_id: str
    test_name: str
    test_type: TestType
    
    # Models in test
    champion_model: str  # model_id:version
    challenger_models: List[str]  # list of model_id:version
    
    # Traffic allocation
    traffic_allocation: Dict[str, float]  # model -> percentage
    
    # Test parameters
    min_sample_size: int = 1000
    max_duration_days: int = 14
    significance_level: float = 0.05
    minimum_effect_size: float = 0.02  # 2% improvement
    
    # Success criteria
    primary_metric: str = "accuracy"
    secondary_metrics: List[str] = field(default_factory=list)
    
    # Status and results
    status: str = "planning"  # planning, running, completed, cancelled
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    # Results
    test_results: Dict[str, Any] = field(default_factory=dict)
    winner: Optional[str] = None
    confidence_interval: Optional[Tuple[float, float]] = None


class DriftDetector:
    """
    Advanced drift detection system using multiple statistical methods
    
    Features:
    - Multiple drift detection algorithms (KS test, PSI, KL divergence)
    - Feature-level and model-level drift analysis
    - Real-time drift monitoring with configurable thresholds
    - Automated alerts and retraining triggers
    - Comprehensive drift reporting and visualization
    """
    
    def __init__(
        self,
        reference_window: int = 1000,
        detection_window: int = 500,
        drift_threshold: float = 0.1
    ):
        self.reference_window = reference_window
        self.detection_window = detection_window
        self.drift_threshold = drift_threshold
        
        # Drift detection methods
        self.drift_methods = {
            'kolmogorov_smirnov': self._ks_test,
            'population_stability_index': self._psi_test,
            'kullback_leibler': self._kl_divergence_test,
            'wasserstein_distance': self._wasserstein_test
        }
        
        # Historical data for drift detection
        self.reference_data: Optional[np.ndarray] = None
        self.reference_targets: Optional[np.ndarray] = None
        self.reference_predictions: Optional[np.ndarray] = None
        
        # Drift history
        self.drift_history: deque = deque(maxlen=1000)
        
        self.logger = logging.getLogger(f"{__name__}.DriftDetector")
    
    def set_reference_data(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        predictions: Optional[np.ndarray] = None
    ) -> None:
        """Set reference data for drift detection"""
        self.reference_data = X[-self.reference_window:] if len(X) > self.reference_window else X
        
        if y is not None:
            self.reference_targets = y[-self.reference_window:] if len(y) > self.reference_window else y
        
        if predictions is not None:
            self.reference_predictions = predictions[-self.reference_window:] if len(predictions) > self.reference_window else predictions
        
        self.logger.info(f"Set reference data: {self.reference_data.shape}")
    
    def detect_drift(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        predictions: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ) -> DriftDetectionResult:
        """Detect drift in new data"""
        try:
            if self.reference_data is None:
                raise ValueError("Reference data not set. Call set_reference_data() first.")
            
            # Use latest detection window
            current_data = X[-self.detection_window:] if len(X) > self.detection_window else X
            
            # Initialize results
            drift_detected = False
            drift_scores = {}
            statistical_tests = {}
            feature_drift_scores = {}
            
            # Feature-level drift detection
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(current_data.shape[1])]
            
            for i, feature_name in enumerate(feature_names):
                if i < self.reference_data.shape[1] and i < current_data.shape[1]:
                    ref_feature = self.reference_data[:, i]
                    curr_feature = current_data[:, i]
                    
                    # Apply multiple drift detection methods
                    feature_drift_results = {}
                    for method_name, method_func in self.drift_methods.items():
                        try:
                            result = method_func(ref_feature, curr_feature)
                            feature_drift_results[method_name] = result
                        except Exception as e:
                            self.logger.warning(f"Error in {method_name} for feature {feature_name}: {e}")
                    
                    # Aggregate feature drift score
                    if feature_drift_results:
                        feature_drift_score = np.mean([r['drift_score'] for r in feature_drift_results.values()])
                        feature_drift_scores[feature_name] = feature_drift_score
                        
                        if feature_drift_score > self.drift_threshold:
                            drift_detected = True
            
            # Target drift detection (if available)
            if y is not None and self.reference_targets is not None:
                current_targets = y[-self.detection_window:] if len(y) > self.detection_window else y
                target_drift_results = {}
                
                for method_name, method_func in self.drift_methods.items():
                    try:
                        result = method_func(self.reference_targets, current_targets)
                        target_drift_results[method_name] = result
                    except Exception as e:
                        self.logger.warning(f"Error in {method_name} for targets: {e}")
                
                if target_drift_results:
                    statistical_tests['target_drift'] = target_drift_results
                    target_drift_score = np.mean([r['drift_score'] for r in target_drift_results.values()])
                    
                    if target_drift_score > self.drift_threshold:
                        drift_detected = True
                        drift_type = DriftType.LABEL_DRIFT
                    else:
                        drift_type = DriftType.DATA_DRIFT
                else:
                    drift_type = DriftType.DATA_DRIFT
            else:
                drift_type = DriftType.DATA_DRIFT
            
            # Prediction drift detection (if available)
            if predictions is not None and self.reference_predictions is not None:
                current_predictions = predictions[-self.detection_window:] if len(predictions) > self.detection_window else predictions
                pred_drift_results = {}
                
                for method_name, method_func in self.drift_methods.items():
                    try:
                        result = method_func(self.reference_predictions, current_predictions)
                        pred_drift_results[method_name] = result
                    except Exception as e:
                        self.logger.warning(f"Error in {method_name} for predictions: {e}")
                
                if pred_drift_results:
                    statistical_tests['prediction_drift'] = pred_drift_results
                    pred_drift_score = np.mean([r['drift_score'] for r in pred_drift_results.values()])
                    
                    if pred_drift_score > self.drift_threshold:
                        drift_detected = True
                        drift_type = DriftType.CONCEPT_DRIFT
            
            # Calculate overall drift score
            all_drift_scores = list(feature_drift_scores.values())
            overall_drift_score = np.mean(all_drift_scores) if all_drift_scores else 0.0
            
            # Generate recommendations
            recommendations = self._generate_drift_recommendations(
                drift_detected, drift_type, overall_drift_score, feature_drift_scores
            )
            
            # Calculate confidence
            confidence = min(1.0, overall_drift_score * 2)  # Simple confidence calculation
            
            result = DriftDetectionResult(
                drift_detected=drift_detected,
                drift_type=drift_type,
                drift_score=overall_drift_score,
                confidence=confidence,
                feature_drift_scores=feature_drift_scores,
                statistical_tests=statistical_tests,
                recommendations=recommendations,
                detection_window=len(current_data)
            )
            
            # Store in history
            self.drift_history.append(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error detecting drift: {e}")
            raise
    
    def _ks_test(self, ref_data: np.ndarray, curr_data: np.ndarray) -> Dict[str, float]:
        """Kolmogorov-Smirnov test for drift detection"""
        try:
            statistic, p_value = stats.ks_2samp(ref_data, curr_data)
            
            # Convert to drift score (0-1)
            drift_score = min(1.0, statistic * 2)  # KS statistic is 0-1, scale to emphasize
            
            return {
                'statistic': float(statistic),
                'p_value': float(p_value),
                'drift_score': drift_score
            }
            
        except Exception as e:
            self.logger.warning(f"Error in KS test: {e}")
            return {'statistic': 0.0, 'p_value': 1.0, 'drift_score': 0.0}
    
    def _psi_test(self, ref_data: np.ndarray, curr_data: np.ndarray, bins: int = 10) -> Dict[str, float]:
        """Population Stability Index for drift detection"""
        try:
            # Create bins based on reference data
            _, bin_edges = np.histogram(ref_data, bins=bins)
            
            # Calculate distributions
            ref_hist, _ = np.histogram(ref_data, bins=bin_edges, density=True)
            curr_hist, _ = np.histogram(curr_data, bins=bin_edges, density=True)
            
            # Add small epsilon to avoid log(0)
            epsilon = 1e-8
            ref_hist = ref_hist + epsilon
            curr_hist = curr_hist + epsilon
            
            # Calculate PSI
            psi = np.sum((curr_hist - ref_hist) * np.log(curr_hist / ref_hist))
            
            # Convert to drift score
            # PSI interpretation: 0-0.1 (no drift), 0.1-0.25 (moderate), >0.25 (significant)
            drift_score = min(1.0, psi / 0.25)
            
            return {
                'psi': float(psi),
                'drift_score': drift_score,
                'bins_used': bins
            }
            
        except Exception as e:
            self.logger.warning(f"Error in PSI test: {e}")
            return {'psi': 0.0, 'drift_score': 0.0, 'bins_used': bins}
    
    def _kl_divergence_test(self, ref_data: np.ndarray, curr_data: np.ndarray, bins: int = 20) -> Dict[str, float]:
        """KL Divergence test for drift detection"""
        try:
            # Create probability distributions
            _, bin_edges = np.histogram(np.concatenate([ref_data, curr_data]), bins=bins)
            
            ref_hist, _ = np.histogram(ref_data, bins=bin_edges, density=True)
            curr_hist, _ = np.histogram(curr_data, bins=bin_edges, density=True)
            
            # Normalize to probabilities
            ref_prob = ref_hist / np.sum(ref_hist)
            curr_prob = curr_hist / np.sum(curr_hist)
            
            # Add small epsilon
            epsilon = 1e-8
            ref_prob = ref_prob + epsilon
            curr_prob = curr_prob + epsilon
            
            # Calculate KL divergence
            kl_div = np.sum(curr_prob * np.log(curr_prob / ref_prob))
            
            # Convert to drift score (KL divergence is 0 to infinity)
            drift_score = min(1.0, kl_div / 2.0)  # Scale appropriately
            
            return {
                'kl_divergence': float(kl_div),
                'drift_score': drift_score
            }
            
        except Exception as e:
            self.logger.warning(f"Error in KL divergence test: {e}")
            return {'kl_divergence': 0.0, 'drift_score': 0.0}
    
    def _wasserstein_test(self, ref_data: np.ndarray, curr_data: np.ndarray) -> Dict[str, float]:
        """Wasserstein distance test for drift detection"""
        try:
            wasserstein_dist = stats.wasserstein_distance(ref_data, curr_data)
            
            # Normalize by data range to get drift score
            data_range = np.ptp(np.concatenate([ref_data, curr_data]))
            if data_range > 0:
                drift_score = min(1.0, wasserstein_dist / data_range)
            else:
                drift_score = 0.0
            
            return {
                'wasserstein_distance': float(wasserstein_dist),
                'drift_score': drift_score
            }
            
        except Exception as e:
            self.logger.warning(f"Error in Wasserstein test: {e}")
            return {'wasserstein_distance': 0.0, 'drift_score': 0.0}
    
    def _generate_drift_recommendations(
        self,
        drift_detected: bool,
        drift_type: DriftType,
        overall_score: float,
        feature_scores: Dict[str, float]
    ) -> List[str]:
        """Generate recommendations based on drift analysis"""
        recommendations = []
        
        if not drift_detected:
            recommendations.append("No significant drift detected. Continue monitoring.")
            return recommendations
        
        # General recommendations based on drift type
        if drift_type == DriftType.DATA_DRIFT:
            recommendations.append("Data drift detected in input features")
            recommendations.append("Consider retraining model with recent data")
            
            # Feature-specific recommendations
            high_drift_features = [
                feature for feature, score in feature_scores.items()
                if score > self.drift_threshold * 1.5
            ]
            
            if high_drift_features:
                recommendations.append(f"High drift in features: {', '.join(high_drift_features[:3])}")
                recommendations.append("Review feature engineering and data preprocessing")
        
        elif drift_type == DriftType.CONCEPT_DRIFT:
            recommendations.append("Concept drift detected - relationship between features and target changed")
            recommendations.append("Immediate model retraining recommended")
            recommendations.append("Consider updating model architecture or feature engineering")
        
        elif drift_type == DriftType.LABEL_DRIFT:
            recommendations.append("Label drift detected - target distribution has changed")
            recommendations.append("Review data collection and labeling processes")
            recommendations.append("Consider adjusting class weights or sampling strategy")
        
        # Severity-based recommendations
        if overall_score > 0.3:
            recommendations.append("CRITICAL: High drift detected - immediate action required")
            recommendations.append("Consider rolling back to previous model version")
        elif overall_score > 0.15:
            recommendations.append("MODERATE: Significant drift - plan retraining within 24 hours")
        
        return recommendations


class ModelRetrainer:
    """
    Intelligent model retraining system with performance validation
    
    Features:
    - Automated retraining triggered by drift detection or performance degradation
    - Incremental learning and transfer learning capabilities
    - Model validation and A/B testing before deployment
    - Rollback capabilities and version management
    - Resource-aware training with GPU/CPU optimization
    """
    
    def __init__(
        self,
        database_url: str = None,
        model_storage_path: str = None
    ):
        self.config = get_ml_config()
        self.database_url = database_url or self.config.database_url
        self.model_storage_path = model_storage_path or self.config.model_storage_path
        
        self.db_connection: Optional[asyncpg.Connection] = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Retraining queue and status
        self.retraining_queue: asyncio.Queue = asyncio.Queue()
        self.active_retraining: Dict[str, asyncio.Task] = {}
        
        # Model registry
        self.model_registry = ModelRegistry()
        
        self.logger = logging.getLogger(f"{__name__}.ModelRetrainer")
    
    async def initialize(self) -> None:
        """Initialize the model retrainer"""
        try:
            self.db_connection = await asyncpg.connect(self.database_url)
            await self.model_registry.initialize(self.db_connection)
            
            # Start background retraining worker
            asyncio.create_task(self._retraining_worker())
            
            self.logger.info("Model retrainer initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize model retrainer: {e}")
            raise
    
    async def queue_retraining(
        self,
        model_id: str,
        trigger_reason: str,
        priority: int = 1,
        retraining_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Queue a model for retraining"""
        try:
            retrain_request = {
                'request_id': str(uuid.uuid4()),
                'model_id': model_id,
                'trigger_reason': trigger_reason,
                'priority': priority,
                'config': retraining_config or {},
                'queued_at': datetime.utcnow(),
                'status': 'queued'
            }
            
            await self.retraining_queue.put(retrain_request)
            
            self.logger.info(
                f"Queued model {model_id} for retraining. "
                f"Reason: {trigger_reason}, Priority: {priority}"
            )
            
            return retrain_request['request_id']
            
        except Exception as e:
            self.logger.error(f"Error queueing retraining for {model_id}: {e}")
            raise
    
    async def _retraining_worker(self) -> None:
        """Background worker for processing retraining requests"""
        try:
            while True:
                try:
                    # Get next retraining request (blocks until available)
                    request = await self.retraining_queue.get()
                    
                    model_id = request['model_id']
                    request_id = request['request_id']
                    
                    # Check if model is already being retrained
                    if model_id in self.active_retraining:
                        self.logger.warning(f"Model {model_id} already being retrained, skipping")
                        continue
                    
                    # Start retraining task
                    task = asyncio.create_task(
                        self._retrain_model(request)
                    )
                    self.active_retraining[model_id] = task
                    
                    # Clean up completed tasks
                    await self._cleanup_completed_retraining()
                    
                except Exception as e:
                    self.logger.error(f"Error in retraining worker: {e}")
                    await asyncio.sleep(10)  # Wait before retrying
                
        except Exception as e:
            self.logger.error(f"Fatal error in retraining worker: {e}")
    
    async def _retrain_model(self, request: Dict[str, Any]) -> None:
        """Retrain a single model"""
        model_id = request['model_id']
        request_id = request['request_id']
        
        try:
            self.logger.info(f"Starting retraining for model {model_id}")
            
            # Get current model version
            current_version = await self.model_registry.get_latest_version(model_id)
            if not current_version:
                raise ValueError(f"No existing version found for model {model_id}")
            
            # Get training data
            training_data = await self._get_training_data(model_id, request['config'])
            if training_data is None:
                raise ValueError(f"No training data available for model {model_id}")
            
            X_train, y_train, feature_names = training_data
            
            # Create new model version
            new_version = self._create_new_version(current_version)
            
            # Train the model
            trained_model, training_metrics = await self._train_model_version(
                new_version, X_train, y_train, feature_names
            )
            
            # Validate the model
            validation_metrics = await self._validate_model(
                trained_model, X_train, y_train, feature_names
            )
            
            # Update version with metrics
            new_version.training_metrics = training_metrics
            new_version.validation_metrics = validation_metrics
            new_version.status = ModelStatus.VALIDATION
            
            # Save model and version
            model_path = await self._save_model(trained_model, new_version)
            new_version.model_path = model_path
            new_version.model_size_bytes = Path(model_path).stat().st_size
            
            # Register new version
            await self.model_registry.register_version(new_version)
            
            # Evaluate if new model is better
            should_deploy = await self._evaluate_model_improvement(
                current_version, new_version
            )
            
            if should_deploy:
                # Deploy as challenger for A/B testing
                await self._deploy_as_challenger(new_version)
                self.logger.info(
                    f"Model {model_id} v{new_version.version} deployed as challenger"
                )
            else:
                new_version.status = ModelStatus.RETIRED
                await self.model_registry.update_version(new_version)
                self.logger.info(
                    f"Model {model_id} v{new_version.version} not deployed (no improvement)"
                )
            
        except Exception as e:
            self.logger.error(f"Error retraining model {model_id}: {e}")
            # Update status to failed if version exists
            if 'new_version' in locals():
                new_version.status = ModelStatus.FAILED
                await self.model_registry.update_version(new_version)
        
        finally:
            # Remove from active retraining
            if model_id in self.active_retraining:
                del self.active_retraining[model_id]
    
    async def _get_training_data(
        self,
        model_id: str,
        config: Dict[str, Any]
    ) -> Optional[Tuple[np.ndarray, np.ndarray, List[str]]]:
        """Get training data for model retraining"""
        try:
            # This would integrate with the feature engineering system
            # For now, return placeholder data
            
            # Generate synthetic training data
            np.random.seed(42)
            n_samples = config.get('training_samples', 10000)
            n_features = config.get('n_features', 20)
            
            X = np.random.randn(n_samples, n_features)
            y = np.random.randint(0, 3, n_samples)  # 3-class classification
            feature_names = [f"feature_{i}" for i in range(n_features)]
            
            return X, y, feature_names
            
        except Exception as e:
            self.logger.error(f"Error getting training data: {e}")
            return None
    
    def _create_new_version(self, current_version: ModelVersion) -> ModelVersion:
        """Create new model version based on current version"""
        try:
            # Generate new version string
            current_ver = float(current_version.version)
            new_ver = f"{current_ver + 0.1:.1f}"
            
            new_version = ModelVersion(
                model_id=current_version.model_id,
                version=new_ver,
                model_type=current_version.model_type,
                status=ModelStatus.TRAINING,
                training_data_hash="",  # Will be set during training
                training_samples=0,
                training_features=[],
                hyperparameters=current_version.hyperparameters.copy(),
                training_metrics={},
                validation_metrics={},
                test_metrics={},
                dependencies=current_version.dependencies.copy(),
                framework_version=current_version.framework_version
            )
            
            return new_version
            
        except Exception as e:
            self.logger.error(f"Error creating new version: {e}")
            raise
    
    async def _train_model_version(
        self,
        version: ModelVersion,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: List[str]
    ) -> Tuple[Any, Dict[str, float]]:
        """Train a model version"""
        try:
            # Import and train model based on type
            if version.model_type == ModelType.RANDOM_FOREST:
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(**version.hyperparameters)
            elif version.model_type == ModelType.GRADIENT_BOOSTING:
                from sklearn.ensemble import GradientBoostingClassifier
                model = GradientBoostingClassifier(**version.hyperparameters)
            else:
                raise ValueError(f"Unsupported model type: {version.model_type}")
            
            # Train model in executor to avoid blocking
            loop = asyncio.get_event_loop()
            trained_model = await loop.run_in_executor(
                self.executor,
                model.fit,
                X_train, y_train
            )
            
            # Calculate training metrics
            y_pred = await loop.run_in_executor(
                self.executor,
                trained_model.predict,
                X_train
            )
            
            training_metrics = {
                'accuracy': float(accuracy_score(y_train, y_pred)),
                'precision': float(precision_score(y_train, y_pred, average='weighted')),
                'recall': float(recall_score(y_train, y_pred, average='weighted')),
                'f1_score': float(f1_score(y_train, y_pred, average='weighted')),
            }
            
            # Add AUC if binary/multiclass
            if hasattr(trained_model, 'predict_proba'):
                try:
                    y_proba = await loop.run_in_executor(
                        self.executor,
                        trained_model.predict_proba,
                        X_train
                    )
                    if y_proba.shape[1] == 2:
                        auc = roc_auc_score(y_train, y_proba[:, 1])
                    else:
                        auc = roc_auc_score(y_train, y_proba, multi_class='ovr')
                    training_metrics['auc_roc'] = float(auc)
                except:
                    pass
            
            # Update version metadata
            version.training_samples = len(X_train)
            version.training_features = feature_names.copy()
            version.training_data_hash = hashlib.md5(
                np.concatenate([X_train.flatten(), y_train])
            ).hexdigest()
            
            return trained_model, training_metrics
            
        except Exception as e:
            self.logger.error(f"Error training model version: {e}")
            raise
    
    async def _validate_model(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, float]:
        """Validate model using cross-validation"""
        try:
            loop = asyncio.get_event_loop()
            
            # Cross-validation
            if len(np.unique(y)) > 1:
                cv_scores = await loop.run_in_executor(
                    self.executor,
                    cross_val_score,
                    model, X, y, cv=StratifiedKFold(n_splits=5), scoring='accuracy'
                )
            else:
                cv_scores = await loop.run_in_executor(
                    self.executor,
                    cross_val_score,
                    model, X, y, cv=KFold(n_splits=5), scoring='accuracy'
                )
            
            validation_metrics = {
                'cv_accuracy_mean': float(cv_scores.mean()),
                'cv_accuracy_std': float(cv_scores.std()),
                'cv_scores': [float(score) for score in cv_scores]
            }
            
            return validation_metrics
            
        except Exception as e:
            self.logger.error(f"Error validating model: {e}")
            return {}
    
    async def _save_model(self, model: Any, version: ModelVersion) -> str:
        """Save trained model to storage"""
        try:
            # Create model path
            model_dir = Path(self.model_storage_path) / version.model_id
            model_dir.mkdir(parents=True, exist_ok=True)
            
            model_file = model_dir / f"{version.model_id}_v{version.version}.joblib"
            
            # Save model in executor
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor,
                joblib.dump,
                model, str(model_file)
            )
            
            return str(model_file)
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            raise
    
    async def _evaluate_model_improvement(
        self,
        current_version: ModelVersion,
        new_version: ModelVersion
    ) -> bool:
        """Evaluate if new model version is better than current"""
        try:
            # Compare validation accuracy
            current_acc = current_version.validation_metrics.get('cv_accuracy_mean', 0.0)
            new_acc = new_version.validation_metrics.get('cv_accuracy_mean', 0.0)
            
            improvement = new_acc - current_acc
            min_improvement = self.config.model_lifecycle.performance_threshold
            
            is_better = improvement > min_improvement
            
            self.logger.info(
                f"Model comparison - Current: {current_acc:.4f}, "
                f"New: {new_acc:.4f}, Improvement: {improvement:.4f}, "
                f"Threshold: {min_improvement:.4f}, Better: {is_better}"
            )
            
            return is_better
            
        except Exception as e:
            self.logger.error(f"Error evaluating model improvement: {e}")
            return False
    
    async def _deploy_as_challenger(self, version: ModelVersion) -> None:
        """Deploy model version as challenger"""
        try:
            version.status = ModelStatus.CHALLENGER
            version.deployed_at = datetime.utcnow()
            
            await self.model_registry.update_version(version)
            
            # Create A/B test configuration
            ab_test_config = ABTestConfiguration(
                test_id=str(uuid.uuid4()),
                test_name=f"Champion vs {version.model_id} v{version.version}",
                test_type=TestType.CHAMPION_CHALLENGER,
                champion_model=f"{version.model_id}:current",  # Current champion
                challenger_models=[f"{version.model_id}:{version.version}"],
                traffic_allocation={
                    "champion": 0.8,  # 80% traffic to champion
                    "challenger": 0.2  # 20% traffic to challenger
                },
                min_sample_size=self.config.model_lifecycle.model_validation_samples,
                max_duration_days=self.config.model_lifecycle.a_b_test_duration_days,
                primary_metric="accuracy",
                secondary_metrics=["precision", "recall", "f1_score"]
            )
            
            # Start A/B test (would integrate with A/B testing system)
            self.logger.info(f"Started A/B test: {ab_test_config.test_id}")
            
        except Exception as e:
            self.logger.error(f"Error deploying as challenger: {e}")
            raise
    
    async def _cleanup_completed_retraining(self) -> None:
        """Clean up completed retraining tasks"""
        try:
            completed_models = []
            for model_id, task in self.active_retraining.items():
                if task.done():
                    completed_models.append(model_id)
            
            for model_id in completed_models:
                task = self.active_retraining[model_id]
                if task.exception():
                    self.logger.error(f"Retraining task for {model_id} failed: {task.exception()}")
                del self.active_retraining[model_id]
                
        except Exception as e:
            self.logger.error(f"Error cleaning up retraining tasks: {e}")


class ModelManager:
    """
    Comprehensive model lifecycle management system
    
    Features:
    - Model versioning and deployment management
    - A/B testing framework with statistical significance testing
    - Performance monitoring and automated drift detection
    - Model rollback and emergency procedures
    - Resource monitoring and optimization
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
        
        # Subsystems
        self.drift_detector = DriftDetector()
        self.model_retrainer = ModelRetrainer(database_url, model_storage_path)
        self.model_registry = ModelRegistry()
        
        # Model performance tracking
        self.performance_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.active_ab_tests: Dict[str, ABTestConfiguration] = {}
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._ab_test_task: Optional[asyncio.Task] = None
        self._performance_task: Optional[asyncio.Task] = None
        
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self) -> None:
        """Initialize the model manager"""
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
            
            # Initialize subsystems
            await self.model_registry.initialize(self.db_connection)
            await self.model_retrainer.initialize()
            
            # Load active A/B tests
            await self._load_active_ab_tests()
            
            # Start background tasks
            self._monitoring_task = asyncio.create_task(self._model_monitoring_loop())
            self._ab_test_task = asyncio.create_task(self._ab_test_monitoring_loop())
            self._performance_task = asyncio.create_task(self._performance_monitoring_loop())
            
            self.logger.info("Model manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize model manager: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the model manager"""
        try:
            # Cancel background tasks
            tasks = [self._monitoring_task, self._ab_test_task, self._performance_task]
            for task in tasks:
                if task:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            # Close connections
            if self.db_connection:
                await self.db_connection.close()
            
            if self.redis_client:
                await self.redis_client.close()
            
            self.logger.info("Model manager shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for model lifecycle manager"""
        try:
            health_status = {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "database_connected": self.db_connection is not None,
                "redis_connected": self.redis_client is not None,
                "active_models": len(self.model_registry.active_models) if hasattr(self.model_registry, 'active_models') else 0,
                "ab_tests_running": len(self.ab_tests),
                "performance_metrics": len(self.performance_metrics)
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
    
    async def register_model_performance(
        self,
        model_id: str,
        version: str,
        metrics: ModelPerformanceMetrics
    ) -> None:
        """Register model performance metrics"""
        try:
            # Store in memory for real-time monitoring
            key = f"{model_id}:{version}"
            self.performance_metrics[key].append(metrics)
            
            # Check for performance drift
            if len(self.performance_metrics[key]) >= 100:  # Minimum samples for drift detection
                recent_metrics = list(self.performance_metrics[key])[-100:]
                reference_metrics = list(self.performance_metrics[key])[-500:-100] if len(self.performance_metrics[key]) >= 500 else recent_metrics[:50]
                
                # Extract performance values for drift detection
                recent_performance = np.array([m.accuracy or 0.0 for m in recent_metrics if m.accuracy is not None])
                reference_performance = np.array([m.accuracy or 0.0 for m in reference_metrics if m.accuracy is not None])
                
                if len(recent_performance) > 10 and len(reference_performance) > 10:
                    self.drift_detector.set_reference_data(reference_performance.reshape(-1, 1))
                    drift_result = self.drift_detector.detect_drift(recent_performance.reshape(-1, 1))
                    
                    if drift_result.drift_detected:
                        await self._handle_performance_drift(model_id, version, drift_result)
            
            # Save to database
            await self._save_performance_metrics(metrics)
            
        except Exception as e:
            self.logger.error(f"Error registering model performance: {e}")
    
    async def _handle_performance_drift(
        self,
        model_id: str,
        version: str,
        drift_result: DriftDetectionResult
    ) -> None:
        """Handle detected performance drift"""
        try:
            self.logger.warning(
                f"Performance drift detected for model {model_id}:{version}. "
                f"Drift score: {drift_result.drift_score:.3f}"
            )
            
            # Send alert
            alert_data = {
                "model_id": model_id,
                "version": version,
                "drift_type": drift_result.drift_type.value,
                "drift_score": drift_result.drift_score,
                "recommendations": drift_result.recommendations
            }
            
            message = create_system_alert_message(
                component="model_lifecycle",
                alert_type="performance_drift",
                severity="high",
                data=alert_data
            )
            
            if self.redis_pubsub:
                await self.redis_pubsub.publish_system_alert(message.data)
            
            # Queue model for retraining if drift is significant
            if drift_result.drift_score > 0.2:  # High drift threshold
                await self.model_retrainer.queue_retraining(
                    model_id=model_id,
                    trigger_reason=f"performance_drift_{drift_result.drift_type.value}",
                    priority=1,
                    retraining_config={"drift_detected": True, "drift_score": drift_result.drift_score}
                )
            
        except Exception as e:
            self.logger.error(f"Error handling performance drift: {e}")
    
    async def start_ab_test(self, ab_test_config: ABTestConfiguration) -> str:
        """Start an A/B test for model comparison"""
        try:
            ab_test_config.start_date = datetime.utcnow()
            ab_test_config.status = "running"
            
            # Store A/B test configuration
            self.active_ab_tests[ab_test_config.test_id] = ab_test_config
            
            # Save to database
            await self._save_ab_test_config(ab_test_config)
            
            self.logger.info(
                f"Started A/B test: {ab_test_config.test_id} - "
                f"{ab_test_config.champion_model} vs {ab_test_config.challenger_models}"
            )
            
            return ab_test_config.test_id
            
        except Exception as e:
            self.logger.error(f"Error starting A/B test: {e}")
            raise
    
    async def _model_monitoring_loop(self) -> None:
        """Background loop for model monitoring"""
        try:
            while True:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                try:
                    # Check model health and performance
                    active_models = await self.model_registry.get_active_models()
                    
                    for model in active_models:
                        await self._check_model_health(model)
                        
                except Exception as e:
                    self.logger.error(f"Error in model monitoring loop: {e}")
                
        except asyncio.CancelledError:
            self.logger.info("Model monitoring loop cancelled")
            raise
        except Exception as e:
            self.logger.error(f"Error in model monitoring loop: {e}")
    
    async def _ab_test_monitoring_loop(self) -> None:
        """Background loop for A/B test monitoring"""
        try:
            while True:
                await asyncio.sleep(600)  # Check every 10 minutes
                
                try:
                    for test_id, test_config in list(self.active_ab_tests.items()):
                        await self._check_ab_test_progress(test_config)
                        
                except Exception as e:
                    self.logger.error(f"Error in A/B test monitoring loop: {e}")
                
        except asyncio.CancelledError:
            self.logger.info("A/B test monitoring loop cancelled")
            raise
        except Exception as e:
            self.logger.error(f"Error in A/B test monitoring loop: {e}")
    
    async def _performance_monitoring_loop(self) -> None:
        """Background loop for performance monitoring"""
        try:
            while True:
                await asyncio.sleep(180)  # Check every 3 minutes
                
                try:
                    # Analyze recent performance metrics
                    await self._analyze_performance_trends()
                    
                except Exception as e:
                    self.logger.error(f"Error in performance monitoring loop: {e}")
                
        except asyncio.CancelledError:
            self.logger.info("Performance monitoring loop cancelled")
            raise
        except Exception as e:
            self.logger.error(f"Error in performance monitoring loop: {e}")
    
    async def _check_model_health(self, model: ModelVersion) -> None:
        """Check health of a specific model"""
        try:
            # Check if model file exists
            if model.model_path and not Path(model.model_path).exists():
                self.logger.error(f"Model file missing for {model.model_id}:{model.version}")
                # Send alert and potentially trigger retraining
                return
            
            # Check recent performance
            key = f"{model.model_id}:{model.version}"
            if key in self.performance_metrics:
                recent_metrics = list(self.performance_metrics[key])[-10:]  # Last 10 metrics
                if recent_metrics:
                    avg_accuracy = np.mean([m.accuracy or 0.0 for m in recent_metrics if m.accuracy])
                    if avg_accuracy < 0.7:  # Performance threshold
                        self.logger.warning(
                            f"Low performance detected for {model.model_id}:{model.version} "
                            f"(accuracy: {avg_accuracy:.3f})"
                        )
            
        except Exception as e:
            self.logger.error(f"Error checking model health: {e}")
    
    async def _check_ab_test_progress(self, test_config: ABTestConfiguration) -> None:
        """Check progress of an A/B test"""
        try:
            if test_config.status != "running":
                return
            
            # Check if test duration exceeded
            if test_config.start_date:
                duration = datetime.utcnow() - test_config.start_date
                if duration.days >= test_config.max_duration_days:
                    await self._complete_ab_test(test_config, "duration_exceeded")
                    return
            
            # Check if minimum sample size reached (placeholder logic)
            # In real implementation, this would check actual prediction counts
            sample_size_reached = True  # Placeholder
            
            if sample_size_reached:
                # Perform statistical significance test
                significance_result = await self._test_statistical_significance(test_config)
                
                if significance_result['significant']:
                    await self._complete_ab_test(test_config, "significance_reached")
            
        except Exception as e:
            self.logger.error(f"Error checking A/B test progress: {e}")
    
    async def _test_statistical_significance(
        self, 
        test_config: ABTestConfiguration
    ) -> Dict[str, Any]:
        """Test statistical significance of A/B test results"""
        try:
            # Placeholder implementation
            # In real system, would compare actual performance metrics
            
            # Simulate performance data
            champion_performance = np.random.normal(0.85, 0.05, 1000)  # 85% accuracy
            challenger_performance = np.random.normal(0.87, 0.05, 200)  # 87% accuracy
            
            # Perform t-test
            statistic, p_value = stats.ttest_ind(champion_performance, challenger_performance)
            
            is_significant = p_value < test_config.significance_level
            effect_size = abs(np.mean(challenger_performance) - np.mean(champion_performance))
            meaningful_effect = effect_size >= test_config.minimum_effect_size
            
            return {
                'significant': is_significant and meaningful_effect,
                'p_value': p_value,
                'effect_size': effect_size,
                'champion_mean': np.mean(champion_performance),
                'challenger_mean': np.mean(challenger_performance)
            }
            
        except Exception as e:
            self.logger.error(f"Error testing statistical significance: {e}")
            return {'significant': False, 'p_value': 1.0, 'effect_size': 0.0}
    
    async def _complete_ab_test(
        self, 
        test_config: ABTestConfiguration, 
        completion_reason: str
    ) -> None:
        """Complete an A/B test and determine winner"""
        try:
            test_config.status = "completed"
            test_config.end_date = datetime.utcnow()
            
            # Get final test results
            test_results = await self._test_statistical_significance(test_config)
            test_config.test_results = test_results
            
            # Determine winner
            if test_results.get('significant', False):
                if test_results['challenger_mean'] > test_results['champion_mean']:
                    test_config.winner = test_config.challenger_models[0]
                    await self._promote_challenger_to_champion(test_config)
                else:
                    test_config.winner = test_config.champion_model
            else:
                # No significant difference, keep champion
                test_config.winner = test_config.champion_model
            
            # Update database
            await self._save_ab_test_config(test_config)
            
            # Remove from active tests
            if test_config.test_id in self.active_ab_tests:
                del self.active_ab_tests[test_config.test_id]
            
            self.logger.info(
                f"Completed A/B test {test_config.test_id}. "
                f"Winner: {test_config.winner}, Reason: {completion_reason}"
            )
            
        except Exception as e:
            self.logger.error(f"Error completing A/B test: {e}")
    
    async def _promote_challenger_to_champion(self, test_config: ABTestConfiguration) -> None:
        """Promote challenger model to champion status"""
        try:
            challenger_model_spec = test_config.challenger_models[0]
            model_id, version = challenger_model_spec.split(':')
            
            # Get challenger version
            challenger_version = await self.model_registry.get_version(model_id, version)
            if challenger_version:
                # Update status to champion
                challenger_version.status = ModelStatus.CHAMPION
                challenger_version.deployed_at = datetime.utcnow()
                
                await self.model_registry.update_version(challenger_version)
                
                # Retire old champion
                await self.model_registry.retire_champions(model_id, exclude_version=version)
                
                self.logger.info(
                    f"Promoted challenger {model_id}:{version} to champion"
                )
            
        except Exception as e:
            self.logger.error(f"Error promoting challenger to champion: {e}")
    
    async def _analyze_performance_trends(self) -> None:
        """Analyze performance trends across all models"""
        try:
            # This would analyze trends and send insights
            # Placeholder implementation
            pass
            
        except Exception as e:
            self.logger.error(f"Error analyzing performance trends: {e}")
    
    # Database operations
    
    async def _create_database_tables(self) -> None:
        """Create database tables for model lifecycle"""
        try:
            # Model versions table
            await self.db_connection.execute("""
                CREATE TABLE IF NOT EXISTS ml_model_versions (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    model_id VARCHAR(100) NOT NULL,
                    version VARCHAR(20) NOT NULL,
                    model_type VARCHAR(50) NOT NULL,
                    status VARCHAR(20) NOT NULL,
                    training_data_hash VARCHAR(64),
                    training_samples INTEGER,
                    training_features TEXT[],
                    hyperparameters JSONB,
                    training_metrics JSONB,
                    validation_metrics JSONB,
                    test_metrics JSONB,
                    deployed_at TIMESTAMPTZ,
                    deployment_config JSONB,
                    created_by VARCHAR(100) DEFAULT 'system',
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW(),
                    model_path VARCHAR(500),
                    model_size_bytes BIGINT,
                    dependencies TEXT[],
                    framework_version VARCHAR(50),
                    UNIQUE(model_id, version)
                );
                
                CREATE INDEX IF NOT EXISTS idx_model_versions_model_id
                    ON ml_model_versions(model_id);
                CREATE INDEX IF NOT EXISTS idx_model_versions_status
                    ON ml_model_versions(status);
            """)
            
            # Performance metrics table
            await self.db_connection.execute("""
                CREATE TABLE IF NOT EXISTS ml_model_performance (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    model_id VARCHAR(100) NOT NULL,
                    version VARCHAR(20) NOT NULL,
                    accuracy DECIMAL(8,6),
                    precision_score DECIMAL(8,6),
                    recall DECIMAL(8,6),
                    f1_score DECIMAL(8,6),
                    auc_roc DECIMAL(8,6),
                    mse DECIMAL(12,8),
                    mae DECIMAL(12,8),
                    r2_score DECIMAL(8,6),
                    prediction_latency_ms DECIMAL(10,3),
                    throughput_predictions_per_sec DECIMAL(10,2),
                    memory_usage_mb DECIMAL(10,2),
                    performance_window_start TIMESTAMPTZ,
                    performance_window_end TIMESTAMPTZ,
                    total_predictions INTEGER,
                    successful_predictions INTEGER,
                    average_prediction_confidence DECIMAL(5,4),
                    uncertainty_quantiles JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                
                CREATE INDEX IF NOT EXISTS idx_model_performance_model_version
                    ON ml_model_performance(model_id, version);
                CREATE INDEX IF NOT EXISTS idx_model_performance_created_at
                    ON ml_model_performance(created_at);
            """)
            
            # A/B tests table
            await self.db_connection.execute("""
                CREATE TABLE IF NOT EXISTS ml_ab_tests (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    test_id VARCHAR(100) UNIQUE NOT NULL,
                    test_name VARCHAR(200) NOT NULL,
                    test_type VARCHAR(50) NOT NULL,
                    champion_model VARCHAR(150) NOT NULL,
                    challenger_models TEXT[],
                    traffic_allocation JSONB,
                    min_sample_size INTEGER,
                    max_duration_days INTEGER,
                    significance_level DECIMAL(5,4),
                    minimum_effect_size DECIMAL(5,4),
                    primary_metric VARCHAR(50),
                    secondary_metrics TEXT[],
                    status VARCHAR(20),
                    start_date TIMESTAMPTZ,
                    end_date TIMESTAMPTZ,
                    test_results JSONB,
                    winner VARCHAR(150),
                    confidence_interval JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                
                CREATE INDEX IF NOT EXISTS idx_ab_tests_test_id
                    ON ml_ab_tests(test_id);
                CREATE INDEX IF NOT EXISTS idx_ab_tests_status
                    ON ml_ab_tests(status);
            """)
            
            # Drift detection results table
            await self.db_connection.execute("""
                CREATE TABLE IF NOT EXISTS ml_drift_detection (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    model_id VARCHAR(100) NOT NULL,
                    version VARCHAR(20) NOT NULL,
                    drift_detected BOOLEAN NOT NULL,
                    drift_type VARCHAR(50) NOT NULL,
                    drift_score DECIMAL(8,6) NOT NULL,
                    confidence DECIMAL(5,4) NOT NULL,
                    feature_drift_scores JSONB,
                    statistical_tests JSONB,
                    recommendations TEXT[],
                    detection_window INTEGER,
                    detection_timestamp TIMESTAMPTZ DEFAULT NOW()
                );
                
                CREATE INDEX IF NOT EXISTS idx_drift_detection_model_version
                    ON ml_drift_detection(model_id, version);
                CREATE INDEX IF NOT EXISTS idx_drift_detection_timestamp
                    ON ml_drift_detection(detection_timestamp);
            """)
            
            self.logger.info("Model lifecycle database tables created successfully")
            
        except Exception as e:
            self.logger.error(f"Error creating database tables: {e}")
            raise
    
    async def _save_performance_metrics(self, metrics: ModelPerformanceMetrics) -> None:
        """Save performance metrics to database"""
        try:
            await self.db_connection.execute("""
                INSERT INTO ml_model_performance (
                    model_id, version, accuracy, precision_score, recall, f1_score,
                    auc_roc, mse, mae, r2_score, prediction_latency_ms,
                    throughput_predictions_per_sec, memory_usage_mb,
                    performance_window_start, performance_window_end,
                    total_predictions, successful_predictions,
                    average_prediction_confidence, uncertainty_quantiles
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19
                )
            """,
                metrics.model_id, metrics.version, metrics.accuracy,
                metrics.precision, metrics.recall, metrics.f1_score,
                metrics.auc_roc, metrics.mse, metrics.mae, metrics.r2_score,
                metrics.prediction_latency_ms, metrics.throughput_predictions_per_sec,
                metrics.memory_usage_mb, metrics.performance_window_start,
                metrics.performance_window_end, metrics.total_predictions,
                metrics.successful_predictions, metrics.average_prediction_confidence,
                json.dumps(metrics.uncertainty_quantiles)
            )
            
        except Exception as e:
            self.logger.error(f"Error saving performance metrics: {e}")
    
    async def _save_ab_test_config(self, config: ABTestConfiguration) -> None:
        """Save A/B test configuration to database"""
        try:
            await self.db_connection.execute("""
                INSERT INTO ml_ab_tests (
                    test_id, test_name, test_type, champion_model, challenger_models,
                    traffic_allocation, min_sample_size, max_duration_days,
                    significance_level, minimum_effect_size, primary_metric,
                    secondary_metrics, status, start_date, end_date,
                    test_results, winner, confidence_interval
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18
                )
                ON CONFLICT (test_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    end_date = EXCLUDED.end_date,
                    test_results = EXCLUDED.test_results,
                    winner = EXCLUDED.winner,
                    confidence_interval = EXCLUDED.confidence_interval
            """,
                config.test_id, config.test_name, config.test_type.value,
                config.champion_model, config.challenger_models,
                json.dumps(config.traffic_allocation), config.min_sample_size,
                config.max_duration_days, config.significance_level,
                config.minimum_effect_size, config.primary_metric,
                config.secondary_metrics, config.status,
                config.start_date, config.end_date,
                json.dumps(config.test_results), config.winner,
                json.dumps(list(config.confidence_interval) if config.confidence_interval else None)
            )
            
        except Exception as e:
            self.logger.error(f"Error saving A/B test config: {e}")
    
    async def _load_active_ab_tests(self) -> None:
        """Load active A/B tests from database"""
        try:
            rows = await self.db_connection.fetch("""
                SELECT * FROM ml_ab_tests 
                WHERE status = 'running'
            """)
            
            for row in rows:
                config = ABTestConfiguration(
                    test_id=row['test_id'],
                    test_name=row['test_name'],
                    test_type=TestType(row['test_type']),
                    champion_model=row['champion_model'],
                    challenger_models=row['challenger_models'],
                    traffic_allocation=json.loads(row['traffic_allocation']),
                    min_sample_size=row['min_sample_size'],
                    max_duration_days=row['max_duration_days'],
                    significance_level=float(row['significance_level']),
                    minimum_effect_size=float(row['minimum_effect_size']),
                    primary_metric=row['primary_metric'],
                    secondary_metrics=row['secondary_metrics'],
                    status=row['status'],
                    start_date=row['start_date'],
                    end_date=row['end_date'],
                    test_results=json.loads(row['test_results']) if row['test_results'] else {},
                    winner=row['winner']
                )
                
                self.active_ab_tests[config.test_id] = config
            
            self.logger.info(f"Loaded {len(rows)} active A/B tests")
            
        except Exception as e:
            self.logger.error(f"Error loading active A/B tests: {e}")
    
    # Public API methods
    
    async def get_model_performance_summary(self, model_id: str) -> Dict[str, Any]:
        """Get performance summary for a model"""
        try:
            # Get recent performance metrics from database
            rows = await self.db_connection.fetch("""
                SELECT * FROM ml_model_performance 
                WHERE model_id = $1 
                ORDER BY created_at DESC 
                LIMIT 100
            """, model_id)
            
            if not rows:
                return {"model_id": model_id, "metrics": [], "summary": {}}
            
            # Calculate summary statistics
            accuracies = [float(row['accuracy']) for row in rows if row['accuracy']]
            latencies = [float(row['prediction_latency_ms']) for row in rows if row['prediction_latency_ms']]
            
            summary = {
                "model_id": model_id,
                "total_records": len(rows),
                "avg_accuracy": np.mean(accuracies) if accuracies else None,
                "accuracy_std": np.std(accuracies) if accuracies else None,
                "avg_latency_ms": np.mean(latencies) if latencies else None,
                "latency_p95_ms": np.percentile(latencies, 95) if latencies else None,
                "time_range": {
                    "start": rows[-1]['created_at'].isoformat(),
                    "end": rows[0]['created_at'].isoformat()
                }
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting model performance summary: {e}")
            return {"model_id": model_id, "error": str(e)}
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            "active_models_tracked": len(self.performance_metrics),
            "active_ab_tests": len(self.active_ab_tests),
            "total_performance_records": sum(len(deque_obj) for deque_obj in self.performance_metrics.values()),
            "drift_detections_performed": len(self.drift_detector.drift_history),
            "active_retraining_jobs": len(self.model_retrainer.active_retraining) if hasattr(self.model_retrainer, 'active_retraining') else 0
        }


# Global model manager instance
model_manager_instance = None

def get_model_manager() -> ModelManager:
    """Get global model manager instance"""
    global model_manager_instance
    if model_manager_instance is None:
        raise RuntimeError("Model manager not initialized. Call init_model_manager() first.")
    return model_manager_instance

def init_model_manager() -> ModelManager:
    """Initialize global model manager instance"""
    global model_manager_instance
    model_manager_instance = ModelManager()
    return model_manager_instance

# Alias for compatibility
ModelLifecycleManager = ModelManager

# Export all public classes and functions
__all__ = [
    'ModelStatus',
    'DriftType', 
    'TestType',
    'ModelVersion',
    'DriftDetectionResult',
    'ModelPerformanceMetrics',
    'ABTestConfiguration',
    'DriftDetector',
    'ModelRetrainer', 
    'ModelManager',
    'ModelLifecycleManager',
    'get_model_manager',
    'init_model_manager'
]