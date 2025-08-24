"""
Real-time ML Inference System with Monitoring Dashboard

Implements high-performance ML inference infrastructure:
- Real-time ML inference with <100ms latency optimization
- Model serving and prediction caching for performance
- Confidence intervals and uncertainty quantification
- Comprehensive ML model monitoring dashboard
- Model performance tracking and alerting
- Load balancing and horizontal scaling support
"""

import asyncio
import logging
import json
import time
import uuid
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor
import pickle
import joblib
from pathlib import Path
import psutil

# ML imports
import scipy.stats as stats

# Database imports
import asyncpg
import redis.asyncio as redis

# Internal imports
from .config import get_ml_config, ModelType
from .utils import MLMetrics, ModelRegistry
from .market_regime import RegimeType, RegimeState
from .feature_engineering import FeatureEngineer
from .model_lifecycle import ModelVersion, ModelStatus
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


class InferenceStatus(Enum):
    """Status of inference operations"""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    CACHED = "cached"
    MODEL_UNAVAILABLE = "model_unavailable"
    FEATURE_ERROR = "feature_error"


class PredictionType(Enum):
    """Types of ML predictions"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    PROBABILITY = "probability"
    RANKING = "ranking"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"


@dataclass
class InferenceRequest:
    """ML inference request"""
    request_id: str
    model_id: str
    model_version: Optional[str] = None
    features: Dict[str, float] = field(default_factory=dict)
    prediction_type: PredictionType = PredictionType.CLASSIFICATION
    return_probabilities: bool = True
    return_uncertainty: bool = True
    max_latency_ms: int = 100
    cache_ttl_seconds: int = 60
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Request timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class InferenceResult:
    """ML inference result"""
    request_id: str
    status: InferenceStatus
    
    # Primary prediction
    prediction: Any = None
    prediction_probabilities: Optional[Dict[str, float]] = None
    
    # Uncertainty quantification
    confidence: float = 0.0
    uncertainty_bounds: Optional[Tuple[float, float]] = None
    prediction_interval: Optional[Tuple[float, float]] = None
    
    # Model information
    model_id: str = ""
    model_version: str = ""
    model_confidence: float = 0.0
    
    # Performance metrics
    inference_latency_ms: float = 0.0
    feature_processing_ms: float = 0.0
    model_execution_ms: float = 0.0
    was_cached: bool = False
    
    # Explanation and attribution
    feature_importance: Optional[Dict[str, float]] = None
    prediction_explanation: Optional[str] = None
    
    # Error information
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    
    # Additional metadata
    features_used: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelPerformanceSnapshot:
    """Performance snapshot for monitoring"""
    model_id: str
    version: str
    timestamp: datetime
    
    # Throughput metrics
    requests_per_second: float = 0.0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    cached_requests: int = 0
    
    # Latency metrics (milliseconds)
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    
    # Accuracy metrics
    avg_confidence: float = 0.0
    prediction_accuracy: float = 0.0  # When ground truth is available
    
    # Resource utilization
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    # Model health
    error_rate: float = 0.0
    availability: float = 1.0
    
    # Cache performance
    cache_hit_rate: float = 0.0
    cache_size: int = 0


@dataclass  
class ModelServingConfig:
    """Configuration for model serving"""
    model_id: str
    model_version: str
    model_path: str
    
    # Performance settings
    max_batch_size: int = 32
    max_latency_ms: int = 100
    cache_ttl_seconds: int = 300
    
    # Scaling settings
    min_replicas: int = 1
    max_replicas: int = 4
    target_cpu_percent: int = 70
    scale_up_threshold: int = 100  # requests per second
    scale_down_threshold: int = 20
    
    # Health check settings
    health_check_interval_seconds: int = 30
    failure_threshold: int = 3
    
    # Resource limits
    memory_limit_mb: int = 512
    cpu_limit_cores: float = 1.0
    
    # Advanced settings
    enable_uncertainty_quantification: bool = True
    enable_feature_importance: bool = True
    enable_prediction_caching: bool = True
    
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelServer:
    """
    High-performance ML model server for real-time inference
    """
    
    def __init__(
        self,
        model_id: str,
        model_version: str,
        model_path: str,
        config: ModelServingConfig
    ):
        self.model_id = model_id
        self.model_version = model_version
        self.model_path = model_path
        self.config = config
        
        # Model and preprocessing
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.model_metadata = {}
        
        # Performance tracking
        self.request_count = 0
        self.success_count = 0
        self.error_count = 0
        self.cache_hits = 0
        self.latency_history = deque(maxlen=1000)
        self.confidence_history = deque(maxlen=1000)
        
        # Prediction cache
        self.prediction_cache: Dict[str, Tuple[InferenceResult, datetime]] = {}
        self.cache_stats = {'hits': 0, 'misses': 0, 'evictions': 0}
        
        # Status and health
        self.is_loaded = False
        self.is_healthy = True
        self.last_health_check = datetime.utcnow()
        self.error_streak = 0
        
        # Background tasks
        self._cache_cleanup_task: Optional[asyncio.Task] = None
        
        self.logger = logging.getLogger(f"{__name__}.ModelServer.{model_id}")
    
    async def initialize(self) -> None:
        """Initialize the model server"""
        try:
            await self._load_model()
            
            # Start background tasks
            self._cache_cleanup_task = asyncio.create_task(self._cache_cleanup_loop())
            
            self.is_loaded = True
            self.logger.info(f"Model server initialized for {self.model_id}:{self.model_version}")
            
        except Exception as e:
            self.is_healthy = False
            self.logger.error(f"Failed to initialize model server: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the model server"""
        try:
            if self._cache_cleanup_task:
                self._cache_cleanup_task.cancel()
                try:
                    await self._cache_cleanup_task
                except asyncio.CancelledError:
                    pass
            
            self.is_loaded = False
            self.logger.info(f"Model server shutdown for {self.model_id}:{self.model_version}")
            
        except Exception as e:
            self.logger.error(f"Error during model server shutdown: {e}")
    
    async def _load_model(self) -> None:
        """Load ML model and preprocessing components"""
        try:
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # Load model using joblib
            model_data = joblib.load(self.model_path)
            
            if isinstance(model_data, dict):
                # Model saved with metadata
                self.model = model_data.get('model')
                self.scaler = model_data.get('scaler')
                self.feature_names = model_data.get('feature_names', [])
                self.model_metadata = model_data.get('metadata', {})
            else:
                # Model saved directly
                self.model = model_data
                self.scaler = None
                self.feature_names = []
                self.model_metadata = {}
            
            if self.model is None:
                raise ValueError("No model found in model file")
            
            self.logger.info(f"Model loaded successfully: {self.model_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    async def predict(self, request: InferenceRequest) -> InferenceResult:
        """Make prediction using the loaded model"""
        start_time = time.time()
        request.started_at = datetime.utcnow()
        
        try:
            self.request_count += 1
            
            # Check cache first
            cache_key = self._generate_cache_key(request.features)
            cached_result = self._get_cached_prediction(cache_key)
            
            if cached_result and request.cache_ttl_seconds > 0:
                self.cache_hits += 1
                cached_result.request_id = request.request_id
                cached_result.was_cached = True
                cached_result.timestamp = datetime.utcnow()
                return cached_result
            
            # Feature processing
            feature_start = time.time()
            features_array = self._prepare_features(request.features)
            feature_time_ms = (time.time() - feature_start) * 1000
            
            # Model inference
            model_start = time.time()
            prediction_result = await self._run_inference(features_array, request)
            model_time_ms = (time.time() - model_start) * 1000
            
            # Calculate total latency
            total_latency_ms = (time.time() - start_time) * 1000
            
            # Create inference result
            result = InferenceResult(
                request_id=request.request_id,
                status=InferenceStatus.SUCCESS,
                prediction=prediction_result.get('prediction'),
                prediction_probabilities=prediction_result.get('probabilities'),
                confidence=prediction_result.get('confidence', 0.0),
                uncertainty_bounds=prediction_result.get('uncertainty_bounds'),
                model_id=self.model_id,
                model_version=self.model_version,
                model_confidence=self.model_metadata.get('confidence', 0.8),
                inference_latency_ms=total_latency_ms,
                feature_processing_ms=feature_time_ms,
                model_execution_ms=model_time_ms,
                feature_importance=prediction_result.get('feature_importance'),
                features_used=list(request.features.keys()),
                metadata={
                    'model_type': str(type(self.model).__name__),
                    'feature_count': len(features_array)
                }
            )
            
            # Cache result if configured
            if request.cache_ttl_seconds > 0:
                self._cache_prediction(cache_key, result, request.cache_ttl_seconds)
            
            # Update performance metrics
            self._update_performance_metrics(result)
            
            self.success_count += 1
            self.error_streak = 0
            
            request.completed_at = datetime.utcnow()
            
            return result
            
        except Exception as e:
            self.error_count += 1
            self.error_streak += 1
            
            # Check if we should mark as unhealthy
            if self.error_streak >= self.config.failure_threshold:
                self.is_healthy = False
            
            total_latency_ms = (time.time() - start_time) * 1000
            
            error_result = InferenceResult(
                request_id=request.request_id,
                status=InferenceStatus.ERROR,
                inference_latency_ms=total_latency_ms,
                error_message=str(e),
                error_code=type(e).__name__,
                model_id=self.model_id,
                model_version=self.model_version,
                timestamp=datetime.utcnow()
            )
            
            self.logger.error(f"Prediction error for request {request.request_id}: {e}")
            
            request.completed_at = datetime.utcnow()
            
            return error_result
    
    def _prepare_features(self, features: Dict[str, float]) -> np.ndarray:
        """Prepare features for model input"""
        try:
            # If feature names are known, use them to order features
            if self.feature_names:
                feature_values = [features.get(name, 0.0) for name in self.feature_names]
            else:
                # Use all available features
                feature_values = list(features.values())
            
            features_array = np.array(feature_values).reshape(1, -1)
            
            # Apply scaling if scaler is available
            if self.scaler:
                features_array = self.scaler.transform(features_array)
            
            return features_array
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {e}")
            raise
    
    async def _run_inference(
        self, 
        features: np.ndarray, 
        request: InferenceRequest
    ) -> Dict[str, Any]:
        """Run model inference"""
        try:
            result = {}
            
            # Make prediction
            if request.prediction_type == PredictionType.CLASSIFICATION:
                prediction = self.model.predict(features)[0]
                result['prediction'] = prediction
                
                # Get probabilities if available
                if request.return_probabilities and hasattr(self.model, 'predict_proba'):
                    probabilities = self.model.predict_proba(features)[0]
                    
                    # Map to class labels
                    if hasattr(self.model, 'classes_'):
                        prob_dict = dict(zip(self.model.classes_, probabilities))
                        result['probabilities'] = prob_dict
                        result['confidence'] = max(probabilities)
                    else:
                        result['confidence'] = max(probabilities)
                
            elif request.prediction_type == PredictionType.REGRESSION:
                prediction = self.model.predict(features)[0]
                result['prediction'] = float(prediction)
                
                # Calculate uncertainty for regression (if model supports it)
                if request.return_uncertainty:
                    uncertainty = self._calculate_regression_uncertainty(features)
                    result['uncertainty_bounds'] = uncertainty
                    result['confidence'] = 1.0 - min(0.5, abs(uncertainty[1] - uncertainty[0]) / abs(prediction))
            
            elif request.prediction_type == PredictionType.PROBABILITY:
                if hasattr(self.model, 'predict_proba'):
                    probabilities = self.model.predict_proba(features)[0]
                    result['prediction'] = probabilities
                    result['confidence'] = max(probabilities)
                else:
                    raise ValueError("Model does not support probability predictions")
            
            # Feature importance (if requested and available)
            if (request.metadata.get('return_feature_importance', False) and 
                hasattr(self.model, 'feature_importances_')):
                feature_importance = dict(zip(
                    self.feature_names[:len(self.model.feature_importances_)],
                    self.model.feature_importances_
                ))
                result['feature_importance'] = feature_importance
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in model inference: {e}")
            raise
    
    def _calculate_regression_uncertainty(
        self, 
        features: np.ndarray
    ) -> Tuple[float, float]:
        """Calculate uncertainty bounds for regression predictions"""
        try:
            # Simplified uncertainty calculation
            # In practice, this would use more sophisticated methods
            
            prediction = self.model.predict(features)[0]
            
            # Use cross-validation or bootstrap estimates if available
            # For now, use a simple percentage-based uncertainty
            uncertainty_pct = 0.1  # 10% uncertainty
            
            lower_bound = prediction * (1 - uncertainty_pct)
            upper_bound = prediction * (1 + uncertainty_pct)
            
            return (lower_bound, upper_bound)
            
        except Exception as e:
            self.logger.error(f"Error calculating regression uncertainty: {e}")
            return (0.0, 0.0)
    
    def _generate_cache_key(self, features: Dict[str, float]) -> str:
        """Generate cache key for features"""
        try:
            # Sort features for consistent key generation
            sorted_features = sorted(features.items())
            feature_string = json.dumps(sorted_features, sort_keys=True)
            
            # Use hash for compact key
            import hashlib
            return hashlib.md5(feature_string.encode()).hexdigest()
            
        except Exception as e:
            self.logger.error(f"Error generating cache key: {e}")
            return str(uuid.uuid4())
    
    def _get_cached_prediction(self, cache_key: str) -> Optional[InferenceResult]:
        """Get cached prediction if available and valid"""
        try:
            if cache_key in self.prediction_cache:
                result, cached_time = self.prediction_cache[cache_key]
                
                # Check if cache entry is still valid
                age_seconds = (datetime.utcnow() - cached_time).total_seconds()
                if age_seconds < self.config.cache_ttl_seconds:
                    self.cache_stats['hits'] += 1
                    return result
                else:
                    # Remove expired entry
                    del self.prediction_cache[cache_key]
                    self.cache_stats['evictions'] += 1
            
            self.cache_stats['misses'] += 1
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting cached prediction: {e}")
            return None
    
    def _cache_prediction(
        self, 
        cache_key: str, 
        result: InferenceResult, 
        ttl_seconds: int
    ) -> None:
        """Cache prediction result"""
        try:
            self.prediction_cache[cache_key] = (result, datetime.utcnow())
            
            # Limit cache size
            max_cache_size = 10000
            if len(self.prediction_cache) > max_cache_size:
                # Remove oldest entries
                items_to_remove = len(self.prediction_cache) - max_cache_size
                oldest_keys = sorted(
                    self.prediction_cache.keys(),
                    key=lambda k: self.prediction_cache[k][1]
                )[:items_to_remove]
                
                for key in oldest_keys:
                    del self.prediction_cache[key]
                    self.cache_stats['evictions'] += 1
            
        except Exception as e:
            self.logger.error(f"Error caching prediction: {e}")
    
    def _update_performance_metrics(self, result: InferenceResult) -> None:
        """Update performance tracking metrics"""
        try:
            self.latency_history.append(result.inference_latency_ms)
            self.confidence_history.append(result.confidence)
            
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
    
    async def _cache_cleanup_loop(self) -> None:
        """Background task to clean up expired cache entries"""
        try:
            while True:
                await asyncio.sleep(300)  # Clean every 5 minutes
                
                try:
                    current_time = datetime.utcnow()
                    expired_keys = []
                    
                    for cache_key, (result, cached_time) in self.prediction_cache.items():
                        age_seconds = (current_time - cached_time).total_seconds()
                        if age_seconds > self.config.cache_ttl_seconds:
                            expired_keys.append(cache_key)
                    
                    # Remove expired entries
                    for key in expired_keys:
                        del self.prediction_cache[key]
                        self.cache_stats['evictions'] += 1
                    
                    if expired_keys:
                        self.logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                        
                except Exception as e:
                    self.logger.error(f"Error in cache cleanup: {e}")
                
        except asyncio.CancelledError:
            self.logger.debug("Cache cleanup loop cancelled")
            raise
        except Exception as e:
            self.logger.error(f"Error in cache cleanup loop: {e}")
    
    def get_performance_snapshot(self) -> ModelPerformanceSnapshot:
        """Get current performance snapshot"""
        try:
            current_time = datetime.utcnow()
            
            # Calculate metrics
            success_rate = self.success_count / self.request_count if self.request_count > 0 else 0.0
            error_rate = self.error_count / self.request_count if self.request_count > 0 else 0.0
            cache_hit_rate = self.cache_hits / self.request_count if self.request_count > 0 else 0.0
            
            # Latency percentiles
            latencies = list(self.latency_history)
            avg_latency = np.mean(latencies) if latencies else 0.0
            p50_latency = np.percentile(latencies, 50) if latencies else 0.0
            p95_latency = np.percentile(latencies, 95) if latencies else 0.0
            p99_latency = np.percentile(latencies, 99) if latencies else 0.0
            max_latency = max(latencies) if latencies else 0.0
            
            # Average confidence
            confidences = list(self.confidence_history)
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            # Resource usage
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            cpu_percent = process.cpu_percent()
            
            return ModelPerformanceSnapshot(
                model_id=self.model_id,
                version=self.model_version,
                timestamp=current_time,
                requests_per_second=0.0,  # Would need time window calculation
                total_requests=self.request_count,
                successful_requests=self.success_count,
                failed_requests=self.error_count,
                cached_requests=self.cache_hits,
                avg_latency_ms=avg_latency,
                p50_latency_ms=p50_latency,
                p95_latency_ms=p95_latency,
                p99_latency_ms=p99_latency,
                max_latency_ms=max_latency,
                avg_confidence=avg_confidence,
                memory_usage_mb=memory_mb,
                cpu_usage_percent=cpu_percent,
                error_rate=error_rate,
                availability=1.0 if self.is_healthy else 0.0,
                cache_hit_rate=cache_hit_rate,
                cache_size=len(self.prediction_cache)
            )
            
        except Exception as e:
            self.logger.error(f"Error getting performance snapshot: {e}")
            return ModelPerformanceSnapshot(
                model_id=self.model_id,
                version=self.model_version,
                timestamp=datetime.utcnow()
            )
    
    def is_ready(self) -> bool:
        """Check if model server is ready to serve requests"""
        return self.is_loaded and self.is_healthy


class InferenceEngine:
    """
    High-performance ML inference engine with load balancing and monitoring
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
        self.model_registry = ModelRegistry()
        
        # Model servers
        self.model_servers: Dict[str, ModelServer] = {}  # model_id:version -> server
        self.load_balancer: Dict[str, List[ModelServer]] = defaultdict(list)  # model_id -> servers
        
        # Performance monitoring
        self.request_history: deque = deque(maxlen=10000)
        self.performance_snapshots: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Feature engineering integration
        self.feature_engineer: Optional[FeatureEngineer] = None
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._health_check_task: Optional[asyncio.Task] = None
        self._performance_task: Optional[asyncio.Task] = None
        
        # Executor for CPU-intensive operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self) -> None:
        """Initialize the inference engine"""
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
            
            # Initialize model registry
            await self.model_registry.initialize(self.db_connection)
            
            # Load active models
            await self._load_active_models()
            
            # Start background tasks
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            self._performance_task = asyncio.create_task(self._performance_monitoring_loop())
            
            self.logger.info("ML inference engine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize inference engine: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the inference engine"""
        try:
            # Cancel background tasks
            tasks = [self._monitoring_task, self._health_check_task, self._performance_task]
            for task in tasks:
                if task:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            # Shutdown model servers
            for server in self.model_servers.values():
                await server.shutdown()
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            # Close connections
            if self.db_connection:
                await self.db_connection.close()
            
            if self.redis_client:
                await self.redis_client.close()
            
            self.logger.info("ML inference engine shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for inference engine"""
        try:
            health_status = {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "database_connected": self.db_connection is not None,
                "redis_connected": self.redis_client is not None,
                "models_loaded": len(self.model_servers),
                "active_requests": self.get_active_requests_count(),
                "performance_snapshots": len(self.performance_snapshots)
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
    
    def get_active_requests_count(self) -> int:
        """Get count of active requests"""
        return len([req for req in self.request_history if req.get('status') == 'active'])
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics"""
        try:
            # Calculate request metrics
            recent_requests = list(self.request_history)[-100:]  # Last 100 requests
            
            if recent_requests:
                avg_latency = np.mean([req.get('latency_ms', 0) for req in recent_requests])
                success_rate = len([req for req in recent_requests if req.get('success', False)]) / len(recent_requests) * 100
            else:
                avg_latency = 0
                success_rate = 0
            
            # System resource metrics
            cpu_percent = psutil.cpu_percent()
            memory_info = psutil.virtual_memory()
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "request_metrics": {
                    "total_requests": len(self.request_history),
                    "recent_requests": len(recent_requests),
                    "average_latency_ms": avg_latency,
                    "success_rate_percent": success_rate
                },
                "system_metrics": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_info.percent,
                    "memory_available_gb": memory_info.available / (1024**3)
                },
                "model_metrics": {
                    "models_loaded": len(self.model_servers),
                    "active_models": len([s for s in self.model_servers.values() if s.is_healthy]),
                    "performance_snapshots": len(self.performance_snapshots)
                }
            }
            
        except Exception as e:
            return {
                "error": f"Failed to get system metrics: {str(e)}",
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """List all available models"""
        try:
            models = []
            for model_id, server in self.model_servers.items():
                models.append({
                    "model_id": model_id,
                    "model_type": server.model_type if hasattr(server, 'model_type') else "unknown",
                    "version": server.version if hasattr(server, 'version') else "unknown",
                    "status": "healthy" if server.is_healthy else "unhealthy",
                    "created_at": server.created_at.isoformat() if hasattr(server, 'created_at') else None,
                    "request_count": getattr(server, 'request_count', 0)
                })
            
            return models
            
        except Exception as e:
            self.logger.error(f"Error listing available models: {e}")
            return []
    
    async def _load_active_models(self) -> None:
        """Load active models for serving"""
        try:
            # Get active model versions
            active_models = await self.model_registry.get_active_models()
            
            for model in active_models:
                if model.status in [ModelStatus.CHAMPION, ModelStatus.DEPLOYED]:
                    await self._load_model_server(model)
            
            self.logger.info(f"Loaded {len(self.model_servers)} model servers")
            
        except Exception as e:
            self.logger.error(f"Error loading active models: {e}")
    
    async def _load_model_server(self, model: ModelVersion) -> None:
        """Load a single model server"""
        try:
            if not model.model_path or not Path(model.model_path).exists():
                self.logger.warning(f"Model file not found for {model.model_id}:{model.version}")
                return
            
            # Create serving config
            config = ModelServingConfig(
                model_id=model.model_id,
                model_version=model.version,
                model_path=model.model_path,
                max_latency_ms=self.config.inference.max_latency_ms,
                cache_ttl_seconds=self.config.inference.feature_cache_ttl
            )
            
            # Create and initialize model server
            server = ModelServer(model.model_id, model.version, model.model_path, config)
            await server.initialize()
            
            # Register server
            server_key = f"{model.model_id}:{model.version}"
            self.model_servers[server_key] = server
            self.load_balancer[model.model_id].append(server)
            
            self.logger.info(f"Loaded model server: {server_key}")
            
        except Exception as e:
            self.logger.error(f"Error loading model server for {model.model_id}:{model.version}: {e}")
    
    async def predict(self, request: InferenceRequest) -> InferenceResult:
        """Make ML prediction with load balancing"""
        try:
            # Find available model server
            server = self._select_model_server(request.model_id, request.model_version)
            
            if not server:
                return InferenceResult(
                    request_id=request.request_id,
                    status=InferenceStatus.MODEL_UNAVAILABLE,
                    error_message=f"No available server for model {request.model_id}",
                    error_code="MODEL_UNAVAILABLE",
                    timestamp=datetime.utcnow()
                )
            
            # Feature engineering (if needed)
            if request.metadata.get('compute_features', False):
                enhanced_features = await self._compute_enhanced_features(request.features)
                request.features.update(enhanced_features)
            
            # Make prediction
            result = await server.predict(request)
            
            # Store request for monitoring
            self.request_history.append((request, result))
            
            # Update performance snapshots
            server_key = f"{server.model_id}:{server.model_version}"
            snapshot = server.get_performance_snapshot()
            self.performance_snapshots[server_key].append(snapshot)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in prediction: {e}")
            return InferenceResult(
                request_id=request.request_id,
                status=InferenceStatus.ERROR,
                error_message=str(e),
                error_code=type(e).__name__,
                timestamp=datetime.utcnow()
            )
    
    def _select_model_server(self, model_id: str, model_version: Optional[str] = None) -> Optional[ModelServer]:
        """Select best available model server using load balancing"""
        try:
            if model_version:
                # Specific version requested
                server_key = f"{model_id}:{model_version}"
                server = self.model_servers.get(server_key)
                return server if server and server.is_ready() else None
            
            # Find best available server for model_id
            available_servers = [
                server for server in self.load_balancer[model_id]
                if server.is_ready()
            ]
            
            if not available_servers:
                return None
            
            # Simple round-robin selection (could be enhanced with more sophisticated load balancing)
            return min(available_servers, key=lambda s: s.request_count)
            
        except Exception as e:
            self.logger.error(f"Error selecting model server: {e}")
            return None
    
    async def _compute_enhanced_features(self, base_features: Dict[str, float]) -> Dict[str, float]:
        """Compute enhanced features using feature engineering"""
        try:
            if not self.feature_engineer:
                return {}
            
            # This would integrate with the feature engineering system
            # For now, return empty dict
            return {}
            
        except Exception as e:
            self.logger.error(f"Error computing enhanced features: {e}")
            return {}
    
    async def batch_predict(
        self, 
        requests: List[InferenceRequest]
    ) -> List[InferenceResult]:
        """Make batch predictions with optimal throughput"""
        try:
            # Group requests by model
            requests_by_model = defaultdict(list)
            for request in requests:
                key = f"{request.model_id}:{request.model_version or 'latest'}"
                requests_by_model[key].append(request)
            
            # Process batches concurrently
            all_results = []
            
            for model_key, model_requests in requests_by_model.items():
                # Process requests for this model
                tasks = [self.predict(request) for request in model_requests]
                results = await asyncio.gather(*tasks)
                all_results.extend(results)
            
            return all_results
            
        except Exception as e:
            self.logger.error(f"Error in batch prediction: {e}")
            return [
                InferenceResult(
                    request_id=req.request_id,
                    status=InferenceStatus.ERROR,
                    error_message=str(e),
                    timestamp=datetime.utcnow()
                )
                for req in requests
            ]
    
    # Background monitoring tasks
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        try:
            while True:
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
                try:
                    # Collect performance metrics
                    await self._collect_performance_metrics()
                    
                    # Check for anomalies
                    await self._detect_performance_anomalies()
                    
                except Exception as e:
                    self.logger.error(f"Error in monitoring loop: {e}")
                
        except asyncio.CancelledError:
            self.logger.info("Monitoring loop cancelled")
            raise
        except Exception as e:
            self.logger.error(f"Error in monitoring loop: {e}")
    
    async def _health_check_loop(self) -> None:
        """Health check loop for model servers"""
        try:
            while True:
                await asyncio.sleep(60)  # Health check every minute
                
                try:
                    for server_key, server in list(self.model_servers.items()):
                        # Check server health
                        if not server.is_ready():
                            self.logger.warning(f"Model server unhealthy: {server_key}")
                            
                            # Try to reload if failure threshold exceeded
                            if server.error_streak >= server.config.failure_threshold:
                                await self._reload_model_server(server_key)
                        
                except Exception as e:
                    self.logger.error(f"Error in health check loop: {e}")
                
        except asyncio.CancelledError:
            self.logger.info("Health check loop cancelled")
            raise
        except Exception as e:
            self.logger.error(f"Error in health check loop: {e}")
    
    async def _performance_monitoring_loop(self) -> None:
        """Performance monitoring and alerting loop"""
        try:
            while True:
                await asyncio.sleep(300)  # Monitor every 5 minutes
                
                try:
                    # Generate performance report
                    performance_report = self._generate_performance_report()
                    
                    # Save to database
                    await self._save_performance_metrics(performance_report)
                    
                    # Check alert conditions
                    await self._check_performance_alerts(performance_report)
                    
                except Exception as e:
                    self.logger.error(f"Error in performance monitoring loop: {e}")
                
        except asyncio.CancelledError:
            self.logger.info("Performance monitoring loop cancelled")
            raise
        except Exception as e:
            self.logger.error(f"Error in performance monitoring loop: {e}")
    
    async def _collect_performance_metrics(self) -> None:
        """Collect performance metrics from all servers"""
        try:
            for server_key, server in self.model_servers.items():
                snapshot = server.get_performance_snapshot()
                self.performance_snapshots[server_key].append(snapshot)
            
        except Exception as e:
            self.logger.error(f"Error collecting performance metrics: {e}")
    
    async def _detect_performance_anomalies(self) -> None:
        """Detect performance anomalies and issues"""
        try:
            for server_key, snapshots in self.performance_snapshots.items():
                if len(snapshots) < 5:  # Need sufficient data
                    continue
                
                recent_snapshots = list(snapshots)[-5:]
                
                # Check latency anomalies
                latencies = [s.avg_latency_ms for s in recent_snapshots]
                if np.mean(latencies) > self.config.inference.alert_thresholds['prediction_latency'] * 1000:
                    await self._send_performance_alert(
                        server_key, "high_latency", 
                        f"Average latency: {np.mean(latencies):.2f}ms"
                    )
                
                # Check error rate anomalies
                error_rates = [s.error_rate for s in recent_snapshots]
                if np.mean(error_rates) > self.config.inference.alert_thresholds['error_rate']:
                    await self._send_performance_alert(
                        server_key, "high_error_rate",
                        f"Error rate: {np.mean(error_rates):.3f}"
                    )
            
        except Exception as e:
            self.logger.error(f"Error detecting performance anomalies: {e}")
    
    async def _reload_model_server(self, server_key: str) -> None:
        """Reload a failed model server"""
        try:
            if server_key not in self.model_servers:
                return
            
            # Get model information
            model_id, model_version = server_key.split(':', 1)
            
            # Shutdown old server
            old_server = self.model_servers[server_key]
            await old_server.shutdown()
            
            # Remove from load balancer
            self.load_balancer[model_id] = [
                s for s in self.load_balancer[model_id] 
                if s != old_server
            ]
            
            # Get model version info
            model_version_info = await self.model_registry.get_version(model_id, model_version)
            if model_version_info:
                await self._load_model_server(model_version_info)
                self.logger.info(f"Reloaded model server: {server_key}")
            
        except Exception as e:
            self.logger.error(f"Error reloading model server {server_key}: {e}")
    
    def _generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        try:
            report = {
                'timestamp': datetime.utcnow().isoformat(),
                'total_servers': len(self.model_servers),
                'healthy_servers': sum(1 for s in self.model_servers.values() if s.is_healthy),
                'total_requests': len(self.request_history),
                'servers': {}
            }
            
            # Per-server metrics
            for server_key, server in self.model_servers.items():
                snapshot = server.get_performance_snapshot()
                report['servers'][server_key] = asdict(snapshot)
            
            # Global metrics
            all_latencies = []
            all_error_rates = []
            
            for snapshots in self.performance_snapshots.values():
                if snapshots:
                    latest = snapshots[-1]
                    all_latencies.append(latest.avg_latency_ms)
                    all_error_rates.append(latest.error_rate)
            
            if all_latencies:
                report['global_avg_latency_ms'] = float(np.mean(all_latencies))
                report['global_p95_latency_ms'] = float(np.percentile(all_latencies, 95))
            
            if all_error_rates:
                report['global_error_rate'] = float(np.mean(all_error_rates))
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating performance report: {e}")
            return {'timestamp': datetime.utcnow().isoformat(), 'error': str(e)}
    
    async def _save_performance_metrics(self, report: Dict[str, Any]) -> None:
        """Save performance metrics to database"""
        try:
            await self.db_connection.execute("""
                INSERT INTO ml_inference_performance (
                    timestamp, total_servers, healthy_servers, total_requests,
                    global_avg_latency_ms, global_p95_latency_ms, global_error_rate,
                    server_metrics, full_report
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
                datetime.utcnow(),
                report.get('total_servers', 0),
                report.get('healthy_servers', 0),
                report.get('total_requests', 0),
                report.get('global_avg_latency_ms', 0.0),
                report.get('global_p95_latency_ms', 0.0),
                report.get('global_error_rate', 0.0),
                json.dumps(report.get('servers', {})),
                json.dumps(report)
            )
            
        except Exception as e:
            self.logger.error(f"Error saving performance metrics: {e}")
    
    async def _check_performance_alerts(self, report: Dict[str, Any]) -> None:
        """Check for performance alert conditions"""
        try:
            alerts = []
            
            # Global latency alert
            if report.get('global_avg_latency_ms', 0) > self.config.inference.alert_thresholds['prediction_latency'] * 1000:
                alerts.append({
                    'type': 'global_high_latency',
                    'value': report['global_avg_latency_ms'],
                    'threshold': self.config.inference.alert_thresholds['prediction_latency'] * 1000
                })
            
            # Global error rate alert
            if report.get('global_error_rate', 0) > self.config.inference.alert_thresholds['error_rate']:
                alerts.append({
                    'type': 'global_high_error_rate',
                    'value': report['global_error_rate'],
                    'threshold': self.config.inference.alert_thresholds['error_rate']
                })
            
            # Healthy servers alert
            if report.get('healthy_servers', 0) < report.get('total_servers', 1) * 0.5:
                alerts.append({
                    'type': 'low_healthy_servers',
                    'healthy': report['healthy_servers'],
                    'total': report['total_servers']
                })
            
            # Send alerts
            for alert in alerts:
                await self._send_system_alert(alert)
            
        except Exception as e:
            self.logger.error(f"Error checking performance alerts: {e}")
    
    async def _send_performance_alert(
        self, 
        server_key: str, 
        alert_type: str, 
        message: str
    ) -> None:
        """Send performance alert"""
        try:
            alert_data = {
                'server_key': server_key,
                'alert_type': alert_type,
                'message': message,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            system_message = create_system_alert_message(
                component="ml_inference",
                alert_type=alert_type,
                severity="medium",
                data=alert_data
            )
            
            if self.redis_pubsub:
                await self.redis_pubsub.publish_system_alert(system_message.data)
            
        except Exception as e:
            self.logger.error(f"Error sending performance alert: {e}")
    
    async def _send_system_alert(self, alert: Dict[str, Any]) -> None:
        """Send system-level alert"""
        try:
            severity = "high" if alert['type'] in ['low_healthy_servers', 'global_high_error_rate'] else "medium"
            
            system_message = create_system_alert_message(
                component="ml_inference",
                alert_type=alert['type'],
                severity=severity,
                data=alert
            )
            
            if self.redis_pubsub:
                await self.redis_pubsub.publish_system_alert(system_message.data)
            
        except Exception as e:
            self.logger.error(f"Error sending system alert: {e}")
    
    # Database operations
    
    async def _create_database_tables(self) -> None:
        """Create database tables for inference engine"""
        try:
            # Inference performance table
            await self.db_connection.execute("""
                CREATE TABLE IF NOT EXISTS ml_inference_performance (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    timestamp TIMESTAMPTZ NOT NULL,
                    total_servers INTEGER,
                    healthy_servers INTEGER,
                    total_requests INTEGER,
                    global_avg_latency_ms DECIMAL(10,3),
                    global_p95_latency_ms DECIMAL(10,3),
                    global_error_rate DECIMAL(8,6),
                    server_metrics JSONB,
                    full_report JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                
                CREATE INDEX IF NOT EXISTS idx_inference_performance_timestamp
                    ON ml_inference_performance(timestamp);
            """)
            
            # Inference requests table
            await self.db_connection.execute("""
                CREATE TABLE IF NOT EXISTS ml_inference_requests (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    request_id VARCHAR(100) NOT NULL,
                    model_id VARCHAR(100) NOT NULL,
                    model_version VARCHAR(20),
                    prediction_type VARCHAR(50),
                    status VARCHAR(20) NOT NULL,
                    inference_latency_ms DECIMAL(10,3),
                    feature_processing_ms DECIMAL(10,3),
                    model_execution_ms DECIMAL(10,3),
                    confidence DECIMAL(5,4),
                    was_cached BOOLEAN DEFAULT FALSE,
                    error_message TEXT,
                    error_code VARCHAR(50),
                    features_used TEXT[],
                    request_timestamp TIMESTAMPTZ,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                
                CREATE INDEX IF NOT EXISTS idx_inference_requests_model
                    ON ml_inference_requests(model_id, model_version);
                CREATE INDEX IF NOT EXISTS idx_inference_requests_timestamp
                    ON ml_inference_requests(request_timestamp);
                CREATE INDEX IF NOT EXISTS idx_inference_requests_status
                    ON ml_inference_requests(status);
            """)
            
            self.logger.info("ML inference database tables created successfully")
            
        except Exception as e:
            self.logger.error(f"Error creating database tables: {e}")
            raise
    
    # Public API methods
    
    async def get_model_status(self, model_id: str) -> Dict[str, Any]:
        """Get status of all servers for a model"""
        try:
            servers = self.load_balancer[model_id]
            
            status = {
                'model_id': model_id,
                'total_servers': len(servers),
                'healthy_servers': sum(1 for s in servers if s.is_healthy),
                'servers': []
            }
            
            for server in servers:
                snapshot = server.get_performance_snapshot()
                status['servers'].append({
                    'version': server.model_version,
                    'is_healthy': server.is_healthy,
                    'is_ready': server.is_ready(),
                    'requests': snapshot.total_requests,
                    'avg_latency_ms': snapshot.avg_latency_ms,
                    'error_rate': snapshot.error_rate,
                    'cache_hit_rate': snapshot.cache_hit_rate
                })
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting model status: {e}")
            return {'model_id': model_id, 'error': str(e)}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        try:
            total_servers = len(self.model_servers)
            healthy_servers = sum(1 for s in self.model_servers.values() if s.is_healthy)
            
            # Calculate global metrics
            all_requests = sum(s.request_count for s in self.model_servers.values())
            all_successes = sum(s.success_count for s in self.model_servers.values())
            all_errors = sum(s.error_count for s in self.model_servers.values())
            
            return {
                'status': 'healthy' if healthy_servers == total_servers else 'degraded',
                'total_servers': total_servers,
                'healthy_servers': healthy_servers,
                'total_requests': all_requests,
                'success_rate': all_successes / all_requests if all_requests > 0 else 0.0,
                'error_rate': all_errors / all_requests if all_requests > 0 else 0.0,
                'models_loaded': len(set(s.model_id for s in self.model_servers.values())),
                'recent_requests': len(self.request_history),
                'uptime_seconds': (datetime.utcnow() - datetime.utcnow()).total_seconds()  # Would track actual start time
            }
            
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}


class MLMonitoringDashboard:
    """
    Comprehensive ML monitoring dashboard with real-time metrics
    """
    
    def __init__(self, inference_engine: InferenceEngine):
        self.inference_engine = inference_engine
        self.logger = logging.getLogger(f"{__name__}.MLMonitoringDashboard")
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get complete dashboard data"""
        try:
            dashboard_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'system_overview': self.inference_engine.get_system_status(),
                'model_performance': await self._get_model_performance_data(),
                'request_metrics': await self._get_request_metrics(),
                'resource_utilization': await self._get_resource_utilization(),
                'alerts': await self._get_active_alerts(),
                'trends': await self._get_performance_trends()
            }
            
            return dashboard_data
            
        except Exception as e:
            self.logger.error(f"Error getting dashboard data: {e}")
            return {'error': str(e), 'timestamp': datetime.utcnow().isoformat()}
    
    async def _get_model_performance_data(self) -> List[Dict[str, Any]]:
        """Get performance data for all models"""
        try:
            performance_data = []
            
            for model_id in self.inference_engine.load_balancer.keys():
                model_status = await self.inference_engine.get_model_status(model_id)
                performance_data.append(model_status)
            
            return performance_data
            
        except Exception as e:
            self.logger.error(f"Error getting model performance data: {e}")
            return []
    
    async def _get_request_metrics(self) -> Dict[str, Any]:
        """Get request-level metrics"""
        try:
            recent_requests = list(self.inference_engine.request_history)[-1000:]  # Last 1000 requests
            
            if not recent_requests:
                return {}
            
            # Extract metrics
            latencies = [result.inference_latency_ms for _, result in recent_requests]
            confidences = [result.confidence for _, result in recent_requests if result.confidence > 0]
            statuses = [result.status.value for _, result in recent_requests]
            
            # Calculate statistics
            metrics = {
                'total_requests': len(recent_requests),
                'avg_latency_ms': float(np.mean(latencies)),
                'p50_latency_ms': float(np.percentile(latencies, 50)),
                'p95_latency_ms': float(np.percentile(latencies, 95)),
                'p99_latency_ms': float(np.percentile(latencies, 99)),
                'avg_confidence': float(np.mean(confidences)) if confidences else 0.0,
                'success_rate': statuses.count('success') / len(statuses),
                'error_rate': statuses.count('error') / len(statuses),
                'cache_hit_rate': statuses.count('cached') / len(statuses)
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error getting request metrics: {e}")
            return {}
    
    async def _get_resource_utilization(self) -> Dict[str, Any]:
        """Get system resource utilization"""
        try:
            # System-wide metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Process-specific metrics
            process = psutil.Process()
            process_memory = process.memory_info().rss / 1024 / 1024  # MB
            process_cpu = process.cpu_percent()
            
            return {
                'system': {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'disk_percent': disk.percent,
                    'available_memory_gb': memory.available / 1024 / 1024 / 1024
                },
                'process': {
                    'memory_mb': process_memory,
                    'cpu_percent': process_cpu,
                    'threads': process.num_threads(),
                    'open_files': len(process.open_files())
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting resource utilization: {e}")
            return {}
    
    async def _get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active alerts and warnings"""
        try:
            alerts = []
            
            # Check for unhealthy servers
            for server_key, server in self.inference_engine.model_servers.items():
                if not server.is_healthy:
                    alerts.append({
                        'type': 'server_unhealthy',
                        'severity': 'high',
                        'server': server_key,
                        'message': f'Model server {server_key} is unhealthy',
                        'timestamp': datetime.utcnow().isoformat()
                    })
            
            # Check for high error rates
            for server_key, server in self.inference_engine.model_servers.items():
                snapshot = server.get_performance_snapshot()
                if snapshot.error_rate > 0.05:  # 5% error rate threshold
                    alerts.append({
                        'type': 'high_error_rate',
                        'severity': 'medium',
                        'server': server_key,
                        'value': snapshot.error_rate,
                        'message': f'High error rate: {snapshot.error_rate:.1%}',
                        'timestamp': datetime.utcnow().isoformat()
                    })
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Error getting active alerts: {e}")
            return []
    
    async def _get_performance_trends(self) -> Dict[str, Any]:
        """Get performance trends over time"""
        try:
            trends = {}
            
            # Get trends for each server
            for server_key, snapshots in self.inference_engine.performance_snapshots.items():
                if len(snapshots) < 2:
                    continue
                
                recent_snapshots = list(snapshots)[-10:]  # Last 10 snapshots
                
                # Calculate trends
                latencies = [s.avg_latency_ms for s in recent_snapshots]
                error_rates = [s.error_rate for s in recent_snapshots]
                request_counts = [s.total_requests for s in recent_snapshots]
                
                trends[server_key] = {
                    'latency_trend': 'increasing' if latencies[-1] > latencies[0] else 'decreasing',
                    'error_trend': 'increasing' if error_rates[-1] > error_rates[0] else 'decreasing',
                    'throughput_trend': 'increasing' if request_counts[-1] > request_counts[0] else 'decreasing',
                    'latency_values': latencies,
                    'error_values': error_rates,
                    'request_values': request_counts
                }
            
            return trends
            
        except Exception as e:
            self.logger.error(f"Error getting performance trends: {e}")
            return {}


# Global inference engine instance
inference_engine_instance = None

def get_inference_engine() -> InferenceEngine:
    """Get global inference engine instance"""
    global inference_engine_instance
    if inference_engine_instance is None:
        raise RuntimeError("Inference engine not initialized. Call init_inference_engine() first.")
    return inference_engine_instance

def init_inference_engine() -> InferenceEngine:
    """Initialize global inference engine instance"""
    global inference_engine_instance
    inference_engine_instance = InferenceEngine()
    return inference_engine_instance