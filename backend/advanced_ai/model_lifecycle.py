"""
AI Model Lifecycle Management with Continuous Learning
Implementation of automated model deployment, monitoring, and continuous learning pipelines
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import logging
import asyncio
import json
import pickle
import os
import shutil
from pathlib import Path
import mlflow
import mlflow.pytorch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import joblib
import redis.asyncio as aioredis
from contextlib import asynccontextmanager
import hashlib
import boto3
from dataclasses import asdict
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for AI models"""
    model_id: str
    model_name: str
    version: str
    model_type: str
    architecture: str
    training_dataset: str
    training_start_time: datetime
    training_end_time: datetime
    performance_metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    feature_importance: Dict[str, float]
    data_schema: Dict[str, str]
    model_size_mb: float
    inference_latency_ms: float
    deployment_status: str = "trained"  # trained, deployed, deprecated
    deployment_timestamp: Optional[datetime] = None
    last_evaluation_time: Optional[datetime] = None
    drift_score: Optional[float] = None
    retraining_trigger: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    artifacts: Dict[str, str] = field(default_factory=dict)


@dataclass
class ModelPerformance:
    """Model performance tracking"""
    model_id: str
    timestamp: datetime
    metrics: Dict[str, float]
    data_quality_score: float
    prediction_distribution: Dict[str, float]
    feature_stability: Dict[str, float]
    business_impact: Dict[str, float]
    drift_metrics: Dict[str, float]
    error_analysis: Dict[str, Any]
    recommendation: str  # continue, retrain, investigate, deprecate


@dataclass
class RetrainingJob:
    """Retraining job configuration"""
    job_id: str
    model_id: str
    trigger_reason: str
    scheduled_time: datetime
    dataset_config: Dict[str, Any]
    training_config: Dict[str, Any]
    validation_config: Dict[str, Any]
    resource_requirements: Dict[str, Any]
    priority: str = "normal"  # low, normal, high, critical
    status: str = "pending"   # pending, running, completed, failed
    estimated_duration: Optional[timedelta] = None
    actual_duration: Optional[timedelta] = None
    results: Optional[Dict[str, Any]] = None


class ModelRegistry:
    """Centralized model registry for tracking and versioning"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.storage_path = Path(config.get('storage_path', '/app/models'))
        self.redis_url = config.get('redis_url', 'redis://localhost:6379')
        self.use_mlflow = config.get('use_mlflow', True)
        self.s3_bucket = config.get('s3_bucket')
        
        # Initialize storage
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize MLflow
        if self.use_mlflow:
            mlflow.set_tracking_uri(config.get('mlflow_uri', 'file:./mlruns'))
        
        # Initialize S3 client
        if self.s3_bucket:
            self.s3_client = boto3.client('s3')
        
    async def register_model(self, model: nn.Module, metadata: ModelMetadata,
                           artifacts: Optional[Dict[str, Any]] = None) -> str:
        """Register a new model in the registry"""
        try:
            # Create model directory
            model_path = self.storage_path / metadata.model_id
            model_path.mkdir(parents=True, exist_ok=True)
            
            # Save model
            model_file = model_path / 'model.pth'
            torch.save(model.state_dict(), model_file)
            
            # Save metadata
            metadata_file = model_path / 'metadata.json'
            with open(metadata_file, 'w') as f:
                json.dump(asdict(metadata), f, indent=2, default=str)
            
            # Save artifacts
            if artifacts:
                artifacts_dir = model_path / 'artifacts'
                artifacts_dir.mkdir(exist_ok=True)
                
                for name, data in artifacts.items():
                    artifact_file = artifacts_dir / f'{name}.pkl'
                    with open(artifact_file, 'wb') as f:
                        pickle.dump(data, f)
            
            # Register in MLflow if enabled
            if self.use_mlflow:
                await self._register_in_mlflow(model, metadata, artifacts)
            
            # Cache in Redis
            await self._cache_metadata(metadata)
            
            # Upload to S3 if configured
            if self.s3_bucket:
                await self._upload_to_s3(metadata.model_id, model_path)
            
            logger.info(f"Model registered: {metadata.model_id}")
            return metadata.model_id
            
        except Exception as e:
            logger.error(f"Error registering model: {e}")
            raise
    
    async def get_model(self, model_id: str, model_class: type) -> Tuple[nn.Module, ModelMetadata]:
        """Load a model from the registry"""
        try:
            model_path = self.storage_path / model_id
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model {model_id} not found")
            
            # Load metadata
            metadata_file = model_path / 'metadata.json'
            with open(metadata_file, 'r') as f:
                metadata_dict = json.load(f)
            
            metadata = ModelMetadata(**metadata_dict)
            
            # Initialize model
            model = model_class()  # Assumes no-args constructor
            
            # Load model weights
            model_file = model_path / 'model.pth'
            model.load_state_dict(torch.load(model_file, map_location='cpu'))
            
            logger.info(f"Model loaded: {model_id}")
            return model, metadata
            
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {e}")
            raise
    
    async def list_models(self, status: Optional[str] = None,
                         model_type: Optional[str] = None) -> List[ModelMetadata]:
        """List models in the registry with optional filtering"""
        models = []
        
        try:
            for model_dir in self.storage_path.iterdir():
                if model_dir.is_dir():
                    metadata_file = model_dir / 'metadata.json'
                    
                    if metadata_file.exists():
                        with open(metadata_file, 'r') as f:
                            metadata_dict = json.load(f)
                        
                        metadata = ModelMetadata(**metadata_dict)
                        
                        # Apply filters
                        if status and metadata.deployment_status != status:
                            continue
                        if model_type and metadata.model_type != model_type:
                            continue
                        
                        models.append(metadata)
            
            return sorted(models, key=lambda x: x.training_start_time, reverse=True)
            
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    async def update_model_status(self, model_id: str, status: str,
                                deployment_timestamp: Optional[datetime] = None) -> None:
        """Update model deployment status"""
        try:
            model_path = self.storage_path / model_id
            metadata_file = model_path / 'metadata.json'
            
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata_dict = json.load(f)
                
                metadata_dict['deployment_status'] = status
                if deployment_timestamp:
                    metadata_dict['deployment_timestamp'] = deployment_timestamp.isoformat()
                
                with open(metadata_file, 'w') as f:
                    json.dump(metadata_dict, f, indent=2, default=str)
                
                logger.info(f"Updated model {model_id} status to {status}")
                
        except Exception as e:
            logger.error(f"Error updating model status: {e}")
            raise
    
    async def deprecate_model(self, model_id: str, reason: str) -> None:
        """Deprecate a model"""
        await self.update_model_status(model_id, 'deprecated')
        
        # Archive model files
        model_path = self.storage_path / model_id
        archive_path = self.storage_path / 'archived' / model_id
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        
        if model_path.exists():
            shutil.move(str(model_path), str(archive_path))
        
        logger.info(f"Model {model_id} deprecated: {reason}")
    
    async def _register_in_mlflow(self, model: nn.Module, metadata: ModelMetadata,
                                artifacts: Optional[Dict[str, Any]]) -> None:
        """Register model in MLflow"""
        try:
            with mlflow.start_run(run_name=f"{metadata.model_name}_{metadata.version}"):
                # Log parameters
                mlflow.log_params(metadata.hyperparameters)
                
                # Log metrics
                mlflow.log_metrics(metadata.performance_metrics)
                
                # Log model
                mlflow.pytorch.log_model(
                    model, 
                    artifact_path="model",
                    registered_model_name=metadata.model_name
                )
                
                # Log artifacts
                if artifacts:
                    for name, data in artifacts.items():
                        artifact_path = f"artifacts/{name}.pkl"
                        with open(artifact_path, 'wb') as f:
                            pickle.dump(data, f)
                        mlflow.log_artifact(artifact_path)
                        os.remove(artifact_path)
                
        except Exception as e:
            logger.warning(f"Failed to register in MLflow: {e}")
    
    async def _cache_metadata(self, metadata: ModelMetadata) -> None:
        """Cache metadata in Redis"""
        try:
            redis = await aioredis.from_url(self.redis_url)
            
            # Cache metadata
            await redis.hset(
                f"model:{metadata.model_id}",
                mapping={
                    "name": metadata.model_name,
                    "version": metadata.version,
                    "type": metadata.model_type,
                    "status": metadata.deployment_status,
                    "performance": json.dumps(metadata.performance_metrics)
                }
            )
            
            # Set expiration
            await redis.expire(f"model:{metadata.model_id}", 86400)  # 24 hours
            
            await redis.close()
            
        except Exception as e:
            logger.warning(f"Failed to cache metadata in Redis: {e}")
    
    async def _upload_to_s3(self, model_id: str, model_path: Path) -> None:
        """Upload model to S3"""
        try:
            if not self.s3_bucket:
                return
            
            # Upload all files in the model directory
            for file_path in model_path.rglob('*'):
                if file_path.is_file():
                    relative_path = file_path.relative_to(model_path)
                    s3_key = f"models/{model_id}/{relative_path}"
                    
                    self.s3_client.upload_file(
                        str(file_path),
                        self.s3_bucket,
                        s3_key
                    )
            
            logger.info(f"Model {model_id} uploaded to S3")
            
        except Exception as e:
            logger.warning(f"Failed to upload to S3: {e}")


class ModelMonitor:
    """Real-time model monitoring and drift detection"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.redis_url = config.get('redis_url', 'redis://localhost:6379')
        self.monitoring_interval = config.get('monitoring_interval', 300)  # 5 minutes
        self.drift_threshold = config.get('drift_threshold', 0.1)
        self.performance_threshold = config.get('performance_threshold', 0.05)
        
        # Performance tracking
        self.performance_history: Dict[str, List[ModelPerformance]] = {}
        
    async def monitor_model(self, model_id: str, predictions: np.ndarray,
                          actuals: Optional[np.ndarray] = None,
                          features: Optional[np.ndarray] = None) -> ModelPerformance:
        """Monitor model performance and detect drift"""
        try:
            timestamp = datetime.now()
            
            # Calculate performance metrics
            metrics = {}
            if actuals is not None:
                metrics.update({
                    'mse': mean_squared_error(actuals, predictions),
                    'mae': mean_absolute_error(actuals, predictions),
                    'r2': r2_score(actuals, predictions),
                    'accuracy': self._calculate_accuracy(actuals, predictions)
                })
            
            # Calculate prediction distribution
            pred_distribution = {
                'mean': float(np.mean(predictions)),
                'std': float(np.std(predictions)),
                'min': float(np.min(predictions)),
                'max': float(np.max(predictions)),
                'q25': float(np.percentile(predictions, 25)),
                'q50': float(np.percentile(predictions, 50)),
                'q75': float(np.percentile(predictions, 75))
            }
            
            # Feature stability analysis
            feature_stability = {}
            if features is not None:
                feature_stability = await self._analyze_feature_stability(
                    model_id, features, timestamp
                )
            
            # Drift detection
            drift_metrics = await self._detect_drift(model_id, predictions, features)
            
            # Data quality assessment
            data_quality_score = self._assess_data_quality(predictions, features)
            
            # Business impact assessment
            business_impact = await self._assess_business_impact(
                model_id, predictions, actuals
            )
            
            # Error analysis
            error_analysis = self._analyze_errors(predictions, actuals) if actuals is not None else {}
            
            # Generate recommendation
            recommendation = self._generate_recommendation(
                metrics, drift_metrics, data_quality_score
            )
            
            # Create performance record
            performance = ModelPerformance(
                model_id=model_id,
                timestamp=timestamp,
                metrics=metrics,
                data_quality_score=data_quality_score,
                prediction_distribution=pred_distribution,
                feature_stability=feature_stability,
                business_impact=business_impact,
                drift_metrics=drift_metrics,
                error_analysis=error_analysis,
                recommendation=recommendation
            )
            
            # Store performance history
            if model_id not in self.performance_history:
                self.performance_history[model_id] = []
            
            self.performance_history[model_id].append(performance)
            
            # Keep only recent history (last 1000 records)
            self.performance_history[model_id] = self.performance_history[model_id][-1000:]
            
            # Cache in Redis
            await self._cache_performance(performance)
            
            # Trigger alerts if necessary
            if recommendation in ['retrain', 'investigate']:
                await self._trigger_alert(model_id, performance)
            
            return performance
            
        except Exception as e:
            logger.error(f"Error monitoring model {model_id}: {e}")
            raise
    
    async def get_model_health(self, model_id: str, 
                             time_window: timedelta = timedelta(hours=24)) -> Dict[str, Any]:
        """Get comprehensive model health report"""
        try:
            cutoff_time = datetime.now() - time_window
            
            # Get recent performance records
            recent_performances = [
                p for p in self.performance_history.get(model_id, [])
                if p.timestamp >= cutoff_time
            ]
            
            if not recent_performances:
                return {
                    'status': 'no_data',
                    'message': 'No recent performance data available'
                }
            
            # Aggregate metrics
            all_metrics = {}
            for perf in recent_performances:
                for metric, value in perf.metrics.items():
                    if metric not in all_metrics:
                        all_metrics[metric] = []
                    all_metrics[metric].append(value)
            
            # Calculate trends
            metric_trends = {}
            for metric, values in all_metrics.items():
                if len(values) >= 2:
                    recent_avg = np.mean(values[-10:])  # Last 10 values
                    older_avg = np.mean(values[:-10]) if len(values) > 10 else np.mean(values)
                    trend = (recent_avg - older_avg) / older_avg if older_avg != 0 else 0
                    metric_trends[metric] = trend
            
            # Drift analysis
            drift_scores = [p.drift_metrics.get('overall_drift', 0) for p in recent_performances]
            avg_drift = np.mean(drift_scores) if drift_scores else 0
            
            # Data quality analysis
            quality_scores = [p.data_quality_score for p in recent_performances]
            avg_quality = np.mean(quality_scores) if quality_scores else 0
            
            # Overall health score
            health_score = self._calculate_health_score(
                metric_trends, avg_drift, avg_quality
            )
            
            return {
                'status': 'healthy' if health_score > 0.7 else 'degraded' if health_score > 0.4 else 'critical',
                'health_score': health_score,
                'metric_trends': metric_trends,
                'avg_drift': avg_drift,
                'avg_data_quality': avg_quality,
                'recommendations': self._get_health_recommendations(health_score, avg_drift, avg_quality),
                'last_update': recent_performances[-1].timestamp,
                'evaluation_count': len(recent_performances)
            }
            
        except Exception as e:
            logger.error(f"Error getting model health for {model_id}: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    async def _analyze_feature_stability(self, model_id: str, features: np.ndarray,
                                       timestamp: datetime) -> Dict[str, float]:
        """Analyze feature stability over time"""
        try:
            # Get historical feature data from cache
            redis = await aioredis.from_url(self.redis_url)
            
            historical_key = f"features:{model_id}"
            historical_data = await redis.lrange(historical_key, 0, 100)
            
            stability_scores = {}
            
            if historical_data:
                # Parse historical features
                historical_features = []
                for data_str in historical_data:
                    data = json.loads(data_str)
                    historical_features.append(np.array(data['features']))
                
                # Compare current features with historical
                for i in range(features.shape[1]):  # For each feature
                    current_feature = features[:, i]
                    historical_values = [hf[:, i] for hf in historical_features if hf.shape[1] > i]
                    
                    if historical_values:
                        historical_combined = np.concatenate(historical_values)
                        
                        # Calculate KL divergence or similar metric
                        stability_score = self._calculate_feature_drift(
                            current_feature, historical_combined
                        )
                        stability_scores[f'feature_{i}'] = stability_score
            
            # Cache current features
            feature_data = {
                'timestamp': timestamp.isoformat(),
                'features': features.tolist()
            }
            
            await redis.lpush(historical_key, json.dumps(feature_data))
            await redis.ltrim(historical_key, 0, 99)  # Keep only last 100
            await redis.expire(historical_key, 604800)  # 1 week
            
            await redis.close()
            
            return stability_scores
            
        except Exception as e:
            logger.error(f"Error analyzing feature stability: {e}")
            return {}
    
    async def _detect_drift(self, model_id: str, predictions: np.ndarray,
                          features: Optional[np.ndarray]) -> Dict[str, float]:
        """Detect various types of drift"""
        drift_metrics = {}
        
        try:
            # Get historical data
            redis = await aioredis.from_url(self.redis_url)
            
            # Prediction drift
            pred_key = f"predictions:{model_id}"
            historical_preds = await redis.lrange(pred_key, 0, 100)
            
            if historical_preds:
                historical_pred_data = [json.loads(p) for p in historical_preds]
                historical_predictions = np.concatenate([
                    np.array(data['predictions']) for data in historical_pred_data
                ])
                
                # Calculate prediction drift
                pred_drift = self._calculate_prediction_drift(
                    predictions, historical_predictions
                )
                drift_metrics['prediction_drift'] = pred_drift
            
            # Feature drift (if features provided)
            if features is not None:
                feature_drift_scores = await self._analyze_feature_stability(
                    model_id, features, datetime.now()
                )
                avg_feature_drift = np.mean(list(feature_drift_scores.values())) if feature_drift_scores else 0
                drift_metrics['feature_drift'] = avg_feature_drift
            
            # Overall drift score
            drift_values = [v for v in drift_metrics.values() if v is not None]
            drift_metrics['overall_drift'] = np.mean(drift_values) if drift_values else 0
            
            # Cache current predictions
            pred_data = {
                'timestamp': datetime.now().isoformat(),
                'predictions': predictions.tolist()
            }
            
            await redis.lpush(pred_key, json.dumps(pred_data))
            await redis.ltrim(pred_key, 0, 99)
            await redis.expire(pred_key, 604800)  # 1 week
            
            await redis.close()
            
        except Exception as e:
            logger.error(f"Error detecting drift: {e}")
        
        return drift_metrics
    
    def _calculate_accuracy(self, actuals: np.ndarray, predictions: np.ndarray) -> float:
        """Calculate classification accuracy or regression accuracy"""
        try:
            # For regression tasks, use relative accuracy
            if len(np.unique(actuals)) > 10:  # Continuous values
                relative_errors = np.abs((predictions - actuals) / (actuals + 1e-8))
                accuracy = 1 - np.mean(relative_errors)
                return max(0, accuracy)  # Ensure non-negative
            else:
                # For classification, use standard accuracy
                return np.mean(np.round(predictions) == actuals)
        except:
            return 0.0
    
    def _assess_data_quality(self, predictions: np.ndarray, 
                           features: Optional[np.ndarray]) -> float:
        """Assess data quality score"""
        quality_score = 1.0
        
        # Check for anomalies in predictions
        if len(predictions) > 0:
            # Check for NaN/inf values
            if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
                quality_score -= 0.3
            
            # Check for extreme outliers
            q1, q3 = np.percentile(predictions, [25, 75])
            iqr = q3 - q1
            outlier_threshold = 3 * iqr
            outlier_ratio = np.sum(np.abs(predictions - np.median(predictions)) > outlier_threshold) / len(predictions)
            quality_score -= outlier_ratio * 0.2
        
        # Check feature quality
        if features is not None:
            # Missing values
            missing_ratio = np.sum(np.isnan(features)) / features.size
            quality_score -= missing_ratio * 0.3
            
            # Constant features
            constant_features = np.sum(np.std(features, axis=0) < 1e-8)
            if features.shape[1] > 0:
                constant_ratio = constant_features / features.shape[1]
                quality_score -= constant_ratio * 0.2
        
        return max(0.0, min(1.0, quality_score))
    
    async def _assess_business_impact(self, model_id: str, predictions: np.ndarray,
                                   actuals: Optional[np.ndarray]) -> Dict[str, float]:
        """Assess business impact of model predictions"""
        business_impact = {
            'prediction_volume': len(predictions),
            'prediction_variance': float(np.var(predictions)),
        }
        
        if actuals is not None:
            # Calculate potential business value
            errors = np.abs(predictions - actuals)
            business_impact.update({
                'avg_error': float(np.mean(errors)),
                'error_variance': float(np.var(errors)),
                'large_error_rate': float(np.mean(errors > np.std(errors) * 2))
            })
        
        return business_impact
    
    def _analyze_errors(self, predictions: np.ndarray, actuals: np.ndarray) -> Dict[str, Any]:
        """Perform detailed error analysis"""
        if actuals is None:
            return {}
        
        errors = predictions - actuals
        
        return {
            'mean_error': float(np.mean(errors)),
            'error_std': float(np.std(errors)),
            'error_skewness': float(self._calculate_skewness(errors)),
            'error_kurtosis': float(self._calculate_kurtosis(errors)),
            'positive_error_rate': float(np.mean(errors > 0)),
            'large_error_rate': float(np.mean(np.abs(errors) > 2 * np.std(errors)))
        }
    
    def _generate_recommendation(self, metrics: Dict[str, float],
                               drift_metrics: Dict[str, float],
                               data_quality_score: float) -> str:
        """Generate recommendation based on monitoring results"""
        # Check data quality
        if data_quality_score < 0.7:
            return 'investigate'
        
        # Check drift
        overall_drift = drift_metrics.get('overall_drift', 0)
        if overall_drift > self.drift_threshold:
            return 'retrain'
        
        # Check performance degradation
        if metrics:
            # Assume we have historical baseline (simplified)
            for metric, value in metrics.items():
                if metric in ['mse', 'mae'] and value > 0.1:  # High error
                    return 'investigate'
                elif metric in ['r2', 'accuracy'] and value < 0.8:  # Low accuracy
                    return 'retrain'
        
        return 'continue'
    
    def _calculate_feature_drift(self, current: np.ndarray, historical: np.ndarray) -> float:
        """Calculate drift score between current and historical features"""
        try:
            from scipy.stats import ks_2samp
            _, p_value = ks_2samp(current, historical)
            drift_score = 1 - p_value  # Higher score means more drift
            return drift_score
        except:
            # Fallback to simple statistical comparison
            current_mean = np.mean(current)
            current_std = np.std(current)
            historical_mean = np.mean(historical)
            historical_std = np.std(historical)
            
            mean_drift = abs(current_mean - historical_mean) / (historical_std + 1e-8)
            std_drift = abs(current_std - historical_std) / (historical_std + 1e-8)
            
            return (mean_drift + std_drift) / 2
    
    def _calculate_prediction_drift(self, current: np.ndarray, historical: np.ndarray) -> float:
        """Calculate drift in prediction distribution"""
        return self._calculate_feature_drift(current, historical)
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data"""
        try:
            from scipy.stats import skew
            return skew(data)
        except:
            # Manual calculation
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0
            return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data"""
        try:
            from scipy.stats import kurtosis
            return kurtosis(data)
        except:
            # Manual calculation
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0
            return np.mean(((data - mean) / std) ** 4) - 3
    
    def _calculate_health_score(self, metric_trends: Dict[str, float],
                              avg_drift: float, avg_quality: float) -> float:
        """Calculate overall model health score"""
        # Base score from data quality
        health_score = avg_quality * 0.4
        
        # Penalty for drift
        health_score += (1 - avg_drift) * 0.3
        
        # Bonus/penalty for metric trends
        if metric_trends:
            positive_trends = sum(1 for trend in metric_trends.values() if trend > 0)
            total_trends = len(metric_trends)
            trend_score = positive_trends / total_trends
            health_score += trend_score * 0.3
        
        return min(1.0, max(0.0, health_score))
    
    def _get_health_recommendations(self, health_score: float, avg_drift: float,
                                  avg_quality: float) -> List[str]:
        """Get health-based recommendations"""
        recommendations = []
        
        if health_score < 0.4:
            recommendations.append("Immediate attention required")
        elif health_score < 0.7:
            recommendations.append("Consider model retraining")
        
        if avg_drift > 0.2:
            recommendations.append("High drift detected - check data sources")
        
        if avg_quality < 0.8:
            recommendations.append("Data quality issues - review input pipeline")
        
        if not recommendations:
            recommendations.append("Model operating normally")
        
        return recommendations
    
    async def _cache_performance(self, performance: ModelPerformance) -> None:
        """Cache performance data in Redis"""
        try:
            redis = await aioredis.from_url(self.redis_url)
            
            perf_key = f"performance:{performance.model_id}"
            perf_data = {
                'timestamp': performance.timestamp.isoformat(),
                'metrics': json.dumps(performance.metrics),
                'drift_score': performance.drift_metrics.get('overall_drift', 0),
                'quality_score': performance.data_quality_score,
                'recommendation': performance.recommendation
            }
            
            await redis.hset(perf_key, mapping=perf_data)
            await redis.expire(perf_key, 86400)  # 24 hours
            
            await redis.close()
            
        except Exception as e:
            logger.warning(f"Failed to cache performance data: {e}")
    
    async def _trigger_alert(self, model_id: str, performance: ModelPerformance) -> None:
        """Trigger alerts for model issues"""
        try:
            # This would integrate with your alerting system
            # For now, just log the alert
            logger.warning(f"ALERT: Model {model_id} requires attention. "
                         f"Recommendation: {performance.recommendation}")
            
            # You could implement:
            # - Email notifications
            # - Slack/Teams messages
            # - PagerDuty alerts
            # - Dashboard notifications
            
        except Exception as e:
            logger.error(f"Error triggering alert: {e}")


class ContinuousLearningPipeline:
    """Continuous learning pipeline for automated model updates"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_registry = ModelRegistry(config.get('registry_config', {}))
        self.model_monitor = ModelMonitor(config.get('monitor_config', {}))
        
        # Pipeline configuration
        self.retraining_schedule = config.get('retraining_schedule', 'weekly')
        self.drift_threshold = config.get('drift_threshold', 0.1)
        self.performance_threshold = config.get('performance_threshold', 0.05)
        self.min_data_points = config.get('min_data_points', 1000)
        
        # Active jobs
        self.active_jobs: Dict[str, RetrainingJob] = {}
        
    async def start_continuous_learning(self, model_ids: List[str]) -> None:
        """Start continuous learning for specified models"""
        logger.info(f"Starting continuous learning for {len(model_ids)} models")
        
        # Create monitoring tasks for each model
        tasks = []
        for model_id in model_ids:
            task = asyncio.create_task(self._monitor_model_continuously(model_id))
            tasks.append(task)
        
        # Wait for all monitoring tasks
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Error in continuous learning pipeline: {e}")
    
    async def _monitor_model_continuously(self, model_id: str) -> None:
        """Continuously monitor a single model"""
        while True:
            try:
                # Check if retraining is needed
                should_retrain = await self._evaluate_retraining_need(model_id)
                
                if should_retrain:
                    job_id = await self.schedule_retraining(
                        model_id, 
                        trigger_reason="continuous_learning_check"
                    )
                    
                    if job_id:
                        await self._execute_retraining_job(job_id)
                
                # Wait before next check
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Error monitoring model {model_id}: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def _evaluate_retraining_need(self, model_id: str) -> bool:
        """Evaluate if model needs retraining"""
        try:
            # Get model health
            health = await self.model_monitor.get_model_health(model_id)
            
            # Check health score
            if health.get('health_score', 1.0) < 0.6:
                logger.info(f"Model {model_id} needs retraining due to low health score")
                return True
            
            # Check drift
            if health.get('avg_drift', 0) > self.drift_threshold:
                logger.info(f"Model {model_id} needs retraining due to drift")
                return True
            
            # Check if scheduled retraining is due
            if await self._is_scheduled_retraining_due(model_id):
                logger.info(f"Model {model_id} scheduled retraining is due")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error evaluating retraining need for {model_id}: {e}")
            return False
    
    async def schedule_retraining(self, model_id: str, trigger_reason: str,
                                priority: str = "normal",
                                custom_config: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Schedule a retraining job"""
        try:
            job_id = f"retrain_{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create retraining job
            job = RetrainingJob(
                job_id=job_id,
                model_id=model_id,
                trigger_reason=trigger_reason,
                scheduled_time=datetime.now(),
                dataset_config=custom_config.get('dataset_config', {}) if custom_config else {},
                training_config=custom_config.get('training_config', {}) if custom_config else {},
                validation_config=custom_config.get('validation_config', {}) if custom_config else {},
                resource_requirements=custom_config.get('resource_requirements', {}) if custom_config else {},
                priority=priority
            )
            
            self.active_jobs[job_id] = job
            logger.info(f"Scheduled retraining job {job_id} for model {model_id}")
            
            return job_id
            
        except Exception as e:
            logger.error(f"Error scheduling retraining: {e}")
            return None
    
    async def _execute_retraining_job(self, job_id: str) -> None:
        """Execute a retraining job"""
        try:
            job = self.active_jobs.get(job_id)
            if not job:
                logger.error(f"Job {job_id} not found")
                return
            
            job.status = "running"
            start_time = datetime.now()
            
            logger.info(f"Executing retraining job {job_id}")
            
            # Load current model
            models = await self.model_registry.list_models()
            current_model = next((m for m in models if m.model_id == job.model_id), None)
            
            if not current_model:
                logger.error(f"Model {job.model_id} not found in registry")
                job.status = "failed"
                job.results = {"error": "Model not found"}
                return
            
            # Simulate retraining process
            # In a real implementation, this would:
            # 1. Fetch new training data
            # 2. Prepare features
            # 3. Train the model
            # 4. Validate the model
            # 5. Compare with existing model
            # 6. Deploy if better
            
            await asyncio.sleep(10)  # Simulate training time
            
            # Generate new model metadata
            new_model_id = f"{job.model_id}_retrained_{start_time.strftime('%Y%m%d_%H%M%S')}"
            
            new_metadata = ModelMetadata(
                model_id=new_model_id,
                model_name=current_model.model_name,
                version=f"v{int(current_model.version.replace('v', '')) + 1}",
                model_type=current_model.model_type,
                architecture=current_model.architecture,
                training_dataset="retrained_dataset",
                training_start_time=start_time,
                training_end_time=datetime.now(),
                performance_metrics={
                    'accuracy': 0.92,  # Simulated improvement
                    'mse': 0.05,
                    'r2': 0.89
                },
                hyperparameters=current_model.hyperparameters,
                feature_importance={},
                data_schema=current_model.data_schema,
                model_size_mb=current_model.model_size_mb,
                inference_latency_ms=current_model.inference_latency_ms
            )
            
            # Mark job as completed
            job.status = "completed"
            job.actual_duration = datetime.now() - start_time
            job.results = {
                "new_model_id": new_model_id,
                "performance_improvement": 0.05,  # Simulated
                "validation_metrics": new_metadata.performance_metrics
            }
            
            logger.info(f"Retraining job {job_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Error executing retraining job {job_id}: {e}")
            if job_id in self.active_jobs:
                self.active_jobs[job_id].status = "failed"
                self.active_jobs[job_id].results = {"error": str(e)}
    
    async def _is_scheduled_retraining_due(self, model_id: str) -> bool:
        """Check if scheduled retraining is due"""
        # Get last retraining time from model metadata
        models = await self.model_registry.list_models()
        model = next((m for m in models if m.model_id == model_id), None)
        
        if not model:
            return False
        
        last_training = model.training_end_time
        now = datetime.now()
        
        if self.retraining_schedule == 'daily':
            return (now - last_training).days >= 1
        elif self.retraining_schedule == 'weekly':
            return (now - last_training).days >= 7
        elif self.retraining_schedule == 'monthly':
            return (now - last_training).days >= 30
        
        return False
    
    async def get_job_status(self, job_id: str) -> Optional[RetrainingJob]:
        """Get status of a retraining job"""
        return self.active_jobs.get(job_id)
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a retraining job"""
        try:
            job = self.active_jobs.get(job_id)
            if job and job.status in ["pending", "running"]:
                job.status = "cancelled"
                logger.info(f"Job {job_id} cancelled")
                return True
            return False
        except Exception as e:
            logger.error(f"Error cancelling job {job_id}: {e}")
            return False


class AIModelManager:
    """Main AI model lifecycle manager"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_registry = ModelRegistry(config.get('registry_config', {}))
        self.model_monitor = ModelMonitor(config.get('monitor_config', {}))
        self.continuous_learning = ContinuousLearningPipeline(config)
        
    async def deploy_model(self, model: nn.Module, metadata: ModelMetadata,
                         artifacts: Optional[Dict[str, Any]] = None) -> str:
        """Deploy a new model"""
        # Register model
        model_id = await self.model_registry.register_model(model, metadata, artifacts)
        
        # Update status to deployed
        await self.model_registry.update_model_status(
            model_id, 'deployed', datetime.now()
        )
        
        logger.info(f"Model deployed: {model_id}")
        return model_id
    
    async def monitor_prediction(self, model_id: str, predictions: np.ndarray,
                               actuals: Optional[np.ndarray] = None,
                               features: Optional[np.ndarray] = None) -> ModelPerformance:
        """Monitor a model prediction"""
        return await self.model_monitor.monitor_model(
            model_id, predictions, actuals, features
        )
    
    async def start_continuous_learning(self, model_ids: List[str]) -> None:
        """Start continuous learning for models"""
        await self.continuous_learning.start_continuous_learning(model_ids)
    
    async def get_model_dashboard(self, model_id: str) -> Dict[str, Any]:
        """Get comprehensive model dashboard data"""
        try:
            # Get model metadata
            models = await self.model_registry.list_models()
            model = next((m for m in models if m.model_id == model_id), None)
            
            if not model:
                return {'error': 'Model not found'}
            
            # Get health information
            health = await self.model_monitor.get_model_health(model_id)
            
            # Get recent performance
            performance_history = self.model_monitor.performance_history.get(model_id, [])
            recent_performance = performance_history[-10:] if performance_history else []
            
            # Get active jobs
            active_jobs = [
                job for job in self.continuous_learning.active_jobs.values()
                if job.model_id == model_id
            ]
            
            return {
                'model_metadata': asdict(model),
                'health': health,
                'recent_performance': [asdict(p) for p in recent_performance],
                'active_jobs': [asdict(job) for job in active_jobs],
                'recommendations': health.get('recommendations', [])
            }
            
        except Exception as e:
            logger.error(f"Error getting model dashboard: {e}")
            return {'error': str(e)}


# Example usage and testing
async def demo_model_lifecycle():
    """Demonstrate model lifecycle management"""
    logger.info("Starting model lifecycle demo")
    
    # Configuration
    config = {
        'registry_config': {
            'storage_path': '/tmp/model_registry',
            'redis_url': 'redis://localhost:6379',
            'use_mlflow': False  # Disable for demo
        },
        'monitor_config': {
            'drift_threshold': 0.1,
            'monitoring_interval': 60
        },
        'retraining_schedule': 'weekly'
    }
    
    # Initialize model manager
    model_manager = AIModelManager(config)
    
    # Create a simple model for demonstration
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)
            
        def forward(self, x):
            return self.linear(x)
    
    # Create model and metadata
    demo_model = SimpleModel()
    
    metadata = ModelMetadata(
        model_id="demo_model_001",
        model_name="Demo Model",
        version="v1",
        model_type="regression",
        architecture="linear",
        training_dataset="demo_dataset",
        training_start_time=datetime.now() - timedelta(hours=1),
        training_end_time=datetime.now(),
        performance_metrics={
            'mse': 0.05,
            'mae': 0.15,
            'r2': 0.85
        },
        hyperparameters={'lr': 0.001, 'epochs': 100},
        feature_importance={'feature_0': 0.5, 'feature_1': 0.3, 'feature_2': 0.2},
        data_schema={'input': 'float32', 'output': 'float32'},
        model_size_mb=0.1,
        inference_latency_ms=5.0
    )
    
    # Deploy model
    model_id = await model_manager.deploy_model(demo_model, metadata)
    logger.info(f"Deployed model: {model_id}")
    
    # Simulate some predictions and monitoring
    for i in range(5):
        # Generate sample data
        features = np.random.randn(100, 10)
        predictions = np.random.randn(100) + 0.1 * i  # Add drift over time
        actuals = predictions + np.random.randn(100) * 0.1
        
        # Monitor predictions
        performance = await model_manager.monitor_prediction(
            model_id, predictions, actuals, features
        )
        
        logger.info(f"Monitoring iteration {i+1}:")
        logger.info(f"  MSE: {performance.metrics.get('mse', 0):.4f}")
        logger.info(f"  Drift: {performance.drift_metrics.get('overall_drift', 0):.4f}")
        logger.info(f"  Quality: {performance.data_quality_score:.4f}")
        logger.info(f"  Recommendation: {performance.recommendation}")
        
        # Wait between iterations
        await asyncio.sleep(1)
    
    # Get model health
    health = await model_manager.model_monitor.get_model_health(model_id)
    logger.info(f"\nModel Health Report:")
    logger.info(f"  Status: {health.get('status')}")
    logger.info(f"  Health Score: {health.get('health_score', 0):.3f}")
    logger.info(f"  Recommendations: {health.get('recommendations', [])}")
    
    # Get dashboard data
    dashboard = await model_manager.get_model_dashboard(model_id)
    logger.info(f"\nDashboard Summary:")
    logger.info(f"  Model Status: {dashboard['model_metadata']['deployment_status']}")
    logger.info(f"  Performance Records: {len(dashboard['recent_performance'])}")
    logger.info(f"  Active Jobs: {len(dashboard['active_jobs'])}")
    
    logger.info("Model lifecycle demo completed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(demo_model_lifecycle())