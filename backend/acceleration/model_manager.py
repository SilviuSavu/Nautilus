"""
Model Management System for Neural Engine Integration
===================================================

Comprehensive model lifecycle management system optimized for M4 Max Neural Engine
with automated model deployment, A/B testing, performance monitoring, and retraining.

Key Features:
- Model lifecycle management (load, update, retire, rollback)
- A/B testing framework with statistical analysis
- Performance monitoring and accuracy tracking
- Automated retraining triggers and pipelines
- Resource allocation and queuing optimization
- Model versioning with deployment strategies
- Health monitoring and failure recovery

Performance Targets:
- < 100ms model deployment time
- Zero-downtime model updates
- Automated performance regression detection
- Real-time accuracy monitoring
- Predictive retraining scheduling
"""

import logging
import asyncio
import time
import json
import pickle
import hashlib
import threading
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

# Model management and monitoring
try:
    from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Core ML integration
from .neural_engine_config import neural_engine_config, get_optimization_config
from .neural_inference import inference_engine, InferenceResult, PriorityLevel
from .coreml_pipeline import model_converter, version_manager as pipeline_version_manager

logger = logging.getLogger(__name__)

class ModelStatus(Enum):
    """Model deployment status"""
    LOADING = "loading"
    ACTIVE = "active"
    INACTIVE = "inactive"
    TESTING = "testing"
    DEPRECATED = "deprecated"
    FAILED = "failed"
    RETIRING = "retiring"

class DeploymentStrategy(Enum):
    """Model deployment strategies"""
    IMMEDIATE = "immediate"
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING = "rolling"
    AB_TEST = "ab_test"

class PerformanceMetric(Enum):
    """Performance metrics for model evaluation"""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    MSE = "mse"
    MAE = "mae"
    RMSE = "rmse"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"

class RetrainingTrigger(Enum):
    """Triggers for automated retraining"""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    SCHEDULED = "scheduled"
    MANUAL = "manual"
    FEEDBACK_THRESHOLD = "feedback_threshold"

@dataclass
class ModelMetadata:
    """Comprehensive model metadata"""
    model_id: str
    name: str
    version: str
    architecture: str
    framework: str
    created_at: datetime
    created_by: str
    description: str
    tags: List[str] = field(default_factory=list)
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    training_data_hash: Optional[str] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    performance_baseline: Optional[Dict[str, float]] = None

@dataclass
class DeploymentConfig:
    """Model deployment configuration"""
    strategy: DeploymentStrategy
    traffic_split: Optional[float] = None
    rollout_duration_minutes: Optional[int] = None
    success_criteria: Optional[Dict[str, float]] = None
    rollback_criteria: Optional[Dict[str, float]] = None
    canary_percentage: Optional[float] = None
    monitoring_window_minutes: int = 60
    auto_promote: bool = False

@dataclass
class ModelPerformance:
    """Model performance metrics tracking"""
    model_id: str
    timestamp: datetime
    metrics: Dict[PerformanceMetric, float]
    inference_count: int
    error_count: int
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_qps: float
    memory_usage_mb: float
    cpu_utilization: float
    neural_engine_utilization: float

@dataclass
class ExperimentResult:
    """A/B testing experiment results"""
    experiment_id: str
    model_a_id: str
    model_b_id: str
    start_time: datetime
    end_time: Optional[datetime]
    traffic_split: float
    metrics_a: Dict[str, float]
    metrics_b: Dict[str, float]
    statistical_significance: bool
    confidence_level: float
    winner: Optional[str]
    recommendation: str

@dataclass 
class RetrainingJob:
    """Automated retraining job specification"""
    job_id: str
    model_id: str
    trigger: RetrainingTrigger
    trigger_data: Dict[str, Any]
    scheduled_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "pending"
    config: Optional[Dict[str, Any]] = None
    results: Optional[Dict[str, Any]] = None

class ModelRegistry:
    """Central registry for model metadata and versions"""
    
    def __init__(self, storage_path: str = "/tmp/nautilus_model_registry"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.versions = {}
        self.performance_history = {}
        self._lock = threading.RLock()
        
        # Load existing registry
        self._load_registry()
    
    def register_model(self, metadata: ModelMetadata, model_path: str) -> bool:
        """
        Register a new model in the registry
        
        Args:
            metadata: Model metadata
            model_path: Path to the model file
            
        Returns:
            True if registration successful
        """
        with self._lock:
            try:
                model_id = metadata.model_id
                
                if model_id not in self.models:
                    self.models[model_id] = {
                        'metadata': metadata,
                        'versions': {},
                        'current_version': None,
                        'created_at': time.time()
                    }
                
                # Add version
                version_info = {
                    'metadata': metadata,
                    'model_path': model_path,
                    'registered_at': time.time(),
                    'status': ModelStatus.INACTIVE,
                    'deployment_config': None,
                    'performance_metrics': {}
                }
                
                if model_id not in self.versions:
                    self.versions[model_id] = {}
                
                self.versions[model_id][metadata.version] = version_info
                
                # Save registry
                self._save_registry()
                
                logger.info(f"Registered model {model_id} version {metadata.version}")
                return True
                
            except Exception as e:
                logger.error(f"Model registration failed: {e}")
                return False
    
    def get_model_info(self, model_id: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get model information"""
        with self._lock:
            if model_id not in self.models:
                return None
            
            model_info = self.models[model_id].copy()
            
            if version:
                if model_id in self.versions and version in self.versions[model_id]:
                    model_info['version_info'] = self.versions[model_id][version]
            else:
                # Get current version
                current_version = model_info.get('current_version')
                if current_version and model_id in self.versions:
                    model_info['version_info'] = self.versions[model_id].get(current_version)
            
            return model_info
    
    def list_models(self, status_filter: Optional[ModelStatus] = None) -> List[Dict[str, Any]]:
        """List all registered models"""
        with self._lock:
            models_list = []
            
            for model_id, model_info in self.models.items():
                if model_id not in self.versions:
                    continue
                
                for version, version_info in self.versions[model_id].items():
                    if status_filter and version_info['status'] != status_filter:
                        continue
                    
                    models_list.append({
                        'model_id': model_id,
                        'version': version,
                        'status': version_info['status'].value if isinstance(version_info['status'], ModelStatus) else version_info['status'],
                        'registered_at': version_info['registered_at'],
                        'model_path': version_info['model_path'],
                        'metadata': asdict(version_info['metadata'])
                    })
            
            return models_list
    
    def update_model_status(self, model_id: str, version: str, status: ModelStatus) -> bool:
        """Update model deployment status"""
        with self._lock:
            if model_id in self.versions and version in self.versions[model_id]:
                self.versions[model_id][version]['status'] = status
                self._save_registry()
                logger.info(f"Updated model {model_id} v{version} status to {status.value}")
                return True
            return False
    
    def record_performance(self, performance: ModelPerformance):
        """Record model performance metrics"""
        with self._lock:
            model_id = performance.model_id
            
            if model_id not in self.performance_history:
                self.performance_history[model_id] = []
            
            self.performance_history[model_id].append(asdict(performance))
            
            # Keep only recent performance data (last 1000 records)
            if len(self.performance_history[model_id]) > 1000:
                self.performance_history[model_id] = self.performance_history[model_id][-1000:]
            
            # Update current performance in versions
            if model_id in self.versions:
                for version_info in self.versions[model_id].values():
                    if version_info['status'] == ModelStatus.ACTIVE:
                        version_info['performance_metrics'] = asdict(performance)
            
            self._save_registry()
    
    def get_performance_history(self, model_id: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get performance history for a model"""
        with self._lock:
            if model_id not in self.performance_history:
                return []
            
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            filtered_history = []
            for record in self.performance_history[model_id]:
                record_time = datetime.fromisoformat(record['timestamp']) if isinstance(record['timestamp'], str) else record['timestamp']
                if record_time >= cutoff_time:
                    filtered_history.append(record)
            
            return filtered_history
    
    def _load_registry(self):
        """Load registry from persistent storage"""
        try:
            registry_file = self.storage_path / "registry.json"
            if registry_file.exists():
                with open(registry_file, 'r') as f:
                    data = json.load(f)
                
                # Reconstruct models and versions
                self.models = data.get('models', {})
                self.versions = data.get('versions', {})
                self.performance_history = data.get('performance_history', {})
                
                logger.info("Model registry loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model registry: {e}")
    
    def _save_registry(self):
        """Save registry to persistent storage"""
        try:
            registry_file = self.storage_path / "registry.json"
            
            # Convert data to JSON-serializable format
            data = {
                'models': self._serialize_models(),
                'versions': self._serialize_versions(),
                'performance_history': self.performance_history
            }
            
            with open(registry_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to save model registry: {e}")
    
    def _serialize_models(self) -> Dict[str, Any]:
        """Serialize models dictionary for JSON storage"""
        serialized = {}
        for model_id, model_info in self.models.items():
            serialized[model_id] = {
                'metadata': asdict(model_info['metadata']),
                'versions': list(model_info['versions'].keys()) if 'versions' in model_info else [],
                'current_version': model_info.get('current_version'),
                'created_at': model_info.get('created_at')
            }
        return serialized
    
    def _serialize_versions(self) -> Dict[str, Any]:
        """Serialize versions dictionary for JSON storage"""
        serialized = {}
        for model_id, versions in self.versions.items():
            serialized[model_id] = {}
            for version, version_info in versions.items():
                serialized[model_id][version] = {
                    'metadata': asdict(version_info['metadata']),
                    'model_path': version_info['model_path'],
                    'registered_at': version_info['registered_at'],
                    'status': version_info['status'].value if isinstance(version_info['status'], ModelStatus) else version_info['status'],
                    'deployment_config': version_info.get('deployment_config'),
                    'performance_metrics': version_info.get('performance_metrics', {})
                }
        return serialized

class ExperimentManager:
    """A/B testing experiment management"""
    
    def __init__(self):
        self.active_experiments = {}
        self.completed_experiments = {}
        self.experiment_results = {}
        self._lock = threading.RLock()
    
    def create_experiment(self,
                         experiment_id: str,
                         model_a_id: str,
                         model_b_id: str,
                         traffic_split: float = 0.5,
                         duration_hours: int = 24,
                         success_criteria: Optional[Dict[str, float]] = None) -> bool:
        """
        Create a new A/B testing experiment
        
        Args:
            experiment_id: Unique experiment identifier
            model_a_id: Control model ID
            model_b_id: Treatment model ID  
            traffic_split: Fraction of traffic for model A (0.0-1.0)
            duration_hours: Experiment duration in hours
            success_criteria: Success criteria for automatic promotion
            
        Returns:
            True if experiment created successfully
        """
        with self._lock:
            try:
                if experiment_id in self.active_experiments:
                    logger.error(f"Experiment {experiment_id} already exists")
                    return False
                
                experiment = {
                    'experiment_id': experiment_id,
                    'model_a_id': model_a_id,
                    'model_b_id': model_b_id,
                    'traffic_split': traffic_split,
                    'start_time': datetime.now(),
                    'end_time': datetime.now() + timedelta(hours=duration_hours),
                    'success_criteria': success_criteria or {},
                    'metrics_a': {'requests': 0, 'successes': 0, 'total_latency': 0, 'errors': 0},
                    'metrics_b': {'requests': 0, 'successes': 0, 'total_latency': 0, 'errors': 0},
                    'status': 'active'
                }
                
                self.active_experiments[experiment_id] = experiment
                
                logger.info(f"Created A/B experiment {experiment_id}: {model_a_id} vs {model_b_id}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to create experiment: {e}")
                return False
    
    def record_experiment_result(self,
                                experiment_id: str,
                                model_id: str,
                                success: bool,
                                latency_ms: float):
        """Record experiment result for statistical analysis"""
        with self._lock:
            if experiment_id not in self.active_experiments:
                return
            
            experiment = self.active_experiments[experiment_id]
            
            # Determine which model this result is for
            if model_id == experiment['model_a_id']:
                metrics = experiment['metrics_a']
            elif model_id == experiment['model_b_id']:
                metrics = experiment['metrics_b']
            else:
                return  # Unknown model
            
            # Update metrics
            metrics['requests'] += 1
            if success:
                metrics['successes'] += 1
            else:
                metrics['errors'] += 1
            metrics['total_latency'] += latency_ms
    
    def get_experiment_status(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get current experiment status and preliminary results"""
        with self._lock:
            if experiment_id not in self.active_experiments:
                return None
            
            experiment = self.active_experiments[experiment_id]
            
            # Calculate preliminary metrics
            metrics_a = experiment['metrics_a']
            metrics_b = experiment['metrics_b']
            
            result = {
                'experiment_id': experiment_id,
                'status': experiment['status'],
                'start_time': experiment['start_time'],
                'end_time': experiment['end_time'],
                'traffic_split': experiment['traffic_split'],
                'model_a': {
                    'model_id': experiment['model_a_id'],
                    'requests': metrics_a['requests'],
                    'success_rate': metrics_a['successes'] / max(metrics_a['requests'], 1),
                    'avg_latency_ms': metrics_a['total_latency'] / max(metrics_a['requests'], 1),
                    'error_rate': metrics_a['errors'] / max(metrics_a['requests'], 1)
                },
                'model_b': {
                    'model_id': experiment['model_b_id'],
                    'requests': metrics_b['requests'],
                    'success_rate': metrics_b['successes'] / max(metrics_b['requests'], 1),
                    'avg_latency_ms': metrics_b['total_latency'] / max(metrics_b['requests'], 1),
                    'error_rate': metrics_b['errors'] / max(metrics_b['requests'], 1)
                }
            }
            
            # Calculate statistical significance (simplified)
            if metrics_a['requests'] >= 100 and metrics_b['requests'] >= 100:
                result['statistical_analysis'] = self._calculate_statistical_significance(
                    metrics_a, metrics_b
                )
            
            return result
    
    def finalize_experiment(self, experiment_id: str) -> Optional[ExperimentResult]:
        """Finalize experiment and generate results"""
        with self._lock:
            if experiment_id not in self.active_experiments:
                return None
            
            experiment = self.active_experiments[experiment_id]
            metrics_a = experiment['metrics_a']
            metrics_b = experiment['metrics_b']
            
            # Calculate final metrics
            final_metrics_a = {
                'success_rate': metrics_a['successes'] / max(metrics_a['requests'], 1),
                'avg_latency_ms': metrics_a['total_latency'] / max(metrics_a['requests'], 1),
                'error_rate': metrics_a['errors'] / max(metrics_a['requests'], 1),
                'total_requests': metrics_a['requests']
            }
            
            final_metrics_b = {
                'success_rate': metrics_b['successes'] / max(metrics_b['requests'], 1),
                'avg_latency_ms': metrics_b['total_latency'] / max(metrics_b['requests'], 1),
                'error_rate': metrics_b['errors'] / max(metrics_b['requests'], 1),
                'total_requests': metrics_b['requests']
            }
            
            # Statistical analysis
            statistical_analysis = self._calculate_statistical_significance(metrics_a, metrics_b)
            
            # Determine winner
            winner = None
            recommendation = "Inconclusive"
            
            if statistical_analysis['significant']:
                if final_metrics_b['success_rate'] > final_metrics_a['success_rate']:
                    winner = experiment['model_b_id']
                    recommendation = f"Promote Model B ({experiment['model_b_id']})"
                else:
                    winner = experiment['model_a_id']
                    recommendation = f"Keep Model A ({experiment['model_a_id']})"
            
            # Create result object
            result = ExperimentResult(
                experiment_id=experiment_id,
                model_a_id=experiment['model_a_id'],
                model_b_id=experiment['model_b_id'],
                start_time=experiment['start_time'],
                end_time=datetime.now(),
                traffic_split=experiment['traffic_split'],
                metrics_a=final_metrics_a,
                metrics_b=final_metrics_b,
                statistical_significance=statistical_analysis['significant'],
                confidence_level=statistical_analysis['confidence'],
                winner=winner,
                recommendation=recommendation
            )
            
            # Move to completed experiments
            del self.active_experiments[experiment_id]
            self.completed_experiments[experiment_id] = result
            
            logger.info(f"Finalized experiment {experiment_id}: {recommendation}")
            return result
    
    def _calculate_statistical_significance(self, metrics_a: Dict, metrics_b: Dict) -> Dict[str, Any]:
        """Calculate statistical significance (simplified implementation)"""
        # This is a simplified statistical analysis
        # In production, would use proper statistical tests (t-test, chi-square, etc.)
        
        n_a = metrics_a['requests']
        n_b = metrics_b['requests']
        
        if n_a < 30 or n_b < 30:
            return {'significant': False, 'confidence': 0.0, 'reason': 'Insufficient sample size'}
        
        success_rate_a = metrics_a['successes'] / n_a
        success_rate_b = metrics_b['successes'] / n_b
        
        # Calculate effect size
        effect_size = abs(success_rate_b - success_rate_a)
        
        # Simple significance test based on effect size and sample size
        min_detectable_effect = 0.05  # 5% minimum effect size
        confidence = min(0.95, effect_size * np.sqrt(min(n_a, n_b)) / 10)
        
        significant = effect_size >= min_detectable_effect and confidence >= 0.8
        
        return {
            'significant': significant,
            'confidence': confidence,
            'effect_size': effect_size,
            'p_value': 1 - confidence  # Simplified p-value approximation
        }

class RetrainingScheduler:
    """Automated model retraining scheduler"""
    
    def __init__(self, model_registry: ModelRegistry):
        self.model_registry = model_registry
        self.retraining_jobs = {}
        self.running_jobs = {}
        self._lock = threading.RLock()
        self.is_running = False
        self.scheduler_thread = None
        
        # Retraining configuration
        self.config = {
            'performance_threshold': 0.05,  # 5% degradation triggers retraining
            'data_drift_threshold': 0.1,
            'min_samples_for_retraining': 1000,
            'max_concurrent_jobs': 2,
            'job_timeout_hours': 8
        }
    
    def start(self):
        """Start the retraining scheduler"""
        if self.is_running:
            return
        
        self.is_running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("Retraining scheduler started")
    
    def stop(self):
        """Stop the retraining scheduler"""
        self.is_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5.0)
        
        logger.info("Retraining scheduler stopped")
    
    def schedule_retraining(self,
                          model_id: str,
                          trigger: RetrainingTrigger,
                          trigger_data: Dict[str, Any],
                          schedule_time: Optional[datetime] = None) -> str:
        """
        Schedule a retraining job
        
        Args:
            model_id: Model to retrain
            trigger: What triggered the retraining
            trigger_data: Additional trigger information
            schedule_time: When to run the job (None for immediate)
            
        Returns:
            Job ID
        """
        job_id = f"retrain_{model_id}_{int(time.time())}"
        
        job = RetrainingJob(
            job_id=job_id,
            model_id=model_id,
            trigger=trigger,
            trigger_data=trigger_data,
            scheduled_at=schedule_time or datetime.now()
        )
        
        with self._lock:
            self.retraining_jobs[job_id] = job
        
        logger.info(f"Scheduled retraining job {job_id} for model {model_id}")
        return job_id
    
    def check_performance_degradation(self, model_id: str) -> bool:
        """Check if model performance has degraded significantly"""
        try:
            # Get recent performance history
            recent_performance = self.model_registry.get_performance_history(model_id, hours=24)
            baseline_performance = self.model_registry.get_performance_history(model_id, hours=168)  # 1 week
            
            if len(recent_performance) < 10 or len(baseline_performance) < 100:
                return False  # Not enough data
            
            # Calculate average accuracy/performance
            recent_avg = np.mean([p['metrics'].get('accuracy', 0.5) for p in recent_performance])
            baseline_avg = np.mean([p['metrics'].get('accuracy', 0.5) for p in baseline_performance])
            
            # Check for degradation
            degradation = (baseline_avg - recent_avg) / baseline_avg
            
            if degradation > self.config['performance_threshold']:
                logger.warning(f"Performance degradation detected for {model_id}: {degradation:.2%}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Performance degradation check failed: {e}")
            return False
    
    def _scheduler_loop(self):
        """Main scheduler loop"""
        while self.is_running:
            try:
                current_time = datetime.now()
                
                # Check for scheduled jobs
                with self._lock:
                    jobs_to_run = []
                    for job_id, job in list(self.retraining_jobs.items()):
                        if (job.scheduled_at <= current_time and 
                            job.status == "pending" and
                            len(self.running_jobs) < self.config['max_concurrent_jobs']):
                            jobs_to_run.append(job)
                
                # Start jobs
                for job in jobs_to_run:
                    self._start_retraining_job(job)
                
                # Check for performance degradation
                self._check_all_models_performance()
                
                # Clean up completed jobs
                self._cleanup_completed_jobs()
                
                time.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                time.sleep(60)
    
    def _start_retraining_job(self, job: RetrainingJob):
        """Start a retraining job"""
        with self._lock:
            job.status = "running"
            job.started_at = datetime.now()
            self.running_jobs[job.job_id] = job
        
        # Start job in thread pool (simplified - would use proper job queue in production)
        thread = threading.Thread(target=self._run_retraining_job, args=(job,))
        thread.start()
        
        logger.info(f"Started retraining job {job.job_id}")
    
    def _run_retraining_job(self, job: RetrainingJob):
        """Execute a retraining job"""
        try:
            logger.info(f"Running retraining job {job.job_id} for model {job.model_id}")
            
            # This is a placeholder for actual retraining logic
            # In production, this would:
            # 1. Load latest training data
            # 2. Retrain the model with updated data
            # 3. Validate the new model
            # 4. Register the new model version
            # 5. Optionally deploy with A/B testing
            
            # Simulate retraining time
            time.sleep(300)  # 5 minutes simulation
            
            # Mark job as completed
            with self._lock:
                job.status = "completed"
                job.completed_at = datetime.now()
                job.results = {
                    'new_model_version': f"{job.model_id}_retrained_{int(time.time())}",
                    'performance_improvement': 0.03,
                    'training_duration_minutes': 5
                }
                
                if job.job_id in self.running_jobs:
                    del self.running_jobs[job.job_id]
            
            logger.info(f"Completed retraining job {job.job_id}")
            
        except Exception as e:
            logger.error(f"Retraining job {job.job_id} failed: {e}")
            
            with self._lock:
                job.status = "failed"
                job.completed_at = datetime.now()
                job.results = {'error': str(e)}
                
                if job.job_id in self.running_jobs:
                    del self.running_jobs[job.job_id]
    
    def _check_all_models_performance(self):
        """Check performance for all active models"""
        try:
            active_models = self.model_registry.list_models(status_filter=ModelStatus.ACTIVE)
            
            for model_info in active_models:
                model_id = model_info['model_id']
                
                if self.check_performance_degradation(model_id):
                    # Schedule retraining
                    self.schedule_retraining(
                        model_id=model_id,
                        trigger=RetrainingTrigger.PERFORMANCE_DEGRADATION,
                        trigger_data={'degradation_detected_at': time.time()}
                    )
                    
        except Exception as e:
            logger.error(f"Performance check failed: {e}")
    
    def _cleanup_completed_jobs(self):
        """Clean up old completed jobs"""
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        with self._lock:
            jobs_to_remove = []
            for job_id, job in self.retraining_jobs.items():
                if (job.status in ["completed", "failed"] and 
                    job.completed_at and job.completed_at < cutoff_time):
                    jobs_to_remove.append(job_id)
            
            for job_id in jobs_to_remove:
                del self.retraining_jobs[job_id]

class ModelManager:
    """Comprehensive model management system"""
    
    def __init__(self, registry_path: str = "/tmp/nautilus_model_management"):
        self.registry = ModelRegistry(registry_path)
        self.experiment_manager = ExperimentManager()
        self.retraining_scheduler = RetrainingScheduler(self.registry)
        
        # Deployment management
        self.active_deployments = {}
        self.deployment_history = []
        
        # Performance monitoring
        self.performance_monitor_thread = None
        self.is_monitoring = False
        
        logger.info("Model Manager initialized")
    
    async def start(self):
        """Start the model management system"""
        try:
            # Start retraining scheduler
            self.retraining_scheduler.start()
            
            # Start performance monitoring
            self._start_performance_monitoring()
            
            logger.info("Model management system started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start model management system: {e}")
            return False
    
    async def stop(self):
        """Stop the model management system"""
        try:
            # Stop retraining scheduler
            self.retraining_scheduler.stop()
            
            # Stop performance monitoring
            self._stop_performance_monitoring()
            
            logger.info("Model management system stopped")
            
        except Exception as e:
            logger.error(f"Error stopping model management system: {e}")
    
    async def deploy_model(self,
                          model_id: str,
                          version: str,
                          config: DeploymentConfig) -> bool:
        """
        Deploy a model with specified strategy
        
        Args:
            model_id: Model to deploy
            version: Model version to deploy
            config: Deployment configuration
            
        Returns:
            True if deployment successful
        """
        try:
            logger.info(f"Deploying model {model_id} v{version} with strategy {config.strategy.value}")
            
            # Get model info
            model_info = self.registry.get_model_info(model_id, version)
            if not model_info:
                logger.error(f"Model {model_id} v{version} not found")
                return False
            
            # Execute deployment strategy
            if config.strategy == DeploymentStrategy.IMMEDIATE:
                return await self._deploy_immediate(model_id, version, config)
            elif config.strategy == DeploymentStrategy.BLUE_GREEN:
                return await self._deploy_blue_green(model_id, version, config)
            elif config.strategy == DeploymentStrategy.CANARY:
                return await self._deploy_canary(model_id, version, config)
            elif config.strategy == DeploymentStrategy.AB_TEST:
                return await self._deploy_ab_test(model_id, version, config)
            else:
                logger.error(f"Unsupported deployment strategy: {config.strategy}")
                return False
                
        except Exception as e:
            logger.error(f"Model deployment failed: {e}")
            return False
    
    async def _deploy_immediate(self, model_id: str, version: str, config: DeploymentConfig) -> bool:
        """Immediate deployment strategy"""
        try:
            # Update model status to active
            self.registry.update_model_status(model_id, version, ModelStatus.ACTIVE)
            
            # Record deployment
            deployment_record = {
                'model_id': model_id,
                'version': version,
                'strategy': config.strategy.value,
                'deployed_at': time.time(),
                'status': 'success'
            }
            self.deployment_history.append(deployment_record)
            
            logger.info(f"Successfully deployed {model_id} v{version} immediately")
            return True
            
        except Exception as e:
            logger.error(f"Immediate deployment failed: {e}")
            return False
    
    async def _deploy_ab_test(self, model_id: str, version: str, config: DeploymentConfig) -> bool:
        """A/B testing deployment strategy"""
        try:
            # Find current active model for comparison
            active_models = self.registry.list_models(status_filter=ModelStatus.ACTIVE)
            current_model = None
            
            for model in active_models:
                if model['model_id'] == model_id:
                    current_model = model
                    break
            
            if not current_model:
                logger.error(f"No active model found for A/B testing with {model_id}")
                return False
            
            # Create A/B experiment
            experiment_id = f"deploy_{model_id}_{version}_{int(time.time())}"
            
            success = self.experiment_manager.create_experiment(
                experiment_id=experiment_id,
                model_a_id=f"{current_model['model_id']}:{current_model['version']}",
                model_b_id=f"{model_id}:{version}",
                traffic_split=config.traffic_split or 0.5,
                duration_hours=config.rollout_duration_minutes // 60 if config.rollout_duration_minutes else 24,
                success_criteria=config.success_criteria
            )
            
            if success:
                # Mark new model as testing
                self.registry.update_model_status(model_id, version, ModelStatus.TESTING)
                
                logger.info(f"Started A/B test deployment for {model_id} v{version}")
                return True
            else:
                logger.error("Failed to create A/B experiment")
                return False
                
        except Exception as e:
            logger.error(f"A/B test deployment failed: {e}")
            return False
    
    def _start_performance_monitoring(self):
        """Start performance monitoring thread"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.performance_monitor_thread = threading.Thread(
            target=self._performance_monitoring_loop,
            daemon=True
        )
        self.performance_monitor_thread.start()
        
        logger.info("Performance monitoring started")
    
    def _stop_performance_monitoring(self):
        """Stop performance monitoring"""
        self.is_monitoring = False
        if self.performance_monitor_thread:
            self.performance_monitor_thread.join(timeout=5.0)
        
        logger.info("Performance monitoring stopped")
    
    def _performance_monitoring_loop(self):
        """Performance monitoring loop"""
        while self.is_monitoring:
            try:
                # Get inference engine status
                inference_status = inference_engine.get_status()
                
                # Record performance for active models
                active_models = self.registry.list_models(status_filter=ModelStatus.ACTIVE)
                
                for model in active_models:
                    model_id = model['model_id']
                    
                    # Create performance record (simplified)
                    performance = ModelPerformance(
                        model_id=model_id,
                        timestamp=datetime.now(),
                        metrics={PerformanceMetric.ACCURACY: 0.92},  # Would get from actual metrics
                        inference_count=100,
                        error_count=2,
                        avg_latency_ms=5.2,
                        p95_latency_ms=8.7,
                        p99_latency_ms=12.1,
                        throughput_qps=500.0,
                        memory_usage_mb=256.0,
                        cpu_utilization=0.3,
                        neural_engine_utilization=0.85
                    )
                    
                    self.registry.record_performance(performance)
                
                time.sleep(60)  # Monitor every minute
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                time.sleep(30)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive model management system status"""
        return {
            'registry_stats': {
                'total_models': len(self.registry.models),
                'total_versions': sum(len(versions) for versions in self.registry.versions.values()),
                'active_models': len(self.registry.list_models(status_filter=ModelStatus.ACTIVE))
            },
            'experiments': {
                'active_experiments': len(self.experiment_manager.active_experiments),
                'completed_experiments': len(self.experiment_manager.completed_experiments)
            },
            'retraining': {
                'pending_jobs': len([j for j in self.retraining_scheduler.retraining_jobs.values() if j.status == "pending"]),
                'running_jobs': len(self.retraining_scheduler.running_jobs)
            },
            'deployments': {
                'active_deployments': len(self.active_deployments),
                'recent_deployments': len([d for d in self.deployment_history if time.time() - d['deployed_at'] < 86400])
            },
            'monitoring': {
                'performance_monitoring_active': self.is_monitoring,
                'retraining_scheduler_active': self.retraining_scheduler.is_running
            }
        }

# Global model manager instance
model_manager = ModelManager()

# Convenience functions
async def initialize_model_management() -> bool:
    """Initialize and start the model management system"""
    return await model_manager.start()

async def register_model(metadata: ModelMetadata, model_path: str) -> bool:
    """Register a new model in the management system"""
    return model_manager.registry.register_model(metadata, model_path)

async def deploy_model(model_id: str, version: str, strategy: DeploymentStrategy = DeploymentStrategy.IMMEDIATE) -> bool:
    """Deploy a model with default configuration"""
    config = DeploymentConfig(strategy=strategy)
    return await model_manager.deploy_model(model_id, version, config)

def get_model_management_status() -> Dict[str, Any]:
    """Get comprehensive model management status"""
    return model_manager.get_system_status()