"""
Workload Classifier for Intelligent Task Distribution
====================================================

Machine learning-based workload classification system that automatically categorizes
tasks and operations for optimal CPU core allocation on M4 Max architecture.
"""

import os
import sys
import time
import threading
import pickle
import json
from typing import Dict, List, Optional, Tuple, Set, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

from .cpu_affinity import WorkloadPriority
from .process_manager import ProcessClass

logger = logging.getLogger(__name__)

class WorkloadCategory(Enum):
    """Categories of workloads for classification"""
    TRADING_EXECUTION = "trading_execution"      # Order placement, execution
    MARKET_DATA = "market_data"                  # Data ingestion, parsing
    RISK_CALCULATION = "risk_calculation"        # Risk metrics, position sizing
    ANALYTICS = "analytics"                      # Performance analysis, reporting
    ML_INFERENCE = "ml_inference"                # Model inference, predictions
    DATA_PROCESSING = "data_processing"          # ETL, data transformation
    BACKGROUND_MAINTENANCE = "background_maintenance"  # Cleanup, archiving
    SYSTEM_MONITORING = "system_monitoring"      # Health checks, monitoring

@dataclass
class WorkloadFeatures:
    """Feature vector for workload classification"""
    # Timing characteristics
    execution_time_ms: float = 0.0
    cpu_time_ms: float = 0.0
    wall_time_ms: float = 0.0
    
    # Resource usage
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    io_read_mb: float = 0.0
    io_write_mb: float = 0.0
    network_bytes: float = 0.0
    
    # Concurrency characteristics
    thread_count: int = 1
    is_blocking: bool = False
    has_external_deps: bool = False
    
    # Frequency and patterns
    invocation_frequency: float = 0.0  # calls per second
    time_of_day: float = 0.0  # 0-23 hour
    market_session: int = 0  # 0=pre, 1=open, 2=close, 3=after
    
    # Context features
    function_name: str = ""
    module_name: str = ""
    stack_depth: int = 0
    has_database_access: bool = False
    has_network_access: bool = False
    
    # Performance requirements
    latency_sensitive: bool = False
    throughput_critical: bool = False
    
    def to_array(self) -> np.ndarray:
        """Convert features to numpy array for ML model"""
        return np.array([
            self.execution_time_ms,
            self.cpu_time_ms,
            self.wall_time_ms,
            self.cpu_usage_percent,
            self.memory_usage_mb,
            self.io_read_mb,
            self.io_write_mb,
            self.network_bytes,
            self.thread_count,
            float(self.is_blocking),
            float(self.has_external_deps),
            self.invocation_frequency,
            self.time_of_day,
            self.market_session,
            self.stack_depth,
            float(self.has_database_access),
            float(self.has_network_access),
            float(self.latency_sensitive),
            float(self.throughput_critical)
        ])

@dataclass
class WorkloadSample:
    """Training sample for the classifier"""
    features: WorkloadFeatures
    category: WorkloadCategory
    timestamp: float = field(default_factory=time.time)
    accuracy: float = 1.0  # Confidence in the labeling

class WorkloadClassifier:
    """
    Machine learning-based workload classifier for trading platform optimization
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or "workload_classifier_model.pkl"
        self.scaler_path = self.model_path.replace('.pkl', '_scaler.pkl')
        
        # ML components
        self.model: Optional[RandomForestClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.is_trained = False
        
        # Training data
        self.training_samples: List[WorkloadSample] = []
        self.feature_buffer: deque = deque(maxlen=10000)
        
        # Classification cache
        self.classification_cache: Dict[str, Tuple[WorkloadCategory, float]] = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Pattern recognition
        self.pattern_rules: Dict[str, WorkloadCategory] = {}
        self.function_patterns: Dict[str, WorkloadCategory] = {}
        
        # Statistics
        self.classification_stats = defaultdict(int)
        self.accuracy_history: deque = deque(maxlen=1000)
        
        self._lock = threading.RLock()
        
        # Initialize with heuristic rules
        self._initialize_heuristic_rules()
        
        # Load existing model if available
        self._load_model()
        
        # Start automatic retraining
        self._start_retraining_thread()
    
    def _initialize_heuristic_rules(self) -> None:
        """Initialize heuristic classification rules"""
        
        # Function name patterns
        self.function_patterns = {
            "execute_order": WorkloadCategory.TRADING_EXECUTION,
            "place_order": WorkloadCategory.TRADING_EXECUTION,
            "cancel_order": WorkloadCategory.TRADING_EXECUTION,
            "process_market_data": WorkloadCategory.MARKET_DATA,
            "parse_tick": WorkloadCategory.MARKET_DATA,
            "update_orderbook": WorkloadCategory.MARKET_DATA,
            "calculate_risk": WorkloadCategory.RISK_CALCULATION,
            "update_positions": WorkloadCategory.RISK_CALCULATION,
            "check_limits": WorkloadCategory.RISK_CALCULATION,
            "generate_report": WorkloadCategory.ANALYTICS,
            "calculate_pnl": WorkloadCategory.ANALYTICS,
            "predict_price": WorkloadCategory.ML_INFERENCE,
            "run_model": WorkloadCategory.ML_INFERENCE,
            "infer": WorkloadCategory.ML_INFERENCE,
            "transform_data": WorkloadCategory.DATA_PROCESSING,
            "extract_features": WorkloadCategory.DATA_PROCESSING,
            "cleanup": WorkloadCategory.BACKGROUND_MAINTENANCE,
            "archive": WorkloadCategory.BACKGROUND_MAINTENANCE,
            "health_check": WorkloadCategory.SYSTEM_MONITORING,
            "monitor": WorkloadCategory.SYSTEM_MONITORING
        }
        
        # Module patterns
        self.pattern_rules = {
            "trading_engine": WorkloadCategory.TRADING_EXECUTION,
            "market_data": WorkloadCategory.MARKET_DATA,
            "risk_management": WorkloadCategory.RISK_CALCULATION,
            "analytics": WorkloadCategory.ANALYTICS,
            "ml": WorkloadCategory.ML_INFERENCE,
            "data_processing": WorkloadCategory.DATA_PROCESSING,
            "monitoring": WorkloadCategory.SYSTEM_MONITORING
        }
        
        logger.info("Initialized heuristic classification rules")
    
    def _load_model(self) -> bool:
        """Load trained model from disk"""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                self.is_trained = True
                
                logger.info("Loaded pre-trained workload classification model")
                return True
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
        
        return False
    
    def _save_model(self) -> bool:
        """Save trained model to disk"""
        try:
            if self.model is not None and self.scaler is not None:
                joblib.dump(self.model, self.model_path)
                joblib.dump(self.scaler, self.scaler_path)
                
                logger.info(f"Saved workload classification model to {self.model_path}")
                return True
                
        except Exception as e:
            logger.error(f"Error saving model: {e}")
        
        return False
    
    def _start_retraining_thread(self) -> None:
        """Start background thread for automatic model retraining"""
        def retrain_loop():
            while True:
                try:
                    time.sleep(3600)  # Retrain every hour
                    
                    with self._lock:
                        if len(self.training_samples) >= 100:  # Minimum samples for retraining
                            self._retrain_model()
                            
                except Exception as e:
                    logger.error(f"Error in retraining loop: {e}")
                    time.sleep(300)  # Wait 5 minutes on error
        
        retrain_thread = threading.Thread(target=retrain_loop, daemon=True, name="WorkloadRetrainer")
        retrain_thread.start()
    
    def extract_features(
        self,
        function_name: str = "",
        module_name: str = "",
        execution_time_ms: float = 0.0,
        cpu_usage: float = 0.0,
        memory_usage_mb: float = 0.0,
        io_stats: Optional[Dict] = None,
        thread_count: int = 1,
        context: Optional[Dict] = None
    ) -> WorkloadFeatures:
        """
        Extract features from execution context
        """
        current_time = time.time()
        io_stats = io_stats or {}
        context = context or {}
        
        features = WorkloadFeatures(
            execution_time_ms=execution_time_ms,
            cpu_usage_percent=cpu_usage,
            memory_usage_mb=memory_usage_mb,
            io_read_mb=io_stats.get('read_mb', 0.0),
            io_write_mb=io_stats.get('write_mb', 0.0),
            network_bytes=io_stats.get('network_bytes', 0.0),
            thread_count=thread_count,
            function_name=function_name,
            module_name=module_name,
            time_of_day=time.localtime(current_time).tm_hour,
            market_session=self._get_market_session(current_time),
            has_database_access=self._check_database_access(function_name, module_name),
            has_network_access=self._check_network_access(function_name, module_name),
            latency_sensitive=self._is_latency_sensitive(function_name, context),
            throughput_critical=self._is_throughput_critical(function_name, context)
        )
        
        # Calculate derived features
        if execution_time_ms > 0:
            features.cpu_time_ms = execution_time_ms * (cpu_usage / 100.0)
            features.wall_time_ms = execution_time_ms
        
        # Determine blocking behavior
        features.is_blocking = self._is_blocking_operation(function_name, context)
        
        # Check for external dependencies
        features.has_external_deps = (
            features.has_database_access or 
            features.has_network_access or
            'external' in function_name.lower()
        )
        
        # Estimate invocation frequency (simplified)
        features.invocation_frequency = self._estimate_frequency(function_name)
        
        return features
    
    def _get_market_session(self, timestamp: float) -> int:
        """Get market session (0=pre, 1=open, 2=close, 3=after)"""
        hour = time.localtime(timestamp).tm_hour
        
        # US market hours (simplified)
        if 4 <= hour < 9:    # 4 AM - 9 AM ET (pre-market)
            return 0
        elif 9 <= hour < 16:  # 9 AM - 4 PM ET (market open)
            return 1
        elif 16 <= hour < 20: # 4 PM - 8 PM ET (after hours)
            return 3
        else:
            return 2  # Market closed
    
    def _check_database_access(self, function_name: str, module_name: str) -> bool:
        """Check if operation involves database access"""
        db_keywords = ['db', 'database', 'sql', 'query', 'insert', 'update', 'select']
        text = f"{function_name} {module_name}".lower()
        return any(keyword in text for keyword in db_keywords)
    
    def _check_network_access(self, function_name: str, module_name: str) -> bool:
        """Check if operation involves network access"""
        net_keywords = ['http', 'api', 'request', 'fetch', 'download', 'upload', 'websocket']
        text = f"{function_name} {module_name}".lower()
        return any(keyword in text for keyword in net_keywords)
    
    def _is_latency_sensitive(self, function_name: str, context: Dict) -> bool:
        """Determine if operation is latency sensitive"""
        latency_keywords = ['order', 'execute', 'trade', 'tick', 'quote', 'urgent']
        return (
            any(keyword in function_name.lower() for keyword in latency_keywords) or
            context.get('latency_sensitive', False)
        )
    
    def _is_throughput_critical(self, function_name: str, context: Dict) -> bool:
        """Determine if operation is throughput critical"""
        throughput_keywords = ['batch', 'bulk', 'process', 'transform', 'calculate']
        return (
            any(keyword in function_name.lower() for keyword in throughput_keywords) or
            context.get('throughput_critical', False)
        )
    
    def _is_blocking_operation(self, function_name: str, context: Dict) -> bool:
        """Determine if operation is blocking"""
        blocking_keywords = ['wait', 'block', 'sync', 'lock']
        return (
            any(keyword in function_name.lower() for keyword in blocking_keywords) or
            context.get('blocking', False)
        )
    
    def _estimate_frequency(self, function_name: str) -> float:
        """Estimate invocation frequency based on function name"""
        # This is a simplified heuristic - in production, you'd track actual frequencies
        frequency_map = {
            'tick': 1000.0,      # Market ticks are very frequent
            'quote': 100.0,      # Quotes are frequent
            'order': 10.0,       # Orders are less frequent
            'report': 0.1,       # Reports are infrequent
            'cleanup': 0.01      # Cleanup is rare
        }
        
        for keyword, freq in frequency_map.items():
            if keyword in function_name.lower():
                return freq
        
        return 1.0  # Default frequency
    
    def classify_workload(
        self,
        features: WorkloadFeatures,
        use_cache: bool = True
    ) -> Tuple[WorkloadCategory, float]:
        """
        Classify a workload based on its features
        Returns (category, confidence)
        """
        # Create cache key
        cache_key = f"{features.function_name}_{features.module_name}"
        
        # Check cache
        if use_cache and cache_key in self.classification_cache:
            category, confidence = self.classification_cache[cache_key]
            # Check if cache is still valid
            if time.time() - confidence < self.cache_ttl:
                return category, confidence
        
        # Try heuristic rules first
        category, confidence = self._classify_heuristic(features)
        
        # If ML model is available and heuristic confidence is low, use ML
        if self.is_trained and confidence < 0.8:
            ml_category, ml_confidence = self._classify_ml(features)
            if ml_confidence > confidence:
                category, confidence = ml_category, ml_confidence
        
        # Cache result
        if use_cache:
            self.classification_cache[cache_key] = (category, time.time())
        
        # Update statistics
        with self._lock:
            self.classification_stats[category] += 1
        
        return category, confidence
    
    def _classify_heuristic(self, features: WorkloadFeatures) -> Tuple[WorkloadCategory, float]:
        """Classify using heuristic rules"""
        
        # Check function name patterns
        for pattern, category in self.function_patterns.items():
            if pattern in features.function_name.lower():
                return category, 0.9
        
        # Check module patterns
        for pattern, category in self.pattern_rules.items():
            if pattern in features.module_name.lower():
                return category, 0.7
        
        # Feature-based heuristics
        if features.latency_sensitive or features.execution_time_ms < 1.0:
            return WorkloadCategory.TRADING_EXECUTION, 0.6
        
        if features.has_database_access and features.throughput_critical:
            return WorkloadCategory.DATA_PROCESSING, 0.6
        
        if features.memory_usage_mb > 100 and 'ml' in features.module_name.lower():
            return WorkloadCategory.ML_INFERENCE, 0.6
        
        if features.execution_time_ms > 1000:  # Long-running tasks
            return WorkloadCategory.BACKGROUND_MAINTENANCE, 0.5
        
        # Default classification
        return WorkloadCategory.ANALYTICS, 0.3
    
    def _classify_ml(self, features: WorkloadFeatures) -> Tuple[WorkloadCategory, float]:
        """Classify using ML model"""
        try:
            if not self.is_trained or self.model is None or self.scaler is None:
                return WorkloadCategory.ANALYTICS, 0.0
            
            # Convert features to array
            feature_array = features.to_array().reshape(1, -1)
            
            # Scale features
            scaled_features = self.scaler.transform(feature_array)
            
            # Predict
            probabilities = self.model.predict_proba(scaled_features)[0]
            predicted_class_idx = np.argmax(probabilities)
            confidence = probabilities[predicted_class_idx]
            
            # Map class index to category
            categories = list(WorkloadCategory)
            predicted_category = categories[predicted_class_idx]
            
            return predicted_category, confidence
            
        except Exception as e:
            logger.error(f"Error in ML classification: {e}")
            return WorkloadCategory.ANALYTICS, 0.0
    
    def add_training_sample(
        self,
        features: WorkloadFeatures,
        actual_category: WorkloadCategory,
        accuracy: float = 1.0
    ) -> None:
        """Add a training sample"""
        sample = WorkloadSample(
            features=features,
            category=actual_category,
            accuracy=accuracy
        )
        
        with self._lock:
            self.training_samples.append(sample)
            
            # Limit training samples to prevent memory issues
            if len(self.training_samples) > 50000:
                # Keep most recent 80% of samples
                keep_count = int(len(self.training_samples) * 0.8)
                self.training_samples = self.training_samples[-keep_count:]
        
        logger.debug(f"Added training sample: {actual_category.value}")
    
    def _retrain_model(self) -> bool:
        """Retrain the ML model with accumulated samples"""
        try:
            logger.info("Starting model retraining...")
            
            if len(self.training_samples) < 50:
                logger.warning("Insufficient training samples for retraining")
                return False
            
            # Prepare training data
            X = []
            y = []
            
            for sample in self.training_samples:
                X.append(sample.features.to_array())
                y.append(list(WorkloadCategory).index(sample.category))
            
            X = np.array(X)
            y = np.array(y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Create and train scaler
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Create and train model
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Update tracking
            self.accuracy_history.append(accuracy)
            self.is_trained = True
            
            # Save model
            self._save_model()
            
            logger.info(f"Model retrained successfully. Accuracy: {accuracy:.3f}")
            
            # Log feature importance
            feature_names = [
                'execution_time_ms', 'cpu_time_ms', 'wall_time_ms', 'cpu_usage_percent',
                'memory_usage_mb', 'io_read_mb', 'io_write_mb', 'network_bytes',
                'thread_count', 'is_blocking', 'has_external_deps', 'invocation_frequency',
                'time_of_day', 'market_session', 'stack_depth', 'has_database_access',
                'has_network_access', 'latency_sensitive', 'throughput_critical'
            ]
            
            importance = self.model.feature_importances_
            top_features = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)[:5]
            
            logger.info(f"Top 5 important features: {top_features}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error retraining model: {e}")
            return False
    
    def get_workload_priority(self, category: WorkloadCategory) -> WorkloadPriority:
        """Map workload category to CPU scheduling priority"""
        priority_map = {
            WorkloadCategory.TRADING_EXECUTION: WorkloadPriority.ULTRA_LOW_LATENCY,
            WorkloadCategory.MARKET_DATA: WorkloadPriority.LOW_LATENCY,
            WorkloadCategory.RISK_CALCULATION: WorkloadPriority.LOW_LATENCY,
            WorkloadCategory.ML_INFERENCE: WorkloadPriority.NORMAL,
            WorkloadCategory.ANALYTICS: WorkloadPriority.NORMAL,
            WorkloadCategory.DATA_PROCESSING: WorkloadPriority.NORMAL,
            WorkloadCategory.SYSTEM_MONITORING: WorkloadPriority.BACKGROUND,
            WorkloadCategory.BACKGROUND_MAINTENANCE: WorkloadPriority.BACKGROUND
        }
        
        return priority_map.get(category, WorkloadPriority.NORMAL)
    
    def get_process_class(self, category: WorkloadCategory) -> ProcessClass:
        """Map workload category to process class"""
        class_map = {
            WorkloadCategory.TRADING_EXECUTION: ProcessClass.TRADING_CORE,
            WorkloadCategory.MARKET_DATA: ProcessClass.TRADING_CORE,
            WorkloadCategory.RISK_CALCULATION: ProcessClass.RISK_MANAGEMENT,
            WorkloadCategory.ML_INFERENCE: ProcessClass.ANALYTICS,
            WorkloadCategory.ANALYTICS: ProcessClass.ANALYTICS,
            WorkloadCategory.DATA_PROCESSING: ProcessClass.DATA_PROCESSING,
            WorkloadCategory.SYSTEM_MONITORING: ProcessClass.BACKGROUND,
            WorkloadCategory.BACKGROUND_MAINTENANCE: ProcessClass.BACKGROUND
        }
        
        return class_map.get(category, ProcessClass.ANALYTICS)
    
    def get_classification_stats(self) -> Dict:
        """Get classification statistics"""
        with self._lock:
            total_classifications = sum(self.classification_stats.values())
            
            stats = {
                "total_classifications": total_classifications,
                "is_trained": self.is_trained,
                "training_samples": len(self.training_samples),
                "cache_size": len(self.classification_cache),
                "category_distribution": dict(self.classification_stats),
                "model_accuracy": list(self.accuracy_history)[-10:] if self.accuracy_history else []
            }
            
            if total_classifications > 0:
                stats["category_percentages"] = {
                    category.value: (count / total_classifications) * 100
                    for category, count in self.classification_stats.items()
                }
            
            return stats
    
    def clear_cache(self) -> None:
        """Clear classification cache"""
        with self._lock:
            self.classification_cache.clear()
        logger.info("Classification cache cleared")
    
    def export_training_data(self, filename: str) -> bool:
        """Export training data for analysis"""
        try:
            with self._lock:
                export_data = {
                    "timestamp": time.time(),
                    "sample_count": len(self.training_samples),
                    "samples": []
                }
                
                for sample in self.training_samples:
                    export_data["samples"].append({
                        "category": sample.category.value,
                        "timestamp": sample.timestamp,
                        "accuracy": sample.accuracy,
                        "features": {
                            "execution_time_ms": sample.features.execution_time_ms,
                            "cpu_usage_percent": sample.features.cpu_usage_percent,
                            "memory_usage_mb": sample.features.memory_usage_mb,
                            "function_name": sample.features.function_name,
                            "module_name": sample.features.module_name,
                            "latency_sensitive": sample.features.latency_sensitive,
                            "throughput_critical": sample.features.throughput_critical
                        }
                    })
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Exported {len(self.training_samples)} training samples to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting training data: {e}")
            return False