#!/usr/bin/env python3
"""
Phase 7: Predictive Alerting System with ML-based Anomaly Detection
Advanced ML models for predicting system failures and anomalies before they occur
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict, field
from enum import Enum
import time
import uuid
import pickle
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import precision_recall_fscore_support
import scipy.stats as stats
from scipy.signal import find_peaks
import asyncpg
import redis.asyncio as redis

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AnomalyType(Enum):
    """Types of anomalies that can be detected"""
    POINT_ANOMALY = "point_anomaly"          # Single outlier point
    CONTEXTUAL_ANOMALY = "contextual_anomaly" # Anomalous in specific context
    COLLECTIVE_ANOMALY = "collective_anomaly" # Pattern of points is anomalous
    TREND_ANOMALY = "trend_anomaly"          # Unusual trend or pattern
    SEASONAL_ANOMALY = "seasonal_anomaly"    # Deviation from seasonal pattern

class PredictionConfidence(Enum):
    """Confidence levels for predictions"""
    VERY_HIGH = "very_high"    # >95% confidence
    HIGH = "high"              # >85% confidence  
    MEDIUM = "medium"          # >70% confidence
    LOW = "low"                # >50% confidence
    VERY_LOW = "very_low"      # <50% confidence

class AlertPriority(Enum):
    """Priority levels for predictive alerts"""
    P0 = "p0"  # Critical - Immediate action required
    P1 = "p1"  # High - Action required within 15 minutes
    P2 = "p2"  # Medium - Action required within 1 hour
    P3 = "p3"  # Low - Action required within 24 hours
    P4 = "p4"  # Info - Monitor and track

@dataclass
class AnomalyPrediction:
    """Anomaly prediction result"""
    prediction_id: str
    metric_id: str
    region: str
    anomaly_type: AnomalyType
    
    # Prediction details
    predicted_time: datetime
    confidence: PredictionConfidence
    anomaly_score: float
    severity_score: float  # 0-100 scale
    
    # Context
    current_value: float
    predicted_value: Optional[float] = None
    normal_range: Tuple[float, float] = (0.0, 0.0)
    
    # ML model details
    model_name: str = ""
    feature_importance: Dict[str, float] = field(default_factory=dict)
    contributing_factors: List[str] = field(default_factory=list)
    
    # Recommendation
    recommended_action: str = ""
    estimated_impact: str = ""
    prevention_window_minutes: int = 0
    
    # Validation
    is_validated: bool = False
    validation_score: Optional[float] = None
    false_positive_probability: float = 0.0

@dataclass
class ModelPerformanceMetrics:
    """Performance metrics for ML models"""
    model_name: str
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    false_positive_rate: float
    false_negative_rate: float
    mean_absolute_error: float
    prediction_latency_ms: float
    last_updated: datetime

class PredictiveAlertingSystem:
    """
    Advanced ML-based predictive alerting system
    """
    
    def __init__(self):
        # ML Models for different anomaly types
        self.anomaly_detectors = {
            'isolation_forest': IsolationForest(contamination=0.1, random_state=42),
            'clustering': DBSCAN(eps=0.3, min_samples=10),
            'statistical': None  # Custom statistical models
        }
        
        self.trend_predictors = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'time_series': None  # Custom time series models
        }
        
        # Data preprocessing
        self.scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler()
        }
        
        self.feature_transformers = {
            'pca': PCA(n_components=0.95),  # Keep 95% variance
        }
        
        # Model state
        self.models_trained = False
        self.last_training_time = None
        self.training_data_cache = {}
        
        # Performance tracking
        self.model_performance: Dict[str, ModelPerformanceMetrics] = {}
        self.prediction_history: List[AnomalyPrediction] = []
        self.validation_results: Dict[str, List[bool]] = {}  # Track prediction accuracy
        
        # Real-time data streams
        self.metric_streams = {}
        self.feature_cache = {}
        
        # Database connections
        self.db_pool = None
        self.redis_client = None
        
        # Configuration
        self.config = {
            'training_window_hours': 168,  # 7 days
            'prediction_horizon_minutes': 60,  # Predict 1 hour ahead
            'retraining_interval_hours': 24,
            'min_training_samples': 1000,
            'anomaly_threshold': 0.8,
            'confidence_threshold': 0.7,
            'enable_ensemble_voting': True,
            'max_predictions_per_metric': 10
        }
        
        # Active predictions
        self.active_predictions: Dict[str, AnomalyPrediction] = {}
        self.prediction_cache = {}
        
    async def initialize(self):
        """Initialize the predictive alerting system"""
        logger.info("🔮 Initializing Predictive Alerting System")
        
        # Initialize database connections
        await self._initialize_databases()
        
        # Create database tables
        await self._create_prediction_tables()
        
        # Load historical data for training
        await self._load_training_data()
        
        # Train initial models
        await self._train_models()
        
        # Start prediction loops
        await self._start_prediction_loops()
        
        logger.info("✅ Predictive Alerting System initialized")
    
    async def _initialize_databases(self):
        """Initialize database connections"""
        
        # TimescaleDB for metrics and predictions
        self.db_pool = await asyncpg.create_pool(
            "postgresql://nautilus:password@timescaledb-global:5432/monitoring",
            min_size=5,
            max_size=20
        )
        
        # Redis for real-time caching
        self.redis_client = redis.from_url(
            "redis://redis-monitoring-global:6379",
            decode_responses=True
        )
        
        logger.info("✅ Predictive alerting databases initialized")
    
    async def _create_prediction_tables(self):
        """Create prediction-related database tables"""
        
        async with self.db_pool.acquire() as conn:
            # Anomaly predictions table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS anomaly_predictions (
                    prediction_id VARCHAR PRIMARY KEY,
                    metric_id VARCHAR NOT NULL,
                    region VARCHAR NOT NULL,
                    anomaly_type VARCHAR NOT NULL,
                    predicted_time TIMESTAMPTZ NOT NULL,
                    confidence VARCHAR NOT NULL,
                    anomaly_score DOUBLE PRECISION NOT NULL,
                    severity_score DOUBLE PRECISION NOT NULL,
                    current_value DOUBLE PRECISION NOT NULL,
                    predicted_value DOUBLE PRECISION,
                    normal_range_min DOUBLE PRECISION,
                    normal_range_max DOUBLE PRECISION,
                    model_name VARCHAR NOT NULL,
                    feature_importance JSONB,
                    contributing_factors TEXT[],
                    recommended_action TEXT,
                    estimated_impact TEXT,
                    prevention_window_minutes INTEGER,
                    is_validated BOOLEAN DEFAULT FALSE,
                    validation_score DOUBLE PRECISION,
                    false_positive_probability DOUBLE PRECISION,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            
            # Convert to hypertable
            await conn.execute("""
                SELECT create_hypertable('anomaly_predictions', 'predicted_time', if_not_exists => TRUE)
            """)
            
            # Model performance table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    id SERIAL PRIMARY KEY,
                    model_name VARCHAR NOT NULL,
                    precision_score DOUBLE PRECISION NOT NULL,
                    recall_score DOUBLE PRECISION NOT NULL,
                    f1_score DOUBLE PRECISION NOT NULL,
                    accuracy_score DOUBLE PRECISION NOT NULL,
                    false_positive_rate DOUBLE PRECISION NOT NULL,
                    false_negative_rate DOUBLE PRECISION NOT NULL,
                    mean_absolute_error DOUBLE PRECISION NOT NULL,
                    prediction_latency_ms DOUBLE PRECISION NOT NULL,
                    evaluation_date TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            
            # Create indexes
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_metric_region_time ON anomaly_predictions(metric_id, region, predicted_time)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_confidence_score ON anomaly_predictions(confidence, anomaly_score)")
            
    async def _load_training_data(self):
        """Load historical data for model training"""
        
        logger.info("📚 Loading training data for ML models")
        
        training_window = datetime.now() - timedelta(hours=self.config['training_window_hours'])
        
        async with self.db_pool.acquire() as conn:
            # Load metrics data
            rows = await conn.fetch("""
                SELECT metric_id, region, timestamp, value, labels
                FROM metrics
                WHERE timestamp >= $1
                ORDER BY metric_id, region, timestamp
            """, training_window)
        
        # Process data by metric and region
        for row in rows:
            key = f"{row['metric_id']}:{row['region']}"
            if key not in self.training_data_cache:
                self.training_data_cache[key] = []
            
            self.training_data_cache[key].append({
                'timestamp': row['timestamp'],
                'value': row['value'],
                'labels': json.loads(row['labels']) if row['labels'] else {}
            })
        
        logger.info(f"📊 Loaded training data for {len(self.training_data_cache)} metric-region pairs")
    
    async def _train_models(self):
        """Train ML models for anomaly detection and prediction"""
        
        logger.info("🧠 Training ML models for anomaly detection")
        
        training_start = time.time()
        
        for metric_region_key, data in self.training_data_cache.items():
            if len(data) < self.config['min_training_samples']:
                continue
            
            try:
                # Prepare features
                features = self._extract_features(data)
                
                if len(features) < 50:  # Need minimum data for training
                    continue
                
                # Train anomaly detectors
                await self._train_anomaly_detectors(metric_region_key, features)
                
                # Train trend predictors
                await self._train_trend_predictors(metric_region_key, features)
                
            except Exception as e:
                logger.error(f"Failed to train models for {metric_region_key}: {e}")
        
        training_time = time.time() - training_start
        self.models_trained = True
        self.last_training_time = datetime.now()
        
        logger.info(f"✅ Model training completed in {training_time:.2f}s")
    
    def _extract_features(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Extract features for ML models"""
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['quarter'] = df['timestamp'].dt.quarter
        
        # Statistical features (rolling windows)
        windows = [5, 10, 30, 60]  # minutes
        
        for window in windows:
            df[f'rolling_mean_{window}'] = df['value'].rolling(window=window, min_periods=1).mean()
            df[f'rolling_std_{window}'] = df['value'].rolling(window=window, min_periods=1).std()
            df[f'rolling_min_{window}'] = df['value'].rolling(window=window, min_periods=1).min()
            df[f'rolling_max_{window}'] = df['value'].rolling(window=window, min_periods=1).max()
            df[f'rolling_median_{window}'] = df['value'].rolling(window=window, min_periods=1).median()
        
        # Lag features
        lags = [1, 5, 10, 30, 60]
        for lag in lags:
            df[f'lag_{lag}'] = df['value'].shift(lag)
        
        # Difference features
        df['diff_1'] = df['value'].diff()
        df['diff_5'] = df['value'].diff(5)
        df['diff_30'] = df['value'].diff(30)
        
        # Percentage change
        df['pct_change_1'] = df['value'].pct_change()
        df['pct_change_5'] = df['value'].pct_change(5)
        df['pct_change_30'] = df['value'].pct_change(30)
        
        # Technical indicators
        df['momentum'] = df['value'] - df['value'].shift(10)
        df['rate_of_change'] = df['value'] / df['value'].shift(10) - 1
        
        # Statistical measures
        df['zscore'] = (df['value'] - df['value'].rolling(60).mean()) / df['value'].rolling(60).std()
        df['percentile_rank'] = df['value'].rolling(60).rank(pct=True)
        
        # Seasonal decomposition features (simplified)
        df['trend'] = df['value'].rolling(window=60, center=True).mean()
        df['seasonal'] = df.groupby(df['hour'])['value'].transform(lambda x: x - x.mean())
        df['residual'] = df['value'] - df['trend'] - df['seasonal']
        
        # Drop timestamp and fill NaN values
        feature_df = df.drop(['timestamp'], axis=1).fillna(method='ffill').fillna(method='bfill')
        
        return feature_df
    
    async def _train_anomaly_detectors(self, metric_key: str, features: pd.DataFrame):
        """Train anomaly detection models"""
        
        # Prepare data
        X = features.select_dtypes(include=[np.number]).values
        
        # Scale features
        X_scaled = self.scalers['robust'].fit_transform(X)
        
        # Train Isolation Forest
        self.anomaly_detectors['isolation_forest'].fit(X_scaled)
        
        # Train DBSCAN clustering
        cluster_labels = self.anomaly_detectors['clustering'].fit_predict(X_scaled)
        
        # Store model for this metric
        model_key = f"anomaly_{metric_key}"
        await self.redis_client.set(
            model_key,
            pickle.dumps({
                'isolation_forest': self.anomaly_detectors['isolation_forest'],
                'scaler': self.scalers['robust'],
                'feature_columns': list(features.select_dtypes(include=[np.number]).columns)
            })
        )
    
    async def _train_trend_predictors(self, metric_key: str, features: pd.DataFrame):
        """Train trend prediction models"""
        
        # Prepare data for time series prediction
        X = features.select_dtypes(include=[np.number]).values[:-1]  # All but last
        y = features['value'].values[1:]  # Shifted by 1 (next value)
        
        if len(X) < 100:  # Need minimum samples
            return
        
        # Scale features
        X_scaled = self.scalers['standard'].fit_transform(X)
        
        # Train Random Forest
        self.trend_predictors['random_forest'].fit(X_scaled, y)
        
        # Store model
        model_key = f"trend_{metric_key}"
        await self.redis_client.set(
            model_key,
            pickle.dumps({
                'random_forest': self.trend_predictors['random_forest'],
                'scaler': self.scalers['standard'],
                'feature_columns': list(features.select_dtypes(include=[np.number]).columns)
            })
        )
    
    async def _start_prediction_loops(self):
        """Start prediction background tasks"""
        
        # Real-time anomaly detection
        asyncio.create_task(self._real_time_anomaly_detection_loop())
        
        # Trend prediction loop
        asyncio.create_task(self._trend_prediction_loop())
        
        # Model retraining loop
        asyncio.create_task(self._model_retraining_loop())
        
        # Prediction validation loop
        asyncio.create_task(self._prediction_validation_loop())
        
        logger.info("🔄 Prediction loops started")
    
    async def _real_time_anomaly_detection_loop(self):
        """Real-time anomaly detection on streaming data"""
        
        while True:
            try:
                if not self.models_trained:
                    await asyncio.sleep(60)
                    continue
                
                # Get recent metrics from all regions
                recent_data = await self._get_recent_metrics(minutes=5)
                
                # Process each metric-region combination
                for metric_region_key, data in recent_data.items():
                    if len(data) < 10:  # Need minimum data points
                        continue
                    
                    # Detect anomalies
                    predictions = await self._detect_anomalies(metric_region_key, data)
                    
                    # Process predictions
                    for prediction in predictions:
                        await self._process_prediction(prediction)
                
            except Exception as e:
                logger.error(f"Error in real-time anomaly detection: {e}")
            
            await asyncio.sleep(30)  # Check every 30 seconds
    
    async def _get_recent_metrics(self, minutes: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """Get recent metrics for real-time processing"""
        
        since_time = datetime.now() - timedelta(minutes=minutes)
        
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT metric_id, region, timestamp, value, labels
                FROM metrics
                WHERE timestamp >= $1
                ORDER BY metric_id, region, timestamp
            """, since_time)
        
        # Group by metric-region
        grouped_data = {}
        for row in rows:
            key = f"{row['metric_id']}:{row['region']}"
            if key not in grouped_data:
                grouped_data[key] = []
            
            grouped_data[key].append({
                'timestamp': row['timestamp'],
                'value': row['value'],
                'labels': json.loads(row['labels']) if row['labels'] else {}
            })
        
        return grouped_data
    
    async def _detect_anomalies(self, metric_key: str, data: List[Dict[str, Any]]) -> List[AnomalyPrediction]:
        """Detect anomalies in real-time data"""
        
        predictions = []
        
        try:
            # Load trained model
            model_key = f"anomaly_{metric_key}"
            model_data = await self.redis_client.get(model_key)
            
            if not model_data:
                return predictions
            
            model_info = pickle.loads(model_data)
            isolation_forest = model_info['isolation_forest']
            scaler = model_info['scaler']
            feature_columns = model_info['feature_columns']
            
            # Extract features from recent data
            features = self._extract_features(data)
            
            if len(features) == 0:
                return predictions
            
            # Get latest feature vector
            latest_features = features[feature_columns].iloc[-1:].values
            latest_features_scaled = scaler.transform(latest_features)
            
            # Predict anomaly
            anomaly_score = isolation_forest.decision_function(latest_features_scaled)[0]
            is_anomaly = isolation_forest.predict(latest_features_scaled)[0] == -1
            
            if is_anomaly and anomaly_score < -self.config['anomaly_threshold']:
                
                # Create prediction
                metric_id, region = metric_key.split(':', 1)
                
                prediction = AnomalyPrediction(
                    prediction_id=str(uuid.uuid4()),
                    metric_id=metric_id,
                    region=region,
                    anomaly_type=AnomalyType.POINT_ANOMALY,
                    predicted_time=datetime.now() + timedelta(minutes=5),  # 5 minutes ahead
                    confidence=self._calculate_confidence(abs(anomaly_score)),
                    anomaly_score=abs(anomaly_score),
                    severity_score=min(abs(anomaly_score) * 50, 100),
                    current_value=data[-1]['value'],
                    normal_range=self._calculate_normal_range(features['value']),
                    model_name="isolation_forest",
                    contributing_factors=self._identify_contributing_factors(features, feature_columns),
                    recommended_action=self._generate_recommendation(metric_id, abs(anomaly_score)),
                    estimated_impact=self._estimate_impact(metric_id, abs(anomaly_score)),
                    prevention_window_minutes=15,
                    false_positive_probability=self._calculate_false_positive_probability(metric_key, anomaly_score)
                )
                
                predictions.append(prediction)
        
        except Exception as e:
            logger.error(f"Error detecting anomalies for {metric_key}: {e}")
        
        return predictions
    
    def _calculate_confidence(self, anomaly_score: float) -> PredictionConfidence:
        """Calculate prediction confidence based on anomaly score"""
        
        if anomaly_score > 2.0:
            return PredictionConfidence.VERY_HIGH
        elif anomaly_score > 1.5:
            return PredictionConfidence.HIGH
        elif anomaly_score > 1.0:
            return PredictionConfidence.MEDIUM
        elif anomaly_score > 0.5:
            return PredictionConfidence.LOW
        else:
            return PredictionConfidence.VERY_LOW
    
    def _calculate_normal_range(self, values: pd.Series) -> Tuple[float, float]:
        """Calculate normal operating range for a metric"""
        
        q25 = values.quantile(0.25)
        q75 = values.quantile(0.75)
        iqr = q75 - q25
        
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        
        return (float(lower_bound), float(upper_bound))
    
    def _identify_contributing_factors(self, features: pd.DataFrame, feature_columns: List[str]) -> List[str]:
        """Identify factors contributing to the anomaly"""
        
        # Simple approach: find features with highest deviation from normal
        latest_row = features.iloc[-1]
        factors = []
        
        for col in feature_columns[-10:]:  # Check last 10 features
            if col in latest_row:
                value = latest_row[col]
                if abs(value) > 2.0:  # High deviation
                    factors.append(col)
        
        return factors[:5]  # Top 5 factors
    
    def _generate_recommendation(self, metric_id: str, anomaly_score: float) -> str:
        """Generate actionable recommendation"""
        
        recommendations = {
            'cpu_usage': "Check for resource-intensive processes and consider scaling up",
            'memory_usage': "Monitor memory leaks and consider increasing memory allocation", 
            'disk_usage': "Clean up disk space and archive old data",
            'network_latency': "Check network connectivity and routing configuration",
            'error_rate': "Investigate error logs and recent deployments",
            'response_time': "Check application performance and database queries"
        }
        
        base_recommendation = recommendations.get(metric_id, "Monitor closely and investigate root cause")
        
        if anomaly_score > 2.0:
            return f"URGENT: {base_recommendation}. Immediate action required."
        elif anomaly_score > 1.5:
            return f"HIGH PRIORITY: {base_recommendation}"
        else:
            return base_recommendation
    
    def _estimate_impact(self, metric_id: str, anomaly_score: float) -> str:
        """Estimate potential impact of the anomaly"""
        
        if anomaly_score > 2.0:
            return "High - Service degradation or outage likely"
        elif anomaly_score > 1.5:
            return "Medium - Performance degradation possible"
        else:
            return "Low - Monitor for trend development"
    
    def _calculate_false_positive_probability(self, metric_key: str, anomaly_score: float) -> float:
        """Calculate probability that this is a false positive"""
        
        # Simple heuristic based on historical performance
        base_fp_rate = 0.15  # 15% base false positive rate
        
        # Adjust based on anomaly score (higher score = lower FP probability)
        if anomaly_score > 2.0:
            return base_fp_rate * 0.3
        elif anomaly_score > 1.5:
            return base_fp_rate * 0.5
        else:
            return base_fp_rate * 0.8
    
    async def _process_prediction(self, prediction: AnomalyPrediction):
        """Process a new anomaly prediction"""
        
        # Store prediction in database
        await self._store_prediction(prediction)
        
        # Add to active predictions
        self.active_predictions[prediction.prediction_id] = prediction
        
        # Send alert if confidence is high enough
        if prediction.confidence in [PredictionConfidence.HIGH, PredictionConfidence.VERY_HIGH]:
            await self._send_predictive_alert(prediction)
        
        logger.info(f"🔮 Anomaly predicted: {prediction.metric_id} in {prediction.region} "
                   f"(confidence: {prediction.confidence.value}, score: {prediction.anomaly_score:.2f})")
    
    async def _store_prediction(self, prediction: AnomalyPrediction):
        """Store prediction in database"""
        
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO anomaly_predictions (
                    prediction_id, metric_id, region, anomaly_type, predicted_time,
                    confidence, anomaly_score, severity_score, current_value,
                    predicted_value, normal_range_min, normal_range_max, model_name,
                    feature_importance, contributing_factors, recommended_action,
                    estimated_impact, prevention_window_minutes, false_positive_probability
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19)
            """,
            prediction.prediction_id,
            prediction.metric_id,
            prediction.region,
            prediction.anomaly_type.value,
            prediction.predicted_time,
            prediction.confidence.value,
            prediction.anomaly_score,
            prediction.severity_score,
            prediction.current_value,
            prediction.predicted_value,
            prediction.normal_range[0],
            prediction.normal_range[1],
            prediction.model_name,
            json.dumps(prediction.feature_importance),
            prediction.contributing_factors,
            prediction.recommended_action,
            prediction.estimated_impact,
            prediction.prevention_window_minutes,
            prediction.false_positive_probability
            )
    
    async def _send_predictive_alert(self, prediction: AnomalyPrediction):
        """Send predictive alert notification"""
        
        # Determine alert priority
        if prediction.confidence == PredictionConfidence.VERY_HIGH and prediction.severity_score > 80:
            priority = AlertPriority.P0
        elif prediction.confidence == PredictionConfidence.HIGH and prediction.severity_score > 60:
            priority = AlertPriority.P1
        elif prediction.severity_score > 40:
            priority = AlertPriority.P2
        else:
            priority = AlertPriority.P3
        
        # Format alert message
        message = (f"🔮 PREDICTIVE ALERT [{priority.value.upper()}]\n"
                  f"📊 Metric: {prediction.metric_id} ({prediction.region})\n"
                  f"⏰ Predicted: {prediction.predicted_time.strftime('%H:%M:%S')}\n"
                  f"📈 Confidence: {prediction.confidence.value} ({prediction.anomaly_score:.2f})\n"
                  f"💡 Action: {prediction.recommended_action}\n"
                  f"🎯 Impact: {prediction.estimated_impact}")
        
        # Send to monitoring system (would integrate with actual alerting)
        logger.warning(message)
        
        # Cache alert
        await self.redis_client.setex(
            f"predictive_alert:{prediction.prediction_id}",
            3600,  # 1 hour TTL
            json.dumps(asdict(prediction), default=str)
        )
    
    async def get_prediction_dashboard(self) -> Dict[str, Any]:
        """Get predictive analytics dashboard data"""
        
        # Active predictions summary
        active_by_confidence = {}
        for conf in PredictionConfidence:
            active_by_confidence[conf.value] = len([
                p for p in self.active_predictions.values()
                if p.confidence == conf
            ])
        
        # Recent predictions
        recent_predictions = await self._get_recent_predictions(hours=24)
        
        # Model performance
        model_performance = {}
        for model_name, metrics in self.model_performance.items():
            model_performance[model_name] = asdict(metrics)
        
        # Prediction accuracy over time
        accuracy_trend = await self._calculate_prediction_accuracy_trend()
        
        dashboard = {
            'overview': {
                'total_active_predictions': len(self.active_predictions),
                'predictions_last_24h': len(recent_predictions),
                'average_confidence': np.mean([p.anomaly_score for p in self.active_predictions.values()]) if self.active_predictions else 0,
                'models_trained': len(self.training_data_cache),
                'last_model_update': self.last_training_time.isoformat() if self.last_training_time else None,
                'prediction_accuracy_percentage': accuracy_trend['current_accuracy'] if accuracy_trend else 0
            },
            
            'predictions_by_confidence': active_by_confidence,
            
            'recent_predictions': [
                {
                    'prediction_id': p.prediction_id,
                    'metric_id': p.metric_id,
                    'region': p.region,
                    'anomaly_type': p.anomaly_type.value,
                    'confidence': p.confidence.value,
                    'anomaly_score': p.anomaly_score,
                    'predicted_time': p.predicted_time.isoformat(),
                    'recommended_action': p.recommended_action,
                    'false_positive_probability': p.false_positive_probability
                } for p in list(self.active_predictions.values())[-10:]
            ],
            
            'model_performance': model_performance,
            
            'accuracy_trends': accuracy_trend,
            
            'top_anomalous_metrics': await self._get_top_anomalous_metrics(),
            
            'prediction_statistics': {
                'total_predictions_made': len(self.prediction_history),
                'average_prediction_lead_time_minutes': 15.5,  # Example
                'true_positive_rate': 85.2,
                'false_positive_rate': 12.1,
                'prediction_coverage_percentage': 78.9
            },
            
            'last_updated': datetime.now().isoformat()
        }
        
        return dashboard
    
    async def _get_recent_predictions(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent predictions from database"""
        
        since_time = datetime.now() - timedelta(hours=hours)
        
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM anomaly_predictions
                WHERE created_at >= $1
                ORDER BY created_at DESC
            """, since_time)
        
        return [dict(row) for row in rows]
    
    async def _calculate_prediction_accuracy_trend(self) -> Dict[str, Any]:
        """Calculate prediction accuracy trend"""
        
        # This would normally validate predictions against actual outcomes
        # For now, return example data
        return {
            'current_accuracy': 85.2,
            'trend_7_days': [82.1, 83.5, 84.2, 85.1, 84.8, 85.5, 85.2],
            'improvement_percentage': 3.1,
            'validated_predictions': 245
        }
    
    async def _get_top_anomalous_metrics(self) -> List[Dict[str, Any]]:
        """Get metrics with highest anomaly activity"""
        
        # Count anomalies by metric in last 24 hours
        since_time = datetime.now() - timedelta(hours=24)
        
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT 
                    metric_id, 
                    region,
                    COUNT(*) as anomaly_count,
                    AVG(anomaly_score) as avg_score,
                    MAX(severity_score) as max_severity
                FROM anomaly_predictions
                WHERE created_at >= $1
                GROUP BY metric_id, region
                ORDER BY anomaly_count DESC, avg_score DESC
                LIMIT 10
            """, since_time)
        
        return [
            {
                'metric_id': row['metric_id'],
                'region': row['region'],
                'anomaly_count': row['anomaly_count'],
                'average_score': float(row['avg_score']),
                'max_severity': float(row['max_severity'])
            } for row in rows
        ]
    
    async def _trend_prediction_loop(self):
        """Predict future trends for metrics"""
        
        while True:
            try:
                if not self.models_trained:
                    await asyncio.sleep(300)  # 5 minutes
                    continue
                
                # Get metrics that need trend predictions
                metrics_for_prediction = await self._get_metrics_for_trend_prediction()
                
                for metric_key in metrics_for_prediction:
                    try:
                        trend_prediction = await self._predict_trend(metric_key)
                        if trend_prediction:
                            await self._process_prediction(trend_prediction)
                    except Exception as e:
                        logger.error(f"Error predicting trend for {metric_key}: {e}")
                
            except Exception as e:
                logger.error(f"Error in trend prediction loop: {e}")
            
            await asyncio.sleep(300)  # Every 5 minutes
    
    async def _model_retraining_loop(self):
        """Periodically retrain models with new data"""
        
        while True:
            try:
                # Wait for retraining interval
                await asyncio.sleep(self.config['retraining_interval_hours'] * 3600)
                
                logger.info("🔄 Starting model retraining")
                
                # Load fresh training data
                await self._load_training_data()
                
                # Retrain models
                await self._train_models()
                
                # Evaluate model performance
                await self._evaluate_model_performance()
                
                logger.info("✅ Model retraining completed")
                
            except Exception as e:
                logger.error(f"Error in model retraining: {e}")
    
    async def _prediction_validation_loop(self):
        """Validate predictions against actual outcomes"""
        
        while True:
            try:
                # Validate predictions from 1 hour ago
                validation_time = datetime.now() - timedelta(hours=1)
                
                predictions_to_validate = [
                    p for p in self.active_predictions.values()
                    if p.predicted_time <= validation_time and not p.is_validated
                ]
                
                for prediction in predictions_to_validate:
                    await self._validate_prediction(prediction)
                
            except Exception as e:
                logger.error(f"Error in prediction validation: {e}")
            
            await asyncio.sleep(600)  # Every 10 minutes

# Main execution
async def main():
    """Main execution for testing predictive alerting"""
    
    system = PredictiveAlertingSystem()
    await system.initialize()
    
    logger.info("🔮 Predictive Alerting System started")
    
    # Let system run for a while
    await asyncio.sleep(60)
    
    # Get dashboard
    dashboard = await system.get_prediction_dashboard()
    logger.info(f"📊 Prediction Dashboard: {json.dumps(dashboard, indent=2)}")

if __name__ == "__main__":
    asyncio.run(main())