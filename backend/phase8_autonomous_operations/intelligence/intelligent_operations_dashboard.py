"""
Intelligent Operations Dashboard for Phase 8 - Autonomous Operations Intelligence
Advanced operational intelligence with predictive insights, anomaly detection, and autonomous decision support.

This module provides:
- Real-time operational intelligence with predictive analytics
- Autonomous system health monitoring with ML-driven insights
- Intelligent alert correlation and root cause analysis
- Predictive capacity planning and resource optimization
- Advanced pattern recognition for operational anomalies
- Automated incident response and remediation suggestions
- Multi-dimensional operational dashboards with AI insights
"""

import asyncio
import logging
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import uuid
import asyncpg
import redis.asyncio as redis
from concurrent.futures import ThreadPoolExecutor
import scipy.stats as stats
import sklearn.cluster as cluster
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class OperationalSeverity(Enum):
    """Operational alert severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class PredictionHorizon(Enum):
    """Prediction time horizons"""
    NEXT_HOUR = "1h"
    NEXT_4_HOURS = "4h"
    NEXT_DAY = "24h"
    NEXT_WEEK = "7d"
    NEXT_MONTH = "30d"


class OperationalDimension(Enum):
    """Operational monitoring dimensions"""
    PERFORMANCE = "performance"
    AVAILABILITY = "availability"
    CAPACITY = "capacity"
    SECURITY = "security"
    COST = "cost"
    QUALITY = "quality"


@dataclass
class OperationalMetric:
    """Operational metric with metadata"""
    metric_id: str
    name: str
    value: float
    unit: str
    timestamp: datetime
    dimension: OperationalDimension
    severity: Optional[OperationalSeverity] = None
    threshold_breached: bool = False
    predicted_value: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    anomaly_score: float = 0.0
    tags: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictiveInsight:
    """Predictive insight with AI-generated recommendations"""
    insight_id: str
    insight_type: str
    title: str
    description: str
    prediction_horizon: PredictionHorizon
    confidence: float
    severity: OperationalSeverity
    affected_systems: List[str]
    predicted_impact: Dict[str, Any]
    recommendations: List[str]
    supporting_data: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None


@dataclass
class OperationalAlert:
    """Intelligent operational alert with correlation"""
    alert_id: str
    alert_type: str
    title: str
    description: str
    severity: OperationalSeverity
    source_systems: List[str]
    affected_metrics: List[str]
    correlation_id: Optional[str] = None
    root_cause_analysis: Optional[Dict[str, Any]] = None
    suggested_actions: List[str] = field(default_factory=list)
    escalation_path: List[str] = field(default_factory=list)
    auto_resolution_attempts: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "active"
    acknowledged_by: Optional[str] = None
    resolved_by: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None


@dataclass
class SystemHealthScore:
    """Comprehensive system health assessment"""
    system_id: str
    overall_score: float  # 0-100
    dimension_scores: Dict[OperationalDimension, float]
    trend_analysis: Dict[str, str]  # improving/degrading/stable
    critical_issues: List[str]
    recommendations: List[str]
    next_maintenance_window: Optional[datetime] = None
    risk_factors: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CapacityForecast:
    """Intelligent capacity planning forecast"""
    resource_type: str
    current_utilization: float
    predicted_utilization: Dict[PredictionHorizon, float]
    capacity_exhaustion_date: Optional[datetime]
    scaling_recommendations: List[Dict[str, Any]]
    cost_projections: Dict[str, float]
    confidence_levels: Dict[PredictionHorizon, float]
    assumptions: List[str] = field(default_factory=list)


class IntelligentAnomalyDetector:
    """Advanced ML-based anomaly detection for operations"""
    
    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.isolation_forest = IsolationForest(contamination=contamination, random_state=42)
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.is_trained = False
        self.training_data_size = 0
        self.last_training_time: Optional[datetime] = None
        
    def train(self, training_data: pd.DataFrame) -> None:
        """Train the anomaly detection model"""
        try:
            if training_data.empty:
                raise ValueError("Training data is empty")
            
            # Prepare features
            numeric_columns = training_data.select_dtypes(include=[np.number]).columns
            training_features = training_data[numeric_columns].fillna(0)
            
            # Store feature names
            self.feature_names = list(training_features.columns)
            
            # Scale features
            scaled_features = self.scaler.fit_transform(training_features)
            
            # Train isolation forest
            self.isolation_forest.fit(scaled_features)
            
            self.is_trained = True
            self.training_data_size = len(training_data)
            self.last_training_time = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Error training anomaly detector: {e}")
            raise
    
    def detect_anomalies(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalies in operational data"""
        try:
            if not self.is_trained:
                raise ValueError("Model is not trained")
            
            # Prepare features
            numeric_columns = [col for col in self.feature_names if col in data.columns]
            features = data[numeric_columns].fillna(0)
            
            # Scale features
            scaled_features = self.scaler.transform(features)
            
            # Detect anomalies
            anomaly_scores = self.isolation_forest.decision_function(scaled_features)
            anomaly_predictions = self.isolation_forest.predict(scaled_features)
            
            # Convert to positive anomaly scores
            normalized_scores = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())
            
            return {
                'anomaly_scores': normalized_scores,
                'is_anomaly': anomaly_predictions == -1,
                'anomaly_indices': np.where(anomaly_predictions == -1)[0].tolist(),
                'severity_levels': self._classify_anomaly_severity(normalized_scores),
                'feature_contributions': self._calculate_feature_contributions(scaled_features, anomaly_scores)
            }
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return {
                'anomaly_scores': np.array([]),
                'is_anomaly': np.array([], dtype=bool),
                'anomaly_indices': [],
                'severity_levels': [],
                'feature_contributions': {}
            }
    
    def _classify_anomaly_severity(self, scores: np.ndarray) -> List[OperationalSeverity]:
        """Classify anomaly severity based on scores"""
        severity_levels = []
        
        for score in scores:
            if score > 0.9:
                severity_levels.append(OperationalSeverity.CRITICAL)
            elif score > 0.7:
                severity_levels.append(OperationalSeverity.HIGH)
            elif score > 0.5:
                severity_levels.append(OperationalSeverity.MEDIUM)
            else:
                severity_levels.append(OperationalSeverity.LOW)
        
        return severity_levels
    
    def _calculate_feature_contributions(self, scaled_features: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
        """Calculate feature contributions to anomaly scores"""
        try:
            contributions = {}
            
            for i, feature_name in enumerate(self.feature_names):
                # Simple correlation between feature values and anomaly scores
                feature_values = scaled_features[:, i]
                correlation = abs(np.corrcoef(feature_values, scores)[0, 1])
                contributions[feature_name] = correlation if not np.isnan(correlation) else 0.0
            
            return contributions
            
        except Exception as e:
            logger.error(f"Error calculating feature contributions: {e}")
            return {}


class PredictiveAnalyzer:
    """Advanced predictive analytics for operational intelligence"""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        
    def build_prediction_models(self, historical_data: Dict[str, pd.DataFrame]) -> None:
        """Build predictive models for different operational metrics"""
        try:
            for metric_name, data in historical_data.items():
                if data.empty or len(data) < 50:  # Need minimum data for training
                    continue
                
                # Prepare time series features
                features = self._create_time_features(data)
                target = data['value'].values
                
                if len(features) != len(target):
                    continue
                
                # Create and train model
                scaler = StandardScaler()
                scaled_features = scaler.fit_transform(features)
                
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(scaled_features, target)
                
                self.models[metric_name] = model
                self.scalers[metric_name] = scaler
                
        except Exception as e:
            logger.error(f"Error building prediction models: {e}")
    
    def predict_metric_values(
        self, 
        metric_name: str, 
        current_data: pd.DataFrame, 
        prediction_horizon: PredictionHorizon
    ) -> Dict[str, Any]:
        """Predict future metric values"""
        try:
            if metric_name not in self.models:
                return {'error': f'No model available for {metric_name}'}
            
            model = self.models[metric_name]
            scaler = self.scalers[metric_name]
            
            # Create features for prediction
            features = self._create_time_features(current_data)
            scaled_features = scaler.transform(features[-1:])  # Use latest data point
            
            # Make prediction
            prediction = model.predict(scaled_features)[0]
            
            # Calculate confidence interval (simplified approach)
            prediction_std = prediction * 0.1  # 10% uncertainty
            confidence_interval = (prediction - 2 * prediction_std, prediction + 2 * prediction_std)
            
            # Calculate feature importance
            feature_importance = dict(zip(
                [f'feature_{i}' for i in range(len(model.feature_importances_))],
                model.feature_importances_
            ))
            
            return {
                'prediction': prediction,
                'confidence_interval': confidence_interval,
                'horizon': prediction_horizon.value,
                'feature_importance': feature_importance,
                'model_score': getattr(model, 'score_', 0.0)
            }
            
        except Exception as e:
            logger.error(f"Error predicting metric values: {e}")
            return {'error': str(e)}
    
    def _create_time_features(self, data: pd.DataFrame) -> np.ndarray:
        """Create time-based features for prediction"""
        try:
            if 'timestamp' not in data.columns:
                # Create sequential features
                return np.arange(len(data)).reshape(-1, 1)
            
            # Convert timestamp to features
            timestamps = pd.to_datetime(data['timestamp'])
            features = []
            
            for ts in timestamps:
                feature_vector = [
                    ts.hour,
                    ts.day_of_week,
                    ts.day,
                    ts.month,
                    ts.weekday(),
                    int(ts.timestamp())
                ]
                features.append(feature_vector)
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Error creating time features: {e}")
            return np.array([]).reshape(0, 1)


class RootCauseAnalyzer:
    """Intelligent root cause analysis for operational issues"""
    
    def __init__(self):
        self.correlation_threshold = 0.7
        self.time_window_minutes = 15
        
    def analyze_incident(
        self, 
        incident_data: Dict[str, Any], 
        historical_metrics: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Perform root cause analysis for an operational incident"""
        try:
            incident_time = incident_data.get('timestamp', datetime.utcnow())
            affected_systems = incident_data.get('affected_systems', [])
            
            # Analyze temporal correlations
            temporal_analysis = self._analyze_temporal_correlations(
                incident_time, historical_metrics
            )
            
            # Analyze system correlations
            system_analysis = self._analyze_system_correlations(
                affected_systems, historical_metrics
            )
            
            # Generate root cause hypotheses
            hypotheses = self._generate_root_cause_hypotheses(
                temporal_analysis, system_analysis, incident_data
            )
            
            # Rank hypotheses by likelihood
            ranked_hypotheses = self._rank_hypotheses(hypotheses)
            
            return {
                'incident_id': incident_data.get('id', str(uuid.uuid4())),
                'analysis_timestamp': datetime.utcnow().isoformat(),
                'temporal_analysis': temporal_analysis,
                'system_analysis': system_analysis,
                'root_cause_hypotheses': ranked_hypotheses,
                'confidence_score': self._calculate_confidence_score(ranked_hypotheses),
                'recommended_actions': self._generate_recommended_actions(ranked_hypotheses)
            }
            
        except Exception as e:
            logger.error(f"Error in root cause analysis: {e}")
            return {'error': str(e)}
    
    def _analyze_temporal_correlations(
        self, 
        incident_time: datetime, 
        metrics: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Analyze temporal correlations around incident time"""
        try:
            correlations = {}
            
            # Define time window
            start_time = incident_time - timedelta(minutes=self.time_window_minutes)
            end_time = incident_time + timedelta(minutes=self.time_window_minutes)
            
            for metric_name, data in metrics.items():
                if 'timestamp' in data.columns and 'value' in data.columns:
                    # Filter data to time window
                    mask = (pd.to_datetime(data['timestamp']) >= start_time) & \
                           (pd.to_datetime(data['timestamp']) <= end_time)
                    window_data = data[mask]
                    
                    if len(window_data) > 2:
                        # Calculate trend and volatility
                        values = window_data['value'].values
                        trend = (values[-1] - values[0]) / len(values) if len(values) > 1 else 0
                        volatility = np.std(values) if len(values) > 1 else 0
                        
                        correlations[metric_name] = {
                            'trend': trend,
                            'volatility': volatility,
                            'max_value': float(np.max(values)),
                            'min_value': float(np.min(values)),
                            'mean_value': float(np.mean(values)),
                            'data_points': len(window_data)
                        }
            
            return correlations
            
        except Exception as e:
            logger.error(f"Error analyzing temporal correlations: {e}")
            return {}
    
    def _analyze_system_correlations(
        self, 
        affected_systems: List[str], 
        metrics: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Analyze correlations between affected systems"""
        try:
            system_correlations = {}
            
            # Group metrics by system
            system_metrics = defaultdict(list)
            for metric_name in metrics.keys():
                # Extract system name from metric name (simplified approach)
                system_name = metric_name.split('_')[0] if '_' in metric_name else 'unknown'
                system_metrics[system_name].append(metric_name)
            
            # Analyze correlations for affected systems
            for system in affected_systems:
                if system in system_metrics:
                    system_data = {}
                    
                    for metric in system_metrics[system]:
                        if metric in metrics:
                            recent_data = metrics[metric].tail(100)  # Last 100 data points
                            if 'value' in recent_data.columns:
                                system_data[metric] = recent_data['value'].values
                    
                    # Calculate cross-correlations
                    correlations = self._calculate_cross_correlations(system_data)
                    system_correlations[system] = correlations
            
            return system_correlations
            
        except Exception as e:
            logger.error(f"Error analyzing system correlations: {e}")
            return {}
    
    def _calculate_cross_correlations(self, data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate cross-correlations between metrics"""
        try:
            correlations = {}
            metric_names = list(data.keys())
            
            for i, metric1 in enumerate(metric_names):
                for j, metric2 in enumerate(metric_names[i+1:], i+1):
                    if len(data[metric1]) > 1 and len(data[metric2]) > 1:
                        # Ensure same length
                        min_len = min(len(data[metric1]), len(data[metric2]))
                        corr = np.corrcoef(
                            data[metric1][-min_len:], 
                            data[metric2][-min_len:]
                        )[0, 1]
                        
                        if not np.isnan(corr):
                            correlations[f"{metric1}_vs_{metric2}"] = float(corr)
            
            return correlations
            
        except Exception as e:
            logger.error(f"Error calculating cross-correlations: {e}")
            return {}
    
    def _generate_root_cause_hypotheses(
        self, 
        temporal_analysis: Dict[str, Any], 
        system_analysis: Dict[str, Any],
        incident_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate root cause hypotheses"""
        hypotheses = []
        
        # Hypothesis 1: High volatility metrics
        high_volatility_metrics = [
            metric for metric, data in temporal_analysis.items()
            if data.get('volatility', 0) > np.mean([d.get('volatility', 0) for d in temporal_analysis.values()]) * 2
        ]
        
        if high_volatility_metrics:
            hypotheses.append({
                'type': 'high_volatility',
                'description': f'High volatility detected in metrics: {", ".join(high_volatility_metrics)}',
                'evidence': high_volatility_metrics,
                'likelihood': 0.8,
                'category': 'performance_degradation'
            })
        
        # Hypothesis 2: Strong correlations
        for system, correlations in system_analysis.items():
            strong_correlations = [
                corr_pair for corr_pair, corr_value in correlations.items()
                if abs(corr_value) > self.correlation_threshold
            ]
            
            if strong_correlations:
                hypotheses.append({
                    'type': 'correlation_cascade',
                    'description': f'Strong correlations in {system}: {", ".join(strong_correlations)}',
                    'evidence': strong_correlations,
                    'likelihood': 0.7,
                    'category': 'cascade_failure'
                })
        
        # Hypothesis 3: Trending metrics
        trending_up = [
            metric for metric, data in temporal_analysis.items()
            if data.get('trend', 0) > 0.1
        ]
        
        if trending_up:
            hypotheses.append({
                'type': 'resource_exhaustion',
                'description': f'Upward trending metrics suggest resource exhaustion: {", ".join(trending_up)}',
                'evidence': trending_up,
                'likelihood': 0.6,
                'category': 'capacity_issue'
            })
        
        return hypotheses
    
    def _rank_hypotheses(self, hypotheses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank hypotheses by likelihood"""
        return sorted(hypotheses, key=lambda x: x.get('likelihood', 0), reverse=True)
    
    def _calculate_confidence_score(self, hypotheses: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence score for the analysis"""
        if not hypotheses:
            return 0.0
        
        # Weight by position and likelihood
        total_confidence = sum(
            h.get('likelihood', 0) * (1.0 / (i + 1))
            for i, h in enumerate(hypotheses)
        )
        
        return min(total_confidence / len(hypotheses), 1.0)
    
    def _generate_recommended_actions(self, hypotheses: List[Dict[str, Any]]) -> List[str]:
        """Generate recommended actions based on hypotheses"""
        actions = []
        
        for hypothesis in hypotheses[:3]:  # Top 3 hypotheses
            category = hypothesis.get('category', 'unknown')
            
            if category == 'performance_degradation':
                actions.extend([
                    'Check system resource utilization',
                    'Review recent configuration changes',
                    'Analyze application performance metrics'
                ])
            elif category == 'cascade_failure':
                actions.extend([
                    'Investigate upstream dependencies',
                    'Check for network connectivity issues',
                    'Review service health checks'
                ])
            elif category == 'capacity_issue':
                actions.extend([
                    'Scale up resources if possible',
                    'Implement load balancing',
                    'Review capacity planning'
                ])
        
        return list(set(actions))  # Remove duplicates


class IntelligentOperationsDashboard:
    """
    Advanced operational intelligence dashboard with predictive insights
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
        
        # AI/ML components
        self.anomaly_detector = IntelligentAnomalyDetector()
        self.predictive_analyzer = PredictiveAnalyzer()
        self.root_cause_analyzer = RootCauseAnalyzer()
        
        # Data storage
        self.operational_metrics: deque = deque(maxlen=10000)
        self.predictive_insights: Dict[str, PredictiveInsight] = {}
        self.active_alerts: Dict[str, OperationalAlert] = {}
        self.system_health_scores: Dict[str, SystemHealthScore] = {}
        self.capacity_forecasts: Dict[str, CapacityForecast] = {}
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._prediction_task: Optional[asyncio.Task] = None
        self._analysis_task: Optional[asyncio.Task] = None
        
        # Thread pool for ML operations
        self.ml_executor = ThreadPoolExecutor(max_workers=4)
        
        # Configuration
        self.update_interval_seconds = 60
        self.prediction_interval_seconds = 300  # 5 minutes
        self.analysis_interval_seconds = 180    # 3 minutes
        
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self) -> None:
        """Initialize the intelligent operations dashboard"""
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
            
            # Initialize ML models with historical data
            await self._initialize_ml_models()
            
            # Start background monitoring tasks
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            self._prediction_task = asyncio.create_task(self._prediction_loop())
            self._analysis_task = asyncio.create_task(self._analysis_loop())
            
            self.logger.info("Intelligent Operations Dashboard initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize dashboard: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the dashboard"""
        try:
            # Cancel background tasks
            tasks = [self._monitoring_task, self._prediction_task, self._analysis_task]
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
            
            self.logger.info("Intelligent Operations Dashboard shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    async def _create_database_tables(self) -> None:
        """Create database tables for operational intelligence"""
        try:
            async with self.db_pool.acquire() as conn:
                # Operational metrics table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS operational_metrics (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        metric_id VARCHAR(100) NOT NULL,
                        name VARCHAR(200) NOT NULL,
                        value DECIMAL(15,6) NOT NULL,
                        unit VARCHAR(20),
                        timestamp TIMESTAMPTZ NOT NULL,
                        dimension VARCHAR(50),
                        severity VARCHAR(20),
                        threshold_breached BOOLEAN DEFAULT FALSE,
                        predicted_value DECIMAL(15,6),
                        confidence_interval_lower DECIMAL(15,6),
                        confidence_interval_upper DECIMAL(15,6),
                        anomaly_score DECIMAL(5,4) DEFAULT 0,
                        tags JSONB,
                        metadata JSONB,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_operational_metrics_timestamp
                        ON operational_metrics(timestamp);
                    CREATE INDEX IF NOT EXISTS idx_operational_metrics_metric_id
                        ON operational_metrics(metric_id);
                    CREATE INDEX IF NOT EXISTS idx_operational_metrics_dimension
                        ON operational_metrics(dimension);
                """)
                
                # Predictive insights table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS predictive_insights (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        insight_id VARCHAR(100) UNIQUE NOT NULL,
                        insight_type VARCHAR(50) NOT NULL,
                        title VARCHAR(200) NOT NULL,
                        description TEXT,
                        prediction_horizon VARCHAR(10),
                        confidence DECIMAL(5,4),
                        severity VARCHAR(20),
                        affected_systems TEXT[],
                        predicted_impact JSONB,
                        recommendations TEXT[],
                        supporting_data JSONB,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        expires_at TIMESTAMPTZ,
                        status VARCHAR(20) DEFAULT 'active'
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_predictive_insights_created_at
                        ON predictive_insights(created_at);
                    CREATE INDEX IF NOT EXISTS idx_predictive_insights_severity
                        ON predictive_insights(severity);
                """)
                
                # Operational alerts table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS operational_alerts (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        alert_id VARCHAR(100) UNIQUE NOT NULL,
                        alert_type VARCHAR(50) NOT NULL,
                        title VARCHAR(200) NOT NULL,
                        description TEXT,
                        severity VARCHAR(20) NOT NULL,
                        source_systems TEXT[],
                        affected_metrics TEXT[],
                        correlation_id VARCHAR(100),
                        root_cause_analysis JSONB,
                        suggested_actions TEXT[],
                        escalation_path TEXT[],
                        auto_resolution_attempts JSONB,
                        status VARCHAR(20) DEFAULT 'active',
                        acknowledged_by VARCHAR(100),
                        resolved_by VARCHAR(100),
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        updated_at TIMESTAMPTZ DEFAULT NOW(),
                        resolved_at TIMESTAMPTZ
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_operational_alerts_created_at
                        ON operational_alerts(created_at);
                    CREATE INDEX IF NOT EXISTS idx_operational_alerts_status
                        ON operational_alerts(status);
                    CREATE INDEX IF NOT EXISTS idx_operational_alerts_severity
                        ON operational_alerts(severity);
                """)
                
                # System health scores table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS system_health_scores (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        system_id VARCHAR(100) NOT NULL,
                        overall_score DECIMAL(5,2),
                        dimension_scores JSONB,
                        trend_analysis JSONB,
                        critical_issues TEXT[],
                        recommendations TEXT[],
                        next_maintenance_window TIMESTAMPTZ,
                        risk_factors TEXT[],
                        timestamp TIMESTAMPTZ NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_system_health_scores_system_id
                        ON system_health_scores(system_id);
                    CREATE INDEX IF NOT EXISTS idx_system_health_scores_timestamp
                        ON system_health_scores(timestamp);
                """)
            
            self.logger.info("Database tables created successfully")
            
        except Exception as e:
            self.logger.error(f"Error creating database tables: {e}")
            raise
    
    async def _initialize_ml_models(self) -> None:
        """Initialize ML models with historical data"""
        try:
            # Load historical operational metrics
            historical_data = await self._load_historical_metrics()
            
            # Train anomaly detection model
            if historical_data:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    self.ml_executor,
                    self.anomaly_detector.train,
                    historical_data
                )
            
            # Build predictive models
            historical_metrics_by_type = await self._load_historical_metrics_by_type()
            if historical_metrics_by_type:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    self.ml_executor,
                    self.predictive_analyzer.build_prediction_models,
                    historical_metrics_by_type
                )
            
            self.logger.info("ML models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing ML models: {e}")
    
    async def _load_historical_metrics(self) -> pd.DataFrame:
        """Load historical operational metrics for training"""
        try:
            async with self.db_pool.acquire() as conn:
                query = """
                    SELECT metric_id, name, value, unit, timestamp, dimension, 
                           anomaly_score, tags, metadata
                    FROM operational_metrics 
                    WHERE timestamp >= NOW() - INTERVAL '30 days'
                    ORDER BY timestamp DESC
                    LIMIT 10000
                """
                
                rows = await conn.fetch(query)
                
                if rows:
                    return pd.DataFrame([dict(row) for row in rows])
                else:
                    return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Error loading historical metrics: {e}")
            return pd.DataFrame()
    
    async def _load_historical_metrics_by_type(self) -> Dict[str, pd.DataFrame]:
        """Load historical metrics grouped by metric type"""
        try:
            async with self.db_pool.acquire() as conn:
                query = """
                    SELECT metric_id, value, timestamp
                    FROM operational_metrics 
                    WHERE timestamp >= NOW() - INTERVAL '7 days'
                    ORDER BY metric_id, timestamp
                """
                
                rows = await conn.fetch(query)
                
                if rows:
                    df = pd.DataFrame([dict(row) for row in rows])
                    return {
                        metric_id: group_df 
                        for metric_id, group_df in df.groupby('metric_id')
                    }
                else:
                    return {}
                
        except Exception as e:
            self.logger.error(f"Error loading historical metrics by type: {e}")
            return {}
    
    # Background monitoring loops
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop for operational intelligence"""
        try:
            while True:
                await asyncio.sleep(self.update_interval_seconds)
                
                try:
                    # Collect current operational metrics
                    current_metrics = await self._collect_operational_metrics()
                    
                    # Detect anomalies
                    if current_metrics:
                        await self._detect_operational_anomalies(current_metrics)
                    
                    # Update system health scores
                    await self._update_system_health_scores()
                    
                    # Process alerts and correlations
                    await self._process_alert_correlations()
                    
                except Exception as e:
                    self.logger.error(f"Error in monitoring loop: {e}")
                
        except asyncio.CancelledError:
            self.logger.info("Monitoring loop cancelled")
            raise
        except Exception as e:
            self.logger.error(f"Error in monitoring loop: {e}")
    
    async def _prediction_loop(self) -> None:
        """Predictive analytics loop"""
        try:
            while True:
                await asyncio.sleep(self.prediction_interval_seconds)
                
                try:
                    # Generate predictive insights
                    await self._generate_predictive_insights()
                    
                    # Update capacity forecasts
                    await self._update_capacity_forecasts()
                    
                    # Cleanup expired insights
                    await self._cleanup_expired_insights()
                    
                except Exception as e:
                    self.logger.error(f"Error in prediction loop: {e}")
                
        except asyncio.CancelledError:
            self.logger.info("Prediction loop cancelled")
            raise
        except Exception as e:
            self.logger.error(f"Error in prediction loop: {e}")
    
    async def _analysis_loop(self) -> None:
        """Root cause analysis loop"""
        try:
            while True:
                await asyncio.sleep(self.analysis_interval_seconds)
                
                try:
                    # Perform root cause analysis for active alerts
                    await self._perform_root_cause_analysis()
                    
                    # Update correlation patterns
                    await self._update_correlation_patterns()
                    
                    # Generate intelligent recommendations
                    await self._generate_intelligent_recommendations()
                    
                except Exception as e:
                    self.logger.error(f"Error in analysis loop: {e}")
                
        except asyncio.CancelledError:
            self.logger.info("Analysis loop cancelled")
            raise
        except Exception as e:
            self.logger.error(f"Error in analysis loop: {e}")
    
    async def _collect_operational_metrics(self) -> List[OperationalMetric]:
        """Collect current operational metrics from various sources"""
        try:
            metrics = []
            
            # Collect from different operational dimensions
            performance_metrics = await self._collect_performance_metrics()
            availability_metrics = await self._collect_availability_metrics()
            capacity_metrics = await self._collect_capacity_metrics()
            security_metrics = await self._collect_security_metrics()
            cost_metrics = await self._collect_cost_metrics()
            quality_metrics = await self._collect_quality_metrics()
            
            metrics.extend(performance_metrics)
            metrics.extend(availability_metrics)
            metrics.extend(capacity_metrics)
            metrics.extend(security_metrics)
            metrics.extend(cost_metrics)
            metrics.extend(quality_metrics)
            
            # Store metrics
            self.operational_metrics.extend(metrics)
            
            # Persist to database
            await self._persist_operational_metrics(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error collecting operational metrics: {e}")
            return []
    
    async def _collect_performance_metrics(self) -> List[OperationalMetric]:
        """Collect performance-related operational metrics"""
        try:
            metrics = []
            current_time = datetime.utcnow()
            
            # Example performance metrics (would integrate with actual monitoring systems)
            metrics.extend([
                OperationalMetric(
                    metric_id="response_time_avg",
                    name="Average Response Time",
                    value=125.5,
                    unit="ms",
                    timestamp=current_time,
                    dimension=OperationalDimension.PERFORMANCE,
                    tags={"service": "api", "environment": "production"}
                ),
                OperationalMetric(
                    metric_id="throughput_rps",
                    name="Requests Per Second",
                    value=850.0,
                    unit="rps",
                    timestamp=current_time,
                    dimension=OperationalDimension.PERFORMANCE,
                    tags={"service": "api", "environment": "production"}
                ),
                OperationalMetric(
                    metric_id="error_rate_pct",
                    name="Error Rate",
                    value=0.5,
                    unit="percent",
                    timestamp=current_time,
                    dimension=OperationalDimension.PERFORMANCE,
                    tags={"service": "api", "environment": "production"}
                )
            ])
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error collecting performance metrics: {e}")
            return []
    
    async def _collect_availability_metrics(self) -> List[OperationalMetric]:
        """Collect availability-related operational metrics"""
        metrics = []
        current_time = datetime.utcnow()
        
        metrics.extend([
            OperationalMetric(
                metric_id="uptime_pct",
                name="System Uptime",
                value=99.95,
                unit="percent",
                timestamp=current_time,
                dimension=OperationalDimension.AVAILABILITY,
                tags={"system": "trading_engine"}
            ),
            OperationalMetric(
                metric_id="service_health_score",
                name="Service Health Score",
                value=98.5,
                unit="score",
                timestamp=current_time,
                dimension=OperationalDimension.AVAILABILITY,
                tags={"service": "all"}
            )
        ])
        
        return metrics
    
    async def _collect_capacity_metrics(self) -> List[OperationalMetric]:
        """Collect capacity-related operational metrics"""
        metrics = []
        current_time = datetime.utcnow()
        
        metrics.extend([
            OperationalMetric(
                metric_id="cpu_utilization",
                name="CPU Utilization",
                value=65.3,
                unit="percent",
                timestamp=current_time,
                dimension=OperationalDimension.CAPACITY,
                tags={"resource": "compute"}
            ),
            OperationalMetric(
                metric_id="memory_utilization",
                name="Memory Utilization",
                value=72.1,
                unit="percent",
                timestamp=current_time,
                dimension=OperationalDimension.CAPACITY,
                tags={"resource": "memory"}
            ),
            OperationalMetric(
                metric_id="disk_utilization",
                name="Disk Utilization",
                value=45.8,
                unit="percent",
                timestamp=current_time,
                dimension=OperationalDimension.CAPACITY,
                tags={"resource": "storage"}
            )
        ])
        
        return metrics
    
    async def _collect_security_metrics(self) -> List[OperationalMetric]:
        """Collect security-related operational metrics"""
        metrics = []
        current_time = datetime.utcnow()
        
        metrics.extend([
            OperationalMetric(
                metric_id="failed_auth_attempts",
                name="Failed Authentication Attempts",
                value=12.0,
                unit="count",
                timestamp=current_time,
                dimension=OperationalDimension.SECURITY,
                tags={"security_event": "authentication"}
            ),
            OperationalMetric(
                metric_id="security_incidents",
                name="Security Incidents",
                value=0.0,
                unit="count",
                timestamp=current_time,
                dimension=OperationalDimension.SECURITY,
                tags={"security_event": "incident"}
            )
        ])
        
        return metrics
    
    async def _collect_cost_metrics(self) -> List[OperationalMetric]:
        """Collect cost-related operational metrics"""
        metrics = []
        current_time = datetime.utcnow()
        
        metrics.extend([
            OperationalMetric(
                metric_id="infrastructure_cost",
                name="Infrastructure Cost",
                value=1250.0,
                unit="usd",
                timestamp=current_time,
                dimension=OperationalDimension.COST,
                tags={"cost_category": "infrastructure"}
            ),
            OperationalMetric(
                metric_id="cost_per_transaction",
                name="Cost Per Transaction",
                value=0.15,
                unit="usd",
                timestamp=current_time,
                dimension=OperationalDimension.COST,
                tags={"cost_category": "transaction"}
            )
        ])
        
        return metrics
    
    async def _collect_quality_metrics(self) -> List[OperationalMetric]:
        """Collect quality-related operational metrics"""
        metrics = []
        current_time = datetime.utcnow()
        
        metrics.extend([
            OperationalMetric(
                metric_id="data_quality_score",
                name="Data Quality Score",
                value=95.2,
                unit="score",
                timestamp=current_time,
                dimension=OperationalDimension.QUALITY,
                tags={"quality_aspect": "data"}
            ),
            OperationalMetric(
                metric_id="service_quality_score",
                name="Service Quality Score",
                value=97.8,
                unit="score",
                timestamp=current_time,
                dimension=OperationalDimension.QUALITY,
                tags={"quality_aspect": "service"}
            )
        ])
        
        return metrics
    
    async def _persist_operational_metrics(self, metrics: List[OperationalMetric]) -> None:
        """Persist operational metrics to database"""
        try:
            if not metrics:
                return
            
            async with self.db_pool.acquire() as conn:
                for metric in metrics:
                    await conn.execute("""
                        INSERT INTO operational_metrics (
                            metric_id, name, value, unit, timestamp, dimension,
                            severity, threshold_breached, predicted_value,
                            confidence_interval_lower, confidence_interval_upper,
                            anomaly_score, tags, metadata
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                    """,
                        metric.metric_id,
                        metric.name,
                        metric.value,
                        metric.unit,
                        metric.timestamp,
                        metric.dimension.value if metric.dimension else None,
                        metric.severity.value if metric.severity else None,
                        metric.threshold_breached,
                        metric.predicted_value,
                        metric.confidence_interval[0] if metric.confidence_interval else None,
                        metric.confidence_interval[1] if metric.confidence_interval else None,
                        metric.anomaly_score,
                        json.dumps(metric.tags),
                        json.dumps(metric.metadata)
                    )
            
        except Exception as e:
            self.logger.error(f"Error persisting operational metrics: {e}")
    
    # Public API methods
    
    async def get_dashboard_data(
        self, 
        time_range: Optional[Tuple[datetime, datetime]] = None,
        dimensions: Optional[List[OperationalDimension]] = None
    ) -> Dict[str, Any]:
        """Get comprehensive dashboard data with predictive insights"""
        try:
            dashboard_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'system_overview': await self._get_system_overview(),
                'operational_metrics': await self._get_operational_metrics_summary(time_range, dimensions),
                'predictive_insights': list(self.predictive_insights.values()),
                'active_alerts': list(self.active_alerts.values()),
                'system_health_scores': list(self.system_health_scores.values()),
                'capacity_forecasts': list(self.capacity_forecasts.values()),
                'anomaly_analysis': await self._get_anomaly_analysis(),
                'performance_trends': await self._get_performance_trends(),
                'recommendations': await self._get_intelligent_recommendations()
            }
            
            return dashboard_data
            
        except Exception as e:
            self.logger.error(f"Error getting dashboard data: {e}")
            return {'error': str(e), 'timestamp': datetime.utcnow().isoformat()}
    
    async def _get_system_overview(self) -> Dict[str, Any]:
        """Get high-level system overview"""
        try:
            recent_metrics = list(self.operational_metrics)[-100:]  # Last 100 metrics
            
            if not recent_metrics:
                return {
                    'status': 'unknown',
                    'overall_health_score': 0.0,
                    'active_alerts': 0,
                    'predictive_insights': 0
                }
            
            # Calculate overall health score
            health_scores = [metric.value for metric in recent_metrics 
                           if 'health' in metric.name.lower() or 'score' in metric.name.lower()]
            overall_health = np.mean(health_scores) if health_scores else 85.0
            
            # Determine system status
            if overall_health >= 95:
                status = 'excellent'
            elif overall_health >= 85:
                status = 'good'
            elif overall_health >= 70:
                status = 'warning'
            else:
                status = 'critical'
            
            return {
                'status': status,
                'overall_health_score': overall_health,
                'active_alerts': len([a for a in self.active_alerts.values() if a.status == 'active']),
                'critical_alerts': len([a for a in self.active_alerts.values() 
                                      if a.status == 'active' and a.severity == OperationalSeverity.CRITICAL]),
                'predictive_insights': len(self.predictive_insights),
                'systems_monitored': len(set(metric.tags.get('system', 'unknown') for metric in recent_metrics)),
                'anomalies_detected': len([m for m in recent_metrics if m.anomaly_score > 0.7]),
                'last_updated': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting system overview: {e}")
            return {'error': str(e)}
    
    async def create_operational_alert(
        self,
        alert_type: str,
        title: str,
        description: str,
        severity: OperationalSeverity,
        source_systems: List[str],
        affected_metrics: List[str] = None
    ) -> str:
        """Create a new operational alert with intelligent analysis"""
        try:
            alert_id = str(uuid.uuid4())
            
            # Perform initial root cause analysis
            incident_data = {
                'id': alert_id,
                'type': alert_type,
                'timestamp': datetime.utcnow(),
                'affected_systems': source_systems,
                'severity': severity.value
            }
            
            # Get recent historical metrics for analysis
            historical_metrics = await self._get_recent_historical_metrics_for_systems(source_systems)
            
            # Perform root cause analysis
            root_cause_analysis = await asyncio.get_event_loop().run_in_executor(
                self.ml_executor,
                self.root_cause_analyzer.analyze_incident,
                incident_data,
                historical_metrics
            )
            
            # Create alert with analysis
            alert = OperationalAlert(
                alert_id=alert_id,
                alert_type=alert_type,
                title=title,
                description=description,
                severity=severity,
                source_systems=source_systems,
                affected_metrics=affected_metrics or [],
                root_cause_analysis=root_cause_analysis,
                suggested_actions=root_cause_analysis.get('recommended_actions', []),
                status='active'
            )
            
            # Store alert
            self.active_alerts[alert_id] = alert
            
            # Persist to database
            await self._persist_operational_alert(alert)
            
            # Publish alert to Redis for real-time notifications
            await self._publish_alert_notification(alert)
            
            self.logger.info(f"Created operational alert: {alert_id} - {title}")
            
            return alert_id
            
        except Exception as e:
            self.logger.error(f"Error creating operational alert: {e}")
            raise
    
    async def _persist_operational_alert(self, alert: OperationalAlert) -> None:
        """Persist operational alert to database"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO operational_alerts (
                        alert_id, alert_type, title, description, severity,
                        source_systems, affected_metrics, correlation_id,
                        root_cause_analysis, suggested_actions, escalation_path,
                        auto_resolution_attempts, status, acknowledged_by,
                        resolved_by, created_at, updated_at, resolved_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)
                """,
                    alert.alert_id,
                    alert.alert_type,
                    alert.title,
                    alert.description,
                    alert.severity.value,
                    alert.source_systems,
                    alert.affected_metrics,
                    alert.correlation_id,
                    json.dumps(alert.root_cause_analysis) if alert.root_cause_analysis else None,
                    alert.suggested_actions,
                    alert.escalation_path,
                    json.dumps(alert.auto_resolution_attempts),
                    alert.status,
                    alert.acknowledged_by,
                    alert.resolved_by,
                    alert.created_at,
                    alert.updated_at,
                    alert.resolved_at
                )
            
        except Exception as e:
            self.logger.error(f"Error persisting operational alert: {e}")
    
    async def _publish_alert_notification(self, alert: OperationalAlert) -> None:
        """Publish alert notification to Redis"""
        try:
            notification = {
                'type': 'operational_alert',
                'alert_id': alert.alert_id,
                'severity': alert.severity.value,
                'title': alert.title,
                'description': alert.description,
                'source_systems': alert.source_systems,
                'suggested_actions': alert.suggested_actions,
                'timestamp': alert.created_at.isoformat()
            }
            
            await self.redis_client.publish(
                'nautilus:operational_alerts',
                json.dumps(notification)
            )
            
        except Exception as e:
            self.logger.error(f"Error publishing alert notification: {e}")


# Global instance
intelligent_operations_dashboard = None

def get_intelligent_operations_dashboard() -> IntelligentOperationsDashboard:
    """Get global intelligent operations dashboard instance"""
    global intelligent_operations_dashboard
    if intelligent_operations_dashboard is None:
        raise RuntimeError("Intelligent operations dashboard not initialized")
    return intelligent_operations_dashboard

def init_intelligent_operations_dashboard(
    database_url: str, 
    redis_url: str
) -> IntelligentOperationsDashboard:
    """Initialize global intelligent operations dashboard instance"""
    global intelligent_operations_dashboard
    intelligent_operations_dashboard = IntelligentOperationsDashboard(database_url, redis_url)
    return intelligent_operations_dashboard