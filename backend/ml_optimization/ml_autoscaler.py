"""
ML-Powered Auto-Scaling System

This module implements intelligent auto-scaling that goes beyond CPU/memory metrics
by predicting trading patterns, market events, and resource demands.
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import redis
import json

# Kubernetes imports
try:
    from kubernetes import client, config
    from kubernetes.client import ApiException
    KUBERNETES_AVAILABLE = True
except ImportError:
    KUBERNETES_AVAILABLE = False
    logging.warning("Kubernetes client not available - using simulation mode")


class TradingPattern(Enum):
    """Trading pattern classifications for scaling decisions"""
    LOW_ACTIVITY = "low_activity"
    NORMAL_TRADING = "normal_trading" 
    HIGH_VOLUME = "high_volume"
    VOLATILE_MARKET = "volatile_market"
    MARKET_OPEN = "market_open"
    MARKET_CLOSE = "market_close"
    NEWS_EVENT = "news_event"
    EARNINGS_SEASON = "earnings_season"
    ECONOMIC_DATA = "economic_data"
    CRISIS_MODE = "crisis_mode"


class ScalingDecision(Enum):
    """Auto-scaling decision types"""
    SCALE_UP_AGGRESSIVE = "scale_up_aggressive"
    SCALE_UP_MODERATE = "scale_up_moderate"
    MAINTAIN = "maintain"
    SCALE_DOWN_MODERATE = "scale_down_moderate"
    SCALE_DOWN_AGGRESSIVE = "scale_down_aggressive"


@dataclass
class ScalingMetrics:
    """Comprehensive metrics for ML-powered scaling decisions"""
    # Traditional metrics
    cpu_utilization: float
    memory_utilization: float
    active_connections: int
    request_rate: float
    
    # Trading-specific metrics
    market_volatility: float
    trading_volume: float
    order_flow_rate: float
    price_movement_velocity: float
    
    # Time-based features
    hour_of_day: int
    day_of_week: int
    is_market_hours: bool
    time_to_market_open: float
    time_to_market_close: float
    
    # Market event indicators
    earnings_events_today: int
    economic_releases_today: int
    fed_meeting_proximity: float
    
    # Historical context
    avg_volume_last_hour: float
    avg_volatility_last_hour: float
    trend_direction: float  # -1 to 1


@dataclass
class MLPrediction:
    """ML model prediction results"""
    predicted_load: float
    confidence: float
    pattern: TradingPattern
    scaling_recommendation: ScalingDecision
    recommended_replicas: int
    prediction_horizon: int  # minutes
    feature_importance: Dict[str, float] = field(default_factory=dict)


class MLAutoScaler:
    """
    Intelligent auto-scaling system using ML to predict resource demands
    based on trading patterns, market conditions, and historical data.
    """
    
    def __init__(self, namespace: str = "nautilus-trading", redis_url: str = "redis://localhost:6379"):
        self.namespace = namespace
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.logger = logging.getLogger(__name__)
        
        # ML Models
        self.load_predictor = None  # Predicts resource load
        self.pattern_classifier = None  # Classifies trading patterns
        self.scaler = StandardScaler()
        
        # Model performance tracking
        self.prediction_accuracy_history = []
        self.scaling_decisions_history = []
        
        # Configuration
        self.prediction_horizon_minutes = 15
        self.min_replicas = 2
        self.max_replicas = 20
        self.scaling_cooldown_seconds = 300
        
        # Kubernetes client
        if KUBERNETES_AVAILABLE:
            try:
                config.load_incluster_config()  # For running in cluster
            except:
                try:
                    config.load_kube_config()  # For local development
                except:
                    self.logger.warning("Could not load Kubernetes config")
            self.k8s_apps_v1 = client.AppsV1Api()
            self.k8s_autoscaling_v2 = client.AutoscalingV2Api()
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models for prediction and classification"""
        # Load predictor - predicts resource requirements
        self.load_predictor = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=8,
            subsample=0.8,
            random_state=42
        )
        
        # Pattern classifier - identifies trading patterns
        self.pattern_classifier = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
    
    async def collect_metrics(self, service_name: str) -> ScalingMetrics:
        """
        Collect comprehensive metrics for scaling decisions
        """
        try:
            # Get basic Kubernetes metrics
            cpu_util, memory_util = await self._get_k8s_metrics(service_name)
            
            # Get trading-specific metrics from Redis
            trading_metrics = await self._get_trading_metrics()
            
            # Get time-based features
            now = datetime.now()
            market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
            
            # Calculate time features
            time_to_open = (market_open - now).total_seconds() / 3600 if now < market_open else 0
            time_to_close = (market_close - now).total_seconds() / 3600 if now < market_close else 0
            is_market_hours = market_open <= now <= market_close
            
            # Get market event data
            events = await self._get_market_events()
            
            return ScalingMetrics(
                cpu_utilization=cpu_util,
                memory_utilization=memory_util,
                active_connections=trading_metrics.get('active_connections', 0),
                request_rate=trading_metrics.get('request_rate', 0),
                market_volatility=trading_metrics.get('volatility', 0),
                trading_volume=trading_metrics.get('volume', 0),
                order_flow_rate=trading_metrics.get('order_flow', 0),
                price_movement_velocity=trading_metrics.get('price_velocity', 0),
                hour_of_day=now.hour,
                day_of_week=now.weekday(),
                is_market_hours=is_market_hours,
                time_to_market_open=time_to_open,
                time_to_market_close=time_to_close,
                earnings_events_today=events.get('earnings', 0),
                economic_releases_today=events.get('economic', 0),
                fed_meeting_proximity=events.get('fed_proximity', 10.0),
                avg_volume_last_hour=trading_metrics.get('avg_volume_1h', 0),
                avg_volatility_last_hour=trading_metrics.get('avg_volatility_1h', 0),
                trend_direction=trading_metrics.get('trend', 0)
            )
            
        except Exception as e:
            self.logger.error(f"Error collecting metrics: {str(e)}")
            # Return default metrics
            now = datetime.now()
            return ScalingMetrics(
                cpu_utilization=50.0,
                memory_utilization=60.0,
                active_connections=100,
                request_rate=50.0,
                market_volatility=0.02,
                trading_volume=1000000,
                order_flow_rate=100,
                price_movement_velocity=0.001,
                hour_of_day=now.hour,
                day_of_week=now.weekday(),
                is_market_hours=True,
                time_to_market_open=2.0,
                time_to_market_close=6.0,
                earnings_events_today=0,
                economic_releases_today=1,
                fed_meeting_proximity=5.0,
                avg_volume_last_hour=800000,
                avg_volatility_last_hour=0.018,
                trend_direction=0.1
            )
    
    async def _get_k8s_metrics(self, service_name: str) -> Tuple[float, float]:
        """Get CPU and memory utilization from Kubernetes metrics"""
        try:
            if not KUBERNETES_AVAILABLE:
                return 65.0, 70.0  # Simulated values
            
            # In a real implementation, you would use metrics-server or Prometheus
            # For now, return simulated values with some variability
            import random
            cpu = random.uniform(40, 90)
            memory = random.uniform(50, 85)
            return cpu, memory
            
        except Exception as e:
            self.logger.error(f"Error getting K8s metrics: {str(e)}")
            return 65.0, 70.0
    
    async def _get_trading_metrics(self) -> Dict[str, float]:
        """Get trading-specific metrics from Redis"""
        try:
            # Get cached trading metrics
            metrics_str = self.redis_client.get("trading:metrics:current")
            if metrics_str:
                return json.loads(metrics_str)
            
            # Return default trading metrics if not available
            import random
            return {
                'active_connections': random.randint(50, 500),
                'request_rate': random.uniform(20, 200),
                'volatility': random.uniform(0.01, 0.05),
                'volume': random.randint(500000, 2000000),
                'order_flow': random.uniform(50, 300),
                'price_velocity': random.uniform(0.0001, 0.005),
                'avg_volume_1h': random.randint(400000, 1500000),
                'avg_volatility_1h': random.uniform(0.008, 0.04),
                'trend': random.uniform(-0.5, 0.5)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting trading metrics: {str(e)}")
            return {}
    
    async def _get_market_events(self) -> Dict[str, Any]:
        """Get upcoming market events that might affect resource needs"""
        try:
            events_str = self.redis_client.get("market:events:today")
            if events_str:
                return json.loads(events_str)
            
            # Return simulated market events
            import random
            return {
                'earnings': random.randint(0, 5),
                'economic': random.randint(0, 3),
                'fed_proximity': random.uniform(1.0, 30.0)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting market events: {str(e)}")
            return {'earnings': 0, 'economic': 0, 'fed_proximity': 10.0}
    
    def _metrics_to_features(self, metrics: ScalingMetrics) -> np.ndarray:
        """Convert scaling metrics to ML features"""
        features = [
            metrics.cpu_utilization,
            metrics.memory_utilization,
            metrics.active_connections,
            metrics.request_rate,
            metrics.market_volatility,
            metrics.trading_volume / 1000000,  # Normalize volume
            metrics.order_flow_rate,
            metrics.price_movement_velocity * 1000,  # Scale for ML
            metrics.hour_of_day,
            metrics.day_of_week,
            float(metrics.is_market_hours),
            metrics.time_to_market_open,
            metrics.time_to_market_close,
            metrics.earnings_events_today,
            metrics.economic_releases_today,
            metrics.fed_meeting_proximity,
            metrics.avg_volume_last_hour / 1000000,
            metrics.avg_volatility_last_hour,
            metrics.trend_direction
        ]
        
        return np.array(features).reshape(1, -1)
    
    async def predict_scaling_needs(self, metrics: ScalingMetrics) -> MLPrediction:
        """
        Use ML models to predict scaling needs based on current metrics
        """
        try:
            # Convert metrics to features
            features = self._metrics_to_features(metrics)
            
            # Scale features if scaler is fitted
            if hasattr(self.scaler, 'mean_'):
                features_scaled = self.scaler.transform(features)
            else:
                features_scaled = features
            
            # Make predictions
            if self.load_predictor is None:
                await self._load_or_train_models()
            
            # Predict resource load
            predicted_load = self.load_predictor.predict(features_scaled)[0]
            
            # Classify trading pattern
            pattern_score = self.pattern_classifier.predict(features_scaled)[0]
            pattern = self._score_to_pattern(pattern_score)
            
            # Calculate confidence based on feature stability
            confidence = self._calculate_confidence(metrics)
            
            # Make scaling decision
            scaling_decision, recommended_replicas = self._make_scaling_decision(
                predicted_load, pattern, metrics
            )
            
            # Get feature importance (simplified)
            feature_importance = self._get_feature_importance(features)
            
            return MLPrediction(
                predicted_load=predicted_load,
                confidence=confidence,
                pattern=pattern,
                scaling_recommendation=scaling_decision,
                recommended_replicas=recommended_replicas,
                prediction_horizon=self.prediction_horizon_minutes,
                feature_importance=feature_importance
            )
            
        except Exception as e:
            self.logger.error(f"Error in ML prediction: {str(e)}")
            # Fallback to rule-based prediction
            return self._fallback_prediction(metrics)
    
    def _score_to_pattern(self, score: float) -> TradingPattern:
        """Convert pattern classifier score to trading pattern enum"""
        if score < 0.1:
            return TradingPattern.LOW_ACTIVITY
        elif score < 0.3:
            return TradingPattern.NORMAL_TRADING
        elif score < 0.5:
            return TradingPattern.HIGH_VOLUME
        elif score < 0.7:
            return TradingPattern.VOLATILE_MARKET
        elif score < 0.8:
            return TradingPattern.MARKET_OPEN
        elif score < 0.85:
            return TradingPattern.MARKET_CLOSE
        elif score < 0.9:
            return TradingPattern.NEWS_EVENT
        elif score < 0.95:
            return TradingPattern.EARNINGS_SEASON
        else:
            return TradingPattern.CRISIS_MODE
    
    def _calculate_confidence(self, metrics: ScalingMetrics) -> float:
        """Calculate prediction confidence based on metric stability"""
        # Simple confidence calculation based on various factors
        confidence = 0.8  # Base confidence
        
        # Adjust based on market hours (higher confidence during market hours)
        if metrics.is_market_hours:
            confidence += 0.1
        else:
            confidence -= 0.1
        
        # Adjust based on volatility (lower confidence in high volatility)
        if metrics.market_volatility > 0.03:
            confidence -= 0.2
        
        # Adjust based on time proximity to market events
        if metrics.time_to_market_open < 1 or metrics.time_to_market_close < 1:
            confidence += 0.1
        
        return max(0.1, min(1.0, confidence))
    
    def _make_scaling_decision(
        self, 
        predicted_load: float, 
        pattern: TradingPattern, 
        metrics: ScalingMetrics
    ) -> Tuple[ScalingDecision, int]:
        """Make intelligent scaling decision based on ML predictions"""
        
        current_replicas = 3  # Would get from actual deployment
        
        # Base decision on predicted load
        if predicted_load > 0.8:
            if pattern in [TradingPattern.CRISIS_MODE, TradingPattern.VOLATILE_MARKET]:
                decision = ScalingDecision.SCALE_UP_AGGRESSIVE
                replicas = min(self.max_replicas, current_replicas * 2)
            else:
                decision = ScalingDecision.SCALE_UP_MODERATE
                replicas = min(self.max_replicas, current_replicas + 2)
        
        elif predicted_load > 0.6:
            if pattern == TradingPattern.HIGH_VOLUME:
                decision = ScalingDecision.SCALE_UP_MODERATE
                replicas = min(self.max_replicas, current_replicas + 1)
            else:
                decision = ScalingDecision.MAINTAIN
                replicas = current_replicas
        
        elif predicted_load < 0.3:
            if pattern == TradingPattern.LOW_ACTIVITY:
                decision = ScalingDecision.SCALE_DOWN_MODERATE
                replicas = max(self.min_replicas, current_replicas - 1)
            else:
                decision = ScalingDecision.MAINTAIN
                replicas = current_replicas
        
        else:
            decision = ScalingDecision.MAINTAIN
            replicas = current_replicas
        
        return decision, replicas
    
    def _get_feature_importance(self, features: np.ndarray) -> Dict[str, float]:
        """Get simplified feature importance for the prediction"""
        feature_names = [
            'cpu_utilization', 'memory_utilization', 'active_connections', 
            'request_rate', 'market_volatility', 'trading_volume',
            'order_flow_rate', 'price_movement_velocity', 'hour_of_day',
            'day_of_week', 'is_market_hours', 'time_to_market_open',
            'time_to_market_close', 'earnings_events_today', 'economic_releases_today',
            'fed_meeting_proximity', 'avg_volume_last_hour', 'avg_volatility_last_hour',
            'trend_direction'
        ]
        
        # Simplified importance based on feature values
        importance = {}
        for i, name in enumerate(feature_names):
            if i < len(features[0]):
                importance[name] = abs(float(features[0][i])) / 100  # Normalize
        
        # Normalize to sum to 1
        total = sum(importance.values())
        if total > 0:
            importance = {k: v/total for k, v in importance.items()}
        
        return importance
    
    def _fallback_prediction(self, metrics: ScalingMetrics) -> MLPrediction:
        """Fallback rule-based prediction when ML models fail"""
        # Simple rule-based logic
        load_score = (metrics.cpu_utilization + metrics.memory_utilization) / 200
        
        if load_score > 0.8:
            decision = ScalingDecision.SCALE_UP_MODERATE
            replicas = 5
        elif load_score < 0.3:
            decision = ScalingDecision.SCALE_DOWN_MODERATE
            replicas = 2
        else:
            decision = ScalingDecision.MAINTAIN
            replicas = 3
        
        return MLPrediction(
            predicted_load=load_score,
            confidence=0.5,
            pattern=TradingPattern.NORMAL_TRADING,
            scaling_recommendation=decision,
            recommended_replicas=replicas,
            prediction_horizon=self.prediction_horizon_minutes
        )
    
    async def _load_or_train_models(self):
        """Load existing models or train new ones"""
        try:
            # Try to load existing models
            self.load_predictor = joblib.load('/app/models/load_predictor.pkl')
            self.pattern_classifier = joblib.load('/app/models/pattern_classifier.pkl')
            self.scaler = joblib.load('/app/models/scaler.pkl')
            self.logger.info("Loaded existing ML models")
        except:
            # Generate synthetic training data and train models
            await self._train_models_with_synthetic_data()
            self.logger.info("Trained new ML models with synthetic data")
    
    async def _train_models_with_synthetic_data(self):
        """Train models with synthetic data until real data is available"""
        # Generate synthetic training data
        n_samples = 10000
        X_synthetic, y_load, y_pattern = self._generate_synthetic_training_data(n_samples)
        
        # Fit scaler
        self.scaler.fit(X_synthetic)
        X_scaled = self.scaler.transform(X_synthetic)
        
        # Train models
        self.load_predictor.fit(X_scaled, y_load)
        self.pattern_classifier.fit(X_scaled, y_pattern)
        
        # Save models
        joblib.dump(self.load_predictor, '/tmp/load_predictor.pkl')
        joblib.dump(self.pattern_classifier, '/tmp/pattern_classifier.pkl')
        joblib.dump(self.scaler, '/tmp/scaler.pkl')
    
    def _generate_synthetic_training_data(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate synthetic training data for model initialization"""
        np.random.seed(42)
        
        X = []
        y_load = []
        y_pattern = []
        
        for _ in range(n_samples):
            # Generate realistic feature combinations
            hour = np.random.randint(0, 24)
            day_of_week = np.random.randint(0, 7)
            is_market_hours = 1.0 if 9 <= hour <= 16 and day_of_week < 5 else 0.0
            
            # Base resource usage varies by time
            base_cpu = 30 + 40 * is_market_hours + 10 * np.sin(hour * np.pi / 12)
            base_memory = 40 + 30 * is_market_hours + 5 * np.sin(hour * np.pi / 12)
            
            # Add volatility effect
            volatility = np.random.uniform(0.005, 0.05)
            volatility_multiplier = 1 + volatility * 5
            
            # Market event effects
            earnings_events = np.random.poisson(1)
            economic_events = np.random.poisson(0.3)
            event_multiplier = 1 + (earnings_events + economic_events) * 0.1
            
            features = [
                base_cpu * volatility_multiplier * event_multiplier + np.random.normal(0, 10),
                base_memory * volatility_multiplier * event_multiplier + np.random.normal(0, 5),
                np.random.randint(10, 800) * is_market_hours + np.random.randint(5, 100),
                np.random.uniform(10, 300) * is_market_hours + np.random.uniform(5, 50),
                volatility,
                np.random.uniform(0.5, 3.0) * is_market_hours + 0.1,  # trading_volume
                np.random.uniform(20, 500) * is_market_hours + 10,
                volatility * np.random.uniform(0.1, 2.0),  # price_velocity
                hour,
                day_of_week,
                is_market_hours,
                max(0, 9 - hour) if hour < 9 else 0,  # time_to_market_open
                max(0, 16 - hour) if hour < 16 else 0,  # time_to_market_close
                earnings_events,
                economic_events,
                np.random.uniform(1, 20),  # fed_meeting_proximity
                np.random.uniform(0.3, 2.5) * is_market_hours + 0.05,  # avg_volume_1h
                np.random.uniform(0.003, 0.04),  # avg_volatility_1h
                np.random.uniform(-0.5, 0.5)  # trend_direction
            ]
            
            X.append(features)
            
            # Target: resource load (normalized 0-1)
            resource_load = min(1.0, (features[0] + features[1]) / 200 * volatility_multiplier * event_multiplier)
            y_load.append(resource_load)
            
            # Pattern classification (0-1 score)
            pattern_score = volatility * 10 + event_multiplier * 0.2 + (1 - is_market_hours) * 0.1
            y_pattern.append(min(1.0, pattern_score))
        
        return np.array(X), np.array(y_load), np.array(y_pattern)
    
    async def execute_scaling_decision(self, service_name: str, prediction: MLPrediction) -> Dict[str, Any]:
        """Execute the scaling decision in Kubernetes"""
        try:
            result = {
                "service": service_name,
                "prediction": prediction,
                "execution_time": datetime.now().isoformat(),
                "success": False,
                "message": ""
            }
            
            if not KUBERNETES_AVAILABLE:
                result["success"] = True
                result["message"] = f"Simulated scaling {service_name} to {prediction.recommended_replicas} replicas"
                self.logger.info(result["message"])
                return result
            
            # Update HPA if the scaling recommendation is significant
            if prediction.scaling_recommendation in [
                ScalingDecision.SCALE_UP_AGGRESSIVE,
                ScalingDecision.SCALE_UP_MODERATE,
                ScalingDecision.SCALE_DOWN_MODERATE
            ]:
                # Get current HPA
                try:
                    hpa = self.k8s_autoscaling_v2.read_namespaced_horizontal_pod_autoscaler(
                        name=f"{service_name}-hpa",
                        namespace=self.namespace
                    )
                    
                    # Update replicas based on ML prediction
                    hpa.spec.min_replicas = max(self.min_replicas, prediction.recommended_replicas - 1)
                    hpa.spec.max_replicas = min(self.max_replicas, prediction.recommended_replicas + 5)
                    
                    # Update HPA
                    self.k8s_autoscaling_v2.patch_namespaced_horizontal_pod_autoscaler(
                        name=f"{service_name}-hpa",
                        namespace=self.namespace,
                        body=hpa
                    )
                    
                    result["success"] = True
                    result["message"] = f"Updated HPA for {service_name} based on ML prediction"
                    
                except ApiException as e:
                    result["message"] = f"Kubernetes API error: {str(e)}"
                    self.logger.error(result["message"])
            
            else:
                result["success"] = True
                result["message"] = f"No scaling action needed for {service_name}"
            
            # Store decision history
            await self._store_scaling_decision(service_name, prediction, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing scaling decision: {str(e)}")
            return {
                "service": service_name,
                "success": False,
                "error": str(e),
                "execution_time": datetime.now().isoformat()
            }
    
    async def _store_scaling_decision(self, service_name: str, prediction: MLPrediction, result: Dict[str, Any]):
        """Store scaling decision for historical analysis and model improvement"""
        decision_record = {
            "timestamp": datetime.now().isoformat(),
            "service": service_name,
            "prediction": {
                "predicted_load": prediction.predicted_load,
                "confidence": prediction.confidence,
                "pattern": prediction.pattern.value,
                "scaling_recommendation": prediction.scaling_recommendation.value,
                "recommended_replicas": prediction.recommended_replicas
            },
            "execution_result": result
        }
        
        # Store in Redis for analysis
        self.redis_client.lpush(
            f"ml_scaling:history:{service_name}",
            json.dumps(decision_record)
        )
        
        # Keep only last 1000 decisions per service
        self.redis_client.ltrim(f"ml_scaling:history:{service_name}", 0, 999)
    
    async def run_continuous_optimization(self, services: List[str], interval_seconds: int = 300):
        """
        Run continuous ML-powered optimization for specified services
        """
        self.logger.info(f"Starting continuous ML optimization for services: {services}")
        
        while True:
            try:
                optimization_start = datetime.now()
                
                for service in services:
                    # Collect current metrics
                    metrics = await self.collect_metrics(service)
                    
                    # Make ML prediction
                    prediction = await self.predict_scaling_needs(metrics)
                    
                    # Execute scaling decision if confidence is high enough
                    if prediction.confidence > 0.6:
                        result = await self.execute_scaling_decision(service, prediction)
                        
                        self.logger.info(
                            f"ML Optimization for {service}: "
                            f"Pattern={prediction.pattern.value}, "
                            f"Load={prediction.predicted_load:.2f}, "
                            f"Confidence={prediction.confidence:.2f}, "
                            f"Action={prediction.scaling_recommendation.value}, "
                            f"Replicas={prediction.recommended_replicas}"
                        )
                    else:
                        self.logger.info(
                            f"Low confidence prediction for {service} "
                            f"({prediction.confidence:.2f}), skipping scaling action"
                        )
                
                # Calculate optimization cycle time
                cycle_time = (datetime.now() - optimization_start).total_seconds()
                self.logger.info(f"Optimization cycle completed in {cycle_time:.2f}s")
                
                # Wait for next cycle
                await asyncio.sleep(max(0, interval_seconds - cycle_time))
                
            except Exception as e:
                self.logger.error(f"Error in continuous optimization: {str(e)}")
                await asyncio.sleep(interval_seconds)


async def main():
    """Test the ML Auto-Scaler"""
    logging.basicConfig(level=logging.INFO)
    
    # Initialize ML Auto-Scaler
    autoscaler = MLAutoScaler()
    
    # Test services
    services = ["nautilus-market-data", "nautilus-strategy-engine", "nautilus-risk-engine"]
    
    print("🤖 Testing ML-Powered Auto-Scaling System")
    print("=" * 50)
    
    # Test individual prediction
    for service in services:
        print(f"\n📊 Testing {service}:")
        
        # Collect metrics
        metrics = await autoscaler.collect_metrics(service)
        print(f"CPU: {metrics.cpu_utilization:.1f}%, Memory: {metrics.memory_utilization:.1f}%")
        print(f"Market Volatility: {metrics.market_volatility:.3f}, Trading Volume: {metrics.trading_volume:,}")
        
        # Make prediction
        prediction = await autoscaler.predict_scaling_needs(metrics)
        print(f"Predicted Load: {prediction.predicted_load:.2f}")
        print(f"Trading Pattern: {prediction.pattern.value}")
        print(f"Confidence: {prediction.confidence:.2f}")
        print(f"Scaling Decision: {prediction.scaling_recommendation.value}")
        print(f"Recommended Replicas: {prediction.recommended_replicas}")
        
        # Execute scaling (simulation)
        result = await autoscaler.execute_scaling_decision(service, prediction)
        print(f"Execution: {result['message']}")
    
    print(f"\n🚀 Starting continuous optimization (simulation)...")
    print("Press Ctrl+C to stop")
    
    try:
        # Run for a short time to demonstrate
        await asyncio.wait_for(
            autoscaler.run_continuous_optimization(services, interval_seconds=30),
            timeout=120  # Run for 2 minutes
        )
    except asyncio.TimeoutError:
        print("\n✅ ML Auto-Scaling test completed successfully!")
    except KeyboardInterrupt:
        print("\n⏹️  Stopped by user")


if __name__ == "__main__":
    asyncio.run(main())