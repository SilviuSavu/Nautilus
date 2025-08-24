"""
Intelligent Fraud Detection System
Real-time Analysis & ML-based Detection for Financial Trading Platforms

Provides comprehensive fraud detection with behavioral analysis, pattern recognition,
and real-time anomaly detection specifically designed for financial trading environments.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
import hashlib
from collections import defaultdict, deque
import numpy as np
import pandas as pd
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession
import aiohttp
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class FraudType(Enum):
    """Types of fraud in financial trading"""
    ACCOUNT_TAKEOVER = "account_takeover"
    IDENTITY_THEFT = "identity_theft"
    PAYMENT_FRAUD = "payment_fraud"
    MARKET_MANIPULATION = "market_manipulation"
    INSIDER_TRADING = "insider_trading"
    PUMP_AND_DUMP = "pump_and_dump"
    WASH_TRADING = "wash_trading"
    SPOOFING = "spoofing"
    LAYERING = "layering"
    FRONT_RUNNING = "front_running"
    UNAUTHORIZED_TRADING = "unauthorized_trading"
    SYNTHETIC_IDENTITY = "synthetic_identity"
    MONEY_LAUNDERING = "money_laundering"
    CHARGEBACK_FRAUD = "chargeback_fraud"
    SOCIAL_ENGINEERING = "social_engineering"


class FraudSeverity(Enum):
    """Fraud severity levels"""
    SUSPICIOUS = "suspicious"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DetectionModel(Enum):
    """Fraud detection model types"""
    STATISTICAL_ANOMALY = "statistical_anomaly"
    MACHINE_LEARNING = "machine_learning"
    BEHAVIORAL_ANALYSIS = "behavioral_analysis"
    PATTERN_MATCHING = "pattern_matching"
    NETWORK_ANALYSIS = "network_analysis"
    TIME_SERIES = "time_series"
    ENSEMBLE = "ensemble"


@dataclass
class FraudAlert:
    """Fraud detection alert"""
    alert_id: str
    timestamp: datetime
    fraud_type: FraudType
    severity: FraudSeverity
    confidence_score: float  # 0.0-1.0
    risk_score: float  # 0.0-100.0
    detection_model: DetectionModel
    affected_entities: Dict[str, List[str]]  # entity_type: [entity_ids]
    evidence: Dict[str, Any]
    behavioral_indicators: List[Dict[str, Any]]
    financial_impact: Optional[float] = None
    recommended_actions: List[str] = field(default_factory=list)
    false_positive_likelihood: float = 0.1
    investigation_priority: int = 5  # 1-10 scale
    related_alerts: List[str] = field(default_factory=list)


@dataclass
class TradingBehaviorProfile:
    """Trading behavior profile for fraud detection"""
    profile_id: str
    user_id: str
    account_id: str
    creation_date: datetime
    last_updated: datetime
    trading_patterns: Dict[str, Any]
    risk_tolerance: Dict[str, float]
    typical_instruments: Set[str]
    typical_position_sizes: Dict[str, float]
    typical_trading_hours: List[Tuple[int, int]]  # (start_hour, end_hour)
    geographic_patterns: Set[str]
    device_patterns: Set[str]
    api_usage_patterns: Dict[str, Any]
    anomaly_history: List[Dict[str, Any]]
    trust_score: float = 0.5
    learning_confidence: float = 0.0


class BehavioralAnalyzer:
    """Analyzes trading behavior for fraud detection"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.profiles = {}
        self.anomaly_detectors = {}
        self.baseline_windows = {
            "short_term": timedelta(days=7),
            "medium_term": timedelta(days=30),
            "long_term": timedelta(days=90)
        }
        
    async def analyze_trading_behavior(self, user_id: str, account_id: str,
                                     trading_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trading behavior for fraud indicators"""
        try:
            # Get or create behavior profile
            profile = await self._get_trading_profile(user_id, account_id)
            
            # Extract behavioral features
            features = self._extract_trading_features(trading_data)
            
            # Detect behavioral anomalies
            anomalies = await self._detect_trading_anomalies(profile, features)
            
            # Update profile with new data
            await self._update_trading_profile(profile, features)
            
            # Calculate fraud risk score
            fraud_risk = self._calculate_fraud_risk(profile, anomalies, features)
            
            return {
                "user_id": user_id,
                "account_id": account_id,
                "fraud_risk_score": fraud_risk,
                "anomalies": anomalies,
                "behavioral_changes": self._detect_behavioral_shifts(profile, features),
                "red_flags": self._identify_red_flags(features, anomalies),
                "trust_score": profile.trust_score
            }
            
        except Exception as e:
            logger.error(f"Trading behavior analysis failed: {e}")
            return {"error": str(e)}
    
    def _extract_trading_features(self, trading_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant features from trading data"""
        features = {
            "timestamp": datetime.now(),
            "order_frequency": trading_data.get("order_count", 0),
            "total_volume": trading_data.get("total_volume", 0.0),
            "position_sizes": trading_data.get("position_sizes", []),
            "instruments_traded": set(trading_data.get("instruments", [])),
            "pnl_realized": trading_data.get("pnl_realized", 0.0),
            "pnl_unrealized": trading_data.get("pnl_unrealized", 0.0),
            "win_rate": trading_data.get("win_rate", 0.0),
            "average_hold_time": trading_data.get("avg_hold_time", 0),
            "risk_metrics": trading_data.get("risk_metrics", {}),
            "geographic_location": trading_data.get("location", "unknown"),
            "device_fingerprint": trading_data.get("device_fingerprint", ""),
            "api_usage": trading_data.get("api_calls", {}),
            "session_duration": trading_data.get("session_duration", 0),
            "concurrent_sessions": trading_data.get("concurrent_sessions", 1)
        }
        
        # Calculate derived features
        if features["position_sizes"]:
            features["position_size_variance"] = np.var(features["position_sizes"])
            features["max_position_size"] = max(features["position_sizes"])
            features["avg_position_size"] = np.mean(features["position_sizes"])
        
        return features
    
    async def _detect_trading_anomalies(self, profile: TradingBehaviorProfile,
                                      features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in trading behavior"""
        anomalies = []
        
        # Volume anomalies
        if await self._is_volume_anomaly(profile, features["total_volume"]):
            anomalies.append({
                "type": "volume_anomaly",
                "feature": "total_volume",
                "current_value": features["total_volume"],
                "baseline": profile.trading_patterns.get("avg_volume", 0),
                "severity": "medium"
            })
        
        # Frequency anomalies
        if await self._is_frequency_anomaly(profile, features["order_frequency"]):
            anomalies.append({
                "type": "frequency_anomaly", 
                "feature": "order_frequency",
                "current_value": features["order_frequency"],
                "baseline": profile.trading_patterns.get("avg_frequency", 0),
                "severity": "medium"
            })
        
        # Instrument diversity anomalies
        if await self._is_instrument_anomaly(profile, features["instruments_traded"]):
            anomalies.append({
                "type": "instrument_anomaly",
                "feature": "instruments_traded",
                "current_value": len(features["instruments_traded"]),
                "baseline": len(profile.typical_instruments),
                "severity": "low"
            })
        
        # Geographic anomalies
        if await self._is_geographic_anomaly(profile, features["geographic_location"]):
            anomalies.append({
                "type": "geographic_anomaly",
                "feature": "geographic_location", 
                "current_value": features["geographic_location"],
                "baseline": list(profile.geographic_patterns),
                "severity": "high"
            })
        
        # Time-based anomalies
        if await self._is_timing_anomaly(profile, features["timestamp"]):
            anomalies.append({
                "type": "timing_anomaly",
                "feature": "trading_time",
                "current_value": features["timestamp"].hour,
                "baseline": profile.typical_trading_hours,
                "severity": "medium"
            })
        
        return anomalies
    
    def _calculate_fraud_risk(self, profile: TradingBehaviorProfile,
                            anomalies: List[Dict[str, Any]],
                            features: Dict[str, Any]) -> float:
        """Calculate overall fraud risk score"""
        base_risk = 10.0  # Base risk score out of 100
        
        # Anomaly contribution
        anomaly_risk = 0.0
        for anomaly in anomalies:
            if anomaly["severity"] == "high":
                anomaly_risk += 25.0
            elif anomaly["severity"] == "medium":
                anomaly_risk += 15.0
            elif anomaly["severity"] == "low":
                anomaly_risk += 5.0
        
        # Trust score impact (inverse relationship)
        trust_impact = (1.0 - profile.trust_score) * 30.0
        
        # Account age factor (newer accounts are riskier)
        account_age = (datetime.now() - profile.creation_date).days
        age_factor = max(0, 20.0 - (account_age / 30.0) * 20.0)
        
        # Recent anomaly history
        recent_anomalies = len([a for a in profile.anomaly_history 
                              if (datetime.now() - datetime.fromisoformat(a["timestamp"])).days <= 7])
        history_risk = min(20.0, recent_anomalies * 5.0)
        
        total_risk = base_risk + anomaly_risk + trust_impact + age_factor + history_risk
        return min(100.0, total_risk)


class PatternMatcher:
    """Matches known fraud patterns in trading data"""
    
    def __init__(self):
        self.fraud_patterns = {}
        self.load_fraud_patterns()
        
    def load_fraud_patterns(self):
        """Load known fraud patterns"""
        self.fraud_patterns = {
            "wash_trading": {
                "description": "Trading between related accounts to create artificial volume",
                "indicators": [
                    "rapid_buy_sell_same_instrument",
                    "matched_order_sizes",
                    "related_account_activity",
                    "minimal_price_movement"
                ],
                "confidence_threshold": 0.8
            },
            "pump_and_dump": {
                "description": "Artificially inflating prices then selling at profit",
                "indicators": [
                    "coordinated_buying_pressure",
                    "social_media_promotion",
                    "rapid_price_increase",
                    "large_sell_orders_after_pump"
                ],
                "confidence_threshold": 0.85
            },
            "spoofing": {
                "description": "Placing large orders with intent to cancel to manipulate prices",
                "indicators": [
                    "large_order_placement",
                    "rapid_order_cancellation",
                    "price_movement_correlation",
                    "repeated_pattern"
                ],
                "confidence_threshold": 0.9
            },
            "layering": {
                "description": "Placing multiple orders at different price levels to manipulate market",
                "indicators": [
                    "multiple_orders_same_side",
                    "price_level_distribution",
                    "order_cancellation_pattern",
                    "market_impact"
                ],
                "confidence_threshold": 0.85
            }
        }
    
    async def detect_patterns(self, trading_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect fraud patterns in trading data"""
        detected_patterns = []
        
        for pattern_name, pattern_config in self.fraud_patterns.items():
            confidence = await self._calculate_pattern_confidence(
                pattern_name, pattern_config, trading_data
            )
            
            if confidence >= pattern_config["confidence_threshold"]:
                detected_patterns.append({
                    "pattern_name": pattern_name,
                    "description": pattern_config["description"],
                    "confidence": confidence,
                    "indicators_matched": await self._get_matched_indicators(
                        pattern_config, trading_data
                    ),
                    "fraud_type": self._pattern_to_fraud_type(pattern_name)
                })
        
        return detected_patterns
    
    async def _calculate_pattern_confidence(self, pattern_name: str,
                                          pattern_config: Dict[str, Any],
                                          trading_data: Dict[str, Any]) -> float:
        """Calculate confidence score for fraud pattern match"""
        indicators = pattern_config["indicators"]
        matched_indicators = 0
        total_weight = 0.0
        matched_weight = 0.0
        
        for indicator in indicators:
            weight = self._get_indicator_weight(indicator)
            total_weight += weight
            
            if await self._check_indicator(indicator, trading_data):
                matched_indicators += 1
                matched_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return matched_weight / total_weight
    
    def _pattern_to_fraud_type(self, pattern_name: str) -> FraudType:
        """Map pattern name to fraud type enum"""
        pattern_mapping = {
            "wash_trading": FraudType.WASH_TRADING,
            "pump_and_dump": FraudType.PUMP_AND_DUMP,
            "spoofing": FraudType.SPOOFING,
            "layering": FraudType.LAYERING,
            "front_running": FraudType.FRONT_RUNNING
        }
        return pattern_mapping.get(pattern_name, FraudType.MARKET_MANIPULATION)


class MachineLearningDetector:
    """Machine learning-based fraud detection"""
    
    def __init__(self):
        self.models = {}
        self.feature_scalers = {}
        self.training_data = defaultdict(list)
        self.model_performance = defaultdict(dict)
        
    async def initialize_models(self):
        """Initialize ML models for fraud detection"""
        try:
            # Isolation Forest for anomaly detection
            self.models["isolation_forest"] = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            )
            
            # Initialize feature scalers
            self.feature_scalers["standard"] = StandardScaler()
            
            logger.info("ML fraud detection models initialized")
            
        except Exception as e:
            logger.error(f"ML model initialization failed: {e}")
    
    async def predict_fraud(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Predict fraud using ML models"""
        try:
            # Prepare feature vector
            feature_vector = self._prepare_feature_vector(features)
            
            if len(feature_vector) == 0:
                return {"error": "No valid features for prediction"}
            
            predictions = {}
            
            # Isolation Forest prediction
            if "isolation_forest" in self.models:
                anomaly_score = self.models["isolation_forest"].decision_function([feature_vector])[0]
                is_anomaly = self.models["isolation_forest"].predict([feature_vector])[0] == -1
                
                predictions["isolation_forest"] = {
                    "is_fraud": is_anomaly,
                    "anomaly_score": float(anomaly_score),
                    "confidence": abs(anomaly_score)
                }
            
            # Ensemble prediction
            fraud_probability = await self._calculate_ensemble_probability(predictions)
            
            return {
                "fraud_probability": fraud_probability,
                "predictions": predictions,
                "feature_importance": await self._get_feature_importance(feature_vector),
                "model_confidence": self._calculate_model_confidence(predictions)
            }
            
        except Exception as e:
            logger.error(f"ML fraud prediction failed: {e}")
            return {"error": str(e)}
    
    def _prepare_feature_vector(self, features: Dict[str, Any]) -> List[float]:
        """Prepare feature vector for ML models"""
        feature_vector = []
        
        # Numerical features
        numerical_features = [
            "order_frequency", "total_volume", "position_size_variance",
            "pnl_realized", "pnl_unrealized", "win_rate", "average_hold_time",
            "session_duration", "concurrent_sessions"
        ]
        
        for feature in numerical_features:
            value = features.get(feature, 0.0)
            if isinstance(value, (int, float)):
                feature_vector.append(float(value))
            else:
                feature_vector.append(0.0)
        
        # Categorical features (encoded)
        if features.get("instruments_traded"):
            feature_vector.append(float(len(features["instruments_traded"])))
        else:
            feature_vector.append(0.0)
        
        return feature_vector
    
    async def _calculate_ensemble_probability(self, predictions: Dict[str, Any]) -> float:
        """Calculate ensemble fraud probability"""
        if not predictions:
            return 0.0
        
        # Simple averaging for now (could be weighted)
        probabilities = []
        
        for model_name, prediction in predictions.items():
            if "anomaly_score" in prediction:
                # Convert anomaly score to probability
                prob = 1.0 / (1.0 + np.exp(-prediction["anomaly_score"]))
                probabilities.append(prob)
        
        if probabilities:
            return float(np.mean(probabilities))
        
        return 0.0


class IntelligentFraudDetection:
    """
    Main intelligent fraud detection system
    """
    
    def __init__(self, redis_url: str, database_url: str):
        self.redis_url = redis_url
        self.database_url = database_url
        self.redis = None
        self.behavioral_analyzer = None
        self.pattern_matcher = None
        self.ml_detector = None
        self.active = False
        self.detection_metrics = defaultdict(int)
        self.alert_queue = deque(maxlen=10000)
        
    async def initialize(self):
        """Initialize fraud detection system"""
        try:
            # Initialize Redis connection
            self.redis = await redis.from_url(self.redis_url)
            
            # Initialize components
            self.behavioral_analyzer = BehavioralAnalyzer(self.redis)
            self.pattern_matcher = PatternMatcher()
            self.ml_detector = MachineLearningDetector()
            
            # Initialize ML models
            await self.ml_detector.initialize_models()
            
            self.active = True
            
            logger.info("Intelligent Fraud Detection system initialized")
            
        except Exception as e:
            logger.error(f"Fraud detection initialization failed: {e}")
            raise
    
    async def analyze_transaction(self, transaction_data: Dict[str, Any]) -> Optional[FraudAlert]:
        """Analyze transaction for fraud indicators"""
        try:
            if not self.active:
                return None
            
            user_id = transaction_data.get("user_id")
            account_id = transaction_data.get("account_id")
            
            # Behavioral analysis
            behavioral_analysis = await self.behavioral_analyzer.analyze_trading_behavior(
                user_id, account_id, transaction_data
            )
            
            # Pattern matching
            pattern_matches = await self.pattern_matcher.detect_patterns(transaction_data)
            
            # ML-based detection
            ml_features = self._extract_ml_features(transaction_data, behavioral_analysis)
            ml_prediction = await self.ml_detector.predict_fraud(ml_features)
            
            # Combine results and generate alert if needed
            alert = await self._generate_fraud_alert(
                transaction_data, behavioral_analysis, pattern_matches, ml_prediction
            )
            
            if alert:
                await self._process_fraud_alert(alert)
                self.detection_metrics[f"fraud_{alert.fraud_type.value}"] += 1
                self.detection_metrics["alerts_total"] += 1
            
            return alert
            
        except Exception as e:
            logger.error(f"Transaction analysis failed: {e}")
            return None
    
    def _extract_ml_features(self, transaction_data: Dict[str, Any],
                           behavioral_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features for ML analysis"""
        features = transaction_data.copy()
        
        # Add behavioral analysis results
        features.update({
            "fraud_risk_score": behavioral_analysis.get("fraud_risk_score", 0.0),
            "anomaly_count": len(behavioral_analysis.get("anomalies", [])),
            "trust_score": behavioral_analysis.get("trust_score", 0.5)
        })
        
        return features
    
    async def _generate_fraud_alert(self, transaction_data: Dict[str, Any],
                                  behavioral_analysis: Dict[str, Any],
                                  pattern_matches: List[Dict[str, Any]],
                                  ml_prediction: Dict[str, Any]) -> Optional[FraudAlert]:
        """Generate fraud alert based on analysis results"""
        try:
            # Calculate overall fraud score
            fraud_score = await self._calculate_fraud_score(
                behavioral_analysis, pattern_matches, ml_prediction
            )
            
            # Determine if alert should be generated
            if fraud_score < 30.0:  # Threshold for alert generation
                return None
            
            # Determine fraud type and severity
            fraud_type = self._determine_fraud_type(pattern_matches, behavioral_analysis)
            severity = self._determine_severity(fraud_score)
            
            # Create alert
            alert = FraudAlert(
                alert_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                fraud_type=fraud_type,
                severity=severity,
                confidence_score=min(1.0, fraud_score / 100.0),
                risk_score=fraud_score,
                detection_model=DetectionModel.ENSEMBLE,
                affected_entities={
                    "users": [transaction_data.get("user_id")],
                    "accounts": [transaction_data.get("account_id")]
                },
                evidence={
                    "behavioral_analysis": behavioral_analysis,
                    "pattern_matches": pattern_matches,
                    "ml_prediction": ml_prediction,
                    "transaction_data": transaction_data
                },
                behavioral_indicators=behavioral_analysis.get("anomalies", []),
                financial_impact=transaction_data.get("transaction_amount"),
                recommended_actions=self._generate_recommendations(fraud_type, severity),
                investigation_priority=self._calculate_priority(fraud_score, fraud_type)
            )
            
            return alert
            
        except Exception as e:
            logger.error(f"Fraud alert generation failed: {e}")
            return None
    
    async def _calculate_fraud_score(self, behavioral_analysis: Dict[str, Any],
                                   pattern_matches: List[Dict[str, Any]],
                                   ml_prediction: Dict[str, Any]) -> float:
        """Calculate overall fraud score"""
        # Behavioral analysis weight: 40%
        behavioral_score = behavioral_analysis.get("fraud_risk_score", 0.0) * 0.4
        
        # Pattern matching weight: 35%
        pattern_score = 0.0
        if pattern_matches:
            max_confidence = max(p.get("confidence", 0.0) for p in pattern_matches)
            pattern_score = max_confidence * 100.0 * 0.35
        
        # ML prediction weight: 25%
        ml_score = 0.0
        if "fraud_probability" in ml_prediction:
            ml_score = ml_prediction["fraud_probability"] * 100.0 * 0.25
        
        return behavioral_score + pattern_score + ml_score
    
    def _determine_fraud_type(self, pattern_matches: List[Dict[str, Any]],
                            behavioral_analysis: Dict[str, Any]) -> FraudType:
        """Determine most likely fraud type"""
        if pattern_matches:
            # Use highest confidence pattern match
            best_match = max(pattern_matches, key=lambda p: p.get("confidence", 0.0))
            return best_match.get("fraud_type", FraudType.UNAUTHORIZED_TRADING)
        
        # Fallback to behavioral indicators
        anomalies = behavioral_analysis.get("anomalies", [])
        if any(a.get("type") == "geographic_anomaly" for a in anomalies):
            return FraudType.ACCOUNT_TAKEOVER
        elif any(a.get("type") == "volume_anomaly" for a in anomalies):
            return FraudType.MARKET_MANIPULATION
        
        return FraudType.UNAUTHORIZED_TRADING
    
    def _determine_severity(self, fraud_score: float) -> FraudSeverity:
        """Determine fraud severity based on score"""
        if fraud_score >= 90.0:
            return FraudSeverity.CRITICAL
        elif fraud_score >= 70.0:
            return FraudSeverity.HIGH
        elif fraud_score >= 50.0:
            return FraudSeverity.MEDIUM
        elif fraud_score >= 30.0:
            return FraudSeverity.LOW
        else:
            return FraudSeverity.SUSPICIOUS
    
    async def _process_fraud_alert(self, alert: FraudAlert):
        """Process and store fraud alert"""
        try:
            # Store alert
            alert_key = f"fraud_alert:{alert.alert_id}"
            alert_data = {
                "alert_id": alert.alert_id,
                "timestamp": alert.timestamp.isoformat(),
                "fraud_type": alert.fraud_type.value,
                "severity": alert.severity.value,
                "confidence_score": alert.confidence_score,
                "risk_score": alert.risk_score,
                "affected_entities": json.dumps(alert.affected_entities),
                "evidence": json.dumps(alert.evidence),
                "recommended_actions": json.dumps(alert.recommended_actions)
            }
            
            await self.redis.hset(alert_key, mapping=alert_data)
            await self.redis.expire(alert_key, 86400 * 30)  # 30 days
            
            # Add to alert queue
            self.alert_queue.append(alert)
            
            # Publish alert
            await self.redis.publish("fraud_alerts", json.dumps({
                "alert_id": alert.alert_id,
                "fraud_type": alert.fraud_type.value,
                "severity": alert.severity.value,
                "confidence": alert.confidence_score,
                "risk_score": alert.risk_score
            }))
            
            logger.warning(f"Fraud alert generated: {alert.alert_id} - {alert.fraud_type.value}")
            
        except Exception as e:
            logger.error(f"Fraud alert processing failed: {e}")
    
    async def get_fraud_detection_status(self) -> Dict[str, Any]:
        """Get fraud detection system status"""
        try:
            status = {
                "active": self.active,
                "detection_metrics": dict(self.detection_metrics),
                "active_alerts": len(self.alert_queue),
                "models_loaded": len(self.ml_detector.models) if self.ml_detector else 0,
                "patterns_loaded": len(self.pattern_matcher.fraud_patterns) if self.pattern_matcher else 0,
                "last_updated": datetime.now().isoformat()
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Status retrieval failed: {e}")
            return {"error": str(e)}
    
    async def shutdown(self):
        """Graceful shutdown of fraud detection system"""
        self.active = False
        
        if self.redis:
            await self.redis.close()
        
        logger.info("Intelligent Fraud Detection system shutdown completed")


# Global fraud detection instance
fraud_detector = None

async def get_fraud_detector() -> IntelligentFraudDetection:
    """Get or create fraud detection instance"""
    global fraud_detector
    if not fraud_detector:
        fraud_detector = IntelligentFraudDetection(
            redis_url="redis://localhost:6379",
            database_url="postgresql://localhost:5432/nautilus"
        )
        await fraud_detector.initialize()
    return fraud_detector


async def analyze_for_fraud(transaction_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry point for fraud analysis
    """
    try:
        detector = await get_fraud_detector()
        alert = await detector.analyze_transaction(transaction_data)
        
        if alert:
            return {
                "fraud_detected": True,
                "alert_id": alert.alert_id,
                "fraud_type": alert.fraud_type.value,
                "severity": alert.severity.value,
                "confidence": alert.confidence_score,
                "risk_score": alert.risk_score,
                "recommended_actions": alert.recommended_actions
            }
        
        return {"fraud_detected": False}
        
    except Exception as e:
        logger.error(f"Fraud analysis failed: {e}")
        return {"error": str(e)}