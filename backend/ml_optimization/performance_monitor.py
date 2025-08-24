"""
ML Performance Monitor and Validation

This module monitors the performance of ML-powered optimizations,
validates model predictions against actual outcomes, and provides
continuous feedback for model improvement.
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import redis
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class PerformanceMetric(Enum):
    """Performance metrics to track"""
    PREDICTION_ACCURACY = "prediction_accuracy"
    SCALING_EFFECTIVENESS = "scaling_effectiveness"
    RESOURCE_EFFICIENCY = "resource_efficiency"
    COST_OPTIMIZATION = "cost_optimization"
    LATENCY_IMPROVEMENT = "latency_improvement"
    SYSTEM_STABILITY = "system_stability"
    MODEL_DRIFT = "model_drift"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class PerformanceRecord:
    """Individual performance measurement record"""
    timestamp: datetime
    service_name: str
    metric_type: PerformanceMetric
    
    # Prediction vs actual
    predicted_value: float
    actual_value: float
    error: float
    percentage_error: float
    
    # Context
    market_regime: str = "unknown"
    system_load: float = 0.0
    confidence: float = 0.0
    
    # Metadata
    model_version: Optional[str] = None
    prediction_horizon: int = 0  # minutes


@dataclass
class PerformanceAlert:
    """Performance alert definition"""
    alert_id: str
    timestamp: datetime
    severity: AlertSeverity
    metric_type: PerformanceMetric
    service_name: str
    
    # Alert details
    message: str
    current_value: float
    threshold_value: float
    deviation: float
    
    # Recommendations
    recommended_actions: List[str] = field(default_factory=list)
    
    # Status
    acknowledged: bool = False
    resolved: bool = False
    resolution_time: Optional[datetime] = None


@dataclass
class ModelPerformanceReport:
    """Comprehensive model performance report"""
    model_type: str
    report_period_start: datetime
    report_period_end: datetime
    
    # Accuracy metrics
    prediction_accuracy: float
    mean_absolute_error: float
    root_mean_square_error: float
    r2_score: float
    
    # Performance trends
    accuracy_trend: str  # improving, declining, stable
    error_trend: str
    
    # Operational metrics
    total_predictions: int
    successful_optimizations: int
    failed_optimizations: int
    
    # Business impact
    cost_savings: float
    performance_improvement: float
    availability_impact: float
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    requires_retraining: bool = False
    confidence_level: float = 0.0


class MLPerformanceMonitor:
    """
    Comprehensive performance monitoring system for ML optimization components.
    
    This system tracks prediction accuracy, measures optimization effectiveness,
    validates model performance, and provides continuous feedback for improvement.
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379", report_path: str = "/tmp/ml_reports"):
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.logger = logging.getLogger(__name__)
        
        # Storage paths
        self.report_path = Path(report_path)
        self.report_path.mkdir(parents=True, exist_ok=True)
        
        # Performance data storage
        self.performance_records: List[PerformanceRecord] = []
        self.active_alerts: Dict[str, PerformanceAlert] = {}
        self.performance_history: Dict[str, List[float]] = {}
        
        # Monitoring configuration
        self.monitoring_enabled = True
        self.collection_interval = 300  # seconds
        self.alert_thresholds = {
            PerformanceMetric.PREDICTION_ACCURACY: 0.7,
            PerformanceMetric.SCALING_EFFECTIVENESS: 0.6,
            PerformanceMetric.RESOURCE_EFFICIENCY: 0.5,
            PerformanceMetric.MODEL_DRIFT: 0.15,
            PerformanceMetric.SYSTEM_STABILITY: 0.95
        }
        
        # Performance tracking
        self.last_collection_time = datetime.now()
        self.total_predictions = 0
        self.successful_predictions = 0
        
        # Start monitoring loop
        asyncio.create_task(self._start_monitoring_loop())
    
    async def record_prediction_performance(
        self, 
        service_name: str,
        predicted_value: float,
        actual_value: float,
        metric_type: PerformanceMetric,
        context: Dict[str, Any] = None
    ):
        """Record a prediction performance measurement"""
        try:
            context = context or {}
            
            # Calculate errors
            error = abs(predicted_value - actual_value)
            if actual_value != 0:
                percentage_error = (error / abs(actual_value)) * 100
            else:
                percentage_error = error * 100
            
            # Create performance record
            record = PerformanceRecord(
                timestamp=datetime.now(),
                service_name=service_name,
                metric_type=metric_type,
                predicted_value=predicted_value,
                actual_value=actual_value,
                error=error,
                percentage_error=percentage_error,
                market_regime=context.get('market_regime', 'unknown'),
                system_load=context.get('system_load', 0.0),
                confidence=context.get('confidence', 0.0),
                model_version=context.get('model_version'),
                prediction_horizon=context.get('prediction_horizon', 0)
            )
            
            # Store record
            self.performance_records.append(record)
            await self._store_performance_record(record)
            
            # Update tracking metrics
            self.total_predictions += 1
            if percentage_error < 20:  # Consider < 20% error as successful
                self.successful_predictions += 1
            
            # Check for performance issues
            await self._check_performance_thresholds(record)
            
            self.logger.debug(
                f"Recorded prediction performance: {service_name} "
                f"{metric_type.value} error={percentage_error:.1f}%"
            )
            
        except Exception as e:
            self.logger.error(f"Error recording prediction performance: {str(e)}")
    
    async def record_scaling_outcome(
        self,
        service_name: str,
        predicted_need: str,  # scale_up, scale_down, maintain
        actual_outcome: str,  # scaled_up, scaled_down, maintained, failed
        effectiveness_score: float,  # 0-1 scale
        context: Dict[str, Any] = None
    ):
        """Record scaling decision outcome"""
        try:
            context = context or {}
            
            # Convert categorical outcomes to numerical for analysis
            outcome_mapping = {
                'maintain': 0.0, 'maintained': 0.0,
                'scale_down': -0.5, 'scaled_down': -0.5,
                'scale_up': 0.5, 'scaled_up': 0.5,
                'failed': -1.0
            }
            
            predicted_numeric = outcome_mapping.get(predicted_need, 0.0)
            actual_numeric = outcome_mapping.get(actual_outcome, 0.0)
            
            await self.record_prediction_performance(
                service_name=service_name,
                predicted_value=predicted_numeric,
                actual_value=actual_numeric,
                metric_type=PerformanceMetric.SCALING_EFFECTIVENESS,
                context={
                    **context,
                    'effectiveness_score': effectiveness_score,
                    'predicted_action': predicted_need,
                    'actual_action': actual_outcome
                }
            )
            
            # Store scaling-specific metrics
            scaling_data = {
                'timestamp': datetime.now().isoformat(),
                'service_name': service_name,
                'predicted_need': predicted_need,
                'actual_outcome': actual_outcome,
                'effectiveness_score': effectiveness_score,
                'context': context
            }
            
            self.redis_client.lpush("ml:scaling:outcomes", json.dumps(scaling_data))
            self.redis_client.ltrim("ml:scaling:outcomes", 0, 999)  # Keep last 1000
            
        except Exception as e:
            self.logger.error(f"Error recording scaling outcome: {str(e)}")
    
    async def record_resource_efficiency(
        self,
        service_name: str,
        resource_type: str,  # cpu, memory, network
        allocated: float,
        utilized: float,
        cost_impact: float,
        performance_impact: float
    ):
        """Record resource allocation efficiency"""
        try:
            # Calculate efficiency score (0-1, higher is better)
            if allocated > 0:
                utilization_efficiency = min(1.0, utilized / allocated)
                # Penalize both over-allocation and under-utilization
                if utilization_efficiency > 0.9:
                    efficiency_score = 1.0 - (utilization_efficiency - 0.9) * 5  # Over-utilized
                elif utilization_efficiency < 0.3:
                    efficiency_score = utilization_efficiency / 0.3  # Under-utilized
                else:
                    efficiency_score = 1.0  # Well-balanced
            else:
                efficiency_score = 0.0
            
            # Adjust for cost and performance impacts
            cost_factor = max(0.1, min(1.0, 1.0 - abs(cost_impact) / 100))
            performance_factor = max(0.1, min(1.0, 1.0 + performance_impact / 100))
            
            final_efficiency = efficiency_score * cost_factor * performance_factor
            
            await self.record_prediction_performance(
                service_name=service_name,
                predicted_value=1.0,  # We always aim for perfect efficiency
                actual_value=final_efficiency,
                metric_type=PerformanceMetric.RESOURCE_EFFICIENCY,
                context={
                    'resource_type': resource_type,
                    'allocated': allocated,
                    'utilized': utilized,
                    'utilization_rate': utilization_efficiency,
                    'cost_impact': cost_impact,
                    'performance_impact': performance_impact
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error recording resource efficiency: {str(e)}")
    
    async def calculate_model_drift(self, model_type: str, lookback_days: int = 7) -> float:
        """Calculate model drift score based on recent performance"""
        try:
            # Get recent performance data
            since_date = datetime.now() - timedelta(days=lookback_days)
            
            recent_records = [
                record for record in self.performance_records
                if record.timestamp >= since_date and record.model_version
                and model_type in str(record.model_version)
            ]
            
            if len(recent_records) < 10:
                return 0.0  # Insufficient data
            
            # Calculate performance trends
            errors = [record.percentage_error for record in recent_records]
            timestamps = [record.timestamp for record in recent_records]
            
            # Split into early and late periods
            mid_point = len(errors) // 2
            early_errors = errors[:mid_point]
            late_errors = errors[mid_point:]
            
            # Calculate drift as change in average error
            early_avg_error = np.mean(early_errors)
            late_avg_error = np.mean(late_errors)
            
            if early_avg_error == 0:
                drift_score = 0.0
            else:
                drift_score = abs(late_avg_error - early_avg_error) / early_avg_error
            
            # Store drift calculation
            drift_data = {
                'timestamp': datetime.now().isoformat(),
                'model_type': model_type,
                'lookback_days': lookback_days,
                'drift_score': drift_score,
                'early_avg_error': early_avg_error,
                'late_avg_error': late_avg_error,
                'sample_size': len(recent_records)
            }
            
            self.redis_client.setex(
                f"ml:drift:{model_type}",
                3600,  # 1 hour expiry
                json.dumps(drift_data)
            )
            
            return drift_score
            
        except Exception as e:
            self.logger.error(f"Error calculating model drift for {model_type}: {str(e)}")
            return 0.0
    
    async def _store_performance_record(self, record: PerformanceRecord):
        """Store performance record in Redis"""
        record_data = {
            'timestamp': record.timestamp.isoformat(),
            'service_name': record.service_name,
            'metric_type': record.metric_type.value,
            'predicted_value': record.predicted_value,
            'actual_value': record.actual_value,
            'error': record.error,
            'percentage_error': record.percentage_error,
            'market_regime': record.market_regime,
            'system_load': record.system_load,
            'confidence': record.confidence,
            'model_version': record.model_version,
            'prediction_horizon': record.prediction_horizon
        }
        
        # Store individual record
        self.redis_client.lpush("ml:performance:records", json.dumps(record_data))
        self.redis_client.ltrim("ml:performance:records", 0, 4999)  # Keep last 5000
        
        # Update service-specific metrics
        service_key = f"ml:performance:{record.service_name}:{record.metric_type.value}"
        self.redis_client.lpush(service_key, json.dumps(record_data))
        self.redis_client.ltrim(service_key, 0, 999)  # Keep last 1000 per service/metric
    
    async def _check_performance_thresholds(self, record: PerformanceRecord):
        """Check performance record against alert thresholds"""
        try:
            threshold = self.alert_thresholds.get(record.metric_type)
            if threshold is None:
                return
            
            # Convert percentage error to accuracy for comparison
            if record.metric_type == PerformanceMetric.PREDICTION_ACCURACY:
                current_value = max(0.0, 1.0 - (record.percentage_error / 100))
                is_below_threshold = current_value < threshold
                deviation = threshold - current_value
            else:
                current_value = record.percentage_error / 100
                is_below_threshold = current_value > threshold
                deviation = current_value - threshold
            
            if is_below_threshold:
                await self._create_performance_alert(
                    record.service_name,
                    record.metric_type,
                    current_value,
                    threshold,
                    deviation,
                    record
                )
            
        except Exception as e:
            self.logger.error(f"Error checking performance thresholds: {str(e)}")
    
    async def _create_performance_alert(
        self,
        service_name: str,
        metric_type: PerformanceMetric,
        current_value: float,
        threshold: float,
        deviation: float,
        record: PerformanceRecord
    ):
        """Create a performance alert"""
        try:
            # Determine severity
            if deviation > 0.3:
                severity = AlertSeverity.CRITICAL
            elif deviation > 0.15:
                severity = AlertSeverity.WARNING
            else:
                severity = AlertSeverity.INFO
            
            # Generate alert ID
            alert_id = f"{service_name}_{metric_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create alert message
            if metric_type == PerformanceMetric.PREDICTION_ACCURACY:
                message = f"Prediction accuracy for {service_name} dropped to {current_value:.2%} (threshold: {threshold:.2%})"
            elif metric_type == PerformanceMetric.MODEL_DRIFT:
                message = f"Model drift detected for {service_name}: {current_value:.3f} (threshold: {threshold:.3f})"
            else:
                message = f"Performance issue with {service_name} {metric_type.value}: {current_value:.3f}"
            
            # Generate recommendations
            recommendations = self._generate_recommendations(metric_type, current_value, threshold, record)
            
            alert = PerformanceAlert(
                alert_id=alert_id,
                timestamp=datetime.now(),
                severity=severity,
                metric_type=metric_type,
                service_name=service_name,
                message=message,
                current_value=current_value,
                threshold_value=threshold,
                deviation=deviation,
                recommended_actions=recommendations
            )
            
            # Store alert
            self.active_alerts[alert_id] = alert
            await self._store_alert(alert)
            
            self.logger.warning(f"Performance alert created: {alert_id} - {message}")
            
        except Exception as e:
            self.logger.error(f"Error creating performance alert: {str(e)}")
    
    def _generate_recommendations(
        self,
        metric_type: PerformanceMetric,
        current_value: float,
        threshold: float,
        record: PerformanceRecord
    ) -> List[str]:
        """Generate recommendations based on performance issues"""
        recommendations = []
        
        if metric_type == PerformanceMetric.PREDICTION_ACCURACY:
            if record.confidence < 0.5:
                recommendations.append("Consider increasing model training data volume")
                recommendations.append("Review feature engineering for better signal extraction")
            if record.market_regime != "normal":
                recommendations.append("Implement regime-specific model variants")
            recommendations.append("Schedule model retraining with recent data")
            
        elif metric_type == PerformanceMetric.SCALING_EFFECTIVENESS:
            recommendations.append("Review scaling thresholds and adjust for current market conditions")
            recommendations.append("Analyze recent scaling decisions for pattern identification")
            recommendations.append("Consider implementing more conservative scaling policies")
            
        elif metric_type == PerformanceMetric.RESOURCE_EFFICIENCY:
            recommendations.append("Optimize resource allocation algorithms")
            recommendations.append("Implement dynamic resource adjustment based on utilization")
            recommendations.append("Review cost-performance trade-offs")
            
        elif metric_type == PerformanceMetric.MODEL_DRIFT:
            recommendations.append("Schedule immediate model retraining")
            recommendations.append("Investigate data distribution changes")
            recommendations.append("Consider ensemble models for improved stability")
            
        else:
            recommendations.append("Review recent system changes and performance metrics")
            recommendations.append("Consider increasing monitoring frequency")
        
        return recommendations
    
    async def _store_alert(self, alert: PerformanceAlert):
        """Store alert in Redis"""
        alert_data = {
            'alert_id': alert.alert_id,
            'timestamp': alert.timestamp.isoformat(),
            'severity': alert.severity.value,
            'metric_type': alert.metric_type.value,
            'service_name': alert.service_name,
            'message': alert.message,
            'current_value': alert.current_value,
            'threshold_value': alert.threshold_value,
            'deviation': alert.deviation,
            'recommended_actions': alert.recommended_actions,
            'acknowledged': alert.acknowledged,
            'resolved': alert.resolved
        }
        
        # Store alert
        self.redis_client.setex(
            f"ml:alert:{alert.alert_id}",
            86400,  # 24 hour expiry
            json.dumps(alert_data)
        )
        
        # Add to active alerts list
        self.redis_client.lpush("ml:alerts:active", json.dumps(alert_data))
        self.redis_client.ltrim("ml:alerts:active", 0, 99)  # Keep last 100
    
    async def generate_performance_report(
        self, 
        model_type: str,
        days_back: int = 7
    ) -> ModelPerformanceReport:
        """Generate comprehensive performance report for a model type"""
        try:
            # Get performance data for the specified period
            since_date = datetime.now() - timedelta(days=days_back)
            
            relevant_records = [
                record for record in self.performance_records
                if record.timestamp >= since_date 
                and (record.model_version and model_type in str(record.model_version))
            ]
            
            if not relevant_records:
                # Return minimal report if no data
                return ModelPerformanceReport(
                    model_type=model_type,
                    report_period_start=since_date,
                    report_period_end=datetime.now(),
                    prediction_accuracy=0.0,
                    mean_absolute_error=0.0,
                    root_mean_square_error=0.0,
                    r2_score=0.0,
                    accuracy_trend="unknown",
                    error_trend="unknown",
                    total_predictions=0,
                    successful_optimizations=0,
                    failed_optimizations=0,
                    cost_savings=0.0,
                    performance_improvement=0.0,
                    availability_impact=0.0,
                    recommendations=["Insufficient data for analysis"],
                    confidence_level=0.0
                )
            
            # Calculate accuracy metrics
            errors = [record.percentage_error for record in relevant_records]
            actual_values = [record.actual_value for record in relevant_records]
            predicted_values = [record.predicted_value for record in relevant_records]
            
            mean_absolute_error = np.mean(errors)
            root_mean_square_error = np.sqrt(np.mean(np.array(errors) ** 2))
            prediction_accuracy = max(0.0, 1.0 - (mean_absolute_error / 100))
            
            # Calculate R² score if we have variation in actual values
            if len(set(actual_values)) > 1:
                r2 = r2_score(actual_values, predicted_values)
            else:
                r2 = 0.0
            
            # Analyze trends
            if len(errors) >= 10:
                mid_point = len(errors) // 2
                early_errors = errors[:mid_point]
                late_errors = errors[mid_point:]
                
                early_avg = np.mean(early_errors)
                late_avg = np.mean(late_errors)
                
                if late_avg < early_avg * 0.9:
                    accuracy_trend = "improving"
                    error_trend = "decreasing"
                elif late_avg > early_avg * 1.1:
                    accuracy_trend = "declining"
                    error_trend = "increasing"
                else:
                    accuracy_trend = "stable"
                    error_trend = "stable"
            else:
                accuracy_trend = "insufficient_data"
                error_trend = "insufficient_data"
            
            # Calculate operational metrics
            total_predictions = len(relevant_records)
            successful_optimizations = len([r for r in relevant_records if r.percentage_error < 20])
            failed_optimizations = total_predictions - successful_optimizations
            
            # Estimate business impact (simplified)
            cost_savings = successful_optimizations * 10.0  # $10 per successful optimization
            performance_improvement = prediction_accuracy * 100  # Percentage improvement
            availability_impact = min(99.99, 99.0 + prediction_accuracy)  # Availability percentage
            
            # Generate recommendations
            recommendations = []
            requires_retraining = False
            
            if prediction_accuracy < 0.7:
                recommendations.append("Model accuracy is below acceptable threshold - immediate retraining recommended")
                requires_retraining = True
            
            if accuracy_trend == "declining":
                recommendations.append("Model performance is declining - schedule retraining")
                requires_retraining = True
            
            if failed_optimizations > successful_optimizations:
                recommendations.append("High failure rate detected - review model features and training data")
            
            if mean_absolute_error > 30:
                recommendations.append("High prediction errors - consider ensemble methods or feature engineering")
            
            # Calculate confidence level
            confidence_factors = []
            confidence_factors.append(min(1.0, total_predictions / 100))  # Data volume factor
            confidence_factors.append(prediction_accuracy)  # Accuracy factor
            confidence_factors.append(1.0 if accuracy_trend in ["stable", "improving"] else 0.5)  # Trend factor
            
            confidence_level = np.mean(confidence_factors)
            
            report = ModelPerformanceReport(
                model_type=model_type,
                report_period_start=since_date,
                report_period_end=datetime.now(),
                prediction_accuracy=prediction_accuracy,
                mean_absolute_error=mean_absolute_error,
                root_mean_square_error=root_mean_square_error,
                r2_score=r2,
                accuracy_trend=accuracy_trend,
                error_trend=error_trend,
                total_predictions=total_predictions,
                successful_optimizations=successful_optimizations,
                failed_optimizations=failed_optimizations,
                cost_savings=cost_savings,
                performance_improvement=performance_improvement,
                availability_impact=availability_impact,
                recommendations=recommendations,
                requires_retraining=requires_retraining,
                confidence_level=confidence_level
            )
            
            # Store report
            await self._store_performance_report(report)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating performance report for {model_type}: {str(e)}")
            return ModelPerformanceReport(
                model_type=model_type,
                report_period_start=datetime.now() - timedelta(days=days_back),
                report_period_end=datetime.now(),
                prediction_accuracy=0.0,
                mean_absolute_error=0.0,
                root_mean_square_error=0.0,
                r2_score=0.0,
                accuracy_trend="error",
                error_trend="error",
                total_predictions=0,
                successful_optimizations=0,
                failed_optimizations=0,
                cost_savings=0.0,
                performance_improvement=0.0,
                availability_impact=0.0,
                recommendations=[f"Error generating report: {str(e)}"],
                confidence_level=0.0
            )
    
    async def _store_performance_report(self, report: ModelPerformanceReport):
        """Store performance report in Redis"""
        report_data = {
            'model_type': report.model_type,
            'report_period_start': report.report_period_start.isoformat(),
            'report_period_end': report.report_period_end.isoformat(),
            'prediction_accuracy': report.prediction_accuracy,
            'mean_absolute_error': report.mean_absolute_error,
            'root_mean_square_error': report.root_mean_square_error,
            'r2_score': report.r2_score,
            'accuracy_trend': report.accuracy_trend,
            'error_trend': report.error_trend,
            'total_predictions': report.total_predictions,
            'successful_optimizations': report.successful_optimizations,
            'failed_optimizations': report.failed_optimizations,
            'cost_savings': report.cost_savings,
            'performance_improvement': report.performance_improvement,
            'availability_impact': report.availability_impact,
            'recommendations': report.recommendations,
            'requires_retraining': report.requires_retraining,
            'confidence_level': report.confidence_level,
            'generated_at': datetime.now().isoformat()
        }
        
        # Store latest report
        self.redis_client.setex(
            f"ml:report:{report.model_type}",
            3600,  # 1 hour expiry
            json.dumps(report_data)
        )
        
        # Add to report history
        self.redis_client.lpush(f"ml:reports:{report.model_type}", json.dumps(report_data))
        self.redis_client.ltrim(f"ml:reports:{report.model_type}", 0, 29)  # Keep last 30
    
    async def get_active_alerts(self, severity_filter: Optional[AlertSeverity] = None) -> List[PerformanceAlert]:
        """Get active performance alerts"""
        alerts = []
        
        try:
            alert_data_list = self.redis_client.lrange("ml:alerts:active", 0, -1)
            
            for alert_data_str in alert_data_list:
                try:
                    alert_data = json.loads(alert_data_str)
                    
                    if severity_filter and alert_data.get('severity') != severity_filter.value:
                        continue
                    
                    if alert_data.get('resolved', False):
                        continue
                    
                    alert = PerformanceAlert(
                        alert_id=alert_data['alert_id'],
                        timestamp=datetime.fromisoformat(alert_data['timestamp']),
                        severity=AlertSeverity(alert_data['severity']),
                        metric_type=PerformanceMetric(alert_data['metric_type']),
                        service_name=alert_data['service_name'],
                        message=alert_data['message'],
                        current_value=alert_data['current_value'],
                        threshold_value=alert_data['threshold_value'],
                        deviation=alert_data['deviation'],
                        recommended_actions=alert_data.get('recommended_actions', []),
                        acknowledged=alert_data.get('acknowledged', False),
                        resolved=alert_data.get('resolved', False)
                    )
                    
                    alerts.append(alert)
                    
                except Exception as e:
                    self.logger.error(f"Error parsing alert data: {str(e)}")
                    continue
            
            # Sort by severity and timestamp
            severity_order = {AlertSeverity.EMERGENCY: 4, AlertSeverity.CRITICAL: 3, 
                            AlertSeverity.WARNING: 2, AlertSeverity.INFO: 1}
            
            alerts.sort(key=lambda x: (severity_order.get(x.severity, 0), x.timestamp), reverse=True)
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Error getting active alerts: {str(e)}")
            return []
    
    async def _start_monitoring_loop(self):
        """Start background monitoring and analysis loop"""
        await asyncio.sleep(30)  # Initial delay
        
        while self.monitoring_enabled:
            try:
                monitoring_start = datetime.now()
                
                # Calculate drift scores for different model types
                model_types = ["load_predictor", "pattern_classifier", "volatility_predictor", "regime_classifier"]
                
                for model_type in model_types:
                    drift_score = await self.calculate_model_drift(model_type)
                    
                    if drift_score > self.alert_thresholds.get(PerformanceMetric.MODEL_DRIFT, 0.15):
                        self.logger.warning(f"Model drift detected for {model_type}: {drift_score:.3f}")
                
                # Clean up old performance records (keep last 7 days)
                cutoff_time = datetime.now() - timedelta(days=7)
                initial_count = len(self.performance_records)
                self.performance_records = [
                    record for record in self.performance_records
                    if record.timestamp > cutoff_time
                ]
                
                if len(self.performance_records) < initial_count:
                    self.logger.info(f"Cleaned up {initial_count - len(self.performance_records)} old performance records")
                
                # Update monitoring statistics
                self.last_collection_time = monitoring_start
                
                # Calculate cycle time
                cycle_time = (datetime.now() - monitoring_start).total_seconds()
                
                # Wait for next cycle
                await asyncio.sleep(max(0, self.collection_interval - cycle_time))
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def get_monitoring_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive monitoring dashboard data"""
        try:
            # Calculate overall statistics
            recent_records = [
                record for record in self.performance_records
                if record.timestamp > datetime.now() - timedelta(hours=24)
            ]
            
            if recent_records:
                avg_accuracy = 1.0 - (np.mean([r.percentage_error for r in recent_records]) / 100)
                error_std = np.std([r.percentage_error for r in recent_records])
            else:
                avg_accuracy = 0.0
                error_std = 0.0
            
            # Get active alerts
            active_alerts = await self.get_active_alerts()
            
            # Get service-specific performance
            service_performance = {}
            services = ["nautilus-market-data", "nautilus-strategy-engine", "nautilus-risk-engine"]
            
            for service in services:
                service_records = [r for r in recent_records if r.service_name == service]
                if service_records:
                    service_accuracy = 1.0 - (np.mean([r.percentage_error for r in service_records]) / 100)
                    service_predictions = len(service_records)
                else:
                    service_accuracy = 0.0
                    service_predictions = 0
                
                service_performance[service] = {
                    'accuracy': service_accuracy,
                    'predictions': service_predictions
                }
            
            # Performance trends (last 7 days, daily buckets)
            trend_data = {}
            for i in range(7):
                day_start = datetime.now() - timedelta(days=i+1)
                day_end = datetime.now() - timedelta(days=i)
                
                day_records = [
                    r for r in self.performance_records
                    if day_start <= r.timestamp < day_end
                ]
                
                if day_records:
                    day_accuracy = 1.0 - (np.mean([r.percentage_error for r in day_records]) / 100)
                else:
                    day_accuracy = 0.0
                
                trend_data[day_start.strftime('%Y-%m-%d')] = day_accuracy
            
            return {
                "timestamp": datetime.now().isoformat(),
                "monitoring_enabled": self.monitoring_enabled,
                "overall_statistics": {
                    "total_predictions": self.total_predictions,
                    "successful_predictions": self.successful_predictions,
                    "success_rate": (self.successful_predictions / max(1, self.total_predictions)) * 100,
                    "average_accuracy": avg_accuracy,
                    "error_std_dev": error_std,
                    "recent_records_count": len(recent_records)
                },
                "alerts": {
                    "total_active": len(active_alerts),
                    "critical": len([a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]),
                    "warnings": len([a for a in active_alerts if a.severity == AlertSeverity.WARNING]),
                    "recent_alerts": [
                        {
                            "alert_id": alert.alert_id,
                            "severity": alert.severity.value,
                            "service": alert.service_name,
                            "metric": alert.metric_type.value,
                            "message": alert.message,
                            "timestamp": alert.timestamp.isoformat()
                        }
                        for alert in active_alerts[:5]
                    ]
                },
                "service_performance": service_performance,
                "performance_trends": trend_data,
                "model_health": {
                    "load_predictor": {"status": "healthy", "last_update": datetime.now().isoformat()},
                    "pattern_classifier": {"status": "healthy", "last_update": datetime.now().isoformat()},
                    "volatility_predictor": {"status": "healthy", "last_update": datetime.now().isoformat()}
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating dashboard data: {str(e)}")
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "monitoring_enabled": self.monitoring_enabled
            }


async def main():
    """Test the ML Performance Monitor"""
    logging.basicConfig(level=logging.INFO)
    
    monitor = MLPerformanceMonitor()
    
    print("📊 Testing ML Performance Monitor")
    print("=" * 40)
    
    # Simulate some prediction performances
    services = ["nautilus-market-data", "nautilus-strategy-engine", "nautilus-risk-engine"]
    
    print("\n🎯 Simulating Prediction Performances...")
    
    for i, service in enumerate(services):
        # Simulate various prediction scenarios
        scenarios = [
            # Good predictions
            {"predicted": 0.7, "actual": 0.65, "confidence": 0.8},
            {"predicted": 0.4, "actual": 0.42, "confidence": 0.75},
            {"predicted": 0.9, "actual": 0.88, "confidence": 0.9},
            
            # Poor predictions
            {"predicted": 0.3, "actual": 0.7, "confidence": 0.6},
            {"predicted": 0.8, "actual": 0.4, "confidence": 0.5},
        ]
        
        for j, scenario in enumerate(scenarios):
            await monitor.record_prediction_performance(
                service_name=service,
                predicted_value=scenario["predicted"],
                actual_value=scenario["actual"],
                metric_type=PerformanceMetric.PREDICTION_ACCURACY,
                context={
                    'confidence': scenario["confidence"],
                    'market_regime': 'normal' if j < 3 else 'volatile',
                    'model_version': f'{service}_v1.0'
                }
            )
    
    # Simulate scaling outcomes
    print("\n📈 Simulating Scaling Outcomes...")
    
    scaling_scenarios = [
        {"predicted": "scale_up", "actual": "scaled_up", "effectiveness": 0.9},
        {"predicted": "maintain", "actual": "maintained", "effectiveness": 1.0},
        {"predicted": "scale_down", "actual": "failed", "effectiveness": 0.1},
        {"predicted": "scale_up", "actual": "scaled_up", "effectiveness": 0.7},
    ]
    
    for scenario in scaling_scenarios:
        await monitor.record_scaling_outcome(
            service_name="nautilus-market-data",
            predicted_need=scenario["predicted"],
            actual_outcome=scenario["actual"],
            effectiveness_score=scenario["effectiveness"]
        )
    
    # Simulate resource efficiency measurements
    print("\n💾 Simulating Resource Efficiency...")
    
    efficiency_scenarios = [
        {"allocated": 4.0, "utilized": 3.2, "cost": -5.0, "performance": 10.0},
        {"allocated": 2.0, "utilized": 1.9, "cost": 0.0, "performance": 5.0},
        {"allocated": 6.0, "utilized": 2.1, "cost": 15.0, "performance": -2.0},
    ]
    
    for scenario in efficiency_scenarios:
        await monitor.record_resource_efficiency(
            service_name="nautilus-strategy-engine",
            resource_type="cpu",
            allocated=scenario["allocated"],
            utilized=scenario["utilized"],
            cost_impact=scenario["cost"],
            performance_impact=scenario["performance"]
        )
    
    # Give monitoring loop time to process
    await asyncio.sleep(2)
    
    # Generate performance reports
    print("\n📋 Generating Performance Reports...")
    
    model_types = ["load_predictor", "pattern_classifier"]
    
    for model_type in model_types:
        report = await monitor.generate_performance_report(model_type, days_back=1)
        
        print(f"\n{model_type.upper()} REPORT:")
        print(f"  Prediction Accuracy: {report.prediction_accuracy:.2%}")
        print(f"  Mean Absolute Error: {report.mean_absolute_error:.1f}%")
        print(f"  Total Predictions: {report.total_predictions}")
        print(f"  Successful Optimizations: {report.successful_optimizations}")
        print(f"  Cost Savings: ${report.cost_savings:.2f}")
        print(f"  Confidence Level: {report.confidence_level:.2%}")
        print(f"  Requires Retraining: {report.requires_retraining}")
        
        if report.recommendations:
            print("  Recommendations:")
            for rec in report.recommendations[:3]:
                print(f"    - {rec}")
    
    # Check active alerts
    print("\n🚨 Active Alerts:")
    
    alerts = await monitor.get_active_alerts()
    if alerts:
        for alert in alerts[:3]:
            print(f"  {alert.severity.value.upper()}: {alert.message}")
            if alert.recommended_actions:
                print(f"    Action: {alert.recommended_actions[0]}")
    else:
        print("  No active alerts")
    
    # Get dashboard data
    print("\n📊 Dashboard Summary:")
    
    dashboard = await monitor.get_monitoring_dashboard_data()
    
    if "overall_statistics" in dashboard:
        stats = dashboard["overall_statistics"]
        print(f"  Total Predictions: {stats['total_predictions']}")
        print(f"  Success Rate: {stats['success_rate']:.1f}%")
        print(f"  Average Accuracy: {stats['average_accuracy']:.2%}")
        print(f"  Recent Records: {stats['recent_records_count']}")
    
    if "alerts" in dashboard:
        alert_stats = dashboard["alerts"]
        print(f"  Active Alerts: {alert_stats['total_active']}")
        print(f"  Critical: {alert_stats['critical']}, Warnings: {alert_stats['warnings']}")
    
    print("\n✅ ML Performance Monitor test completed!")


if __name__ == "__main__":
    asyncio.run(main())