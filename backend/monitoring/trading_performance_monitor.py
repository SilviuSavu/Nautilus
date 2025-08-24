"""
Trading Performance Monitor for Nautilus Trading Platform
Comprehensive monitoring for trading operations with M4 Max optimization focus:

- Order execution latency monitoring
- Market data processing throughput
- Risk assessment performance tracking
- ML inference performance monitoring
- Strategy execution metrics
- WebSocket message throughput
- Database query performance
- Real-time trading metrics
"""

import asyncio
import json
import logging
import time
import statistics
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import redis
import psycopg2
from psycopg2.extras import RealDictCursor
from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram, Summary
from prometheus_client.exposition import generate_latest

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PerformanceLevel(Enum):
    EXCELLENT = "excellent"    # < 1ms
    GOOD = "good"             # 1-10ms
    ACCEPTABLE = "acceptable"  # 10-50ms
    POOR = "poor"             # 50-100ms
    CRITICAL = "critical"     # > 100ms

@dataclass
class TradingLatencyMetrics:
    """Trading operation latency metrics"""
    operation_type: str
    average_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    max_latency_ms: float
    min_latency_ms: float
    total_operations: int
    failed_operations: int
    success_rate: float
    timestamp: datetime
    performance_level: PerformanceLevel

@dataclass
class ThroughputMetrics:
    """System throughput metrics"""
    component: str
    messages_per_second: float
    bytes_per_second: float
    peak_throughput: float
    average_throughput: float
    queue_depth: int
    processing_time_ms: float
    backlog_size: int
    timestamp: datetime

@dataclass
class RiskMetrics:
    """Risk assessment performance metrics"""
    risk_checks_per_second: float
    average_risk_calc_time_ms: float
    breach_detection_time_ms: float
    portfolio_risk_calc_time_ms: float
    var_calculation_time_ms: float
    stress_test_time_ms: float
    ml_risk_prediction_time_ms: float
    active_risk_monitors: int
    failed_risk_checks: int
    timestamp: datetime

@dataclass
class MLInferenceMetrics:
    """Machine learning inference performance metrics"""
    model_name: str
    inference_time_ms: float
    preprocessing_time_ms: float
    neural_engine_utilization: float
    batch_size: int
    predictions_per_second: float
    model_accuracy: Optional[float]
    cache_hit_rate: float
    memory_usage_mb: float
    timestamp: datetime

@dataclass
class MarketDataMetrics:
    """Market data processing metrics"""
    venue: str
    symbols_tracked: int
    updates_per_second: float
    latency_ms: float
    missed_updates: int
    data_quality_score: float
    compression_ratio: float
    storage_size_mb: float
    timestamp: datetime

class TradingPerformanceMonitor:
    """Comprehensive trading performance monitoring system"""
    
    def __init__(self, redis_host: str = "redis", redis_port: int = 6379,
                 db_host: str = "postgres", db_port: int = 5432,
                 db_name: str = "nautilus", db_user: str = "nautilus", db_password: str = "nautilus123"):
        
        # Redis connection
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        
        # Database connection
        self.db_config = {
            'host': db_host,
            'port': db_port,
            'database': db_name,
            'user': db_user,
            'password': db_password
        }
        
        # Prometheus metrics setup
        self.registry = CollectorRegistry()
        self._setup_prometheus_metrics()
        
        # Performance tracking
        self.latency_samples = {}
        self.throughput_samples = {}
        self.monitoring = False
        
        # Performance thresholds (in milliseconds)
        self.performance_thresholds = {
            'order_execution': {'excellent': 1, 'good': 10, 'acceptable': 50, 'poor': 100},
            'market_data': {'excellent': 0.5, 'good': 2, 'acceptable': 10, 'poor': 50},
            'risk_calculation': {'excellent': 5, 'good': 20, 'acceptable': 100, 'poor': 500},
            'ml_inference': {'excellent': 10, 'good': 50, 'acceptable': 200, 'poor': 1000}
        }
        
        logger.info("Trading Performance Monitor initialized")
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics for trading performance monitoring"""
        
        # Latency Metrics
        self.order_execution_latency = Histogram(
            'nautilus_order_execution_latency_seconds',
            'Order execution latency in seconds',
            ['order_type', 'venue'],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
            registry=self.registry
        )
        
        self.market_data_latency = Histogram(
            'nautilus_market_data_latency_seconds',
            'Market data processing latency in seconds',
            ['venue', 'data_type'],
            buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
            registry=self.registry
        )
        
        self.risk_calculation_latency = Histogram(
            'nautilus_risk_calculation_latency_seconds',
            'Risk calculation latency in seconds',
            ['risk_type'],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
            registry=self.registry
        )
        
        self.ml_inference_latency = Histogram(
            'nautilus_ml_inference_latency_seconds',
            'ML inference latency in seconds',
            ['model_name'],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0],
            registry=self.registry
        )
        
        # Throughput Metrics
        self.message_throughput = Gauge(
            'nautilus_message_throughput_per_second',
            'Messages processed per second',
            ['component'],
            registry=self.registry
        )
        
        self.data_throughput = Gauge(
            'nautilus_data_throughput_bytes_per_second',
            'Data throughput in bytes per second',
            ['component'],
            registry=self.registry
        )
        
        # Trading Metrics
        self.active_orders = Gauge(
            'nautilus_active_orders',
            'Number of active orders',
            ['venue'],
            registry=self.registry
        )
        
        self.order_success_rate = Gauge(
            'nautilus_order_success_rate',
            'Order success rate percentage',
            ['venue', 'order_type'],
            registry=self.registry
        )
        
        self.trade_volume = Counter(
            'nautilus_trade_volume_total',
            'Total trade volume',
            ['venue', 'symbol'],
            registry=self.registry
        )
        
        # Risk Metrics
        self.risk_checks_per_second = Gauge(
            'nautilus_risk_checks_per_second',
            'Risk checks performed per second',
            registry=self.registry
        )
        
        self.risk_breach_alerts = Counter(
            'nautilus_risk_breach_alerts_total',
            'Total risk breach alerts',
            ['breach_type'],
            registry=self.registry
        )
        
        self.portfolio_value_at_risk = Gauge(
            'nautilus_portfolio_var',
            'Portfolio Value at Risk',
            ['confidence_level'],
            registry=self.registry
        )
        
        # ML Performance Metrics
        self.ml_predictions_per_second = Gauge(
            'nautilus_ml_predictions_per_second',
            'ML predictions per second',
            ['model_name'],
            registry=self.registry
        )
        
        self.neural_engine_utilization = Gauge(
            'nautilus_neural_engine_utilization_percent',
            'Neural Engine utilization percentage',
            registry=self.registry
        )
        
        self.model_accuracy = Gauge(
            'nautilus_ml_model_accuracy',
            'ML model accuracy score',
            ['model_name'],
            registry=self.registry
        )
        
        # System Performance
        self.database_query_time = Histogram(
            'nautilus_database_query_duration_seconds',
            'Database query execution time',
            ['query_type'],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
            registry=self.registry
        )
        
        self.websocket_connections = Gauge(
            'nautilus_websocket_connections',
            'Active WebSocket connections',
            registry=self.registry
        )
        
        self.websocket_message_rate = Gauge(
            'nautilus_websocket_messages_per_second',
            'WebSocket messages per second',
            ['message_type'],
            registry=self.registry
        )
        
        # Performance Summary
        self.overall_performance_score = Gauge(
            'nautilus_overall_performance_score',
            'Overall system performance score (0-100)',
            registry=self.registry
        )
        
        # Error Tracking
        self.trading_errors = Counter(
            'nautilus_trading_errors_total',
            'Total trading errors',
            ['error_type', 'component'],
            registry=self.registry
        )
    
    def _get_database_connection(self):
        """Get database connection with error handling"""
        try:
            return psycopg2.connect(**self.db_config)
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            return None
    
    def _classify_performance(self, latency_ms: float, operation_type: str) -> PerformanceLevel:
        """Classify performance level based on latency"""
        thresholds = self.performance_thresholds.get(operation_type, self.performance_thresholds['order_execution'])
        
        if latency_ms <= thresholds['excellent']:
            return PerformanceLevel.EXCELLENT
        elif latency_ms <= thresholds['good']:
            return PerformanceLevel.GOOD
        elif latency_ms <= thresholds['acceptable']:
            return PerformanceLevel.ACCEPTABLE
        elif latency_ms <= thresholds['poor']:
            return PerformanceLevel.POOR
        else:
            return PerformanceLevel.CRITICAL
    
    async def collect_order_execution_metrics(self) -> List[TradingLatencyMetrics]:
        """Collect order execution latency metrics"""
        metrics_list = []
        
        try:
            conn = self._get_database_connection()
            if not conn:
                return metrics_list
            
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Query recent order execution times
                query = """
                SELECT 
                    order_type,
                    venue,
                    EXTRACT(EPOCH FROM (filled_time - submitted_time)) * 1000 as latency_ms,
                    status
                FROM orders 
                WHERE submitted_time >= NOW() - INTERVAL '5 minutes'
                AND filled_time IS NOT NULL
                ORDER BY submitted_time DESC
                LIMIT 1000
                """
                
                cursor.execute(query)
                orders = cursor.fetchall()
                
                if orders:
                    # Group by order type and venue
                    order_groups = {}
                    for order in orders:
                        key = f"{order['order_type']}_{order['venue']}"
                        if key not in order_groups:
                            order_groups[key] = []
                        order_groups[key].append({
                            'latency_ms': float(order['latency_ms']),
                            'success': order['status'] == 'FILLED'
                        })
                    
                    # Calculate metrics for each group
                    for key, order_list in order_groups.items():
                        if len(order_list) >= 5:  # Minimum sample size
                            latencies = [o['latency_ms'] for o in order_list]
                            successes = sum(1 for o in order_list if o['success'])
                            
                            metrics = TradingLatencyMetrics(
                                operation_type=key,
                                average_latency_ms=statistics.mean(latencies),
                                p50_latency_ms=statistics.median(latencies),
                                p95_latency_ms=statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies),
                                p99_latency_ms=statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 100 else max(latencies),
                                max_latency_ms=max(latencies),
                                min_latency_ms=min(latencies),
                                total_operations=len(order_list),
                                failed_operations=len(order_list) - successes,
                                success_rate=(successes / len(order_list)) * 100,
                                timestamp=datetime.now(),
                                performance_level=self._classify_performance(statistics.mean(latencies), 'order_execution')
                            )
                            
                            metrics_list.append(metrics)
                            
                            # Update Prometheus metrics
                            order_type, venue = key.split('_', 1)
                            self.order_success_rate.labels(venue=venue, order_type=order_type).set(metrics.success_rate)
                
            conn.close()
            
        except Exception as e:
            logger.error(f"Error collecting order execution metrics: {e}")
        
        return metrics_list
    
    async def collect_market_data_metrics(self) -> List[MarketDataMetrics]:
        """Collect market data processing metrics"""
        metrics_list = []
        
        try:
            # Get market data metrics from Redis
            venues = ['IBKR', 'YFINANCE', 'ALPHA_VANTAGE']
            
            for venue in venues:
                # Get recent market data updates
                updates_key = f"market_data:{venue.lower()}:updates"
                recent_updates = self.redis_client.zrevrange(updates_key, 0, 100, withscores=True)
                
                if recent_updates:
                    # Calculate throughput
                    now = time.time()
                    minute_ago = now - 60
                    recent_count = len([u for u in recent_updates if u[1] > minute_ago])
                    updates_per_second = recent_count / 60.0
                    
                    # Get latency info
                    latency_key = f"market_data:{venue.lower()}:latency"
                    avg_latency = float(self.redis_client.get(latency_key) or 0)
                    
                    # Get symbol count
                    symbols_key = f"market_data:{venue.lower()}:symbols"
                    symbols_count = self.redis_client.scard(symbols_key) or 0
                    
                    metrics = MarketDataMetrics(
                        venue=venue,
                        symbols_tracked=symbols_count,
                        updates_per_second=updates_per_second,
                        latency_ms=avg_latency,
                        missed_updates=0,  # Would need specific tracking
                        data_quality_score=95.0,  # Would need quality assessment
                        compression_ratio=3.2,    # Estimate
                        storage_size_mb=len(recent_updates) * 0.001,  # Rough estimate
                        timestamp=datetime.now()
                    )
                    
                    metrics_list.append(metrics)
                    
                    # Update Prometheus metrics
                    self.market_data_latency.labels(venue=venue, data_type='quote').observe(avg_latency / 1000)
                    
        except Exception as e:
            logger.error(f"Error collecting market data metrics: {e}")
        
        return metrics_list
    
    async def collect_risk_metrics(self) -> RiskMetrics:
        """Collect risk assessment performance metrics"""
        try:
            # Get risk calculation metrics from Redis
            risk_checks_count = int(self.redis_client.get("risk:checks:count") or 0)
            risk_calc_times = self.redis_client.lrange("risk:calc_times", 0, 99)
            
            avg_risk_calc_time = 0.0
            if risk_calc_times:
                times = [float(t) for t in risk_calc_times]
                avg_risk_calc_time = statistics.mean(times)
            
            # Get specific risk calculation times
            breach_detection_time = float(self.redis_client.get("risk:breach_detection_time") or 0)
            portfolio_risk_time = float(self.redis_client.get("risk:portfolio_calc_time") or 0)
            var_calc_time = float(self.redis_client.get("risk:var_calc_time") or 0)
            stress_test_time = float(self.redis_client.get("risk:stress_test_time") or 0)
            ml_risk_time = float(self.redis_client.get("risk:ml_prediction_time") or 0)
            
            active_monitors = int(self.redis_client.get("risk:active_monitors") or 0)
            failed_checks = int(self.redis_client.get("risk:failed_checks") or 0)
            
            # Calculate checks per second (last minute)
            current_time = time.time()
            minute_ago_checks = int(self.redis_client.get(f"risk:checks:{int(current_time - 60)}") or 0)
            current_checks = int(self.redis_client.get(f"risk:checks:{int(current_time)}") or 0)
            checks_per_second = max(0, current_checks - minute_ago_checks) / 60.0
            
            metrics = RiskMetrics(
                risk_checks_per_second=checks_per_second,
                average_risk_calc_time_ms=avg_risk_calc_time,
                breach_detection_time_ms=breach_detection_time,
                portfolio_risk_calc_time_ms=portfolio_risk_time,
                var_calculation_time_ms=var_calc_time,
                stress_test_time_ms=stress_test_time,
                ml_risk_prediction_time_ms=ml_risk_time,
                active_risk_monitors=active_monitors,
                failed_risk_checks=failed_checks,
                timestamp=datetime.now()
            )
            
            # Update Prometheus metrics
            self.risk_checks_per_second.set(checks_per_second)
            self.risk_calculation_latency.labels(risk_type='portfolio').observe(portfolio_risk_time / 1000)
            self.risk_calculation_latency.labels(risk_type='var').observe(var_calc_time / 1000)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting risk metrics: {e}")
            return RiskMetrics(
                risk_checks_per_second=0.0,
                average_risk_calc_time_ms=0.0,
                breach_detection_time_ms=0.0,
                portfolio_risk_calc_time_ms=0.0,
                var_calculation_time_ms=0.0,
                stress_test_time_ms=0.0,
                ml_risk_prediction_time_ms=0.0,
                active_risk_monitors=0,
                failed_risk_checks=0,
                timestamp=datetime.now()
            )
    
    async def collect_ml_inference_metrics(self) -> List[MLInferenceMetrics]:
        """Collect ML inference performance metrics"""
        metrics_list = []
        
        try:
            # Get ML model performance metrics
            models = ['risk_predictor', 'price_forecaster', 'sentiment_analyzer', 'portfolio_optimizer']
            
            for model_name in models:
                # Get inference times
                inference_times_key = f"ml:{model_name}:inference_times"
                inference_times = self.redis_client.lrange(inference_times_key, 0, 99)
                
                if inference_times:
                    times = [float(t) for t in inference_times]
                    avg_inference_time = statistics.mean(times)
                    
                    # Get other metrics
                    preprocessing_time = float(self.redis_client.get(f"ml:{model_name}:preprocessing_time") or 0)
                    neural_engine_util = float(self.redis_client.get(f"ml:{model_name}:neural_engine_util") or 0)
                    batch_size = int(self.redis_client.get(f"ml:{model_name}:batch_size") or 1)
                    predictions_count = int(self.redis_client.get(f"ml:{model_name}:predictions_count") or 0)
                    accuracy = float(self.redis_client.get(f"ml:{model_name}:accuracy") or 0.0)
                    cache_hits = int(self.redis_client.get(f"ml:{model_name}:cache_hits") or 0)
                    cache_total = int(self.redis_client.get(f"ml:{model_name}:cache_total") or 1)
                    memory_usage = float(self.redis_client.get(f"ml:{model_name}:memory_usage") or 0)
                    
                    # Calculate predictions per second (last minute)
                    predictions_per_second = predictions_count / 60.0  # Rough estimate
                    cache_hit_rate = (cache_hits / cache_total) * 100 if cache_total > 0 else 0.0
                    
                    metrics = MLInferenceMetrics(
                        model_name=model_name,
                        inference_time_ms=avg_inference_time,
                        preprocessing_time_ms=preprocessing_time,
                        neural_engine_utilization=neural_engine_util,
                        batch_size=batch_size,
                        predictions_per_second=predictions_per_second,
                        model_accuracy=accuracy if accuracy > 0 else None,
                        cache_hit_rate=cache_hit_rate,
                        memory_usage_mb=memory_usage,
                        timestamp=datetime.now()
                    )
                    
                    metrics_list.append(metrics)
                    
                    # Update Prometheus metrics
                    self.ml_inference_latency.labels(model_name=model_name).observe(avg_inference_time / 1000)
                    self.ml_predictions_per_second.labels(model_name=model_name).set(predictions_per_second)
                    if accuracy > 0:
                        self.model_accuracy.labels(model_name=model_name).set(accuracy)
                    
        except Exception as e:
            logger.error(f"Error collecting ML inference metrics: {e}")
        
        return metrics_list
    
    async def collect_throughput_metrics(self) -> List[ThroughputMetrics]:
        """Collect system throughput metrics"""
        metrics_list = []
        
        try:
            components = [
                'websocket_engine', 'market_data_engine', 'risk_engine', 
                'analytics_engine', 'factor_engine', 'ml_engine'
            ]
            
            for component in components:
                # Get message throughput
                msg_count_key = f"throughput:{component}:messages"
                msg_counts = self.redis_client.lrange(msg_count_key, 0, 59)  # Last minute
                
                if msg_counts:
                    total_messages = sum(int(count) for count in msg_counts)
                    messages_per_second = total_messages / 60.0
                    
                    # Get bytes throughput
                    bytes_count_key = f"throughput:{component}:bytes"
                    bytes_counts = self.redis_client.lrange(bytes_count_key, 0, 59)
                    total_bytes = sum(int(count) for count in bytes_counts) if bytes_counts else 0
                    bytes_per_second = total_bytes / 60.0
                    
                    # Get queue depth and processing time
                    queue_depth = int(self.redis_client.get(f"throughput:{component}:queue_depth") or 0)
                    processing_time = float(self.redis_client.get(f"throughput:{component}:processing_time") or 0)
                    backlog_size = int(self.redis_client.get(f"throughput:{component}:backlog") or 0)
                    
                    # Calculate peak and average throughput
                    peak_throughput = max([int(count) for count in msg_counts]) if msg_counts else 0
                    average_throughput = messages_per_second
                    
                    metrics = ThroughputMetrics(
                        component=component,
                        messages_per_second=messages_per_second,
                        bytes_per_second=bytes_per_second,
                        peak_throughput=peak_throughput,
                        average_throughput=average_throughput,
                        queue_depth=queue_depth,
                        processing_time_ms=processing_time,
                        backlog_size=backlog_size,
                        timestamp=datetime.now()
                    )
                    
                    metrics_list.append(metrics)
                    
                    # Update Prometheus metrics
                    self.message_throughput.labels(component=component).set(messages_per_second)
                    self.data_throughput.labels(component=component).set(bytes_per_second)
                    
        except Exception as e:
            logger.error(f"Error collecting throughput metrics: {e}")
        
        return metrics_list
    
    async def calculate_overall_performance_score(self, 
                                                latency_metrics: List[TradingLatencyMetrics],
                                                risk_metrics: RiskMetrics,
                                                ml_metrics: List[MLInferenceMetrics],
                                                throughput_metrics: List[ThroughputMetrics]) -> float:
        """Calculate overall system performance score (0-100)"""
        try:
            score_components = []
            
            # Latency score (40% weight)
            if latency_metrics:
                latency_scores = []
                for metrics in latency_metrics:
                    if metrics.performance_level == PerformanceLevel.EXCELLENT:
                        latency_scores.append(100)
                    elif metrics.performance_level == PerformanceLevel.GOOD:
                        latency_scores.append(80)
                    elif metrics.performance_level == PerformanceLevel.ACCEPTABLE:
                        latency_scores.append(60)
                    elif metrics.performance_level == PerformanceLevel.POOR:
                        latency_scores.append(40)
                    else:  # CRITICAL
                        latency_scores.append(20)
                
                avg_latency_score = statistics.mean(latency_scores) if latency_scores else 50
                score_components.append(('latency', avg_latency_score, 0.4))
            
            # Risk performance score (20% weight)
            risk_score = 80  # Default good score
            if risk_metrics.risk_checks_per_second > 10:
                risk_score = 100
            elif risk_metrics.risk_checks_per_second > 5:
                risk_score = 80
            elif risk_metrics.risk_checks_per_second > 1:
                risk_score = 60
            else:
                risk_score = 40
            
            score_components.append(('risk', risk_score, 0.2))
            
            # ML performance score (20% weight)
            if ml_metrics:
                ml_scores = []
                for metrics in ml_metrics:
                    if metrics.inference_time_ms < 10:
                        ml_scores.append(100)
                    elif metrics.inference_time_ms < 50:
                        ml_scores.append(80)
                    elif metrics.inference_time_ms < 200:
                        ml_scores.append(60)
                    else:
                        ml_scores.append(40)
                
                avg_ml_score = statistics.mean(ml_scores) if ml_scores else 50
                score_components.append(('ml', avg_ml_score, 0.2))
            
            # Throughput score (20% weight)
            if throughput_metrics:
                throughput_scores = []
                for metrics in throughput_metrics:
                    if metrics.messages_per_second > 1000:
                        throughput_scores.append(100)
                    elif metrics.messages_per_second > 500:
                        throughput_scores.append(80)
                    elif metrics.messages_per_second > 100:
                        throughput_scores.append(60)
                    else:
                        throughput_scores.append(40)
                
                avg_throughput_score = statistics.mean(throughput_scores) if throughput_scores else 50
                score_components.append(('throughput', avg_throughput_score, 0.2))
            
            # Calculate weighted average
            total_score = sum(score * weight for _, score, weight in score_components)
            total_weight = sum(weight for _, _, weight in score_components)
            
            overall_score = total_score / total_weight if total_weight > 0 else 50
            
            # Update Prometheus metric
            self.overall_performance_score.set(overall_score)
            
            return overall_score
            
        except Exception as e:
            logger.error(f"Error calculating performance score: {e}")
            return 50.0
    
    def _store_metrics_redis(self, all_metrics: Dict[str, Any]):
        """Store all metrics in Redis"""
        try:
            timestamp = int(time.time())
            
            # Store current snapshot
            self.redis_client.set(
                "trading:performance:current",
                json.dumps(all_metrics, default=str),
                ex=300  # 5 minute expiry
            )
            
            # Store historical data
            self.redis_client.lpush("trading:performance:history", json.dumps(all_metrics, default=str))
            self.redis_client.ltrim("trading:performance:history", 0, 1439)  # Keep 24 hours (1 min intervals)
            
            # Store specific metric time series
            if 'overall_score' in all_metrics:
                self.redis_client.zadd("trading:performance:scores", {timestamp: all_metrics['overall_score']})
                
            # Cleanup old time series data
            cutoff_time = timestamp - (24 * 60 * 60)
            self.redis_client.zremrangebyscore("trading:performance:scores", 0, cutoff_time)
            
        except Exception as e:
            logger.error(f"Error storing metrics in Redis: {e}")
    
    async def collect_all_trading_metrics(self) -> Dict[str, Any]:
        """Collect all trading performance metrics"""
        logger.debug("Collecting comprehensive trading performance metrics...")
        
        try:
            # Collect all metrics concurrently
            latency_task = asyncio.create_task(self.collect_order_execution_metrics())
            market_data_task = asyncio.create_task(self.collect_market_data_metrics())
            risk_task = asyncio.create_task(self.collect_risk_metrics())
            ml_task = asyncio.create_task(self.collect_ml_inference_metrics())
            throughput_task = asyncio.create_task(self.collect_throughput_metrics())
            
            latency_metrics, market_data_metrics, risk_metrics, ml_metrics, throughput_metrics = await asyncio.gather(
                latency_task, market_data_task, risk_task, ml_task, throughput_task
            )
            
            # Calculate overall performance score
            overall_score = await self.calculate_overall_performance_score(
                latency_metrics, risk_metrics, ml_metrics, throughput_metrics
            )
            
            # Compile all metrics
            all_metrics = {
                'timestamp': datetime.now().isoformat(),
                'latency_metrics': [asdict(m) for m in latency_metrics],
                'market_data_metrics': [asdict(m) for m in market_data_metrics],
                'risk_metrics': asdict(risk_metrics),
                'ml_metrics': [asdict(m) for m in ml_metrics],
                'throughput_metrics': [asdict(m) for m in throughput_metrics],
                'overall_score': overall_score
            }
            
            # Store in Redis
            self._store_metrics_redis(all_metrics)
            
            logger.debug(f"Trading performance metrics collected - Overall score: {overall_score:.1f}")
            
            return all_metrics
            
        except Exception as e:
            logger.error(f"Error collecting trading metrics: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    async def start_monitoring(self, interval: float = 60.0):
        """Start continuous trading performance monitoring"""
        logger.info(f"Starting trading performance monitoring (interval: {interval}s)")
        self.monitoring = True
        
        while self.monitoring:
            try:
                await self.collect_all_trading_metrics()
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                logger.info("Trading performance monitoring cancelled")
                break
            except Exception as e:
                logger.error(f"Error in trading monitoring loop: {e}")
                await asyncio.sleep(interval)
    
    def stop_monitoring(self):
        """Stop monitoring"""
        logger.info("Stopping trading performance monitoring")
        self.monitoring = False
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics in text format"""
        return generate_latest(self.registry).decode('utf-8')
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get current performance summary"""
        try:
            current_metrics = self.redis_client.get("trading:performance:current")
            if current_metrics:
                return json.loads(current_metrics)
            return {'error': 'No current metrics available'}
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {'error': str(e)}

# Example usage
if __name__ == "__main__":
    async def main():
        monitor = TradingPerformanceMonitor()
        
        # Test collection
        metrics = await monitor.collect_all_trading_metrics()
        
        print(f"\n=== Trading Performance Summary ===")
        print(f"Overall Score: {metrics.get('overall_score', 'N/A'):.1f}")
        print(f"Latency Metrics: {len(metrics.get('latency_metrics', []))}")
        print(f"Market Data Metrics: {len(metrics.get('market_data_metrics', []))}")
        print(f"ML Metrics: {len(metrics.get('ml_metrics', []))}")
        print(f"Throughput Metrics: {len(metrics.get('throughput_metrics', []))}")
        
        # Start monitoring for 120 seconds
        print(f"\n=== Starting continuous monitoring for 2 minutes ===")
        monitoring_task = asyncio.create_task(monitor.start_monitoring(interval=30.0))
        await asyncio.sleep(120)
        monitor.stop_monitoring()
        monitoring_task.cancel()
        
        print("Trading Performance Monitor test completed.")
    
    asyncio.run(main())