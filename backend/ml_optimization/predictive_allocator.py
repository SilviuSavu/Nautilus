"""
Predictive Resource Allocation Engine

This module implements intelligent resource allocation that anticipates
trading events, market conditions, and system demands before they occur.
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import redis
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error
import joblib
from scipy.optimize import minimize
import yfinance as yf


class ResourceType(Enum):
    """Types of resources to allocate"""
    CPU = "cpu"
    MEMORY = "memory"
    NETWORK = "network"
    STORAGE = "storage"
    GPU = "gpu"


class AllocationStrategy(Enum):
    """Resource allocation strategies"""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    EVENT_DRIVEN = "event_driven"
    ML_OPTIMIZED = "ml_optimized"


@dataclass
class ResourceDemand:
    """Predicted resource demand for a specific service"""
    service_name: str
    timestamp: datetime
    prediction_horizon_minutes: int
    
    # Resource predictions
    cpu_demand: float  # CPU cores
    memory_demand: float  # GB
    network_demand: float  # Mbps
    storage_demand: float  # GB
    
    # Confidence metrics
    prediction_confidence: float
    demand_volatility: float
    
    # Context
    triggering_events: List[str] = field(default_factory=list)
    market_regime: str = "normal"
    risk_level: str = "low"


@dataclass
class ResourceAllocation:
    """Resource allocation decision"""
    service_name: str
    resource_type: ResourceType
    current_allocation: float
    recommended_allocation: float
    allocation_change: float
    priority: int  # 1-10, higher is more important
    cost_impact: float
    performance_impact: float
    justification: str


@dataclass
class AllocationPlan:
    """Complete resource allocation plan"""
    plan_id: str
    timestamp: datetime
    strategy: AllocationStrategy
    total_cost: float
    expected_performance_gain: float
    
    allocations: List[ResourceAllocation] = field(default_factory=list)
    risk_assessment: Dict[str, float] = field(default_factory=dict)
    contingency_plans: List[str] = field(default_factory=list)


class PredictiveResourceAllocator:
    """
    Intelligent resource allocator that predicts future demands and
    optimizes allocations across the entire trading infrastructure.
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.logger = logging.getLogger(__name__)
        
        # ML Models for different prediction horizons
        self.short_term_predictor = None  # 1-15 minutes
        self.medium_term_predictor = None  # 15-60 minutes  
        self.long_term_predictor = None  # 1-24 hours
        
        # Feature scalers
        self.feature_scaler = StandardScaler()
        self.target_scaler = MinMaxScaler()
        
        # Resource optimization
        self.resource_optimizer = None
        
        # Configuration
        self.services = [
            "nautilus-market-data",
            "nautilus-strategy-engine", 
            "nautilus-risk-engine",
            "nautilus-order-manager",
            "nautilus-position-keeper"
        ]
        
        self.resource_costs = {
            ResourceType.CPU: 0.05,  # Cost per CPU hour
            ResourceType.MEMORY: 0.01,  # Cost per GB hour
            ResourceType.NETWORK: 0.001,  # Cost per Mbps hour
            ResourceType.STORAGE: 0.0001,  # Cost per GB hour
            ResourceType.GPU: 0.50  # Cost per GPU hour
        }
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models for different prediction horizons"""
        # Short-term predictor (high frequency, low latency)
        self.short_term_predictor = RandomForestRegressor(
            n_estimators=50,
            max_depth=8,
            min_samples_split=5,
            random_state=42
        )
        
        # Medium-term predictor (balanced accuracy and speed)
        self.medium_term_predictor = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            random_state=42
        )
        
        # Long-term predictor (high accuracy)
        self.long_term_predictor = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=10,
            subsample=0.9,
            random_state=42
        )
    
    async def collect_market_indicators(self) -> Dict[str, float]:
        """Collect market indicators that affect resource demands"""
        try:
            indicators = {}
            
            # Get VIX (volatility indicator)
            vix = yf.Ticker("^VIX")
            vix_data = vix.history(period="2d")
            if not vix_data.empty:
                indicators['vix_current'] = float(vix_data['Close'].iloc[-1])
                indicators['vix_change'] = float(vix_data['Close'].iloc[-1] - vix_data['Close'].iloc[-2])
            
            # Get major indices for trend analysis
            spy = yf.Ticker("SPY")
            spy_data = spy.history(period="5d")
            if not spy_data.empty:
                indicators['spy_return_1d'] = float((spy_data['Close'].iloc[-1] / spy_data['Close'].iloc[-2] - 1) * 100)
                indicators['spy_volatility'] = float(spy_data['Close'].pct_change().std() * 100)
                
            # Trading volume indicators
            if not spy_data.empty:
                recent_volume = spy_data['Volume'].rolling(window=3).mean().iloc[-1]
                avg_volume = spy_data['Volume'].mean()
                indicators['volume_ratio'] = float(recent_volume / avg_volume)
            
            # Get from Redis if available (real-time updates)
            cached_indicators = self.redis_client.get("market:indicators:realtime")
            if cached_indicators:
                cached_data = json.loads(cached_indicators)
                indicators.update(cached_data)
            
            # Add default values for missing indicators
            default_indicators = {
                'vix_current': 20.0,
                'vix_change': 0.0,
                'spy_return_1d': 0.0,
                'spy_volatility': 1.0,
                'volume_ratio': 1.0,
                'interest_rates': 5.0,
                'dollar_index': 100.0,
                'commodity_momentum': 0.0
            }
            
            for key, default_value in default_indicators.items():
                if key not in indicators:
                    indicators[key] = default_value
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error collecting market indicators: {str(e)}")
            return {
                'vix_current': 20.0,
                'vix_change': 0.0, 
                'spy_return_1d': 0.0,
                'spy_volatility': 1.0,
                'volume_ratio': 1.0,
                'interest_rates': 5.0,
                'dollar_index': 100.0,
                'commodity_momentum': 0.0
            }
    
    async def get_scheduled_events(self) -> List[Dict[str, Any]]:
        """Get scheduled events that will affect resource demands"""
        try:
            events = []
            now = datetime.now()
            
            # Market events from Redis
            events_str = self.redis_client.get("market:events:scheduled")
            if events_str:
                cached_events = json.loads(events_str)
                events.extend(cached_events)
            
            # Add known recurring events
            market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
            
            # Check if market events are approaching
            if now < market_open:
                minutes_to_open = (market_open - now).total_seconds() / 60
                if minutes_to_open <= 60:  # Within 1 hour
                    events.append({
                        'event_type': 'market_open',
                        'minutes_until': minutes_to_open,
                        'impact_level': 'high',
                        'affected_services': self.services
                    })
            
            if now < market_close:
                minutes_to_close = (market_close - now).total_seconds() / 60
                if minutes_to_close <= 60:  # Within 1 hour
                    events.append({
                        'event_type': 'market_close',
                        'minutes_until': minutes_to_close,
                        'impact_level': 'high',
                        'affected_services': self.services
                    })
            
            # Economic data releases (simulated)
            economic_events = [
                {'time': '08:30', 'event': 'Employment Report', 'impact': 'very_high'},
                {'time': '10:00', 'event': 'ISM Manufacturing', 'impact': 'high'},
                {'time': '14:00', 'event': 'FOMC Minutes', 'impact': 'very_high'}
            ]
            
            for event in economic_events:
                event_time = now.replace(hour=int(event['time'].split(':')[0]), 
                                       minute=int(event['time'].split(':')[1]), 
                                       second=0, microsecond=0)
                if now < event_time:
                    minutes_until = (event_time - now).total_seconds() / 60
                    if minutes_until <= 180:  # Within 3 hours
                        events.append({
                            'event_type': 'economic_data',
                            'name': event['event'],
                            'minutes_until': minutes_until,
                            'impact_level': event['impact'],
                            'affected_services': ['nautilus-market-data', 'nautilus-risk-engine']
                        })
            
            return events
            
        except Exception as e:
            self.logger.error(f"Error getting scheduled events: {str(e)}")
            return []
    
    async def collect_historical_metrics(self, service_name: str, hours: int = 24) -> pd.DataFrame:
        """Collect historical resource usage metrics"""
        try:
            # Get historical data from Redis
            metrics_key = f"metrics:history:{service_name}"
            historical_data = self.redis_client.lrange(metrics_key, 0, hours * 12)  # 5-minute intervals
            
            if historical_data:
                data_points = []
                for record in historical_data:
                    try:
                        data = json.loads(record)
                        data_points.append(data)
                    except:
                        continue
                
                if data_points:
                    df = pd.DataFrame(data_points)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    return df.sort_values('timestamp')
            
            # Generate synthetic historical data if not available
            return self._generate_synthetic_history(service_name, hours)
            
        except Exception as e:
            self.logger.error(f"Error collecting historical metrics: {str(e)}")
            return self._generate_synthetic_history(service_name, hours)
    
    def _generate_synthetic_history(self, service_name: str, hours: int) -> pd.DataFrame:
        """Generate synthetic historical data for testing"""
        timestamps = pd.date_range(
            start=datetime.now() - timedelta(hours=hours),
            end=datetime.now(),
            freq='5T'  # 5-minute intervals
        )
        
        data = []
        for i, ts in enumerate(timestamps):
            # Simulate realistic patterns
            hour = ts.hour
            is_market_hours = 9 <= hour <= 16
            
            # Base resource usage with daily patterns
            base_cpu = 30 + 40 * is_market_hours + 10 * np.sin(hour * np.pi / 12)
            base_memory = 40 + 30 * is_market_hours + 5 * np.sin(hour * np.pi / 12)
            base_network = 50 + 100 * is_market_hours + 20 * np.sin(hour * np.pi / 12)
            
            # Add noise and volatility spikes
            volatility = np.random.uniform(0.8, 1.2)
            if np.random.random() < 0.05:  # 5% chance of spike
                volatility *= 2
            
            data.append({
                'timestamp': ts.isoformat(),
                'cpu_usage': max(10, base_cpu * volatility + np.random.normal(0, 5)),
                'memory_usage': max(10, base_memory * volatility + np.random.normal(0, 3)),
                'network_usage': max(5, base_network * volatility + np.random.normal(0, 10)),
                'storage_usage': np.random.uniform(10, 50),
                'request_rate': np.random.uniform(10, 500) * is_market_hours + np.random.uniform(5, 50),
                'response_time': np.random.uniform(10, 200),
                'error_rate': np.random.uniform(0, 0.05)
            })
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    
    def _create_features(
        self, 
        historical_df: pd.DataFrame, 
        market_indicators: Dict[str, float],
        scheduled_events: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Create features for ML prediction"""
        
        if historical_df.empty:
            return np.array([]).reshape(0, -1)
        
        # Time-based features
        latest_time = historical_df['timestamp'].iloc[-1]
        features = [
            latest_time.hour,
            latest_time.weekday(),
            float(9 <= latest_time.hour <= 16 and latest_time.weekday() < 5),  # is_market_hours
        ]
        
        # Historical usage features (rolling averages)
        if len(historical_df) >= 12:  # At least 1 hour of data
            recent_data = historical_df.tail(12)  # Last hour
            features.extend([
                recent_data['cpu_usage'].mean(),
                recent_data['memory_usage'].mean(),
                recent_data['network_usage'].mean(),
                recent_data['request_rate'].mean(),
                recent_data['response_time'].mean(),
                recent_data['cpu_usage'].std(),
                recent_data['memory_usage'].std(),
                recent_data['network_usage'].std()
            ])
        else:
            # Pad with zeros if insufficient data
            features.extend([0.0] * 8)
        
        # Trend features
        if len(historical_df) >= 24:  # At least 2 hours
            old_avg = historical_df.head(12)['cpu_usage'].mean()
            new_avg = historical_df.tail(12)['cpu_usage'].mean()
            cpu_trend = (new_avg - old_avg) / old_avg if old_avg > 0 else 0
            
            old_mem = historical_df.head(12)['memory_usage'].mean()
            new_mem = historical_df.tail(12)['memory_usage'].mean()
            memory_trend = (new_mem - old_mem) / old_mem if old_mem > 0 else 0
            
            features.extend([cpu_trend, memory_trend])
        else:
            features.extend([0.0, 0.0])
        
        # Market indicators
        features.extend([
            market_indicators.get('vix_current', 20.0),
            market_indicators.get('vix_change', 0.0),
            market_indicators.get('spy_return_1d', 0.0),
            market_indicators.get('spy_volatility', 1.0),
            market_indicators.get('volume_ratio', 1.0)
        ])
        
        # Event features
        high_impact_events_1h = len([e for e in scheduled_events 
                                    if e.get('minutes_until', 999) <= 60 
                                    and e.get('impact_level') in ['high', 'very_high']])
        
        medium_impact_events_1h = len([e for e in scheduled_events 
                                      if e.get('minutes_until', 999) <= 60 
                                      and e.get('impact_level') == 'medium'])
        
        features.extend([
            float(high_impact_events_1h),
            float(medium_impact_events_1h)
        ])
        
        return np.array(features).reshape(1, -1)
    
    async def predict_demand(self, service_name: str, horizon_minutes: int) -> ResourceDemand:
        """Predict resource demand for a specific service and time horizon"""
        try:
            # Collect input data
            historical_df = await self.collect_historical_metrics(service_name)
            market_indicators = await self.collect_market_indicators()
            scheduled_events = await self.get_scheduled_events()
            
            # Create features
            features = self._create_features(historical_df, market_indicators, scheduled_events)
            
            if features.size == 0:
                # Return default prediction if no data
                return ResourceDemand(
                    service_name=service_name,
                    timestamp=datetime.now(),
                    prediction_horizon_minutes=horizon_minutes,
                    cpu_demand=2.0,
                    memory_demand=4.0,
                    network_demand=100.0,
                    storage_demand=50.0,
                    prediction_confidence=0.5,
                    demand_volatility=0.2,
                    triggering_events=[],
                    market_regime="normal",
                    risk_level="low"
                )
            
            # Select appropriate model based on horizon
            if horizon_minutes <= 15:
                model = self.short_term_predictor
            elif horizon_minutes <= 60:
                model = self.medium_term_predictor
            else:
                model = self.long_term_predictor
            
            # Ensure model is trained
            if model is None:
                await self._train_models_if_needed(service_name)
            
            # Make predictions (simplified - in reality would predict each resource type)
            try:
                if hasattr(self.feature_scaler, 'mean_'):
                    features_scaled = self.feature_scaler.transform(features)
                else:
                    features_scaled = features
                
                # Predict normalized demand (0-1)
                demand_prediction = model.predict(features_scaled)[0] if model else 0.5
            except:
                demand_prediction = 0.5
            
            # Convert to actual resource demands
            base_demands = self._get_base_demands(service_name)
            
            cpu_demand = base_demands['cpu'] * (1 + demand_prediction)
            memory_demand = base_demands['memory'] * (1 + demand_prediction)
            network_demand = base_demands['network'] * (1 + demand_prediction)
            storage_demand = base_demands['storage']
            
            # Calculate confidence
            confidence = self._calculate_prediction_confidence(
                historical_df, market_indicators, scheduled_events
            )
            
            # Assess volatility
            volatility = self._assess_demand_volatility(historical_df, market_indicators)
            
            # Identify triggering events
            triggering_events = [
                event['event_type'] for event in scheduled_events
                if event.get('minutes_until', 999) <= horizon_minutes
                and service_name in event.get('affected_services', [])
            ]
            
            # Determine market regime
            market_regime = self._classify_market_regime(market_indicators)
            
            # Assess risk level
            risk_level = self._assess_risk_level(volatility, market_indicators, scheduled_events)
            
            return ResourceDemand(
                service_name=service_name,
                timestamp=datetime.now(),
                prediction_horizon_minutes=horizon_minutes,
                cpu_demand=cpu_demand,
                memory_demand=memory_demand,
                network_demand=network_demand,
                storage_demand=storage_demand,
                prediction_confidence=confidence,
                demand_volatility=volatility,
                triggering_events=triggering_events,
                market_regime=market_regime,
                risk_level=risk_level
            )
            
        except Exception as e:
            self.logger.error(f"Error predicting demand for {service_name}: {str(e)}")
            # Return fallback prediction
            return ResourceDemand(
                service_name=service_name,
                timestamp=datetime.now(),
                prediction_horizon_minutes=horizon_minutes,
                cpu_demand=2.0,
                memory_demand=4.0,
                network_demand=100.0,
                storage_demand=50.0,
                prediction_confidence=0.3,
                demand_volatility=0.3,
                triggering_events=[],
                market_regime="uncertain",
                risk_level="medium"
            )
    
    def _get_base_demands(self, service_name: str) -> Dict[str, float]:
        """Get baseline resource demands for each service"""
        base_demands = {
            "nautilus-market-data": {
                'cpu': 2.0, 'memory': 4.0, 'network': 200.0, 'storage': 100.0
            },
            "nautilus-strategy-engine": {
                'cpu': 3.0, 'memory': 6.0, 'network': 50.0, 'storage': 20.0
            },
            "nautilus-risk-engine": {
                'cpu': 2.5, 'memory': 3.0, 'network': 30.0, 'storage': 10.0
            },
            "nautilus-order-manager": {
                'cpu': 2.0, 'memory': 2.0, 'network': 100.0, 'storage': 5.0
            },
            "nautilus-position-keeper": {
                'cpu': 1.5, 'memory': 2.0, 'network': 20.0, 'storage': 10.0
            }
        }
        
        return base_demands.get(service_name, {
            'cpu': 2.0, 'memory': 4.0, 'network': 100.0, 'storage': 50.0
        })
    
    def _calculate_prediction_confidence(
        self, 
        historical_df: pd.DataFrame, 
        market_indicators: Dict[str, float],
        scheduled_events: List[Dict[str, Any]]
    ) -> float:
        """Calculate confidence in the demand prediction"""
        confidence = 0.8  # Base confidence
        
        # Adjust based on data availability
        if len(historical_df) < 12:  # Less than 1 hour of data
            confidence -= 0.3
        elif len(historical_df) < 48:  # Less than 4 hours of data
            confidence -= 0.1
        
        # Adjust based on market volatility
        vix = market_indicators.get('vix_current', 20)
        if vix > 30:  # High volatility
            confidence -= 0.2
        elif vix < 15:  # Low volatility
            confidence += 0.1
        
        # Adjust based on upcoming events
        high_impact_events = len([e for e in scheduled_events 
                                 if e.get('impact_level') in ['high', 'very_high']])
        if high_impact_events > 0:
            confidence -= 0.1 * high_impact_events
        
        return max(0.1, min(1.0, confidence))
    
    def _assess_demand_volatility(
        self, 
        historical_df: pd.DataFrame, 
        market_indicators: Dict[str, float]
    ) -> float:
        """Assess expected volatility in resource demand"""
        base_volatility = 0.1
        
        # Historical volatility
        if len(historical_df) >= 12:
            cpu_std = historical_df.tail(12)['cpu_usage'].std()
            base_volatility += cpu_std / 100  # Normalize
        
        # Market-based volatility
        vix = market_indicators.get('vix_current', 20)
        market_volatility = (vix - 15) / 100  # Normalize around VIX 15
        
        total_volatility = base_volatility + market_volatility
        return max(0.01, min(1.0, total_volatility))
    
    def _classify_market_regime(self, market_indicators: Dict[str, float]) -> str:
        """Classify current market regime"""
        vix = market_indicators.get('vix_current', 20)
        spy_return = market_indicators.get('spy_return_1d', 0)
        volume_ratio = market_indicators.get('volume_ratio', 1.0)
        
        if vix > 35:
            return "crisis"
        elif vix > 25:
            return "volatile"
        elif spy_return > 2:
            return "bullish"
        elif spy_return < -2:
            return "bearish"
        elif volume_ratio > 1.5:
            return "active"
        else:
            return "normal"
    
    def _assess_risk_level(
        self, 
        volatility: float, 
        market_indicators: Dict[str, float],
        scheduled_events: List[Dict[str, Any]]
    ) -> str:
        """Assess risk level for resource allocation"""
        risk_score = 0
        
        # Volatility component
        if volatility > 0.3:
            risk_score += 2
        elif volatility > 0.2:
            risk_score += 1
        
        # Market component
        vix = market_indicators.get('vix_current', 20)
        if vix > 30:
            risk_score += 2
        elif vix > 25:
            risk_score += 1
        
        # Event component
        high_impact_events = len([e for e in scheduled_events 
                                 if e.get('impact_level') in ['high', 'very_high']])
        risk_score += min(2, high_impact_events)
        
        if risk_score >= 4:
            return "high"
        elif risk_score >= 2:
            return "medium"
        else:
            return "low"
    
    async def _train_models_if_needed(self, service_name: str):
        """Train models if they haven't been trained yet"""
        if self.short_term_predictor is None:
            self._initialize_models()
        
        # Generate synthetic training data
        n_samples = 5000
        X_train, y_train = self._generate_training_data(service_name, n_samples)
        
        # Fit scaler
        self.feature_scaler.fit(X_train)
        X_scaled = self.feature_scaler.transform(X_train)
        
        # Train models
        self.short_term_predictor.fit(X_scaled, y_train)
        self.medium_term_predictor.fit(X_scaled, y_train)
        self.long_term_predictor.fit(X_scaled, y_train)
    
    def _generate_training_data(self, service_name: str, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic training data for model training"""
        np.random.seed(42)
        
        X = []
        y = []
        
        for _ in range(n_samples):
            # Generate realistic feature combinations
            hour = np.random.randint(0, 24)
            day_of_week = np.random.randint(0, 7)
            is_market_hours = float(9 <= hour <= 16 and day_of_week < 5)
            
            # Historical usage patterns
            base_cpu = 30 + 40 * is_market_hours + 10 * np.sin(hour * np.pi / 12)
            base_memory = 40 + 30 * is_market_hours + 5 * np.sin(hour * np.pi / 12)
            base_network = 50 + 100 * is_market_hours + 20 * np.sin(hour * np.pi / 12)
            
            # Market indicators
            vix = np.random.uniform(10, 40)
            vix_change = np.random.normal(0, 2)
            spy_return = np.random.normal(0, 1.5)
            spy_volatility = np.random.uniform(0.5, 3.0)
            volume_ratio = np.random.uniform(0.5, 2.0)
            
            # Events
            high_impact_events = np.random.poisson(0.1)
            medium_impact_events = np.random.poisson(0.3)
            
            features = [
                hour, day_of_week, is_market_hours,
                base_cpu, base_memory, base_network,
                np.random.uniform(10, 200),  # request_rate
                np.random.uniform(10, 500),  # response_time
                np.random.uniform(5, 25),    # cpu_std
                np.random.uniform(3, 15),    # memory_std
                np.random.uniform(10, 50),   # network_std
                np.random.normal(0, 0.2),    # cpu_trend
                np.random.normal(0, 0.2),    # memory_trend
                vix, vix_change, spy_return, spy_volatility, volume_ratio,
                high_impact_events, medium_impact_events
            ]
            
            X.append(features)
            
            # Target: normalized demand multiplier (0-1)
            market_stress = min(1.0, vix / 30)
            event_impact = min(0.5, (high_impact_events * 0.3 + medium_impact_events * 0.1))
            time_impact = is_market_hours * 0.3
            
            demand = market_stress * 0.4 + event_impact + time_impact + np.random.normal(0, 0.1)
            y.append(max(0, min(1, demand)))
        
        return np.array(X), np.array(y)
    
    async def create_allocation_plan(
        self, 
        strategy: AllocationStrategy = AllocationStrategy.ML_OPTIMIZED
    ) -> AllocationPlan:
        """Create optimized resource allocation plan for all services"""
        try:
            plan_id = f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            allocations = []
            total_cost = 0.0
            performance_gain = 0.0
            
            # Predict demands for all services
            predictions = {}
            for service in self.services:
                # Get predictions for different horizons
                short_term = await self.predict_demand(service, 15)
                medium_term = await self.predict_demand(service, 60)
                predictions[service] = {
                    'short': short_term,
                    'medium': medium_term
                }
            
            # Create allocations for each service and resource type
            for service in self.services:
                short_pred = predictions[service]['short']
                medium_pred = predictions[service]['medium']
                
                # CPU allocation
                current_cpu = self._get_current_allocation(service, ResourceType.CPU)
                recommended_cpu = self._optimize_cpu_allocation(short_pred, medium_pred, strategy)
                
                if abs(recommended_cpu - current_cpu) > 0.1:  # Significant change
                    cpu_allocation = ResourceAllocation(
                        service_name=service,
                        resource_type=ResourceType.CPU,
                        current_allocation=current_cpu,
                        recommended_allocation=recommended_cpu,
                        allocation_change=recommended_cpu - current_cpu,
                        priority=self._calculate_priority(service, short_pred.risk_level),
                        cost_impact=(recommended_cpu - current_cpu) * self.resource_costs[ResourceType.CPU],
                        performance_impact=self._estimate_performance_impact(service, 'cpu', recommended_cpu - current_cpu),
                        justification=self._generate_justification(service, 'cpu', short_pred, medium_pred)
                    )
                    allocations.append(cpu_allocation)
                    total_cost += cpu_allocation.cost_impact
                    performance_gain += cpu_allocation.performance_impact
                
                # Memory allocation
                current_memory = self._get_current_allocation(service, ResourceType.MEMORY)
                recommended_memory = self._optimize_memory_allocation(short_pred, medium_pred, strategy)
                
                if abs(recommended_memory - current_memory) > 0.5:  # Significant change
                    memory_allocation = ResourceAllocation(
                        service_name=service,
                        resource_type=ResourceType.MEMORY,
                        current_allocation=current_memory,
                        recommended_allocation=recommended_memory,
                        allocation_change=recommended_memory - current_memory,
                        priority=self._calculate_priority(service, short_pred.risk_level),
                        cost_impact=(recommended_memory - current_memory) * self.resource_costs[ResourceType.MEMORY],
                        performance_impact=self._estimate_performance_impact(service, 'memory', recommended_memory - current_memory),
                        justification=self._generate_justification(service, 'memory', short_pred, medium_pred)
                    )
                    allocations.append(memory_allocation)
                    total_cost += memory_allocation.cost_impact
                    performance_gain += memory_allocation.performance_impact
            
            # Risk assessment
            risk_assessment = self._assess_plan_risks(predictions, allocations)
            
            # Contingency plans
            contingency_plans = self._create_contingency_plans(predictions, allocations)
            
            return AllocationPlan(
                plan_id=plan_id,
                timestamp=datetime.now(),
                strategy=strategy,
                total_cost=total_cost,
                expected_performance_gain=performance_gain,
                allocations=allocations,
                risk_assessment=risk_assessment,
                contingency_plans=contingency_plans
            )
            
        except Exception as e:
            self.logger.error(f"Error creating allocation plan: {str(e)}")
            return AllocationPlan(
                plan_id="fallback_plan",
                timestamp=datetime.now(),
                strategy=strategy,
                total_cost=0.0,
                expected_performance_gain=0.0,
                allocations=[],
                risk_assessment={'error': 1.0},
                contingency_plans=['Review allocation logic']
            )
    
    def _get_current_allocation(self, service: str, resource_type: ResourceType) -> float:
        """Get current resource allocation for a service"""
        # In a real system, this would query Kubernetes or monitoring system
        defaults = {
            "nautilus-market-data": {ResourceType.CPU: 2.0, ResourceType.MEMORY: 4.0},
            "nautilus-strategy-engine": {ResourceType.CPU: 3.0, ResourceType.MEMORY: 6.0},
            "nautilus-risk-engine": {ResourceType.CPU: 2.5, ResourceType.MEMORY: 3.0},
            "nautilus-order-manager": {ResourceType.CPU: 2.0, ResourceType.MEMORY: 2.0},
            "nautilus-position-keeper": {ResourceType.CPU: 1.5, ResourceType.MEMORY: 2.0}
        }
        
        return defaults.get(service, {}).get(resource_type, 2.0)
    
    def _optimize_cpu_allocation(
        self, 
        short_pred: ResourceDemand, 
        medium_pred: ResourceDemand, 
        strategy: AllocationStrategy
    ) -> float:
        """Optimize CPU allocation based on predictions and strategy"""
        base_demand = (short_pred.cpu_demand + medium_pred.cpu_demand) / 2
        
        if strategy == AllocationStrategy.CONSERVATIVE:
            return base_demand * 1.5  # 50% buffer
        elif strategy == AllocationStrategy.BALANCED:
            return base_demand * 1.2  # 20% buffer
        elif strategy == AllocationStrategy.AGGRESSIVE:
            return base_demand * 1.0  # No buffer
        elif strategy == AllocationStrategy.EVENT_DRIVEN:
            event_multiplier = 1.3 if short_pred.triggering_events else 1.1
            return base_demand * event_multiplier
        else:  # ML_OPTIMIZED
            confidence_multiplier = 1.0 + (1.0 - short_pred.prediction_confidence) * 0.5
            volatility_multiplier = 1.0 + short_pred.demand_volatility * 0.3
            return base_demand * confidence_multiplier * volatility_multiplier
    
    def _optimize_memory_allocation(
        self, 
        short_pred: ResourceDemand, 
        medium_pred: ResourceDemand, 
        strategy: AllocationStrategy
    ) -> float:
        """Optimize memory allocation based on predictions and strategy"""
        base_demand = (short_pred.memory_demand + medium_pred.memory_demand) / 2
        
        if strategy == AllocationStrategy.CONSERVATIVE:
            return base_demand * 1.4  # 40% buffer
        elif strategy == AllocationStrategy.BALANCED:
            return base_demand * 1.15  # 15% buffer
        elif strategy == AllocationStrategy.AGGRESSIVE:
            return base_demand * 1.0  # No buffer
        elif strategy == AllocationStrategy.EVENT_DRIVEN:
            event_multiplier = 1.25 if short_pred.triggering_events else 1.05
            return base_demand * event_multiplier
        else:  # ML_OPTIMIZED
            confidence_multiplier = 1.0 + (1.0 - short_pred.prediction_confidence) * 0.4
            volatility_multiplier = 1.0 + short_pred.demand_volatility * 0.2
            return base_demand * confidence_multiplier * volatility_multiplier
    
    def _calculate_priority(self, service: str, risk_level: str) -> int:
        """Calculate allocation priority (1-10)"""
        # Base priorities by service criticality
        base_priorities = {
            "nautilus-risk-engine": 10,
            "nautilus-order-manager": 9,
            "nautilus-market-data": 8,
            "nautilus-strategy-engine": 7,
            "nautilus-position-keeper": 6
        }
        
        base_priority = base_priorities.get(service, 5)
        
        # Adjust based on risk level
        risk_adjustments = {"low": 0, "medium": 1, "high": 2}
        risk_adjustment = risk_adjustments.get(risk_level, 0)
        
        return min(10, base_priority + risk_adjustment)
    
    def _estimate_performance_impact(self, service: str, resource_type: str, change: float) -> float:
        """Estimate performance impact of resource allocation change"""
        # Simplified performance impact estimation
        if change > 0:  # Increase allocation
            return min(50.0, abs(change) * 10)  # Max 50% improvement
        else:  # Decrease allocation
            return max(-30.0, change * 15)  # Max 30% degradation
    
    def _generate_justification(
        self, 
        service: str, 
        resource_type: str, 
        short_pred: ResourceDemand, 
        medium_pred: ResourceDemand
    ) -> str:
        """Generate justification text for allocation decision"""
        reasons = []
        
        if short_pred.triggering_events:
            reasons.append(f"Upcoming events: {', '.join(short_pred.triggering_events)}")
        
        if short_pred.market_regime != "normal":
            reasons.append(f"Market regime: {short_pred.market_regime}")
        
        if short_pred.risk_level == "high":
            reasons.append("High risk environment detected")
        
        confidence_desc = "high" if short_pred.prediction_confidence > 0.7 else "medium" if short_pred.prediction_confidence > 0.5 else "low"
        reasons.append(f"Prediction confidence: {confidence_desc}")
        
        if not reasons:
            reasons.append("Standard optimization based on predicted demand")
        
        return "; ".join(reasons)
    
    def _assess_plan_risks(
        self, 
        predictions: Dict[str, Dict[str, ResourceDemand]], 
        allocations: List[ResourceAllocation]
    ) -> Dict[str, float]:
        """Assess risks associated with the allocation plan"""
        risks = {}
        
        # Cost overrun risk
        total_cost_increase = sum(max(0, alloc.cost_impact) for alloc in allocations)
        risks['cost_overrun'] = min(1.0, total_cost_increase / 100)  # Normalize
        
        # Under-allocation risk
        high_confidence_predictions = [
            pred for service_preds in predictions.values()
            for pred in service_preds.values()
            if pred.prediction_confidence > 0.8
        ]
        
        if high_confidence_predictions:
            avg_volatility = sum(pred.demand_volatility for pred in high_confidence_predictions) / len(high_confidence_predictions)
            risks['under_allocation'] = avg_volatility
        else:
            risks['under_allocation'] = 0.5
        
        # Market event risk
        event_services = [
            pred for service_preds in predictions.values()
            for pred in service_preds.values()
            if pred.triggering_events
        ]
        risks['event_impact'] = min(1.0, len(event_services) / len(self.services))
        
        return risks
    
    def _create_contingency_plans(
        self, 
        predictions: Dict[str, Dict[str, ResourceDemand]], 
        allocations: List[ResourceAllocation]
    ) -> List[str]:
        """Create contingency plans for various scenarios"""
        plans = []
        
        # High volatility contingency
        high_vol_services = [
            service for service, preds in predictions.items()
            if any(pred.demand_volatility > 0.3 for pred in preds.values())
        ]
        
        if high_vol_services:
            plans.append(f"High volatility detected for {', '.join(high_vol_services)}: Monitor closely and be ready to scale up aggressively")
        
        # Event-based contingency
        event_services = [
            service for service, preds in predictions.items()
            if any(pred.triggering_events for pred in preds.values())
        ]
        
        if event_services:
            plans.append(f"Market events affecting {', '.join(event_services)}: Pre-position additional resources")
        
        # Cost management contingency
        high_cost_allocations = [alloc for alloc in allocations if alloc.cost_impact > 10]
        if high_cost_allocations:
            plans.append("High cost allocations detected: Review and approve before implementation")
        
        # Performance monitoring
        plans.append("Monitor actual vs predicted performance and adjust models accordingly")
        
        return plans
    
    async def execute_allocation_plan(self, plan: AllocationPlan) -> Dict[str, Any]:
        """Execute the resource allocation plan"""
        execution_results = {
            "plan_id": plan.plan_id,
            "execution_time": datetime.now().isoformat(),
            "total_allocations": len(plan.allocations),
            "successful_allocations": 0,
            "failed_allocations": 0,
            "results": []
        }
        
        # Sort allocations by priority (highest first)
        sorted_allocations = sorted(plan.allocations, key=lambda x: x.priority, reverse=True)
        
        for allocation in sorted_allocations:
            try:
                # In a real system, this would update Kubernetes resource limits
                result = {
                    "service": allocation.service_name,
                    "resource_type": allocation.resource_type.value,
                    "change": allocation.allocation_change,
                    "success": True,
                    "message": f"Updated {allocation.resource_type.value} allocation by {allocation.allocation_change:.2f}"
                }
                
                # Simulate some failures for testing
                if np.random.random() < 0.05:  # 5% failure rate
                    result["success"] = False
                    result["message"] = "Simulated allocation failure"
                    execution_results["failed_allocations"] += 1
                else:
                    execution_results["successful_allocations"] += 1
                
                execution_results["results"].append(result)
                
                # Store allocation history
                await self._store_allocation_history(allocation, result)
                
            except Exception as e:
                execution_results["failed_allocations"] += 1
                execution_results["results"].append({
                    "service": allocation.service_name,
                    "resource_type": allocation.resource_type.value,
                    "success": False,
                    "error": str(e)
                })
        
        return execution_results
    
    async def _store_allocation_history(self, allocation: ResourceAllocation, result: Dict[str, Any]):
        """Store allocation history for analysis and learning"""
        history_record = {
            "timestamp": datetime.now().isoformat(),
            "service": allocation.service_name,
            "resource_type": allocation.resource_type.value,
            "allocation_change": allocation.allocation_change,
            "priority": allocation.priority,
            "cost_impact": allocation.cost_impact,
            "expected_performance_impact": allocation.performance_impact,
            "execution_result": result
        }
        
        # Store in Redis
        self.redis_client.lpush(
            f"allocation:history:{allocation.service_name}",
            json.dumps(history_record)
        )
        
        # Keep only last 500 records per service
        self.redis_client.ltrim(f"allocation:history:{allocation.service_name}", 0, 499)


async def main():
    """Test the Predictive Resource Allocator"""
    logging.basicConfig(level=logging.INFO)
    
    allocator = PredictiveResourceAllocator()
    
    print("🔮 Testing Predictive Resource Allocation System")
    print("=" * 55)
    
    # Test demand prediction for each service
    for service in allocator.services:
        print(f"\n📈 Demand Prediction for {service}:")
        
        # Short-term prediction
        short_pred = await allocator.predict_demand(service, 15)
        print(f"15-min horizon: CPU={short_pred.cpu_demand:.2f}, Memory={short_pred.memory_demand:.1f}GB")
        print(f"Confidence: {short_pred.prediction_confidence:.2f}, Volatility: {short_pred.demand_volatility:.2f}")
        print(f"Market Regime: {short_pred.market_regime}, Risk: {short_pred.risk_level}")
        
        if short_pred.triggering_events:
            print(f"Triggering Events: {', '.join(short_pred.triggering_events)}")
    
    # Create and test allocation plan
    print(f"\n🎯 Creating ML-Optimized Allocation Plan...")
    plan = await allocator.create_allocation_plan(AllocationStrategy.ML_OPTIMIZED)
    
    print(f"Plan ID: {plan.plan_id}")
    print(f"Total Cost Impact: ${plan.total_cost:.2f}/hour")
    print(f"Expected Performance Gain: {plan.expected_performance_gain:.1f}%")
    print(f"Number of Allocations: {len(plan.allocations)}")
    
    if plan.allocations:
        print("\nTop Priority Allocations:")
        sorted_allocations = sorted(plan.allocations, key=lambda x: x.priority, reverse=True)
        for i, alloc in enumerate(sorted_allocations[:3]):
            print(f"{i+1}. {alloc.service_name} ({alloc.resource_type.value}): "
                  f"{alloc.allocation_change:+.2f} (Priority: {alloc.priority})")
            print(f"   Justification: {alloc.justification}")
    
    print(f"\nRisk Assessment:")
    for risk_type, risk_value in plan.risk_assessment.items():
        print(f"- {risk_type}: {risk_value:.2f}")
    
    print(f"\nContingency Plans:")
    for i, plan_text in enumerate(plan.contingency_plans, 1):
        print(f"{i}. {plan_text}")
    
    # Execute the plan (simulation)
    print(f"\n🚀 Executing Allocation Plan...")
    execution_result = await allocator.execute_allocation_plan(plan)
    
    print(f"Execution completed: {execution_result['successful_allocations']}/{execution_result['total_allocations']} successful")
    
    if execution_result["failed_allocations"] > 0:
        print(f"Failed allocations: {execution_result['failed_allocations']}")
    
    print("\n✅ Predictive Resource Allocation test completed!")


if __name__ == "__main__":
    asyncio.run(main())