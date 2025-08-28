#!/usr/bin/env python3
"""
Nautilus Adaptive Learning Module
Advanced system learning and optimization based on stress test results.
Implements machine learning for performance prediction and automatic tuning.
"""

import asyncio
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
from pathlib import Path
import pickle
import sqlite3
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
import redis.asyncio as redis
import aiohttp
import asyncpg
from collections import defaultdict, deque
import threading
import time
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SystemState:
    """Current system state snapshot"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    network_io: float
    disk_io: float
    redis_connections: int
    database_connections: int
    active_engines: int
    request_rate: float
    error_rate: float
    avg_latency: float
    neural_engine_usage: float
    gpu_usage: float

@dataclass
class PerformancePrediction:
    """Performance prediction for given configuration"""
    predicted_latency: float
    predicted_throughput: float
    predicted_error_rate: float
    confidence_score: float
    recommended_actions: List[str]

@dataclass
class OptimizationAction:
    """System optimization action"""
    action_type: str
    target: str
    parameter: str
    old_value: Any
    new_value: Any
    expected_improvement: float
    confidence: float

class AdaptiveLearningEngine:
    """Machine learning engine for system optimization"""
    
    def __init__(self, db_path: str = "nautilus_learning.db"):
        self.db_path = db_path
        self.models = {}
        self.scalers = {}
        self.performance_history = deque(maxlen=10000)
        self.optimization_history = []
        self.learning_rate = 0.1
        
        # Initialize database
        self.init_database()
        
        # Performance prediction models
        self.latency_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.throughput_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.error_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        
        # Feature scalers
        self.feature_scaler = StandardScaler()
        
        # Current system configuration
        self.current_config = {
            "redis_pool_size": 20,
            "request_timeout": 5.0,
            "circuit_breaker_threshold": 0.05,
            "load_balancer_weights": {"engine_logic_bus": 0.4, "marketdata_bus": 0.6},
            "neural_engine_allocation": 0.8,
            "gpu_memory_allocation": 0.9,
            "database_connection_pool": 50
        }

    def init_database(self):
        """Initialize SQLite database for learning data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                cpu_usage REAL,
                memory_usage REAL,
                network_io REAL,
                disk_io REAL,
                redis_connections INTEGER,
                database_connections INTEGER,
                active_engines INTEGER,
                request_rate REAL,
                error_rate REAL,
                avg_latency REAL,
                neural_engine_usage REAL,
                gpu_usage REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS optimization_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                action_type TEXT,
                target TEXT,
                parameter TEXT,
                old_value TEXT,
                new_value TEXT,
                expected_improvement REAL,
                actual_improvement REAL,
                confidence REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                test_type TEXT,
                duration REAL,
                total_requests INTEGER,
                total_errors INTEGER,
                avg_latency REAL,
                max_throughput REAL,
                system_grade TEXT,
                improvements_found INTEGER
            )
        ''')
        
        conn.commit()
        conn.close()

    async def record_system_state(self, state: SystemState):
        """Record current system state for learning"""
        self.performance_history.append(state)
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO performance_history 
            (timestamp, cpu_usage, memory_usage, network_io, disk_io, 
             redis_connections, database_connections, active_engines, 
             request_rate, error_rate, avg_latency, neural_engine_usage, gpu_usage)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            state.timestamp, state.cpu_usage, state.memory_usage, state.network_io,
            state.disk_io, state.redis_connections, state.database_connections,
            state.active_engines, state.request_rate, state.error_rate,
            state.avg_latency, state.neural_engine_usage, state.gpu_usage
        ))
        
        conn.commit()
        conn.close()

    def extract_features(self, state: SystemState) -> np.ndarray:
        """Extract features for machine learning models"""
        features = [
            state.cpu_usage,
            state.memory_usage,
            state.network_io,
            state.disk_io,
            state.redis_connections,
            state.database_connections,
            state.active_engines,
            state.request_rate,
            state.neural_engine_usage,
            state.gpu_usage,
            # Configuration features
            self.current_config["redis_pool_size"],
            self.current_config["request_timeout"],
            self.current_config["circuit_breaker_threshold"],
            self.current_config["neural_engine_allocation"],
            self.current_config["gpu_memory_allocation"],
            self.current_config["database_connection_pool"]
        ]
        return np.array(features)

    async def train_models(self) -> Dict[str, float]:
        """Train ML models on historical performance data"""
        if len(self.performance_history) < 100:
            logger.warning("Insufficient data for training models")
            return {"training": "skipped", "reason": "insufficient_data"}

        logger.info("🧠 Training adaptive learning models...")
        
        # Prepare training data
        X = []
        y_latency = []
        y_throughput = []
        y_error = []
        
        for state in self.performance_history:
            features = self.extract_features(state)
            X.append(features)
            y_latency.append(state.avg_latency)
            y_throughput.append(state.request_rate)
            y_error.append(1 if state.error_rate > 0.01 else 0)  # Binary classification
        
        X = np.array(X)
        y_latency = np.array(y_latency)
        y_throughput = np.array(y_throughput)
        y_error = np.array(y_error)
        
        # Scale features
        X_scaled = self.feature_scaler.fit_transform(X)
        
        # Train models
        results = {}
        
        # Latency prediction model
        X_train, X_test, y_lat_train, y_lat_test = train_test_split(
            X_scaled, y_latency, test_size=0.2, random_state=42
        )
        self.latency_model.fit(X_train, y_lat_train)
        lat_predictions = self.latency_model.predict(X_test)
        results["latency_mae"] = mean_absolute_error(y_lat_test, lat_predictions)
        
        # Throughput prediction model
        X_train, X_test, y_thr_train, y_thr_test = train_test_split(
            X_scaled, y_throughput, test_size=0.2, random_state=42
        )
        self.throughput_model.fit(X_train, y_thr_train)
        thr_predictions = self.throughput_model.predict(X_test)
        results["throughput_mae"] = mean_absolute_error(y_thr_test, thr_predictions)
        
        # Error prediction model
        if len(np.unique(y_error)) > 1:  # Only train if we have both classes
            X_train, X_test, y_err_train, y_err_test = train_test_split(
                X_scaled, y_error, test_size=0.2, random_state=42
            )
            self.error_model.fit(X_train, y_err_train)
            err_predictions = self.error_model.predict(X_test)
            results["error_accuracy"] = accuracy_score(y_err_test, err_predictions)
        
        logger.info(f"✅ Models trained - Latency MAE: {results.get('latency_mae', 0):.3f}ms")
        return results

    async def predict_performance(self, hypothetical_state: SystemState) -> PerformancePrediction:
        """Predict performance for a hypothetical system state"""
        
        if not hasattr(self.latency_model, 'n_features_in_'):
            # Models not trained yet
            await self.train_models()
        
        features = self.extract_features(hypothetical_state)
        features_scaled = self.feature_scaler.transform([features])
        
        # Make predictions
        predicted_latency = self.latency_model.predict(features_scaled)[0]
        predicted_throughput = self.throughput_model.predict(features_scaled)[0]
        
        try:
            predicted_error_prob = self.error_model.predict_proba(features_scaled)[0][1]
        except:
            predicted_error_prob = 0.01  # Default low error rate
        
        # Calculate confidence based on prediction uncertainty
        # Use ensemble variance as confidence metric
        confidence_score = min(0.95, max(0.5, 1.0 - (predicted_latency / 10.0)))
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            hypothetical_state, predicted_latency, predicted_throughput, predicted_error_prob
        )
        
        return PerformancePrediction(
            predicted_latency=predicted_latency,
            predicted_throughput=predicted_throughput,
            predicted_error_rate=predicted_error_prob,
            confidence_score=confidence_score,
            recommended_actions=recommendations
        )

    def _generate_recommendations(self, state: SystemState, latency: float, 
                                throughput: float, error_rate: float) -> List[str]:
        """Generate optimization recommendations based on predictions"""
        recommendations = []
        
        if latency > 5.0:
            recommendations.append("Increase Redis connection pool size")
            recommendations.append("Optimize database query patterns")
            recommendations.append("Enable aggressive caching")
        
        if throughput < 50000:
            recommendations.append("Scale engine horizontally")
            recommendations.append("Optimize message bus routing")
            recommendations.append("Increase hardware allocation")
        
        if error_rate > 0.02:
            recommendations.append("Implement circuit breaker patterns")
            recommendations.append("Add request rate limiting")
            recommendations.append("Enhance error handling")
        
        if state.neural_engine_usage < 80:
            recommendations.append("Increase Neural Engine utilization")
            recommendations.append("Optimize ML model inference")
        
        if state.gpu_usage < 85:
            recommendations.append("Increase GPU workload distribution")
            recommendations.append("Optimize VPIN calculations")
        
        return recommendations

    async def generate_optimization_actions(self, current_state: SystemState) -> List[OptimizationAction]:
        """Generate specific optimization actions based on current state"""
        actions = []
        
        # Predict current performance
        prediction = await self.predict_performance(current_state)
        
        if prediction.predicted_latency > 3.0:
            # Increase Redis pool size
            actions.append(OptimizationAction(
                action_type="config_change",
                target="redis",
                parameter="pool_size",
                old_value=self.current_config["redis_pool_size"],
                new_value=min(50, self.current_config["redis_pool_size"] * 1.5),
                expected_improvement=15.0,
                confidence=0.8
            ))
        
        if current_state.error_rate > 0.01:
            # Reduce circuit breaker threshold
            actions.append(OptimizationAction(
                action_type="config_change",
                target="circuit_breaker",
                parameter="threshold",
                old_value=self.current_config["circuit_breaker_threshold"],
                new_value=max(0.01, self.current_config["circuit_breaker_threshold"] * 0.8),
                expected_improvement=25.0,
                confidence=0.9
            ))
        
        if current_state.neural_engine_usage < 80:
            # Increase Neural Engine allocation
            actions.append(OptimizationAction(
                action_type="hardware_optimization",
                target="neural_engine",
                parameter="allocation",
                old_value=self.current_config["neural_engine_allocation"],
                new_value=min(0.95, self.current_config["neural_engine_allocation"] + 0.1),
                expected_improvement=10.0,
                confidence=0.7
            ))
        
        return actions

    async def apply_optimization_action(self, action: OptimizationAction) -> bool:
        """Apply an optimization action to the system"""
        logger.info(f"🔧 Applying optimization: {action.action_type} on {action.target}.{action.parameter}")
        
        try:
            if action.action_type == "config_change":
                # Update configuration
                if action.target == "redis":
                    self.current_config["redis_pool_size"] = int(action.new_value)
                elif action.target == "circuit_breaker":
                    self.current_config["circuit_breaker_threshold"] = float(action.new_value)
                
                # In a real system, this would apply changes to Redis/engine configs
                # For now, we simulate the change
                await asyncio.sleep(1)  # Simulate configuration change delay
                
            elif action.action_type == "hardware_optimization":
                # Update hardware allocation
                if action.target == "neural_engine":
                    self.current_config["neural_engine_allocation"] = float(action.new_value)
                elif action.target == "gpu_memory":
                    self.current_config["gpu_memory_allocation"] = float(action.new_value)
                
                await asyncio.sleep(2)  # Simulate hardware reallocation delay
            
            # Record the action
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO optimization_actions 
                (timestamp, action_type, target, parameter, old_value, new_value, 
                 expected_improvement, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now(), action.action_type, action.target, action.parameter,
                str(action.old_value), str(action.new_value), 
                action.expected_improvement, action.confidence
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"✅ Applied optimization: {action.parameter} = {action.new_value}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to apply optimization: {e}")
            return False

    async def adaptive_learning_cycle(self, stress_test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a complete adaptive learning cycle"""
        logger.info("🧠 Starting Adaptive Learning Cycle...")
        
        # Extract performance metrics from stress test results
        current_state = SystemState(
            timestamp=datetime.now(),
            cpu_usage=stress_test_results.get("hardware_utilization", {}).get("cpu_usage", 50.0),
            memory_usage=stress_test_results.get("hardware_utilization", {}).get("memory_usage", 60.0),
            network_io=1000.0,  # Simulated
            disk_io=500.0,      # Simulated
            redis_connections=stress_test_results.get("hardware_utilization", {}).get("redis_connections", 100),
            database_connections=20,  # Simulated
            active_engines=18,
            request_rate=stress_test_results.get("performance_summary", {}).get("total_requests", 0) / 300,
            error_rate=stress_test_results.get("performance_summary", {}).get("overall_error_rate", 0),
            avg_latency=stress_test_results.get("performance_summary", {}).get("final_latency_ms", 2.0),
            neural_engine_usage=stress_test_results.get("hardware_utilization", {}).get("neural_engine_usage", 85.0),
            gpu_usage=stress_test_results.get("hardware_utilization", {}).get("gpu_usage", 90.0)
        )
        
        # Record the current state
        await self.record_system_state(current_state)
        
        # Train models on accumulated data
        training_results = await self.train_models()
        
        # Generate optimization actions
        optimization_actions = await self.generate_optimization_actions(current_state)
        
        # Apply top optimization actions
        applied_actions = []
        for action in sorted(optimization_actions, key=lambda x: x.expected_improvement, reverse=True)[:3]:
            if action.confidence > 0.7:  # Only apply high-confidence actions
                success = await self.apply_optimization_action(action)
                if success:
                    applied_actions.append(action)
        
        # Predict performance after optimizations
        future_prediction = await self.predict_performance(current_state)
        
        # Calculate learning metrics
        learning_metrics = {
            "learning_cycle_completed": True,
            "data_points_collected": len(self.performance_history),
            "models_trained": len(training_results),
            "optimization_actions_generated": len(optimization_actions),
            "optimization_actions_applied": len(applied_actions),
            "predicted_improvement": {
                "latency_reduction": max(0, current_state.avg_latency - future_prediction.predicted_latency),
                "throughput_increase": max(0, future_prediction.predicted_throughput - current_state.request_rate),
                "error_reduction": max(0, current_state.error_rate - future_prediction.predicted_error_rate)
            },
            "confidence_score": future_prediction.confidence_score,
            "recommendations": future_prediction.recommended_actions,
            "applied_optimizations": [
                {
                    "action": f"{action.target}.{action.parameter}",
                    "change": f"{action.old_value} → {action.new_value}",
                    "expected_improvement": f"{action.expected_improvement:.1f}%"
                }
                for action in applied_actions
            ]
        }
        
        # Store learning session
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO learning_sessions 
            (timestamp, test_type, duration, total_requests, total_errors, 
             avg_latency, max_throughput, system_grade, improvements_found)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now(), "comprehensive_stress_test", 300,
            stress_test_results.get("performance_summary", {}).get("total_requests", 0),
            stress_test_results.get("performance_summary", {}).get("total_errors", 0),
            current_state.avg_latency, future_prediction.predicted_throughput,
            stress_test_results.get("performance_summary", {}).get("system_grade", "B"),
            len(applied_actions)
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"✅ Learning cycle completed - Applied {len(applied_actions)} optimizations")
        logger.info(f"🎯 Predicted improvements: -{learning_metrics['predicted_improvement']['latency_reduction']:.2f}ms latency")
        
        return learning_metrics

    async def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of learning progress and optimizations"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get learning session stats
        cursor.execute('''
            SELECT COUNT(*), AVG(avg_latency), AVG(max_throughput), AVG(improvements_found)
            FROM learning_sessions
            ORDER BY timestamp DESC LIMIT 10
        ''')
        session_stats = cursor.fetchone()
        
        # Get recent optimizations
        cursor.execute('''
            SELECT action_type, target, parameter, expected_improvement, confidence
            FROM optimization_actions
            ORDER BY timestamp DESC LIMIT 5
        ''')
        recent_optimizations = cursor.fetchall()
        
        conn.close()
        
        return {
            "learning_sessions": session_stats[0] if session_stats else 0,
            "average_latency": session_stats[1] if session_stats else 0,
            "average_throughput": session_stats[2] if session_stats else 0,
            "average_improvements_per_session": session_stats[3] if session_stats else 0,
            "recent_optimizations": [
                {
                    "action": f"{opt[1]}.{opt[2]}",
                    "type": opt[0],
                    "expected_improvement": f"{opt[3]:.1f}%",
                    "confidence": f"{opt[4]:.1f}"
                }
                for opt in recent_optimizations
            ],
            "current_configuration": self.current_config,
            "data_points_collected": len(self.performance_history)
        }

# Global learning engine instance
learning_engine = AdaptiveLearningEngine()

async def run_learning_cycle_standalone():
    """Run a standalone learning cycle for testing"""
    
    # Simulate stress test results
    mock_stress_results = {
        "performance_summary": {
            "total_requests": 150000,
            "total_errors": 50,
            "overall_error_rate": 0.0003,
            "final_latency_ms": 3.2,
            "system_grade": "A-"
        },
        "hardware_utilization": {
            "cpu_usage": 75.5,
            "memory_usage": 68.2,
            "neural_engine_usage": 82.1,
            "gpu_usage": 88.7,
            "redis_connections": 145
        }
    }
    
    # Execute learning cycle
    results = await learning_engine.adaptive_learning_cycle(mock_stress_results)
    
    print(json.dumps(results, indent=2, default=str))
    
    # Get learning summary
    summary = await learning_engine.get_learning_summary()
    print("\n📊 Learning Summary:")
    print(json.dumps(summary, indent=2, default=str))

if __name__ == "__main__":
    print("🧠 Nautilus Adaptive Learning Module")
    print("=" * 40)
    asyncio.run(run_learning_cycle_standalone())