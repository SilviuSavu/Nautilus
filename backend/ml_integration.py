"""
ML Integration Module for Nautilus Trading Platform
Integrates ML framework with existing risk management, WebSocket infrastructure,
and data sources for seamless real-time operation.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json

# ML Framework imports
from ml.config import MLConfig
from ml.market_regime import MarketRegimeDetector
from ml.feature_engineering import FeatureEngineer
from ml.model_lifecycle import ModelLifecycleManager
from ml.risk_prediction import RiskPredictor
from ml.inference_engine import InferenceEngine

# Existing Nautilus imports (with fallbacks for standalone testing)
try:
    from risk.breach_detector import AdvancedBreachDetector
except ImportError:
    # Fallback for standalone testing
    class AdvancedBreachDetector:
        def register_ml_callback(self, callback_type, callback_func):
            pass
        
try:
    from websocket.websocket_routes import WebSocketManager
except ImportError:
    # Fallback for standalone testing
    class WebSocketManager:
        async def broadcast_to_topic(self, topic, message):
            pass

import redis.asyncio as redis

logger = logging.getLogger(__name__)

class MLNautilusIntegrator:
    """
    Main integration class that connects ML framework with existing Nautilus infrastructure.
    Handles real-time ML predictions, risk integration, and WebSocket streaming.
    """
    
    def __init__(self, config: Optional[MLConfig] = None):
        self.config = config or MLConfig()
        
        # Initialize ML components
        self.regime_detector = MarketRegimeDetector(self.config)
        self.feature_engineer = FeatureEngineer(self.config)
        self.lifecycle_manager = ModelLifecycleManager(self.config)
        self.risk_predictor = RiskPredictor()
        self.inference_engine = InferenceEngine(self.config)
        
        # Integration components
        self.redis_client = None
        self.websocket_manager = None
        self.breach_detector = None
        
        # State management
        self.is_running = False
        self.background_tasks = []
        self.last_regime_update = None
        self.last_risk_check = None
        
    async def initialize(self):
        """Initialize all ML components and integrations."""
        try:
            logger.info("Initializing ML-Nautilus integrator...")
            
            # Initialize Redis connection
            self.redis_client = redis.from_url(self.config.redis_url)
            await self.redis_client.ping()
            logger.info("Redis connection established")
            
            # Initialize ML components
            await self.regime_detector.initialize()
            await self.feature_engineer.initialize()
            await self.lifecycle_manager.initialize()
            await self.risk_predictor.initialize()
            await self.inference_engine.initialize()
            
            # Initialize WebSocket manager (if available)
            try:
                self.websocket_manager = WebSocketManager()
                logger.info("WebSocket manager initialized")
            except Exception as e:
                logger.warning(f"WebSocket manager not available: {e}")
            
            # Initialize breach detector integration
            try:
                self.breach_detector = AdvancedBreachDetector()
                await self._integrate_with_breach_detector()
                logger.info("Breach detector integration complete")
            except Exception as e:
                logger.warning(f"Breach detector not available: {e}")
            
            self.is_running = True
            logger.info("ML-Nautilus integrator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ML integrator: {e}")
            raise
    
    async def start_background_tasks(self):
        """Start all background ML tasks."""
        if not self.is_running:
            await self.initialize()
        
        # Start regime monitoring
        regime_task = asyncio.create_task(self._regime_monitoring_loop())
        self.background_tasks.append(regime_task)
        
        # Start risk prediction updates
        risk_task = asyncio.create_task(self._risk_prediction_loop())
        self.background_tasks.append(risk_task)
        
        # Start model health monitoring
        health_task = asyncio.create_task(self._model_health_monitoring_loop())
        self.background_tasks.append(health_task)
        
        # Start drift detection
        drift_task = asyncio.create_task(self._drift_detection_loop())
        self.background_tasks.append(drift_task)
        
        # Start feature engineering updates
        feature_task = asyncio.create_task(self._feature_update_loop())
        self.background_tasks.append(feature_task)
        
        logger.info(f"Started {len(self.background_tasks)} ML background tasks")
    
    async def stop_background_tasks(self):
        """Stop all background ML tasks."""
        logger.info("Stopping ML background tasks...")
        
        for task in self.background_tasks:
            task.cancel()
        
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        self.background_tasks.clear()
        self.is_running = False
        
        logger.info("All ML background tasks stopped")
    
    async def _regime_monitoring_loop(self):
        """Background task for continuous regime monitoring."""
        while self.is_running:
            try:
                # Get current regime
                regime_state = await self.regime_detector.get_current_regime()
                
                # Check for regime changes
                if self._is_significant_regime_change(regime_state):
                    await self._handle_regime_change(regime_state)
                
                # Publish regime update via WebSocket
                if self.websocket_manager:
                    await self._broadcast_regime_update(regime_state)
                
                # Store regime data in Redis
                await self._cache_regime_data(regime_state)
                
                self.last_regime_update = datetime.utcnow()
                
                # Sleep until next update
                await asyncio.sleep(self.config.regime_detection.update_frequency)
                
            except Exception as e:
                logger.error(f"Error in regime monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _risk_prediction_loop(self):
        """Background task for continuous risk prediction updates."""
        while self.is_running:
            try:
                # Get current portfolio positions (you'll need to implement this)
                portfolio = await self._get_current_portfolio()
                
                if portfolio:
                    # Calculate VaR and risk metrics
                    var_result = await self.risk_predictor.calculate_var(portfolio)
                    
                    # Check for risk threshold breaches
                    if self.breach_detector:
                        await self._check_ml_risk_breaches(var_result)
                    
                    # Broadcast risk updates
                    if self.websocket_manager:
                        await self._broadcast_risk_update(var_result)
                    
                    # Cache risk data
                    await self._cache_risk_data(var_result)
                
                self.last_risk_check = datetime.utcnow()
                
                # Sleep until next check
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in risk prediction loop: {e}")
                await asyncio.sleep(60)
    
    async def _model_health_monitoring_loop(self):
        """Background task for monitoring ML model health."""
        while self.is_running:
            try:
                # Check inference engine health
                engine_metrics = await self.inference_engine.get_system_metrics()
                
                # Check for performance degradation
                if self._is_performance_degraded(engine_metrics):
                    await self._handle_performance_degradation(engine_metrics)
                
                # Monitor model server health
                for model_name in await self.inference_engine.list_available_models():
                    model_health = await self._check_model_health(model_name)
                    
                    if not model_health['healthy']:
                        await self._handle_unhealthy_model(model_name, model_health)
                
                # Sleep until next health check
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in model health monitoring: {e}")
                await asyncio.sleep(300)
    
    async def _drift_detection_loop(self):
        """Background task for model drift detection."""
        while self.is_running:
            try:
                # Check drift for all active models
                active_models = await self.lifecycle_manager.get_active_models()
                
                for model in active_models:
                    drift_result = await self.lifecycle_manager.check_model_drift(
                        model_type=model['type']
                    )
                    
                    if drift_result.drift_detected:
                        await self._handle_model_drift(model, drift_result)
                
                # Sleep until next drift check
                await asyncio.sleep(3600)  # Check hourly
                
            except Exception as e:
                logger.error(f"Error in drift detection loop: {e}")
                await asyncio.sleep(3600)
    
    async def _feature_update_loop(self):
        """Background task for updating feature importance and correlation analysis."""
        while self.is_running:
            try:
                # Update feature importance for all models
                await self.feature_engineer.update_feature_importance()
                
                # Run correlation analysis for top symbols
                top_symbols = await self._get_top_trading_symbols()
                if top_symbols:
                    await self.feature_engineer.correlation_analyzer.analyze_cross_asset_correlation(
                        symbols=top_symbols,
                        lookback_days=30
                    )
                
                # Sleep until next update
                await asyncio.sleep(1800)  # Update every 30 minutes
                
            except Exception as e:
                logger.error(f"Error in feature update loop: {e}")
                await asyncio.sleep(1800)
    
    async def _integrate_with_breach_detector(self):
        """Integrate ML predictions with existing breach detector."""
        if not self.breach_detector:
            return
        
        # Register ML-based breach detection callbacks
        self.breach_detector.register_ml_callback(
            'regime_change', self._ml_regime_breach_check
        )
        self.breach_detector.register_ml_callback(
            'risk_prediction', self._ml_risk_breach_check
        )
        self.breach_detector.register_ml_callback(
            'volatility_spike', self._ml_volatility_breach_check
        )
        
        logger.info("ML callbacks registered with breach detector")
    
    async def _ml_regime_breach_check(self, symbol: str, price: float, **kwargs) -> Dict[str, Any]:
        """ML-enhanced regime-based breach detection."""
        try:
            # Get current regime with confidence
            regime_state = await self.regime_detector.get_current_regime()
            
            # Adjust breach thresholds based on regime
            regime_multipliers = {
                'bull': 0.8,    # Looser thresholds in bull market
                'bear': 1.3,    # Tighter thresholds in bear market
                'volatile': 1.5, # Much tighter in volatile conditions
                'crisis': 2.0   # Very tight in crisis
            }
            
            multiplier = regime_multipliers.get(regime_state.regime.value, 1.0)
            confidence_factor = regime_state.confidence
            
            return {
                'regime': regime_state.regime.value,
                'confidence': confidence_factor,
                'threshold_multiplier': multiplier * confidence_factor,
                'recommendation': f'Adjust thresholds by {multiplier:.2f}x due to {regime_state.regime.value} regime'
            }
            
        except Exception as e:
            logger.error(f"Error in ML regime breach check: {e}")
            return {'error': str(e)}
    
    async def _ml_risk_breach_check(self, portfolio: Dict[str, float], **kwargs) -> Dict[str, Any]:
        """ML-enhanced portfolio risk breach detection."""
        try:
            # Calculate ML-based VaR
            var_result = await self.risk_predictor.calculate_var(portfolio)
            
            # Get stress test scenarios
            stress_result = await self.risk_predictor.run_stress_test(portfolio)
            
            breach_risk = {
                'var_95': var_result.var,
                'expected_shortfall': var_result.expected_shortfall,
                'worst_case_loss': stress_result.worst_case_loss,
                'risk_score': min(abs(var_result.var) / 10000, 1.0),  # Normalize to 0-1
                'recommendation': 'Reduce position sizes' if abs(var_result.var) > 50000 else 'Risk levels acceptable'
            }
            
            return breach_risk
            
        except Exception as e:
            logger.error(f"Error in ML risk breach check: {e}")
            return {'error': str(e)}
    
    async def _ml_volatility_breach_check(self, symbol: str, price: float, **kwargs) -> Dict[str, Any]:
        """ML-enhanced volatility breach detection."""
        try:
            # Compute real-time features for the symbol
            features = await self.feature_engineer.compute_features(symbol)
            
            # Extract volatility-related features
            volatility_features = {
                'realized_volatility': features.features.get('volatility', {}).get('realized_volatility_20', 0),
                'garch_volatility': features.features.get('volatility', {}).get('garch_volatility', 0),
                'volatility_regime': features.features.get('volatility', {}).get('volatility_regime', 'normal')
            }
            
            # Use ML model to predict volatility spike probability
            spike_probability = await self._predict_volatility_spike(symbol, volatility_features)
            
            return {
                'volatility_features': volatility_features,
                'spike_probability': spike_probability,
                'alert_level': 'high' if spike_probability > 0.8 else 'medium' if spike_probability > 0.5 else 'low',
                'recommendation': f'Volatility spike probability: {spike_probability:.2%}'
            }
            
        except Exception as e:
            logger.error(f"Error in ML volatility breach check: {e}")
            return {'error': str(e)}
    
    # Helper methods
    def _is_significant_regime_change(self, regime_state) -> bool:
        """Check if there's a significant regime change worth broadcasting."""
        if not hasattr(self, '_last_regime') or not hasattr(self, '_last_regime_confidence'):
            self._last_regime = regime_state.regime
            self._last_regime_confidence = regime_state.confidence
            return True
        
        regime_changed = self._last_regime != regime_state.regime
        confidence_significant = abs(self._last_regime_confidence - regime_state.confidence) > 0.1
        
        if regime_changed or confidence_significant:
            self._last_regime = regime_state.regime
            self._last_regime_confidence = regime_state.confidence
            return True
        
        return False
    
    async def _handle_regime_change(self, regime_state):
        """Handle significant regime changes."""
        logger.info(f"Significant regime change detected: {regime_state.regime.value} (confidence: {regime_state.confidence:.3f})")
        
        # Publish alert to Redis
        alert_data = {
            'type': 'regime_change',
            'regime': regime_state.regime.value,
            'confidence': regime_state.confidence,
            'timestamp': datetime.utcnow().isoformat(),
            'action_required': regime_state.regime.value in ['crisis', 'volatile']
        }
        
        await self.redis_client.publish('ml_alerts', json.dumps(alert_data))
    
    async def _broadcast_regime_update(self, regime_state):
        """Broadcast regime update via WebSocket."""
        if not self.websocket_manager:
            return
        
        message = {
            'type': 'regime_update',
            'data': {
                'regime': regime_state.regime.value,
                'confidence': regime_state.confidence,
                'probabilities': regime_state.probabilities,
                'timestamp': regime_state.timestamp.isoformat()
            }
        }
        
        await self.websocket_manager.broadcast_to_topic('market_regime', message)
    
    async def _cache_regime_data(self, regime_state):
        """Cache regime data in Redis for fast access."""
        cache_data = {
            'regime': regime_state.regime.value,
            'confidence': regime_state.confidence,
            'probabilities': regime_state.probabilities,
            'timestamp': regime_state.timestamp.isoformat()
        }
        
        await self.redis_client.setex(
            'ml:current_regime',
            300,  # 5-minute TTL
            json.dumps(cache_data)
        )
    
    async def _get_current_portfolio(self) -> Optional[Dict[str, float]]:
        """Get current portfolio positions - integrate with your position management system."""
        # This should integrate with your actual portfolio/position management
        # For now, return a mock portfolio
        try:
            # Try to get from Redis cache first
            portfolio_data = await self.redis_client.get('current_portfolio')
            if portfolio_data:
                return json.loads(portfolio_data)
            
            # If not cached, return default/mock portfolio
            return {
                'AAPL': 1000,
                'GOOGL': 500,
                'MSFT': 750,
                'TSLA': 200
            }
        except Exception as e:
            logger.error(f"Error getting current portfolio: {e}")
            return None
    
    async def _check_ml_risk_breaches(self, var_result):
        """Check ML-calculated risk metrics against thresholds."""
        # Define risk thresholds
        var_threshold = 100000  # $100k VaR threshold
        es_threshold = 150000   # $150k Expected Shortfall threshold
        
        breaches = []
        
        if abs(var_result.var) > var_threshold:
            breaches.append({
                'type': 'var_breach',
                'value': var_result.var,
                'threshold': var_threshold,
                'severity': 'high'
            })
        
        if abs(var_result.expected_shortfall) > es_threshold:
            breaches.append({
                'type': 'expected_shortfall_breach',
                'value': var_result.expected_shortfall,
                'threshold': es_threshold,
                'severity': 'critical'
            })
        
        if breaches:
            await self._handle_risk_breaches(breaches)
    
    async def _handle_risk_breaches(self, breaches):
        """Handle detected risk breaches."""
        for breach in breaches:
            logger.warning(f"ML Risk breach detected: {breach}")
            
            # Publish breach alert
            alert_data = {
                'type': 'ml_risk_breach',
                'breach_type': breach['type'],
                'value': breach['value'],
                'threshold': breach['threshold'],
                'severity': breach['severity'],
                'timestamp': datetime.utcnow().isoformat()
            }
            
            await self.redis_client.publish('risk_alerts', json.dumps(alert_data))
    
    async def _predict_volatility_spike(self, symbol: str, volatility_features: Dict) -> float:
        """Predict probability of volatility spike using ML model."""
        try:
            # Use inference engine to predict volatility spike
            inference_request = {
                'model_name': 'volatility_predictor',
                'features': volatility_features,
                'symbol': symbol
            }
            
            # If model doesn't exist, return heuristic-based probability
            if 'volatility_predictor' not in await self.inference_engine.list_available_models():
                realized_vol = volatility_features.get('realized_volatility', 0)
                # Simple heuristic: high probability if realized vol > 30%
                return min(realized_vol / 0.3, 1.0) if realized_vol > 0.2 else 0.1
            
            result = await self.inference_engine.predict(inference_request)
            return result.prediction.get('spike_probability', 0.1)
            
        except Exception as e:
            logger.error(f"Error predicting volatility spike: {e}")
            return 0.1  # Default low probability
    
    async def _get_top_trading_symbols(self) -> List[str]:
        """Get list of most actively traded symbols."""
        # This should integrate with your trading data
        # For now, return common symbols
        return ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'SPY', 'QQQ', 'IWM']
    
    def _is_performance_degraded(self, metrics: Dict) -> bool:
        """Check if system performance is degraded."""
        avg_latency = metrics.get('average_latency_ms', 0)
        error_rate = metrics.get('error_rate', 0)
        
        return avg_latency > 1000 or error_rate > 0.05  # 1s latency or 5% error rate
    
    async def _handle_performance_degradation(self, metrics: Dict):
        """Handle performance degradation."""
        logger.warning(f"ML system performance degraded: {metrics}")
        
        # Scale up resources or restart models if needed
        # This would integrate with your scaling infrastructure
        
        alert_data = {
            'type': 'performance_degradation',
            'metrics': metrics,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        await self.redis_client.publish('system_alerts', json.dumps(alert_data))
    
    async def _check_model_health(self, model_name: str) -> Dict[str, Any]:
        """Check health of individual model."""
        try:
            # Get model server metrics
            server = self.inference_engine.model_servers.get(model_name)
            if not server:
                return {'healthy': False, 'reason': 'Model server not found'}
            
            # Check if model is responsive
            test_request = {'model_name': model_name, 'features': {}}
            try:
                await asyncio.wait_for(
                    self.inference_engine.predict(test_request),
                    timeout=5.0
                )
                return {'healthy': True}
            except asyncio.TimeoutError:
                return {'healthy': False, 'reason': 'Model response timeout'}
            except Exception as e:
                return {'healthy': False, 'reason': f'Model error: {str(e)}'}
                
        except Exception as e:
            return {'healthy': False, 'reason': f'Health check failed: {str(e)}'}
    
    async def _handle_unhealthy_model(self, model_name: str, health_info: Dict):
        """Handle unhealthy model detection."""
        logger.error(f"Unhealthy model detected: {model_name} - {health_info}")
        
        # Attempt to restart model
        try:
            await self.inference_engine.unload_model(model_name)
            await asyncio.sleep(5)
            await self.inference_engine.load_model(model_name, model_type='auto')
            logger.info(f"Restarted unhealthy model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to restart model {model_name}: {e}")
    
    async def _handle_model_drift(self, model: Dict, drift_result):
        """Handle detected model drift."""
        logger.warning(f"Model drift detected: {model['name']} - {drift_result.drift_types}")
        
        if drift_result.recommendation == 'retrain':
            # Trigger automatic retraining
            await self.lifecycle_manager.trigger_retraining(
                model_type=model['type'],
                force=True
            )
            logger.info(f"Triggered automatic retraining for {model['name']}")

# Global integrator instance
ml_integrator = MLNautilusIntegrator()

# FastAPI startup/shutdown handlers
async def startup_ml_integration():
    """Startup handler for ML integration."""
    try:
        await ml_integrator.start_background_tasks()
        logger.info("ML integration started successfully")
    except Exception as e:
        logger.error(f"Failed to start ML integration: {e}")
        raise

async def shutdown_ml_integration():
    """Shutdown handler for ML integration."""
    try:
        await ml_integrator.stop_background_tasks()
        logger.info("ML integration shutdown complete")
    except Exception as e:
        logger.error(f"Error during ML integration shutdown: {e}")

# Export for use in main app
__all__ = ['MLNautilusIntegrator', 'ml_integrator', 'startup_ml_integration', 'shutdown_ml_integration']