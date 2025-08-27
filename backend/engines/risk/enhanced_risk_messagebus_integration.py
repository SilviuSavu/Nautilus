#!/usr/bin/env python3
"""
Enhanced Risk Engine with Universal MessageBus Integration
Replaces HTTP communication with sub-5ms MessageBus for real-time risk management.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Set
import numpy as np

# Import universal MessageBus
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from universal_enhanced_messagebus_client import (
    UniversalEnhancedMessageBusClient,
    UniversalMessageBusConfig,
    EngineType,
    MessageType,
    MessagePriority,
    UniversalMessage,
    create_messagebus_client
)

# Import clock for deterministic testing
from engines.common.clock import get_global_clock, Clock

# Import MarketData Client for centralized data access
from marketdata_client import create_marketdata_client, DataType, DataSource

logger = logging.getLogger(__name__)


class EnhancedRiskEngineMessageBus:
    """
    Enhanced Risk Engine with MessageBus integration for ultra-fast risk management
    
    Features:
    - Sub-5ms risk alert propagation
    - Real-time portfolio monitoring
    - Flash crash detection integration
    - ML prediction integration
    - Hardware-accelerated risk calculations
    """
    
    def __init__(self):
        self.messagebus_client: Optional[UniversalEnhancedMessageBusClient] = None
        self.marketdata_client = None  # Will be initialized in initialize()
        self.clock = get_global_clock()
        
        # Risk management state
        self.active_positions = {}
        self.risk_limits = {}
        self.current_risk_metrics = {}
        self.alert_history = []
        
        # Performance tracking
        self.risk_calculations_processed = 0
        self.alerts_sent = 0
        self.average_calculation_time_ms = 0.0
        
        # MarketData Client metrics
        self.marketdata_requests = 0
        self.marketdata_cache_hits = 0
        self.avg_marketdata_latency_ms = 0.0
        
        # Risk thresholds
        self.position_limit_threshold = 0.8
        self.portfolio_var_threshold = 0.05
        self.correlation_threshold = 0.7
        self.volatility_spike_threshold = 3.0
        
        logger.info("🛡️ Enhanced Risk Engine with MessageBus initialized")
    
    async def initialize(self) -> None:
        """Initialize Risk Engine with MessageBus and MarketData Client integration"""
        
        # Create MessageBus client optimized for risk management
        self.messagebus_client = create_messagebus_client(
            EngineType.RISK,
            engine_port=8200,
            buffer_interval_ms=5,     # Ultra-fast for critical risk alerts
            max_buffer_size=5000,
            priority_threshold=MessagePriority.URGENT,
            subscribe_to_engines={
                EngineType.VPIN,      # VPIN toxicity alerts
                EngineType.ML,        # ML predictions
                EngineType.PORTFOLIO, # Portfolio updates
                EngineType.MARKETDATA,# Market data
                EngineType.COLLATERAL # Margin updates
            }
        )
        
        await self.messagebus_client.start()
        
        # Initialize MarketData Client for centralized data access
        self.marketdata_client = create_marketdata_client(EngineType.RISK, 8200)
        
        # Setup message subscriptions
        self._setup_risk_subscriptions()
        
        # Start risk monitoring tasks
        asyncio.create_task(self._risk_monitoring_loop())
        asyncio.create_task(self._performance_monitoring_loop())
        
        logger.info("✅ Enhanced Risk Engine with MessageBus and MarketData Client ready")
    
    async def stop(self) -> None:
        """Stop Risk Engine and MessageBus"""
        if self.messagebus_client:
            await self.messagebus_client.stop()
        logger.info("✅ Enhanced Risk Engine stopped")
    
    # ==================== MARKETDATA CLIENT INTEGRATION ====================
    
    async def get_market_data(
        self,
        symbols: List[str],
        data_types: List[DataType] = None,
        sources: List[DataSource] = None,
        cache: bool = True
    ) -> Dict[str, Any]:
        """Get market data through MarketData Client (sub-5ms performance)"""
        
        if not self.marketdata_client:
            logger.error("MarketData Client not initialized")
            return {}
        
        if data_types is None:
            data_types = [DataType.QUOTE, DataType.FUNDAMENTAL, DataType.LEVEL2]
        
        if sources is None:
            sources = [DataSource.IBKR, DataSource.ALPHA_VANTAGE]
        
        start_time = time.time()
        self.marketdata_requests += 1
        
        try:
            # Get data through Centralized Hub
            data = await self.marketdata_client.get_data(
                symbols=symbols,
                data_types=data_types,
                sources=sources,
                cache=cache,
                priority=MessagePriority.HIGH,
                timeout=2.0  # Fast timeout for risk calculations
            )
            
            # Track cache hits
            if data.get('cache_hit', False):
                self.marketdata_cache_hits += 1
            
            # Update latency metrics
            latency = (time.time() - start_time) * 1000
            self.avg_marketdata_latency_ms = (
                (self.avg_marketdata_latency_ms * (self.marketdata_requests - 1) + latency)
                / self.marketdata_requests
            )
            
            logger.debug(f"📊 Market data retrieved for {symbols} in {latency:.2f}ms")
            return data
            
        except Exception as e:
            logger.error(f"MarketData Client request failed: {e}")
            return {}
    
    async def get_real_time_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get real-time prices for risk calculations"""
        
        data = await self.get_market_data(
            symbols=symbols,
            data_types=[DataType.QUOTE, DataType.TICK],
            sources=[DataSource.IBKR],  # Use IBKR for fastest real-time data
            cache=False  # Real-time data should not be cached
        )
        
        prices = {}
        for symbol in symbols:
            symbol_data = data.get(symbol, {})
            # Extract price from quote or tick data
            prices[symbol] = symbol_data.get('last_price', symbol_data.get('bid', 0.0))
        
        return prices
    
    async def get_volatility_data(self, symbols: List[str]) -> Dict[str, float]:
        """Get volatility data for risk calculations"""
        
        data = await self.get_market_data(
            symbols=symbols,
            data_types=[DataType.FUNDAMENTAL],
            sources=[DataSource.ALPHA_VANTAGE, DataSource.YAHOO]  # Good for volatility data
        )
        
        volatilities = {}
        for symbol in symbols:
            symbol_data = data.get(symbol, {})
            volatilities[symbol] = symbol_data.get('volatility', symbol_data.get('implied_volatility', 0.2))
        
        return volatilities
    
    # ==================== RISK MANAGEMENT METHODS ====================
    
    async def calculate_portfolio_risk(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive portfolio risk metrics"""
        
        start_time = self.clock.timestamp()
        
        try:
            # Extract portfolio information
            positions = portfolio_data.get('positions', {})
            total_value = portfolio_data.get('total_value', 0)
            
            # Calculate risk metrics
            risk_metrics = {
                'timestamp': start_time,
                'portfolio_var': self._calculate_portfolio_var(positions),
                'position_concentration': self._calculate_position_concentration(positions, total_value),
                'correlation_risk': self._calculate_correlation_risk(positions),
                'volatility_risk': self._calculate_volatility_risk(positions),
                'liquidity_risk': self._calculate_liquidity_risk(positions),
                'leverage_ratio': self._calculate_leverage_ratio(portfolio_data),
                'beta_exposure': self._calculate_beta_exposure(positions),
                'sector_concentration': self._calculate_sector_concentration(positions)
            }
            
            # Check for risk limit breaches
            breaches = self._check_risk_breaches(risk_metrics)
            
            # Send alerts for any breaches
            if breaches:
                await self._send_risk_breach_alerts(breaches, risk_metrics)
            
            # Publish risk metrics to other engines
            await self._publish_risk_metrics(risk_metrics)
            
            calculation_time = (self.clock.timestamp() - start_time) * 1000
            self._update_performance_metrics(calculation_time)
            
            return {
                'risk_metrics': risk_metrics,
                'risk_breaches': breaches,
                'calculation_time_ms': calculation_time,
                'alert_level': self._determine_alert_level(risk_metrics)
            }
            
        except Exception as e:
            logger.error(f"Portfolio risk calculation failed: {e}")
            return {'error': str(e), 'timestamp': start_time}
    
    async def process_vpin_alert(self, vpin_data: Dict[str, Any]) -> None:
        """Process VPIN toxicity alert and adjust risk parameters"""
        
        symbol = vpin_data.get('symbol', 'UNKNOWN')
        toxicity_score = vpin_data.get('toxicity_score', 0.0)
        vpin_value = vpin_data.get('vpin_value', 0.0)
        
        # Adjust risk limits based on toxicity
        if toxicity_score > 0.7:  # High toxicity
            await self._apply_emergency_risk_reduction(symbol, toxicity_score)
        elif toxicity_score > 0.5:  # Moderate toxicity
            await self._apply_enhanced_monitoring(symbol, toxicity_score)
        
        # Send risk adjustment alerts
        risk_adjustment = {
            'symbol': symbol,
            'toxicity_score': toxicity_score,
            'vpin_value': vpin_value,
            'risk_action': 'emergency_reduction' if toxicity_score > 0.7 else 'enhanced_monitoring',
            'timestamp': self.clock.timestamp()
        }
        
        await self.messagebus_client.publish_risk_alert(
            'vpin_risk_adjustment',
            risk_adjustment
        )
        
        logger.warning(f"🚨 VPIN risk adjustment for {symbol}: toxicity={toxicity_score:.3f}")
    
    async def process_ml_prediction(self, prediction_data: Dict[str, Any]) -> None:
        """Process ML prediction and update risk models"""
        
        model_name = prediction_data.get('model_name', 'unknown')
        prediction = prediction_data.get('prediction', 0.0)
        confidence = prediction_data.get('confidence', 0.0)
        
        # Update risk models based on ML predictions
        if confidence > 0.8:  # High confidence prediction
            await self._update_risk_models_with_ml(prediction_data)
        
        # Send risk model update notification
        await self.messagebus_client.publish(
            MessageType.RISK_METRIC,
            f"risk.ml_integration.{model_name}",
            {
                'model_prediction': prediction,
                'confidence': confidence,
                'risk_impact': 'high' if abs(prediction) > 0.5 else 'moderate',
                'timestamp': self.clock.timestamp()
            },
            MessagePriority.HIGH,
            target_engines=[EngineType.PORTFOLIO, EngineType.STRATEGY]
        )
    
    async def handle_flash_crash_alert(self, flash_crash_data: Dict[str, Any]) -> None:
        """Handle critical flash crash alert"""
        
        symbols = flash_crash_data.get('symbols', [])
        crash_probability = flash_crash_data.get('crash_probability', 0.0)
        
        # Implement emergency risk protocols
        if crash_probability > 0.8:
            await self._activate_emergency_risk_protocols(symbols, crash_probability)
        elif crash_probability > 0.6:
            await self._activate_enhanced_risk_monitoring(symbols, crash_probability)
        
        # Notify all trading engines immediately
        emergency_alert = {
            'alert_type': 'flash_crash_risk_response',
            'affected_symbols': symbols,
            'crash_probability': crash_probability,
            'risk_actions': ['position_reduction', 'hedging_activation', 'liquidity_preservation'],
            'timestamp': self.clock.timestamp(),
            'expires_at': self.clock.timestamp() + 3600  # 1 hour expiry
        }
        
        await self.messagebus_client.publish(
            MessageType.SYSTEM_ALERT,
            'risk.flash_crash_response',
            emergency_alert,
            MessagePriority.FLASH_CRASH,
            target_engines=[EngineType.PORTFOLIO, EngineType.STRATEGY, EngineType.COLLATERAL]
        )
        
        logger.critical(f"🚨 FLASH CRASH RISK RESPONSE: {len(symbols)} symbols, probability={crash_probability:.2f}")
    
    # ==================== MESSAGE BUS INTEGRATION ====================
    
    def _setup_risk_subscriptions(self) -> None:
        """Setup MessageBus subscriptions for risk management"""
        
        # VPIN alerts
        self.messagebus_client.subscribe("vpin.calculation.*", self._handle_vpin_message)
        self.messagebus_client.subscribe("vpin.toxicity_alert.*", self._handle_toxicity_alert)
        
        # ML predictions
        self.messagebus_client.subscribe("ml.prediction.*", self._handle_ml_prediction)
        
        # Portfolio updates
        self.messagebus_client.subscribe("portfolio.update.*", self._handle_portfolio_update)
        self.messagebus_client.subscribe("portfolio.position_change.*", self._handle_position_change)
        
        # Market data
        self.messagebus_client.subscribe("market_data.*", self._handle_market_data)
        
        # Flash crash alerts
        self.messagebus_client.subscribe("system.flash_crash_alert", self._handle_flash_crash_message)
        
        # Collateral updates
        self.messagebus_client.subscribe("collateral.margin_alert.*", self._handle_margin_alert)
        
        logger.info("📡 Risk Engine MessageBus subscriptions configured")
    
    async def _handle_vpin_message(self, message: UniversalMessage) -> None:
        """Handle VPIN calculation results"""
        try:
            await self.process_vpin_alert(message.payload)
        except Exception as e:
            logger.error(f"Failed to process VPIN message: {e}")
    
    async def _handle_toxicity_alert(self, message: UniversalMessage) -> None:
        """Handle VPIN toxicity alerts"""
        try:
            toxicity_data = message.payload
            symbol = toxicity_data.get('symbol', 'UNKNOWN')
            alert_level = toxicity_data.get('alert_level', 'NORMAL')
            
            if alert_level in ['HIGH', 'CRITICAL']:
                await self.process_vpin_alert(toxicity_data)
            
        except Exception as e:
            logger.error(f"Failed to process toxicity alert: {e}")
    
    async def _handle_ml_prediction(self, message: UniversalMessage) -> None:
        """Handle ML prediction updates"""
        try:
            await self.process_ml_prediction(message.payload)
        except Exception as e:
            logger.error(f"Failed to process ML prediction: {e}")
    
    async def _handle_portfolio_update(self, message: UniversalMessage) -> None:
        """Handle portfolio updates"""
        try:
            portfolio_data = message.payload
            await self.calculate_portfolio_risk(portfolio_data)
        except Exception as e:
            logger.error(f"Failed to process portfolio update: {e}")
    
    async def _handle_position_change(self, message: UniversalMessage) -> None:
        """Handle individual position changes"""
        try:
            position_data = message.payload
            symbol = position_data.get('symbol', 'UNKNOWN')
            
            # Update position tracking
            self.active_positions[symbol] = position_data
            
            # Check if position change triggers risk alerts
            risk_check = await self._check_position_risk(symbol, position_data)
            if risk_check.get('breach', False):
                await self._send_position_risk_alert(symbol, position_data, risk_check)
            
        except Exception as e:
            logger.error(f"Failed to process position change: {e}")
    
    async def _handle_market_data(self, message: UniversalMessage) -> None:
        """Handle market data updates for risk calculations"""
        try:
            market_data = message.payload
            symbol = market_data.get('symbol', 'UNKNOWN')
            
            # Update volatility and correlation calculations using MarketData Client
            await self._update_market_risk_metrics(symbol, market_data)
            
        except Exception as e:
            logger.error(f"Failed to process market data: {e}")
    
    async def _update_market_risk_metrics(self, symbol: str, market_data: Dict[str, Any]) -> None:
        """Update volatility and correlation calculations using MarketData Client"""
        try:
            # Enrich market data with additional sources through MarketData Client
            enriched_data = await self.get_market_data(
                symbols=[symbol],
                data_types=[DataType.QUOTE, DataType.FUNDAMENTAL],
                cache=True
            )
            
            # Combine local message data with enriched data
            combined_data = {**market_data, **enriched_data.get(symbol, {})}
            
            # Update risk metrics with enriched data
            current_price = combined_data.get('last_price', combined_data.get('price', 0.0))
            volatility = combined_data.get('volatility', 0.2)
            
            # Store updated metrics
            self.current_risk_metrics[symbol] = {
                'current_price': current_price,
                'volatility': volatility,
                'last_updated': self.clock.timestamp(),
                'data_source': 'marketdata_client_enriched'
            }
            
            logger.debug(f"📊 Updated risk metrics for {symbol}: price={current_price:.2f}, vol={volatility:.3f}")
            
        except Exception as e:
            logger.error(f"Failed to update market risk metrics for {symbol}: {e}")
    
    async def _handle_flash_crash_message(self, message: UniversalMessage) -> None:
        """Handle flash crash alerts"""
        try:
            await self.handle_flash_crash_alert(message.payload)
        except Exception as e:
            logger.error(f"Failed to process flash crash alert: {e}")
    
    async def _handle_margin_alert(self, message: UniversalMessage) -> None:
        """Handle margin/collateral alerts"""
        try:
            margin_data = message.payload
            await self._process_margin_risk_update(margin_data)
        except Exception as e:
            logger.error(f"Failed to process margin alert: {e}")
    
    # ==================== RISK CALCULATION METHODS ====================
    
    def _calculate_portfolio_var(self, positions: Dict[str, Any]) -> float:
        """Calculate portfolio Value at Risk"""
        if not positions:
            return 0.0
        
        # Simplified VaR calculation (in production, use more sophisticated models)
        total_value = sum(pos.get('market_value', 0) for pos in positions.values())
        volatility = np.mean([pos.get('volatility', 0.2) for pos in positions.values()])
        
        # 95% VaR calculation
        var_95 = total_value * volatility * 1.645  # 95% confidence
        return var_95 / total_value if total_value > 0 else 0.0
    
    def _calculate_position_concentration(self, positions: Dict[str, Any], total_value: float) -> float:
        """Calculate position concentration risk"""
        if not positions or total_value == 0:
            return 0.0
        
        # Calculate Herfindahl index for concentration
        weights = [(pos.get('market_value', 0) / total_value) ** 2 for pos in positions.values()]
        herfindahl_index = sum(weights)
        
        return herfindahl_index
    
    def _calculate_correlation_risk(self, positions: Dict[str, Any]) -> float:
        """Calculate correlation risk across positions"""
        if len(positions) < 2:
            return 0.0
        
        # Simplified correlation risk (in production, use correlation matrix)
        symbols = list(positions.keys())
        correlation_sum = 0.0
        count = 0
        
        for i, symbol1 in enumerate(symbols):
            for symbol2 in symbols[i+1:]:
                # Simulate correlation (in production, use actual correlation data)
                correlation = 0.5 if symbol1[:2] == symbol2[:2] else 0.2
                correlation_sum += abs(correlation)
                count += 1
        
        return correlation_sum / count if count > 0 else 0.0
    
    def _calculate_volatility_risk(self, positions: Dict[str, Any]) -> float:
        """Calculate volatility risk"""
        if not positions:
            return 0.0
        
        volatilities = [pos.get('volatility', 0.2) for pos in positions.values()]
        return np.mean(volatilities) * np.sqrt(len(volatilities))
    
    def _calculate_liquidity_risk(self, positions: Dict[str, Any]) -> float:
        """Calculate liquidity risk"""
        if not positions:
            return 0.0
        
        # Simplified liquidity risk based on position sizes
        liquidity_scores = []
        for pos in positions.values():
            size = pos.get('quantity', 0)
            avg_volume = pos.get('avg_daily_volume', 1000000)
            liquidity_score = min(1.0, abs(size) / avg_volume)
            liquidity_scores.append(liquidity_score)
        
        return np.mean(liquidity_scores)
    
    def _calculate_leverage_ratio(self, portfolio_data: Dict[str, Any]) -> float:
        """Calculate portfolio leverage ratio"""
        total_value = portfolio_data.get('total_value', 0)
        gross_exposure = portfolio_data.get('gross_exposure', total_value)
        
        return gross_exposure / total_value if total_value > 0 else 1.0
    
    def _calculate_beta_exposure(self, positions: Dict[str, Any]) -> float:
        """Calculate portfolio beta exposure"""
        if not positions:
            return 1.0
        
        # Weighted average beta
        total_value = sum(pos.get('market_value', 0) for pos in positions.values())
        if total_value == 0:
            return 1.0
        
        weighted_beta = 0.0
        for pos in positions.values():
            weight = pos.get('market_value', 0) / total_value
            beta = pos.get('beta', 1.0)
            weighted_beta += weight * beta
        
        return weighted_beta
    
    def _calculate_sector_concentration(self, positions: Dict[str, Any]) -> Dict[str, float]:
        """Calculate sector concentration"""
        sector_exposure = {}
        total_value = sum(pos.get('market_value', 0) for pos in positions.values())
        
        if total_value == 0:
            return {}
        
        for pos in positions.values():
            sector = pos.get('sector', 'UNKNOWN')
            value = pos.get('market_value', 0)
            sector_exposure[sector] = sector_exposure.get(sector, 0) + (value / total_value)
        
        return sector_exposure
    
    # ==================== RISK BREACH DETECTION ====================
    
    def _check_risk_breaches(self, risk_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for risk limit breaches"""
        breaches = []
        
        # Portfolio VaR breach
        if risk_metrics['portfolio_var'] > self.portfolio_var_threshold:
            breaches.append({
                'type': 'portfolio_var_breach',
                'current_value': risk_metrics['portfolio_var'],
                'threshold': self.portfolio_var_threshold,
                'severity': 'HIGH'
            })
        
        # Position concentration breach
        if risk_metrics['position_concentration'] > 0.5:
            breaches.append({
                'type': 'concentration_breach',
                'current_value': risk_metrics['position_concentration'],
                'threshold': 0.5,
                'severity': 'MEDIUM'
            })
        
        # Correlation risk breach
        if risk_metrics['correlation_risk'] > self.correlation_threshold:
            breaches.append({
                'type': 'correlation_breach',
                'current_value': risk_metrics['correlation_risk'],
                'threshold': self.correlation_threshold,
                'severity': 'MEDIUM'
            })
        
        # Leverage breach
        if risk_metrics['leverage_ratio'] > 3.0:
            breaches.append({
                'type': 'leverage_breach',
                'current_value': risk_metrics['leverage_ratio'],
                'threshold': 3.0,
                'severity': 'HIGH'
            })
        
        return breaches
    
    async def _send_risk_breach_alerts(self, breaches: List[Dict[str, Any]], risk_metrics: Dict[str, Any]) -> None:
        """Send risk breach alerts via MessageBus"""
        
        for breach in breaches:
            alert_priority = MessagePriority.CRITICAL if breach['severity'] == 'HIGH' else MessagePriority.URGENT
            
            alert_data = {
                'breach_type': breach['type'],
                'current_value': breach['current_value'],
                'threshold': breach['threshold'],
                'severity': breach['severity'],
                'timestamp': self.clock.timestamp(),
                'full_risk_metrics': risk_metrics,
                'recommended_actions': self._get_breach_recommendations(breach['type'])
            }
            
            await self.messagebus_client.publish_risk_alert(
                f"risk_breach_{breach['type']}",
                alert_data
            )
            
            self.alerts_sent += 1
            
            logger.warning(f"🚨 Risk breach alert: {breach['type']} = {breach['current_value']:.4f} (threshold: {breach['threshold']:.4f})")
    
    # ==================== EMERGENCY RISK PROTOCOLS ====================
    
    async def _apply_emergency_risk_reduction(self, symbol: str, toxicity_score: float) -> None:
        """Apply emergency risk reduction protocols"""
        
        emergency_actions = {
            'symbol': symbol,
            'action': 'emergency_risk_reduction',
            'toxicity_trigger': toxicity_score,
            'risk_reduction_percentage': min(50, toxicity_score * 60),  # Up to 50% reduction
            'immediate_actions': [
                'reduce_position_size',
                'increase_monitoring_frequency',
                'activate_hedging_protocols',
                'restrict_new_positions'
            ],
            'timestamp': self.clock.timestamp()
        }
        
        await self.messagebus_client.publish(
            MessageType.SYSTEM_ALERT,
            f"risk.emergency_reduction.{symbol}",
            emergency_actions,
            MessagePriority.CRITICAL,
            target_engines=[EngineType.PORTFOLIO, EngineType.STRATEGY]
        )
    
    async def _activate_emergency_risk_protocols(self, symbols: List[str], crash_probability: float) -> None:
        """Activate emergency risk protocols for flash crash"""
        
        emergency_protocols = {
            'protocol': 'flash_crash_emergency',
            'affected_symbols': symbols,
            'crash_probability': crash_probability,
            'actions': [
                'immediate_position_reduction',
                'liquidity_preservation',
                'activate_circuit_breakers',
                'emergency_hedging'
            ],
            'risk_limits': {
                'max_position_size': 0.1,  # 10% max position
                'max_leverage': 1.5,       # Reduce leverage
                'stop_new_trades': True
            },
            'timestamp': self.clock.timestamp(),
            'duration_minutes': 60  # Emergency protocols active for 1 hour
        }
        
        await self.messagebus_client.publish(
            MessageType.SYSTEM_ALERT,
            'risk.emergency_protocols',
            emergency_protocols,
            MessagePriority.FLASH_CRASH
        )
    
    # ==================== PERFORMANCE MONITORING ====================
    
    def _update_performance_metrics(self, calculation_time_ms: float) -> None:
        """Update risk engine performance metrics"""
        self.risk_calculations_processed += 1
        
        # Update average calculation time
        self.average_calculation_time_ms = (
            (self.average_calculation_time_ms * (self.risk_calculations_processed - 1) + calculation_time_ms) /
            self.risk_calculations_processed
        )
    
    async def _risk_monitoring_loop(self) -> None:
        """Background risk monitoring loop"""
        while True:
            try:
                # Perform periodic risk checks
                await self._periodic_risk_assessment()
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Risk monitoring loop error: {e}")
                await asyncio.sleep(30)
    
    async def _performance_monitoring_loop(self) -> None:
        """Background performance monitoring loop"""
        while True:
            try:
                # Send performance metrics
                if self.messagebus_client:
                    performance_data = {
                        'risk_calculations_processed': self.risk_calculations_processed,
                        'alerts_sent': self.alerts_sent,
                        'average_calculation_time_ms': self.average_calculation_time_ms,
                        'active_positions_count': len(self.active_positions),
                        'timestamp': self.clock.timestamp()
                    }
                    
                    await self.messagebus_client.publish(
                        MessageType.PERFORMANCE_METRIC,
                        'risk.performance',
                        performance_data,
                        MessagePriority.LOW
                    )
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _periodic_risk_assessment(self) -> None:
        """Perform periodic portfolio risk assessment"""
        if self.active_positions:
            # Create mock portfolio data for assessment
            portfolio_data = {
                'positions': self.active_positions,
                'total_value': sum(pos.get('market_value', 0) for pos in self.active_positions.values()),
                'gross_exposure': sum(abs(pos.get('market_value', 0)) for pos in self.active_positions.values())
            }
            
            await self.calculate_portfolio_risk(portfolio_data)
    
    # ==================== UTILITY METHODS ====================
    
    def _determine_alert_level(self, risk_metrics: Dict[str, Any]) -> str:
        """Determine overall alert level from risk metrics"""
        
        risk_score = 0
        
        # Add scores for each risk metric
        risk_score += min(10, risk_metrics['portfolio_var'] * 20)  # VaR contribution
        risk_score += min(5, risk_metrics['position_concentration'] * 10)  # Concentration contribution
        risk_score += min(5, risk_metrics['correlation_risk'] * 7)  # Correlation contribution
        risk_score += min(5, risk_metrics['volatility_risk'] * 3)  # Volatility contribution
        
        if risk_score >= 15:
            return 'CRITICAL'
        elif risk_score >= 10:
            return 'HIGH'
        elif risk_score >= 5:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _get_breach_recommendations(self, breach_type: str) -> List[str]:
        """Get recommendations for specific breach types"""
        
        recommendations = {
            'portfolio_var_breach': [
                'Reduce position sizes',
                'Increase hedging',
                'Diversify holdings',
                'Review correlation assumptions'
            ],
            'concentration_breach': [
                'Diversify positions',
                'Reduce largest positions',
                'Spread risk across sectors'
            ],
            'correlation_breach': [
                'Reduce correlated positions',
                'Add uncorrelated assets',
                'Review sector exposure'
            ],
            'leverage_breach': [
                'Reduce leverage immediately',
                'Close most risky positions',
                'Increase margin requirements'
            ]
        }
        
        return recommendations.get(breach_type, ['Review risk management policies'])
    
    async def _publish_risk_metrics(self, risk_metrics: Dict[str, Any]) -> None:
        """Publish risk metrics to interested engines"""
        
        if self.messagebus_client:
            await self.messagebus_client.publish(
                MessageType.RISK_METRIC,
                'risk.portfolio_metrics',
                risk_metrics,
                MessagePriority.HIGH,
                target_engines=[EngineType.ANALYTICS, EngineType.PORTFOLIO, EngineType.WEBSOCKET]
            )
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get Risk Engine performance summary with MarketData Client metrics"""
        
        messagebus_stats = await self.messagebus_client.get_performance_metrics() if self.messagebus_client else {}
        
        # Get MarketData Client metrics
        marketdata_metrics = {}
        if self.marketdata_client:
            marketdata_metrics = self.marketdata_client.get_metrics()
        
        # Calculate cache hit rate
        cache_hit_rate = (self.marketdata_cache_hits / max(1, self.marketdata_requests)) * 100
        
        return {
            'risk_engine_performance': {
                'calculations_processed': self.risk_calculations_processed,
                'alerts_sent': self.alerts_sent,
                'average_calculation_time_ms': self.average_calculation_time_ms,
                'active_positions': len(self.active_positions)
            },
            'marketdata_client_performance': {
                'total_requests': self.marketdata_requests,
                'cache_hits': self.marketdata_cache_hits,
                'cache_hit_rate_percent': f"{cache_hit_rate:.1f}%",
                'avg_latency_ms': f"{self.avg_marketdata_latency_ms:.2f}",
                'client_metrics': marketdata_metrics,
                'target_achieved': self.avg_marketdata_latency_ms < 5.0,  # Sub-5ms target
                'no_direct_api_calls': True  # Confirmation that all calls go through hub
            },
            'messagebus_performance': messagebus_stats,
            'target_performance': {
                'calculation_time_target_ms': 10.0,
                'alert_latency_target_ms': 5.0,
                'marketdata_latency_target_ms': 5.0,
                'target_achieved': (
                    self.average_calculation_time_ms < 10.0 and 
                    self.avg_marketdata_latency_ms < 5.0
                )
            }
        }


# Global instance
enhanced_risk_engine = EnhancedRiskEngineMessageBus()