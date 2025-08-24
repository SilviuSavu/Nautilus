#!/usr/bin/env python3
"""
Risk Engine Services - Business logic for risk management
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import asdict
import numpy as np

from models import RiskLimit, RiskBreach, RiskLimitType, BreachSeverity
from enhanced_messagebus_client import BufferedMessageBusClient, MessagePriority, EnhancedMessageBusConfig
from hybrid_risk_analytics import HybridRiskAnalyticsEngine, ComputationMode
from advanced_risk_analytics import RiskAnalyticsActor
from pyfolio_integration import PyFolioAnalytics
from supervised_knn_optimizer import SupervisedKNNOptimizer, create_supervised_optimizer
from professional_risk_reporter import (
    ProfessionalRiskReporter, 
    create_professional_risk_reporter,
    ReportConfiguration,
    ReportType,
    ReportFormat
)


logger = logging.getLogger(__name__)


class RiskCalculationService:
    """Service for risk calculations and analytics"""
    
    def __init__(self):
        self.active_limits: Dict[str, RiskLimit] = {}
        self.active_breaches: Dict[str, RiskBreach] = {}
        self.risk_checks_processed = 0
        self.breaches_detected = 0
        
    def add_limit(self, limit: RiskLimit):
        """Add a new risk limit"""
        self.active_limits[limit.limit_id] = limit
        
    def remove_limit(self, limit_id: str):
        """Remove a risk limit"""
        if limit_id in self.active_limits:
            del self.active_limits[limit_id]
            
    def check_position_risk(self, portfolio_id: str, position_data: Dict[str, Any]) -> List[RiskBreach]:
        """Check position against all applicable risk limits"""
        breaches = []
        self.risk_checks_processed += 1
        
        for limit in self.active_limits.values():
            if not limit.enabled:
                continue
                
            # Skip if limit doesn't apply to this portfolio
            if limit.portfolio_id and limit.portfolio_id != portfolio_id:
                continue
                
            # Calculate current value based on limit type
            current_value = self._calculate_limit_value(limit, position_data)
            
            # Check for breach
            if current_value > limit.limit_value * limit.threshold_breach:
                breach = self._create_breach(limit, current_value)
                breaches.append(breach)
                self.active_breaches[breach.breach_id] = breach
                self.breaches_detected += 1
                
        return breaches
        
    def _calculate_limit_value(self, limit: RiskLimit, position_data: Dict[str, Any]) -> float:
        """Calculate current value for a specific limit type"""
        if limit.limit_type == RiskLimitType.POSITION_SIZE:
            return abs(position_data.get("quantity", 0))
        elif limit.limit_type == RiskLimitType.PORTFOLIO_VALUE:
            return position_data.get("market_value", 0)
        elif limit.limit_type == RiskLimitType.DAILY_LOSS:
            return abs(position_data.get("unrealized_pnl", 0))
        elif limit.limit_type == RiskLimitType.LEVERAGE:
            return position_data.get("leverage", 1.0)
        else:
            return 0.0
            
    def _create_breach(self, limit: RiskLimit, current_value: float) -> RiskBreach:
        """Create a breach record"""
        breach_percentage = (current_value / limit.limit_value) * 100
        
        # Determine severity
        if breach_percentage >= 150:
            severity = BreachSeverity.CRITICAL
        elif breach_percentage >= 120:
            severity = BreachSeverity.HIGH
        elif breach_percentage >= 100:
            severity = BreachSeverity.MEDIUM
        else:
            severity = BreachSeverity.LOW
            
        breach_id = f"breach_{limit.limit_id}_{int(time.time())}"
        
        return RiskBreach(
            breach_id=breach_id,
            limit_id=limit.limit_id,
            breach_time=datetime.now(),
            severity=severity,
            breach_value=current_value,
            limit_value=limit.limit_value,
            breach_percentage=breach_percentage
        )
        
    def resolve_breach(self, breach_id: str, resolution_data: Dict[str, Any]):
        """Resolve a breach"""
        if breach_id in self.active_breaches:
            breach = self.active_breaches[breach_id]
            breach.resolved = True
            breach.resolution_time = datetime.now()
            breach.action_taken = resolution_data.get("action_taken", "Manual resolution")
            del self.active_breaches[breach_id]


class RiskMonitoringService:
    """Service for continuous risk monitoring"""
    
    def __init__(self, calculation_service: RiskCalculationService, messagebus: BufferedMessageBusClient):
        self.calculation_service = calculation_service
        self.messagebus = messagebus
        self.monitoring_active = False
        self.monitor_task = None
        
    async def start_monitoring(self):
        """Start continuous risk monitoring"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_task = asyncio.create_task(self._continuous_monitoring())
            
    async def stop_monitoring(self):
        """Stop continuous risk monitoring"""
        self.monitoring_active = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
                
    async def _continuous_monitoring(self):
        """Continuous monitoring loop"""
        while self.monitoring_active:
            try:
                # Check for any critical breaches
                critical_breaches = [
                    breach for breach in self.calculation_service.active_breaches.values()
                    if breach.severity == BreachSeverity.CRITICAL and not breach.resolved
                ]
                
                if critical_breaches:
                    # Publish critical alerts
                    for breach in critical_breaches:
                        await self.messagebus.publish(
                            "risk.breach.critical",
                            asdict(breach),
                            priority=MessagePriority.URGENT
                        )
                        
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(5)  # Back off on error


class RiskAnalyticsService:
    """Service for advanced risk analytics"""
    
    def __init__(self):
        self.hybrid_engine = None
        self.pyfolio = None
        self.supervised_optimizer = None
        self.professional_reporter = None
        
    async def initialize(self):
        """Initialize all analytics components"""
        try:
            # Initialize Hybrid Analytics Engine
            self.hybrid_engine = await create_production_hybrid_engine()
            
            # Initialize PyFolio Analytics
            self.pyfolio = PyFolioAnalytics()
            
            # Initialize Supervised k-NN Optimizer
            self.supervised_optimizer = create_supervised_optimizer()
            
            # Initialize Professional Risk Reporter
            self.professional_reporter = await create_professional_risk_reporter()
            
            logger.info("Risk analytics services initialized successfully")
            
        except Exception as e:
            logger.error(f"Analytics initialization error: {e}")
            
    async def compute_hybrid_analytics(self, portfolio_id: str, request_data: Dict[str, Any]):
        """Compute hybrid risk analytics"""
        if not self.hybrid_engine:
            raise Exception("Hybrid analytics engine not initialized")
            
        try:
            import pandas as pd
            
            # Extract data from request
            returns = pd.Series(request_data.get("returns", []))
            positions = pd.DataFrame(request_data.get("positions", []))
            
            # Compute analytics
            result = await self.hybrid_engine.compute_portfolio_analytics(
                returns=returns,
                positions=positions,
                computation_mode=ComputationMode(request_data.get("computation_mode", "hybrid_auto"))
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Hybrid analytics error: {e}")
            raise
            
    async def generate_professional_report(self, portfolio_id: str, report_type: str, format_type: str):
        """Generate professional risk report"""
        if not self.professional_reporter:
            raise Exception("Professional reporter not initialized")
            
        try:
            config = ReportConfiguration(
                report_type=ReportType(report_type),
                format=ReportFormat(format_type),
                include_charts=True,
                include_analytics=True
            )
            
            report = await self.professional_reporter.generate_report(
                portfolio_id=portfolio_id,
                config=config
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Professional report generation error: {e}")
            raise