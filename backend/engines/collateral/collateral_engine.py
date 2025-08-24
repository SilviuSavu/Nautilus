"""
Collateral Management Engine
===========================

Main orchestrator for the collateral management system, coordinating:
- Margin calculations and monitoring
- Cross-margining optimization
- Regulatory compliance
- Real-time alerts and notifications
- Integration with existing Nautilus infrastructure
"""

import asyncio
import logging
import time
from decimal import Decimal
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timezone

from models import Portfolio, Position, MarginRequirement, MarginAlert, AlertSeverity
from margin_calculator import MarginCalculator
from collateral_optimizer import CollateralOptimizer, OptimizationResult
from margin_monitor import RealTimeMarginMonitor, MonitoringConfig
from regulatory_calculator import RegulatoryCapitalCalculator

# Import existing system components
try:
    from backend.hardware_router import hardware_accelerated, WorkloadType
    HARDWARE_ACCELERATION_AVAILABLE = True
except ImportError:
    HARDWARE_ACCELERATION_AVAILABLE = False
    
    # Create dummy WorkloadType enum
    class WorkloadType:
        RISK_CALCULATION = "risk_calculation"
        MONTE_CARLO = "monte_carlo"
        ML_INFERENCE = "ml_inference"
    
    def hardware_accelerated(workload_type, **kwargs):
        def decorator(func):
            return func
        return decorator

# Import Risk Engine integration
try:
    from backend.engines.risk.enhanced_risk_api import EnhancedRiskEngine
    RISK_ENGINE_AVAILABLE = True
except ImportError:
    RISK_ENGINE_AVAILABLE = False

logger = logging.getLogger(__name__)


class CollateralManagementEngine:
    """
    Main collateral management engine that orchestrates all collateral-related operations.
    
    This engine serves as the central hub for:
    - Real-time margin monitoring and calculations
    - Cross-margining optimization for capital efficiency
    - Regulatory compliance across multiple jurisdictions
    - Integration with existing Nautilus Risk Engine
    - M4 Max hardware acceleration for performance
    """
    
    def __init__(self, 
                 jurisdiction: str = "US",
                 entity_type: str = "hedge_fund",
                 monitoring_config: Optional[MonitoringConfig] = None):
        
        # Core components
        self.margin_calculator = MarginCalculator()
        self.collateral_optimizer = CollateralOptimizer()
        self.margin_monitor = RealTimeMarginMonitor(monitoring_config)
        self.regulatory_calculator = RegulatoryCapitalCalculator(jurisdiction, entity_type)
        
        # Integration with existing systems
        self.risk_engine = None
        if RISK_ENGINE_AVAILABLE:
            try:
                self.risk_engine = EnhancedRiskEngine()
                logger.info("Risk Engine integration enabled")
            except Exception as e:
                logger.warning(f"Risk Engine integration failed: {e}")
        
        # Engine state
        self.active_portfolios = {}
        self.optimization_cache = {}
        self.performance_metrics = {
            'calculations_performed': 0,
            'alerts_generated': 0,
            'optimizations_run': 0,
            'average_calculation_time_ms': 0.0
        }
        
        # Event callbacks
        self.alert_callbacks = []
        self.margin_update_callbacks = []
        
        logger.info("Collateral Management Engine initialized successfully")
    
    async def initialize(self):
        """Initialize the collateral management engine"""
        try:
            # Initialize components
            await self._initialize_components()
            
            logger.info("Collateral Management Engine initialization complete")
            
        except Exception as e:
            logger.error(f"Failed to initialize Collateral Management Engine: {e}")
            raise
    
    async def _initialize_components(self):
        """Initialize all engine components"""
        
        # Initialize optimization cache
        self.optimization_cache.clear()
        
        # Set up callbacks
        self.alert_callbacks.append(self._handle_margin_alert)
        
        logger.debug("All components initialized successfully")
    
    @hardware_accelerated(WorkloadType.RISK_CALCULATION, data_size=5000)
    async def calculate_portfolio_margin(self, portfolio: Portfolio, optimize: bool = True) -> Dict[str, Any]:
        """
        Calculate comprehensive margin requirements for a portfolio with optimization
        
        Args:
            portfolio: Portfolio to analyze
            optimize: Whether to apply cross-margining optimization
            
        Returns:
            Complete margin analysis including optimization results
        """
        start_time = time.time()
        
        try:
            # Calculate base margin requirements
            margin_req = await self.margin_calculator.calculate_portfolio_margin_requirement(portfolio)
            
            # Apply optimization if requested
            optimization_result = None
            if optimize:
                optimization_result = await self.collateral_optimizer.optimize_portfolio_margin(portfolio)
            
            # Calculate regulatory requirements
            regulatory_req = await self.regulatory_calculator.calculate_comprehensive_regulatory_capital(portfolio)
            
            # Integration with Risk Engine for enhanced analysis
            risk_analysis = None
            if self.risk_engine:
                try:
                    risk_analysis = await self._integrate_with_risk_engine(portfolio, margin_req)
                except Exception as e:
                    logger.warning(f"Risk engine integration failed: {e}")
            
            calculation_time = (time.time() - start_time) * 1000
            
            # Update performance metrics
            self.performance_metrics['calculations_performed'] += 1
            self.performance_metrics['average_calculation_time_ms'] = (
                (self.performance_metrics['average_calculation_time_ms'] * 
                 (self.performance_metrics['calculations_performed'] - 1) + calculation_time) /
                self.performance_metrics['calculations_performed']
            )
            
            result = {
                'portfolio_id': portfolio.id,
                'margin_requirement': {
                    'total_margin': float(margin_req.total_margin_requirement),
                    'net_initial_margin': float(margin_req.net_initial_margin),
                    'variation_margin': float(margin_req.variation_margin),
                    'cross_margin_offset': float(margin_req.cross_margin_offset),
                    'margin_utilization': float(margin_req.margin_utilization),
                    'margin_utilization_percent': margin_req.margin_utilization_percent,
                    'margin_excess': float(margin_req.margin_excess),
                    'time_to_margin_call_minutes': margin_req.time_to_margin_call_minutes,
                    'position_margins': {
                        pos_id: {k: float(v) for k, v in margins.items()}
                        for pos_id, margins in margin_req.position_margins.items()
                    }
                },
                'regulatory_capital': {
                    'basel_iii_requirement': float(regulatory_req.basel_iii_requirement),
                    'dodd_frank_requirement': float(regulatory_req.dodd_frank_requirement),
                    'emir_requirement': float(regulatory_req.emir_requirement),
                    'total_regulatory_capital': float(regulatory_req.total_regulatory_capital),
                    'capital_adequacy_ratio': float(regulatory_req.capital_adequacy_ratio),
                    'is_compliant': regulatory_req.is_compliant
                },
                'calculation_metadata': {
                    'calculation_time_ms': calculation_time,
                    'hardware_acceleration': HARDWARE_ACCELERATION_AVAILABLE,
                    'risk_engine_integration': risk_analysis is not None,
                    'calculated_at': datetime.now(timezone.utc).isoformat()
                }
            }
            
            # Add optimization results if available
            if optimization_result:
                result['optimization'] = {
                    'original_margin': float(optimization_result.original_margin),
                    'optimized_margin': float(optimization_result.optimized_margin),
                    'margin_savings': float(optimization_result.margin_savings),
                    'capital_efficiency_improvement': float(optimization_result.capital_efficiency_improvement),
                    'cross_margin_benefits': [
                        {
                            'asset_class': benefit.asset_class.value,
                            'position_count': len(benefit.position_ids),
                            'gross_margin': float(benefit.gross_margin),
                            'cross_margin_offset': float(benefit.cross_margin_offset),
                            'offset_percentage': float(benefit.offset_percentage),
                            'capital_efficiency_improvement': benefit.capital_efficiency_improvement
                        }
                        for benefit in optimization_result.cross_margin_benefits
                    ],
                    'computation_time_ms': optimization_result.computation_time_ms
                }
                
                self.performance_metrics['optimizations_run'] += 1
            
            # Add risk engine analysis if available
            if risk_analysis:
                result['risk_analysis'] = risk_analysis
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating portfolio margin for {portfolio.id}: {e}")
            raise
    
    async def _integrate_with_risk_engine(self, portfolio: Portfolio, margin_req: MarginRequirement) -> Dict[str, Any]:
        """Integrate with existing Risk Engine for enhanced analysis"""
        
        if not self.risk_engine:
            return None
        
        try:
            # Get VaR calculations from Risk Engine
            var_analysis = await self.risk_engine.calculate_portfolio_var(portfolio)
            
            # Get stress test results
            stress_results = await self.risk_engine.run_stress_tests(portfolio)
            
            # Calculate margin-VaR correlation
            margin_var_ratio = float(margin_req.total_margin_requirement / var_analysis.var_95) if var_analysis.var_95 > 0 else 0
            
            return {
                'var_analysis': {
                    'var_95': float(var_analysis.var_95),
                    'var_99': float(var_analysis.var_99),
                    'expected_shortfall': float(var_analysis.expected_shortfall),
                    'margin_var_ratio': margin_var_ratio
                },
                'stress_test_results': [
                    {
                        'scenario': result.scenario_name,
                        'portfolio_change': float(result.portfolio_change_percent),
                        'margin_impact': float(result.margin_impact) if hasattr(result, 'margin_impact') else 0
                    }
                    for result in stress_results
                ],
                'risk_adjusted_margin': {
                    'recommended_buffer': float(margin_req.total_margin_requirement * Decimal('0.20')),  # 20% buffer
                    'stress_adjusted_margin': float(margin_req.total_margin_requirement * Decimal('1.30'))  # 30% stress buffer
                }
            }
            
        except Exception as e:
            logger.error(f"Risk engine integration error: {e}")
            return None
    
    async def start_real_time_monitoring(self, portfolio: Portfolio, alert_callback: Optional[Callable] = None) -> bool:
        """
        Start real-time margin monitoring for a portfolio
        
        Args:
            portfolio: Portfolio to monitor
            alert_callback: Optional callback for alerts
            
        Returns:
            True if monitoring started successfully
        """
        try:
            # Store portfolio reference
            self.active_portfolios[portfolio.id] = portfolio
            
            # Start monitoring
            await self.margin_monitor.start_monitoring(portfolio, alert_callback)
            
            logger.info(f"Started real-time monitoring for portfolio {portfolio.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start monitoring for portfolio {portfolio.id}: {e}")
            return False
    
    async def stop_real_time_monitoring(self, portfolio_id: str) -> bool:
        """Stop real-time monitoring for a portfolio"""
        try:
            await self.margin_monitor.stop_monitoring(portfolio_id)
            
            if portfolio_id in self.active_portfolios:
                del self.active_portfolios[portfolio_id]
            
            logger.info(f"Stopped monitoring for portfolio {portfolio_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop monitoring for portfolio {portfolio_id}: {e}")
            return False
    
    async def _handle_margin_alert(self, alert: MarginAlert):
        """Handle margin alerts generated by the monitoring system"""
        
        self.performance_metrics['alerts_generated'] += 1
        
        # Log alert
        log_level = logging.ERROR if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY] else logging.WARNING
        logger.log(log_level, f"Margin Alert [{alert.severity.value.upper()}]: {alert.message}")
        
        # Execute alert callbacks
        for callback in self.alert_callbacks:
            try:
                await callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
        
        # For critical alerts, suggest immediate actions
        if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]:
            await self._suggest_emergency_actions(alert)
    
    async def _suggest_emergency_actions(self, alert: MarginAlert):
        """Suggest emergency actions for critical margin alerts"""
        
        if alert.portfolio_id not in self.active_portfolios:
            return
        
        portfolio = self.active_portfolios[alert.portfolio_id]
        
        try:
            # Generate portfolio rebalancing suggestions
            if alert.required_action_amount:
                rebalancing_suggestions = await self.collateral_optimizer.suggest_portfolio_rebalancing(
                    portfolio, alert.required_action_amount
                )
                
                logger.critical(f"EMERGENCY REBALANCING SUGGESTIONS for {alert.portfolio_id}:")
                for suggestion in rebalancing_suggestions['rebalancing_suggestions']:
                    logger.critical(f"  - {suggestion['action'].upper()} {suggestion['reduction_percent']:.1f}% of {suggestion['symbol']}")
                
        except Exception as e:
            logger.error(f"Error generating emergency suggestions: {e}")
    
    async def optimize_collateral_allocation(self, portfolio: Portfolio, available_collateral: Dict[str, Decimal]) -> Dict[str, Any]:
        """Optimize allocation of available collateral"""
        return await self.collateral_optimizer.optimize_collateral_allocation(portfolio, available_collateral)
    
    async def run_margin_stress_test(self, portfolio: Portfolio, scenarios: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run comprehensive margin stress tests"""
        
        scenarios = scenarios or ["Market Crash", "Volatility Spike", "Liquidity Crisis"]
        stress_results = []
        
        for scenario in scenarios:
            try:
                result = await self.margin_calculator.run_margin_stress_test(portfolio, scenario)
                stress_results.append({
                    'scenario': scenario,
                    'base_margin': float(result.base_margin),
                    'stressed_margin': float(result.stressed_margin),
                    'margin_increase': float(result.margin_increase),
                    'margin_increase_percent': float(result.margin_increase_percent * 100),
                    'positions_at_risk': result.positions_at_risk,
                    'estimated_liquidation_value': float(result.estimated_liquidation_value),
                    'passes_test': result.passes_stress_test
                })
            except Exception as e:
                logger.error(f"Stress test failed for scenario {scenario}: {e}")
        
        # Overall stress test assessment
        all_passed = all(result['passes_test'] for result in stress_results)
        max_margin_increase = max((result['margin_increase_percent'] for result in stress_results), default=0)
        
        return {
            'portfolio_id': portfolio.id,
            'stress_test_results': stress_results,
            'overall_assessment': {
                'all_scenarios_passed': all_passed,
                'max_margin_increase_percent': max_margin_increase,
                'recommendation': 'PASS' if all_passed else 'INCREASE_BUFFER'
            },
            'tested_scenarios': scenarios,
            'test_timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    async def get_engine_status(self) -> Dict[str, Any]:
        """Get comprehensive engine status and metrics"""
        
        monitoring_status = {}
        for portfolio_id in self.active_portfolios.keys():
            monitoring_status[portfolio_id] = await self.margin_monitor.get_monitoring_status(portfolio_id)
        
        return {
            'engine_status': 'operational',
            'active_portfolios': len(self.active_portfolios),
            'performance_metrics': self.performance_metrics,
            'monitoring_status': monitoring_status,
            'hardware_acceleration': HARDWARE_ACCELERATION_AVAILABLE,
            'risk_engine_integration': RISK_ENGINE_AVAILABLE and self.risk_engine is not None,
            'components': {
                'margin_calculator': 'operational',
                'collateral_optimizer': 'operational',
                'margin_monitor': 'operational',
                'regulatory_calculator': 'operational'
            },
            'uptime': time.time()  # Simple uptime tracking
        }
    
    async def generate_comprehensive_report(self, portfolio: Portfolio) -> Dict[str, Any]:
        """Generate comprehensive collateral management report"""
        
        # Calculate all margin metrics
        margin_analysis = await self.calculate_portfolio_margin(portfolio, optimize=True)
        
        # Run stress tests
        stress_test_results = await self.run_margin_stress_test(portfolio)
        
        # Generate regulatory report
        regulatory_report = await self.regulatory_calculator.generate_regulatory_report(portfolio)
        
        # Get current monitoring status
        monitoring_status = await self.margin_monitor.get_monitoring_status(portfolio.id)
        
        return {
            'portfolio_id': portfolio.id,
            'report_timestamp': datetime.now(timezone.utc).isoformat(),
            'executive_summary': {
                'total_margin_requirement': margin_analysis['margin_requirement']['total_margin'],
                'margin_utilization_percent': margin_analysis['margin_requirement']['margin_utilization_percent'],
                'capital_efficiency_improvement': margin_analysis.get('optimization', {}).get('capital_efficiency_improvement', 0),
                'regulatory_compliant': margin_analysis['regulatory_capital']['is_compliant'],
                'stress_test_passed': stress_test_results['overall_assessment']['all_scenarios_passed']
            },
            'detailed_analysis': {
                'margin_analysis': margin_analysis,
                'stress_test_results': stress_test_results,
                'regulatory_report': regulatory_report,
                'monitoring_status': monitoring_status
            },
            'recommendations': self._generate_recommendations(margin_analysis, stress_test_results, regulatory_report)
        }
    
    def _generate_recommendations(self, margin_analysis: Dict, stress_results: Dict, regulatory_report: Dict) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        # Margin utilization recommendations
        utilization = margin_analysis['margin_requirement']['margin_utilization_percent']
        if utilization > 90:
            recommendations.append("URGENT: Reduce position sizes or add collateral - margin call risk is high")
        elif utilization > 80:
            recommendations.append("Consider reducing leverage or diversifying positions")
        elif utilization < 50:
            recommendations.append("Margin utilization is low - consider increasing leverage for better capital efficiency")
        
        # Optimization recommendations
        if 'optimization' in margin_analysis:
            savings = margin_analysis['optimization']['capital_efficiency_improvement']
            if savings > 10:
                recommendations.append(f"Cross-margining optimization could improve capital efficiency by {savings:.1f}%")
        
        # Stress test recommendations
        if not stress_results['overall_assessment']['all_scenarios_passed']:
            recommendations.append("Portfolio fails some stress scenarios - consider increasing margin buffer")
        
        # Regulatory recommendations
        if not regulatory_report['capital_adequacy']['status'] == 'COMPLIANT':
            recommendations.extend(regulatory_report['recommendations'])
        
        return recommendations
    
    async def shutdown(self):
        """Gracefully shutdown the collateral management engine"""
        logger.info("Shutting down Collateral Management Engine...")
        
        # Stop all monitoring
        for portfolio_id in list(self.active_portfolios.keys()):
            await self.stop_real_time_monitoring(portfolio_id)
        
        # Emergency stop margin monitor
        await self.margin_monitor.emergency_stop_monitoring()
        
        logger.info("Collateral Management Engine shutdown complete")