"""
Advanced Performance Analytics Engine for Sprint 3 Priority 2
Comprehensive analytics module for real-time performance, risk, strategy, and execution analysis
"""

from .performance_calculator import (
    PerformanceCalculator,
    PerformanceSnapshot,
    PositionPerformance,
    PerformanceMetricType,
    get_performance_calculator,
    init_performance_calculator
)

from .risk_analytics import (
    RiskAnalytics,
    VaRResult,
    ExposureAnalysis,
    CorrelationAnalysis,
    StressTestResult,
    VaRMethod,
    StressScenario,
    get_risk_analytics,
    init_risk_analytics
)

from .strategy_analytics import (
    StrategyAnalytics,
    StrategyPerformance,
    StrategyComparison,
    StrategyAttribution,
    AlphaBetaAnalysis,
    StrategyStatus,
    PerformancePeriod,
    get_strategy_analytics,
    init_strategy_analytics
)

from .execution_analytics import (
    ExecutionAnalytics,
    SlippageAnalysis,
    ExecutionMetrics,
    VenueAnalysis,
    MarketImpactAnalysis,
    TimingAnalysis,
    OrderType,
    OrderSide,
    ExecutionQuality,
    get_execution_analytics,
    init_execution_analytics
)

from .analytics_aggregator import (
    AnalyticsAggregator,
    AggregationJob,
    AggregatedMetrics,
    QueryOptimization,
    AggregationInterval,
    DataType,
    CompressionLevel,
    get_analytics_aggregator,
    init_analytics_aggregator
)

import asyncio
import logging
from typing import Optional
import asyncpg

logger = logging.getLogger(__name__)

class AnalyticsEngine:
    """
    Unified Analytics Engine that coordinates all analytics components
    """
    
    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool
        self.logger = logging.getLogger(__name__)
        
        # Initialize all analytics components
        self.performance_calculator = init_performance_calculator(db_pool)
        self.risk_analytics = init_risk_analytics(db_pool)
        self.strategy_analytics = init_strategy_analytics(db_pool)
        self.execution_analytics = init_execution_analytics(db_pool)
        self.analytics_aggregator = init_analytics_aggregator(db_pool)
        
        self.logger.info("Analytics Engine initialized with all components")
    
    async def get_comprehensive_portfolio_analytics(
        self,
        portfolio_id: str,
        include_risk: bool = True,
        include_performance: bool = True,
        include_execution: bool = True
    ) -> dict:
        """
        Get comprehensive analytics for a portfolio
        
        Args:
            portfolio_id: Portfolio identifier
            include_risk: Include risk analytics
            include_performance: Include performance analytics
            include_execution: Include execution analytics
            
        Returns:
            Comprehensive analytics dictionary
        """
        try:
            analytics_result = {
                'portfolio_id': portfolio_id,
                'timestamp': None,
                'performance': None,
                'risk': None,
                'execution': None
            }
            
            # Performance Analytics
            if include_performance:
                try:
                    performance_snapshot = await self.performance_calculator.calculate_portfolio_metrics(portfolio_id)
                    analytics_result['performance'] = {
                        'snapshot': {
                            'total_pnl': float(performance_snapshot.total_pnl),
                            'unrealized_pnl': float(performance_snapshot.unrealized_pnl),
                            'realized_pnl': float(performance_snapshot.realized_pnl),
                            'total_return': performance_snapshot.total_return,
                            'sharpe_ratio': performance_snapshot.sharpe_ratio,
                            'max_drawdown': performance_snapshot.max_drawdown,
                            'win_rate': performance_snapshot.win_rate,
                            'profit_factor': performance_snapshot.profit_factor,
                            'volatility': performance_snapshot.volatility,
                            'alpha': performance_snapshot.alpha,
                            'beta': performance_snapshot.beta,
                            'information_ratio': performance_snapshot.information_ratio
                        },
                        'attribution': await self.performance_calculator.calculate_performance_attribution(portfolio_id)
                    }
                    analytics_result['timestamp'] = performance_snapshot.timestamp
                except Exception as e:
                    self.logger.warning(f"Performance analytics failed for {portfolio_id}: {e}")
                    analytics_result['performance'] = {'error': str(e)}
            
            # Risk Analytics
            if include_risk:
                try:
                    var_result = await self.risk_analytics.calculate_var(portfolio_id)
                    exposure_analysis = await self.risk_analytics.analyze_portfolio_exposure(portfolio_id)
                    correlation_analysis = await self.risk_analytics.calculate_correlation_analysis(portfolio_id)
                    
                    analytics_result['risk'] = {
                        'var': {
                            'method': var_result.method.value,
                            'confidence_level': var_result.confidence_level,
                            'time_horizon': var_result.time_horizon,
                            'var_amount': float(var_result.var_amount),
                            'expected_shortfall': float(var_result.expected_shortfall),
                            'observations_used': var_result.observations_used
                        },
                        'exposure': {
                            'total_exposure': float(exposure_analysis.total_exposure),
                            'net_exposure': float(exposure_analysis.net_exposure),
                            'gross_exposure': float(exposure_analysis.gross_exposure),
                            'long_exposure': float(exposure_analysis.long_exposure),
                            'short_exposure': float(exposure_analysis.short_exposure),
                            'exposure_by_asset_class': exposure_analysis.exposure_by_asset_class,
                            'exposure_by_sector': exposure_analysis.exposure_by_sector,
                            'concentration_metrics': exposure_analysis.concentration_metrics
                        },
                        'correlation': {
                            'average_correlation': correlation_analysis.average_correlation,
                            'max_correlation': correlation_analysis.max_correlation,
                            'min_correlation': correlation_analysis.min_correlation,
                            'diversification_ratio': correlation_analysis.diversification_ratio,
                            'asset_count': len(correlation_analysis.asset_ids)
                        }
                    }
                except Exception as e:
                    self.logger.warning(f"Risk analytics failed for {portfolio_id}: {e}")
                    analytics_result['risk'] = {'error': str(e)}
            
            # Execution Analytics (if available)
            if include_execution:
                try:
                    execution_metrics = await self.execution_analytics.calculate_execution_metrics()
                    
                    analytics_result['execution'] = {
                        'metrics': {
                            'total_orders': execution_metrics.total_orders,
                            'filled_orders': execution_metrics.filled_orders,
                            'fill_rate': execution_metrics.fill_rate,
                            'avg_execution_time_ms': execution_metrics.avg_execution_time_ms,
                            'avg_slippage_bps': execution_metrics.avg_slippage_bps,
                            'total_slippage_cost': float(execution_metrics.total_slippage_cost),
                            'market_impact_bps': execution_metrics.market_impact_bps
                        }
                    }
                except Exception as e:
                    self.logger.warning(f"Execution analytics failed: {e}")
                    analytics_result['execution'] = {'error': str(e)}
            
            return analytics_result
            
        except Exception as e:
            self.logger.error(f"Error getting comprehensive analytics for {portfolio_id}: {e}")
            raise
    
    async def run_stress_test_suite(
        self,
        portfolio_id: str,
        scenarios: Optional[list] = None
    ) -> dict:
        """
        Run comprehensive stress test suite
        """
        if scenarios is None:
            scenarios = [
                StressScenario.MARKET_CRASH,
                StressScenario.INTEREST_RATE_SHOCK,
                StressScenario.VOLATILITY_SPIKE,
                StressScenario.LIQUIDITY_CRISIS
            ]
        
        try:
            stress_results = {}
            
            for scenario in scenarios:
                try:
                    result = await self.risk_analytics.run_stress_test(portfolio_id, scenario)
                    stress_results[scenario.value] = {
                        'portfolio_impact': float(result.portfolio_impact),
                        'impact_percentage': result.impact_percentage,
                        'positions_affected': result.positions_affected,
                        'worst_position_impact': float(result.worst_position_impact),
                        'var_breach_probability': result.var_breach_probability,
                        'recovery_time_estimate': result.recovery_time_estimate,
                        'stress_factors': result.stress_factors
                    }
                except Exception as e:
                    self.logger.warning(f"Stress test {scenario.value} failed: {e}")
                    stress_results[scenario.value] = {'error': str(e)}
            
            return {
                'portfolio_id': portfolio_id,
                'stress_test_timestamp': None,
                'scenarios': stress_results,
                'summary': {
                    'worst_scenario': max(
                        stress_results.items(),
                        key=lambda x: abs(x[1].get('impact_percentage', 0))
                    )[0] if stress_results else None,
                    'total_scenarios_tested': len(scenarios),
                    'scenarios_completed': len([r for r in stress_results.values() if 'error' not in r])
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error running stress test suite for {portfolio_id}: {e}")
            raise
    
    async def generate_analytics_report(
        self,
        portfolio_id: str,
        report_type: str = "comprehensive"
    ) -> dict:
        """
        Generate comprehensive analytics report
        """
        try:
            if report_type == "comprehensive":
                # Get all analytics
                analytics = await self.get_comprehensive_portfolio_analytics(
                    portfolio_id, 
                    include_risk=True,
                    include_performance=True,
                    include_execution=True
                )
                
                # Run stress tests
                stress_results = await self.run_stress_test_suite(portfolio_id)
                
                # Combine into report
                report = {
                    'report_type': 'comprehensive',
                    'portfolio_id': portfolio_id,
                    'generated_at': analytics.get('timestamp'),
                    'analytics': analytics,
                    'stress_tests': stress_results,
                    'summary': {
                        'overall_score': self._calculate_overall_score(analytics),
                        'risk_level': self._assess_risk_level(analytics.get('risk', {})),
                        'performance_grade': self._grade_performance(analytics.get('performance', {})),
                        'key_insights': self._generate_key_insights(analytics, stress_results)
                    }
                }
                
                return report
            
            else:
                raise ValueError(f"Unsupported report type: {report_type}")
                
        except Exception as e:
            self.logger.error(f"Error generating analytics report for {portfolio_id}: {e}")
            raise
    
    def _calculate_overall_score(self, analytics: dict) -> float:
        """Calculate overall portfolio score (0-100)"""
        try:
            score = 50.0  # Base score
            
            performance = analytics.get('performance', {}).get('snapshot', {})
            risk = analytics.get('risk', {})
            
            # Performance factors (40% weight)
            if 'sharpe_ratio' in performance:
                sharpe = performance['sharpe_ratio']
                score += min(sharpe * 10, 20)  # Max 20 points for Sharpe
            
            if 'total_return' in performance:
                ret = performance['total_return']
                score += min(ret * 50, 15)  # Max 15 points for return
            
            # Risk factors (30% weight)
            if 'var' in risk:
                # Lower VaR is better, so subtract from score
                var_impact = risk['var'].get('var_amount', 0)
                score -= min(abs(var_impact) / 10000, 10)  # Max -10 points
            
            if 'exposure' in risk:
                # Good diversification adds points
                concentration = risk['exposure'].get('concentration_metrics', {})
                if 'effective_positions' in concentration:
                    eff_pos = concentration['effective_positions']
                    score += min(eff_pos / 10, 10)  # Max 10 points
            
            # Execution factors (10% weight)
            execution = analytics.get('execution', {}).get('metrics', {})
            if 'fill_rate' in execution:
                score += execution['fill_rate'] * 5  # Max 5 points
            
            return max(0, min(100, score))
            
        except Exception as e:
            self.logger.warning(f"Error calculating overall score: {e}")
            return 50.0  # Default neutral score
    
    def _assess_risk_level(self, risk_data: dict) -> str:
        """Assess overall risk level"""
        try:
            risk_score = 0
            
            # VaR assessment
            var_data = risk_data.get('var', {})
            if 'var_amount' in var_data:
                var_amount = abs(var_data['var_amount'])
                if var_amount > 50000:
                    risk_score += 3
                elif var_amount > 20000:
                    risk_score += 2
                elif var_amount > 10000:
                    risk_score += 1
            
            # Concentration assessment
            exposure_data = risk_data.get('exposure', {})
            concentration = exposure_data.get('concentration_metrics', {})
            if 'effective_positions' in concentration:
                eff_pos = concentration['effective_positions']
                if eff_pos < 5:
                    risk_score += 2
                elif eff_pos < 10:
                    risk_score += 1
            
            # Correlation assessment
            corr_data = risk_data.get('correlation', {})
            if 'average_correlation' in corr_data:
                avg_corr = corr_data['average_correlation']
                if avg_corr > 0.8:
                    risk_score += 2
                elif avg_corr > 0.6:
                    risk_score += 1
            
            if risk_score >= 5:
                return "HIGH"
            elif risk_score >= 3:
                return "MEDIUM"
            elif risk_score >= 1:
                return "LOW"
            else:
                return "VERY_LOW"
                
        except Exception as e:
            self.logger.warning(f"Error assessing risk level: {e}")
            return "UNKNOWN"
    
    def _grade_performance(self, performance_data: dict) -> str:
        """Grade portfolio performance"""
        try:
            snapshot = performance_data.get('snapshot', {})
            
            grade_points = 0
            
            # Sharpe ratio grading
            sharpe = snapshot.get('sharpe_ratio', 0)
            if sharpe > 2.0:
                grade_points += 4
            elif sharpe > 1.0:
                grade_points += 3
            elif sharpe > 0.5:
                grade_points += 2
            elif sharpe > 0:
                grade_points += 1
            
            # Return grading
            total_return = snapshot.get('total_return', 0)
            if total_return > 0.20:  # 20%+
                grade_points += 3
            elif total_return > 0.10:  # 10%+
                grade_points += 2
            elif total_return > 0.05:  # 5%+
                grade_points += 1
            
            # Drawdown penalty
            max_drawdown = snapshot.get('max_drawdown', 0)
            if max_drawdown < -0.20:  # -20% or worse
                grade_points -= 2
            elif max_drawdown < -0.10:  # -10% or worse
                grade_points -= 1
            
            if grade_points >= 6:
                return "A"
            elif grade_points >= 4:
                return "B"
            elif grade_points >= 2:
                return "C"
            elif grade_points >= 0:
                return "D"
            else:
                return "F"
                
        except Exception as e:
            self.logger.warning(f"Error grading performance: {e}")
            return "N/A"
    
    def _generate_key_insights(self, analytics: dict, stress_results: dict) -> list:
        """Generate key insights from analytics"""
        insights = []
        
        try:
            performance = analytics.get('performance', {}).get('snapshot', {})
            risk = analytics.get('risk', {})
            
            # Performance insights
            sharpe = performance.get('sharpe_ratio', 0)
            if sharpe > 1.5:
                insights.append("Excellent risk-adjusted returns (Sharpe > 1.5)")
            elif sharpe < 0.5:
                insights.append("Poor risk-adjusted returns (Sharpe < 0.5)")
            
            # Risk insights
            var_data = risk.get('var', {})
            if 'var_amount' in var_data:
                var_amount = abs(var_data['var_amount'])
                if var_amount > 100000:
                    insights.append("High Value at Risk - consider risk reduction")
            
            # Concentration insights
            concentration = risk.get('exposure', {}).get('concentration_metrics', {})
            if 'effective_positions' in concentration:
                eff_pos = concentration['effective_positions']
                if eff_pos < 5:
                    insights.append("High concentration risk - consider diversification")
            
            # Stress test insights
            stress_scenarios = stress_results.get('scenarios', {})
            worst_scenario = None
            worst_impact = 0
            
            for scenario, result in stress_scenarios.items():
                if 'impact_percentage' in result:
                    impact = abs(result['impact_percentage'])
                    if impact > worst_impact:
                        worst_impact = impact
                        worst_scenario = scenario
            
            if worst_scenario and worst_impact > 15:
                insights.append(f"Vulnerable to {worst_scenario} scenario ({worst_impact:.1f}% impact)")
            
            return insights
            
        except Exception as e:
            self.logger.warning(f"Error generating insights: {e}")
            return ["Analytics processing completed"]

# Global analytics engine instance
analytics_engine = None

def get_analytics_engine() -> AnalyticsEngine:
    """Get global analytics engine instance"""
    global analytics_engine
    if analytics_engine is None:
        raise RuntimeError("Analytics engine not initialized. Call init_analytics_engine() first.")
    return analytics_engine

def init_analytics_engine(db_pool: asyncpg.Pool) -> AnalyticsEngine:
    """Initialize global analytics engine instance"""
    global analytics_engine
    analytics_engine = AnalyticsEngine(db_pool)
    return analytics_engine

__all__ = [
    # Main engine
    'AnalyticsEngine',
    'get_analytics_engine',
    'init_analytics_engine',
    
    # Performance analytics
    'PerformanceCalculator',
    'PerformanceSnapshot',
    'PositionPerformance',
    'PerformanceMetricType',
    'get_performance_calculator',
    'init_performance_calculator',
    
    # Risk analytics
    'RiskAnalytics',
    'VaRResult',
    'ExposureAnalysis',
    'CorrelationAnalysis',
    'StressTestResult',
    'VaRMethod',
    'StressScenario',
    'get_risk_analytics',
    'init_risk_analytics',
    
    # Strategy analytics
    'StrategyAnalytics',
    'StrategyPerformance',
    'StrategyComparison',
    'StrategyAttribution',
    'AlphaBetaAnalysis',
    'StrategyStatus',
    'PerformancePeriod',
    'get_strategy_analytics',
    'init_strategy_analytics',
    
    # Execution analytics
    'ExecutionAnalytics',
    'SlippageAnalysis',
    'ExecutionMetrics',
    'VenueAnalysis',
    'MarketImpactAnalysis',
    'TimingAnalysis',
    'OrderType',
    'OrderSide',
    'ExecutionQuality',
    'get_execution_analytics',
    'init_execution_analytics',
    
    # Aggregation
    'AnalyticsAggregator',
    'AggregationJob',
    'AggregatedMetrics',
    'QueryOptimization',
    'AggregationInterval',
    'DataType',
    'CompressionLevel',
    'get_analytics_aggregator',
    'init_analytics_aggregator'
]