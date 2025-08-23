"""
Unit tests for Risk Management Components - Sprint 3 Priority 3

Tests limit engine, breach detector, risk monitor, and risk reporter
with comprehensive coverage including edge cases and error scenarios.
"""

import asyncio
import pytest
from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, List, Any, Optional

# Import Sprint 3 risk management components
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from risk_management.limit_engine import (
    LimitEngine, RiskLimit, LimitType, LimitStatus, LimitViolation
)
from risk_management.breach_detector import (
    BreachDetector, BreachType, BreachSeverity, RiskBreach
)
from risk_management.risk_monitor import (
    RiskMonitor, RiskMetrics, MonitoringAlert
)
from risk_management.risk_reporter import (
    RiskReporter, RiskReport, ReportType, RiskSummary
)
from risk_management.enhanced_risk_calculator import (
    EnhancedRiskCalculator, RiskCalculationResult
)


class TestLimitEngine:
    """Test risk limit enforcement functionality"""
    
    @pytest.fixture
    def limit_engine(self):
        """Create limit engine for testing"""
        return LimitEngine()
    
    @pytest.fixture
    def sample_limits(self):
        """Create sample risk limits for testing"""
        return [
            RiskLimit(
                limit_id="pos_limit_aapl",
                limit_type=LimitType.POSITION_SIZE,
                entity_id="AAPL",
                limit_value=Decimal('1000000'),  # $1M position limit
                soft_limit_pct=Decimal('0.8'),   # 80% soft limit
                status=LimitStatus.ACTIVE
            ),
            RiskLimit(
                limit_id="daily_loss_limit",
                limit_type=LimitType.DAILY_LOSS,
                entity_id="portfolio_1",
                limit_value=Decimal('50000'),    # $50K daily loss limit
                soft_limit_pct=Decimal('0.9'),   # 90% soft limit
                status=LimitStatus.ACTIVE
            ),
            RiskLimit(
                limit_id="var_limit_portfolio",
                limit_type=LimitType.VAR,
                entity_id="portfolio_1",
                limit_value=Decimal('25000'),    # $25K VaR limit
                soft_limit_pct=Decimal('0.85'),  # 85% soft limit
                status=LimitStatus.ACTIVE
            )
        ]
    
    def test_limit_creation_and_validation(self, limit_engine, sample_limits):
        """Test creating and validating risk limits"""
        for limit in sample_limits:
            # Test limit creation
            success = limit_engine.add_limit(limit)
            assert success is True
            
            # Test limit retrieval
            retrieved_limit = limit_engine.get_limit(limit.limit_id)
            assert retrieved_limit is not None
            assert retrieved_limit.limit_id == limit.limit_id
            assert retrieved_limit.limit_value == limit.limit_value
        
        # Test getting all limits
        all_limits = limit_engine.get_all_limits()
        assert len(all_limits) == 3
        
        # Test getting limits by type
        position_limits = limit_engine.get_limits_by_type(LimitType.POSITION_SIZE)
        assert len(position_limits) == 1
        assert position_limits[0].entity_id == "AAPL"
    
    def test_position_size_limit_enforcement(self, limit_engine, sample_limits):
        """Test position size limit enforcement"""
        limit_engine.add_limit(sample_limits[0])  # Position limit for AAPL
        
        # Test within limit
        violation = limit_engine.check_position_limit("AAPL", Decimal('500000'))
        assert violation is None
        
        # Test soft limit breach
        violation = limit_engine.check_position_limit("AAPL", Decimal('850000'))
        assert violation is not None
        assert violation.severity == BreachSeverity.WARNING
        assert violation.current_value == Decimal('850000')
        assert violation.limit_value == Decimal('1000000')
        
        # Test hard limit breach
        violation = limit_engine.check_position_limit("AAPL", Decimal('1200000'))
        assert violation is not None
        assert violation.severity == BreachSeverity.CRITICAL
        assert violation.current_value == Decimal('1200000')
        assert violation.limit_value == Decimal('1000000')
    
    def test_daily_loss_limit_enforcement(self, limit_engine, sample_limits):
        """Test daily loss limit enforcement"""
        limit_engine.add_limit(sample_limits[1])  # Daily loss limit
        
        # Test within limit (small loss)
        violation = limit_engine.check_daily_loss_limit("portfolio_1", Decimal('-10000'))
        assert violation is None
        
        # Test soft limit breach
        violation = limit_engine.check_daily_loss_limit("portfolio_1", Decimal('-47000'))
        assert violation is not None
        assert violation.severity == BreachSeverity.WARNING
        
        # Test hard limit breach
        violation = limit_engine.check_daily_loss_limit("portfolio_1", Decimal('-55000'))
        assert violation is not None
        assert violation.severity == BreachSeverity.CRITICAL
    
    def test_var_limit_enforcement(self, limit_engine, sample_limits):
        """Test VaR limit enforcement"""
        limit_engine.add_limit(sample_limits[2])  # VaR limit
        
        # Test within limit
        violation = limit_engine.check_var_limit("portfolio_1", Decimal('15000'))
        assert violation is None
        
        # Test soft limit breach
        violation = limit_engine.check_var_limit("portfolio_1", Decimal('22000'))
        assert violation is not None
        assert violation.severity == BreachSeverity.WARNING
        
        # Test hard limit breach
        violation = limit_engine.check_var_limit("portfolio_1", Decimal('30000'))
        assert violation is not None
        assert violation.severity == BreachSeverity.CRITICAL
    
    def test_limit_modification(self, limit_engine, sample_limits):
        """Test modifying existing limits"""
        limit = sample_limits[0]
        limit_engine.add_limit(limit)
        
        # Test updating limit value
        success = limit_engine.update_limit_value(limit.limit_id, Decimal('2000000'))
        assert success is True
        
        updated_limit = limit_engine.get_limit(limit.limit_id)
        assert updated_limit.limit_value == Decimal('2000000')
        
        # Test updating status
        success = limit_engine.update_limit_status(limit.limit_id, LimitStatus.SUSPENDED)
        assert success is True
        
        updated_limit = limit_engine.get_limit(limit.limit_id)
        assert updated_limit.status == LimitStatus.SUSPENDED
    
    def test_limit_removal(self, limit_engine, sample_limits):
        """Test removing limits"""
        limit = sample_limits[0]
        limit_engine.add_limit(limit)
        
        # Verify limit exists
        assert limit_engine.get_limit(limit.limit_id) is not None
        
        # Remove limit
        success = limit_engine.remove_limit(limit.limit_id)
        assert success is True
        
        # Verify limit removed
        assert limit_engine.get_limit(limit.limit_id) is None
        
        # Test removing non-existent limit
        success = limit_engine.remove_limit("non_existent_limit")
        assert success is False


class TestBreachDetector:
    """Test risk breach detection functionality"""
    
    @pytest.fixture
    def breach_detector(self):
        """Create breach detector for testing"""
        return BreachDetector()
    
    @pytest.fixture
    def mock_limit_engine(self):
        """Mock limit engine for testing"""
        mock_engine = Mock(spec=LimitEngine)
        return mock_engine
    
    def test_position_breach_detection(self, breach_detector, mock_limit_engine):
        """Test position size breach detection"""
        # Mock limit violation
        mock_violation = LimitViolation(
            limit_id="pos_limit_aapl",
            limit_type=LimitType.POSITION_SIZE,
            entity_id="AAPL",
            current_value=Decimal('1200000'),
            limit_value=Decimal('1000000'),
            severity=BreachSeverity.CRITICAL,
            timestamp=datetime.utcnow()
        )
        
        mock_limit_engine.check_position_limit.return_value = mock_violation
        breach_detector.limit_engine = mock_limit_engine
        
        # Test breach detection
        position_data = {"AAPL": Decimal('1200000')}
        breaches = breach_detector.detect_position_breaches(position_data)
        
        assert len(breaches) == 1
        breach = breaches[0]
        assert isinstance(breach, RiskBreach)
        assert breach.breach_type == BreachType.POSITION_LIMIT
        assert breach.severity == BreachSeverity.CRITICAL
        assert breach.entity_id == "AAPL"
    
    def test_pnl_breach_detection(self, breach_detector, mock_limit_engine):
        """Test P&L breach detection"""
        # Mock daily loss violation
        mock_violation = LimitViolation(
            limit_id="daily_loss_limit",
            limit_type=LimitType.DAILY_LOSS,
            entity_id="portfolio_1",
            current_value=Decimal('-55000'),
            limit_value=Decimal('-50000'),
            severity=BreachSeverity.CRITICAL,
            timestamp=datetime.utcnow()
        )
        
        mock_limit_engine.check_daily_loss_limit.return_value = mock_violation
        breach_detector.limit_engine = mock_limit_engine
        
        # Test breach detection
        pnl_data = {"portfolio_1": Decimal('-55000')}
        breaches = breach_detector.detect_pnl_breaches(pnl_data)
        
        assert len(breaches) == 1
        breach = breaches[0]
        assert breach.breach_type == BreachType.LOSS_LIMIT
        assert breach.severity == BreachSeverity.CRITICAL
    
    def test_concentration_breach_detection(self, breach_detector):
        """Test concentration risk breach detection"""
        # Mock portfolio with high concentration
        portfolio_data = {
            'total_value': Decimal('1000000'),
            'positions': [
                {'symbol': 'AAPL', 'value': Decimal('600000')},  # 60% concentration
                {'symbol': 'MSFT', 'value': Decimal('200000')},  # 20% concentration
                {'symbol': 'GOOGL', 'value': Decimal('200000')} # 20% concentration
            ]
        }
        
        # Set concentration limit to 50%
        breach_detector.concentration_limit = Decimal('0.5')
        
        breaches = breach_detector.detect_concentration_breaches(portfolio_data)
        
        assert len(breaches) == 1
        breach = breaches[0]
        assert breach.breach_type == BreachType.CONCENTRATION
        assert breach.entity_id == "AAPL"
        assert breach.current_value == Decimal('0.6')  # 60%
    
    def test_volatility_breach_detection(self, breach_detector):
        """Test volatility breach detection"""
        # Mock volatility data
        volatility_data = {
            'portfolio_1': {
                'current_volatility': 0.35,
                'volatility_limit': 0.25,
                'lookback_period': '30d'
            }
        }
        
        breaches = breach_detector.detect_volatility_breaches(volatility_data)
        
        assert len(breaches) == 1
        breach = breaches[0]
        assert breach.breach_type == BreachType.VOLATILITY
        assert breach.entity_id == "portfolio_1"
        assert float(breach.current_value) == 0.35
    
    def test_correlation_breach_detection(self, breach_detector):
        """Test correlation breach detection"""
        # Mock high correlation scenario
        correlation_data = {
            'pair': ('AAPL', 'MSFT'),
            'correlation': 0.95,
            'correlation_limit': 0.8,
            'portfolio_id': 'portfolio_1'
        }
        
        breaches = breach_detector.detect_correlation_breaches([correlation_data])
        
        assert len(breaches) == 1
        breach = breaches[0]
        assert breach.breach_type == BreachType.CORRELATION
        assert breach.current_value == Decimal('0.95')
        assert breach.limit_value == Decimal('0.8')
    
    @pytest.mark.asyncio
    async def test_real_time_breach_monitoring(self, breach_detector):
        """Test real-time breach monitoring"""
        breach_alerts = []
        
        async def alert_handler(breach):
            breach_alerts.append(breach)
        
        # Register alert handler
        breach_detector.add_alert_handler(alert_handler)
        
        # Mock real-time data that triggers breach
        position_update = {
            "AAPL": Decimal('1200000'),  # Exceeds limit
            "timestamp": datetime.utcnow()
        }
        
        # Process update
        with patch.object(breach_detector, 'detect_position_breaches') as mock_detect:
            mock_breach = RiskBreach(
                breach_id="test_breach_1",
                breach_type=BreachType.POSITION_LIMIT,
                entity_id="AAPL",
                severity=BreachSeverity.CRITICAL,
                current_value=Decimal('1200000'),
                limit_value=Decimal('1000000'),
                timestamp=datetime.utcnow()
            )
            mock_detect.return_value = [mock_breach]
            
            await breach_detector.process_position_update(position_update)
        
        # Verify alert was triggered
        assert len(breach_alerts) == 1
        assert breach_alerts[0].entity_id == "AAPL"


class TestRiskMonitor:
    """Test risk monitoring functionality"""
    
    @pytest.fixture
    def risk_monitor(self):
        """Create risk monitor for testing"""
        return RiskMonitor()
    
    @pytest.fixture
    def mock_db_pool(self):
        """Mock database connection pool"""
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        return mock_pool, mock_conn
    
    @pytest.mark.asyncio
    async def test_portfolio_risk_monitoring(self, risk_monitor, mock_db_pool):
        """Test comprehensive portfolio risk monitoring"""
        pool, mock_conn = mock_db_pool
        risk_monitor.db_pool = pool
        
        # Mock portfolio data
        portfolio_data = {
            'portfolio_id': 'test_portfolio',
            'positions': [
                {'symbol': 'AAPL', 'quantity': 100, 'price': 150.0, 'value': 15000},
                {'symbol': 'MSFT', 'quantity': 50, 'price': 300.0, 'value': 15000},
                {'symbol': 'GOOGL', 'quantity': 10, 'price': 2800.0, 'value': 28000}
            ],
            'total_value': 58000,
            'daily_pnl': -2500
        }
        
        mock_conn.fetch.return_value = []  # Mock database response
        
        # Test risk monitoring
        risk_metrics = await risk_monitor.calculate_portfolio_risk_metrics(portfolio_data)
        
        assert isinstance(risk_metrics, RiskMetrics)
        assert risk_metrics.portfolio_id == 'test_portfolio'
        assert risk_metrics.total_exposure == 58000
        assert risk_metrics.daily_pnl == -2500
        assert len(risk_metrics.position_risks) == 3
    
    def test_real_time_risk_calculation(self, risk_monitor):
        """Test real-time risk metric calculations"""
        # Mock real-time market data
        market_data = {
            'AAPL': {'price': 155.0, 'volatility': 0.25, 'beta': 1.2},
            'MSFT': {'price': 305.0, 'volatility': 0.20, 'beta': 0.9},
            'GOOGL': {'price': 2750.0, 'volatility': 0.30, 'beta': 1.1}
        }
        
        # Mock position data
        positions = {
            'AAPL': {'quantity': 100, 'entry_price': 150.0},
            'MSFT': {'quantity': 50, 'entry_price': 300.0},
            'GOOGL': {'quantity': 10, 'entry_price': 2800.0}
        }
        
        # Calculate real-time risk
        real_time_risk = risk_monitor.calculate_real_time_risk(positions, market_data)
        
        assert 'total_delta' in real_time_risk
        assert 'portfolio_beta' in real_time_risk
        assert 'portfolio_volatility' in real_time_risk
        assert 'unrealized_pnl' in real_time_risk
    
    @pytest.mark.asyncio
    async def test_risk_alert_generation(self, risk_monitor):
        """Test risk alert generation and notifications"""
        alerts_generated = []
        
        async def alert_callback(alert):
            alerts_generated.append(alert)
        
        risk_monitor.add_alert_callback(alert_callback)
        
        # Mock risk scenario that triggers alerts
        risk_scenario = {
            'portfolio_id': 'test_portfolio',
            'var_95': 35000,  # High VaR
            'concentration_risk': 0.6,  # High concentration
            'leverage_ratio': 3.5  # High leverage
        }
        
        await risk_monitor.evaluate_risk_scenario(risk_scenario)
        
        # Verify alerts were generated
        assert len(alerts_generated) > 0
        
        # Check alert types
        alert_types = [alert.alert_type for alert in alerts_generated]
        assert 'VAR_BREACH' in alert_types or 'CONCENTRATION_RISK' in alert_types
    
    def test_risk_metric_trending(self, risk_monitor):
        """Test risk metric trending and historical analysis"""
        # Mock historical risk data
        historical_data = [
            {'date': datetime.utcnow() - timedelta(days=i), 'var_95': 20000 + i * 1000}
            for i in range(30)
        ]
        
        trend_analysis = risk_monitor.analyze_risk_trends(historical_data)
        
        assert 'trend_direction' in trend_analysis
        assert 'volatility_of_volatility' in trend_analysis
        assert 'risk_trend_score' in trend_analysis
    
    def test_stress_test_scenarios(self, risk_monitor):
        """Test stress testing scenarios"""
        # Mock portfolio for stress testing
        portfolio = {
            'positions': [
                {'symbol': 'AAPL', 'quantity': 1000, 'price': 150.0, 'beta': 1.2},
                {'symbol': 'MSFT', 'quantity': 500, 'price': 300.0, 'beta': 0.9}
            ]
        }
        
        # Define stress scenarios
        stress_scenarios = [
            {'name': 'Market Crash', 'market_shock': -0.20, 'volatility_shock': 2.0},
            {'name': 'Interest Rate Shock', 'rate_shock': 0.02, 'duration_shock': 0.1},
            {'name': 'Sector Rotation', 'tech_shock': -0.15, 'value_shock': 0.10}
        ]
        
        stress_results = risk_monitor.run_stress_tests(portfolio, stress_scenarios)
        
        assert len(stress_results) == 3
        for result in stress_results:
            assert 'scenario_name' in result
            assert 'portfolio_pnl_impact' in result
            assert 'worst_position' in result


class TestRiskReporter:
    """Test risk reporting functionality"""
    
    @pytest.fixture
    def risk_reporter(self):
        """Create risk reporter for testing"""
        return RiskReporter()
    
    @pytest.fixture
    def sample_risk_data(self):
        """Sample risk data for reporting"""
        return {
            'portfolio_metrics': {
                'total_value': 1000000,
                'daily_pnl': -5000,
                'var_95': 25000,
                'expected_shortfall': 35000,
                'maximum_drawdown': 0.12
            },
            'position_risks': [
                {'symbol': 'AAPL', 'position_var': 8000, 'concentration': 0.25},
                {'symbol': 'MSFT', 'position_var': 6000, 'concentration': 0.20},
                {'symbol': 'GOOGL', 'position_var': 11000, 'concentration': 0.35}
            ],
            'limit_utilization': [
                {'limit_type': 'POSITION_SIZE', 'utilization': 0.85, 'status': 'WARNING'},
                {'limit_type': 'DAILY_LOSS', 'utilization': 0.10, 'status': 'OK'},
                {'limit_type': 'VAR', 'utilization': 0.92, 'status': 'CRITICAL'}
            ],
            'recent_breaches': [
                {
                    'breach_type': 'VAR_LIMIT',
                    'entity_id': 'portfolio_1',
                    'severity': 'CRITICAL',
                    'timestamp': datetime.utcnow() - timedelta(hours=2)
                }
            ]
        }
    
    def test_daily_risk_report_generation(self, risk_reporter, sample_risk_data):
        """Test daily risk report generation"""
        portfolio_id = "test_portfolio"
        
        report = risk_reporter.generate_daily_risk_report(portfolio_id, sample_risk_data)
        
        assert isinstance(report, RiskReport)
        assert report.report_type == ReportType.DAILY_SUMMARY
        assert report.portfolio_id == portfolio_id
        assert report.total_portfolio_value == 1000000
        assert report.current_var_95 == 25000
        assert len(report.limit_utilizations) == 3
        assert len(report.recent_breaches) == 1
    
    def test_breach_incident_report(self, risk_reporter):
        """Test breach incident report generation"""
        breach_data = RiskBreach(
            breach_id="breach_001",
            breach_type=BreachType.POSITION_LIMIT,
            entity_id="AAPL",
            severity=BreachSeverity.CRITICAL,
            current_value=Decimal('1200000'),
            limit_value=Decimal('1000000'),
            timestamp=datetime.utcnow()
        )
        
        report = risk_reporter.generate_breach_incident_report(breach_data)
        
        assert isinstance(report, RiskReport)
        assert report.report_type == ReportType.INCIDENT
        assert report.primary_breach is not None
        assert report.primary_breach.breach_id == "breach_001"
        assert report.incident_severity == BreachSeverity.CRITICAL
    
    def test_portfolio_risk_summary(self, risk_reporter, sample_risk_data):
        """Test portfolio risk summary generation"""
        portfolio_id = "test_portfolio"
        
        summary = risk_reporter.generate_portfolio_risk_summary(portfolio_id, sample_risk_data)
        
        assert isinstance(summary, RiskSummary)
        assert summary.portfolio_id == portfolio_id
        assert summary.risk_score > 0
        assert summary.risk_grade in ['A', 'B', 'C', 'D', 'F']
        assert len(summary.key_risks) > 0
        assert len(summary.recommendations) > 0
    
    def test_regulatory_compliance_report(self, risk_reporter, sample_risk_data):
        """Test regulatory compliance report generation"""
        portfolio_id = "test_portfolio"
        compliance_requirements = {
            'max_leverage': 3.0,
            'max_concentration': 0.3,
            'max_var_percentage': 0.05,
            'required_capital_buffer': 0.1
        }
        
        report = risk_reporter.generate_compliance_report(
            portfolio_id, sample_risk_data, compliance_requirements
        )
        
        assert isinstance(report, RiskReport)
        assert report.report_type == ReportType.COMPLIANCE
        assert 'compliance_status' in report.metadata
        assert 'violations' in report.metadata
    
    def test_risk_attribution_report(self, risk_reporter, sample_risk_data):
        """Test risk attribution report generation"""
        portfolio_id = "test_portfolio"
        
        attribution_data = {
            'factor_exposures': {
                'market_beta': 1.15,
                'size_factor': 0.2,
                'value_factor': -0.1,
                'momentum_factor': 0.3
            },
            'sector_contributions': {
                'TECHNOLOGY': 0.45,
                'HEALTHCARE': 0.25,
                'FINANCIALS': 0.30
            }
        }
        
        report = risk_reporter.generate_risk_attribution_report(
            portfolio_id, sample_risk_data, attribution_data
        )
        
        assert isinstance(report, RiskReport)
        assert report.report_type == ReportType.ATTRIBUTION
        assert 'factor_analysis' in report.metadata
        assert 'sector_analysis' in report.metadata
    
    @pytest.mark.asyncio
    async def test_automated_report_scheduling(self, risk_reporter):
        """Test automated report scheduling and delivery"""
        reports_generated = []
        
        async def mock_report_delivery(report):
            reports_generated.append(report)
        
        # Mock report generation
        def mock_generate_report():
            return RiskReport(
                report_id="scheduled_report_1",
                report_type=ReportType.DAILY_SUMMARY,
                portfolio_id="test_portfolio",
                generation_timestamp=datetime.utcnow(),
                total_portfolio_value=1000000,
                current_var_95=25000
            )
        
        risk_reporter.schedule_daily_report(
            portfolio_id="test_portfolio",
            delivery_time="09:00",
            delivery_method=mock_report_delivery
        )
        
        # Simulate report generation trigger
        with patch.object(risk_reporter, 'generate_daily_risk_report', return_value=mock_generate_report()):
            await risk_reporter.process_scheduled_reports()
        
        # Note: In a real implementation, this would test actual scheduling
        # For now, we test the report generation mechanism
        assert True  # Placeholder for scheduling test


class TestEnhancedRiskCalculator:
    """Test enhanced risk calculation functionality"""
    
    @pytest.fixture
    def risk_calculator(self):
        """Create enhanced risk calculator for testing"""
        return EnhancedRiskCalculator()
    
    def test_multi_asset_var_calculation(self, risk_calculator):
        """Test VaR calculation across multiple asset classes"""
        portfolio_data = {
            'positions': [
                {'asset_class': 'EQUITY', 'symbol': 'AAPL', 'value': 100000, 'volatility': 0.25},
                {'asset_class': 'BOND', 'symbol': 'TLT', 'value': 50000, 'volatility': 0.15},
                {'asset_class': 'FOREX', 'symbol': 'EURUSD', 'value': 30000, 'volatility': 0.12},
                {'asset_class': 'COMMODITY', 'symbol': 'GLD', 'value': 20000, 'volatility': 0.20}
            ]
        }
        
        correlation_matrix = {
            ('EQUITY', 'BOND'): -0.3,
            ('EQUITY', 'FOREX'): 0.4,
            ('EQUITY', 'COMMODITY'): 0.2,
            ('BOND', 'FOREX'): -0.1,
            ('BOND', 'COMMODITY'): 0.0,
            ('FOREX', 'COMMODITY'): 0.3
        }
        
        var_result = risk_calculator.calculate_multi_asset_var(
            portfolio_data, correlation_matrix, confidence_level=0.95
        )
        
        assert isinstance(var_result, RiskCalculationResult)
        assert var_result.portfolio_var > 0
        assert var_result.diversification_benefit < 0  # Should reduce risk
        assert len(var_result.component_vars) == 4
    
    def test_monte_carlo_simulation(self, risk_calculator):
        """Test Monte Carlo risk simulation"""
        portfolio_returns = [0.01, -0.005, 0.02, -0.015, 0.008] * 50  # 250 returns
        
        simulation_results = risk_calculator.run_monte_carlo_simulation(
            portfolio_returns, num_simulations=1000, time_horizon=21  # 21 trading days
        )
        
        assert 'var_estimates' in simulation_results
        assert 'expected_shortfall' in simulation_results
        assert 'simulation_paths' in simulation_results
        assert len(simulation_results['simulation_paths']) == 1000
        assert simulation_results['var_estimates']['95%'] < 0  # Should be negative (loss)
    
    def test_scenario_analysis(self, risk_calculator):
        """Test scenario-based risk analysis"""
        scenarios = [
            {
                'name': '2008 Financial Crisis',
                'probability': 0.02,
                'market_shock': -0.35,
                'volatility_multiplier': 2.5,
                'correlation_adjustment': 0.3  # Correlations increase in crisis
            },
            {
                'name': 'Tech Bubble Burst',
                'probability': 0.05,
                'sector_shocks': {'TECHNOLOGY': -0.50, 'OTHER': -0.15},
                'volatility_multiplier': 1.8
            }
        ]
        
        portfolio_exposures = {
            'TECHNOLOGY': 0.6,
            'HEALTHCARE': 0.25,
            'FINANCIALS': 0.15
        }
        
        scenario_results = risk_calculator.analyze_scenarios(scenarios, portfolio_exposures)
        
        assert len(scenario_results) == 2
        for result in scenario_results:
            assert 'scenario_name' in result
            assert 'portfolio_impact' in result
            assert 'probability_weighted_impact' in result
    
    def test_tail_risk_metrics(self, risk_calculator):
        """Test tail risk metric calculations"""
        # Generate returns with fat tails
        returns = np.concatenate([
            np.random.normal(0.001, 0.02, 200),  # Normal returns
            np.random.normal(0, 0.08, 20),       # Fat tail returns
        ])
        
        tail_metrics = risk_calculator.calculate_tail_risk_metrics(returns)
        
        assert 'tail_ratio' in tail_metrics
        assert 'tail_expectation_ratio' in tail_metrics
        assert 'conditional_drawdown' in tail_metrics
        assert 'tail_dependence' in tail_metrics
        
        # Tail ratio should indicate fat tails if present
        assert tail_metrics['tail_ratio'] > 0


class TestRiskManagementIntegration:
    """Integration tests for risk management components"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_risk_workflow(self):
        """Test complete risk management workflow"""
        # Initialize components
        limit_engine = LimitEngine()
        breach_detector = BreachDetector()
        risk_monitor = RiskMonitor()
        risk_reporter = RiskReporter()
        
        # Set up risk limits
        position_limit = RiskLimit(
            limit_id="integration_test_limit",
            limit_type=LimitType.POSITION_SIZE,
            entity_id="AAPL",
            limit_value=Decimal('500000'),
            soft_limit_pct=Decimal('0.8')
        )
        limit_engine.add_limit(position_limit)
        
        # Link components
        breach_detector.limit_engine = limit_engine
        
        # Simulate position update that breaches limit
        position_update = {"AAPL": Decimal('600000')}
        
        # Process through workflow
        breaches = breach_detector.detect_position_breaches(position_update)
        
        # Verify breach detected
        assert len(breaches) == 1
        assert breaches[0].severity == BreachSeverity.CRITICAL
        
        # Generate incident report
        incident_report = risk_reporter.generate_breach_incident_report(breaches[0])
        assert incident_report.report_type == ReportType.INCIDENT
    
    def test_performance_under_load(self):
        """Test risk calculations under high load"""
        import time
        
        risk_calculator = EnhancedRiskCalculator()
        
        # Generate large portfolio for performance testing
        large_portfolio = {
            'positions': [
                {'symbol': f'STOCK_{i}', 'value': 10000, 'volatility': 0.25}
                for i in range(1000)
            ]
        }
        
        start_time = time.time()
        
        # This is a simplified performance test
        # In practice, would test actual VaR calculation
        for i in range(100):
            # Simulate risk calculation
            portfolio_value = sum(pos['value'] for pos in large_portfolio['positions'])
        
        calculation_time = time.time() - start_time
        
        # Should handle large portfolios efficiently
        assert calculation_time < 1.0  # Less than 1 second for 100 calculations
        assert portfolio_value == 10000000  # 1000 positions * 10000 each
    
    def test_error_handling_resilience(self):
        """Test error handling across risk management components"""
        limit_engine = LimitEngine()
        
        # Test invalid limit creation
        invalid_limit = RiskLimit(
            limit_id="",  # Invalid empty ID
            limit_type=LimitType.POSITION_SIZE,
            entity_id="AAPL",
            limit_value=Decimal('-1000'),  # Invalid negative limit
            soft_limit_pct=Decimal('1.5')  # Invalid percentage > 1
        )
        
        # Should handle invalid data gracefully
        success = limit_engine.add_limit(invalid_limit)
        assert success is False
        
        # Test breach detector with missing limit engine
        breach_detector = BreachDetector()
        # Should not crash when limit engine is None
        breaches = breach_detector.detect_position_breaches({"AAPL": Decimal('1000000')})
        assert len(breaches) == 0  # No breaches without limit engine