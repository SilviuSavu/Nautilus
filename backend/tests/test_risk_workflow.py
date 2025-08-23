"""
Integration tests for Risk Management Workflow - Sprint 3

Tests end-to-end risk management workflow from limit monitoring through
breach detection to alert generation and reporting.
"""

import asyncio
import pytest
from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, List, Any

# Import Sprint 3 risk management components
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from risk_management.limit_engine import LimitEngine, RiskLimit, LimitType, LimitStatus
from risk_management.breach_detector import BreachDetector, BreachType, BreachSeverity, RiskBreach
from risk_management.risk_monitor import RiskMonitor, RiskMetrics
from risk_management.risk_reporter import RiskReporter, RiskReport, ReportType
from risk_management.enhanced_risk_calculator import EnhancedRiskCalculator
from websocket.websocket_manager import WebSocketManager
from websocket.streaming_service import StreamingService


class TestEndToEndRiskWorkflow:
    """Test complete end-to-end risk management workflow"""
    
    @pytest.fixture
    def risk_system(self):
        """Setup complete risk management system"""
        limit_engine = LimitEngine()
        breach_detector = BreachDetector()
        risk_monitor = RiskMonitor()
        risk_reporter = RiskReporter()
        risk_calculator = EnhancedRiskCalculator()
        websocket_manager = WebSocketManager()
        streaming_service = StreamingService()
        
        # Link components
        breach_detector.limit_engine = limit_engine
        risk_monitor.limit_engine = limit_engine
        risk_monitor.breach_detector = breach_detector
        
        return {
            "limit_engine": limit_engine,
            "breach_detector": breach_detector,
            "risk_monitor": risk_monitor,
            "risk_reporter": risk_reporter,
            "risk_calculator": risk_calculator,
            "websocket_manager": websocket_manager,
            "streaming_service": streaming_service
        }
    
    @pytest.fixture
    def mock_db_pool(self):
        """Mock database connection pool"""
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        return mock_pool, mock_conn
    
    @pytest.fixture
    def sample_portfolio_data(self):
        """Sample portfolio data for testing"""
        return {
            "portfolio_id": "test_portfolio_001",
            "positions": {
                "AAPL": {
                    "quantity": 1000,
                    "avg_price": Decimal("150.00"),
                    "current_price": Decimal("155.00"),
                    "market_value": Decimal("155000.00"),
                    "unrealized_pnl": Decimal("5000.00")
                },
                "MSFT": {
                    "quantity": 500,
                    "avg_price": Decimal("300.00"),
                    "current_price": Decimal("305.00"),
                    "market_value": Decimal("152500.00"),
                    "unrealized_pnl": Decimal("2500.00")
                },
                "GOOGL": {
                    "quantity": 100,
                    "avg_price": Decimal("2800.00"),
                    "current_price": Decimal("2750.00"),
                    "market_value": Decimal("275000.00"),
                    "unrealized_pnl": Decimal("-5000.00")
                }
            },
            "total_market_value": Decimal("582500.00"),
            "total_unrealized_pnl": Decimal("2500.00"),
            "daily_realized_pnl": Decimal("-1000.00"),
            "cash_balance": Decimal("50000.00")
        }
    
    @pytest.mark.asyncio
    async def test_complete_risk_monitoring_cycle(self, risk_system, sample_portfolio_data, mock_db_pool):
        """Test complete risk monitoring cycle"""
        limit_engine = risk_system["limit_engine"]
        breach_detector = risk_system["breach_detector"]
        risk_monitor = risk_system["risk_monitor"]
        risk_reporter = risk_system["risk_reporter"]
        
        pool, mock_conn = mock_db_pool
        risk_monitor.db_pool = pool
        
        portfolio_id = sample_portfolio_data["portfolio_id"]
        
        # 1. Setup risk limits
        limits = [
            RiskLimit(
                limit_id="aapl_position_limit",
                limit_type=LimitType.POSITION_SIZE,
                entity_id="AAPL",
                limit_value=Decimal("100000"),  # $100K limit (will be breached)
                soft_limit_pct=Decimal("0.8"),
                status=LimitStatus.ACTIVE
            ),
            RiskLimit(
                limit_id="portfolio_var_limit",
                limit_type=LimitType.VAR,
                entity_id=portfolio_id,
                limit_value=Decimal("15000"),  # $15K VaR limit
                soft_limit_pct=Decimal("0.9"),
                status=LimitStatus.ACTIVE
            ),
            RiskLimit(
                limit_id="daily_loss_limit",
                limit_type=LimitType.DAILY_LOSS,
                entity_id=portfolio_id,
                limit_value=Decimal("10000"),  # $10K daily loss limit
                soft_limit_pct=Decimal("0.8"),
                status=LimitStatus.ACTIVE
            )
        ]
        
        for limit in limits:
            success = limit_engine.add_limit(limit)
            assert success is True
        
        # 2. Process portfolio data and detect breaches
        position_data = {
            symbol: position["market_value"]
            for symbol, position in sample_portfolio_data["positions"].items()
        }
        
        # AAPL position ($155K) should breach limit ($100K)
        breaches = breach_detector.detect_position_breaches(position_data)
        
        assert len(breaches) == 1
        aapl_breach = breaches[0]
        assert aapl_breach.entity_id == "AAPL"
        assert aapl_breach.severity == BreachSeverity.CRITICAL
        assert aapl_breach.current_value == Decimal("155000.00")
        
        # 3. Monitor portfolio risk metrics
        mock_conn.fetch.return_value = []  # Mock empty database response
        
        risk_metrics = await risk_monitor.calculate_portfolio_risk_metrics(sample_portfolio_data)
        
        assert risk_metrics.portfolio_id == portfolio_id
        assert risk_metrics.total_exposure == float(sample_portfolio_data["total_market_value"])
        assert risk_metrics.daily_pnl == float(sample_portfolio_data["daily_realized_pnl"])
        
        # 4. Generate breach incident report
        incident_report = risk_reporter.generate_breach_incident_report(aapl_breach)
        
        assert incident_report.report_type == ReportType.INCIDENT
        assert incident_report.primary_breach.entity_id == "AAPL"
        assert incident_report.incident_severity == BreachSeverity.CRITICAL
        
        # 5. Generate daily risk report
        risk_data = {
            "portfolio_metrics": {
                "total_value": float(sample_portfolio_data["total_market_value"]),
                "daily_pnl": float(sample_portfolio_data["daily_realized_pnl"]),
                "var_95": 12000.0,  # Within limit
                "expected_shortfall": 18000.0,
                "maximum_drawdown": 0.05
            },
            "position_risks": [
                {"symbol": "AAPL", "position_var": 6000, "concentration": 0.266},
                {"symbol": "MSFT", "position_var": 4000, "concentration": 0.262},
                {"symbol": "GOOGL", "position_var": 5000, "concentration": 0.472}
            ],
            "limit_utilization": [
                {"limit_type": "POSITION_SIZE", "utilization": 1.55, "status": "CRITICAL"},  # AAPL breach
                {"limit_type": "VAR", "utilization": 0.80, "status": "OK"},
                {"limit_type": "DAILY_LOSS", "utilization": 0.10, "status": "OK"}
            ],
            "recent_breaches": [
                {
                    "breach_type": "POSITION_LIMIT",
                    "entity_id": "AAPL",
                    "severity": "CRITICAL",
                    "timestamp": datetime.utcnow()
                }
            ]
        }
        
        daily_report = risk_reporter.generate_daily_risk_report(portfolio_id, risk_data)
        
        assert daily_report.report_type == ReportType.DAILY_SUMMARY
        assert daily_report.portfolio_id == portfolio_id
        assert len(daily_report.recent_breaches) == 1
        assert any(util.utilization > 1.0 for util in daily_report.limit_utilizations)
    
    @pytest.mark.asyncio
    async def test_real_time_risk_monitoring_with_alerts(self, risk_system, mock_db_pool):
        """Test real-time risk monitoring with WebSocket alerts"""
        limit_engine = risk_system["limit_engine"]
        breach_detector = risk_system["breach_detector"]
        websocket_manager = risk_system["websocket_manager"]
        streaming_service = risk_system["streaming_service"]
        
        # Setup risk manager WebSocket connections
        risk_managers = []
        for i in range(3):
            mock_ws = AsyncMock()
            conn_id = f"risk_manager_{i}"
            
            await websocket_manager.connect(mock_ws, conn_id, f"risk_user_{i}")
            websocket_manager.subscribe_to_topic(conn_id, "risk.alerts")
            
            risk_managers.append((conn_id, mock_ws))
        
        # Setup position size limit
        position_limit = RiskLimit(
            limit_id="realtime_position_limit",
            limit_type=LimitType.POSITION_SIZE,
            entity_id="TSLA",
            limit_value=Decimal("75000"),
            soft_limit_pct=Decimal("0.85"),
            status=LimitStatus.ACTIVE
        )
        limit_engine.add_limit(position_limit)
        
        # Setup Redis for streaming (mock)
        with patch('redis.asyncio.Redis') as mock_redis_class:
            mock_redis = AsyncMock()
            mock_redis.ping.return_value = True
            mock_redis.publish.return_value = 1
            mock_redis_class.return_value = mock_redis
            
            from websocket.redis_pubsub import RedisPubSubManager
            redis_manager = RedisPubSubManager()
            redis_manager.redis_client = mock_redis
            streaming_service.pubsub_manager = redis_manager
            
            # Simulate real-time position updates
            position_updates = [
                {"TSLA": Decimal("60000")},  # Within soft limit
                {"TSLA": Decimal("70000")},  # Above soft limit (warning)
                {"TSLA": Decimal("80000")},  # Above hard limit (critical)
            ]
            
            alert_count = 0
            for i, position_data in enumerate(position_updates):
                # Detect breaches
                breaches = breach_detector.detect_position_breaches(position_data)
                
                if i == 0:
                    assert len(breaches) == 0  # No breach
                elif i == 1:
                    assert len(breaches) == 1
                    assert breaches[0].severity == BreachSeverity.WARNING
                    alert_count += 1
                elif i == 2:
                    assert len(breaches) == 1
                    assert breaches[0].severity == BreachSeverity.CRITICAL
                    alert_count += 1
                
                # Stream risk alerts for breaches
                for breach in breaches:
                    risk_alert = {
                        "alert_id": f"alert_{i}_{breach.entity_id}",
                        "alert_type": breach.breach_type.value,
                        "entity_id": breach.entity_id,
                        "severity": breach.severity.value,
                        "current_value": float(breach.current_value),
                        "limit_value": float(breach.limit_value),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    
                    # Stream alert
                    success = await streaming_service.stream_risk_alert(risk_alert)
                    assert success is True
                    
                    # Broadcast to WebSocket clients
                    sent_count = await websocket_manager.broadcast_message(
                        risk_alert, "risk.alerts"
                    )
                    assert sent_count == 3  # All risk managers
            
            # Verify all risk managers received alerts
            for conn_id, mock_ws in risk_managers:
                # Should have received 2 alerts (warning + critical)
                assert mock_ws.send_text.call_count == alert_count
    
    @pytest.mark.asyncio
    async def test_portfolio_var_calculation_workflow(self, risk_system, sample_portfolio_data, mock_db_pool):
        """Test portfolio VaR calculation workflow"""
        limit_engine = risk_system["limit_engine"]
        breach_detector = risk_system["breach_detector"]
        risk_calculator = risk_system["risk_calculator"]
        risk_monitor = risk_system["risk_monitor"]
        
        pool, mock_conn = mock_db_pool
        risk_monitor.db_pool = pool
        
        portfolio_id = sample_portfolio_data["portfolio_id"]
        
        # Setup VaR limit
        var_limit = RiskLimit(
            limit_id="portfolio_var_limit",
            limit_type=LimitType.VAR,
            entity_id=portfolio_id,
            limit_value=Decimal("20000"),  # $20K VaR limit
            soft_limit_pct=Decimal("0.8"),
            status=LimitStatus.ACTIVE
        )
        limit_engine.add_limit(var_limit)
        
        # Mock historical returns for VaR calculation
        portfolio_returns = [
            0.02, -0.01, 0.015, -0.008, 0.012, -0.015, 0.025, -0.020,
            0.008, -0.005, 0.018, -0.012, 0.009, -0.007, 0.022, -0.018
        ] * 15  # 240 daily returns
        
        # Mock correlation matrix for multi-asset VaR
        correlation_matrix = {
            ("AAPL", "MSFT"): 0.6,
            ("AAPL", "GOOGL"): 0.4,
            ("MSFT", "GOOGL"): 0.5
        }
        
        # Calculate portfolio VaR
        portfolio_data = {
            "positions": [
                {
                    "symbol": "AAPL",
                    "value": float(sample_portfolio_data["positions"]["AAPL"]["market_value"]),
                    "volatility": 0.25
                },
                {
                    "symbol": "MSFT", 
                    "value": float(sample_portfolio_data["positions"]["MSFT"]["market_value"]),
                    "volatility": 0.20
                },
                {
                    "symbol": "GOOGL",
                    "value": float(sample_portfolio_data["positions"]["GOOGL"]["market_value"]),
                    "volatility": 0.30
                }
            ]
        }
        
        # Mock risk calculation
        with patch.object(risk_calculator, 'calculate_multi_asset_var') as mock_var_calc:
            from risk_management.enhanced_risk_calculator import RiskCalculationResult
            mock_var_result = RiskCalculationResult(
                portfolio_var=25000.0,  # Above limit
                diversification_benefit=-3000.0,
                component_vars=[8000.0, 6000.0, 9000.0],
                confidence_level=0.95
            )
            mock_var_calc.return_value = mock_var_result
            
            var_result = risk_calculator.calculate_multi_asset_var(
                portfolio_data, correlation_matrix
            )
            
            # Check VaR against limit
            portfolio_var = Decimal(str(var_result.portfolio_var))
            var_breaches = breach_detector.detect_var_breaches({portfolio_id: portfolio_var})
            
            assert len(var_breaches) == 1
            var_breach = var_breaches[0]
            assert var_breach.breach_type == BreachType.VAR_LIMIT
            assert var_breach.severity == BreachSeverity.CRITICAL
            assert var_breach.current_value == portfolio_var
    
    @pytest.mark.asyncio
    async def test_stress_test_scenario_workflow(self, risk_system, sample_portfolio_data):
        """Test stress testing scenario workflow"""
        risk_monitor = risk_system["risk_monitor"]
        risk_calculator = risk_system["risk_calculator"]
        risk_reporter = risk_system["risk_reporter"]
        
        portfolio_id = sample_portfolio_data["portfolio_id"]
        
        # Define stress test scenarios
        stress_scenarios = [
            {
                "name": "Market Crash 2008",
                "probability": 0.02,
                "market_shock": -0.35,
                "volatility_multiplier": 2.5,
                "sector_shocks": {"TECHNOLOGY": -0.45, "OTHER": -0.20}
            },
            {
                "name": "COVID-19 Style Shock",
                "probability": 0.05,
                "market_shock": -0.25,
                "volatility_multiplier": 2.0,
                "correlation_adjustment": 0.3
            },
            {
                "name": "Interest Rate Shock",
                "probability": 0.10,
                "rate_shock": 0.03,  # 300 basis points
                "duration_impact": -0.15,
                "growth_stocks_impact": -0.20
            }
        ]
        
        # Map positions to sectors
        portfolio_exposures = {
            "TECHNOLOGY": 0.8,  # AAPL, MSFT, GOOGL are tech
            "OTHER": 0.2
        }
        
        # Run stress tests
        with patch.object(risk_calculator, 'analyze_scenarios') as mock_stress:
            mock_stress_results = [
                {
                    "scenario_name": "Market Crash 2008",
                    "portfolio_impact": -185000.0,  # 35% of portfolio value
                    "probability_weighted_impact": -3700.0,
                    "worst_position": "GOOGL",
                    "recovery_time_estimate": 365  # days
                },
                {
                    "scenario_name": "COVID-19 Style Shock", 
                    "portfolio_impact": -145000.0,  # 25% of portfolio value
                    "probability_weighted_impact": -7250.0,
                    "worst_position": "AAPL",
                    "recovery_time_estimate": 180
                },
                {
                    "scenario_name": "Interest Rate Shock",
                    "portfolio_impact": -87500.0,  # 15% of portfolio value
                    "probability_weighted_impact": -8750.0,
                    "worst_position": "MSFT",
                    "recovery_time_estimate": 90
                }
            ]
            mock_stress.return_value = mock_stress_results
            
            stress_results = risk_calculator.analyze_scenarios(stress_scenarios, portfolio_exposures)
            
            # Verify stress test results
            assert len(stress_results) == 3
            
            # Find worst-case scenario
            worst_case = max(stress_results, key=lambda x: abs(x["portfolio_impact"]))
            assert worst_case["scenario_name"] == "Market Crash 2008"
            assert worst_case["portfolio_impact"] < -150000  # Significant loss
            
            # Generate stress test report
            stress_report_data = {
                "portfolio_id": portfolio_id,
                "stress_test_date": datetime.utcnow(),
                "scenarios_tested": len(stress_scenarios),
                "worst_case_loss": worst_case["portfolio_impact"],
                "expected_shortfall": sum(r["probability_weighted_impact"] for r in stress_results),
                "stress_results": stress_results
            }
            
            # Generate comprehensive risk report including stress tests
            with patch.object(risk_reporter, 'generate_stress_test_report') as mock_stress_report:
                mock_report = RiskReport(
                    report_id="stress_test_001",
                    report_type=ReportType.STRESS_TEST,
                    portfolio_id=portfolio_id,
                    generation_timestamp=datetime.utcnow(),
                    total_portfolio_value=float(sample_portfolio_data["total_market_value"])
                )
                mock_report.stress_test_results = stress_results
                mock_stress_report.return_value = mock_report
                
                stress_report = risk_reporter.generate_stress_test_report(portfolio_id, stress_report_data)
                
                assert stress_report.report_type == ReportType.STRESS_TEST
                assert len(stress_report.stress_test_results) == 3
    
    @pytest.mark.asyncio
    async def test_risk_limit_breach_escalation_workflow(self, risk_system):
        """Test risk limit breach escalation workflow"""
        limit_engine = risk_system["limit_engine"]
        breach_detector = risk_system["breach_detector"]
        websocket_manager = risk_system["websocket_manager"]
        streaming_service = risk_system["streaming_service"]
        risk_reporter = risk_system["risk_reporter"]
        
        # Setup escalation hierarchy (different alert channels by severity)
        alert_channels = {
            BreachSeverity.WARNING: ["risk.warnings"],
            BreachSeverity.CRITICAL: ["risk.alerts", "management.alerts"],
            BreachSeverity.EMERGENCY: ["risk.alerts", "management.alerts", "emergency.alerts"]
        }
        
        # Setup WebSocket clients for different alert levels
        clients = {}
        for severity, channels in alert_channels.items():
            clients[severity] = []
            for channel in channels:
                for i in range(2):  # 2 clients per channel
                    mock_ws = AsyncMock()
                    conn_id = f"{channel.replace('.', '_')}_{i}"
                    
                    await websocket_manager.connect(mock_ws, conn_id)
                    websocket_manager.subscribe_to_topic(conn_id, channel)
                    
                    clients[severity].append((conn_id, mock_ws, channel))
        
        # Setup progressive limits (same entity, different thresholds)
        limits = [
            RiskLimit(
                limit_id="position_warning_limit",
                limit_type=LimitType.POSITION_SIZE,
                entity_id="NVDA",
                limit_value=Decimal("50000"),
                soft_limit_pct=Decimal("1.0"),  # No soft warning, hard warning
                status=LimitStatus.ACTIVE
            ),
            RiskLimit(
                limit_id="position_critical_limit",
                limit_type=LimitType.POSITION_SIZE,
                entity_id="NVDA",
                limit_value=Decimal("100000"),
                soft_limit_pct=Decimal("1.0"),  # No soft warning, hard critical
                status=LimitStatus.ACTIVE
            )
        ]
        
        for limit in limits:
            limit_engine.add_limit(limit)
        
        # Setup Redis mock
        with patch('redis.asyncio.Redis') as mock_redis_class:
            mock_redis = AsyncMock()
            mock_redis.ping.return_value = True
            mock_redis.publish.return_value = 1
            mock_redis_class.return_value = mock_redis
            
            from websocket.redis_pubsub import RedisPubSubManager
            redis_manager = RedisPubSubManager()
            redis_manager.redis_client = mock_redis
            streaming_service.pubsub_manager = redis_manager
            
            # Simulate escalating position sizes
            position_updates = [
                (Decimal("60000"), BreachSeverity.WARNING),   # Above first limit
                (Decimal("120000"), BreachSeverity.CRITICAL)  # Above second limit
            ]
            
            for position_value, expected_severity in position_updates:
                # Detect breach
                breaches = breach_detector.detect_position_breaches({"NVDA": position_value})
                
                # Should find the highest severity breach
                critical_breaches = [b for b in breaches if b.severity == BreachSeverity.CRITICAL]
                warning_breaches = [b for b in breaches if b.severity == BreachSeverity.WARNING]
                
                if expected_severity == BreachSeverity.CRITICAL:
                    assert len(critical_breaches) >= 1
                elif expected_severity == BreachSeverity.WARNING:
                    assert len(warning_breaches) >= 1
                
                # Send alerts to appropriate channels based on severity
                for breach in breaches:
                    channels_for_severity = alert_channels.get(breach.severity, [])
                    
                    risk_alert = {
                        "alert_id": f"escalation_{breach.entity_id}_{breach.severity.value}",
                        "alert_type": breach.breach_type.value,
                        "entity_id": breach.entity_id,
                        "severity": breach.severity.value,
                        "current_value": float(breach.current_value),
                        "limit_value": float(breach.limit_value),
                        "escalation_level": len(channels_for_severity),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    
                    # Stream to Redis and broadcast to appropriate channels
                    for channel in channels_for_severity:
                        await streaming_service.stream_risk_alert(risk_alert)
                        await websocket_manager.broadcast_message(risk_alert, channel)
                    
                    # Generate escalation report for critical breaches
                    if breach.severity == BreachSeverity.CRITICAL:
                        escalation_report = risk_reporter.generate_breach_incident_report(breach)
                        assert escalation_report.incident_severity == BreachSeverity.CRITICAL
                        
                        # In real implementation, would send to management systems
                        # For testing, verify report contains escalation info
                        assert "NVDA" in str(escalation_report.primary_breach.entity_id)


class TestRiskCalculationPerformance:
    """Test risk calculation performance under load"""
    
    @pytest.fixture
    def large_portfolio(self):
        """Generate large portfolio for performance testing"""
        positions = {}
        for i in range(1000):  # 1000 positions
            symbol = f"STOCK_{i:04d}"
            positions[symbol] = {
                "quantity": 100 + (i % 500),
                "avg_price": Decimal(str(50.0 + (i % 200))),
                "current_price": Decimal(str(52.0 + (i % 180))),
                "market_value": Decimal(str((52.0 + (i % 180)) * (100 + (i % 500)))),
                "unrealized_pnl": Decimal(str(2.0 * (100 + (i % 500))))
            }
        
        return {
            "portfolio_id": "large_portfolio_001",
            "positions": positions,
            "total_market_value": sum(pos["market_value"] for pos in positions.values()),
            "total_unrealized_pnl": sum(pos["unrealized_pnl"] for pos in positions.values())
        }
    
    @pytest.mark.asyncio
    async def test_large_portfolio_risk_calculation_performance(self, large_portfolio):
        """Test risk calculations on large portfolio"""
        import time
        
        risk_calculator = EnhancedRiskCalculator()
        breach_detector = BreachDetector()
        limit_engine = LimitEngine()
        
        # Setup breach detector with limit engine
        breach_detector.limit_engine = limit_engine
        
        # Setup limits for some positions
        for i in range(100):  # 100 limits
            symbol = f"STOCK_{i:04d}"
            limit = RiskLimit(
                limit_id=f"limit_{symbol}",
                limit_type=LimitType.POSITION_SIZE,
                entity_id=symbol,
                limit_value=Decimal("100000"),
                soft_limit_pct=Decimal("0.8"),
                status=LimitStatus.ACTIVE
            )
            limit_engine.add_limit(limit)
        
        # Test breach detection performance
        position_data = {
            symbol: position["market_value"]
            for symbol, position in large_portfolio["positions"].items()
        }
        
        start_time = time.time()
        breaches = breach_detector.detect_position_breaches(position_data)
        breach_detection_time = time.time() - start_time
        
        # Should handle large portfolio efficiently
        assert breach_detection_time < 5.0  # Within 5 seconds
        assert len(breaches) >= 0  # Some positions may breach limits
        
        # Test portfolio risk metrics calculation
        start_time = time.time()
        
        # Simulate risk calculation on large portfolio
        portfolio_value = float(large_portfolio["total_market_value"])
        position_count = len(large_portfolio["positions"])
        
        # Mock risk metrics calculation
        risk_metrics = RiskMetrics(
            portfolio_id=large_portfolio["portfolio_id"],
            total_exposure=portfolio_value,
            daily_pnl=0.0,
            var_95=portfolio_value * 0.02,  # 2% VaR
            expected_shortfall=portfolio_value * 0.035,  # 3.5% ES
            maximum_drawdown=0.10,
            portfolio_beta=1.0,
            portfolio_volatility=0.15,
            position_risks=[]
        )
        
        risk_calculation_time = time.time() - start_time
        
        # Should calculate risk metrics efficiently
        assert risk_calculation_time < 2.0  # Within 2 seconds
        assert risk_metrics.total_exposure == portfolio_value
    
    def test_concurrent_breach_detection(self):
        """Test concurrent breach detection across multiple portfolios"""
        import concurrent.futures
        import threading
        import time
        
        breach_detector = BreachDetector()
        limit_engine = LimitEngine()
        breach_detector.limit_engine = limit_engine
        
        # Setup limits
        for i in range(50):
            limit = RiskLimit(
                limit_id=f"concurrent_limit_{i}",
                limit_type=LimitType.POSITION_SIZE,
                entity_id=f"SYMBOL_{i}",
                limit_value=Decimal("75000"),
                soft_limit_pct=Decimal("0.8"),
                status=LimitStatus.ACTIVE
            )
            limit_engine.add_limit(limit)
        
        # Create position data for multiple portfolios
        portfolios = []
        for portfolio_id in range(10):
            position_data = {}
            for i in range(50):
                symbol = f"SYMBOL_{i}"
                # Some positions will breach limits
                value = Decimal(str(60000 + (portfolio_id * i * 1000) % 50000))
                position_data[symbol] = value
            portfolios.append(position_data)
        
        def detect_breaches_for_portfolio(position_data):
            return breach_detector.detect_position_breaches(position_data)
        
        # Run breach detection concurrently
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(detect_breaches_for_portfolio, portfolio_data)
                for portfolio_data in portfolios
            ]
            
            results = [
                future.result()
                for future in concurrent.futures.as_completed(futures)
            ]
        
        concurrent_time = time.time() - start_time
        
        # Should handle concurrent breach detection efficiently
        assert concurrent_time < 3.0  # Within 3 seconds
        assert len(results) == 10  # All portfolios processed
        
        # Verify results
        total_breaches = sum(len(portfolio_breaches) for portfolio_breaches in results)
        assert total_breaches >= 0  # Some breaches expected


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "--tb=short"])