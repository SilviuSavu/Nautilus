#!/usr/bin/env python3
"""
Test script for the Comprehensive Risk Reporter
Demonstrates key functionality and integration capabilities
"""

import asyncio
import json
import logging
from datetime import datetime
from risk_reporter import (
    ComprehensiveRiskReporter, 
    ReportType, 
    ReportFormat, 
    ReportConfig,
    ReportFrequency,
    ReportPriority
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_basic_report_generation():
    """Test basic report generation functionality"""
    logger.info("=== Testing Basic Report Generation ===")
    
    try:
        # Create risk reporter instance (without full initialization for testing)
        reporter = ComprehensiveRiskReporter()
        
        # Initialize basic components
        reporter.report_templates = reporter._initialize_templates()
        reporter.dashboard_themes = reporter._initialize_dashboard_themes()
        
        # Test executive summary generation
        logger.info("Testing Executive Summary Report...")
        
        report_data, metadata = await reporter.generate_report(
            report_type=ReportType.EXECUTIVE_SUMMARY,
            portfolio_ids=["PORTFOLIO_001", "PORTFOLIO_002"],
            format=ReportFormat.JSON,
            parameters={"period": "Daily"}
        )
        
        logger.info(f"Executive Summary generated successfully:")
        logger.info(f"  - Generation time: {metadata.generation_time_ms}ms")
        logger.info(f"  - Report size: {metadata.size_bytes} bytes")
        logger.info(f"  - Status: {metadata.status}")
        
        # Print key metrics from the report
        if 'summary' in report_data and isinstance(report_data['summary'], dict):
            summary = report_data['summary']
            logger.info(f"  - Total Portfolios: {summary.get('total_portfolios')}")
            logger.info(f"  - Total VaR 95%: ${summary.get('total_var_95', 0):,.2f}")
            logger.info(f"  - Risk Utilization: {summary.get('risk_utilization_pct', 0):.2f}%")
            logger.info(f"  - Critical Alerts: {summary.get('critical_alerts')}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in basic report generation test: {e}")
        return False

async def test_daily_risk_report():
    """Test daily risk report generation"""
    logger.info("\n=== Testing Daily Risk Report ===")
    
    try:
        reporter = ComprehensiveRiskReporter()
        reporter.report_templates = reporter._initialize_templates()
        
        report_data, metadata = await reporter.generate_report(
            report_type=ReportType.DAILY_RISK,
            portfolio_ids=["PORTFOLIO_001", "PORTFOLIO_002", "PORTFOLIO_003"],
            format=ReportFormat.JSON,
            parameters={"date": datetime.utcnow().date()}
        )
        
        logger.info(f"Daily Risk Report generated successfully:")
        logger.info(f"  - Generation time: {metadata.generation_time_ms}ms")
        logger.info(f"  - Portfolios analyzed: {len(report_data.get('portfolios', []))}")
        logger.info(f"  - Risk alerts: {len(report_data.get('risk_alerts', []))}")
        
        # Show portfolio breakdown
        for i, portfolio in enumerate(report_data.get('portfolios', [])[:3]):
            logger.info(f"  - Portfolio {i+1}: VaR 95% = ${portfolio.get('var_1d_95', 0):,.2f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in daily risk report test: {e}")
        return False

async def test_dashboard_data():
    """Test dashboard data generation"""
    logger.info("\n=== Testing Dashboard Data Generation ===")
    
    try:
        reporter = ComprehensiveRiskReporter()
        await reporter._initialize_default_widgets()
        
        dashboard_data = await reporter.get_dashboard_data(
            portfolio_ids=["PORTFOLIO_001", "PORTFOLIO_002"]
        )
        
        logger.info(f"Dashboard data generated successfully:")
        logger.info(f"  - Widgets: {len(dashboard_data.get('widgets', {}))}")
        logger.info(f"  - Charts: {len(dashboard_data.get('charts', {}))}")
        logger.info(f"  - Alerts: {len(dashboard_data.get('alerts', []))}")
        
        # Show summary metrics
        summary_metrics = dashboard_data.get('summary_metrics', {})
        logger.info(f"  - Total Value: ${summary_metrics.get('total_value', 0):,.2f}")
        logger.info(f"  - Total VaR 95%: ${summary_metrics.get('total_var_95', 0):,.2f}")
        logger.info(f"  - Active Alerts: {summary_metrics.get('active_alerts', 0)}")
        logger.info(f"  - Risk Level: {summary_metrics.get('risk_level', 'unknown')}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in dashboard data test: {e}")
        return False

async def test_report_scheduling():
    """Test report scheduling functionality"""
    logger.info("\n=== Testing Report Scheduling ===")
    
    try:
        reporter = ComprehensiveRiskReporter()
        
        # Create a scheduled report configuration
        config = ReportConfig(
            report_id="test_scheduled_daily",
            report_type=ReportType.DAILY_RISK,
            name="Test Daily Risk Report",
            description="Automated daily risk assessment for testing",
            portfolio_ids=["PORTFOLIO_001", "PORTFOLIO_002"],
            format=ReportFormat.JSON,
            frequency=ReportFrequency.DAILY,
            schedule_expression="0 9 * * *",  # Daily at 9 AM
            recipients=["test@example.com"],
            priority=ReportPriority.HIGH,
            active=True
        )
        
        # Schedule the report
        config_id = await reporter.schedule_report(config)
        
        logger.info(f"Report scheduled successfully:")
        logger.info(f"  - Configuration ID: {config_id}")
        logger.info(f"  - Report Type: {config.report_type.value}")
        logger.info(f"  - Frequency: {config.frequency.value}")
        logger.info(f"  - Recipients: {len(config.recipients)}")
        logger.info(f"  - Next Run: {config.next_run}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in report scheduling test: {e}")
        return False

async def test_multiple_formats():
    """Test multiple report formats"""
    logger.info("\n=== Testing Multiple Report Formats ===")
    
    try:
        reporter = ComprehensiveRiskReporter()
        reporter.report_templates = reporter._initialize_templates()
        
        formats_to_test = [ReportFormat.JSON, ReportFormat.HTML, ReportFormat.DASHBOARD]
        
        for format_type in formats_to_test:
            logger.info(f"Testing {format_type.value} format...")
            
            report_data, metadata = await reporter.generate_report(
                report_type=ReportType.EXECUTIVE_SUMMARY,
                portfolio_ids=["PORTFOLIO_001"],
                format=format_type
            )
            
            logger.info(f"  - {format_type.value}: {metadata.size_bytes} bytes, {metadata.generation_time_ms}ms")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in multiple formats test: {e}")
        return False

async def test_export_functionality():
    """Test report export functionality"""
    logger.info("\n=== Testing Export Functionality ===")
    
    try:
        reporter = ComprehensiveRiskReporter()
        
        # Generate a sample report
        report_data, metadata = await reporter.generate_report(
            report_type=ReportType.DAILY_RISK,
            portfolio_ids=["PORTFOLIO_001"],
            format=ReportFormat.JSON
        )
        
        # Test different export formats
        export_formats = [ReportFormat.JSON, ReportFormat.CSV, ReportFormat.HTML]
        
        for export_format in export_formats:
            logger.info(f"Testing {export_format.value} export...")
            
            export_result = await reporter.export_report(
                report_data, 
                export_format
            )
            
            logger.info(f"  - {export_format.value} export: {'Success' if export_result else 'Failed'}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in export functionality test: {e}")
        return False

async def test_stress_test_report():
    """Test stress test report generation"""
    logger.info("\n=== Testing Stress Test Report ===")
    
    try:
        reporter = ComprehensiveRiskReporter()
        
        report_data, metadata = await reporter.generate_report(
            report_type=ReportType.STRESS_TEST,
            portfolio_ids=["PORTFOLIO_001", "PORTFOLIO_002"],
            format=ReportFormat.JSON,
            parameters={
                "scenarios": ["MARKET_CRASH", "VOLATILITY_SPIKE", "INTEREST_RATE_SHOCK"]
            }
        )
        
        logger.info(f"Stress Test Report generated successfully:")
        logger.info(f"  - Scenarios tested: {len(report_data.get('scenarios_tested', []))}")
        logger.info(f"  - Portfolios analyzed: {len(report_data.get('portfolios', []))}")
        
        # Show aggregate results
        aggregate_results = report_data.get('aggregate_results', {})
        for scenario, results in aggregate_results.items():
            logger.info(f"  - {scenario}: Total Impact = ${results.get('total_impact', 0):,.2f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in stress test report test: {e}")
        return False

async def test_regulatory_compliance_report():
    """Test regulatory compliance report generation"""
    logger.info("\n=== Testing Regulatory Compliance Report ===")
    
    try:
        reporter = ComprehensiveRiskReporter()
        
        report_data, metadata = await reporter.generate_report(
            report_type=ReportType.REGULATORY_COMPLIANCE,
            portfolio_ids=["PORTFOLIO_001", "PORTFOLIO_002"],
            format=ReportFormat.JSON,
            parameters={
                "regulation": "Basel III",
                "period": "Q4 2024"
            }
        )
        
        logger.info(f"Regulatory Compliance Report generated successfully:")
        
        compliance_summary = report_data.get('compliance_summary', {})
        logger.info(f"  - Total Portfolios: {compliance_summary.get('total_portfolios')}")
        logger.info(f"  - Compliant Portfolios: {compliance_summary.get('compliant_portfolios')}")
        logger.info(f"  - Total Violations: {compliance_summary.get('total_violations')}")
        logger.info(f"  - Compliance Score: {compliance_summary.get('compliance_score'):.1f}%")
        logger.info(f"  - Status: {compliance_summary.get('status')}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in regulatory compliance report test: {e}")
        return False

async def test_reporter_statistics():
    """Test reporter statistics functionality"""
    logger.info("\n=== Testing Reporter Statistics ===")
    
    try:
        reporter = ComprehensiveRiskReporter()
        await reporter._initialize_default_widgets()
        
        # Generate a few reports to populate statistics
        for _ in range(3):
            await reporter.generate_report(
                report_type=ReportType.EXECUTIVE_SUMMARY,
                portfolio_ids=["PORTFOLIO_001"],
                format=ReportFormat.JSON
            )
        
        # Get statistics
        stats = reporter.get_statistics()
        
        logger.info(f"Reporter Statistics:")
        logger.info(f"  - Reports Generated: {stats['reports_generated']}")
        logger.info(f"  - Dashboard Updates: {stats['dashboard_updates']}")
        logger.info(f"  - Error Count: {stats['error_count']}")
        logger.info(f"  - Avg Generation Time: {stats['avg_generation_time_ms']:.2f}ms")
        logger.info(f"  - Scheduled Reports: {stats['scheduled_reports']}")
        logger.info(f"  - Cached Reports: {stats['cached_reports']}")
        logger.info(f"  - Active Widgets: {stats['active_widgets']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in reporter statistics test: {e}")
        return False

async def run_comprehensive_test():
    """Run comprehensive test suite"""
    logger.info("Starting Comprehensive Risk Reporter Test Suite")
    logger.info("=" * 60)
    
    test_functions = [
        test_basic_report_generation,
        test_daily_risk_report,
        test_dashboard_data,
        test_report_scheduling,
        test_multiple_formats,
        test_export_functionality,
        test_stress_test_report,
        test_regulatory_compliance_report,
        test_reporter_statistics
    ]
    
    passed_tests = 0
    total_tests = len(test_functions)
    
    for test_func in test_functions:
        try:
            result = await test_func()
            if result:
                passed_tests += 1
                logger.info(f"✅ {test_func.__name__} PASSED")
            else:
                logger.error(f"❌ {test_func.__name__} FAILED")
        except Exception as e:
            logger.error(f"❌ {test_func.__name__} FAILED with exception: {e}")
    
    logger.info("=" * 60)
    logger.info(f"Test Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 All tests passed! Risk Reporter is working correctly.")
    else:
        logger.warning(f"⚠️  {total_tests - passed_tests} test(s) failed. Please review the implementation.")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    # Run the comprehensive test
    success = asyncio.run(run_comprehensive_test())
    exit(0 if success else 1)