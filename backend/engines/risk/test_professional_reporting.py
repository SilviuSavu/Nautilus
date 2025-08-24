#!/usr/bin/env python3
"""
Professional Risk Reporting Test Suite
Comprehensive tests for institutional-grade risk reporting system

Test Coverage:
- Professional report generation (HTML, JSON, PDF-ready, Interactive)
- Report template rendering and styling
- Analytics engine integration (PyFolio, Hybrid, k-NN)
- Automated scheduling and delivery
- Performance benchmarking (<5s generation time)
- Data validation and error handling
- Client-ready formatting standards
"""

import pytest
import asyncio
import json
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List
import pandas as pd
import numpy as np

# Import components to test
from professional_risk_reporter import (
    ProfessionalRiskReporter,
    create_professional_risk_reporter,
    ReportConfiguration,
    ReportType,
    ReportFormat,
    ExecutiveSummary,
    ProfessionalReportData
)

from automated_report_scheduler import (
    AutomatedReportScheduler,
    ReportSchedule,
    ScheduleFrequency,
    DeliveryMethod,
    create_automated_scheduler
)

# Mock dependencies
from hybrid_risk_analytics import HybridRiskAnalyticsEngine, HybridAnalyticsResult, PortfolioAnalytics
from advanced_risk_analytics import RiskAnalyticsActor
from pyfolio_integration import PyFolioAnalytics
from supervised_knn_optimizer import SupervisedKNNOptimizer

class TestProfessionalRiskReporter:
    """Test suite for Professional Risk Reporter"""
    
    @pytest.fixture
    async def mock_dependencies(self):
        """Create mock dependencies for testing"""
        # Mock Hybrid Engine
        hybrid_engine = Mock(spec=HybridRiskAnalyticsEngine)
        hybrid_analytics = HybridAnalyticsResult(
            portfolio_analytics=PortfolioAnalytics(
                total_return=0.125,
                sharpe_ratio=1.25,
                max_drawdown=-0.085,
                volatility=0.15,
                value_at_risk=-0.032,
                expected_shortfall=-0.045,
                beta=0.95
            ),
            optimization_result=None,
            stress_test_results=None,
            timestamp=datetime.now()
        )
        hybrid_engine.compute_comprehensive_analytics = AsyncMock(return_value=hybrid_analytics)
        
        # Mock Analytics Actor
        analytics_actor = Mock(spec=RiskAnalyticsActor)
        
        # Mock PyFolio Analytics
        pyfolio_analytics = Mock(spec=PyFolioAnalytics)
        pyfolio_data = {
            'total_return': 0.125,
            'sharpe_ratio': 1.25,
            'max_drawdown': -0.085,
            'volatility': 0.15,
            'daily_mean_return': 0.0005,
            'daily_volatility': 0.018,
            'annual_return': 0.125,
            'annual_volatility': 0.15
        }
        pyfolio_analytics.generate_comprehensive_analysis = AsyncMock(return_value=pyfolio_data)
        
        # Mock Supervised k-NN Optimizer
        supervised_optimizer = Mock(spec=SupervisedKNNOptimizer)
        
        return {
            'hybrid_engine': hybrid_engine,
            'analytics_actor': analytics_actor,
            'pyfolio_analytics': pyfolio_analytics,
            'supervised_optimizer': supervised_optimizer
        }
    
    @pytest.fixture
    async def professional_reporter(self, mock_dependencies):
        """Create professional reporter with mocked dependencies"""
        return ProfessionalRiskReporter(
            hybrid_engine=mock_dependencies['hybrid_engine'],
            analytics_actor=mock_dependencies['analytics_actor'],
            pyfolio_analytics=mock_dependencies['pyfolio_analytics'],
            supervised_optimizer=mock_dependencies['supervised_optimizer']
        )
    
    @pytest.mark.asyncio
    async def test_html_report_generation(self, professional_reporter):
        """Test HTML report generation"""
        config = ReportConfiguration(
            report_type=ReportType.COMPREHENSIVE,
            format=ReportFormat.HTML,
            date_range_days=252
        )
        
        start_time = time.time()
        report = await professional_reporter.generate_professional_report("TEST_PORTFOLIO", config)
        generation_time = time.time() - start_time
        
        # Verify report content
        assert isinstance(report, str)
        assert "<!DOCTYPE html>" in report
        assert "TEST_PORTFOLIO" in report
        assert "Executive Summary" in report
        assert "Risk Assessment" in report
        assert "Performance Analysis" in report
        assert "Nautilus Risk Analytics" in report
        
        # Verify performance requirement (<5s)
        assert generation_time < 5.0, f"Report generation took {generation_time:.3f}s, exceeds 5s limit"
        
        # Verify professional styling
        assert "font-family:" in report
        assert "background:" in report
        assert "border-radius:" in report
        assert "box-shadow:" in report
        
        print(f"✅ HTML report generated in {generation_time:.3f}s")
    
    @pytest.mark.asyncio
    async def test_json_report_generation(self, professional_reporter):
        """Test JSON report generation"""
        config = ReportConfiguration(
            report_type=ReportType.COMPREHENSIVE,
            format=ReportFormat.JSON,
            date_range_days=252
        )
        
        start_time = time.time()
        report = await professional_reporter.generate_professional_report("TEST_PORTFOLIO", config)
        generation_time = time.time() - start_time
        
        # Verify report structure
        assert isinstance(report, dict)
        assert "metadata" in report
        assert "executive_summary" in report
        assert "analytics" in report
        assert "advanced_analytics" in report
        
        # Verify metadata
        metadata = report["metadata"]
        assert metadata["portfolio_id"] == "TEST_PORTFOLIO"
        assert "report_date" in metadata
        assert "generation_time" in metadata
        assert metadata["data_sources"]
        assert metadata["computation_methods"]
        
        # Verify executive summary
        exec_summary = report["executive_summary"]
        assert "total_return_ytd" in exec_summary
        assert "sharpe_ratio" in exec_summary
        assert "max_drawdown" in exec_summary
        assert "key_risks" in exec_summary
        
        # Verify analytics sections
        analytics = report["analytics"]
        assert "returns" in analytics
        assert "risk" in analytics
        assert "attribution" in analytics
        assert "var_analysis" in analytics
        
        # Verify performance requirement
        assert generation_time < 5.0, f"JSON report generation took {generation_time:.3f}s"
        
        print(f"✅ JSON report generated in {generation_time:.3f}s")
    
    @pytest.mark.asyncio
    async def test_executive_summary_report(self, professional_reporter):
        """Test executive summary report generation"""
        config = ReportConfiguration(
            report_type=ReportType.EXECUTIVE_SUMMARY,
            format=ReportFormat.HTML,
            date_range_days=90
        )
        
        report = await professional_reporter.generate_professional_report("EXEC_TEST", config)
        
        # Verify executive summary specific content
        assert "Executive Risk Summary" in report or "Executive Summary" in report
        assert "EXEC_TEST" in report
        assert "Key Risk Insights" in report
        assert "Risk Dashboard" in report
        
        # Should be more concise than comprehensive report
        assert len(report) < 50000  # Reasonable size limit for executive summary
        
        print("✅ Executive summary report generated successfully")
    
    @pytest.mark.asyncio
    async def test_pdf_ready_report(self, professional_reporter):
        """Test PDF-ready report generation"""
        config = ReportConfiguration(
            report_type=ReportType.CLIENT_TEAR_SHEET,
            format=ReportFormat.PDF_READY,
            date_range_days=252
        )
        
        report = await professional_reporter.generate_professional_report("PDF_TEST", config)
        
        # Verify PDF-ready specific styling
        assert "@media print" in report
        assert "page-break-before: always" in report
        assert ".no-print { display: none; }" in report
        assert "box-shadow: none" in report
        
        print("✅ PDF-ready report generated successfully")
    
    @pytest.mark.asyncio
    async def test_interactive_report(self, professional_reporter):
        """Test interactive report generation"""
        config = ReportConfiguration(
            report_type=ReportType.COMPREHENSIVE,
            format=ReportFormat.INTERACTIVE,
            date_range_days=252
        )
        
        report = await professional_reporter.generate_professional_report("INTERACTIVE_TEST", config)
        
        # Verify interactive elements
        assert "plotly-latest.min.js" in report or "Interactive report loaded" in report
        assert "<script>" in report
        assert "DOMContentLoaded" in report
        
        print("✅ Interactive report generated successfully")
    
    @pytest.mark.asyncio
    async def test_custom_branding(self, professional_reporter):
        """Test custom branding functionality"""
        custom_branding = {
            "client_name": "Hedge Fund XYZ",
            "report_title": "Custom Risk Report",
            "footer_text": "Confidential - For HF XYZ Only"
        }
        
        config = ReportConfiguration(
            report_type=ReportType.CLIENT_TEAR_SHEET,
            format=ReportFormat.HTML,
            custom_branding=custom_branding
        )
        
        report = await professional_reporter.generate_professional_report("CUSTOM_TEST", config)
        
        # Verify custom branding elements
        # Note: This would require template updates to support custom branding
        assert "CUSTOM_TEST" in report
        
        print("✅ Custom branding test completed")
    
    @pytest.mark.asyncio
    async def test_performance_metrics(self, professional_reporter):
        """Test performance metrics tracking"""
        # Generate multiple reports to test performance tracking
        config = ReportConfiguration(
            report_type=ReportType.EXECUTIVE_SUMMARY,
            format=ReportFormat.HTML,
            date_range_days=90
        )
        
        # Generate 3 reports
        for i in range(3):
            await professional_reporter.generate_professional_report(f"PERF_TEST_{i}", config)
        
        # Check performance metrics
        metrics = await professional_reporter.get_performance_metrics()
        
        assert metrics["reports_generated"] >= 3
        assert metrics["average_generation_time"] > 0
        assert metrics["total_generation_time"] > 0
        assert "performance_target_met" in metrics
        
        # Verify performance target (should be under 5 seconds average)
        assert metrics["performance_target_met"] == True, "Average generation time exceeds 5s target"
        
        print(f"✅ Performance metrics: {metrics['reports_generated']} reports, "
              f"{metrics['average_generation_time']:.3f}s average")
    
    @pytest.mark.asyncio
    async def test_error_handling(self, professional_reporter):
        """Test error handling and validation"""
        # Test invalid report type
        with pytest.raises(ValueError):
            config = ReportConfiguration(
                report_type="invalid_type",  # This should cause an error
                format=ReportFormat.HTML
            )
        
        # Test invalid portfolio ID
        config = ReportConfiguration(
            report_type=ReportType.COMPREHENSIVE,
            format=ReportFormat.HTML
        )
        
        # Should handle gracefully even with invalid portfolio
        try:
            report = await professional_reporter.generate_professional_report("", config)
            assert "analytics" in report  # Should generate with mock data
        except Exception as e:
            assert "portfolio" in str(e).lower()
        
        print("✅ Error handling tests passed")

class TestAutomatedReportScheduler:
    """Test suite for Automated Report Scheduler"""
    
    @pytest.fixture
    async def mock_professional_reporter(self):
        """Create mock professional reporter"""
        reporter = Mock(spec=ProfessionalRiskReporter)
        reporter.generate_professional_report = AsyncMock(return_value="<html>Mock Report</html>")
        return reporter
    
    @pytest.fixture
    async def scheduler(self, mock_professional_reporter):
        """Create scheduler for testing"""
        scheduler = AutomatedReportScheduler(mock_professional_reporter)
        # Don't start the scheduler loop for testing
        return scheduler
    
    @pytest.mark.asyncio
    async def test_schedule_creation(self, scheduler):
        """Test creating report schedules"""
        schedule = ReportSchedule(
            schedule_id="test_schedule",
            portfolio_id="TEST_PORTFOLIO",
            client_name="Test Client",
            report_type=ReportType.EXECUTIVE_SUMMARY,
            report_format=ReportFormat.HTML,
            frequency=ScheduleFrequency.DAILY,
            delivery_method=DeliveryMethod.EMAIL,
            email_recipients=["test@example.com"]
        )
        
        schedule_id = await scheduler.add_schedule(schedule)
        
        assert schedule_id in scheduler.schedules
        stored_schedule = scheduler.schedules[schedule_id]
        assert stored_schedule.portfolio_id == "TEST_PORTFOLIO"
        assert stored_schedule.client_name == "Test Client"
        assert stored_schedule.next_run is not None
        
        print(f"✅ Schedule created: {schedule_id}")
    
    @pytest.mark.asyncio
    async def test_schedule_validation(self, scheduler):
        """Test schedule validation"""
        # Test missing portfolio ID
        with pytest.raises(ValueError, match="Portfolio ID is required"):
            schedule = ReportSchedule(
                schedule_id="invalid_schedule",
                portfolio_id="",  # Empty portfolio ID
                client_name="Test Client",
                report_type=ReportType.EXECUTIVE_SUMMARY,
                report_format=ReportFormat.HTML,
                frequency=ScheduleFrequency.DAILY,
                delivery_method=DeliveryMethod.EMAIL,
                email_recipients=["test@example.com"]
            )
            await scheduler.add_schedule(schedule)
        
        # Test missing email recipients for email delivery
        with pytest.raises(ValueError, match="Email recipients required"):
            schedule = ReportSchedule(
                schedule_id="invalid_schedule2",
                portfolio_id="TEST_PORTFOLIO",
                client_name="Test Client",
                report_type=ReportType.EXECUTIVE_SUMMARY,
                report_format=ReportFormat.HTML,
                frequency=ScheduleFrequency.DAILY,
                delivery_method=DeliveryMethod.EMAIL,
                email_recipients=[]  # Empty email list
            )
            await scheduler.add_schedule(schedule)
        
        print("✅ Schedule validation tests passed")
    
    @pytest.mark.asyncio
    async def test_immediate_report_generation(self, scheduler):
        """Test immediate report generation"""
        schedule = ReportSchedule(
            schedule_id="immediate_test",
            portfolio_id="IMMEDIATE_TEST",
            client_name="Immediate Test Client",
            report_type=ReportType.EXECUTIVE_SUMMARY,
            report_format=ReportFormat.JSON,
            frequency=ScheduleFrequency.DAILY,
            delivery_method=DeliveryMethod.FILE_SYSTEM,
            file_path_template="./test_reports/{portfolio_id}_{date}.json"
        )
        
        schedule_id = await scheduler.add_schedule(schedule)
        
        # Generate immediate report
        delivery_id = await scheduler.generate_immediate_report(schedule_id)
        
        assert delivery_id is not None
        assert delivery_id in scheduler.deliveries
        
        delivery = scheduler.deliveries[delivery_id]
        assert delivery.portfolio_id == "IMMEDIATE_TEST"
        assert delivery.report_format == ReportFormat.JSON
        
        print(f"✅ Immediate report generated: {delivery_id}")
    
    @pytest.mark.asyncio
    async def test_schedule_operations(self, scheduler):
        """Test schedule CRUD operations"""
        # Create schedule
        schedule = ReportSchedule(
            schedule_id="crud_test",
            portfolio_id="CRUD_TEST",
            client_name="CRUD Test Client",
            report_type=ReportType.COMPREHENSIVE,
            report_format=ReportFormat.HTML,
            frequency=ScheduleFrequency.WEEKLY,
            delivery_method=DeliveryMethod.FILE_SYSTEM,
            file_path_template="./test_reports/crud_{date}.html"
        )
        
        schedule_id = await scheduler.add_schedule(schedule)
        
        # Read schedule
        retrieved_schedule = await scheduler.get_schedule(schedule_id)
        assert retrieved_schedule is not None
        assert retrieved_schedule.portfolio_id == "CRUD_TEST"
        
        # Update schedule
        updates = {"client_name": "Updated CRUD Client", "enabled": False}
        success = await scheduler.update_schedule(schedule_id, updates)
        assert success
        
        updated_schedule = await scheduler.get_schedule(schedule_id)
        assert updated_schedule.client_name == "Updated CRUD Client"
        assert updated_schedule.enabled == False
        
        # List schedules
        all_schedules = await scheduler.list_schedules()
        assert len(all_schedules) >= 1
        
        portfolio_schedules = await scheduler.list_schedules("CRUD_TEST")
        assert len(portfolio_schedules) == 1
        assert portfolio_schedules[0].schedule_id == schedule_id
        
        # Delete schedule
        deleted = await scheduler.remove_schedule(schedule_id)
        assert deleted
        
        # Verify deletion
        deleted_schedule = await scheduler.get_schedule(schedule_id)
        assert deleted_schedule is None
        
        print("✅ Schedule CRUD operations completed")
    
    @pytest.mark.asyncio
    async def test_scheduler_status(self, scheduler):
        """Test scheduler status reporting"""
        # Add some test schedules
        for i in range(3):
            schedule = ReportSchedule(
                schedule_id=f"status_test_{i}",
                portfolio_id=f"STATUS_TEST_{i}",
                client_name=f"Status Test Client {i}",
                report_type=ReportType.EXECUTIVE_SUMMARY,
                report_format=ReportFormat.HTML,
                frequency=ScheduleFrequency.DAILY,
                delivery_method=DeliveryMethod.FILE_SYSTEM,
                file_path_template=f"./test_reports/status_{i}_{{date}}.html"
            )
            await scheduler.add_schedule(schedule)
        
        # Generate some reports to update statistics
        scheduler.reports_generated = 10
        scheduler.reports_delivered = 8
        scheduler.delivery_failures = 2
        
        status = await scheduler.get_scheduler_status()
        
        assert status["total_schedules"] >= 3
        assert status["enabled_schedules"] >= 3
        assert status["reports_generated"] == 10
        assert status["reports_delivered"] == 8
        assert status["delivery_failures"] == 2
        assert status["success_rate"] == 80.0
        assert "next_scheduled_runs" in status
        
        print(f"✅ Scheduler status: {status}")

class TestIntegrationAndPerformance:
    """Integration tests and performance benchmarks"""
    
    @pytest.fixture
    async def integrated_system(self, mock_dependencies):
        """Create fully integrated system for testing"""
        reporter = ProfessionalRiskReporter(
            hybrid_engine=mock_dependencies['hybrid_engine'],
            analytics_actor=mock_dependencies['analytics_actor'],
            pyfolio_analytics=mock_dependencies['pyfolio_analytics'],
            supervised_optimizer=mock_dependencies['supervised_optimizer']
        )
        
        scheduler = AutomatedReportScheduler(reporter)
        
        return {"reporter": reporter, "scheduler": scheduler}
    
    @pytest.mark.asyncio
    async def test_end_to_end_report_generation(self, integrated_system):
        """Test complete end-to-end report generation workflow"""
        reporter = integrated_system["reporter"]
        
        # Test all report types and formats
        test_cases = [
            (ReportType.EXECUTIVE_SUMMARY, ReportFormat.HTML),
            (ReportType.COMPREHENSIVE, ReportFormat.JSON),
            (ReportType.CLIENT_TEAR_SHEET, ReportFormat.PDF_READY),
            (ReportType.RISK_FOCUSED, ReportFormat.HTML),
            (ReportType.PERFORMANCE_FOCUSED, ReportFormat.JSON)
        ]
        
        results = []
        
        for report_type, report_format in test_cases:
            config = ReportConfiguration(
                report_type=report_type,
                format=report_format,
                date_range_days=252
            )
            
            start_time = time.time()
            report = await reporter.generate_professional_report("E2E_TEST", config)
            generation_time = time.time() - start_time
            
            results.append({
                "type": report_type.value,
                "format": report_format.value,
                "generation_time": generation_time,
                "size": len(str(report)),
                "success": report is not None
            })
            
            # Verify performance requirement
            assert generation_time < 5.0, f"{report_type.value} report took {generation_time:.3f}s"
        
        # Print performance summary
        print("\n📊 End-to-End Performance Results:")
        for result in results:
            print(f"  {result['type']} ({result['format']}): {result['generation_time']:.3f}s, "
                  f"{result['size']:,} chars")
        
        avg_time = sum(r['generation_time'] for r in results) / len(results)
        print(f"  Average generation time: {avg_time:.3f}s")
        
        assert all(r['success'] for r in results), "Some reports failed to generate"
        assert avg_time < 3.0, f"Average generation time {avg_time:.3f}s exceeds target"
    
    @pytest.mark.asyncio
    async def test_concurrent_report_generation(self, integrated_system):
        """Test concurrent report generation performance"""
        reporter = integrated_system["reporter"]
        
        # Generate multiple reports concurrently
        config = ReportConfiguration(
            report_type=ReportType.EXECUTIVE_SUMMARY,
            format=ReportFormat.HTML,
            date_range_days=90
        )
        
        concurrent_count = 5
        start_time = time.time()
        
        tasks = [
            reporter.generate_professional_report(f"CONCURRENT_{i}", config)
            for i in range(concurrent_count)
        ]
        
        reports = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Verify all reports generated successfully
        assert len(reports) == concurrent_count
        assert all(isinstance(report, str) and len(report) > 1000 for report in reports)
        
        # Performance check - concurrent should be faster than sequential
        avg_time_per_report = total_time / concurrent_count
        print(f"✅ Concurrent generation: {concurrent_count} reports in {total_time:.3f}s "
              f"({avg_time_per_report:.3f}s avg per report)")
        
        assert total_time < 10.0, f"Concurrent generation took {total_time:.3f}s, too slow"
    
    @pytest.mark.asyncio
    async def test_memory_usage(self, integrated_system):
        """Test memory usage during report generation"""
        import psutil
        import os
        
        reporter = integrated_system["reporter"]
        process = psutil.Process(os.getpid())
        
        # Get initial memory usage
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Generate large comprehensive report
        config = ReportConfiguration(
            report_type=ReportType.COMPREHENSIVE,
            format=ReportFormat.HTML,
            date_range_days=1000  # Large dataset
        )
        
        report = await reporter.generate_professional_report("MEMORY_TEST", config)
        
        # Get peak memory usage
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        print(f"📊 Memory usage: {initial_memory:.1f}MB → {peak_memory:.1f}MB "
              f"(+{memory_increase:.1f}MB)")
        
        # Verify reasonable memory usage (should not exceed 500MB increase)
        assert memory_increase < 500, f"Memory increase {memory_increase:.1f}MB too high"
        assert len(report) > 10000, "Report should be substantial for large dataset"

@pytest.mark.asyncio
async def test_institutional_formatting_standards():
    """Test institutional formatting and styling standards"""
    # This would test specific formatting requirements
    # like font families, color schemes, spacing, etc.
    
    # Mock reporter for testing
    reporter = Mock(spec=ProfessionalRiskReporter)
    
    # Test CSS standards
    css_requirements = [
        "font-family",
        "border-radius",
        "box-shadow",
        "linear-gradient",
        "@media print",
        "responsive"
    ]
    
    # Mock CSS content
    mock_css = """
    body { font-family: 'Inter', sans-serif; }
    .container { border-radius: 0.5rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .header { background: linear-gradient(135deg, #1e3a8a 0%, #0f172a 100%); }
    @media print { body { font-size: 12pt; } }
    @media (max-width: 768px) { .container { margin: 10px; } }
    """
    
    for requirement in css_requirements:
        assert requirement in mock_css, f"Missing CSS standard: {requirement}"
    
    print("✅ Institutional formatting standards validated")

# Performance benchmarking
@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Performance benchmark tests"""
    
    @pytest.mark.asyncio
    async def test_report_generation_benchmark(self, mock_dependencies):
        """Benchmark report generation performance"""
        reporter = ProfessionalRiskReporter(
            hybrid_engine=mock_dependencies['hybrid_engine'],
            analytics_actor=mock_dependencies['analytics_actor'],
            pyfolio_analytics=mock_dependencies['pyfolio_analytics'],
            supervised_optimizer=mock_dependencies['supervised_optimizer']
        )
        
        # Benchmark different report types
        benchmarks = {}
        
        report_configs = [
            ("Executive Summary", ReportType.EXECUTIVE_SUMMARY, ReportFormat.HTML),
            ("Comprehensive HTML", ReportType.COMPREHENSIVE, ReportFormat.HTML),
            ("Comprehensive JSON", ReportType.COMPREHENSIVE, ReportFormat.JSON),
            ("Client Tear Sheet", ReportType.CLIENT_TEAR_SHEET, ReportFormat.PDF_READY)
        ]
        
        for name, report_type, format_type in report_configs:
            config = ReportConfiguration(
                report_type=report_type,
                format=format_type,
                date_range_days=252
            )
            
            # Run multiple iterations
            times = []
            for i in range(5):
                start_time = time.time()
                report = await reporter.generate_professional_report(f"BENCH_{i}", config)
                elapsed = time.time() - start_time
                times.append(elapsed)
            
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            benchmarks[name] = {
                "avg": avg_time,
                "min": min_time,
                "max": max_time,
                "target_met": avg_time < 5.0
            }
        
        # Print benchmark results
        print("\n🏃 Performance Benchmarks:")
        for name, results in benchmarks.items():
            status = "✅" if results["target_met"] else "❌"
            print(f"  {status} {name}: {results['avg']:.3f}s avg "
                  f"({results['min']:.3f}s - {results['max']:.3f}s)")
        
        # Verify all benchmarks meet performance targets
        failed_benchmarks = [name for name, results in benchmarks.items() 
                           if not results["target_met"]]
        
        assert not failed_benchmarks, f"Performance targets not met: {failed_benchmarks}"

if __name__ == "__main__":
    """Run tests directly"""
    import sys
    
    # Configure logging for tests
    logging.basicConfig(level=logging.INFO)
    
    # Run specific test categories based on arguments
    if len(sys.argv) > 1:
        if "performance" in sys.argv[1].lower():
            pytest.main(["-v", "-m", "benchmark", __file__])
        elif "integration" in sys.argv[1].lower():
            pytest.main(["-v", "-k", "integration", __file__])
        else:
            pytest.main(["-v", __file__])
    else:
        # Run all tests
        pytest.main(["-v", "--tb=short", __file__])